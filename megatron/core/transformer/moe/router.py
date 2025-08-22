# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from abc import ABC, abstractmethod

import torch
import torch.distributed

from megatron.core import parallel_state
from megatron.core.tensor_parallel import (
    gather_from_sequence_parallel_region,
    get_cuda_rng_tracker,
    get_data_parallel_rng_tracker_name,
)
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe.moe_utils import (
    MoEAuxLossAutoScaler,
    save_to_aux_losses_tracker,
    sinkhorn,
    switch_load_balancing_loss_func,
    topk_softmax_with_capacity,
    z_loss_func,
    moe_lpr_loss_func,
)
from megatron.core.transformer.transformer_config import TransformerConfig
import torch.nn.functional as F
import torch.nn as nn

class Router(ABC, MegatronModule):
    """Base Router class"""

    def __init__(self, config: TransformerConfig, is_language_router=False) -> None:
        """
        Initialize the Router module.

        Args:
            config (TransformerConfig): Configuration object for the Transformer model.
        """
        super().__init__(config)
        self.config = config
        self.num_experts = self.config.num_moe_experts if not is_language_router else 2
        self.moe_aux_loss_func = None
        self.layer_number = None

        # Initialize the gate weights.
        self.weight = torch.nn.Parameter(
            torch.empty((self.num_experts, self.config.hidden_size), dtype=torch.float32)
        )
        if config.perform_initialization:
            if get_cuda_rng_tracker().is_initialized():
                with get_cuda_rng_tracker().fork(get_data_parallel_rng_tracker_name()):
                    config.init_method(self.weight)
        else:
            config.init_method(self.weight)
        self.weight.data = self.weight.data.to(dtype=config.params_dtype)
        setattr(self.weight, 'sequence_parallel', config.sequence_parallel)

    def gating(self, input: torch.Tensor):
        """Forward pass of the router gate.

        Args:
            input (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Logits tensor.
        """
        if self.weight.device.type == 'cpu':
            # move weights to GPU
            self.weight.data = self.weight.data.to(device=torch.cuda.current_device())
        
        logits_token = torch.nn.functional.linear(input, self.weight[:self.num_experts // 2, :])
        pool = nn.MaxPool1d(kernel_size=self.config.context_window, stride=self.config.context_window)
        logits_segment = torch.nn.functional.linear(pool(input.permute(2, 1, 0)).permute(2, 1, 0),self.weight[self.num_experts // 2:, :])
        return logits_token, logits_segment


    @abstractmethod
    def routing(self, logits: torch.Tensor):
        """Routing function.

        Args:
            logits (torch.Tensor): Logits tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple of tensors representing max probs and the indices.
        """
        raise NotImplementedError("Routing function not implemented.")

    @abstractmethod
    def forward(self, input: torch.Tensor):
        """
        Forward pass of the router.

        Args:
            input (torch.Tensor): Input tensor.
        """
        raise NotImplementedError("Forward function not implemented.")

    def set_layer_number(self, layer_number: int):
        """Set the layer number for the router."""
        self.layer_number = layer_number


class TopKRouter(Router):
    """Route each token to the top-k experts."""

    def __init__(self, config: TransformerConfig) -> None:
        """Initialize the zero token dropping router.

        Args:
            config (TransformerConfig): The configuration for the transformer model.
        """
        super().__init__(config=config)
        self.topk = self.config.moe_router_topk
        self.routing_type = self.config.moe_router_load_balancing_type
        self.input_jitter = None

    def sinkhorn_load_balancing(self, logits: torch.Tensor):
        """Apply sinkhorn routing to the logits tensor.

        Args:
            logits (torch.Tensor): The logits tensor.

        Returns:
            torch.Tensor: The logits tensor after applying sinkhorn routing.
        """

        def _sinkhorn_activation(logits):
            if self.topk == 1:
                logits = torch.sigmoid(logits)
            else:  # k > 1
                logits = torch.softmax(logits, dim=-1, dtype=torch.float32).type_as(logits)
            return logits

        assert self.config.moe_aux_loss_coeff == 0, "Sinkhorn routing does not support aux loss."
        if self.training:
            with torch.no_grad():
                norm_logits = sinkhorn(
                    logits.to(dtype=torch.float32)
                )  # explicit fp32 conversion for stability
                _, indices = torch.topk(norm_logits, k=self.topk, dim=1)
            logits = _sinkhorn_activation(logits)
            scores = torch.gather(logits, 1, indices)
        else:
            logits = _sinkhorn_activation(logits)
            scores, indices = torch.topk(logits, k=self.topk, dim=1)
        return scores, indices

    def aux_loss_load_balancing(self, logits: torch.Tensor, lang_mask: torch.Tensor = None):
        """Apply loss-based load balancing to the logits tensor.

        Args:
            logits (torch.Tensor): the logits tensor after gating, shape: [num_tokens, num_experts].

        Returns:
            probs (torch.Tensor): the probabilities tensor after load balancing.
            indices (torch.Tensor): the indices tensor after top-k selection.
        """
        probs, indices, tokens_per_expert,  scores = topk_softmax_with_capacity(
            logits,
            self.topk,
            capacity_factor=self.config.moe_expert_capacity_factor,
            pad_to_capacity=self.config.moe_pad_expert_input_to_capacity,
            drop_policy=self.config.moe_token_drop_policy,
            use_pre_softmax=self.config.moe_router_pre_softmax,
        )
        if self.training:
            # Apply load balancing loss
            if self.config.moe_lpr_stage != 2:
                probs = self.apply_load_balancing_loss(scores, tokens_per_expert, activation=probs)
            else:
                probs = self.apply_lpr_loss(scores, lang_mask, activation=probs)

        return probs, indices

    def apply_lpr_loss(
        self,
        probs: torch.Tensor,
        lang_mask: torch.Tensor,
        activation: torch.Tensor,
    ):
        """Applies auxiliary loss to the MoE layer.

        Args:
            probs (torch.Tensor): The probs output by the router for each token. [num_tokens, num_experts]
            num_local_tokens_per_expert (torch.Tensor): The number of tokens per expert. [num_experts]
            activation (torch.Tensor): The activation tensor to attach the gradient function to.

        Returns:
            torch.Tensor: The activation tensor with the attached gradient function.
        """
        moe_lpr_coef = self.config.moe_lpr_loss_coeff
        sequence_partition_group = None
        if self.config.moe_token_dispatcher_type == "alltoall_seq":
            sequence_partition_group = parallel_state.get_context_parallel_group()
            moe_lpr_coef /= parallel_state.get_tensor_model_parallel_world_size()
        else:
            sequence_partition_group = parallel_state.get_tensor_and_context_parallel_group()

        if parallel_state.get_tensor_model_parallel_world_size() > 1:
            lang_mask = lang_mask.transpose(0, 1).contiguous()
            lang_mask = torch.chunk(lang_mask, parallel_state.get_tensor_model_parallel_world_size(), dim=0)
            lang_mask = lang_mask[parallel_state.get_tensor_model_parallel_rank()]

        if lang_mask.any():
            lpr_loss = moe_lpr_loss_func(
                probs,
                moe_lpr_coef,
                lang_mask,
            )
            activation = MoEAuxLossAutoScaler.apply(activation, lpr_loss)
        else:
            lpr_loss = torch.tensor(0, device=probs.device, dtype=probs.dtype)

        save_to_aux_losses_tracker(
            "lpr_loss",
            lpr_loss / moe_lpr_coef,
            self.layer_number,
            self.config.num_layers,
            reduce_group=sequence_partition_group,
        )
        
        return activation

    def apply_load_balancing_loss(
        self,
        probs: torch.Tensor,
        num_local_tokens_per_expert: torch.Tensor,
        activation: torch.Tensor,
    ):
        """Applies auxiliary loss to the MoE layer.

        Args:
            probs (torch.Tensor): The probs output by the router for each token. [num_tokens, num_experts]
            num_local_tokens_per_expert (torch.Tensor): The number of tokens per expert. [num_experts]
            activation (torch.Tensor): The activation tensor to attach the gradient function to.

        Returns:
            torch.Tensor: The activation tensor with the attached gradient function.
        """
        moe_aux_loss_coeff = self.config.moe_aux_loss_coeff
        sequence_partition_group = None
        if self.config.moe_token_dispatcher_type == "alltoall_seq":
            sequence_partition_group = parallel_state.get_context_parallel_group()
            moe_aux_loss_coeff /= parallel_state.get_tensor_model_parallel_world_size()
        else:
            sequence_partition_group = parallel_state.get_tensor_and_context_parallel_group()

        aux_loss = switch_load_balancing_loss_func(
            probs,
            num_local_tokens_per_expert,
            self.topk,
            moe_aux_loss_coeff,
            sequence_partition_group=sequence_partition_group,
        )
        save_to_aux_losses_tracker(
            "load_balancing_loss",
            aux_loss / moe_aux_loss_coeff,
            self.layer_number,
            self.config.num_layers,
            reduce_group=sequence_partition_group,
        )
        activation = MoEAuxLossAutoScaler.apply(activation, aux_loss)
        return activation

    def apply_z_loss(self, logits):
        """Encourages the router's logits to remain small to enhance stability.
        Please refer to the ST-MoE paper (https://arxiv.org/pdf/2202.08906.pdf) for details.

        Args:
            logits (torch.Tensor): The logits of the router.

        Returns:
            torch.Tensor: The logits after applying the z-loss.
        """
        if self.config.moe_z_loss_coeff is not None and self.training:
            moe_z_loss_coeff = (
                self.config.moe_z_loss_coeff
                / parallel_state.get_tensor_and_context_parallel_world_size()
            )
            z_loss = z_loss_func(logits, moe_z_loss_coeff)
            logits = MoEAuxLossAutoScaler.apply(logits, z_loss)
            save_to_aux_losses_tracker(
                "z_loss", z_loss / moe_z_loss_coeff, self.layer_number, self.config.num_layers
            )
        return logits

    def apply_input_jitter(self, input: torch.Tensor):
        """Add noise to the input tensor.
        Refer to https://arxiv.org/abs/2101.03961.

        Args:
            input (Tensor): Input tensor.

        Returns:
            Tensor: Jittered input.
        """
        if self.config.moe_input_jitter_eps is not None:
            eps = self.config.moe_input_jitter_eps
            if self.input_jitter is None:
                self.input_jitter = torch.distributions.uniform.Uniform(
                    torch.tensor(1.0 - eps, device=input.device),
                    torch.tensor(1.0 + eps, device=input.device),
                ).rsample
            return input * self.input_jitter(input.shape)
        else:
            return input

    def routing(self, logits_token: torch.Tensor, logits_segment: torch.Tensor, lang_mask=None):
        """Top-k routing function

        Args:
            logits (torch.Tensor): Logits tensor after gating.

        Returns:
            probs (torch.Tensor): the probabilities tensor after load balancing.
            indices (torch.Tensor): the indices tensor after top-k selection.
        """
        
        # ======token level======

        # Apply Z-Loss
        logits_token = self.apply_z_loss(logits_token)

        if self.config.moe_token_dispatcher_type == "alltoall_seq":  # alltoall
            # Gather the logits from the TP region
            logits_token = gather_from_sequence_parallel_region(logits_token)

        if self.routing_type == "sinkhorn":
            scores, indices = self.sinkhorn_load_balancing(logits_token)
        elif self.routing_type == "aux_loss":  # here
            scores_token, indices_token = self.aux_loss_load_balancing(logits_token, lang_mask)
        elif self.routing_type == "none":
            # A naive top-k routing without load balancing
            scores, indices, _ = topk_softmax_with_capacity(
                logits_token,
                self.topk,
                capacity_factor=self.config.moe_expert_capacity_factor,
                pad_to_capacity=self.config.moe_pad_expert_input_to_capacity,
                drop_policy=self.config.moe_token_drop_policy,
                use_pre_softmax=self.config.moe_router_pre_softmax,
            )
        else:
            raise ValueError(f"Unsupported MoE routing type: {self.routing_type}")
            
            
        # =======sentence level============
        v = logits_segment.shape[0]
        topk_segment = int(v * self.config.capacity//(self.config.num_moe_experts//2))  # 8

        logits_segment = torch.softmax(logits_segment, dim=-1, dtype=torch.float32).transpose(0, 1)  # [num_expert,batch*p]
        
        # topk_gates, topk_indices :[num_experts,topk_segment] 
        token_selection_count = torch.zeros(logits_segment.shape[1], dtype=torch.long)
        topk_gates = torch.zeros((logits_segment.shape[0], topk_segment),device = logits_segment.device,dtype = torch.bfloat16)
        topk_indices = torch.zeros((logits_segment.shape[0], topk_segment), dtype=torch.long)

        for expert_idx in range(logits_segment.shape[0]):
            current_gate = logits_segment[expert_idx,:]
            sorted_scores, sorted_indices = current_gate.sort(descending=True)
            selected_count = 0
            idx = 0
            while selected_count < topk_segment and idx < logits_segment.shape[1]:
                token_idx = sorted_indices[idx]
                if token_selection_count[token_idx] < self.config.expert_choise_bucket:
                    topk_gates[expert_idx, selected_count] = sorted_scores[idx]
                    topk_indices[expert_idx, selected_count] = token_idx
                    selected_count += 1
                    token_selection_count[token_idx] += 1
                idx += 1
        
        # topk_indices的onehot
        expert_indices = torch.arange(self.config.num_moe_experts // 2).unsqueeze(1).expand(-1, topk_segment)
        token_indices = torch.arange(topk_segment).unsqueeze(0).expand(self.config.num_moe_experts // 2, -1)
        seg_select = torch.zeros(self.config.num_moe_experts // 2, topk_segment, v, device=logits_segment.device, dtype=torch.long)
        seg_select[expert_indices, token_indices, topk_indices] = 1         
                
        seg_select = torch.repeat_interleave(seg_select, self.config.context_window, dim=-1)

        ones_indices = (seg_select == 1).nonzero(as_tuple=True)  
        n_indices = ones_indices[0]
        s_indices = ones_indices[1] * self.config.context_window + torch.arange(len(ones_indices[2]), device=seg_select.device) % self.config.context_window
        idx_indices = ones_indices[2]
        output_size = (self.config.num_moe_experts // 2, topk_segment * self.config.context_window, seg_select.size(-1))
        output_seg_select = torch.sparse_coo_tensor(
            torch.stack([n_indices, s_indices, idx_indices]),
            torch.ones(len(n_indices), dtype=seg_select.dtype, device=seg_select.device), 
            size=output_size
        )

        output_seg_select = output_seg_select.to_dense()

        topk_gates = torch.repeat_interleave(topk_gates, self.config.context_window, dim=-1) / self.config.context_window # [num_experts,topk_segment*context_window]
        return scores_token, indices_token, topk_gates, output_seg_select

    def forward(self, input: torch.Tensor, lang_mask=None):
        """
        Forward pass of the router.

        Args:
            input (torch.Tensor): Input tensor.
        """
        self.hidden = input.shape[-1]

        # Apply input jitter
        input = self.apply_input_jitter(input)
        logits_token, logits_segment = self.gating(input) 
        
        logits_token = logits_token.view(-1, self.num_experts//2) # [batch*seq_length,num_experts//2]
        logits_segment = logits_segment.view(-1, self.num_experts//2) # [batch*p,num_experts//2]

        # [batch*seq_length, topk//2]  [batch*seq_length, topk//2]  [num_experts//2, topk_segment*context_window]  [num_experts//2, topk_segment*context_window,bacth*sequence_length]
        scores_token, indices_token, scores_segment, indices_segment = self.routing(logits_token, logits_segment, lang_mask)

        return scores_token, indices_token, scores_segment, indices_segment


class LanguageRouter(Router):
    """Route each token to the top-k experts."""

    def __init__(self, config: TransformerConfig) -> None:
        """Initialize the zero token dropping router.

        Args:
            config (TransformerConfig): The configuration for the transformer model.
        """
        super().__init__(config=config, is_language_router=True)

    def apply_lpr_loss(
        self,
        probs: torch.Tensor,
        lang_mask: torch.Tensor,
        activation: torch.Tensor,
    ):
        """Applies auxiliary loss to the MoE layer.

        Args:
            probs (torch.Tensor): The probs output by the router for each token. [num_tokens, num_experts]
            num_local_tokens_per_expert (torch.Tensor): The number of tokens per expert. [num_experts]
            activation (torch.Tensor): The activation tensor to attach the gradient function to.

        Returns:
            torch.Tensor: The activation tensor with the attached gradient function.
        """
        moe_lpr_coef = self.config.moe_lpr_loss_coeff
        sequence_partition_group = None
        if self.config.moe_token_dispatcher_type == "alltoall_seq":
            sequence_partition_group = parallel_state.get_context_parallel_group()
            moe_lpr_coef /= parallel_state.get_tensor_model_parallel_world_size()
        else:
            sequence_partition_group = parallel_state.get_tensor_and_context_parallel_group()

        if parallel_state.get_tensor_model_parallel_world_size() > 1:
            lang_mask = lang_mask.transpose(0, 1).contiguous()
            lang_mask = torch.chunk(lang_mask, parallel_state.get_tensor_model_parallel_world_size(), dim=0)
            lang_mask = lang_mask[parallel_state.get_tensor_model_parallel_rank()]
        
        if lang_mask.any():
            lpr_loss = moe_lpr_loss_func(
                probs,
                moe_lpr_coef,
                lang_mask,
            )
            activation = MoEAuxLossAutoScaler.apply(activation, lpr_loss)
        else:
            lpr_loss = torch.tensor(0, device=probs.device, dtype=probs.dtype)

        save_to_aux_losses_tracker(
            "lpr_loss",
            lpr_loss / moe_lpr_coef,
            self.layer_number,
            self.config.num_layers,
            reduce_group=sequence_partition_group,
        )
        
        return activation

    def routing(self, logits: torch.Tensor, lang_mask=None):
        """Top-k routing function

        Args:
            logits (torch.Tensor): Logits tensor after gating.

        Returns:
            probs (torch.Tensor): the probabilities tensor after load balancing.
            indices (torch.Tensor): the indices tensor after top-k selection.
        """
        original_shape = logits.shape
        logits = logits.view(-1, 2)

        if self.config.moe_token_dispatcher_type == "alltoall_seq":
            # Gather the logits from the TP region
            logits = gather_from_sequence_parallel_region(logits)

        scores = torch.softmax(logits, dim=-1, dtype=torch.float)
        scores = scores.to(logits.dtype)
        if self.training and self.config.moe_lpr_stage == 2:
            scores = self.apply_lpr_loss(scores, lang_mask, activation=scores)

        scores = scores.reshape(original_shape)
        return scores

    def forward(self, input: torch.Tensor, lang_mask=None):
        """
        Forward pass of the router.

        Args:
            input (torch.Tensor): Input tensor.
        """
        self.hidden = input.shape[-1]

        logits = self.gating(input)

        scores = self.routing(logits, lang_mask)

        return scores