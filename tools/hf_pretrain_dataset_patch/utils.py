import torch
import torch.distributed
from megatron.core import mpu
from megatron.training import get_args
from megatron.training.utils import get_ltor_masks_and_position_ids
from megatron.training import get_tokenizer

def get_batch_on_this_tp_rank_original(data_iterator):
    args = get_args()
    tokenizer = get_tokenizer()
    def _broadcast(item):
        torch.distributed.broadcast(item, mpu.get_tensor_model_parallel_src_rank(),
                                    group=mpu.get_tensor_model_parallel_group())

    if mpu.get_tensor_model_parallel_rank() == 0:

        if isinstance(data_iterator, dict):
            data = data_iterator
        else:
            data = next(data_iterator)

        tokens_ = data['input_ids'].long()
        labels_ = data['labels'].long()
        lang_mask_ = data['lang_mask'].bool() if args.moe_lpr_stage == 2 else None
        tokens = tokens_[:, :-1].contiguous()
        labels = labels_[:, 1:].contiguous()
        lang_mask = lang_mask_[:, :-1].contiguous() if lang_mask_ is not None else None
        # core/tensor_parallel/cross_entropy.py, target_mask = (target < vocab_start_index) | (target >= vocab_end_index)
        labels[labels == tokenizer.eos_token_id] = -100
        labels[labels == tokenizer.pad_token_id] = -100

        attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
            labels,
            -100,
            args.reset_position_ids,
            args.reset_attention_mask,
            args.eod_mask_loss)

        batch = {
            'tokens': tokens.cuda(non_blocking=True),
            'labels': labels.cuda(non_blocking=True),
            'loss_mask': loss_mask.cuda(non_blocking=True),
            'attention_mask': attention_mask.cuda(non_blocking=True),
            'position_ids': position_ids.cuda(non_blocking=True),
        }
        if lang_mask is not None:
            batch['lang_mask'] = lang_mask.cuda(non_blocking=True)

        #for k, v in batch.items():
        #    print(f"{torch.distributed.get_rank()}   {k}: {v.shape}  {v.dtype}")
        if args.pipeline_model_parallel_size == 1:
            _broadcast(batch['tokens'])
            _broadcast(batch['labels'])
            _broadcast(batch['loss_mask'])
            _broadcast(batch['attention_mask'])
            _broadcast(batch['position_ids'])
            if lang_mask is not None:
                _broadcast(batch['lang_mask'])

        elif mpu.is_pipeline_first_stage():
            _broadcast(batch['tokens'])
            _broadcast(batch['attention_mask'])
            _broadcast(batch['position_ids'])
            if lang_mask is not None:
                _broadcast(batch['lang_mask'])

        elif mpu.is_pipeline_last_stage():
            _broadcast(batch['labels'])
            _broadcast(batch['loss_mask'])
            _broadcast(batch['attention_mask'])
            if lang_mask is not None:
                _broadcast(batch['lang_mask'])

    else:

        tokens = torch.empty((args.micro_batch_size, args.seq_length), dtype=torch.int64,
                             device=torch.cuda.current_device())
        labels = torch.empty((args.micro_batch_size, args.seq_length), dtype=torch.int64,
                             device=torch.cuda.current_device())
        loss_mask = torch.empty((args.micro_batch_size, args.seq_length), dtype=torch.float32,
                                device=torch.cuda.current_device())
        
        attn_mask_bsz = args.micro_batch_size if args.reset_attention_mask else 1
        attention_mask = torch.empty((attn_mask_bsz, 1, args.seq_length, args.seq_length), dtype=torch.bool,
                                     device=torch.cuda.current_device())
        position_ids = torch.empty((args.micro_batch_size, args.seq_length), dtype=torch.int64,
                                   device=torch.cuda.current_device())
        
        if args.moe_lpr_stage == 2:
            lang_mask = torch.empty((args.micro_batch_size, args.seq_length), dtype=torch.bool,
                                 device=torch.cuda.current_device())
        else:
            lang_mask = None
        
        #print(f"{torch.distributed.get_rank()}   tokens: {tokens.shape}  {tokens.dtype}")
        #print(f"{torch.distributed.get_rank()}   labels: {labels.shape}  {labels.dtype}")
        #print(f"{torch.distributed.get_rank()}   loss_mask: {loss_mask.shape}  {loss_mask.dtype}")
        #print(f"{torch.distributed.get_rank()}   attention_mask: {attention_mask.shape}  {attention_mask.dtype}")
        #print(f"{torch.distributed.get_rank()}   position_ids: {position_ids.shape}  {position_ids.dtype}")
        #print(f"{torch.distributed.get_rank()}   lang_mask: {lang_mask.shape}  {lang_mask.dtype}")
        if args.pipeline_model_parallel_size == 1:
            _broadcast(tokens)
            _broadcast(labels)
            _broadcast(loss_mask)
            _broadcast(attention_mask)
            _broadcast(position_ids)
            if lang_mask is not None:
                _broadcast(lang_mask)

        elif mpu.is_pipeline_first_stage():
            labels = None
            loss_mask = None

            _broadcast(tokens)
            _broadcast(attention_mask)
            _broadcast(position_ids)
            if lang_mask is not None:
                _broadcast(lang_mask)

        elif mpu.is_pipeline_last_stage():
            tokens = None
            position_ids = None

            _broadcast(labels)
            _broadcast(loss_mask)
            _broadcast(attention_mask)
            if lang_mask is not None:
                _broadcast(lang_mask)

        batch = {
            'tokens': tokens,
            'labels': labels,
            'loss_mask': loss_mask,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
        }
        if lang_mask is not None:
            batch['lang_mask'] = lang_mask.cuda(non_blocking=True)

    return batch
