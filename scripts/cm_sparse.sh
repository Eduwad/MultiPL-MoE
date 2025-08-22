#!/bin/bash

# Runs Mixtral 8x7B model
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PATH=/usr/local/cuda-12.1/bin:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-12.1/lib64
export CUDA_HOME=/usr/local/cuda-12.1
export LD_LIBRARY_PATH=/nju-megatron/lib/gcc/x86_64-conda-linux-gnu/11.4.0:$LD_LIBRARY_PATH
# python path
WORK_DIR=
MEGATRONN_PATH=$WORK_DIR/Megatron-LM
HF_DATA_PATCH_PATH=$MEGATRONN_PATH/tools
export PYTHONPATH=${MEGATRONN_PATH}:${HF_DATA_PATCH_PATH}:$PYTHONPATH
export NCCL_IB_DISABLE=0
export NCCL_PXN_DISABLE=1
export NCCL_IB_TIMEOUT=22
export NCCL_IB_RETRY_CNT=13
export NCCL_DEBUG=INFO
export NCCL_DEBUG_FILE=
export NCCL_TIMEOUT=1000000000
export NCCL_IB_GID_INDEX=3

CHECKPOINT_PATH=
TOKENIZER_MODEL=
DATA_PATH=
SAVE_PATH=

LPR_LOSS_COEFF=1e-3
AUX_LOSS_COEFF=1e-2
LPR_STAGE=1
NUM_EXPERTS=12
TOPK=4

LR=7e-6
MICRO_BSZ=8
GLOBAL_BSZ=64
TRAIN_STEPS=93000
SAVE_STEPS=40000
EVAL_STEPS=1000
WARMUP_STEPS=100


TP=2
EP=1
PP=1

GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=9007
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))


DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NNODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)


MODEL_ARGS=(
    --transformer-impl transformer_engine
    --use-mcore-models
    --disable-bias-linear
    --seq-length 1024
    --max-position-embeddings 1024
    --num-layers 24
    --hidden-size 2048
    --ffn-hidden-size 5504
    --num-attention-heads 16
    --init-method-std 0.02
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --normalization RMSNorm
    --position-embedding-type rope
    --no-rope-fusion
    --use-rotary-position-embeddings
    --swiglu
    --untie-embeddings-and-output-weights
    --no-masked-softmax-fusion
    --no-position-embedding
    --rotary-base 500000
    --ckpt-format torch
    --add-qkv-bias
)


MOE_ARGS=(
    --num-experts $NUM_EXPERTS
    --moe-router-topk $TOPK
    --moe-router-load-balancing-type aux_loss
    --moe-aux-loss-coeff $AUX_LOSS_COEFF
    --moe-lpr-loss-coeff $LPR_LOSS_COEFF
    --moe-lpr-stage $LPR_STAGE
    --moe-token-dispatcher-type alltoall
    --overlap-param-gather
    --overlap-grad-reduce
    --moe-router-pre-softmax
    --context-window 64
    --capacity 1.0
    --expert-choise-bucket 4
    --weight-token 0.7
    --weight-sentence 0.3
)


DATA_ARGS=(
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model ${TOKENIZER_MODEL}
    --data-path $DATA_PATH
    --split 998,1,1
)


TRAINING_ARGS=(
    --micro-batch-size $MICRO_BSZ
    --global-batch-size $GLOBAL_BSZ
    --lr $LR
    --train-iters $TRAIN_STEPS
    --lr-decay-iters $TRAIN_STEPS
    --lr-decay-style cosine
    --min-lr 0
    --weight-decay 0.1
    --lr-warmup-iters $WARMUP_STEPS
    --clip-grad 1.0
    --use-flash-attn
    --bf16
    --distributed-timeout-minutes 1000000
    --no-save-optim
    --no-save-rng
)


MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size $TP
    --pipeline-model-parallel-size $PP
    --expert-model-parallel-size $EP
    --use-distributed-optimizer
    --sequence-parallel 
)

LOGGING_ARGS=(
    --log-interval 1 \
    --save-interval $SAVE_STEPS \
    --eval-interval 100000 \
    --eval-iters $EVAL_STEPS \
    --save $SAVE_PATH \
    --load $CHECKPOINT_PATH \
    --tensorboard-dir "${SAVE_PATH}/tensorboard" \
    --no-load-optim \
    --no-load-rng \
    --log-throughput \
    --log-progress \
    --log-memory-to-tensorboard  \
    --log-timers-to-tensorboard
)

if [ -n "${WANDB_API_KEY}" ]; then
    LOGGING_ARGS+=( 
        --wandb-project ${WANDB_PROJECT:-"sparse_expert"}
        --wandb-exp-name ${WANDB_NAME:-"cm_3b_sparse_4exp_2tp_2ep"}
    )
fi

if [ ${NODE_RANK} -eq 1 ];then
torchrun ${DISTRIBUTED_ARGS[@]} ${MEGATRONN_PATH}/pretrain_gpt.py \
    ${MODEL_ARGS[@]} \
    ${MOE_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${LOGGING_ARGS[@]} > $WORK_DIR/Megatron-LM/scripts/llama_4exp_top2_sparse.out 2>&1
else
echo "training"
torchrun ${DISTRIBUTED_ARGS[@]} ${MEGATRONN_PATH}/pretrain_gpt.py \
    ${MODEL_ARGS[@]} \
    ${MOE_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${LOGGING_ARGS[@]}
fi
exit
