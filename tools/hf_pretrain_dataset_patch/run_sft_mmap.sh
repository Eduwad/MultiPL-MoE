#! /bin/bash
START_TIME=$SECONDS

WORK_DIR=/root/work/huangxin/nanda
MEGATRONN_PATH=$WORK_DIR/Megatron-LM
HF_DATA_PATCH_PATH=$MEGATRONN_PATH/tools
export PYTHONPATH=${MEGATRONN_PATH}:${HF_DATA_PATCH_PATH}:$PYTHONPATH

input_data_path=$1
tokenizer=$2
seq_len=$3
output_data_path=$4
load_dir=$5

python sft_mmap_data.py\
  --input ${input_data_path} \
  --output-prefix ${output_data_path} \
  --tokenizer-type HuggingFaceTokenizer \
  --tokenizer-model $tokenizer \
  --load ${load_dir} \
  --seq-length ${seq_len} \
  --workers 8 \
  --partitions 1 \

ELAPSED_TIME=$(($SECONDS - $START_TIME))
echo "$(($ELAPSED_TIME/60)) min $(($ELAPSED_TIME%60)) sec"
