#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Guide:
# This script supports distributed training on multi-gpu workers (as well as single-worker training).
# Please set the options below according to the comments.
# For multi-gpu workers training, these options should be manually set for each worker.
# After setting the options, please run the script on each worker.

# Number of GPUs per GPU worker
GPUS_PER_NODE=$(python -c 'import torch; print(torch.cuda.device_count())')

# Number of GPU workers, for single-worker training, please set to 1
NNODES=${NNODES:-1}

# The rank of this worker, should be in {0, ..., WORKER_CNT-1}, for single-worker training, please set to 0
NODE_RANK=${NODE_RANK:-0}

# The ip address of the rank-0 worker, for single-worker training, please set to localhost
MASTER_ADDR=${MASTER_ADDR:-localhost}

# The port for communication
MASTER_PORT=${MASTER_PORT:-6001}


MODEL="/home/zkding/projects/llm_model_resource/Qwen/Qwen2-7B-Instruct" # Set the path if you do not want to load from huggingface directly
# ATTENTION: specify the path to your training data, which should be a json file consisting of a list of conversations.
# See https://qwen.readthedocs.io/en/latest/training/SFT/example.html#data-preparation for more information.

BASE_PATH="/home/zkding/projects/coding/python/learn/learn_LLM/qwen/qwen2/finetune"
PY_PATH="finetune.py"

DATA="io_file/input_data/example_data.jsonl"
DS_CONFIG_PATH="ds_config_zero3.json"
OUTPUT_DIR="io_file/output/v1"

USE_LORA=True
Q_LORA=False

# 路径修复
BASE_PATH=$(echo "$BASE_PATH" | sed 's:/*$::')          # 去除末尾 /

DATA=$(echo "$DATA" | sed 's:^/::')                     # 去除开头 /
DS_CONFIG_PATH=$(echo "$DS_CONFIG_PATH" | sed 's:^/::') # 去除开头 /
OUTPUT_DIR=$(echo "$OUTPUT_DIR" | sed 's:^/::')         # 去除开头 /
PY_PATH=$(echo "$PY_PATH" | sed 's:^/::')               # 去除开头 /

DATA="$BASE_PATH/$DATA"                                 # 路径拼接
DS_CONFIG_PATH="$BASE_PATH/$DS_CONFIG_PATH"             # 路径拼接
OUTPUT_DIR="$BASE_PATH/$OUTPUT_DIR"                     # 路径拼接
PY_PATH="$BASE_PATH/$PY_PATH"                           # 路径拼接


function usage() {
    echo '
Usage: bash finetune.sh [-m MODEL_PATH] [-d DATA_PATH] [--deepspeed DS_CONFIG_PATH] [--use_lora USE_LORA] [--q_lora Q_LORA]
'
}

while [[ "$1" != "" ]]; do
    case $1 in
        -m | --model )
            shift
            MODEL=$1
            ;;
        -d | --data )
            shift
            DATA=$1
            ;;
        --deepspeed )
            shift
            DS_CONFIG_PATH=$1
            ;;
        --use_lora  )
            shift
            USE_LORA=$1
            ;;
        --q_lora    )
            shift
            Q_LORA=$1
            ;;
        -h | --help )
            usage
            exit 0
            ;;
        * )
            echo "Unknown argument ${1}"
            exit 1
            ;;
    esac
    shift
done

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

torchrun $DISTRIBUTED_ARGS $PY_PATH \
    --model_name_or_path $MODEL \
    --data_path $DATA \
    --bf16 True \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 5 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 10 \
    --save_total_limit 10 \
    --learning_rate 3e-4 \
    --weight_decay 0.01 \
    --adam_beta2 0.95 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --report_to "none" \
    --model_max_length 512 \
    --lazy_preprocess True \
    --use_lora ${USE_LORA} \
    --q_lora ${Q_LORA} \
    --gradient_checkpointing \
    --deepspeed ${DS_CONFIG_PATH}
