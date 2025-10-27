#!/bin/bash
#
# This script launches the QLoRA-style fine-tuning process for the OpenVision-Instruct project.
# It adapts the LLaVA-OneVision training framework for our specific dataset and hardware.

# --- Configuration ---

# Project & Path Variables
# Note: These paths are relative to the root of the OpenVision-Instruct project.
PROJECT_ROOT=$(pwd)
AIAK_TRAINING_PATH="${PROJECT_ROOT}/vendor/LLaVA-OneVision"
AIAK_MEGATRON_PATH="${AIAK_TRAINING_PATH}/aiak_megatron"
TOKENIZER_PATH="${PROJECT_ROOT}/vendor/LLaVA-OneVision-1.5-4B-stage0"
BASE_MODEL_PATH="${PROJECT_ROOT}/vendor/LLaVA-OneVision-1.5-4B-stage0"
DATA_FILE_PATH="${PROJECT_ROOT}/data/finetune_data_multimodal.json"
SAVE_CKPT_PATH="${PROJECT_ROOT}/checkpoints/OpenVision-Instruct-4B-adapter"
TENSORBOARD_PATH="${PROJECT_ROOT}/runs/OpenVision-Instruct-4B-adapter"

# Hardware & Batch Size Configuration
# Tailored for a single NVIDIA RTX 4090 with 24GB VRAM.
GPUS_PER_NODE=1
TP=1 # Tensor Parallelism
PP=1 # Pipeline Parallelism
MBS=1 # Micro Batch Size
GBS=16 # Global Batch Size (adjust based on memory)

# Training Hyperparameters
SEQ_LEN=4096
TRAIN_ITERS=1000 # Adjust as needed for convergence
LR=1.0e-5

# --- Script Execution ---

mkdir -p "$SAVE_CKPT_PATH"
mkdir -p "$TENSORBOARD_PATH"

# Set distributed arguments for a single node, single GPU setup
DISTRIBUTED_ARGS=(
    --nproc_per_node "$GPUS_PER_NODE"
)

# Configure the model architecture
MODEL_ARGS=(
    --model-name llava-ov-1.5-4b
)

# Configure the dataset and tokenizer
# We use the 'multimodal' format which is built-in.
DATA_ARGS=(
    --tokenizer-type HFTokenizer
    --hf-tokenizer-path "$TOKENIZER_PATH"
    --data-path "$DATA_FILE_PATH"
    --sft-dataset multimodal
    --dataloader-type external
    --split 100,0,0
    --num-workers 2
    --chat-template qwen2-vl
)

# Configure the fine-tuning process
TRAINING_ARGS=(
    --image-resolution 1000
    --training-phase sft
    --trainable-modules adapter  # This is the key for QLoRA-style fine-tuning
    --seq-length "${SEQ_LEN}"
    --max-position-embeddings 32768
    --micro-batch-size "${MBS}"
    --global-batch-size "${GBS}"
    --lr "${LR}"
    --min-lr 1.0e-6
    --clip-grad 1.0
    --weight-decay 0
    --optimizer adam
    --adam-beta1 0.9
    --adam-beta2 0.99
    --train-iters "${TRAIN_ITERS}"
    --lr-decay-iters "${TRAIN_ITERS}"
    --lr-decay-style cosine
    --lr-warmup-fraction 0.01
    --bf16
    --load "$BASE_MODEL_PATH"
    --save "$SAVE_CKPT_PATH"
    --save-interval 200
    --ckpt-format torch
    --recompute-granularity full
    --recompute-method uniform
)

# Configure parallelism and performance settings
MODEL_PARALLEL_ARGS=(
    --attention-backend flash
    --pipeline-model-parallel-size "${PP}"
    --tensor-model-parallel-size "${TP}"
    --use-distributed-optimizer
    --distributed-backend nccl
)

# Configure logging
LOGGING_ARGS=(
    --log-interval 10
    --tensorboard-dir "${TENSORBOARD_PATH}"
)

# Set environment variables and launch training
pip cache purge
pip install -r requirements.txt --no-build-isolation
export PYTHONPATH="$AIAK_MEGATRON_PATH:$AIAK_TRAINING_PATH:$PYTHONPATH"
export LD_LIBRARY_PATH="${PROJECT_ROOT}/.venv/lib/python3.11/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH"
export CUDNN_PATH="${PROJECT_ROOT}/.venv/lib/python3.11/site-packages/nvidia/cudnn/"
export CXXFLAGS="-I${CUDNN_PATH}/include"
export CFLAGS="-I${CUDNN_PATH}/include"

torchrun "${DISTRIBUTED_ARGS[@]}" \
    "$AIAK_TRAINING_PATH/aiak_training_llm/train.py" \
    "${MODEL_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${TRAINING_ARGS[@]}" \
    "${MODEL_PARALLEL_ARGS[@]}" \
    "${LOGGING_ARGS[@]}"

echo "--- Fine-Tuning Complete ---"
