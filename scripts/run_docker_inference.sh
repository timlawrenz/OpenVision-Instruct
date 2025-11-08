#!/bin/bash
# Docker Inference Script - Run with sudo
# Usage: sudo ./scripts/run_docker_inference.sh

set -e

# Create logs directory
mkdir -p logs

# Log file with timestamp
LOGFILE="logs/docker_inference_$(date +%Y%m%d_%H%M%S).log"

echo "================================================================================"
echo "Running Megatron Inference in Docker"
echo "================================================================================"
echo ""
echo "Container: nvcr.io/nvidia/pytorch:24.02-py3"
echo "Checkpoint: stage_2_instruct_llava_ov_4b/iter_0000500"
echo "Test samples: 2"
echo "Log file: $LOGFILE"
echo ""

# Check if running with sudo
if [ "$EUID" -ne 0 ]; then 
    echo "❌ Please run with sudo:"
    echo "   sudo bash scripts/run_docker_inference.sh"
    exit 1
fi

# Get the actual user (not root when using sudo)
ACTUAL_USER=${SUDO_USER:-$USER}
ACTUAL_HOME=$(eval echo ~$ACTUAL_USER)

echo "Running as user: $ACTUAL_USER"
echo ""

docker run --gpus all --rm \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v $(pwd):/workspace \
  -w /workspace \
  nvcr.io/nvidia/pytorch:24.02-py3 \
  bash -c '
    echo "=== Installing Python dependencies ==="
    pip install -q transformers datasets pillow einops einops-exts sentencepiece webdataset accelerate megatron-energon
    
    echo ""
    echo "=== Adding vendor paths to Python path ==="
    export PYTHONPATH=/workspace/vendor/LLaVA-OneVision:/workspace/vendor/LLaVA-OneVision/aiak_megatron:$PYTHONPATH
    
    echo ""
    echo "=== Setting up distributed environment ==="
    export MASTER_ADDR=localhost
    export MASTER_PORT=6000
    export RANK=0
    export WORLD_SIZE=1
    
    echo ""
    echo "=== Running Megatron inference ==="
    python scripts/run_megatron_inference_simple.py \
      --load stage_2_instruct_llava_ov_4b/iter_0000500 \
      --tokenizer-path LLaVA-OneVision-1.5-4B-stage0 \
      --num-layers 36 \
      --hidden-size 2560 \
      --num-attention-heads 32 \
      --seq-length 32768 \
      --max-position-embeddings 32768 \
      --test-samples evaluation/test_samples/test_samples.json \
      --output evaluation/finetuned_results.json \
      --num-samples 2 \
      --max-new-tokens 256 \
      --tensor-model-parallel-size 1 \
      --pipeline-model-parallel-size 1
  ' 2>&1 | tee "$LOGFILE"

echo ""
echo "================================================================================"
echo "✅ Inference complete!"
echo "================================================================================"
echo ""
echo "Results saved to: evaluation/finetuned_results.json"
echo "Log saved to: $LOGFILE"
echo ""
echo "View results:"
echo "  cat evaluation/finetuned_results.json | jq '.'"
echo ""
echo "View full log:"
echo "  cat $LOGFILE"
echo ""
