#!/bin/bash
# Quick test of Megatron inference with 2 samples

set -e

echo "========================================"
echo "Testing Megatron Inference (2 samples)"
echo "========================================"

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

CHECKPOINT_PATH="stage_2_instruct_llava_ov_4b/iter_0000500"
TOKENIZER_PATH="LLaVA-OneVision-1.5-4B-stage0"
TEST_SAMPLES="evaluation/test_samples/test_samples.json"
OUTPUT_FILE="evaluation/test_inference_results.json"

# Check paths exist
if [ ! -d "$CHECKPOINT_PATH" ]; then
    echo "ERROR: Checkpoint not found: $CHECKPOINT_PATH"
    exit 1
fi

if [ ! -d "$TOKENIZER_PATH" ]; then
    echo "ERROR: Tokenizer not found: $TOKENIZER_PATH"
    exit 1
fi

if [ ! -f "$TEST_SAMPLES" ]; then
    echo "ERROR: Test samples not found: $TEST_SAMPLES"
    exit 1
fi

echo "Checkpoint: $CHECKPOINT_PATH"
echo "Tokenizer: $TOKENIZER_PATH"
echo "Test samples: $TEST_SAMPLES"
echo "Output: $OUTPUT_FILE"
echo ""

# Activate virtual environment
source .venv/bin/activate

# Add cuDNN 9 libraries from venv to LD_LIBRARY_PATH
CUDNN_PATH="$PROJECT_ROOT/.venv/lib/python3.11/site-packages/nvidia/cudnn/lib"
export LD_LIBRARY_PATH="$CUDNN_PATH:$LD_LIBRARY_PATH"

# Run inference with torchrun (single GPU)
export CUDA_DEVICE_MAX_CONNECTIONS=1

python -u scripts/run_megatron_inference.py \
    --load "$CHECKPOINT_PATH" \
    --hf-tokenizer-path "$TOKENIZER_PATH" \
    --tokenizer-type HFTokenizer \
    --test-samples "$TEST_SAMPLES" \
    --output "$OUTPUT_FILE" \
    --num-samples 2 \
    --temperature 0.7 \
    --top-k 50 \
    --top-p 0.9 \
    --num-tokens-to-generate 512 \
    --max-batch-size 1 \
    --use-checkpoint-args \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1

echo ""
echo "========================================"
echo "Inference test complete!"
echo "Results saved to: $OUTPUT_FILE"
echo "========================================"
