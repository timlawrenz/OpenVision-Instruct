#!/bin/bash
# Quick test script for visual comparison

CHECKPOINT=${1:-"checkpoints/hf_models/iter_0003500"}
SAMPLE=${2:-"sample_604"}

echo "========================================================================"
echo "Visual Comparison Test"
echo "========================================================================"
echo ""
echo "Checkpoint: $CHECKPOINT"
echo "Sample: $SAMPLE"
echo ""
echo "This will:"
echo "  1. Load your trained model"
echo "  2. Send the input image + instruction"
echo "  3. Generate model response"
echo "  4. Create a comparison image showing:"
echo "     - Input image"
echo "     - Expected output image"
echo "     - Model's text response"
echo ""
echo "Starting test..."
echo ""

python3 scripts/test_with_visual_comparison.py \
    --checkpoint "$CHECKPOINT" \
    --sample-id "$SAMPLE" \
    --samples-dir evaluation/test_samples \
    --output-dir evaluation/visual_tests

echo ""
echo "========================================================================"
echo "✅ Test complete!"
echo "========================================================================"
echo ""
echo "View the comparison image:"
echo "  evaluation/visual_tests/${SAMPLE}_comparison.jpg"
echo ""
echo "View results JSON:"
echo "  cat evaluation/visual_tests/test_results.json | jq '.'"
echo ""
echo "Test other samples:"
echo "  ./test_checkpoint.sh $CHECKPOINT sample_126"
echo "  ./test_checkpoint.sh $CHECKPOINT sample_282"
echo ""
echo "Available samples:"
ls evaluation/test_samples/*.json | xargs -n1 basename | sed 's/.json$//' | sed 's/^/  /'
echo ""
