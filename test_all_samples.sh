#!/bin/bash
# Test all samples and create visual comparisons

CHECKPOINT=${1:-"checkpoints/hf_models/iter_0003500"}

echo "========================================================================"
echo "Testing ALL Samples - Visual Comparison"
echo "========================================================================"
echo ""
echo "Checkpoint: $CHECKPOINT"
echo ""
echo "This will test all 10 samples and create comparison images."
echo "Estimated time: 5-10 minutes"
echo ""
read -p "Continue? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

echo ""
echo "Starting comprehensive test..."
echo ""

python3 scripts/test_with_visual_comparison.py \
    --checkpoint "$CHECKPOINT" \
    --samples-dir evaluation/test_samples \
    --output-dir evaluation/visual_tests \
    --all-samples

echo ""
echo "========================================================================"
echo "✅ All tests complete!"
echo "========================================================================"
echo ""
echo "Comparison images saved to: evaluation/visual_tests/"
echo ""
echo "View all comparisons:"
echo "  ls -lh evaluation/visual_tests/*.jpg"
echo ""
echo "View results summary:"
echo "  cat evaluation/visual_tests/test_results.json | jq '.[] | {id: .sample_id, instruction: .instruction, response: .model_response[:100]}'"
echo ""
