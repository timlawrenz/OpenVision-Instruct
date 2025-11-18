# 🎯 Test Your Checkpoints - One Command

## TL;DR

```bash
./test_checkpoint.sh checkpoints/hf_models/iter_0003500 sample_604
```

This will:
1. Load your trained model
2. Send it the input image + instruction: "Remove the word 'Alice' from the center"
3. Show you the model's text response
4. Create a comparison image showing INPUT | EXPECTED | MODEL RESPONSE

**Time:** 2-3 minutes

## What You'll Get

A comparison image saved to:
```
evaluation/visual_tests/sample_604_comparison.jpg
```

Showing:
- **Left:** Input image (with "Alice")
- **Right:** Expected output (without "Alice")
- **Bottom:** Model's text response

## What to Look For

### ✅ Success (Model Learned):
```
Model Output: "I will edit the image as requested to remove Alice."
```
or
```
Model Output: "Acknowledged. I will remove the word 'Alice' from the center."
```

### ❌ Failure (Model Collapsed):
```
Model Output: ']>;\n tritur.arr>Add_server进城맵...'
```

## Test More Samples

```bash
# Test sample_282 (remove "Sfice" from table)
./test_checkpoint.sh checkpoints/hf_models/iter_0003500 sample_282

# Test sample_126 (remove "DATE" from card)
./test_checkpoint.sh checkpoints/hf_models/iter_0003500 sample_126

# Test all 10 samples
./test_all_samples.sh checkpoints/hf_models/iter_0003500
```

## Compare Checkpoints

```bash
# Test iter_500
./test_checkpoint.sh checkpoints/hf_models/iter_0000500 sample_604

# Test iter_3500
./test_checkpoint.sh checkpoints/hf_models/iter_0003500 sample_604

# Compare results
cat evaluation/visual_tests/test_results.json | jq '.[] | {id, response: .model_response}'
```

## Files Created

✅ **test_checkpoint.sh** - Test single sample (easiest)
✅ **test_all_samples.sh** - Test all 10 samples
✅ **scripts/test_with_visual_comparison.py** - Python implementation
✅ **TEST_CHECKPOINTS.md** - Complete guide

## Full Documentation

See **TEST_CHECKPOINTS.md** for:
- Detailed explanation
- Troubleshooting
- How to interpret results
- All available samples

---

**Run the command above to test your checkpoint now!** 🚀
