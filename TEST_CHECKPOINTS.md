# How to Test Your Image Editing Checkpoints

## What This Does

Tests your trained model with **actual image editing instructions** and shows you:
1. **Input image** (e.g., image with "Alice" text)
2. **Expected output** (image without "Alice")  
3. **Model's response** (text describing what to edit)

## Important Understanding

Your model is trained to **generate editing instructions** (text), NOT to produce edited images.

**What to expect:**
- ✅ Model describes the edit: "Remove Alice from the center"
- ✅ Model acknowledges: "I will edit the image as requested"
- ❌ Model does NOT output an edited image (that's a future enhancement)

## Quick Start - Test Single Sample

```bash
# Test sample_604 with iter_3500 (best checkpoint)
./test_checkpoint.sh checkpoints/hf_models/iter_0003500 sample_604

# Test sample_126 with iter_500
./test_checkpoint.sh checkpoints/hf_models/iter_0000500 sample_126
```

**Time:** ~2-3 minutes per sample

**Output:** 
- Comparison image: `evaluation/visual_tests/sample_604_comparison.jpg`
- Results JSON: `evaluation/visual_tests/test_results.json`

## Test All 10 Samples

```bash
# Test all samples with iter_3500
./test_all_samples.sh checkpoints/hf_models/iter_0003500
```

**Time:** ~5-10 minutes

**Output:**
- 10 comparison images in `evaluation/visual_tests/`
- Complete results in `evaluation/visual_tests/test_results.json`

## Available Test Samples

All samples in `evaluation/test_samples/`:

1. **sample_126** - "Remove the word 'DATE' from the card."
2. **sample_147** - "Change the background color to blue"
3. **sample_170** - [view sample_170.json]
4. **sample_194** - [view sample_194.json]
5. **sample_282** - "Remove the word 'Sfice' under the Location column."
6. **sample_451** - [view sample_451.json]
7. **sample_470** - [view sample_470.json]
8. **sample_524** - [view sample_524.json]
9. **sample_601** - [view sample_601.json]
10. **sample_604** - "Remove the word \"Alice\" from the center of the image."

## What the Comparison Image Shows

```
┌─────────────────────────────────────────────────────────────┐
│ Sample: sample_604                                          │
│ Instruction: Remove the word "Alice" from center of image  │
├─────────────────┬───────────────────────────────────────────┤
│                 │                                           │
│  INPUT IMAGE    │      EXPECTED OUTPUT IMAGE                │
│  (with "Alice") │      (without "Alice")                    │
│                 │                                           │
├─────────────────┴───────────────────────────────────────────┤
│ MODEL OUTPUT (text):                                        │
│ "I will edit the image as requested to remove Alice..."    │
└─────────────────────────────────────────────────────────────┘
```

## How to Evaluate the Results

### ✅ Signs the Model Learned:

1. **Coherent text responses**
   - Complete sentences in English
   - Grammatically correct
   - Not random tokens/gibberish

2. **Task-relevant**
   - Mentions the editing operation
   - References elements from the instruction
   - Acknowledges the task

3. **Examples of good responses:**
   - "I will edit the image to remove Alice from the center"
   - "Acknowledged. I will remove the word 'Alice' as requested"
   - "To edit: locate 'Alice' text and delete it"

### ❌ Signs of Model Collapse:

1. **Gibberish**
   - Random characters: `']>;\n tritur.arr>Add_server进城맵...`
   - Mixed languages randomly
   - No structure

2. **Completely irrelevant**
   - Describes the image instead: "This image shows..."
   - Answers a question: "Yes" / "No"
   - Unrelated content

3. **Silent failure**
   - Empty response
   - Only special tokens
   - Repetitive loops

## Compare Checkpoints

```bash
# Test both checkpoints on same sample
./test_checkpoint.sh checkpoints/hf_models/iter_0000500 sample_604
./test_checkpoint.sh checkpoints/hf_models/iter_0003500 sample_604

# Compare outputs
cat evaluation/visual_tests/test_results.json | jq '.[] | {id, response: .model_response[:150]}'
```

**Expected:** iter_0003500 should have better/more detailed responses than iter_0000500

## View Results

```bash
# View comparison images
ls -lh evaluation/visual_tests/*.jpg

# Open with image viewer
eog evaluation/visual_tests/sample_604_comparison.jpg  # Linux
open evaluation/visual_tests/sample_604_comparison.jpg  # macOS

# View JSON results
cat evaluation/visual_tests/test_results.json | jq '.'

# View just responses
cat evaluation/visual_tests/test_results.json | jq '.[] | {id: .sample_id, response: .model_response}'
```

## Direct Python Usage

```bash
# Custom test
python3 scripts/test_with_visual_comparison.py \
    --checkpoint checkpoints/hf_models/iter_0003500 \
    --sample-id sample_604 \
    --samples-dir evaluation/test_samples \
    --output-dir evaluation/visual_tests \
    --max-new-tokens 512
```

## Troubleshooting

### "ModuleNotFoundError: No module named 'transformers'"

```bash
pip install transformers torch pillow accelerate
```

### "CUDA out of memory"

```bash
# Test fewer samples at once, or use CPU
export CUDA_VISIBLE_DEVICES=""
```

### "Model outputs gibberish"

This means the checkpoint didn't train properly. Try:
- Testing an earlier checkpoint (iter_500 vs iter_3500)
- Checking training logs for issues
- Verifying the checkpoint files exist

### Image viewer not opening

```bash
# Copy to a location you can access
cp evaluation/visual_tests/sample_604_comparison.jpg ~/Desktop/
```

## What Success Looks Like

Based on your MME evaluation, both checkpoints should output something like:

**"I will edit the image as requested."**

This is **correct** - it shows the model learned to acknowledge editing instructions!

The visual comparison helps you see:
1. The model responds coherently (not gibberish) ✅
2. The response relates to the editing task ✅
3. There's consistency across samples ✅

## Next Steps After Validation

1. **If responses are coherent:** Training succeeded! 🎉
   - Document results
   - Prepare model for release
   - Consider training for actual image generation (future work)

2. **If responses are gibberish:** Investigate training
   - Check training logs
   - Review dataset format
   - Try different hyperparameters

## Quick Reference

```bash
# Single sample test (fastest)
./test_checkpoint.sh checkpoints/hf_models/iter_0003500 sample_604

# All samples test (comprehensive)
./test_all_samples.sh checkpoints/hf_models/iter_0003500

# View results
cat evaluation/visual_tests/test_results.json | jq '.'
```

---

**Ready to test your checkpoints!** 🚀
