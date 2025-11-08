# Inference Setup Progress

**Date:** 2025-11-04  
**Status:** ⏳ Moving files to NFS, inference script ready

---

## 🔧 Disk Space Management

### Issue
Local disk was 100% full (3.3T/3.5T used)

### Solution
Moving large files to NFS mounts:

1. **Base Model (17GB)** ✅ MOVED
   - From: `LLaVA-OneVision-1.5-4B-stage0/`
   - To: `data/checkpoints/LLaVA-OneVision-1.5-4B-stage0/`
   - Symlink created

2. **Training Checkpoint (9GB)** ⏳ IN PROGRESS
   - From: `stage_2_instruct_llava_ov_4b/`
   - To: `data/checkpoints/stage_2_instruct_llava_ov_4b/`
   - Needs: `sudo chown -R tim:tim` (files owned by root)

3. **WebDataset (82GB)** ⏳ IN PROGRESS  
   - From: `data/OpenGPT-4o-Image-wds/`
   - To: `data/OpenGPT-4o-Image/OpenGPT-4o-Image-wds/`
   - Currently moving over NFS (will take time)

### Expected Result
- ~108GB freed from local disk
- All large files on NFS mounts
- Symlinks in place for convenience

---

## 📝 Inference Script Created

**Location:** `scripts/run_inference.py`

### Features
- ✅ Loads HuggingFace model (base or fine-tuned)
- ✅ Runs inference on test samples
- ✅ Saves results to JSON
- ✅ Handles errors gracefully
- ✅ Supports CUDA or CPU

### Usage

```bash
# Test on base model (baseline)
python scripts/run_inference.py \
    --model LLaVA-OneVision-1.5-4B-stage0 \
    --test-samples evaluation/test_samples/test_samples.json \
    --output evaluation/baseline_results.json

# Test on 2 samples first (quick test)
python scripts/run_inference.py \
    --model LLaVA-OneVision-1.5-4B-stage0 \
    --num-samples 2 \
    --output evaluation/baseline_quick.json
```

### What It Does

For each test sample:
1. Loads the image
2. Formats instruction with `<image>` tag
3. Runs model.generate()
4. Saves response to JSON

Output format:
```json
[
  {
    "id": "sample_126",
    "instruction": "Remove the word 'DATE' from the card.",
    "response": "Model's response here...",
    "image_path": "evaluation/test_samples/sample_126.input.jpg"
  }
]
```

---

## 🎯 Next Steps

### 1. Wait for File Moves to Complete

Check progress:
```bash
# Monitor moves
ps aux | grep mv

# Check disk space
df -h /

# When complete, should have ~100GB+ free
```

### 2. Fix Checkpoint Permissions (if needed)

```bash
sudo chown -R tim:tim /home/tim/source/activity/OpenVision-Instruct/stage_2_instruct_llava_ov_4b/
```

### 3. Create Symlinks

```bash
cd /home/tim/source/activity/OpenVision-Instruct

# If not already created
ln -s data/checkpoints/LLaVA-OneVision-1.5-4B-stage0 LLaVA-OneVision-1.5-4B-stage0
ln -s data/checkpoints/stage_2_instruct_llava_ov_4b stage_2_instruct_llava_ov_4b
ln -s data/OpenGPT-4o-Image/OpenGPT-4o-Image-wds data/OpenGPT-4o-Image-wds
```

### 4. Run Baseline Inference

```bash
# Quick test with 2 samples
python scripts/run_inference.py \
    --model LLaVA-OneVision-1.5-4B-stage0 \
    --num-samples 2 \
    --output evaluation/baseline_quick.json

# If successful, run full test
python scripts/run_inference.py \
    --model LLaVA-OneVision-1.5-4B-stage0 \
    --output evaluation/baseline_results.json
```

### 5. Load Fine-tuned Model

**Challenge:** Need to figure out how to load Megatron checkpoint

**Options:**
a. Convert Megatron checkpoint to HuggingFace format
b. Load base model + adapter weights separately
c. Use Megatron inference API

**For now:** Baseline with base model gives us comparison point

---

## 📊 Expected Baseline Results

The **base model** (pre-fine-tuning) should give:
- Generic responses like "I'll help you with that"
- Not specifically about image editing
- No mention of specific elements to edit

This establishes the baseline to compare against fine-tuned model.

---

## 🔍 Model Loading Strategy

### Current Approach
Using HuggingFace `transformers` library:
```python
from transformers import AutoProcessor, AutoModel

processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
    trust_remote_code=True
)
```

### Fine-tuned Model Challenge

The training saved:
- Format: Megatron checkpoint (`.pt` files)
- Location: `stage_2_instruct_llava_ov_4b/iter_0000500/`
- Type: Adapter weights only (not full model)

Need to either:
1. Convert to HuggingFace format
2. Load adapter on top of base model
3. Use Megatron inference pipeline

---

## 📁 File Organization After Cleanup

```
OpenVision-Instruct/
├── data/
│   ├── checkpoints/ (NFS mount - 11T available)
│   │   ├── LLaVA-OneVision-1.5-4B-stage0/     # Base model (17GB)
│   │   └── stage_2_instruct_llava_ov_4b/       # Fine-tuned (9GB)
│   ├── OpenGPT-4o-Image/ (NFS mount)
│   │   └── OpenGPT-4o-Image-wds/               # Dataset (82GB)
│   └── OpenGPT-4o-Image-wds -> ...             # Symlink
├── evaluation/
│   ├── test_samples/                            # Test images
│   ├── baseline_results.json                    # To be created
│   └── finetuned_results.json                   # To be created
├── scripts/
│   └── run_inference.py                         # ✅ Ready
├── LLaVA-OneVision-1.5-4B-stage0 -> ...        # Symlink
└── stage_2_instruct_llava_ov_4b -> ...         # Symlink
```

---

## ⏱️ Estimated Timeline

1. **File moves complete:** ~10-20 minutes (82GB over NFS)
2. **Baseline inference:** ~5-10 minutes (10 samples)
3. **Figure out fine-tuned loading:** TBD (research needed)
4. **Full evaluation:** ~15-30 minutes

**Total:** ~1-2 hours to complete validation

---

## 🎓 Key Learnings

1. **NFS mounts are key:** Keep large files on NFS
2. **Training creates files as root:** Need permission fixes
3. **Base model is HuggingFace format:** Easy to load
4. **Fine-tuned is Megatron format:** Requires conversion or special loading

---

**Status:** Waiting for file moves to complete, then ready to run inference! 🚀
