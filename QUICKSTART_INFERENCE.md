# Quick Start: Docker Inference

## ✅ Ready to Run!

Everything is set up. Just run this one command:

```bash
sudo bash scripts/run_docker_inference.sh
```

That's it! The script will:
1. ✅ Use the Docker container (nvcr.io/nvidia/pytorch:24.02-py3)
2. ✅ Install dependencies (transformers, pillow, etc.)
3. ✅ Run inference on 2 test samples
4. ✅ Save results to `evaluation/finetuned_results.json`

**Expected time:** 5-10 minutes (first run downloads packages)

---

## What Happens

```
1. Container starts with GPU access
2. Installs Python packages (~2 min)
3. Loads your fine-tuned checkpoint (9.2GB)
4. Runs inference on 2 samples (~3-5 min)
5. Saves results
```

---

## View Results

After it completes:

```bash
# View all results
cat evaluation/finetuned_results.json | jq '.'

# View just the responses
cat evaluation/finetuned_results.json | jq '.[].generated_text'

# Compare to baseline (gibberish)
cat evaluation/baseline_hf_results.json | jq '.[0].response'
```

---

## Expected Output

### Fine-tuned Model (What you should get):
```
Instruction: "Remove the word 'Sfice' under the Location column."
Response: <Coherent image editing instructions>
```

### Base Model (What we saw before):
```
Response: ']>;\n tritur.arr>Add_server进城맵...' (gibberish)
```

---

## Troubleshooting

### If you get "docker: command not found"
```bash
# Docker is installed, just needs sudo
which docker  # Should show /usr/bin/docker
```

### If you get permission errors
```bash
# Make sure script is executable
chmod +x scripts/run_docker_inference.sh

# Run with sudo
sudo bash scripts/run_docker_inference.sh
```

### If Docker image isn't found
```bash
# Pull it first (5-10 minutes)
sudo docker pull nvcr.io/nvidia/pytorch:24.02-py3
```

---

## Run on All 10 Samples

After the 2-sample test works:

```bash
# Edit the script and change --num-samples 2 to --num-samples 10
# Or run directly:

sudo docker run --gpus all --rm \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v $(pwd):/workspace -w /workspace \
  nvcr.io/nvidia/pytorch:24.02-py3 \
  bash -c "
    pip install -q transformers pillow einops einops-exts sentencepiece webdataset
    export PYTHONPATH=/workspace/vendor/LLaVA-OneVision:/workspace/vendor/LLaVA-OneVision/aiak_megatron:\$PYTHONPATH
    python scripts/run_megatron_inference.py \
      --load stage_2_instruct_llava_ov_4b/iter_0000500 \
      --hf-tokenizer-path LLaVA-OneVision-1.5-4B-stage0 \
      --test-samples evaluation/test_samples/test_samples.json \
      --output evaluation/finetuned_results_all.json \
      --num-samples 10 \
      --use-checkpoint-args
  "
```

---

## Files

- **Script:** `scripts/run_docker_inference.sh`
- **Inference code:** `scripts/run_megatron_inference.py`
- **Checkpoint:** `stage_2_instruct_llava_ov_4b/iter_0000500/` (9.2GB)
- **Test samples:** `evaluation/test_samples/test_samples.json`
- **Output:** `evaluation/finetuned_results.json`

---

## Next Steps

1. **Run the script** (see command above)
2. **Check results** - compare to baseline
3. **Evaluate quality** - does it generate good instructions?
4. **Run on all samples** if quality looks good
5. **Celebrate!** 🎉

---

**Ready to see your fine-tuned model in action!**
