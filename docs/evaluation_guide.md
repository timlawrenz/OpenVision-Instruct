# Evaluation Guide

This document explains how to evaluate the fine-tuned model to track its performance over time.

The evaluation process requires two main steps:
1.  Converting a training checkpoint from its native `Megatron` format to the standard `Hugging Face` format.
2.  Running the `lmms-eval` tool on the converted Hugging Face model.

All commands should be run from within the Docker container at the `/workspace/OpenVision-Instruct` directory.

## Step 1: Convert a Training Checkpoint

The training script saves checkpoints in a highly efficient format optimized for resuming training. To evaluate a checkpoint, you must first convert it into the standard Hugging Face format that evaluation tools expect.

**Command Template:**

```bash
# 1. Set the path to the training checkpoint you want to evaluate
INPUT_CHECKPOINT_PATH=/workspace/OpenVision-Instruct/data/checkpoints/OpenVision-Instruct-4B-adapter/iter_...

# 2. Set a path for the new Hugging Face model directory
OUTPUT_HF_MODEL_PATH=/workspace/OpenVision-Instruct/data/checkpoints/OpenVision-Instruct-4B-adapter-hf-iter...

# 3. Run the conversion script
AIAK_TRAINING_PATH=/workspace/OpenVision-Instruct/vendor/LLaVA-OneVision \
bash vendor/LLaVA-OneVision/examples/llava_ov_1_5/convert/convert_4b_mcore_to_hf.sh \
$INPUT_CHECKPOINT_PATH \
$OUTPUT_HF_MODEL_PATH \
1 1

# 4. Copy the tokenizer and other necessary configuration files to the new directory
find /workspace/OpenVision-Instruct/vendor/LLaVA-OneVision-1.5-4B-stage0/ -type f -not -iname '*safetensors*' -exec cp {} $OUTPUT_HF_MODEL_PATH/ ";"
```

Replace the `iter_...` parts with the actual iteration number of the checkpoint you wish to evaluate (e.g., `iter_0000500`).

## Step 2: Run the Evaluation

After the conversion is complete, you can run the evaluation.

**1. Install the Evaluation Tool:**

If you haven't already, install the `lmms-eval` library:
```bash
pip install git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git
```

**2. Run the Evaluation Script:**

This command will run the `mme` (MM-Eval) benchmark on your converted model.

```bash
# Use the same output path from the conversion step above
HF_MODEL_PATH=/workspace/OpenVision-Instruct/data/checkpoints/OpenVision-Instruct-4B-adapter-hf-iter...

accelerate launch --num_processes=1 -m lmms_eval \
--model=llava_onevision1_5 \
--model_args=pretrained=$HF_MODEL_PATH \
--tasks=mme \
--batch_size=1
```

## Tracking Progress

You can repeat this process for different checkpoints saved during your training run (e.g., at 500, 1000, and 1500 iterations). By comparing the evaluation scores from the `mme` benchmark at different stages, you can effectively track whether your model's performance is improving over time.

```