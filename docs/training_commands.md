# Training and Data Preparation Commands

This document provides a quick reference for the commands used to prepare the data and to start, stop, and resume the fine-tuning process.

All commands should be run from within the Docker container, at the `/workspace/OpenVision-Instruct` directory.

## 1. Data Preparation

This script transforms the raw `editing.json` into a single `prepared_data.yaml` file that contains a list of all data points. This is the format expected by the training script's data loader.

```bash
# First, ensure the pyyaml package is installed
pip install pyyaml

# Run the preparation script
python scripts/prepare_data.py
```

## 2. Starting a New Training Run

This command starts the fine-tuning process. The `DATA_PATH` variable points to a configuration file that, in turn, points to the actual data.

```bash
AIAK_TRAINING_PATH=/workspace/OpenVision-Instruct/vendor/LLaVA-OneVision \
DATA_PATH=/workspace/OpenVision-Instruct/data/dataset_config.yaml \
TOKENIZER_PATH=/workspace/OpenVision-Instruct/vendor/LLaVA-OneVision-1.5-4B-stage0 \
CHECKPOINT_PATH=/workspace/OpenVision-Instruct/data/checkpoints/LLaVA-OneVision-1.5-4B-stage0_mcore_tp1_pp1 \
SAVE_CKPT_PATH=/workspace/OpenVision-Instruct/data/checkpoints/OpenVision-Instruct-4B-adapter \
bash vendor/LLaVA-OneVision/examples/llava_ov_1_5/quick_start/stage_2_instruct_llava_ov_.sh
```

## 3. Interrupting Training

To stop the training process at any time, press `Ctrl+C`. The script will catch the signal, save a final checkpoint at the current training step, and exit gracefully.

## 4. Resuming from a Checkpoint

To continue training from where you left off, you first need to identify the latest checkpoint.

**Find the latest checkpoint:**
```bash
# This command lists the checkpoint directories and shows the most recent one
ls -td data/checkpoints/OpenVision-Instruct-4B-adapter/iter_* | head -n 1
```

Copy the output path from the command above. Now, use that path for the `CHECKPOINT_PATH` variable in the training command.

**Resume training command:**
```bash
# Replace this path with the output from the command above
LATEST_CHECKPOINT=data/checkpoints/OpenVision-Instruct-4B-adapter/iter_...

AIAK_TRAINING_PATH=/workspace/OpenVision-Instruct/vendor/LLaVA-OneVision \
DATA_PATH=/workspace/OpenVision-Instruct/data/dataset_config.yaml \
TOKENIZER_PATH=/workspace/OpenVision-Instruct/vendor/LLaVA-OneVision-1.5-4B-stage0 \
CHECKPOINT_PATH=$LATEST_CHECKPOINT \
SAVE_CKPT_PATH=/workspace/OpenVision-Instruct/data/checkpoints/OpenVision-Instruct-4B-adapter \
bash vendor/LLaVA-OneVision/examples/llava_ov_1_5/quick_start/stage_2_instruct_llava_ov_4b.sh
```
