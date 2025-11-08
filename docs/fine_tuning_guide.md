# Fine-Tuning Guide

This document provides a high-level overview of the fine-tuning process for the OpenVision-Instruct project. For specific commands and step-by-step instructions, please refer to the [Training Commands](./training_commands.md) and [Evaluation Guide](./evaluation_guide.md).

## 1. The Training Environment

We use the official Docker container provided by the `LLaVA-OneVision` repository. This approach ensures a consistent, reproducible environment with all necessary dependencies and CUDA libraries pre-configured. All training and utility scripts are run from within this container.

## 2. The Core Training Script

The primary script for our fine-tuning task is `vendor/LLaVA-OneVision/examples/llava_ov_1_5/quick_start/stage_2_instruct_llava_ov_4b.sh`.

This is a high-level wrapper script that orchestrates the "Stage 2" instruction fine-tuning process. It is responsible for launching the underlying Python training code (`aiak_training_llm/train.py`) with the correct set of default hyperparameters for this phase.

## 3. Configuration via Environment Variables

Instead of creating a custom script or passing a long list of command-line arguments, we configure the training process by setting environment variables before calling the script. This is the standard method used by the `LLaVA-OneVision` framework.

The key variables we use are:
-   `AIAK_TRAINING_PATH`: The root of the `LLaVA-OneVision` vendor code.
-   `DATA_PATH`: The path to our prepared dataset (`prepared_data.jsonl`).
-   `TOKENIZER_PATH`: The path to the base model's tokenizer files.
-   `CHECKPOINT_PATH`: The path to the model checkpoint to start from. For a new run, this is the converted base model. For resuming, this is the path to a saved training checkpoint.
-   `SAVE_CKPT_PATH`: The directory where new training checkpoints will be saved.

## 4. Parameter-Efficient Fine-Tuning (PEFT)

The `stage_2_instruct_llava_ov_4b.sh` script, by default, attempts to perform full-model fine-tuning, which requires a very large amount of GPU memory. To make training feasible on a single consumer GPU, we have modified the script to perform Parameter-Efficient Fine-Tuning (PEFT).

This was achieved by changing the `--trainable-modules` argument within the script from `language_model adapter vision_model` to simply `adapter`. This crucial change instructs the framework to freeze the vast majority of the base model's weights and only train the small, lightweight "adapter" layers. This is the key technique that makes fine-tuning possible on a 24GB GPU and is analogous to the QLoRA methodology.

## 5. Checkpointing and Resuming

The training framework has built-in support for saving and resuming progress. By setting the `--save-interval` argument, we instruct the script to save a complete checkpoint periodically.

To resume, we simply update the `CHECKPOINT_PATH` to point to the desired saved checkpoint. The script handles the rest, loading the model weights, optimizer state, and learning rate to continue training seamlessly.