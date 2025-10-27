# Fine-Tuning Guide

This document details the process for fine-tuning the LLaVA-OneVision model using the QLoRA methodology. It covers the discovery of the correct training script, the configuration of the training process, and the final script used to launch the fine-tuning.

## 1. Discovering the Training Script

Initial exploration of the `vendor/LLaVA-OneVision` repository did not reveal a straightforward `finetune.py` or `train.py` script in the root directory. The `README.md` and the file structure pointed towards a more complex, multi-stage training process managed by shell scripts.

The key discovery was the `examples/llava_ov_1_5/quick_start/stage_2_instruct_llava_ov_4b.sh` script. Analysis of this script revealed that it is a wrapper that launches the main Python training program: `aiak_training_llm/train.py`. This became our target for customization.

## 2. Understanding the Training Configuration

By examining the launch script and the argument parsing logic in `aiak_training_llm/train/arguments.py`, we identified the key parameters to control the fine-tuning process.

### QLoRA-Style Fine-Tuning

The most critical finding was that the framework handles Parameter-Efficient Fine-Tuning (PEFT) through the `--trainable-modules` argument. While there is no explicit `--qlora` flag, setting this parameter to `adapter` instructs the training script to freeze the base model and only train the lightweight adapter layers. This is the core mechanism that makes fine-tuning on consumer hardware feasible and is equivalent to the QLoRA approach outlined in the project's `README.md`.

### Data Configuration

The training script uses a flexible data loading system that is configured via a JSON file. The default configuration is located at `vendor/LLaVA-OneVision/configs/sft_dataset_config.json`. Our analysis showed that our prepared dataset needed to conform to the `multimodal` format defined in this file to be correctly interpreted by the training script.

## 3. The Fine-Tuning Script

Based on these findings, we will create a custom launch script, `scripts/run_finetune.sh`, to orchestrate the fine-tuning process. This script will:

1.  **Set Environment Variables**: Define paths to the training scripts, base model, tokenizer, and our custom dataset.
2.  **Configure Training Parameters**:
    -   Load the pre-trained `LLaVA-OneVision-1.5-4B-stage0` model.
    -   Point to our `data/finetune_data_multimodal.json` file.
    -   Set `--training-phase` to `sft` (Supervised Fine-Tuning).
    -   Set `--trainable-modules` to `adapter` to enable our QLoRA-style fine-tuning.
    -   Use the built-in `multimodal` dataset configuration.
    -   Configure hyperparameters such as learning rate, batch size, and sequence length for efficient training on a 24GB GPU.
3.  **Launch the Training**: Execute the `aiak_training_llm/train.py` script with the specified configuration.

This approach allows us to leverage the power of the LLaVA-OneVision training framework while adapting it to our specific dataset and hardware constraints, staying true to the project's goal of efficient, democratized AI.
