# Environment Setup

This document details the steps required to set up the Python environment for the OpenVision-Instruct project. We prioritize using a virtual environment to ensure dependencies are isolated and reproducible.

## 1. Python Version

The deep learning ecosystem, particularly libraries with complex CUDA backends, can be sensitive to the Python version. To ensure maximum compatibility and access to pre-compiled binaries, this project is standardized on **Python 3.11**.

## 2. Virtual Environment

First, create a virtual environment in the project's root directory:

```bash
python3.11 -m venv .venv
```

This command creates a `./.venv/` directory containing a private copy of the Python interpreter and its libraries.

To activate the environment, use the following command:

```bash
source .venv/bin/activate
```

All subsequent commands should be run within this activated environment.

## 3. Core Dependencies

The fine-tuning process relies on the PyTorch and Hugging Face ecosystems. Install the core libraries with the following commands.

### 3.1. PyTorch

Install PyTorch with support for CUDA 12.1, which is required to leverage NVIDIA GPUs like the RTX 4090.

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 3.2. Hugging Face Ecosystem & QLoRA Tooling

Install the necessary libraries for loading the LLaVA model, applying the QLoRA technique, and managing the training process.

```bash
pip install packaging transformers peft bitsandbytes accelerate
```

- **packaging**: A core dependency for many Python projects.
- **transformers**: Provides the LLaVA model implementation and infrastructure.
- **peft**: The Parameter-Efficient Fine-Tuning library, which contains the LoRA/QLoRA implementation.
- **bitsandbytes**: Provides the 4-bit quantization functions that make QLoRA memory-efficient.
- **accelerate**: Simplifies running PyTorch training on any hardware setup.

## 4. Asset Acquisition

With the environment set up, the next step is to acquire the base model and the fine-tuning dataset.

### 4.1. Base Model Repository

Clone the official LLaVA-OneVision repository. This contains the necessary model architecture, training scripts, and utilities. We will place it in a `vendor/` directory to keep it separate from our project-specific code.

```bash
git clone https://github.com/EvolvingLMMs-Lab/LLaVA-OneVision-1.5.git vendor/LLaVA-OneVision
```

### 4.2. Fine-Tuning Dataset

Download the OpenGPT-4o-Image dataset from the Hugging Face Hub. The command below uses the `huggingface-hub` library to download the dataset files into a local `data/` directory.

```bash
.venv/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='WINDop/OpenGPT-4o-Image', repo_type='dataset', local_dir='data/OpenGPT-4o-Image')"
```
