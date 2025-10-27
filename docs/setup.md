# Environment Setup

This document details the steps required to set up the Python environment for the OpenVision-Instruct project. We prioritize using a virtual environment to ensure dependencies are isolated and reproducible.

## 1. Python Version

The deep learning ecosystem, particularly libraries with complex CUDA backends, can be sensitive to the Python version. To ensure maximum compatibility and access to pre-compiled binaries, this project is standardized on **Python 3.11**.

## 2. System-level Prerequisites

Before creating the virtual environment, ensure you have the necessary system-level packages installed. These are required to compile some of the Python dependencies from source.

### 2.1. Python Development Headers

The build process for several libraries requires the Python C development headers. Install them using your system's package manager.

For Debian/Ubuntu-based systems:
```bash
sudo apt-get update && sudo apt-get install python3.11-dev
```

### 2.2. NVIDIA CUDA Toolkit

While many CUDA libraries are installed via `pip`, the build process for `transformer_engine` requires the CUDA Toolkit to be available in the system's path to find necessary headers like `cudnn.h`. A system-wide installation is recommended. Please ensure you have the NVIDIA CUDA Toolkit installed from the official NVIDIA website.

## 3. Virtual Environment

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

## 4. Core Dependencies

To ensure a reproducible environment and prevent dependency conflicts, all required Python packages are listed in the `requirements.txt` file.

Install all core dependencies, including PyTorch, the Hugging Face ecosystem, and QLoRA tooling, by running the following command from the root of the project:

```bash
pip install -r requirements.txt
```

This single command installs the correct, pinned versions of all necessary libraries, ensuring the environment is consistent and stable.

## 5. Asset Acquisition

With the environment set up, the next step is to acquire the base model and the fine-tuning dataset.

### 5.1. Base Model Repository

Clone the official LLaVA-OneVision repository. This contains the necessary model architecture, training scripts, and utilities. We will place it in a `vendor/` directory to keep it separate from our project-specific code.

```bash
git clone https://github.com/EvolvingLMMs-Lab/LLaVA-OneVision-1.5.git vendor/LLaVA-OneVision
```

### 5.2. Fine-Tuning Dataset

Download the OpenGPT-4o-Image dataset from the Hugging Face Hub. The command below uses the `huggingface-hub` library to download the dataset files into a local `data/` directory.

```bash
.venv/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='WINDop/OpenGPT-4o-Image', repo_type='dataset', local_dir='data/OpenGPT-4o-Image')"
```
