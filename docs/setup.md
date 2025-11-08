# Environment Setup

This document details the steps required to set up the environment for the OpenVision-Instruct project. We will follow the recommended Docker-based setup from the official LLaVA-OneVision-1.5 repository to ensure a reproducible and stable environment.

## 1. Prerequisites

-   **Docker**: Ensure you have Docker installed and running on your system.
-   **NVIDIA GPU**: A CUDA-enabled NVIDIA GPU is required for training. The instructions are tailored for an A100 80GB GPU, but other GPUs with sufficient VRAM (e.g., RTX 4090 24GB) should work.
-   **NVIDIA Container Toolkit**: You must have the NVIDIA Container Toolkit installed to enable GPU access within Docker containers.

## 2. Clone the LLaVA-OneVision-1.5 Repository

First, clone the official LLaVA-OneVision-1.5 repository. We will place it in a `vendor/` directory to keep it separate from our project-specific code.

```bash
git clone https://github.com/EvolvingLMMs-Lab/LLaVA-OneVision-1.5.git vendor/LLaVA-OneVision
```

## 3. Build and Run the Docker Container

Navigate into the cloned repository and build the Docker image.

```bash
cd vendor/LLaVA-OneVision
docker build -t llava_megatron:25.04 .
```

After the build is complete, run the container. This command mounts the current project directory (`OpenVision-Instruct`) into the container at `/workspace/OpenVision-Instruct`.

```bash
# Make sure you are in the root of the OpenVision-Instruct project directory
docker run -it --gpus all \
--ipc host --net host --privileged --cap-add IPC_LOCK \
--ulimit memlock=-1 --ulimit stack=67108864 --rm \
-v $(pwd):/workspace/OpenVision-Instruct \
-w /workspace/OpenVision-Instruct \
--name "llava_megatron_container" \
llava_megatron:25.04 /bin/bash
```

All subsequent commands should be run from within this Docker container.

## 4. Asset Acquisition

### 4.1. Base Model

Inside the container, you need to acquire the base model. You have two options:

**Option 1: Download pre-trained model from Hugging Face**

Download the `LLaVA-OneVision-1.5-4B-stage0` model directly from Hugging Face.

```bash
# You may need to install huggingface-cli first: pip install huggingface-cli
# Then login: huggingface-cli login
huggingface-cli download EvolvingLMMs-Lab/LLaVA-OneVision-1.5-4B-stage0 --local-dir LLaVA-OneVision-1.5-4B-stage0
```

**Option 2: Merge initial weights yourself**

Alternatively, you can merge the initial weights from the original ViT and LLM:

```bash
python vendor/LLaVA-OneVision/ds/merge_model.py \
--vit_path DeepGlint-AI/rice-vit-large-patch14-560 \
--llm_path Qwen/Qwen3-4B-Instruct-2507 \
--output LLaVA-OneVision-1.5-4B-stage0
```

### 4.2. Fine-Tuning Dataset

Download the OpenGPT-4o-Image dataset from the Hugging Face Hub into the `data/` directory.

```bash
# Make sure you are in /workspace/OpenVision-Instruct inside the container
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='WINDop/OpenGPT-4o-Image', repo_type='dataset', local_dir='data/OpenGPT-4o-Image')"
```

## 5. Model Format Conversion



The LLaVA-OneVision training scripts require the model to be in the Megatron format. Before running the conversion, create a directory on your NFS share to store the large checkpoint files.



```bash

# Create a directory for the converted checkpoints on your mounted data volume

mkdir -p data/checkpoints

```



Now, convert the downloaded Hugging Face model to the Megatron format, ensuring the output is saved to the directory you just created.



```bash

AIAK_TRAINING_PATH=/workspace/OpenVision-Instruct/vendor/LLaVA-OneVision \

bash vendor/LLaVA-OneVision/examples/llava_ov_1_5/convert/convert_4b_hf_to_mcore.sh \

/workspace/OpenVision-Instruct/LLaVA-OneVision-1.5-4B-stage0 \

/workspace/OpenVision-Instruct/data/checkpoints/LLaVA-OneVision-1.5-4B-stage0_mcore_tp1_pp1 \

1 1

```



After these steps, the environment is set up and you are ready to proceed with data preparation and fine-tuning.
