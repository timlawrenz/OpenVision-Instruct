# Project Context

## Purpose
OpenVision-Instruct aims to democratize advanced visual AI by fine-tuning the LLaVA-OneVision-1.5 model on the OpenGPT-4o-Image dataset. The project creates a powerful, open-source model for instruction-based image understanding and editing, providing a transparent alternative to proprietary "magic edit" features. Core goals include:
- Making state-of-the-art visual instruction-following capabilities accessible to the open-source community
- Demonstrating efficient AI training using Parameter-Efficient Fine-Tuning (PEFT) techniques on consumer hardware
- Providing a fully transparent, auditable model where training data and processes are public
- Enabling researchers, developers, and creators to build applications without expensive proprietary APIs

## Tech Stack
- **Base Model**: LLaVA-OneVision-1.5-4B (efficient open-source Large Multimodal Model)
- **Training Framework**: Megatron-LM / NeMo framework via Docker container
- **Fine-tuning Method**: Parameter-Efficient Fine-Tuning (PEFT) using adapter layers (analogous to QLoRA)
- **Python**: 3.11+ with PyTorch and CUDA support
- **Key Libraries**: 
  - transformers==4.53.1
  - accelerate==1.9.0
  - datasets==2.19.2
  - megatron-energon==5.0.0
  - wandb==0.21.0 (for experiment tracking)
- **Data Format**: YAML-based conversation format for training data
- **Infrastructure**: Docker (llava_megatron:25.04), NVIDIA Container Toolkit
- **Hardware**: NVIDIA GPU with CUDA (e.g., RTX 4090 24GB or A100 80GB)
- **Dataset**: OpenGPT-4o-Image from Hugging Face (WINDop/OpenGPT-4o-Image)

## Project Conventions

### Code Style
- Python scripts follow standard PEP 8 conventions
- Use explicit variable names that describe the data or purpose
- Scripts should be self-contained with clear main entry points
- Prefer environment variables for configuration over hardcoded paths
- Keep vendor code (LLaVA-OneVision) separate in `vendor/` directory

### Architecture Patterns
- **Vendor Isolation**: Official LLaVA-OneVision code lives in `vendor/LLaVA-OneVision` and should not be modified
- **Data Pipeline**: Raw dataset → preprocessing script → prepared YAML → training
- **Model Lifecycle**: HF model → Megatron conversion → PEFT training → checkpoint saving → evaluation → merge & release
- **Docker-first**: All training and evaluation runs inside the official Docker container for reproducibility
- **Checkpointing**: Regular checkpoint saves with ability to resume from any checkpoint
- **Three-phase approach**:
  1. Environment & Data Preparation
  2. Fine-tuning with PEFT
  3. Evaluation against MME benchmark

### Testing Strategy
- Model performance tracked via MME (MM-Eval) benchmark evaluation
- Checkpoints converted to HuggingFace format before evaluation
- Training progress monitored via WandB integration
- No unit tests currently; validation happens through model evaluation

### Git Workflow
- Single main branch (`main`)
- Descriptive commit messages focusing on documentation and data preparation steps
- Remote: origin/main on GitHub
- Keep data files and large checkpoints out of git (use .gitignore)

## Domain Context
- **Vision-Language Models (VLMs)**: Models that process both images and text, enabling visual understanding and reasoning
- **Instruction Tuning**: Fine-tuning models to follow human instructions for specific tasks (Stage 2 training)
- **PEFT/QLoRA**: Training only adapter layers while freezing base model weights to reduce memory requirements
- **Megatron Format**: NVIDIA's checkpoint format for large-scale model training with tensor/pipeline parallelism
- **ShareGPT Format**: Conversation-style data format with alternating user/assistant messages
- **MME Benchmark**: Standard evaluation suite for multimodal models measuring perception and cognition
- **Image Editing Tasks**: The OpenGPT-4o-Image dataset contains hierarchical taxonomy of visual editing and generation tasks

## Important Constraints
- **Memory**: Training must fit on consumer GPU (24GB VRAM) via PEFT approach
- **Adapter-only Training**: Only train adapter modules, not full model (`--trainable-modules adapter`)
- **Vendor Code**: Must not modify official LLaVA-OneVision scripts; configure via environment variables
- **Data Format**: Must match the multimodal format expected by the training framework (images + messages structure)
- **Reproducibility**: All training happens in official Docker environment to ensure consistent results
- **Path Management**: Image paths in dataset must be relative to training script working directory
- **License Compliance**: Final model must be released under permissive license compatible with base model

## External Dependencies
- **Hugging Face Hub**: 
  - Base model: EvolvingLMMs-Lab/LLaVA-OneVision-1.5-4B-stage0
  - Dataset: WINDop/OpenGPT-4o-Image
  - Model components: DeepGlint-AI/rice-vit-large-patch14-560, Qwen/Qwen3-4B-Instruct-2507
- **GitHub**: LLaVA-OneVision-1.5 official repository (EvolvingLMMs-Lab/LLaVA-OneVision-1.5)
- **Docker Hub**: NVIDIA CUDA base images for container
- **WandB**: Optional experiment tracking and logging
- **NVIDIA Container Toolkit**: Required for GPU access in Docker
