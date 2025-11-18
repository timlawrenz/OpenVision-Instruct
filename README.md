# Project OpenVision-Instruct

> **⚠️ PROJECT STATUS: DISCONTINUED (November 2025)**
> 
> **TL;DR:** This project successfully trained LLaVA-OneVision on image editing instructions, but discovered a fundamental architectural limitation: **LLaVA can only output text, not images**. The model learned to understand and acknowledge editing instructions perfectly, but cannot generate actual edited images.
> 
> **What Was Achieved:**
> - ✅ Successfully fine-tuned LLaVA-OneVision-1.5-4B (3,500 iterations)
> - ✅ Model correctly understands and acknowledges image editing instructions
> - ✅ Training pipeline and infrastructure fully functional
> - ✅ Data preparation and evaluation framework complete
> 
> **Why Discontinued:**
> - ❌ LLaVA architecture has no image decoder - fundamentally cannot generate images
> - ❌ Training data included output images, but model architecture cannot use them
> - ❌ Goal was to replicate Qwen2-VL image editing (Apache licensed), which requires image generation capability
> - ❌ Would need different architecture (diffusion-based or with image decoder) to achieve actual image editing
> 
> **What This Repo Provides:**
> - Complete working training pipeline for LLaVA-OneVision
> - Data preparation scripts for OpenGPT-4o-Image dataset
> - Evaluation and validation tools
> - Documentation of what works and what doesn't
> - Could be useful for: two-stage editing pipelines, instruction understanding, or as foundation for adding image generation
> 
> **For Future Work:**
> - Consider InstructPix2Pix, MGIE, or similar architectures that can actually generate images
> - Or use this as stage 1 (understanding) + diffusion model as stage 2 (generation)
> - See [REALITY_CHECK.md](./REALITY_CHECK.md) for detailed analysis and alternatives
> 
> ---

## Original Project Vision

Fine-tuning LLaVA-OneVision on the OpenGPT-4o-Image dataset to create a powerful, open-source model for advanced, instruction-based image understanding and editing.

The Vision: Democratizing Advanced Visual AI

In the current AI landscape, the most advanced capabilities for nuanced, instruction-based image manipulation often remain within the closed ecosystems of large tech companies. These "magic edit" features, while powerful, are typically offered as black-box APIs, limiting transparency, accessibility, and community-driven innovation.

This project aims to bridge that gap.

Our core mission is to take a highly efficient, open-source vision-language model and specialize it on a cutting-edge, publicly available dataset. The goal is to create a transparent, powerful, and accessible tool that provides state-of-the-art visual instruction-following capabilities, putting this power back into the hands of the open-source community, researchers, and individual creators.
The Components

This project stands on the shoulders of two incredible open-source efforts:

The Base Model: LLaVA-OneVision-1.5

*What it is*: A family of highly efficient, open-source Large Multimodal Models (LMMs).

*Why we chose it*: Its framework is explicitly designed for democratized, cost-effective training. It provides a state-of-the-art foundation in visual understanding with a parameter size that is feasible for fine-tuning on prosumer hardware.

The Fine-Tuning Dataset: OpenGPT-4o-Image

*What it is*: A comprehensive dataset designed for advanced image generation and editing. Its paper (arxiv:2509.24900) details a hierarchical taxonomy of complex visual tasks.

*Why we chose it*: It contains high-quality, diverse instruction-image pairs that are perfect for teaching a model to perform complex, multi-step, and nuanced visual reasoning and manipulation tasks.

## The Technical Approach

Our methodology is broken down into several key phases, from setting up the environment to fine-tuning and evaluating the model.

### Phase 1: Environment and Data Preparation

*   **Environment Setup**: We use the official Docker-based environment provided by the LLaVA-OneVision team to ensure reproducibility. For detailed instructions, see the [**Environment Setup Guide**](./docs/setup.md).

*   **Data Formatting**: A custom script parses the OpenGPT-4o-Image dataset into the specific JSONL format required by the training scripts. See the [Data Preparation Plan](./docs/data_preparation.md) for more details on the data structure.

### Phase 2: Fine-Tuning

*   **Core Technology**: The fine-tuning process uses a Parameter-Efficient Fine-Tuning (PEFT) approach, analogous to QLoRA. This freezes the base model and trains only a small number of "adapter" layers, making it possible to train on a single prosumer GPU (e.g., an NVIDIA RTX 4090). For a high-level overview, see the [**Fine-Tuning Guide**](./docs/fine_tuning_guide.md).

*   **Training Process**: The training is managed by a shell script that handles checkpointing, allowing you to stop and resume the process. For the specific commands to start, stop, and resume training, refer to the [**Training Commands Guide**](./docs/training_commands.md).

### Phase 3: Evaluation

*   **Performance Tracking**: To measure the model's improvement over time, we evaluate saved checkpoints against the `mme` (MM-Eval) benchmark. This requires converting the training checkpoints to a standard Hugging Face format before running the evaluation. The full process is detailed in the [**Evaluation Guide**](./docs/evaluation_guide.md).

### Phase 4: Merging & Release

*   **Model Release**: Upon successful fine-tuning, the trained adapter weights will be merged with the original base model weights to create a final, standalone model.

*   **Distribution**: The final model will be released on Hugging Face under a permissive license, complete with a detailed model card explaining its capabilities, limitations, and usage.

## Our Philosophy & The Value We Hope to Generate

*Democratizing Technology*: To provide a powerful, open-source alternative to proprietary visual AI systems.

*Enabling Innovation*: To create a foundational tool that developers, researchers, and artists can use to build new applications and explore creative frontiers without relying on expensive APIs.

*Promoting Transparency*: To offer an auditable model where the training data and process are public, fostering trust and further research.

*Efficient AI*: To demonstrate that meaningful contributions to the AI space don't always require massive-scale data centers. By leveraging efficient techniques like QLoRA, we can create valuable assets with a minimal carbon footprint relative to training a model from scratch.

## How to Contribute

This is an open project, and community involvement is welcome. For now, the best way to contribute is to:

*Open an Issue*: To suggest features, report bugs, or discuss the project direction.

*Submit a Pull Request*: To contribute directly to the data processing scripts, training configurations, or documentation.

## References

```bibtex
 @inproceedings{LLaVA-OneVision-1.5,
  title={LLaVA-OneVision-1.5: Fully Open Framework for Democratized Multimodal Training},
  author={An, Xiang and Xie, Yin and Yang, Kaicheng and Zhang, Wenkang and Zhao, Xiuwei and Cheng, Zheng and Wang, Yirui and Xu, Songcen and Chen, Changrui and Wu, Chunsheng and Tan, Huajie and Li, Chunyuan and Yang, Jing and Yu, Jie and Wang, Xiyao and Qin, Bin and Wang, Yumeng and Yan, Zizhen and Feng, Ziyong and Liu, Ziwei and Li, Bo and Deng, Jiankang},
  booktitle={arXiv},  
  year={2025}
 }

 @inproceedings{xie2025region,
  title={Region-based Cluster Discrimination for Visual Representation Learning},
  author={Xie, Yin and Yang, Kaicheng and An, Xiang and Wu, Kun and Zhao, Yongle and Deng, Weimo and Ran, Zimin and Wang, Yumeng and Feng, Ziyong and Miles, Roy and Elezi, Ismail and Deng, Jiankang},
  booktitle={ICCV},
  year={2025}
}

 @.venv/lib/python3.11/site-packages/sympy/physics/mechanics/__pycache__/particle.cpython-311.pyc{lillava,
  title={LLaVA-OneVision: Easy Visual Task Transfer},
  author={Li, Bo and Zhang, Yuanhan and Guo, Dong and Zhang, Renrui and Li, Feng and Zhang, Hao and Zhang, Kaichen and Zhang, Peiyuan and Li, Yanwei and Liu, Ziwei and Li, Chunyuan},
  journal={Transactions on Machine Learning Research}
  year={2024}
}
```
