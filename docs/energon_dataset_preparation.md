# WebDataset Preparation for Megatron Energon

## Critical Discovery: Energon Requires Prepared Datasets

**IMPORTANT**: Megatron Energon cannot directly load WebDataset tar files. They must be prepared using the `energon prepare` command, which creates the `.nv-meta` directory with required metadata.

## Problem We Encountered

When pointing the training config directly at a directory of WebDataset tar files, Energon raised:
```
FileNotFoundError: /workspace/OpenVision-Instruct/data/OpenGPT-4o-Image-wds
```

Even though the directory existed with 29 tar files (sft-0.tar through sft-28.tar).

## Solution: Use `energon prepare`

### Step 1: Install Megatron Energon (if not already installed)
```bash
pip install megatron-energon
```

### Step 2: Run the prepare command
```bash
energon prepare /path/to/your/webdataset/directory
```

### Step 3: Interactive Configuration

The tool will ask several questions:

**1. Train/Val/Test Split**
```
Please enter a desired train/val/test split like "0.5, 0.2, 0.3" or "8,1,1":
```
Enter: `0.9, 0.1, 0` (90% train, 10% validation, 0% test)

**2. Sample Type Selection**
```
Please enter a number to choose a class:
```
For conversational instruction-following data, choose: `9` (VQASample)

**3. Sample Loader Type**
```
Do you want to set a simple field_map[Y] (or write your own sample_loader [n])? [Y/n]:
```
Enter: `n` (we need custom loader for conversation format)

### Step 4: Create Custom Sample Loader

The tool creates `.nv-meta/sample_loader.py`. Edit it to match your data format:

```python
import torch
from typing import Dict, Any
from megatron.energon import VQASample

def sample_loader(sample: Dict[str, Any]) -> VQASample:
    """
    Load a sample from the WebDataset format and convert to VQASample.
    The JSON contains conversations with 'from' and 'value' fields.
    """
    # Get the conversation from JSON
    conversations = sample['json']['conversations']
    
    # Build context from the conversations
    context = ""
    answers = []
    
    for msg in conversations:
        if msg['from'] == 'human':
            # Extract the instruction (remove <image> token if present)
            context = msg['value'].replace('<image>', '').strip()
        elif msg['from'] == 'gpt':
            # The assistant's response
            answers.append(msg['value'])
    
    # Load the input image
    import io
    from PIL import Image
    import torchvision.transforms as transforms
    
    # Convert image bytes to tensor
    image_bytes = sample['input.jpg']
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    
    # Convert to tensor (C, H, W)
    transform = transforms.ToTensor()
    image_tensor = transform(image)
    
    return VQASample(
        image=image_tensor,
        context=context,
        answers=answers if answers else None,
        answer_weights=None
    )

def part_filter(part: str) -> bool:
    """Filter which tar files to include"""
    return True
```

### Step 5: Update Dataset Config

In your `dataset_config.yaml`, use **relative paths** from the config file's location:

```yaml
__module__: megatron.energon
__class__: Metadataset

splits:
  train:
    datasets:
      - weight: 1.0
        path: "OpenGPT-4o-Image-wds"  # Relative path
        subflavors:
          augmentation: false
  val:
    datasets:
      - weight: 1.0
        path: "OpenGPT-4o-Image-wds"
        subflavors:
          augmentation: false
```

## Verification

Test that the dataset loads correctly:

```python
from megatron.energon import load_dataset

dataset = load_dataset('/path/to/dataset_config.yaml')
print(f"SUCCESS! Dataset type: {type(dataset)}")
```

## Directory Structure After Preparation

```
OpenGPT-4o-Image-wds/
├── .nv-meta/               # Created by energon prepare
│   ├── dataset.yaml        # Metadata config
│   ├── sample_loader.py    # Custom sample loader
│   └── ...                 # Other metadata files
├── sft-0.tar               # Original WebDataset tar files
├── sft-0.tar.idx           # Created indices
├── sft-1.tar
├── sft-1.tar.idx
└── ...
```

## Key Lessons

1. **Energon is opinionated**: It requires datasets to be prepared with its tool
2. **WebDataset ≠ Energon-ready**: Raw WebDataset tars need the `.nv-meta` preparation
3. **Custom loaders are necessary**: For non-standard formats like conversational data
4. **Relative paths work better**: Use paths relative to the config file location
5. **The error is misleading**: `FileNotFoundError` doesn't mean the directory is missing, but rather that it's not recognized as a valid Energon dataset

## Official Documentation

- [Megatron Energon GitHub](https://github.com/NVIDIA/Megatron-Energon)
- [NVIDIA NeMo Energon Docs](https://docs.nvidia.com/nemo-framework/user-guide/latest/vlms/energondataprep.html)
- [Dataset Preparation Guide](https://deepwiki.com/NVIDIA/Megatron-Energon/7.1-dataset-preparation)

## Time Saved

Following this guide should save hours of debugging. The key insight: **always run `energon prepare` on WebDataset directories before using them for training**.
