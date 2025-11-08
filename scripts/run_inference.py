#!/usr/bin/env python3
"""
Simple inference script for LLaVA-OneVision model

This script loads the base HuggingFace model and runs inference on test samples.
Note: We're using the base model first to establish baseline, then we'll figure out
how to load the fine-tuned adapter weights.
"""

import argparse
import json
import sys
from pathlib import Path
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel

def load_model(model_path, device="cuda"):
    """Load model and processor"""
    print(f"Loading model from: {model_path}")
    print("This may take a few minutes...")
    
    try:
        processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True
        )
        
        model.eval()
        print("✅ Model loaded successfully")
        return model, processor
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        sys.exit(1)

def run_inference(model, processor, image_path, instruction, device="cuda"):
    """Run inference on a single image"""
    
    # Load image
    image = Image.open(image_path).convert("RGB")
    
    # Format as conversation with proper structure
    # LLaVA-OneVision expects messages with content as list
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": instruction}
            ]
        }
    ]
    
    # Apply chat template
    prompt = processor.apply_chat_template(
        messages,
        add_generation_prompt=True
    )
    
    # Prepare inputs
    inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt",
        padding=True
    )
    
    # Move to device
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
              for k, v in inputs.items()}
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=processor.tokenizer.pad_token_id
        )
    
    # Decode response (skip the prompt)
    response = processor.batch_decode(
        outputs[:, inputs['input_ids'].shape[1]:],
        skip_special_tokens=True
    )[0].strip()
    
    return response

def main():
    parser = argparse.ArgumentParser(description='Run inference on test samples')
    parser.add_argument(
        '--model',
        default='LLaVA-OneVision-1.5-4B-stage0',
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--test-samples',
        default='evaluation/test_samples/test_samples.json',
        help='Path to test samples JSON'
    )
    parser.add_argument(
        '--output',
        default='evaluation/inference_results.json',
        help='Path to save results'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=None,
        help='Number of samples to test (default: all)'
    )
    parser.add_argument(
        '--device',
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to run inference on'
    )
    args = parser.parse_args()
    
    # Check if CUDA is available
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA not available, falling back to CPU")
        args.device = 'cpu'
    
    # Load test samples
    test_samples_path = Path(args.test_samples)
    if not test_samples_path.exists():
        print(f"❌ Test samples not found: {test_samples_path}")
        sys.exit(1)
    
    with open(test_samples_path) as f:
        test_samples = json.load(f)
    
    if args.num_samples:
        test_samples = test_samples[:args.num_samples]
    
    print(f"\n{'='*70}")
    print(f"INFERENCE EVALUATION")
    print(f"{'='*70}")
    print(f"Model: {args.model}")
    print(f"Test samples: {len(test_samples)}")
    print(f"Device: {args.device}")
    print(f"{'='*70}\n")
    
    # Load model
    model, processor = load_model(args.model, args.device)
    
    # Run inference on each sample
    results = []
    test_dir = test_samples_path.parent
    
    for i, sample in enumerate(test_samples, 1):
        sample_id = sample['id']
        instruction = sample['instruction']
        image_path = test_dir / f"{sample_id}.input.jpg"
        
        if not image_path.exists():
            print(f"⚠️  Image not found: {image_path}, skipping...")
            continue
        
        print(f"\n[{i}/{len(test_samples)}] {sample_id}")
        print(f"Instruction: {instruction}")
        print(f"Running inference...", end=" ", flush=True)
        
        try:
            response = run_inference(
                model,
                processor,
                image_path,
                instruction,
                args.device
            )
            
            print("✅")
            print(f"Response: {response[:200]}..." if len(response) > 200 else f"Response: {response}")
            
            results.append({
                'id': sample_id,
                'instruction': instruction,
                'response': response,
                'image_path': str(image_path)
            })
            
        except Exception as e:
            print(f"❌ Error: {e}")
            results.append({
                'id': sample_id,
                'instruction': instruction,
                'error': str(e),
                'image_path': str(image_path)
            })
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"✅ Results saved to: {output_path}")
    print(f"{'='*70}\n")
    
    # Print summary
    successful = sum(1 for r in results if 'error' not in r)
    print(f"Summary:")
    print(f"  Total samples: {len(test_samples)}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {len(results) - successful}")

if __name__ == '__main__':
    main()
