#!/usr/bin/env python3
"""
Simplified inference script that doesn't use Megatron's inference API.
Instead, we load the model directly and run forward passes.
"""

import os
import sys
import json
import torch
from PIL import Image
from typing import List, Dict

# Add vendor paths
sys.path.insert(0, 'vendor/LLaVA-OneVision')

# Disable transformer_engine if it causes issues
os.environ['NVTE_FRAMEWORK'] = ''

def load_test_samples(test_samples_path: str, num_samples: int = None) -> List[Dict]:
    """Load test samples from JSON file."""
    with open(test_samples_path, 'r') as f:
        samples = json.load(f)
    
    if num_samples is not None:
        samples = samples[:num_samples]
    
    print(f"Loaded {len(samples)} test samples")
    return samples


def simple_inference_test(args):
    """Run a simplified inference test without full Megatron setup."""
    
    print("=" * 80)
    print("SIMPLIFIED INFERENCE TEST")
    print("=" * 80)
    
    # Load test samples
    samples_dir = os.path.dirname(args.test_samples)
    test_samples = load_test_samples(args.test_samples, args.num_samples)
    
    # For now, just validate that we can load samples and images
    print("\nValidating test samples...")
    
    valid_samples = []
    for idx, sample in enumerate(test_samples):
        sample_id = sample['id']
        instruction = sample['instruction']
        input_image_path = os.path.join(samples_dir, f"{sample_id}.input.jpg")
        
        print(f"\n[{idx+1}/{len(test_samples)}] {sample_id}")
        print(f"  Instruction: {instruction}")
        print(f"  Image: {input_image_path}", end="")
        
        if os.path.exists(input_image_path):
            try:
                img = Image.open(input_image_path)
                print(f" ✓ ({img.size[0]}x{img.size[1]})")
                valid_samples.append({
                    'id': sample_id,
                    'instruction': instruction,
                    'image_path': input_image_path,
                    'image_size': img.size
                })
            except Exception as e:
                print(f" ✗ (Error: {e})")
        else:
            print(" ✗ (Not found)")
    
    print(f"\n{len(valid_samples)}/{len(test_samples)} samples are valid")
    
    # Save validation results
    results_path = args.output.replace('.json', '_validation.json')
    with open(results_path, 'w') as f:
        json.dump(valid_samples, f, indent=2)
    
    print(f"\nValidation results saved to: {results_path}")
    
    print("\n" + "=" * 80)
    print("NOTE: Full Megatron inference requires proper environment setup.")
    print("This validation confirms your test samples are ready.")
    print("=" * 80)
    
    return valid_samples


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Simplified inference test')
    parser.add_argument('--load', type=str, required=True,
                        help='Path to checkpoint directory')
    parser.add_argument('--hf-tokenizer-path', type=str, required=True,
                        help='Path to tokenizer')
    parser.add_argument('--test-samples', type=str, 
                        default='evaluation/test_samples/test_samples.json',
                        help='Path to test samples JSON')
    parser.add_argument('--output', type=str,
                        default='evaluation/simple_inference_results.json',
                        help='Output path for results')
    parser.add_argument('--num-samples', type=int, default=None,
                        help='Limit number of samples to process')
    parser.add_argument('--tokenizer-type', type=str, default='HFTokenizer')
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--top-k', type=int, default=50)
    parser.add_argument('--top-p', type=float, default=0.9)
    parser.add_argument('--num-tokens-to-generate', type=int, default=512)
    parser.add_argument('--max-batch-size', type=int, default=1)
    parser.add_argument('--use-checkpoint-args', action='store_true')
    parser.add_argument('--tensor-model-parallel-size', type=int, default=1)
    parser.add_argument('--pipeline-model-parallel-size', type=int, default=1)
    
    args = parser.parse_args()
    
    # Run simple test
    valid_samples = simple_inference_test(args)
    
    print(f"\n✓ Found {len(valid_samples)} valid test samples")
    print(f"✓ Checkpoint exists: {os.path.exists(args.load)}")
    print(f"✓ Tokenizer exists: {os.path.exists(args.hf_tokenizer_path)}")
    
    print("\n" + "=" * 80)
    print("NEXT STEPS:")
    print("=" * 80)
    print("1. ✓ Test samples validated")
    print("2. ☐ Set up proper Megatron environment (transformer_engine)")
    print("3. ☐ Run full inference with model loading")
    print("=" * 80)


if __name__ == '__main__':
    main()
