#!/usr/bin/env python3
"""
Megatron-based inference for LLaVA-OneVision fine-tuned model
Adapted from: aiak_megatron/examples/inference/gpt/simple_gpt_batch_inference.py
"""

import os
import sys
import json
import torch
from argparse import Namespace
from typing import List, Dict
from PIL import Image

# Add LLaVA-OneVision to path
sys.path.insert(0, 'vendor/LLaVA-OneVision')
sys.path.insert(0, 'vendor/LLaVA-OneVision/aiak_megatron')

from megatron.training import get_args
from megatron.training import get_tokenizer
from megatron.training.checkpointing import load_checkpoint
from megatron.training.initialize import initialize_megatron
from megatron.training import get_model
from megatron.core.inference.engines.mcore_engine import MCoreEngine
from megatron.core.inference.common_inference_params import CommonInferenceParams
from megatron.core.inference.model_inference_wrappers.inference_wrapper_config import InferenceWrapperConfig
from megatron.core.inference.model_inference_wrappers.multimodal.vlm_inference_wrapper import VLMInferenceWrapper
from megatron.core.inference.text_generation_controllers.simple_text_generation_controller import SimpleTextGenerationController
from megatron.core.inference.inference_request import InferenceRequest
from megatron.core.transformer.module import MegatronModule

# Import model builder from training code
from aiak_training_llm.train import parse_train_args
from aiak_training_llm.models import build_model


def add_inference_args(parser):
    """Add inference-specific arguments."""
    group = parser.add_argument_group(title='inference')
    
    group.add_argument("--temperature", type=float, default=0.7,
                       help='Sampling temperature.')
    group.add_argument("--top-k", type=int, default=50,
                       help='Top k sampling.')
    group.add_argument("--top-p", type=float, default=0.9,
                       help='Top p sampling.')
    group.add_argument("--num-tokens-to-generate", type=int, default=512,
                       help='Number of tokens to generate for each prompt')
    group.add_argument("--max-batch-size", type=int, default=1,
                       help='Max number of prompts to process at once')
    group.add_argument("--test-samples", type=str, default="evaluation/test_samples/test_samples.json",
                       help='Path to test samples JSON file')
    group.add_argument("--output", type=str, default="evaluation/inference_results.json",
                       help='Path to save inference results')
    group.add_argument("--num-samples", type=int, default=None,
                       help='Limit number of test samples (for quick testing)')
    
    return parser


def model_provider(pre_process=True, post_process=True):
    """Build the VLM model.
    
    This uses the same model building logic as training.
    """
    args = get_args()
    model = build_model(args, pre_process, post_process)
    return model


def load_test_samples(test_samples_path: str, num_samples: int = None) -> List[Dict]:
    """Load test samples from JSON file.
    
    Args:
        test_samples_path: Path to test_samples.json
        num_samples: Optional limit on number of samples to load
        
    Returns:
        List of test sample dictionaries
    """
    with open(test_samples_path, 'r') as f:
        samples = json.load(f)
    
    if num_samples is not None:
        samples = samples[:num_samples]
    
    print(f"Loaded {len(samples)} test samples")
    return samples


def load_image(image_path: str) -> Image.Image:
    """Load an image from disk.
    
    Args:
        image_path: Path to image file
        
    Returns:
        PIL Image object
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    return Image.open(image_path).convert('RGB')


def format_prompt(instruction: str) -> str:
    """Format the instruction as a prompt.
    
    Args:
        instruction: The image editing instruction
        
    Returns:
        Formatted prompt string
    """
    # Use the same format as training
    prompt = f"<image>\nUser: {instruction}\nAssistant:"
    return prompt


def get_inference_engine(args: Namespace, model: MegatronModule) -> MCoreEngine:
    """Create the inference engine with VLM wrapper.
    
    Args:
        args: Command line arguments
        model: The Megatron model
        
    Returns:
        MCoreEngine configured for VLM inference
    """
    tokenizer = get_tokenizer()
    
    # Create inference wrapper config
    inference_wrapper_config = InferenceWrapperConfig(
        hidden_size=args.hidden_size,
        inference_batch_times_seqlen_threshold=args.inference_batch_times_seqlen_threshold,
        fp32_residual_connection=args.fp32_residual_connection,
        params_dtype=args.params_dtype,
        padded_vocab_size=args.padded_vocab_size
    )
    
    # Create VLM inference wrapper (not GPT wrapper!)
    inference_wrapped_model = VLMInferenceWrapper(model, inference_wrapper_config)
    
    # Create text generation controller
    text_generation_controller = SimpleTextGenerationController(
        inference_wrapped_model=inference_wrapped_model,
        tokenizer=tokenizer
    )
    
    # Create and return engine
    return MCoreEngine(
        text_generation_controller=text_generation_controller,
        max_batch_size=args.max_batch_size
    )


def run_inference(args):
    """Main inference loop.
    
    Args:
        args: Command line arguments
    """
    # Load test samples
    samples_dir = os.path.dirname(args.test_samples)
    test_samples = load_test_samples(args.test_samples, args.num_samples)
    
    # Set up model and load checkpoint
    print("Loading model...")
    model = get_model(model_provider, wrap_with_ddp=False)
    
    print(f"Loading checkpoint from {args.load}...")
    load_checkpoint(model, None, None)
    model = model[0]
    
    print("Setting up inference engine...")
    inference_engine = get_inference_engine(args, model)
    
    # Set up inference parameters
    common_inference_params = CommonInferenceParams(
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        num_tokens_to_generate=args.num_tokens_to_generate
    )
    
    # Run inference on each sample
    results = []
    
    for idx, sample in enumerate(test_samples):
        sample_id = sample['id']
        instruction = sample['instruction']
        
        # Construct image path
        input_image_path = os.path.join(samples_dir, f"{sample_id}.input.jpg")
        
        print(f"\n{'='*80}")
        print(f"[{idx+1}/{len(test_samples)}] Processing {sample_id}")
        print(f"Instruction: {instruction}")
        print(f"Image: {input_image_path}")
        
        # Check if image exists
        if not os.path.exists(input_image_path):
            print(f"WARNING: Image not found, skipping: {input_image_path}")
            continue
        
        # Format prompt
        prompt = format_prompt(instruction)
        
        try:
            # Run inference
            # NOTE: For now we're just testing text generation
            # Full multimodal inference will require image processing
            inference_results: List[InferenceRequest] = inference_engine.generate(
                prompts=[prompt],
                common_inference_params=common_inference_params
            )
            
            if torch.distributed.get_rank() == 0:
                result = inference_results[0]
                generated_text = result.generated_text
                
                print(f"\nGenerated response:")
                print(f"{generated_text}")
                
                # Store result
                results.append({
                    'id': sample_id,
                    'instruction': instruction,
                    'prompt': prompt,
                    'generated_text': generated_text,
                    'generated_tokens': result.generated_tokens
                })
        
        except Exception as e:
            print(f"ERROR during inference: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save results
    if torch.distributed.get_rank() == 0:
        print(f"\n{'='*80}")
        print(f"Saving results to {args.output}")
        
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Inference complete! Processed {len(results)}/{len(test_samples)} samples")


def main():
    """Main entry point."""
    
    # Initialize Megatron with inference-specific defaults
    initialize_megatron(
        extra_args_provider=add_inference_args,
        args_defaults={
            'no_load_rng': True,
            'no_load_optim': True,
            'micro_batch_size': 1,
            'exit_on_missing_checkpoint': True,
            # Model configuration (same as training)
            'model_name': 'llava-ov-1.5-4b',
            'tokenizer_type': 'HFTokenizer',
            'image_resolution': 1000,
            'training_phase': 'sft',
            'seq_length': 32768,
            'max_position_embeddings': 32768,
        }
    )
    
    args = get_args()
    
    # Run inference
    run_inference(args)


if __name__ == "__main__":
    main()
