#!/usr/bin/env python3
"""
Simplified Megatron inference with HuggingFace tokenizer
"""

import os
import sys
import json
import torch

# Add paths
sys.path.insert(0, 'vendor/LLaVA-OneVision')
sys.path.insert(0, 'vendor/LLaVA-OneVision/aiak_megatron')

from transformers import AutoTokenizer

# Megatron imports
from megatron.training import get_args
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.checkpointing import load_checkpoint
from megatron.training.initialize import initialize_megatron
from megatron.training import get_model
from megatron.core.models.gpt import GPTModel


def add_inference_args(parser):
    """Add inference-specific arguments."""
    group = parser.add_argument_group(title='inference')
    
    group.add_argument("--test-samples", type=str, default="evaluation/test_samples/test_samples.json",
                       help='Path to test samples JSON file')
    group.add_argument("--output", type=str, default="evaluation/inference_results.json",
                       help='Path to save inference results')
    group.add_argument("--num-samples", type=int, default=None,
                       help='Limit number of test samples (for quick testing)')
    group.add_argument("--temperature", type=float, default=0.7,
                       help='Sampling temperature.')
    group.add_argument("--max-new-tokens", type=int, default=512,
                       help='Max tokens to generate')
    
    return parser


def model_provider(pre_process=True, post_process=True):
    """Build model."""
    from megatron.core.transformer.spec_utils import import_module
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
    
    args = get_args()
    config = core_transformer_config_from_args(args)
    
    # Get the transformer layer spec
    transformer_layer_spec = get_gpt_layer_local_spec()
    
    model = GPTModel(
        config=config,
        transformer_layer_spec=transformer_layer_spec,
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        pre_process=pre_process,
        post_process=post_process,
        fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
        parallel_output=True,
        share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        position_embedding_type=args.position_embedding_type,
        rotary_percent=args.rotary_percent
    )
    
    return model


def generate_with_model(model, tokenizer, prompt, args):
    """Generate text using the model directly."""
    # Tokenize
    inputs = tokenizer(prompt, return_tensors='pt')
    input_ids = inputs['input_ids'].cuda()
    
    # Create attention mask (all ones)
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool).cuda()
    
    # Simple greedy generation
    generated_ids = input_ids.clone()
    
    with torch.no_grad():
        for _ in range(args.max_new_tokens):
            # Forward pass
            logits = model(
                input_ids=generated_ids,
                position_ids=torch.arange(generated_ids.size(1), device='cuda').unsqueeze(0),
                attention_mask=attention_mask
            )
            
            # Get next token (greedy)
            next_token_logits = logits[0, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Append
            generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0)], dim=-1)
            
            # Expand attention mask
            attention_mask = torch.cat([attention_mask, torch.ones((1, 1), dtype=torch.bool, device='cuda')], dim=-1)
            
            # Check for EOS
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    # Decode
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return generated_text


def run_inference(args):
    """Main inference loop."""
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    
    print("Loading test samples...")
    with open(args.test_samples, 'r') as f:
        test_samples = json.load(f)
    
    if args.num_samples:
        test_samples = test_samples[:args.num_samples]
    
    print(f"Processing {len(test_samples)} samples")
    
    # Load model
    print("Loading model...")
    model = get_model(model_provider, wrap_with_ddp=False)
    
    print(f"Loading checkpoint from {args.load}...")
    load_checkpoint(model, None, None)
    model = model[0]
    model.eval()
    
    # Run inference
    results = []
    
    for idx, sample in enumerate(test_samples):
        sample_id = sample['id']
        instruction = sample['instruction']
        
        print(f"\n{'='*80}")
        print(f"[{idx+1}/{len(test_samples)}] {sample_id}")
        print(f"Instruction: {instruction}")
        
        prompt = f"User: {instruction}\nAssistant:"
        
        try:
            print("Generating...")
            generated = generate_with_model(model, tokenizer, prompt, args)
            
            # Extract just the response (after "Assistant:")
            if "Assistant:" in generated:
                response = generated.split("Assistant:")[-1].strip()
            else:
                response = generated
            
            print(f"\nResponse: {response[:200]}...")
            
            results.append({
                'id': sample_id,
                'instruction': instruction,
                'prompt': prompt,
                'generated_text': response
            })
            
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save results
    print(f"\n{'='*80}")
    print(f"Saving results to {args.output}")
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Inference complete! Processed {len(results)}/{len(test_samples)} samples")


def main():
    """Main entry point."""
    
    # Don't initialize full Megatron - just use minimal args
    import argparse
    parser = argparse.ArgumentParser()
    
    # Model args
    parser.add_argument('--load', type=str, required=True)
    parser.add_argument('--tokenizer-path', type=str, required=True)
    parser.add_argument('--num-layers', type=int, default=36)
    parser.add_argument('--hidden-size', type=int, default=2560)
    parser.add_argument('--num-attention-heads', type=int, default=32)
    parser.add_argument('--seq-length', type=int, default=32768)
    parser.add_argument('--max-position-embeddings', type=int, default=32768)
    
    # Inference args
    parser.add_argument('--test-samples', type=str, default='evaluation/test_samples/test_samples.json')
    parser.add_argument('--output', type=str, default='evaluation/finetuned_results.json')
    parser.add_argument('--num-samples', type=int, default=None)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--max-new-tokens', type=int, default=256)
    
    # Required Megatron args
    parser.add_argument('--tensor-model-parallel-size', type=int, default=1)
    parser.add_argument('--pipeline-model-parallel-size', type=int, default=1)
    
    args = parser.parse_args()
    
    # Initialize minimal Megatron
    initialize_megatron(
        args_defaults={
            'no_load_rng': True,
            'no_load_optim': True,
            'micro_batch_size': 1,
            'tokenizer_type': 'NullTokenizer',
            'vocab_size': 151936,
            'num_layers': args.num_layers,
            'hidden_size': args.hidden_size,
            'num_attention_heads': args.num_attention_heads,
            'seq_length': args.seq_length,
            'max_position_embeddings': args.max_position_embeddings,
            'load': args.load,
            'tensor_model_parallel_size': args.tensor_model_parallel_size,
            'pipeline_model_parallel_size': args.pipeline_model_parallel_size,
        },
        ignore_unknown_args=True
    )
    
    # Update args with parsed values
    megatron_args = get_args()
    megatron_args.tokenizer_path = args.tokenizer_path
    megatron_args.test_samples = args.test_samples
    megatron_args.output = args.output
    megatron_args.num_samples = args.num_samples
    megatron_args.temperature = args.temperature
    megatron_args.max_new_tokens = args.max_new_tokens
    
    # Run inference
    run_inference(megatron_args)


if __name__ == "__main__":
    main()

