#!/usr/bin/env python3
"""
Visual comparison tool for image editing model
Shows: Input Image | Model Output | Expected Output
"""

import json
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
import argparse

def create_comparison_image(input_img, model_response, expected_img, instruction, sample_id):
    """Create a side-by-side comparison image with the model's text response"""
    
    # Resize images to same height
    target_height = 400
    
    def resize_keeping_aspect(img, height):
        aspect = img.width / img.height
        new_width = int(height * aspect)
        return img.resize((new_width, height), Image.Resampling.LANCZOS)
    
    input_resized = resize_keeping_aspect(input_img, target_height)
    expected_resized = resize_keeping_aspect(expected_img, target_height)
    
    # Create canvas
    padding = 20
    text_height = 150
    total_width = input_resized.width + expected_resized.width + padding * 3
    total_height = target_height + text_height + padding * 3
    
    canvas = Image.new('RGB', (total_width, total_height), 'white')
    draw = ImageDraw.Draw(canvas)
    
    # Try to use a nice font, fallback to default
    try:
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        text_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except:
        title_font = ImageFont.load_default()
        text_font = ImageFont.load_default()
    
    # Draw instruction at top
    y_pos = padding
    draw.text((padding, y_pos), f"Sample: {sample_id}", fill='black', font=title_font)
    y_pos += 25
    
    # Wrap instruction text
    instruction_wrapped = instruction[:80] + "..." if len(instruction) > 80 else instruction
    draw.text((padding, y_pos), f"Instruction: {instruction_wrapped}", fill='black', font=text_font)
    y_pos += 30
    
    # Place images side by side
    img_y = y_pos + padding
    
    # Input image
    canvas.paste(input_resized, (padding, img_y))
    draw.text((padding, img_y - 20), "INPUT", fill='blue', font=title_font)
    
    # Expected output image
    expected_x = padding * 2 + input_resized.width
    canvas.paste(expected_resized, (expected_x, img_y))
    draw.text((expected_x, img_y - 20), "EXPECTED OUTPUT", fill='green', font=title_font)
    
    # Model response text at bottom
    response_y = img_y + target_height + padding
    draw.text((padding, response_y), "MODEL OUTPUT:", fill='red', font=title_font)
    
    # Wrap model response
    response_text = model_response[:200] + "..." if len(model_response) > 200 else model_response
    y_offset = response_y + 25
    max_width = total_width - padding * 2
    
    # Simple text wrapping
    words = response_text.split()
    lines = []
    current_line = []
    for word in words:
        test_line = ' '.join(current_line + [word])
        bbox = draw.textbbox((0, 0), test_line, font=text_font)
        if bbox[2] - bbox[0] < max_width:
            current_line.append(word)
        else:
            if current_line:
                lines.append(' '.join(current_line))
            current_line = [word]
    if current_line:
        lines.append(' '.join(current_line))
    
    for line in lines[:3]:  # Max 3 lines
        draw.text((padding, y_offset), line, fill='black', font=text_font)
        y_offset += 20
    
    return canvas

def test_sample(model, processor, sample_dir, sample_id, output_dir, max_new_tokens=512):
    """Test a single sample and create comparison"""
    
    sample_file = sample_dir / f"{sample_id}.json"
    input_image_file = sample_dir / f"{sample_id}.input.jpg"
    expected_image_file = sample_dir / f"{sample_id}.output.jpg"
    
    # Load instruction
    with open(sample_file) as f:
        instruction = f.read().strip().strip('"')
    
    print(f"\n{'='*80}")
    print(f"Testing Sample: {sample_id}")
    print(f"{'='*80}")
    print(f"Instruction: {instruction}")
    print()
    
    # Load images
    input_img = Image.open(input_image_file).convert("RGB")
    expected_img = Image.open(expected_image_file).convert("RGB")
    
    print("Generating model response...")
    
    # Format prompt using chat template
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": instruction}
            ]
        }
    ]
    
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=prompt, images=[input_img], return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id
        )
    
    # Decode only the new tokens (skip input)
    input_length = inputs['input_ids'].shape[1]
    generated_ids = output_ids[0][input_length:]
    response = processor.tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    print(f"\nModel Response:")
    print(f"  {response[:300]}")
    if len(response) > 300:
        print(f"  ... (truncated, total {len(response)} chars)")
    print()
    
    # Create comparison image
    comparison = create_comparison_image(
        input_img, response, expected_img, instruction, sample_id
    )
    
    # Save comparison
    output_file = output_dir / f"{sample_id}_comparison.jpg"
    comparison.save(output_file, quality=95)
    
    print(f"✅ Comparison saved to: {output_file}")
    print()
    print("WHAT TO CHECK:")
    print("  1. Does the model response make sense?")
    print("  2. Is it describing an editing operation?")
    print("  3. Does it mention the specific elements from the instruction?")
    print("  4. Is it coherent (not gibberish)?")
    print()
    print("NOTE: Your model only generates TEXT instructions, not edited images.")
    print("      Compare the response quality, not actual image edits.")
    print()
    
    return {
        "sample_id": sample_id,
        "instruction": instruction,
        "model_response": response,
        "comparison_image": str(output_file)
    }

def main():
    parser = argparse.ArgumentParser(description="Test image editing model with visual comparison")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to HF checkpoint")
    parser.add_argument("--sample-id", type=str, default="sample_604",
                        help="Sample ID to test (default: sample_604)")
    parser.add_argument("--samples-dir", type=str, default="evaluation/test_samples",
                        help="Directory containing test samples")
    parser.add_argument("--output-dir", type=str, default="evaluation/visual_tests",
                        help="Output directory for comparison images")
    parser.add_argument("--max-new-tokens", type=int, default=512,
                        help="Maximum new tokens to generate")
    parser.add_argument("--all-samples", action="store_true",
                        help="Test all samples in the directory")
    args = parser.parse_args()
    
    print("="*80)
    print("IMAGE EDITING MODEL - VISUAL COMPARISON TEST")
    print("="*80)
    print()
    
    # Load model
    print(f"Loading model from: {args.checkpoint}")
    processor = AutoProcessor.from_pretrained(args.checkpoint, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    print(f"✅ Model loaded: {model.num_parameters() / 1e9:.2f}B parameters")
    print()
    
    # Setup directories
    sample_dir = Path(args.samples_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get samples to test
    if args.all_samples:
        sample_files = sorted(sample_dir.glob("*.json"))
        sample_ids = [f.stem for f in sample_files]
    else:
        sample_ids = [args.sample_id]
    
    print(f"Testing {len(sample_ids)} sample(s)...")
    print()
    
    # Test each sample
    results = []
    for sample_id in sample_ids:
        try:
            result = test_sample(
                model, processor, sample_dir, sample_id, output_dir, args.max_new_tokens
            )
            results.append(result)
        except Exception as e:
            print(f"❌ Error testing {sample_id}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save results JSON
    results_file = output_dir / "test_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("="*80)
    print("TESTING COMPLETE")
    print("="*80)
    print()
    print(f"Results: {results_file}")
    print(f"Comparison images: {output_dir}/")
    print()
    print("To view comparisons, open the generated JPG files:")
    for result in results:
        print(f"  {result['comparison_image']}")
    print()

if __name__ == "__main__":
    main()
