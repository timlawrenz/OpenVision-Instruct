#!/usr/bin/env python3
"""
Simple visual inspection tool for test samples

Shows test images and instructions to help with manual quality assessment
"""

import json
from pathlib import Path
from PIL import Image

def view_test_samples():
    """Display test samples for visual inspection"""
    
    test_dir = Path('evaluation/test_samples')
    metadata_file = test_dir / 'test_samples.json'
    
    if not metadata_file.exists():
        print(f"❌ Metadata file not found: {metadata_file}")
        return
    
    with open(metadata_file) as f:
        samples = json.load(f)
    
    print("=" * 70)
    print("TEST SAMPLE VISUAL INSPECTION")
    print("=" * 70)
    print(f"\nTotal samples: {len(samples)}\n")
    
    for i, sample in enumerate(samples, 1):
        sample_id = sample['id']
        instruction = sample['instruction']
        
        input_img = test_dir / f"{sample_id}.input.jpg"
        output_img = test_dir / f"{sample_id}.output.jpg"
        
        print(f"\n{'='*70}")
        print(f"Sample {i}/{len(samples)}: {sample_id}")
        print(f"{'='*70}")
        print(f"\n📝 Instruction:")
        print(f"   {instruction}")
        
        if input_img.exists():
            img = Image.open(input_img)
            print(f"\n🖼️  Input Image:")
            print(f"   File: {input_img.name}")
            print(f"   Size: {img.size[0]}x{img.size[1]} pixels")
            print(f"   Path: {input_img}")
        else:
            print(f"\n❌ Input image not found: {input_img}")
        
        if output_img.exists():
            img = Image.open(output_img)
            print(f"\n🎯 Expected Output:")
            print(f"   File: {output_img.name}")
            print(f"   Size: {img.size[0]}x{img.size[1]} pixels")
            print(f"   Path: {output_img}")
        else:
            print(f"\n❌ Output image not found: {output_img}")
        
        print(f"\n💭 Expected Model Behavior:")
        print(f"   Should understand: Text removal from image")
        print(f"   Should mention: 'edit', 'remove', or similar action")
        print(f"   Should reference: The specific word/element to remove")
        
    print(f"\n{'='*70}")
    print("END OF TEST SAMPLES")
    print(f"{'='*70}\n")
    
    print("📋 Summary of instructions:")
    for i, sample in enumerate(samples, 1):
        print(f"  {i}. {sample['instruction']}")
    
    print(f"\n🔍 To view images, open them with:")
    print(f"   cd evaluation/test_samples/")
    print(f"   xdg-open sample_126.input.jpg  # Or your image viewer")
    
    print(f"\n📊 Next step: Run inference to get model responses")

if __name__ == '__main__':
    view_test_samples()
