#!/usr/bin/env python3
"""
Quick test script to validate sample_loader without waiting for model loading.
"""
import sys
sys.path.insert(0, '/workspace/OpenVision-Instruct/data/OpenGPT-4o-Image-wds/.nv-meta')

from sample_loader import sample_loader
import tarfile
import io

def test_sample_loader():
    """Test the sample_loader with actual data from the tar files"""
    
    # Open the first tar file
    tar_path = '/workspace/OpenVision-Instruct/data/OpenGPT-4o-Image-wds/sft-0.tar'
    print(f"Opening tar file: {tar_path}")
    
    with tarfile.open(tar_path, 'r') as tar:
        # Get list of members
        members = tar.getmembers()
        print(f"Found {len(members)} members in tar file")
        
        # Group by sample ID (e.g., "sample_0" from "sample_0.json")
        samples = {}
        for member in members:
            if not member.isfile():
                continue
            # Extract sample ID (e.g., "sample_0" from "sample_0.json" or "sample_0.input.jpg")
            name = member.name
            # Split by first dot to get base name
            sample_id = name.split('.')[0]
            
            if sample_id not in samples:
                samples[sample_id] = {}
            
            # Determine the key
            if name.endswith('.json'):
                samples[sample_id]['json'] = member
            elif 'input' in name:
                samples[sample_id]['input.jpg'] = member
            elif 'output' in name:
                samples[sample_id]['output.jpg'] = member
        
        print(f"Found {len(samples)} samples")
        
        # Test first sample
        sample_id = sorted(samples.keys())[0]
        print(f"\nTesting sample: {sample_id}")
        print(f"Keys: {list(samples[sample_id].keys())}")
        
        # Load the sample data
        sample_data = {}
        for key, member in samples[sample_id].items():
            content = tar.extractfile(member).read()
            sample_data[key] = content
            if key == 'json':
                print(f"JSON content: {content[:200]}")
            else:
                print(f"{key}: {len(content)} bytes")
        
        # Try to load with sample_loader
        print("\nAttempting to load sample...")
        try:
            result = sample_loader(sample_data)
            print("✓ SUCCESS!")
            print(f"Result type: {type(result)}")
            print(f"Video data: {len(result.video)} frame(s)")
            print(f"Messages: {result.messages}")
            print(f"System: {result.system}")
        except Exception as e:
            print(f"✗ ERROR: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    return True

if __name__ == '__main__':
    success = test_sample_loader()
    sys.exit(0 if success else 1)
