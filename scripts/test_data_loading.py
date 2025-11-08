import os
import sys

# Check if path exists
path = '/workspace/OpenVision-Instruct/data/OpenGPT-4o-Image-wds'
print(f"Checking if path exists: {path}")
print(f"os.path.exists: {os.path.exists(path)}")
print(f"os.path.isdir: {os.path.isdir(path)}")
if os.path.exists(path):
    print(f"Contents: {os.listdir(path)[:10]}")
    print(f"\nChecking for .wds subdirectory:")
    wds_path = os.path.join(path, '.wds')
    print(f"  .wds exists: {os.path.exists(wds_path)}")
    if os.path.exists(wds_path):
        print(f"  .wds contents: {os.listdir(wds_path)}")

# Try loading with Energon
try:
    from megatron.energon import load_dataset
    from megatron.energon.epathlib import EPath
    
    print("\nChecking EPath resolution:")
    epath = EPath(path)
    print(f"  EPath: {epath}")
    print(f"  EPath type: {type(epath)}")
    
    print("\nTrying to load dataset...")
    dataset = load_dataset('/workspace/OpenVision-Instruct/data/dataset_config.yaml')
    print("SUCCESS! Dataset loaded.")
    print(f"Dataset type: {type(dataset)}")
    print(f"Dataset: {dataset}")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
