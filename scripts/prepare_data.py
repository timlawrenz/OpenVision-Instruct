import json
import os
from pathlib import Path
import argparse

# The LLaVA training scripts are typically run from the root of their repository.
# We need to make the image paths relative to that location.
# Root of our project -> vendor/LLaVA-OneVision
PATH_TO_TRAINING_ROOT = Path("../..")
DATA_ROOT = "data/OpenGPT-4o-Image"

def transform_data(source_path: Path, output_path: Path):
    """
    Loads the OpenGPT-4o-Image dataset and transforms it into the LLaVA
    conversational format for fine-tuning.

    Args:
        source_path: Path to the directory containing the source JSON files.
        output_path: Path to save the transformed JSON file.
    """
    print(f"Loading source data from {source_path}...")
    
    editing_json = source_path / "editing.json"
    gen_json = source_path / "gen.json"
    
    source_data = []
    if editing_json.exists():
        with open(editing_json, 'r', encoding='utf-8') as f:
            source_data.extend(json.load(f))
        print(f"Loaded {len(source_data)} records from editing.json")
    else:
        print("Warning: editing.json not found.")

    if gen_json.exists():
        start_len = len(source_data)
        with open(gen_json, 'r', encoding='utf-8') as f:
            source_data.extend(json.load(f))
        print(f"Loaded {len(source_data) - start_len} records from gen.json")
    else:
        print("Warning: gen.json not found.")

    if not source_data:
        print("Error: No source data found. Exiting.")
        return

    converted_data = []
    print("Transforming data to LLaVA conversational format...")

    for item in source_data:
        # Ensure input_image is a list and not empty
        if not isinstance(item.get("input_image"), list) or not item["input_image"]:
            continue

        # The image path from the source file is relative to the dataset root,
        # e.g., "editing/input_0.png"
        original_image_path = item["input_image"][0]
        
        # We construct the full relative path from the training script's perspective
        relative_image_path = PATH_TO_TRAINING_ROOT / DATA_ROOT / original_image_path

        transformed_item = {
            "images": [str(relative_image_path)],
            "messages": [
                {
                    "role": "user",
                    "content": f"<image>\n{item['input_prompt']}"
                },
                {
                    "role": "assistant",
                    "content": "Acknowledged."
                }
            ]
        }
        converted_data.append(transformed_item)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, indent=2)

    print(f"Successfully transformed {len(converted_data)} records.")
    print(f"Output saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transform OpenGPT-4o-Image data to LLaVA multimodal format.")
    parser.add_argument(
        "--source_dir",
        type=Path,
        default=Path("data/OpenGPT-4o-Image"),
        help="Path to the directory containing the source dataset files."
    )
    parser.add_argument(
        "--output_file",
        type=Path,
        default=Path("data/finetune_data_multimodal.json"),
        help="Path to save the transformed JSON file."
    )
    args = parser.parse_args()

    transform_data(args.source_dir, args.output_file)
