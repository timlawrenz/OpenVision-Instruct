import json

def prepare_data(input_json_path, output_jsonl_path):
    """
    Prepares the OpenGPT-4o-Image dataset for LLaVA-OneVision fine-tuning.
    Outputs JSONL format with one conversation per line.
    """
    with open(input_json_path, 'r') as f_in:
        data = json.load(f_in)
        
    with open(output_jsonl_path, 'w') as f_out:
        for i, item in enumerate(data):
            if not isinstance(item.get('input_image'), list) or not item['input_image']:
                continue

            input_image_path = item['input_image'][0]
            
            conversation = {
                "id": f"identity_{i}",
                "image": f"data/OpenGPT-4o-Image/{input_image_path}",
                "conversations": [
                    {
                        "from": "human",
                        "value": f"<image>\n{item['input_prompt']}"
                    },
                    {
                        "from": "gpt",
                        "value": "Acknowledged."
                    }
                ]
            }
            f_out.write(json.dumps(conversation) + '\n')

if __name__ == '__main__':
    prepare_data(
        'data/OpenGPT-4o-Image/editing.json',
        'data/prepared_data.jsonl'
    )
    print("Data preparation complete. Output written to data/prepared_data.jsonl")
