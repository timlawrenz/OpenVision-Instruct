# Data Preparation Plan

This document outlines the process for transforming the OpenGPT-4o-Image dataset into the specific format required by the LLaVA-OneVision fine-tuning script.

## 1. Source Format Analysis (OpenGPT-4o-Image)

The source dataset, specifically the `editing.json` and `gen.json` files, is a JSON array of objects. Each object represents a single data point and has the following structure:

```json
{
  "input_prompt": "Remove the word 'SALAD' at the top of the chalkboard.",
  "input_image": [
    "editing/input_0.png"
  ],
  "output_image": "editing/output_0.png"
}
```

- **`input_prompt`**: The textual instruction for the model.
- **`input_image`**: A list containing the relative path to the source image.
- **`output_image`**: The relative path to the target (edited) image. For our fine-tuning purpose, this field will be ignored as we are training the model to understand the instruction, not to compare input/output images directly in this conversational format.

## 2. Target Format Analysis (LLaVA-OneVision)

**Update:** After deeper analysis of the `vendor/LLaVA-OneVision` codebase, specifically the `aiak_training_llm/train/arguments.py` script and its default `configs/sft_dataset_config.json`, it was discovered that the training framework has a predefined `multimodal` configuration. To avoid modifying the vendor code, we will adapt our dataset to this expected format.

The required structure is as follows:

```json
{
  "images": ["path/to/input_image.png"],
  "messages": [
    {
      "role": "user",
      "content": "<image>\nYour instruction text here."
    },
    {
      "role": "assistant",
      "content": "A confirmation or descriptive response."
    }
  ]
}
```

- **`images`**: A list containing the relative path(s) to the input image(s).
- **`messages`**: A list of conversation turns, similar to the ShareGPT format.
    - The "human" role is mapped to `"user"`.
    - The "gpt" role is mapped to `"assistant"`.
    - The keys for speaker and text are `"role"` and `"content"`, respectively.

This change simplifies the integration, as we can now use a built-in data format.

## 3. Pre-processing: Data Extraction

The dataset is downloaded as a multi-part `tar.gz` archive (e.g., `editing.tar.gz.00`, `editing.tar.gz.01`, etc.). Before the data can be processed, these parts must be combined and extracted.

Use the following command to concatenate the archive parts and extract their contents into the dataset directory:

```bash
cat data/OpenGPT-4o-Image/editing.tar.gz.* | tar -xzvf - -C data/OpenGPT-4o-Image/
```

This command will unpack the actual image files (e.g., `editing/input_0.png`) into the `data/OpenGPT-4o-Image/` directory, making them accessible for the transformation script.

## 4. Transformation Plan

We will create a Python script (`scripts/prepare_data.py`) to perform the conversion. The script will execute the following steps:

1.  **Load Source Data**: Read the `data/OpenGPT-4o-Image/editing.json` and `data/OpenGPT-4o-Image/gen.json` files.
2.  **Initialize Target List**: Create an empty list that will hold the converted data objects.
3.  **Iterate and Transform**: Loop through each object in the source JSON array. For each object:
    a. Create a new dictionary for the target `multimodal` format.
    b. Set the `"images"` key to a list containing the first element of the source `"input_image"` list.
    c. Construct the `"messages"` list:
        i. Create the "user" turn. The `"content"` will be `f"<image>\n{source_object['input_prompt']}"`.
        ii. Create the "assistant" turn. We will use a simple, consistent response, such as `"Acknowledged."`.
    d. Append the newly created dictionary to the target list.
4.  **Save Target Data**: After processing all source objects, save the target list to a new file, `data/finetune_data_multimodal.json`, in the required format.
5.  **Handle Image Paths**: The script must ensure that the image paths in the final JSON are relative to the directory where the training script will be run (likely the root of the LLaVA repository). We will need to adjust the paths accordingly (e.g., `../../data/OpenGPT-4o-Image/editing/input_0.png`).

```