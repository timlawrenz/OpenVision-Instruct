# Model Validation and Quality Testing Guide

**Purpose:** Validate that training improves image editing instruction-following quality

**Last Updated:** 2025-11-04

---

## Overview

This guide covers how to evaluate whether the trained model has improved on the image editing task compared to the base model.

## Evaluation Strategy

### 1. Qualitative Evaluation (Visual Inspection)

**Purpose:** Human assessment of editing quality

**Steps:**

#### A. Create a Test Set
```bash
# Select diverse test samples (not in training data)
mkdir -p evaluation/test_samples
# Copy 50-100 representative samples
```

#### B. Run Inference on Base Model
```python
# inference_base.py
from PIL import Image
import torch

def test_base_model(checkpoint_path, test_images, instructions):
    """
    Run base model (before fine-tuning) on test set
    """
    # Load base checkpoint
    model = load_checkpoint(checkpoint_path_base)
    
    results = []
    for img, instruction in zip(test_images, instructions):
        # Generate response
        response = model.generate(
            image=img,
            prompt=f"<image>\n{instruction}"
        )
        results.append({
            'instruction': instruction,
            'response': response,
            'image': img
        })
    
    return results
```

#### C. Run Inference on Fine-tuned Model
```python
# inference_finetuned.py
# Same as above but load fine-tuned checkpoint
model = load_checkpoint(checkpoint_path_finetuned)
```

#### D. Compare Results
Create side-by-side comparison:
```
Input Image | Instruction | Base Model Response | Fine-tuned Model Response
-----------|-------------|---------------------|-------------------------
[image]    | "Remove X"  | "I will process..." | "I will edit the image..."
```

**Metrics to assess:**
- ✅ Instruction understanding (does it understand what to edit?)
- ✅ Response relevance (appropriate for image editing task?)
- ✅ Specificity (mentions specific elements to edit?)
- ✅ Confidence (clear vs. vague language?)

---

### 2. Quantitative Evaluation (Automated Metrics)

#### A. Response Perplexity
Measure how confident the model is on held-out test data:

```python
def calculate_perplexity(model, test_dataloader):
    """
    Lower perplexity = better fit to data distribution
    """
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in test_dataloader:
            loss = model.compute_loss(batch)
            total_loss += loss.item() * batch['num_tokens']
            total_tokens += batch['num_tokens']
    
    perplexity = torch.exp(torch.tensor(total_loss / total_tokens))
    return perplexity.item()

# Compare
base_ppl = calculate_perplexity(base_model, test_loader)
finetuned_ppl = calculate_perplexity(finetuned_model, test_loader)

print(f"Base model perplexity: {base_ppl:.2f}")
print(f"Fine-tuned perplexity: {finetuned_ppl:.2f}")
print(f"Improvement: {(base_ppl - finetuned_ppl) / base_ppl * 100:.1f}%")
```

#### B. Response Quality Metrics

**BLEU/ROUGE scores** (if you have reference responses):
```python
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge

def evaluate_response_quality(predictions, references):
    """
    Compare generated responses to reference responses
    """
    rouge = Rouge()
    
    bleu_scores = []
    rouge_scores = []
    
    for pred, ref in zip(predictions, references):
        # BLEU score
        bleu = sentence_bleu([ref.split()], pred.split())
        bleu_scores.append(bleu)
        
        # ROUGE score
        rouge_score = rouge.get_scores(pred, ref)[0]
        rouge_scores.append(rouge_score['rouge-l']['f'])
    
    return {
        'avg_bleu': np.mean(bleu_scores),
        'avg_rouge_l': np.mean(rouge_scores)
    }
```

#### C. Task-Specific Accuracy

For image editing, measure:
- **Instruction parsing accuracy:** Does the model correctly identify what to edit?
- **Action correctness:** Does it choose appropriate action verbs (remove, add, change)?

```python
def evaluate_instruction_understanding(response, instruction):
    """
    Check if model understood the editing task
    """
    # Extract key elements from instruction
    action_words = ['remove', 'add', 'change', 'replace', 'delete']
    instruction_lower = instruction.lower()
    
    # Check if model mentions the action
    action_mentioned = any(action in response.lower() 
                          for action in action_words 
                          if action in instruction_lower)
    
    # Check if model mentions the target object
    # (simplified - you'd use NER or object detection)
    objects_in_instruction = extract_objects(instruction)
    objects_in_response = extract_objects(response)
    object_overlap = len(set(objects_in_instruction) & 
                        set(objects_in_response))
    
    return {
        'action_mentioned': action_mentioned,
        'object_recall': object_overlap / len(objects_in_instruction)
    }
```

---

### 3. Benchmark Evaluation

#### Standard Vision-Language Benchmarks

Even though this is a specialized task, test on standard benchmarks to ensure no degradation:

```bash
# VQAv2 (Visual Question Answering)
python evaluate_vqa.py \
    --checkpoint checkpoints/finetuned_model \
    --dataset vqav2 \
    --split val

# GQA (General Question Answering)
python evaluate_gqa.py \
    --checkpoint checkpoints/finetuned_model \
    --dataset gqa \
    --split val
```

**Expected result:** Performance should be maintained or slightly degraded (acceptable for specialized fine-tuning)

---

### 4. A/B Testing with Human Evaluators

#### Setup Human Evaluation Study

Create evaluation interface:
```
Question: Which response better follows the instruction?

Input Image: [show image]
Instruction: "Remove the word 'SALE' from the sign"

Response A: "I will edit the image as requested."
Response B: "I'll remove the 'SALE' text from the sign in the image."

[ ] Response A is better
[ ] Response B is better  
[ ] Both are equally good
[ ] Both are equally bad
```

**Metrics:**
- Win rate: % of time fine-tuned model is preferred
- Agreement: Inter-rater reliability (Cohen's kappa)

---

## Practical Evaluation Script

Create a simple evaluation script:

```python
#!/usr/bin/env python3
"""
evaluate_model.py - Compare base and fine-tuned models
"""

import torch
from pathlib import Path
import json
from tqdm import tqdm

def load_test_data(test_file):
    """Load test samples from JSON"""
    with open(test_file) as f:
        return json.load(f)

def run_evaluation(base_ckpt, finetuned_ckpt, test_data):
    """Compare models on test data"""
    
    results = {
        'base': [],
        'finetuned': [],
        'test_samples': []
    }
    
    # Load models
    print("Loading base model...")
    base_model = load_model(base_ckpt)
    
    print("Loading fine-tuned model...")
    finetuned_model = load_model(finetuned_ckpt)
    
    # Evaluate
    print("Running evaluation...")
    for sample in tqdm(test_data):
        image = load_image(sample['image_path'])
        instruction = sample['instruction']
        
        # Base model
        base_response = base_model.generate(
            image=image,
            instruction=instruction
        )
        
        # Fine-tuned model
        finetuned_response = finetuned_model.generate(
            image=image,
            instruction=instruction
        )
        
        results['test_samples'].append({
            'instruction': instruction,
            'image_path': sample['image_path'],
        })
        results['base'].append(base_response)
        results['finetuned'].append(finetuned_response)
    
    return results

def analyze_results(results):
    """Compute metrics"""
    
    # Length comparison (more specific = longer)
    base_lengths = [len(r.split()) for r in results['base']]
    finetuned_lengths = [len(r.split()) for r in results['finetuned']]
    
    print("\n=== Evaluation Results ===")
    print(f"Base model avg response length: {np.mean(base_lengths):.1f} words")
    print(f"Fine-tuned avg response length: {np.mean(finetuned_lengths):.1f} words")
    
    # Task-specific keywords
    task_keywords = ['edit', 'remove', 'add', 'change', 'modify']
    base_keyword_rate = sum(any(kw in r.lower() for kw in task_keywords) 
                           for r in results['base']) / len(results['base'])
    finetuned_keyword_rate = sum(any(kw in r.lower() for kw in task_keywords) 
                                 for r in results['finetuned']) / len(results['finetuned'])
    
    print(f"\nTask keyword usage:")
    print(f"Base model: {base_keyword_rate:.1%}")
    print(f"Fine-tuned: {finetuned_keyword_rate:.1%}")
    
    # Save detailed results
    with open('evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nDetailed results saved to evaluation_results.json")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-ckpt', required=True)
    parser.add_argument('--finetuned-ckpt', required=True)
    parser.add_argument('--test-data', required=True)
    args = parser.parse_args()
    
    test_data = load_test_data(args.test_data)
    results = run_evaluation(args.base_ckpt, args.finetuned_ckpt, test_data)
    analyze_results(results)
```

**Usage:**
```bash
python evaluate_model.py \
    --base-ckpt checkpoints/LLaVA-OneVision-1.5-4B-stage0 \
    --finetuned-ckpt checkpoints/OpenVision-Instruct-4B-adapter \
    --test-data evaluation/test_samples.json
```

---

## Success Criteria

Your fine-tuning is successful if:

✅ **Loss decreased:** Training loss dropped significantly (1.1 → 0.03)  
✅ **Task alignment:** Responses use appropriate editing terminology  
✅ **Specificity:** Model mentions specific elements to edit  
✅ **No catastrophic forgetting:** Maintains base capability on general tasks  
✅ **Human preference:** Evaluators prefer fine-tuned responses 60%+ of time  

---

## Quick Validation Checklist

Before running full evaluation:

```bash
# 1. Check training loss converged
tail -100 runs/training_*.log | grep "lm loss"

# 2. Verify checkpoint saved
ls -lh checkpoints/OpenVision-Instruct-4B-adapter/

# 3. Quick sanity test on single sample
python scripts/quick_inference_test.py \
    --checkpoint checkpoints/OpenVision-Instruct-4B-adapter \
    --image evaluation/sample.jpg \
    --instruction "Remove the word 'STOP' from the sign"

# 4. Compare with base model
python scripts/quick_inference_test.py \
    --checkpoint checkpoints/LLaVA-OneVision-1.5-4B-stage0 \
    --image evaluation/sample.jpg \
    --instruction "Remove the word 'STOP' from the sign"
```

---

## Next Steps

1. **Create test set:** Select 100-200 diverse samples
2. **Run qualitative evaluation:** Visual inspection of results
3. **Run quantitative evaluation:** Automated metrics
4. **Document findings:** Record improvements and failure cases
5. **Iterate:** If quality is insufficient, continue training or adjust hyperparameters

---

## References

- Training logs: `runs/training_20251103_*.log`
- Checkpoint: `checkpoints/OpenVision-Instruct-4B-adapter/`
- Base model: `checkpoints/LLaVA-OneVision-1.5-4B-stage0/`
