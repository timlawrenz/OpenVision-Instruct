# Image Editing Dataset Dataloader

## 🎯 Quick Reference

**Working sample loader location:**
```
data/OpenGPT-4o-Image-wds/.nv-meta/sample_loader.py
```

**Dataset configuration:**
```
data/OpenGPT-4o-Image-wds/.nv-meta/dataset.yaml
```

**Documentation:**
```
docs/DATALOADER_FIXES.md
```

## ✅ Status

**Training Status:** Successfully running as of 2025-11-04  
**Loss:** Converging from 1.12 → 0.03 in first 60 iterations  
**Errors:** Zero skipped or NaN iterations  

## 🔑 Key Implementation Points

### 1. Use Correct Sample Type
```yaml
sample_type:
  __class__: MultiMixQASample  # NOT MultiVidQASample
```

### 2. Handle WebDataset Auto-Decoding
WebDataset automatically converts images to tensors. Convert them back:
```python
if isinstance(image_data, torch.Tensor):
    array = (image_data.permute(1, 2, 0).numpy() * 255).astype('uint8')
    image = Image.fromarray(array, mode='RGB')
```

### 3. Smart `<image>` Tag Handling
Dataset is inconsistent. Check before adding:
```python
if '<image>' not in instruction:
    content = f"<image>\n{instruction}"
```

### 4. Return PIL Images
Task encoder needs PIL Images, not tensors:
```python
return dict(
    image=[input_image],  # PIL Image object
    video=None,
    ...
)
```

## 📊 Dataset Structure

```
OpenGPT-4o-Image-wds/
├── sft-0.tar
│   ├── sample_0.input.jpg    # Input image
│   ├── sample_0.output.jpg   # Target result (loaded but not used in training)
│   └── sample_0.json         # Instruction (may or may not have <image> tag)
├── sft-1.tar
└── ...
```

## 🚀 For Researchers

If you're working with similar image editing datasets:

1. **Copy the sample loader:** `data/OpenGPT-4o-Image-wds/.nv-meta/sample_loader.py`
2. **Adapt the message format:** Modify lines 74-88 for your instruction format
3. **Update dataset config:** Use `MultiMixQASample` for image-based tasks
4. **Test thoroughly:** Use `scripts/test_sample_loader.py`

## 🐛 Troubleshooting

| Error | Solution |
|-------|----------|
| `JSONDecodeError` | Handle both quoted strings and plain text (see lines 36-40) |
| `AttributeError: 'Tensor' object has no attribute 'height'` | Convert tensors to PIL Images (lines 51-55) |
| `IndexError: index N is out of bounds` | Mismatch between `<image>` tags and images (lines 76-79) |
| `TypeError: a bytes-like object is required` | Handle multiple input types (lines 50-72) |

## 📝 Complete Documentation

See `docs/DATALOADER_FIXES.md` for:
- Detailed error analysis
- Step-by-step fixes applied
- Training metrics
- Code explanations

## 🧪 Model Validation

See `docs/MODEL_VALIDATION.md` for:
- Quality evaluation strategies
- Quantitative metrics (perplexity, BLEU, ROUGE)
- Qualitative evaluation (human assessment)
- A/B testing methodology
- Sample evaluation scripts

## 🎓 Lessons Learned

1. **Framework auto-processing:** WebDataset may decode images before your loader runs
2. **Dataset inconsistencies:** Always validate input format assumptions
3. **Type expectations:** Task encoders may expect specific types (PIL vs tensor)
4. **Token counting:** Ensure 1:1 mapping between `<image>` tags and actual images

---

**Last Updated:** 2025-11-04  
**Training Run:** `runs/training_20251103_235510.log`
