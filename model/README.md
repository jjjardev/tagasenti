---
language:
  - tl
  - en
license: apache-2.0
library_name: transformers
pipeline_tag: text-classification
tags:
  - sentiment-analysis
  - filipino
  - tagalog
  - taglish
  - code-switching
  - xlm-roberta
  - low-resource
base_model: xlm-roberta-large
datasets:
  - jjjardev/tagasenti
metrics:
  - accuracy
  - f1
model-index:
  - name: tagasenti_model
    results:
      - task:
          type: text-classification
          name: Sentiment Analysis (Tagalog/Taglish 3-class)
        dataset:
          type: jjjardev/tagasenti
          name: TagaSenti (deduplicated, 80/10/10 split, seed 42)
          split: test
        metrics:
          - type: accuracy
            value: 0.866
            name: Accuracy
          - type: f1
            value: 0.866
            name: Macro F1
---

# tagasenti_model

Fine-tuned **XLM-RoBERTa-large** (355M) for **Tagalog / Taglish 3-class sentiment** (Negative / Neutral / Positive).

Trained on [jjjardev/tagasenti](https://huggingface.co/datasets/jjjardev/tagasenti) — 35,686 sentences (deduplicated to ~34,945) across e-commerce, news, social and 9,378 adversarial examples. Held-out test: **86.6% accuracy**, **macro-F1 0.866** (stratified 80/10/10, seed 42). Zero-shot on Hiligaynon (HiliSenti): 62.4% accuracy.

**Base model:** `xlm-roberta-large` · **Labels:** `0=Negative`, `1=Neutral`, `2=Positive` · **Max length:** 128 · **License:** Apache 2.0 (weights), CC BY-SA 4.0 (dataset)

## Download

**Hugging Face Hub:** https://huggingface.co/jjjardev/tagasenti_model

### Option 1 — `transformers` (recommended)

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_id = "jjjardev/tagasenti_model"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id)
```

### Option 2 — Pipeline (one-liner)

```python
from transformers import pipeline

clf = pipeline("text-classification", model="jjjardev/tagasenti_model")
clf("Ang ganda ng quality ng tela, worth it ang price!")  # -> {'label': 'Positive', 'score': 0.96}
clf("Wala pa ring update ang order ko.")                   # -> {'label': 'Negative', 'score': 0.88}
```

### Option 3 — `huggingface_hub` CLI / Git

```bash
# CLI download to local folder
pip install -U huggingface_hub
huggingface-cli download jjjardev/tagasenti_model --local-dir ./tagasenti_model

# or clone with Git LFS
git lfs install
git clone https://huggingface.co/jjjardev/tagasenti_model
```

> Requires `transformers>=4.40`, `torch>=2.2`, `huggingface_hub>=0.22`. No extra tokenizer install needed.

## Quick inference

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tok = AutoTokenizer.from_pretrained("jjjardev/tagasenti_model")
model = AutoModelForSequenceClassification.from_pretrained("jjjardev/tagasenti_model")
model.eval()

labels = ["Negative", "Neutral", "Positive"]

def predict(texts):
    enc = tok(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        probs = torch.softmax(model(**enc).logits, dim=-1)
    for t, p in zip(texts, probs):
        print(f"{t} -> {labels[p.argmax()]} ({p.max():.2%})")

predict(["Ang ganda ng tela!", "Hindi maganda, sayang pera."])
```

More examples and training script: https://github.com/jjjardev/tagasenti

## Citation

```bibtex
@misc{jessie_james_jarder_2026,
  author = {Jessie James Jarder},
  title  = {tagasenti (Revision 3ebda33)},
  year   = {2026},
  url    = {https://huggingface.co/datasets/jjjardev/tagasenti},
  doi    = {10.57967/hf/9620},
  publisher = {Hugging Face}
}
```
