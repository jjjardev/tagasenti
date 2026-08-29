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

**TagaSenti — Tagalog/Taglish sentiment analysis** fine-tuned from `xlm-roberta-large` (355M) on [jjjardev/tagasenti](https://huggingface.co/datasets/jjjardev/tagasenti) (35,686 sentences, deduplicated to ~34,945 for training).

- **3 classes:** `Negative (0)` / `Neutral (1)` / `Positive (2)`
- **Test:** 86.6% accuracy, macro-F1 **0.866** (stratified 80/10/10, seed 42, deduplicated before split)
- **Zero-shot HiliSenti (Hiligaynon):** 62.4% acc, F1 0.624
- **License:** Apache 2.0 (weights), CC BY-SA 4.0 (dataset), MIT (code)

## Intended use

Classify Tagalog or Taglish (code-switched) sentences. Works out-of-domain for news/social/e-commerce. Not tuned for other Philippine languages (use zero-shot with caution). No PII scrubbing in training data — apply your own NER if needed.

## Usage

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

hf_id = "jjjardev/tagasenti_model"
tok = AutoTokenizer.from_pretrained(hf_id)
model = AutoModelForSequenceClassification.from_pretrained(hf_id)
model.eval()

def predict(texts):
    enc = tok(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        probs = torch.softmax(model(**enc).logits, dim=-1)
    labels = ["Negative", "Neutral", "Positive"]
    for t, p in zip(texts, probs):
        pred = p.argmax().item()
        print(f"{t} -> {labels[pred]} ({p[pred]:.2%})")

predict(["Ang ganda ng quality ng tela, worth it ang price!", "Wala pa ring update ang order ko."])
# Ang ganda ... -> Positive (96.2%)
# Wala pa ... -> Negative (88.1%)
```

CLI:

```bash
pip install -r requirements.txt  # or pip install -e .
python scripts/inference.py "Ang ganda ng tela!" --hf-id jjjardev/tagasenti_model
python scripts/inference.py --file sentences.txt --json
```

## Training

See `scripts/train/TagaSenti.py` — full pipeline:

- Unicode NFKC, laughter canonicalization (`hahaha`), `word2`/`abot-abot` expansion, slang (`wla→wala`, `dko→di ko`, …), char dedup, whitespace collapse — **casing preserved** (XLM-R cased).
- Deduplication (exact sentence, keep first) before split to avoid leakage (~741 dupes, 113 conflicts).
- Dynamic max length p99 capped at 128.
- Weighted CE + label smoothing 0.10, cosine schedule with warmup (2e-5 → 1e-6), AdamW fused, grad checkpointing, FP16, early stopping (patience 3, metric f1_macro).

```bash
pip install -r requirements.txt
python scripts/train/TagaSenti.py          # outputs to models/tagasenti_model
python scripts/upload_model.py --local models/tagasenti_model --repo jjjardev/tagasenti_model --dry-run
python scripts/upload_model.py --local models/tagasenti_model --repo jjjardev/tagasenti_model
```

## Evaluation

- **Test set:** held-out stratified split (see dataset card). Confusion matrix saved as `final_test_confusion_matrix.png`.
- **Cross-lingual:** `jjjardev/hilisenti-v1` test set zero-shot (no fine-tuning) — 62.4% acc.
- Limitations: single seed (no variance), LLM-labeled subsets (NewsPH, TikTok) not human-validated, translation artifacts in Amazon subset, adversarial bias.

## Citation

```bibtex
@misc{jessie_james_jarder_2026,
  author       = {Jessie James Jarder},
  title        = {tagasenti (Revision 3ebda33)},
  year         = 2026,
  url          = {https://huggingface.co/datasets/jjjardev/tagasenti},
  doi          = {10.57967/hf/9620},
  publisher    = {Hugging Face}
}
```

## Contact

Issues: https://github.com/jjjardev/tagasenti — PRs welcome.
