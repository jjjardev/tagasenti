# TagaSenti

A multi‑domain sentiment analysis dataset and model for **Tagalog and Taglish**.

## Dataset

**[jjjardev/tagasenti](https://huggingface.co/datasets/jjjardev/tagasenti)** on Hugging Face — DOI `10.57967/hf/9620`

- **35,686 sentences** (final v6: `v5.1 34,287 + 1,399 selective`) across 3 base domains + adversarial — deduplicated to ~34,945 for training
- 3 classes: Negative (0), Neutral (1), Positive (2) — perfectly balanced at v1 (33/33/33), 37/33/30 at v6
- ~10k targeted adversarial rows over 4 generations (`v3 3,766 + v4 3,413 + v5 2,417 + v6 1,399`) for sarcasm/negation/idiom robustness
- Tagalog / Taglish code‑switching
- CC BY‑SA 4.0 — see [`paper_notes/COMPLETE_DOCUMENTATION.md`](paper_notes/COMPLETE_DOCUMENTATION.md) (private, 9-day log) distilled here

## Model

**[jjjardev/tagasenti_model](https://huggingface.co/jjjardev/tagasenti_model)** on Hugging Face — **v6 release** (matches dataset v6 35,686)

- Fine‑tuned `xlm-roberta-large` (355M parameters)
- **84.8% test F1 / 84.8% accuracy** — **v6 final checkpoint** (35,686 rows: `v5.1 34,287 + 1,399 selective`, 3 epochs, val 0.863 at step 2400) — HiliSenti zero-shot **F1 0.586**, adversarial 100-set ~83%. Released for dataset-version alignment (Neu hedging bias fixed, 0.9% error). Project-best remains v4 (32,179 rows, 0.866 test / 0.624 Hili, val peak 0.878 unsaved) and v5.1 best saved (0.856) — see [`paper_notes/COMPLETE_DOCUMENTATION.md`](paper_notes/COMPLETE_DOCUMENTATION.md) §5–6.
- Adversarial ceiling: **83%** (idioms 8/17 errors remain across v4/v5.1/v6 — template generation cannot teach non-compositional semantics)
- Apache 2.0 — local staging: [`model/`](model/) — ready to `huggingface-cli upload`
- **INT8 ONNX for on-device/CPU**: [`tagasenti_int8.onnx`](https://huggingface.co/jjjardev/tagasenti_model/resolve/main/tagasenti_int8.onnx) (~537 MB, opset 18, dynamic INT8 `per_channel=True`) hosted in the same model repo — no torch needed, see Option 4 in the [model card](model/README.md). Quantize it yourself with [`scripts/quantize_tagasenti_colab.py`](scripts/quantize_tagasenti_colab.py) (Colab T4 cell, pulls straight from HF Hub).

## Training Script

[`scripts/train/TagaSenti.py`](scripts/train/TagaSenti.py) is a complete pipeline that:

1. **Loads the dataset** from Hugging Face Hub via `load_dataset()` (or local `tagasenti_dataset.csv`)
2. **Deduplicates** exact sentence duplicates (keep first, logs 741 duplicates / 113 label conflicts) before splitting to avoid leakage
3. **Splits** 80/10/10 stratified train/val/test (seed 42) on the deduplicated full set
4. **Normalizes** text — Unicode NFKC, laughter canonicalization, reduplication expansion, slang expansion, character deduplication (casing preserved for XLM-RoBERTa)
5. **Tokenizes** with dynamic max length (p99 percentile, cap 128)
6. **Trains** `xlm-roberta-large` with:
   - Weighted cross-entropy + label smoothing (0.10)
   - Cosine LR schedule with warmup (2e-5 → 1e-6)
   - Gradient checkpointing, FP16, AdamW fused
   - Early stopping (patience 3)
7. **Evaluates** on the held-out test set with confusion matrix
8. **Zero-shot evaluates** on HiliSenti (optional)

### Usage

```bash
pip install -r requirements.txt        # or: pip install -e .
python scripts/train/TagaSenti.py     # trains to models/tagasenti_model
python scripts/validate_dataset.py    # checks duplicates / label conflicts
python scripts/inference.py "Ang ganda ng tela!"  # quick HF model inference
```

### Quantize to ONNX INT8 (Colab)

```bash
# Paste scripts/quantize_tagasenti_colab.py into one Colab (T4) cell and run.
# Pulls jjjardev/tagasenti_model from Hub → exports FP32 ONNX → INT8
# (isolated subprocess, T4-safe) → tagasenti_int8.onnx + tokenizer.
# Then: huggingface-cli upload jjjardev/tagasenti_model tagasenti_int8.onnx
```

### Use the INT8 model (no torch)

```python
import numpy as np, onnxruntime as ort
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("jjjardev/tagasenti_model")
session = ort.InferenceSession("tagasenti_int8.onnx", providers=["CPUExecutionProvider"])
inputs = tokenizer("Ang ganda ng tela!", return_tensors="np", truncation=True, max_length=128)
logits = session.run(["logits"], {
    "input_ids": inputs["input_ids"].astype(np.int64),
    "attention_mask": inputs["attention_mask"].astype(np.int64),
})[0]
print(["Negative", "Neutral", "Positive"][logits.argmax().item()])
```

### Requirements

- Python 3.10+
- GPU recommended (trained on Tesla T4)

## Repository Structure

```
├── README.md
├── LICENSE                     # MIT (code), CC BY-SA 4.0 (data), Apache 2.0 (weights)
├── requirements.txt
├── pyproject.toml
├── scripts/
│   ├── train/TagaSenti.py      # Training pipeline
│   ├── quantize_tagasenti_colab.py  # ONNX INT8 quantization (Colab T4 cell, HF Hub source)
│   ├── validate_dataset.py     # Duplicate / integrity checks
│   ├── inference.py            # HF model inference demo
│   └── upload_model.py         # Stage & upload to jjjardev/tagasenti_model
├── model/README.md             # HF model card (for jjjardev/tagasenti_model)
├── database/README.md          # → HF dataset
├── tests/                      # pytest: normalization, dedup, tokenization
└── examples/
    └── app.py                  # (optional) Flask demo
```

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

## License

- Dataset: CC BY‑SA 4.0
- Model weights: Apache 2.0
- Code: MIT
