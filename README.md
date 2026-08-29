# TagaSenti

A multi‑domain sentiment analysis dataset and model for **Tagalog and Taglish**.

## Dataset

**[jjjardev/tagasenti](https://huggingface.co/datasets/jjjardev/tagasenti)** on Hugging Face — DOI `10.57967/hf/9620`

- 35,686 sentences across 5 domains (e‑commerce, news, social media, adversarial, synthetic)
- 3 classes: Negative (0), Neutral (1), Positive (2)
- 9,378 targeted adversarial rows over 4 generations for robust edge-case handling
- Tagalog / Taglish code‑switching
- CC BY‑SA 4.0

## Model

**[jjjardev/tagasenti_model](https://huggingface.co/jjjardev/tagasenti_model)** on Hugging Face

- Fine‑tuned `xlm-roberta-large` (355M parameters)
- **86.6%** test accuracy, macro‑F1 **0.866** (deduplicated, stratified 80/10/10, seed 42)
- Cross-lingual zero-shot on HiliSenti (Hiligaynon): **62.4%** accuracy, F1 **0.624**
- Apache 2.0
- Local staging: [`model/`](model/) — ready to `huggingface-cli upload`

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
