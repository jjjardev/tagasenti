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

**[jjjardev/tagasenti-model](https://huggingface.co/jjjardev/tagasenti-model)** on Hugging Face

- Fine‑tuned `xlm-roberta-large` (355M parameters)
- **86.6%** test accuracy, macro‑F1 **0.866**
- Cross-lingual zero-shot on HiliSenti (Hiligaynon): **62.4%** accuracy, F1 **0.624**
- Apache 2.0

## Training Script

[`scripts/train/TagaSenti.py`](scripts/train/TagaSenti.py) is a complete pipeline that:

1. **Loads the dataset** from Hugging Face Hub via `load_dataset()`
2. **Splits** 80/10/10 stratified train/val/test
3. **Normalizes** text — Unicode NFKC, laughter canonicalization, reduplication expansion, slang expansion, character deduplication (casing preserved for XLM-RoBERTa)
4. **Tokenizes** with dynamic max length (p99 percentile)
5. **Trains** `xlm-roberta-large` with:
   - Weighted cross-entropy + label smoothing (0.10)
   - Cosine LR schedule with warmup (2e-5 → 1e-6)
   - Gradient checkpointing, FP16, AdamW fused
   - Early stopping (patience 3)
6. **Evaluates** on the held-out test set with confusion matrix
7. **Zero-shot evaluates** on HiliSenti (optional)

### Usage

```bash
pip install -r requirements.txt
python scripts/train/TagaSenti.py
```

### Requirements

- Python 3.10+
- GPU recommended (trained on Tesla T4)

## Repository Structure

```
├── README.md
├── LICENSE                     # MIT
├── requirements.txt
├── scripts/train/
│   └── TagaSenti.py            # Training pipeline
├── database/README.md          # → HF dataset
└── model/README.md             # → HF model
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
