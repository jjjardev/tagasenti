# TagaSenti Dataset

**Hugging Face:** https://huggingface.co/datasets/jjjardev/tagasenti — DOI `10.57967/hf/9620`

35,686 sentences (deduplicated ~34,945 for training) — Tagalog/Taglish sentiment, 3 classes: Negative (0), Neutral (1), Positive (2).

- 5 domains: e-commerce (FiReCS, SentiTaglish, Amazon translated), news (NewsPH), social (TikTok), plus 9,378 adversarial across 4 generations.
- Single `train` split on Hub; recommended reproducible 80/10/10 stratified split via `scripts/train/TagaSenti.py` (seed 42, dedup before split).
- Companion model: [jjjardev/tagasenti_model](https://huggingface.co/jjjardev/tagasenti_model) (xlm-roberta-large, 86.6% acc, F1 0.866).

Validate locally:

```bash
python scripts/validate_dataset.py --csv tagasenti_dataset.csv
python scripts/validate_dataset.py --csv tagasenti_dataset.csv --fix-dedup tagasenti_dataset.dedup.csv
```

Full card: [`HFREADME.md`](../HFREADME.md) (published to Hub) and [HF dataset page](https://huggingface.co/datasets/jjjardev/tagasenti).
