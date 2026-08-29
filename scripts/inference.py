"""
inference.py — quick inference with the HF model jjjardev/tagasenti_model.

Usage:
  python scripts/inference.py "Ang ganda ng tela, worth it!"
  python scripts/inference.py --file sentences.txt
  echo "hindi maganda" | python scripts/inference.py --stdin
  python scripts/inference.py --hf-id jjjardev/tagasenti_model --batch-size 8 "ok lang" "pangit"

Requires: transformers, torch
"""

import argparse
import sys
from typing import List

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


LABEL_NAMES = ["Negative", "Neutral", "Positive"]


def load_model(hf_id: str):
    print(f"Loading {hf_id} ...", file=sys.stderr)
    tok = AutoTokenizer.from_pretrained(hf_id)
    model = AutoModelForSequenceClassification.from_pretrained(hf_id)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return tok, model, device


def predict(texts: List[str], tok, model, device: str, max_length: int = 128):
    enc = tok(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        preds = probs.argmax(axis=-1)
    results = []
    for text, pred, prob in zip(texts, preds, probs):
        results.append(
            {
                "sentence": text,
                "label": int(pred),
                "sentiment": LABEL_NAMES[int(pred)],
                "confidence": float(prob[pred]),
                "probs": {LABEL_NAMES[i]: float(prob[i]) for i in range(3)},
            }
        )
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sentences", nargs="*", help="Sentences to classify")
    ap.add_argument("--hf-id", default="jjjardev/tagasenti_model", help="HF model repo")
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--file", type=str, default=None, help="File with one sentence per line")
    ap.add_argument("--stdin", action="store_true", help="Read sentences from stdin")
    ap.add_argument("--json", action="store_true", help="Output JSON lines")
    args = ap.parse_args()

    texts: List[str] = list(args.sentences)
    if args.file:
        with open(args.file, encoding="utf-8") as f:
            texts += [line.strip() for line in f if line.strip()]
    if args.stdin:
        texts += [line.strip() for line in sys.stdin if line.strip()]

    if not texts:
        ap.error("No sentences provided. Pass as args, --file, or --stdin.")

    tok, model, device = load_model(args.hf_id)

    # batch
    all_results = []
    for i in range(0, len(texts), args.batch_size):
        batch = texts[i : i + args.batch_size]
        all_results.extend(predict(batch, tok, model, device, max_length=args.max_length))

    if args.json:
        import json

        for r in all_results:
            print(json.dumps(r, ensure_ascii=False))
    else:
        for r in all_results:
            print(f"[{r['sentiment']:<8} {r['confidence']:.2%}] {r['sentence']}")
            # uncomment to debug probs:
            # print(f"  probs: {r['probs']}")


if __name__ == "__main__":
    main()
