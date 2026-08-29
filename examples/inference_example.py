"""
Minimal inference example for jjjardev/tagasenti_model.

Run: python examples/inference_example.py
Requires: transformers, torch, accelerate (optional)
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

HF_ID = "jjjardev/tagasenti_model"

# Option A: pipeline (simplest)
clf = pipeline("text-classification", model=HF_ID, tokenizer=HF_ID, truncation=True, max_length=128)
for text in [
    "Ang ganda ng quality ng tela, worth it ang price!",
    "Wala pa ring update ang order ko hanggang ngayon.",
    "Sa aking palagay, hindi naman ito gaanong importante.",
]:
    print(clf(text))

# Option B: manual (logits + probs)
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# import torch
# tok = AutoTokenizer.from_pretrained(HF_ID)
# model = AutoModelForSequenceClassification.from_pretrained(HF_ID)
# labels = ["Negative", "Neutral", "Positive"]
# def predict(texts):
#     enc = tok(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
#     with torch.no_grad():
#         probs = torch.softmax(model(**enc).logits, dim=-1)
#     for t, p in zip(texts, probs):
#         print(t, "->", labels[p.argmax()], f"{p.max():.2%}")
# predict(["okay lang", "pangit eh"])
