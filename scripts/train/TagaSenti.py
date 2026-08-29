"""
TagaSenti — Tagalog Sentiment Analysis Pipeline
Fine-tuning XLM-RoBERTa-Large for 3-class Filipino sentiment classification.

All design decisions are documented inline. The normalisation section in
particular explains what is applied, what was removed, and why.
"""

import os
import re
import unicodedata
import multiprocessing

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns

from datasets import Dataset, Features, Value, load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
# ============================================================
# 1. CONFIGURATION
# ============================================================
CFG = {
    "hf_repo":        "jjjardev/tagasenti",
    "hf_model_id":    "jjjardev/tagasenti_model",  # HF model repo (already exists)
    "output_dir":     "models/tagasenti_model",
    "model_ckpt":     "xlm-roberta-large",
    "test_size":      0.20,
    "val_from_held":  0.50,
    "seed":           42,
    "deduplicate":    True,   # drop exact sentence duplicates (keep first); logs conflicts
    "max_len_percentile": 99,
    "max_len_cap":         128,
    "max_len_sample":      2_000,
    "learning_rate":    2e-5,
    "train_batch":      16,
    "eval_batch":       32,
    "grad_accum":       2,
    "num_epochs":       3,
    "weight_decay":     0.05,
    "warmup_steps":     312,
    "min_lr":           1e-6,
    "eval_steps":       200,
    "early_stop":       3,
    "label_smoothing":  0.10,
    "class_weights":    [0.897, 1.021, 1.104],
    "eval_hilisenti":  True,
    "hilisenti_path":  "jjjardev/hilisenti-v1",
}

LABEL_NAMES = ["Negative", "Neutral", "Positive"]
ID2LABEL    = {i: n for i, n in enumerate(LABEL_NAMES)}
LABEL2ID    = {n: i for i, n in enumerate(LABEL_NAMES)}

os.makedirs(CFG["output_dir"], exist_ok=True)

# ============================================================
# 2. DATA LOADING & CLEANING
# ============================================================
def clean_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce the label column to int ∈ {0, 1, 2}; drop malformed rows."""
    df = df.copy()
    df["label"] = (
        df["label"].astype(str).str.strip().str.replace('"', "", regex=False)
    )
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    return df[df["label"].isin([0, 1, 2])].reset_index(drop=True)


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop exact duplicate sentences (keep first occurrence).
    - Logs how many were removed and how many had conflicting labels.
    - Does NOT normalize text before dedup (exact match only); normalization
      happens later and may collapse additional near-duplicates — that's
      intentional and logged via the training script's histogram probe.
    - If a `domain` or `source` column exists, adversarial-only-train filtering
      should be done *before* this call (kept for backward compat).
    """
    if not CFG.get("deduplicate", True):
        return df
    dup_mask = df.duplicated(subset=["sentence"], keep=False)
    if not dup_mask.any():
        print("Deduplication: no exact duplicates found.")
        return df
    dup_groups = df[dup_mask].groupby("sentence")
    conflict_groups = sum(1 for _, g in dup_groups if g["label"].nunique() > 1)
    exact_dup_rows = int(df.duplicated(subset=["sentence"], keep="first").sum())
    print(
        f"Deduplication: {exact_dup_rows} exact duplicate rows found "
        f"across {dup_groups.ngroups} unique sentences "
        f"({conflict_groups} with conflicting labels — keeping first label)."
    )
    # Keep first occurrence per sentence
    deduped = df.drop_duplicates(subset=["sentence"], keep="first").reset_index(drop=True)
    print(f"After dedup: {len(deduped)} rows (removed {len(df) - len(deduped)})")
    print(f"Label distribution (post-dedup):\n{deduped['label'].value_counts().sort_index()}\n")
    return deduped


print(f"Loading dataset from HuggingFace Hub: {CFG['hf_repo']}")
features = Features({"sentence": Value("string"), "label": Value("string")})
ds = load_dataset(CFG["hf_repo"], split="train", features=features)
df = clean_labels(pd.DataFrame(ds))
print(f"Loaded {len(df)} rows")
print(f"Label distribution:\n{df['label'].value_counts().sort_index()}\n")

# Deduplicate before splitting to avoid train/val/test leakage
df = deduplicate(df)

# Stratified 80 / 10 / 10 split on the *full* dataset (including adversarial).
# NOTE: Earlier dataset cards described adversarial as "train-only" — that applied
# to the offline v1-v4 curated splits stored locally. The HF Hub distribution
# `jjjardev/tagasenti` is a single `train` split with all domains mixed; this
# script therefore does a reproducible stratified random split for training.
# To reproduce a strict adversarial-only-train setup, filter by a `domain`
# column before this split (if present) or use the frozen `tagasenti_dataset.csv`
# splits from the GitHub release.
train_df, temp_df = train_test_split(
    df, test_size=CFG["test_size"], stratify=df["label"], random_state=CFG["seed"]
)
val_df, test_df = train_test_split(
    temp_df, test_size=CFG["val_from_held"], stratify=temp_df["label"], random_state=CFG["seed"]
)
train_df = train_df.reset_index(drop=True)
val_df   = val_df.reset_index(drop=True)
test_df  = test_df.reset_index(drop=True)

print(f"Train : {len(train_df)} | Val : {len(val_df)} | Test : {len(test_df)}")
print(f"Label distribution (train):\n{train_df['label'].value_counts().sort_index()}\n")

# ============================================================
# 3. TEXT NORMALISATION
# ============================================================
# ── Casing policy — WHY WE DO NOT LOWERCASE ──────────────────────────────────
# XLM-RoBERTa-large is a CASED SentencePiece model. Its vocabulary contains
# separate entries for mixed-case forms (e.g. "Ganda", "GANDA", "ganda").
# Lowercasing collapses those entries, erasing the emphasis signal that is
# extremely common in Filipino social-media sentiment ("AYAW KO NA" ≠ "ayaw
# ko na" in terms of emotional intensity). Lowercasing was present in the
# original pipeline and has been removed here.
#
# ── What we DO normalise ─────────────────────────────────────────
# 1. Unicode NFKC  — collapses ligatures, homoglyphs, half-width chars
# 2. Laughter strings — "hahahaha…" → "hahaha" (IGNORECASE so HAHAHA works)
# 3. Filipino "word2" notation — "abot2" → "abot abot"
# 4. Hyphenated reduplication — "abot-abot" → "abot abot"
# 5. Selective slang expansion — high-confidence, low-ambiguity only (see below)
# 6. Char deduplication — ≥ 3 consecutive identical chars → 2
# 7. Whitespace collapse
#
# ── Abbreviations REMOVED from the original ────────────────────────────
# • r'\bpro\b'  → "pero"  REMOVED: "pro" is an English loanword in Filipino
#                          ("pro player", "pros and cons"). Blind replacement
#                          would corrupt unambiguously English uses.
# • r'\bmn\b'   → "man"   REMOVED: a 2-char token with a very high collision
#                          rate against English text.
# • r'\bala\b'  → "wala"  REMOVED: "ala" appears as a mid-word syllable in
#                          many Tagalog and Spanish-derived words and collides
#                          with the interjection "ala" (expression of surprise).
#
# ── Pre-compiled regex patterns ──────────────────────────────
# The function is called for every row (~25 k texts). Compiling each pattern
# once at module load time avoids repeating the compile step per call.

_SLANG = {
    r"\bwla\b":     "wala",
    r"\blng\b":     "lang",
    r"\bnlng\b":    "nalang",
    r"\bna lng\b":  "nalang",
    r"\bdko\b":     "di ko",
    r"\bgnito\b":   "ganito",
    r"\bgnyan\b":   "ganyan",
    r"\bgnun\b":    "ganun",
    r"\bsakin\b":   "sa akin",
    r"\bpru\b":     "pero",
    r"\bkc\b":      "kasi",
    r"\bksi\b":     "kasi",
    r"\bky\b":      "kay",
    r"\bnman\b":    "naman",
    r"\bnmn\b":     "naman",
    r"\bmeron\b":   "mayroon",
    r"\bsya\b":     "siya",
    r"\bxa\b":      "siya",
    r"\bxia\b":     "siya",
    r"\bnya\b":     "niya",
    r"\bikw\b":     "ikaw",
    r"\bmng\b":     "mang",
    r"\bdpt\b":     "dapat",
    r"\bdba\b":     "di ba",
    r"\bdiba\b":    "di ba",
}

_COMPILED_SLANG = [(re.compile(p), r) for p, r in _SLANG.items()]


def normalize_tagalog(text) -> str:
    """
    Light normalisation for Filipino social-media text.
    Casing is intentionally preserved — see module docstring.
    """
    if pd.isna(text):
        return ""

    text = unicodedata.normalize("NFKC", str(text))

    # Canonical laughter forms — IGNORECASE catches HAHAHA, Hahaha, etc.
    text = re.sub(r"\b(?:ha){2,}h*\b", "hahaha", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:he){2,}h*\b", "hehehe", text, flags=re.IGNORECASE)

    # Filipino "word2" reduplication (≥3-char words to avoid "ko2", "a2")
    text = re.sub(r"\b([A-Za-z]{3,})2\b", r"\1 \1", text)

    # Hyphenated reduplication: "abot-abot" → "abot abot"
    text = re.sub(r"\b(\w+)-\1\b", r"\1 \1", text)

    # Slang / abbreviation expansion
    for pattern, replacement in _COMPILED_SLANG:
        text = pattern.sub(replacement, text)

    # Char deduplication: "graaaabe" → "graabe" (keeps ≤ 2 consecutive)
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)

    # Whitespace
    return re.sub(r"\s+", " ", text).strip()


print("Normalising text …")
for split_df in (train_df, val_df, test_df):
    split_df["sentence"] = split_df["sentence"].apply(normalize_tagalog)

# ============================================================
# 4. CLASS WEIGHTS
# ============================================================
# v2 uses manual weights [1.2, 1.0, 0.8] to penalise false positives
# (the positive-default bias found in benchmark eval). Balanced weights
# from compute_class_weight produce near-uniform values when the training
# set is already perfectly balanced (33/33/33) — so they had no effect in v1.
class_weights_tensor = torch.tensor(CFG["class_weights"], dtype=torch.float)
print(f"Class weights: {class_weights_tensor}  (Neg={CFG['class_weights'][0]}, Neu={CFG['class_weights'][1]}, Pos={CFG['class_weights'][2]})")

# ============================================================
# 5. TOKENISER & OPTIMAL MAX LENGTH
# ============================================================
tokenizer = AutoTokenizer.from_pretrained(CFG["model_ckpt"])


def compute_optimal_length(texts: list, tok, percentile: int, cap: int) -> int:
    """Return the p-th percentile token length across `texts`, capped at `cap`."""
    lengths = [
        len(tok.encode(t, add_special_tokens=True, truncation=False))
        for t in texts
    ]
    return min(int(np.percentile(lengths, percentile)), cap)


# Guard against datasets smaller than the sample size
sample_n = min(CFG["max_len_sample"], len(train_df))
sample_texts = train_df["sentence"].sample(sample_n, random_state=CFG["seed"]).tolist()

# Histogram probe — print distribution before committing to a percentile
probe_lengths = [len(tokenizer.encode(t, add_special_tokens=True, truncation=False)) for t in sample_texts]
print(
    f"Token-length distribution (n={sample_n}):  "
    f"p50={np.percentile(probe_lengths, 50):.0f}  "
    f"p90={np.percentile(probe_lengths, 90):.0f}  "
    f"p95={np.percentile(probe_lengths, 95):.0f}  "
    f"p99={np.percentile(probe_lengths, 99):.0f}  "
    f"max={max(probe_lengths)}"
)

optimal_max_length = compute_optimal_length(
    sample_texts,
    tokenizer,
    percentile=CFG["max_len_percentile"],
    cap=CFG["max_len_cap"],
)
print(f"Optimal max length (p{CFG['max_len_percentile']}, ≤{CFG['max_len_cap']}): {optimal_max_length}\n")

# ============================================================
# 6. TOKENISATION
# ============================================================
# NOTE: `add_prefix_space` is a BPE convention (RoBERTa/GPT-2). XLM-RoBERTa
# uses SentencePiece and does not use that parameter — it was present in the
# original pipeline and has been removed here.

def tokenize_fn(batch):
    return tokenizer(
        batch["sentence"],
        padding=False,       # DataCollatorWithPadding handles dynamic padding
        truncation=True,
        max_length=optimal_max_length,
    )


def build_dataset(split_df: pd.DataFrame) -> Dataset:
    """Convert a dataframe slice to a HuggingFace Dataset with a clean index."""
    return Dataset.from_pandas(
        split_df[["sentence", "label"]].reset_index(drop=True)
    )


# Cap at 4 workers — using all cores can cause instability in Colab's
# shared-memory environment when the dataset is large.
num_proc = min(multiprocessing.cpu_count(), 4)
print(f"Tokenising with {num_proc} workers …")

_map_kw = dict(batched=True, remove_columns=["sentence"], num_proc=num_proc)
tokenized_train = build_dataset(train_df).map(tokenize_fn, **_map_kw)
tokenized_val   = build_dataset(val_df).map(tokenize_fn, **_map_kw)
tokenized_test  = build_dataset(test_df).map(tokenize_fn, **_map_kw)

# ============================================================
# 7. MODEL
# ============================================================
model = AutoModelForSequenceClassification.from_pretrained(
    CFG["model_ckpt"],
    num_labels=3,
    id2label=ID2LABEL,
    label2id=LABEL2ID,
)

# ============================================================
# 8. METRICS
# ============================================================
# Shared mutable cache: compute_metrics writes predictions here so the
# ConfusionMatrixCallback can read them without a second forward pass.
_last_eval: dict = {}


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    # Cache for the confusion-matrix callback (avoids a redundant predict call)
    _last_eval["preds"]  = preds
    _last_eval["labels"] = labels

    f1_cls   = f1_score(labels, preds, average=None, labels=[0, 1, 2], zero_division=0)
    prec_cls = precision_score(labels, preds, average=None, labels=[0, 1, 2], zero_division=0)
    rec_cls  = recall_score(labels, preds, average=None, labels=[0, 1, 2], zero_division=0)

    return {
        "accuracy":           accuracy_score(labels, preds),
        "balanced_accuracy":  balanced_accuracy_score(labels, preds),
        "f1_macro":           f1_score(labels, preds, average="macro", zero_division=0),
        "f1_negative":        f1_cls[0],
        "f1_neutral":         f1_cls[1],
        "f1_positive":        f1_cls[2],
        "precision_negative": prec_cls[0],
        "precision_neutral":  prec_cls[1],
        "precision_positive": prec_cls[2],
        "recall_negative":    rec_cls[0],
        "recall_neutral":     rec_cls[1],
        "recall_positive":    rec_cls[2],
    }


# ============================================================
# 9. CONFUSION MATRIX CALLBACK
# ============================================================
class ConfusionMatrixCallback(TrainerCallback):
    """
    Saves a confusion matrix PNG and prints a classification report after
    every evaluation step.

    Reads from `_last_eval` (populated inside `compute_metrics`) instead of
    calling trainer.predict() again — this avoids a full second forward pass
    over the validation set on every eval checkpoint.

    Background: HuggingFace's on_evaluate hook does NOT receive `eval_dataset`
    in **kwargs (the original pipeline checked for it and silently returned
    early every time, making the callback a no-op). The _last_eval dict is
    the clean fix.
    """

    def on_evaluate(self, args, state, control, **kwargs):
        if not _last_eval:
            return  # compute_metrics hasn't run yet (shouldn't happen)

        preds  = _last_eval["preds"]
        labels = _last_eval["labels"]
        step   = state.global_step

        self._save_cm(labels, preds, step, args.output_dir)
        print(f"\n--- Classification Report (step {step}) ---")
        print(classification_report(labels, preds, target_names=LABEL_NAMES, digits=4))
        print(f"Prediction distribution:\n{pd.Series(preds).value_counts().sort_index()}\n")

    @staticmethod
    def _save_cm(labels, preds, step: int, output_dir: str):
        cm = confusion_matrix(labels, preds)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES, ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"Confusion Matrix — step {step}")

        cm_dir = os.path.join(output_dir, "confusion_matrices")
        os.makedirs(cm_dir, exist_ok=True)
        save_path = os.path.join(cm_dir, f"cm_step_{step}.png")
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        print(f"  CM saved → {save_path}")


# ============================================================
# 10. CUSTOM TRAINER — weighted + smoothed cross-entropy
# ============================================================
class CustomTrainer(Trainer):
    """
    Injects per-class weights into the cross-entropy loss.

    IMPORTANT: HuggingFace Trainer applies `label_smoothing_factor` only
    through its own default compute_loss implementation. Because we override
    that method, we must pass the smoothing value to nn.CrossEntropyLoss
    explicitly — otherwise the setting in TrainingArguments silently has no
    effect. The original pipeline had this bug; it is fixed here.
    """

    def __init__(self, class_weights: torch.Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits  = outputs.get("logits")

        loss_fct = nn.CrossEntropyLoss(
            weight=self.class_weights.to(logits.device),
            label_smoothing=self.args.label_smoothing_factor,  # honour TrainingArguments
        )
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


# ============================================================
# 11. TRAINING ARGUMENTS
# ============================================================
training_args = TrainingArguments(
    output_dir=CFG["output_dir"],
    # ── Eval & checkpointing ────────────────────────────────
    eval_strategy="steps",
    eval_steps=CFG["eval_steps"],
    save_strategy="steps",
    save_steps=CFG["eval_steps"],
    save_total_limit=0,   # unlimited — prevents best checkpoint deletion
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True,
    # ── Optimisation ───────────────────────────────────────
    learning_rate=CFG["learning_rate"],
    per_device_train_batch_size=CFG["train_batch"],
    per_device_eval_batch_size=CFG["eval_batch"],
    gradient_accumulation_steps=CFG["grad_accum"],
    num_train_epochs=CFG["num_epochs"],
    weight_decay=CFG["weight_decay"],
    warmup_steps=CFG["warmup_steps"],
    lr_scheduler_type="cosine_with_min_lr",
    lr_scheduler_kwargs={"min_lr": CFG["min_lr"]},
    optim="adamw_torch_fused",
    gradient_checkpointing=True,
    # ── Precision & regularisation ───────────────────────────────
    fp16=torch.cuda.is_available(),
    label_smoothing_factor=CFG["label_smoothing"],  # applied in CustomTrainer.compute_loss
    # ── Logging ───────────────────────────────────────────
    logging_strategy="steps",
    logging_steps=50,
    report_to="tensorboard",
    seed=CFG["seed"],
)

# ============================================================
# 12. BUILD TRAINER & TRAIN
# ============================================================
trainer = CustomTrainer(
    class_weights=class_weights_tensor,
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics,
    callbacks=[
        EarlyStoppingCallback(early_stopping_patience=CFG["early_stop"]),
        ConfusionMatrixCallback(),
    ],
)

print("Starting training …\n")
trainer.train()

# ============================================================
# 13. TEST SET EVALUATION
# ============================================================
# We use trainer.predict() rather than trainer.evaluate() so that the
# ConfusionMatrixCallback is NOT re-triggered with test-set predictions
# (the callback is designed for validation-time monitoring, not final eval).
# predict() still calls compute_metrics and returns the metrics in .metrics.
print("\nEvaluating on test set …")
test_output   = trainer.predict(tokenized_test, metric_key_prefix="test")
test_labels   = test_output.label_ids
test_preds_np = np.argmax(test_output.predictions, axis=-1)
print(f"Test metrics: {test_output.metrics}\n")

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(
    confusion_matrix(test_labels, test_preds_np),
    annot=True, fmt="d", cmap="Greens",
    xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES, ax=ax,
)
ax.set_title("Final Test Set Confusion Matrix — TagaSenti")
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
fig.savefig(
    os.path.join(CFG["output_dir"], "final_test_confusion_matrix.png"),
    bbox_inches="tight",
)
plt.show()
plt.close(fig)

print("\nFinal test classification report:")
print(classification_report(test_labels, test_preds_np, target_names=LABEL_NAMES))

# ============================================================
# 14. SAVE
# ============================================================
trainer.save_model(CFG["output_dir"])
tokenizer.save_pretrained(CFG["output_dir"])
print(f"\nModel + tokeniser saved → {CFG['output_dir']}")

# ============================================================
# 15. OPTIONAL: ZERO-SHOT TRANSFER EVAL ON HILISENTI
# ============================================================
if CFG["eval_hilisenti"]:
    print("\n" + "=" * 60)
    print("ZERO-SHOT EVALUATION — HILISENTI TEST SET")
    print("=" * 60)

    hili_path = CFG["hilisenti_path"]
    # Handle HuggingFace dataset loading if path is a repo ID
    if not os.path.exists(hili_path) and "/" in hili_path:
        print(f"Loading HuggingFace dataset: {hili_path}")
        # Load label as string — HF CSV datasets often have quoted labels
        # ("0" instead of 0) that PyArrow can't cast to int64.
        features = Features({"sentence": Value("string"), "label": Value("string")})
        hili_raw = load_dataset(hili_path, split="test", features=features)
        hili_df = clean_labels(pd.DataFrame(hili_raw))
    elif not os.path.exists(hili_path):
        print(f"WARNING: file not found — {hili_path}\nSkipping.")
        hili_df = None
    else:
        hili_df = clean_labels(pd.read_csv(hili_path))

    if hili_df is not None:
        # normalize_tagalog is applied to Hiligaynon text here.
        hili_df["sentence"] = hili_df["sentence"].apply(normalize_tagalog)
        print(f"HiliSenti samples: {len(hili_df)}")
        print(f"Label distribution:\n{hili_df['label'].value_counts().sort_index()}\n")

        tokenized_hili = build_dataset(hili_df).map(tokenize_fn, **_map_kw)

        hili_output   = trainer.predict(tokenized_hili, metric_key_prefix="hilisenti")
        hili_labels   = hili_output.label_ids
        hili_preds_np = np.argmax(hili_output.predictions, axis=-1)
        print(f"HiliSenti zero-shot metrics: {hili_output.metrics}\n")

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            confusion_matrix(hili_labels, hili_preds_np),
            annot=True, fmt="d", cmap="Oranges",
            xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES, ax=ax,
        )
        ax.set_title("Zero-Shot: TagaSenti → HiliSenti Test Set")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        fig.savefig(
            os.path.join(CFG["output_dir"], "hilisenti_zero_shot_cm.png"),
            bbox_inches="tight",
        )
        plt.show()
        plt.close(fig)

        print("\nHiliSenti zero-shot classification report:")
        print(classification_report(hili_labels, hili_preds_np, target_names=LABEL_NAMES))

    print("\n" + "=" * 60)
    print("ZERO-SHOT EVALUATION COMPLETE")
    print("=" * 60)
