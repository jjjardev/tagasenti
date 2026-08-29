"""
validate_dataset.py — integrity checks for TagaSenti.

Checks:
- label domain {0,1,2}
- nulls / empty sentences
- exact duplicate sentences (and conflicting labels)
- post-normalization duplicates (optional)
- length outliers

Usage:
  python scripts/validate_dataset.py                          # HF Hub
  python scripts/validate_dataset.py --csv tagasenti_dataset.csv
  python scripts/validate_dataset.py --csv tagasenti_dataset.csv --fix-dedup tagasenti_dataset.dedup.csv
"""

import argparse
import re
import unicodedata
from collections import Counter

import pandas as pd

try:
    from datasets import Features, Value, load_dataset

    HAS_HF = True
except ImportError:
    HAS_HF = False


def normalize_tagalog_light(text: str) -> str:
    """Minimal normalization used only to detect near-duplicates."""
    if pd.isna(text):
        return ""
    text = unicodedata.normalize("NFKC", str(text))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def validate(df: pd.DataFrame, verbose: bool = True) -> dict:
    report = {}
    n = len(df)
    report["rows"] = n

    # nulls
    report["null_sentence"] = int(df["sentence"].isna().sum())
    report["null_label"] = int(df["label"].isna().sum()) if "label" in df.columns else None
    report["empty_sentence"] = int((df["sentence"].astype(str).str.strip() == "").sum())

    # label domain
    if "label" in df.columns:
        vals = pd.to_numeric(df["label"], errors="coerce")
        report["invalid_labels"] = int((~vals.isin([0, 1, 2])).sum())
        report["label_dist"] = Counter(vals.dropna().astype(int).tolist())
    else:
        report["invalid_labels"] = None
        report["label_dist"] = None

    # exact duplicates
    dup_mask = df.duplicated(subset=["sentence"], keep=False)
    dup_groups = df[dup_mask].groupby("sentence") if dup_mask.any() else []
    n_dup_rows = int(df.duplicated(subset=["sentence"], keep="first").sum())
    n_dup_groups = int(dup_groups.ngroups) if dup_mask.any() else 0
    # conflicting labels
    n_conflict = 0
    conflict_examples = []
    if dup_mask.any() and "label" in df.columns:
        for sent, g in dup_groups:
            if g["label"].nunique() > 1:
                n_conflict += 1
                if len(conflict_examples) < 5:
                    conflict_examples.append((sent[:120], g["label"].tolist()))

    report["exact_dup_rows"] = n_dup_rows
    report["exact_dup_groups"] = n_dup_groups
    report["conflict_groups"] = n_conflict
    report["conflict_examples"] = conflict_examples

    # post-normalization duplicates (light)
    df["_norm"] = df["sentence"].apply(normalize_tagalog_light)
    norm_dup_rows = int(df.duplicated(subset=["_norm"], keep="first").sum())
    report["norm_dup_rows"] = norm_dup_rows
    df = df.drop(columns=["_norm"])

    # length stats
    lens = df["sentence"].astype(str).str.len()
    report["len_min"] = int(lens.min())
    report["len_max"] = int(lens.max())
    report["len_mean"] = float(lens.mean())
    report["len_p99_chars"] = float(lens.quantile(0.99))

    if verbose:
        print(f"Rows: {n}")
        print(f"Null sentence: {report['null_sentence']}, empty: {report['empty_sentence']}, null label: {report['null_label']}")
        print(f"Invalid labels (not in {{0,1,2}}): {report['invalid_labels']}")
        print(f"Label dist: {dict(report['label_dist'] or {})}")
        print(f"Exact duplicates: {n_dup_rows} rows across {n_dup_groups} sentences; {n_conflict} with conflicting labels")
        if conflict_examples:
            print("  Conflict examples (truncated):")
            for sent, labs in conflict_examples:
                print(f"    {repr(sent)} -> {labs}")
        print(f"Near-dup after NFKC+whitespace collapse: {norm_dup_rows} rows")
        print(f"Length chars: min {report['len_min']}, max {report['len_max']}, mean {report['len_mean']:.1f}, p99 {report['len_p99_chars']:.0f}")

    # health check
    ok = report["null_sentence"] == 0 and report["empty_sentence"] == 0 and report["invalid_labels"] == 0
    # duplicates are WARNING not FAIL (training script dedupes by default), but conflicts are actionable
    if n_conflict > 0:
        print(f"\nWARNING: {n_conflict} duplicate sentences have conflicting labels — training will keep first label (see dedup log).")
    if not ok:
        print("\nFAIL: null/empty/invalid labels found.")
    else:
        print("\nPASS: basic integrity ok (duplicates are handled by training dedup).")

    report["ok"] = ok
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default=None, help="Local CSV path; if omitted, loads from HF Hub")
    ap.add_argument("--hf-repo", type=str, default="jjjardev/tagasenti")
    ap.add_argument("--fix-dedup", type=str, default=None, help="Write deduplicated CSV to this path (keep first)")
    args = ap.parse_args()

    if args.csv:
        print(f"Loading CSV: {args.csv}")
        df = pd.read_csv(args.csv)
        # normalize label column name if needed
        if "label" not in df.columns and "labels" in df.columns:
            df = df.rename(columns={"labels": "label"})
    else:
        if not HAS_HF:
            raise SystemExit("datasets not installed; provide --csv or pip install datasets")
        print(f"Loading HF Hub: {args.hf_repo}")
        features = Features({"sentence": Value("string"), "label": Value("string")})
        ds = load_dataset(args.hf_repo, split="train", features=features)
        df = pd.DataFrame(ds)

    # coerce label for validation
    if "label" in df.columns:
        df["label"] = df["label"].astype(str).str.strip().str.replace('"', "", regex=False)
        df["label"] = pd.to_numeric(df["label"], errors="coerce").astype("Int64")

    report = validate(df)

    if args.fix_dedup:
        before = len(df)
        deduped = df.drop_duplicates(subset=["sentence"], keep="first").reset_index(drop=True)
        deduped.to_csv(args.fix_dedup, index=False)
        print(f"\nWrote deduplicated CSV: {args.fix_dedup} ({before} -> {len(deduped)} rows, removed {before - len(deduped)})")


if __name__ == "__main__":
    main()
