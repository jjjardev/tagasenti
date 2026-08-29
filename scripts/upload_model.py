"""
upload_model.py — stage and upload a local checkpoint to Hugging Face Hub.

Assumes you trained with scripts/train/TagaSenti.py and have a folder like
  models/tagasenti_model/
containing:
  config.json, tokenizer.json, tokenizer_config.json, model.safetensors (or pytorch_model.bin)

It will:
  1. Verify required files exist
  2. Patch config.json with correct id2label/label2id if needed
  3. Ensure model/README.md exists (copies from repo model/README.md if needed)
  4. Upload via huggingface_hub (requires `hf auth login` or HF_TOKEN env)

Usage:
  python scripts/upload_model.py --local models/tagasenti_model --repo jjjardev/tagasenti_model
  python scripts/upload_model.py --local models/tagasenti_model --repo jjjardev/tagasenti_model --create-pr
  python scripts/upload_model.py --local models/tagasenti_model --repo jjjardev/tagasenti_model --dry-run

For first-time model creation (repo already exists per user context), this does a direct upload to main.
"""

import argparse
import json
import os
import sys
from pathlib import Path

REQUIRED = ["config.json", "tokenizer.json", "tokenizer_config.json"]
WEIGHTS_CANDIDATES = ["model.safetensors", "pytorch_model.bin", "model.safetensors.index.json"]


def check_local(local: Path):
    missing = [f for f in REQUIRED if not (local / f).exists()]
    has_weights = any((local / w).exists() for w in WEIGHTS_CANDIDATES)
    # also allow sharded safetensors
    if not has_weights:
        has_weights = any(local.glob("model-*.safetensors")) or any(local.glob("pytorch_model-*.bin"))
    return missing, has_weights


def patch_config(local: Path):
    cfg_path = local / "config.json"
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    # ensure labels are correct for TagaSenti
    expected_id2label = {"0": "Negative", "1": "Neutral", "2": "Positive"}
    expected_label2id = {"Negative": 0, "Neutral": 1, "Positive": 2}
    changed = False
    if cfg.get("id2label") != expected_id2label:
        cfg["id2label"] = expected_id2label
        changed = True
    if cfg.get("label2id") != expected_label2id:
        cfg["label2id"] = expected_label2id
        changed = True
    if cfg.get("num_labels") != 3:
        cfg["num_labels"] = 3
        changed = True
    if changed:
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
            f.write("\n")
        print(f"Patched {cfg_path} (id2label/label2id/num_labels)")
    else:
        print(f"Config OK: {cfg_path}")


def ensure_model_card(local: Path, repo: str):
    readme = local / "README.md"
    if readme.exists():
        print(f"Model card exists: {readme}")
        return
    # try to copy from repo root model/README.md
    candidates = [Path("model/README.md"), Path("model/README.md")]
    src = None
    for c in candidates:
        if c.exists():
            src = c
            break
    if src:
        import shutil

        shutil.copy(src, readme)
        print(f"Copied model card {src} -> {readme}")
    else:
        # minimal card
        readme.write_text(
            f"---\nlanguage: [tl, en]\nlicense: apache-2.0\ntags: [text-classification, sentiment-analysis, tagalog, taglish]\nbase_model: xlm-roberta-large\n---\n\n# {repo}\n\nTagaSenti sentiment model — see https://github.com/jjjardev/tagasenti\n",
            encoding="utf-8",
        )
        print(f"Created minimal card: {readme}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", type=str, default="models/tagasenti_model", help="Local checkpoint dir")
    ap.add_argument("--repo", type=str, default="jjjardev/tagasenti_model", help="HF repo id (already exists)")
    ap.add_argument("--dry-run", action="store_true", help="Validate only, don't upload")
    ap.add_argument("--create-pr", action="store_true", help="Upload as PR instead of direct to main")
    ap.add_argument("--commit-message", type=str, default="Add/Update tagasenti_model weights + card")
    ap.add_argument("--private", action="store_true", help="If repo creation needed, make private (ignored if exists)")
    args = ap.parse_args()

    local = Path(args.local)
    if not local.exists():
        print(f"ERROR: local path not found: {local}", file=sys.stderr)
        sys.exit(1)

    missing, has_weights = check_local(local)
    if missing:
        print(f"ERROR: missing required files in {local}: {missing}", file=sys.stderr)
        print(f"  Contents: {sorted(os.listdir(local))}", file=sys.stderr)
        sys.exit(1)
    if not has_weights:
        print(f"ERROR: no weights found in {local} (expected one of {WEIGHTS_CANDIDATES} or sharded)", file=sys.stderr)
        sys.exit(1)

    patch_config(local)
    ensure_model_card(local, args.repo)

    print(f"\nStaging OK: {local} -> {args.repo}")
    print(f"  Files: {sorted(os.listdir(local))}")

    if args.dry_run:
        print("\nDry-run: not uploading. Run without --dry-run to push.")
        print(f"  Next: huggingface-cli upload {args.repo} {local} --repo-type model")
        return

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("huggingface-hub not installed. Install with: pip install huggingface-hub", file=sys.stderr)
        sys.exit(1)

    api = HfApi()
    # verify auth
    try:
        user = api.whoami()
        print(f"Authenticated as: {user.get('name') or user.get('fullname') or 'unknown'}")
    except Exception as e:
        print(f"ERROR: not authenticated. Run `hf auth login` or set HF_TOKEN. ({e})", file=sys.stderr)
        sys.exit(1)

    # upload_folder is simplest and respects .gitattributes
    print(f"\nUploading {local} -> {args.repo} (PR={args.create_pr}) ...")
    url = api.upload_folder(
        folder_path=str(local),
        repo_id=args.repo,
        repo_type="model",
        commit_message=args.commit_message,
        create_pr=args.create_pr,
    )
    print(f"Done: {url}")


if __name__ == "__main__":
    main()
