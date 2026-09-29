# ============================================================
# TAGASENTI — Use / Take Out the INT8 ONNX in Colab (no torch)
# ============================================================
# Purpose: load the quantized model produced by
#   01_quantize_tagasenti_colab.py and run inference — i.e. "take out"
#   the INT8 file for verification, download, or app wiring.
# No PyTorch, no quantization here — just onnxruntime (CPU) +
# transformers tokenizer. Safe to run on a fresh Colab runtime.
#
# Usage: paste into ONE Colab cell and run.
# It will: mount Drive → load tagasenti_int8.onnx → run 4 Tagalog
#   spot-checks (Pos/Neg/Neu + sarcasm) → copy to /content for
#   easy download → print file sizes.

!pip install -q "protobuf>=5.26,<6" onnxruntime transformers
# NOTE: protobuf pinned <6 — same reason as 01 script (unpinned onnx
# pulls protobuf 7.x and breaks Colab deps).

import os, shutil
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
from google.colab import drive

try:
    drive.mount("/content/drive", force_remount=False)
except Exception as e:
    print(f"Drive mount failed ({e}) → checking /content fallback.")

# ── Paths (must match 01_quantize script) ────────────────────
INT8_NAME = "tagasenti_int8.onnx"
HF_MODEL_ID = "jjjardev/tagasenti_model"   # tokenizer fallback if Drive copy missing
candidates = [
    "/content/drive/MyDrive/TagaSenti/model_onnx/tagasenti_int8.onnx",
    "/content/tagasenti_onnx/tagasenti_int8.onnx",
    f"/content/{INT8_NAME}",
]
int8_path = next((p for p in candidates if os.path.exists(p)), candidates[0])
ONNX_OUTPUT_DIR = os.path.dirname(int8_path)

assert os.path.exists(int8_path), (
    f"INT8 model not found at {int8_path}\n"
    "Run 01_quantize_tagasenti_colab.py first, or upload tagasenti_int8.onnx there."
)

# ── Tokenizer (Drive copy first, Hub fallback) ───────────────
try:
    tokenizer = AutoTokenizer.from_pretrained(ONNX_OUTPUT_DIR, local_files_only=True)
    print("Tokenizer loaded from Drive output dir.")
except Exception:
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)
    print(f"Drive tokenizer not found → loaded from Hub: {HF_MODEL_ID}")

# ── Load INT8 session (CPU only — quantized model runs on CPU) ─
sess = ort.InferenceSession(int8_path, providers=["CPUExecutionProvider"])
print(f"Loaded: {INT8_NAME}")
print(f"  Inputs : {[(i.name, i.shape, i.type) for i in sess.get_inputs()]}")
print(f"  Outputs: {[(o.name, o.shape, o.type) for o in sess.get_outputs()]}")

LABELS = ["Negative", "Neutral", "Positive"]

def predict(texts, max_length=128):
    """Run INT8 ONNX inference. Returns (labels, probs)."""
    if isinstance(texts, str):
        texts = [texts]
    enc = tokenizer(texts, padding=True, truncation=True,
                    max_length=max_length, return_tensors="np")
    ort_inputs = {
        "input_ids": enc["input_ids"].astype(np.int64),
        "attention_mask": enc["attention_mask"].astype(np.int64),
    }
    logits = sess.run(["logits"], ort_inputs)[0]
    # softmax
    e = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probs = e / e.sum(axis=-1, keepdims=True)
    preds = probs.argmax(axis=-1)
    return [LABELS[p] for p in preds], probs

# ── Spot-check (examples from jjjardev/tagasenti_model card) ──
demo = [
    "Ang ganda ng quality ng tela, worth it ang price!",          # Positive
    "Wala pa ring update ang order ko hanggang ngayon.",          # Negative
    "Sa aking palagay, hindi naman ito gaanong importante.",      # Neutral
    "Ang bait mo naman, pinagbigyan mo ako sa wakas.",            # Negative (sarcasm, adversarial)
]
labels, probs = predict(demo)
print("\n--- INT8 spot-check ---")
for sent, lab, pr in zip(demo, labels, probs):
    print(f"{lab:>8} ({pr.max():.1%}) : {sent}")

# ── Take out: copy to /content for download / adb push ───────
local_copy = f"/content/{INT8_NAME}"
shutil.copyfile(int8_path, local_copy)
drive_mb = os.path.getsize(int8_path) / 1e6
print(f"\n✅ INT8 ready: {drive_mb:.1f} MB")
print(f"   Drive : {int8_path}")
print(f"   Local : {local_copy}  ← Colab file browser → Download, or adb push to phone")
print("   Load anywhere with: ort.InferenceSession('tagasenti_int8.onnx', providers=['CPUExecutionProvider'])")
