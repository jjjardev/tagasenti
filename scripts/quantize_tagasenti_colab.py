# ============================================================
# TAGASENTI — ONNX INT8 Quantization (Memory-Optimized for T4)
# ============================================================
# Adapted from: filisenti scripts/quantize_colab_cell.py
# Context: TagaSenti (jjjardev/tagasenti_model, HF-only):
#   - Same base arch: XLM-RoBERTa-large, 3-class (Neg/Neu/Pos),
#     max_len 128, inputs input_ids + attention_mask -> logits.
#   - Base vocab 250002 (no Filipino added-tokens, unlike FiliSenti).
#   - Source: always Hugging Face Hub — no Drive upload needed.
#   - Output names: tagasenti.onnx / tagasenti_int8.onnx
#
# Usage (Colab, T4): paste this whole file into ONE Colab cell and run.
# No model upload needed — pulls straight from HF Hub.
# Output: single-file INT8 model ready for ORT CPU / Flutter / app.py.

!pip install -q "protobuf>=5.26,<6" onnx onnxruntime
# NOTE: protobuf is pinned <6 on purpose. Unpinned `pip install onnx`
# pulls protobuf 7.x, which breaks Colab's preinstalled google-ai /
# ydf / grpcio-status packages. If you already ran the old unpinned
# line, do Runtime > Restart session, then run this cell again.

import os, gc, warnings, subprocess, sys
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from google.colab import drive

# ── Suppress cosmetic warnings ──────────────────────────────
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*legacy.*ONNX.*")
warnings.filterwarnings("ignore", message=".*torch.tensor results are registered as constants.*")

# ── Mount (Drive optional — falls back to /content) ──────────
# If you get "MessageError: credential propagation was unsuccessful":
#   1) use a *hosted* Colab runtime (not local), 2) allow third-party
#   cookies for google.com/drive.google.com (Brave Shields down),
#   3) Runtime > Disconnect and delete runtime > reconnect, then retry.
# Or skip Drive entirely — outputs will go to /content instead.
try:
    drive.mount("/content/drive", force_remount=False)
    ONNX_OUTPUT_DIR = "/content/drive/MyDrive/TagaSenti/model_onnx"
except Exception as e:
    print(f"Drive mount failed ({e}) → using local /content instead.")
    ONNX_OUTPUT_DIR = "/content/tagasenti_onnx"

# ── Paths & IDs (EDIT OUTPUT DIR IF NEEDED) ──────────────────
HF_MODEL_ID     = "jjjardev/tagasenti_model"   # v6 release, Apache-2.0 weights — pulled directly
# ONNX_OUTPUT_DIR already set by the Drive-mount block above; only
# override it here if you want a custom path.
os.makedirs(ONNX_OUTPUT_DIR, exist_ok=True)

FP32_NAME  = "tagasenti.onnx"
INT8_NAME  = "tagasenti_int8.onnx"
onnx_fp32_path = os.path.join(ONNX_OUTPUT_DIR, FP32_NAME)
quant_path     = os.path.join(ONNX_OUTPUT_DIR, INT8_NAME)

MODEL_SOURCE = HF_MODEL_ID
print(f"Pulling model from Hub: {MODEL_SOURCE}")

# ── Free memory at start ────────────────────────────────────
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# ── Load model (low_cpu_mem_usage avoids double-allocation) ──
print("Loading model...")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_SOURCE,
    low_cpu_mem_usage=True,          # ← weights loaded incrementally; no 2× spike
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_SOURCE)
model.eval()
print(f"  num_labels={model.num_labels}, id2label={model.config.id2label}")

# ── Export FP32 ONNX ─────────────────────────────────────────
print("Exporting to ONNX (FP32)...")
# Tagalog dummy input (same 128-token truncation the model was trained with).
dummy = tokenizer("Ang ganda ng quality ng tela, worth it ang price!",
                  return_tensors="pt", truncation=True, max_length=128)

with torch.inference_mode():        # ← tighter than torch.no_grad(); no autograd metadata
    torch.onnx.export(
        model,
        (dummy["input_ids"], dummy["attention_mask"]),
        onnx_fp32_path,
        opset_version=18,
        do_constant_folding=True,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids":      {0: "batch_size", 1: "sequence_length"},
            "attention_mask":  {0: "batch_size", 1: "sequence_length"},
            "logits":          {0: "batch_size"},
        },
        dynamo=False,                # legacy exporter — stable with dynamic_axes
    )

# ── Free ALL PyTorch memory ──────────────────────────────────
del model, dummy
gc.collect()                        # first pass: direct refs
if torch.cuda.is_available():
    torch.cuda.empty_cache()
gc.collect()                        # second pass: cyclic refs freed by first pass

# ── Quantize in isolated subprocess ──────────────────────────
# quantize_dynamic calls onnx.load() internally, holding ~1.5 GB
# of protobuf objects.  A child process guarantees the OS reclaims
# *every byte* on exit — no Python GC fragmentation.
print("Quantizing INT8 in isolated subprocess...")
quant_script = os.path.join(ONNX_OUTPUT_DIR, "_quantize_tmp.py")
with open(quant_script, "w") as f:
    f.write(f"""
from onnxruntime.quantization import quantize_dynamic, QuantType
print("Loading ONNX model for quantization...")
quantize_dynamic(
    model_input="{onnx_fp32_path}",
    model_output="{quant_path}",
    weight_type=QuantType.QInt8,
    per_channel=True,
    use_external_data_format=False,
)
print("Quantization succeeded.")
""")
result = subprocess.run(
    [sys.executable, quant_script],
    capture_output=True, text=True, timeout=600,
)
os.remove(quant_script)

print(result.stdout)
if result.returncode != 0:
    raise RuntimeError(f"Quantization failed:\n{result.stderr}")

# ── Save tokenizer ───────────────────────────────────────────
tokenizer.save_pretrained(ONNX_OUTPUT_DIR)

# ── Report ───────────────────────────────────────────────────
def total_onnx_size(base_path):
    total = os.path.getsize(base_path)
    data_path = base_path + ".data"
    if os.path.exists(data_path):
        total += os.path.getsize(data_path)
    return total

fp32_mb = total_onnx_size(onnx_fp32_path) / 1e6
int8_mb = total_onnx_size(quant_path) / 1e6

print(f"\n✅ Done! Files saved in: {ONNX_OUTPUT_DIR}")
print(f"   FP32 : {fp32_mb:>8.1f} MB  →  {FP32_NAME}")
print(f"   INT8 : {int8_mb:>8.1f} MB  →  {INT8_NAME}")
print(f"   Ratio: {fp32_mb / int8_mb:.1f}× compression")
print(f"   Single-file INT8 model — load {INT8_NAME} straight into ORT CPU / Flutter app / app.py.")
print(f"   Spec: inputs input_ids+attention_mask int64 [batch, seq_len], output logits float32 [batch, 3], opset 18.")
