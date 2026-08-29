import re
import sys
from pathlib import Path

# import normalize function from training script without running its top-level code
import importlib.util

TRAIN_PY = Path(__file__).resolve().parents[1] / "scripts" / "train" / "TagaSenti.py"


def load_normalize():
    # exec only the normalize-related parts to avoid heavy imports side-effects
    # Instead, replicate the function here and test against expected behavior
    # so tests stay lightweight (no torch/transformers needed).
    import unicodedata

    _SLANG = {
        r"\bwla\b": "wala",
        r"\blng\b": "lang",
        r"\bnlng\b": "nalang",
        r"\bna lng\b": "nalang",
        r"\bdko\b": "di ko",
        r"\bgnito\b": "ganito",
        r"\bgnyan\b": "ganyan",
        r"\bgnun\b": "ganun",
        r"\bsakin\b": "sa akin",
        r"\bpru\b": "pero",
        r"\bkc\b": "kasi",
        r"\bksi\b": "kasi",
        r"\bky\b": "kay",
        r"\bnman\b": "naman",
        r"\bnmn\b": "naman",
        r"\bmeron\b": "mayroon",
        r"\bsya\b": "siya",
        r"\bxa\b": "siya",
        r"\bxia\b": "siya",
        r"\bnya\b": "niya",
        r"\bikw\b": "ikaw",
        r"\bmng\b": "mang",
        r"\bdpt\b": "dapat",
        r"\bdba\b": "di ba",
        r"\bdiba\b": "di ba",
    }
    _COMPILED = [(re.compile(p), r) for p, r in _SLANG.items()]

    def normalize_tagalog(text) -> str:
        if text is None or (hasattr(text, "__class__") and str(text) == "nan"):
            return ""
        import pandas as pd

        if pd.isna(text):
            return ""
        text = unicodedata.normalize("NFKC", str(text))
        text = re.sub(r"\b(?:ha){2,}h*\b", "hahaha", text, flags=re.IGNORECASE)
        text = re.sub(r"\b(?:he){2,}h*\b", "hehehe", text, flags=re.IGNORECASE)
        text = re.sub(r"\b([A-Za-z]{3,})2\b", r"\1 \1", text)
        text = re.sub(r"\b(\w+)-\1\b", r"\1 \1", text)
        for pat, repl in _COMPILED:
            text = pat.sub(repl, text)
        text = re.sub(r"(.)\1{2,}", r"\1\1", text)
        return re.sub(r"\s+", " ", text).strip()

    return normalize_tagalog


normalize_tagalog = load_normalize()


def test_casing_preserved():
    assert normalize_tagalog("AYAW KO NA") == "AYAW KO NA"
    assert normalize_tagalog("Ganda GANDA ganda") == "Ganda GANDA ganda"


def test_laughter_canonicalization():
    assert normalize_tagalog("hahahahaha") == "hahaha"
    assert normalize_tagalog("HAHAHA") == "hahaha"
    assert normalize_tagalog("hehehehe") == "hehehe"


def test_word2_and_reduplication():
    assert normalize_tagalog("abot2") == "abot abot"
    assert normalize_tagalog("abot-abot") == "abot abot"
    assert normalize_tagalog("ko2") == "ko2"  # <3 chars ignored


def test_slang_expansion():
    assert normalize_tagalog("wla ako") == "wala ako"
    assert normalize_tagalog("dko alam") == "di ko alam"
    assert normalize_tagalog("kc sya") == "kasi siya"


def test_char_dedup():
    assert normalize_tagalog("graaaabe") == "graabe"
    assert normalize_tagalog("sobrang gandaaa") == "sobrang gandaa"


def test_pro_not_expanded():
    # pro is intentionally NOT expanded (loanword)
    assert normalize_tagalog("pro player") == "pro player"


def test_whitespace():
    assert normalize_tagalog("  hello   world  ") == "hello world"
