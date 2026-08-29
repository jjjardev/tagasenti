import pandas as pd


def test_clean_labels():
    from pathlib import Path
    import sys

    # replicate clean_labels logic
    def clean_labels(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["label"] = df["label"].astype(str).str.strip().str.replace('"', "", regex=False)
        df["label"] = pd.to_numeric(df["label"], errors="coerce")
        df = df.dropna(subset=["label"])
        df["label"] = df["label"].astype(int)
        return df[df["label"].isin([0, 1, 2])].reset_index(drop=True)

    df = pd.DataFrame({"sentence": ["a", "b", "c", "d"], "label": ["0", '"1"', " 2 ", "bad"]})
    out = clean_labels(df)
    assert len(out) == 3
    assert out["label"].tolist() == [0, 1, 2]


def test_dedup_keeps_first():
    df = pd.DataFrame(
        {
            "sentence": ["hello", "hello", "world", "hello"],
            "label": [0, 1, 1, 2],
        }
    )
    dedup = df.drop_duplicates(subset=["sentence"], keep="first").reset_index(drop=True)
    assert len(dedup) == 2
    assert dedup.loc[dedup["sentence"] == "hello", "label"].iloc[0] == 0  # first wins
    # conflict detection
    dup_mask = df.duplicated(subset=["sentence"], keep=False)
    groups = df[dup_mask].groupby("sentence")
    conflicts = sum(1 for _, g in groups if g["label"].nunique() > 1)
    assert conflicts == 1


def test_dataset_csv_exists_and_has_columns():
    import os

    csv = "tagasenti_dataset.csv"
    if not os.path.exists(csv):
        # HF Hub only - skip
        return
    df = pd.read_csv(csv, nrows=5)
    assert "sentence" in df.columns
    assert "label" in df.columns
