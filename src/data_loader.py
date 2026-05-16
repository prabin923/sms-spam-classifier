"""
CONCEPT: Supervised learning needs labeled examples.
Each row = one SMS (feature: text) + label (ham or spam).
"""

import pandas as pd

from config import DATA_DIR, RAW_DATA_FILE


def load_dataset(path=None) -> pd.DataFrame:
    path = path or RAW_DATA_FILE
    if not path.exists():
        sample = DATA_DIR / "sample_smsspamcollection"
        if sample.exists():
            path = sample
        else:
            raise FileNotFoundError(
                f"No dataset at {RAW_DATA_FILE}. Run: python download_data.py"
            )

    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["label", "message"],
        encoding="latin-1",
    )
    df["label"] = df["label"].str.strip().str.lower()
    return df
