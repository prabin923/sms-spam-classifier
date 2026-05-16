"""
LESSON 1 — Data exploration (EDA)

AI/ML concepts:
  • Dataset: collection of examples the model learns from
  • Features (X): inputs (here, the SMS text)
  • Labels (y): correct answers (ham vs spam) — supervised learning
  • Class imbalance: when one class has far more samples than another
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_loader import load_dataset


def run():
    df = load_dataset()
    print("=" * 60)
    print("LESSON 1: Explore your data before training")
    print("=" * 60)

    print(f"\nRows (examples): {len(df)}")
    print(f"Columns: {list(df.columns)}")

    print("\n--- First 3 messages ---")
    for _, row in df.head(3).iterrows():
        print(f"  [{row['label']}] {row['message'][:60]}...")

    print("\n--- Label distribution (class balance) ---")
    counts = df["label"].value_counts()
    for label, count in counts.items():
        pct = 100 * count / len(df)
        print(f"  {label}: {count} ({pct:.1f}%)")

    print("\n--- Message length stats ---")
    df["length"] = df["message"].str.len()
    print(df.groupby("label")["length"].describe()[["mean", "min", "max"]])

    print("\nWhy this matters:")
    print("  • Models learn patterns from data — garbage in, garbage out")
    print("  • Imbalanced classes → accuracy can mislead (see Lesson 4)")
    print("  • Spam often has ALL CAPS, URLs, 'WIN', 'FREE' — we'll capture this numerically")


if __name__ == "__main__":
    run()
