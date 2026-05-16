"""
LESSON 4 — Evaluating model quality

AI/ML concepts:
  • Accuracy: % correct — misleading if classes are imbalanced
  • Precision: of predicted spam, how many are truly spam?
  • Recall: of actual spam, how many did we catch?
  • F1-score: balance of precision and recall
  • Confusion matrix: TP, FP, TN, FN at a glance
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split

from config import MODEL_PATH
from src.data_loader import load_dataset


def run(save_plot: bool = True):
    if not MODEL_PATH.exists():
        print("No trained model. Run: python src/lesson_03_train.py")
        return

    df = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        df["message"],
        df["label"],
        test_size=0.2,
        random_state=42,
        stratify=df["label"],
    )

    pipeline = joblib.load(MODEL_PATH)
    y_pred = pipeline.predict(X_test)

    print("=" * 60)
    print("LESSON 4: Evaluate with the right metrics")
    print("=" * 60)

    print("\n--- Classification report ---")
    print(classification_report(y_test, y_pred, target_names=["ham", "spam"]))

    cm = confusion_matrix(y_test, y_pred, labels=["ham", "spam"])
    print("--- Confusion matrix [rows=actual, cols=predicted] ---")
    print("              pred_ham  pred_spam")
    print(f"  actual_ham     {cm[0,0]:5d}     {cm[0,1]:5d}   ← false positive hurts users")
    print(f"  actual_spam    {cm[1,0]:5d}     {cm[1,1]:5d}   ← false negative lets spam through")

    print("\nWhy this matters:")
    print("  • Blocking real SMS (false positive) is often worse than missing spam")
    print("  • Tune threshold / model based on your product goals")

    if save_plot:
        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay(cm, display_labels=["ham", "spam"]).plot(ax=ax, cmap="Blues")
        ax.set_title("Spam classifier — confusion matrix")
        out = Path(__file__).resolve().parents[1] / "data" / "confusion_matrix.png"
        fig.savefig(out, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"\nSaved plot → {out}")


if __name__ == "__main__":
    run()
