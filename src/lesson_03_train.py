"""
LESSON 3 — Training a classifier

AI/ML concepts:
  • Training: algorithm finds patterns linking features → labels
  • Train/test split: hold out data the model never sees during training
  • Naive Bayes: classic for text/spam; fast, probabilistic word counts
  • Logistic Regression: learns weighted combination of features
  • Pipeline: chains preprocessing + model so they stay in sync
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from config import MODEL_DIR, MODEL_PATH
from src.data_loader import load_dataset


def build_pipeline(estimator_name: str = "naive_bayes") -> Pipeline:
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        max_features=5000,
        ngram_range=(1, 2),
    )
    if estimator_name == "logistic_regression":
        classifier = LogisticRegression(max_iter=1000, class_weight="balanced")
    else:
        classifier = MultinomialNB()

    return Pipeline([
        ("tfidf", vectorizer),
        ("classifier", classifier),
    ])


def run(estimator: str = "naive_bayes", test_size: float = 0.2):
    df = load_dataset()
    X = df["message"]
    y = df["label"]

    print("=" * 60)
    print("LESSON 3: Train a spam classifier")
    print("=" * 60)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    print(f"\nTrain: {len(X_train)} | Test: {len(X_test)} (held out for honest evaluation)")

    pipeline = build_pipeline(estimator)
    print(f"\nTraining {estimator} ...")
    pipeline.fit(X_train, y_train)

    train_acc = pipeline.score(X_train, y_train)
    test_acc = pipeline.score(X_test, y_test)
    print(f"  Train accuracy: {train_acc:.3f}")
    print(f"  Test accuracy:  {test_acc:.3f}")

    if train_acc - test_acc > 0.15:
        print("  ⚠ Large train/test gap → possible overfitting on small data")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"\nSaved model → {MODEL_PATH}")

    print("\nWhy this matters:")
    print("  • Test set simulates new, unseen SMS messages")
    print("  • Pipeline ensures prediction uses the same TF-IDF as training")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["naive_bayes", "logistic_regression"],
        default="naive_bayes",
    )
    args = parser.parse_args()
    run(estimator=args.model)
