"""
LESSON 2 — Turning text into numbers (feature engineering)

AI/ML concepts:
  • Models work on numbers, not raw strings
  • Bag of Words: count how often each word appears (ignores order)
  • TF-IDF: down-weights common words ("the"), up-weights distinctive ones ("winner")
  • Vectorization: transform all messages into a matrix (rows=SMS, cols=terms)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.feature_extraction.text import TfidfVectorizer

from src.data_loader import load_dataset


def run():
    df = load_dataset()
    messages = df["message"].tolist()

    print("=" * 60)
    print("LESSON 2: Text → numeric features (TF-IDF)")
    print("=" * 60)

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        max_features=5000,
        ngram_range=(1, 2),
    )

    X = vectorizer.fit_transform(messages)
    print(f"\nShape of feature matrix: {X.shape}")
    print(f"  → {X.shape[0]} messages × {X.shape[1]} features (words/phrases)")

    sample = "WINNER!! You have won a free prize"
    sample_vec = vectorizer.transform([sample])
    feature_names = vectorizer.get_feature_names_out()
    scores = sample_vec.toarray().flatten()
    top_idx = scores.argsort()[-8:][::-1]

    print(f"\nTop TF-IDF terms for: \"{sample}\"")
    for i in top_idx:
        if scores[i] > 0:
            print(f"  {feature_names[i]}: {scores[i]:.3f}")

    print("\nWhy this matters:")
    print("  • 'free prize' as a phrase (bigram) catches spam patterns")
    print("  • Same vectorizer must be used at train AND predict time (saved in pipeline)")


if __name__ == "__main__":
    run()
