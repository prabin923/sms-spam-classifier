"""
LESSON 5 — Inference (using the trained model)

AI/ML concepts:
  • Inference / prediction: apply learned patterns to new inputs
  • predict_proba: confidence scores (not just ham/spam label)
  • Production: load saved pipeline once, score many messages
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib

from config import MODEL_PATH


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run: python src/lesson_03_train.py"
        )
    return joblib.load(MODEL_PATH)


def predict(message: str, pipeline=None) -> dict:
    pipeline = pipeline or load_model()
    label = pipeline.predict([message])[0]
    proba = pipeline.predict_proba([message])[0]
    classes = list(pipeline.classes_)
    scores = {c: float(p) for c, p in zip(classes, proba)}
    return {
        "message": message,
        "label": label,
        "confidence": scores[label],
        "probabilities": scores,
    }


def run_demo():
    print("=" * 60)
    print("LESSON 5: Predict on new messages")
    print("=" * 60)

    examples = [
        "Hey, are we still on for dinner tonight?",
        "URGENT! You won £1000. Text CLAIM to 80999 now!!!",
        "Your package will arrive tomorrow between 2-4pm.",
        "FREE ringtone! Reply STOP to opt out. Winner selected!",
    ]

    pipeline = load_model()
    for msg in examples:
        result = predict(msg, pipeline)
        bar = "█" * int(result["confidence"] * 20)
        print(f"\n[{result['label'].upper()}] ({result['confidence']:.0%}) {bar}")
        print(f"  {msg}")
        print(f"  P(ham)={result['probabilities'].get('ham', 0):.2f}  "
              f"P(spam)={result['probabilities'].get('spam', 0):.2f}")


if __name__ == "__main__":
    run_demo()
