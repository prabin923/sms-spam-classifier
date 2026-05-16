#!/usr/bin/env python3
"""Interactive CLI — classify SMS messages with your trained model."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.lesson_05_predict import predict


def main():
    parser = argparse.ArgumentParser(description="Classify an SMS as ham or spam")
    parser.add_argument("message", nargs="?", help="SMS text to classify")
    parser.add_argument("-i", "--interactive", action="store_true", help="REPL mode")
    args = parser.parse_args()

    if args.interactive or not args.message:
        print("SMS Spam Classifier (type 'quit' to exit)\n")
        while True:
            try:
                text = input("SMS> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not text or text.lower() in ("quit", "exit", "q"):
                break
            _print_result(predict(text))
        return

    _print_result(predict(args.message))


def _print_result(result: dict):
    label = result["label"].upper()
    conf = result["confidence"]
    emoji = "🚫" if result["label"] == "spam" else "✓"
    print(f"{emoji} {label} ({conf:.0%} confident)")
    print(f"   P(ham)={result['probabilities'].get('ham', 0):.3f}  "
          f"P(spam)={result['probabilities'].get('spam', 0):.3f}")


if __name__ == "__main__":
    main()
