#!/usr/bin/env python3
"""Run the full learning pipeline: explore → features → train → evaluate → predict."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description="SMS Spam Classifier — ML learning pipeline")
    parser.add_argument(
        "--step",
        choices=["all", "1", "2", "3", "4", "5", "download"],
        default="all",
        help="Which lesson to run (default: all)",
    )
    parser.add_argument(
        "--model",
        choices=["naive_bayes", "logistic_regression"],
        default="naive_bayes",
    )
    args = parser.parse_args()

    if args.step in ("all", "download"):
        try:
            from download_data import download
            download()
        except Exception as e:
            print(f"Download skipped ({e}). Using sample data in data/")

    if args.step in ("all", "1"):
        from src.lesson_01_explore import run as explore
        explore()

    if args.step in ("all", "2"):
        from src.lesson_02_features import run as features
        features()

    if args.step in ("all", "3"):
        from src.lesson_03_train import run as train
        train(estimator=args.model)

    if args.step in ("all", "4"):
        from src.lesson_04_evaluate import run as evaluate
        evaluate()

    if args.step in ("all", "5"):
        from src.lesson_05_predict import run_demo
        run_demo()


if __name__ == "__main__":
    main()
