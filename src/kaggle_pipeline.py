"""One-command Kaggle pipeline runner."""

import argparse
import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import INPUT_ROOT, OUTPUT_DATA_FEATURES, OUTPUT_DATA_PROCESSED, OUTPUT_MODELS, OUTPUT_REPORTS
from src.features import feature_engineering
from src.training import train_model


def _check_paths():
    print(f"[INFO] INPUT_ROOT: {INPUT_ROOT}")
    print(f"[INFO] OUTPUT_DATA_PROCESSED: {OUTPUT_DATA_PROCESSED}")
    print(f"[INFO] OUTPUT_DATA_FEATURES: {OUTPUT_DATA_FEATURES}")
    print(f"[INFO] OUTPUT_MODELS: {OUTPUT_MODELS}")
    print(f"[INFO] OUTPUT_REPORTS: {OUTPUT_REPORTS}")
    if not Path(INPUT_ROOT).exists():
        raise FileNotFoundError(
            f"Dataset root not found: {INPUT_ROOT}\n"
            "Please attach Le2i dataset in Kaggle Input."
        )


def main():
    parser = argparse.ArgumentParser(description="Run Le2i pipeline on Kaggle")
    parser.add_argument("--skip-extract", action="store_true", help="Skip pose+feature extraction")
    parser.add_argument("--skip-train", action="store_true", help="Skip model training")
    args = parser.parse_args()

    _check_paths()

    if not args.skip_extract:
        print("[STEP] Extract pose + build features...")
        feature_engineering.main()
    else:
        print("[STEP] Skip extraction.")

    if not args.skip_train:
        print("[STEP] Train BiLSTM + Attention...")
        train_model.main()
    else:
        print("[STEP] Skip training.")

    print("[DONE] Kaggle pipeline completed.")


if __name__ == "__main__":
    main()
