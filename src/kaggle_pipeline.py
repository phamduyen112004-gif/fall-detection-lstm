"""One-command Kaggle pipeline runner."""

import argparse
import os
import sys
from pathlib import Path
from time import perf_counter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import INPUT_ROOT, OUTPUT_ROOT, OUTPUT_DATA_FEATURES, OUTPUT_DATA_PROCESSED, OUTPUT_MODELS, OUTPUT_REPORTS
from src.features import feature_engineering
from src import kaggle_sanity
from src.training import train_model
from src.pose.pose_extraction import collect_le2i_video_annotation_pairs


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

    if not str(INPUT_ROOT).startswith("/kaggle/input"):
        print(f"[WARN] INPUT_ROOT is not under /kaggle/input: {INPUT_ROOT}")

    if not str(OUTPUT_ROOT).startswith("/kaggle/working"):
        print(f"[WARN] OUTPUT_ROOT is not under /kaggle/working: {OUTPUT_ROOT}")
    pairs = collect_le2i_video_annotation_pairs(INPUT_ROOT)
    print(f"[INFO] Video pairs discovered: {len(pairs)}")
    if not pairs:
        raise RuntimeError(
            "No videos found under INPUT_ROOT.\n"
            "Expected nested structure: [Room]/[Room]/Videos/*.avi"
        )


def main():
    parser = argparse.ArgumentParser(description="Run Le2i pipeline on Kaggle")
    parser.add_argument("--skip-extract", action="store_true", help="Skip pose+feature extraction")
    parser.add_argument("--skip-train", action="store_true", help="Skip model training")
    parser.add_argument("--extract-only", action="store_true", help="Run extraction/feature stage only")
    parser.add_argument("--train-only", action="store_true", help="Run training stage only")
    parser.add_argument("--skip-sanity", action="store_true", help="Skip final sanity checks")
    parser.add_argument("--strict", action="store_true", help="Fail pipeline if sanity check fails")
    args = parser.parse_args()

    if args.extract_only and args.train_only:
        raise ValueError("Choose only one of --extract-only or --train-only.")
    if args.extract_only:
        args.skip_train = True
        args.skip_extract = False
    if args.train_only:
        args.skip_extract = True
        args.skip_train = False

    _check_paths()

    if not args.skip_extract:
        print("[STEP] Extract pose + build features...")
        t0 = perf_counter()
        feature_engineering.main()
        print(f"[STEP] Extraction finished in {perf_counter() - t0:.1f}s")
    else:
        print("[STEP] Skip extraction.")

    if not args.skip_train:
        feat_file = OUTPUT_DATA_FEATURES / "features_final.npy"
        label_file = OUTPUT_DATA_PROCESSED / "y_data.npy"
        if not feat_file.exists() or not label_file.exists():
            raise FileNotFoundError(
                f"Missing training inputs:\n- {feat_file}\n- {label_file}\n"
                "Run extraction first or remove --skip-extract."
            )
        print("[STEP] Train BiLSTM + Attention...")
        t0 = perf_counter()
        train_model.main()
        print(f"[STEP] Training finished in {perf_counter() - t0:.1f}s")
    else:
        print("[STEP] Skip training.")

    if not args.skip_sanity:
        print("[STEP] Running sanity checks...")
        original_argv = sys.argv[:]
        try:
            sys.argv = [original_argv[0]]
            if args.strict:
                sys.argv.append("--strict")
            kaggle_sanity.main()
        finally:
            sys.argv = original_argv
    else:
        print("[STEP] Skip sanity checks.")

    print("[DONE] Kaggle pipeline completed.")


if __name__ == "__main__":
    main()
