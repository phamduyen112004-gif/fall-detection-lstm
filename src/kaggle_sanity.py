"""Quick sanity checks for Kaggle outputs."""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import OUTPUT_DATA_FEATURES, OUTPUT_DATA_PROCESSED, OUTPUT_MODELS, OUTPUT_REPORTS


def _print_file_status(path: Path, label: str):
    print(f"[{label}] {path} -> {'OK' if path.exists() else 'MISSING'}")


def _check_arrays():
    x_path = OUTPUT_DATA_PROCESSED / "x_data.npy"
    x_ns_path = OUTPUT_DATA_PROCESSED / "x_data_nosmooth.npy"
    y_path = OUTPUT_DATA_PROCESSED / "y_data.npy"
    s_path = OUTPUT_DATA_PROCESSED / "scene_ids.npy"
    f_path = OUTPUT_DATA_FEATURES / "features_final.npy"
    f_ns_path = OUTPUT_DATA_FEATURES / "features_final_nosmooth.npy"

    _print_file_status(x_path, "X_DATA")
    _print_file_status(x_ns_path, "X_DATA_NOSMOOTH")
    _print_file_status(y_path, "Y_DATA")
    _print_file_status(s_path, "SCENE_IDS")
    _print_file_status(f_path, "FEATURES")
    _print_file_status(f_ns_path, "FEATURES_NOSMOOTH")

    ok = True
    if not (x_path.exists() and x_ns_path.exists() and y_path.exists() and f_path.exists() and f_ns_path.exists()):
        print("[WARN] Missing one or more .npy files. Run extraction stage first.")
        return False

    x = np.load(x_path)
    x_ns = np.load(x_ns_path)
    y = np.load(y_path).reshape(-1)
    f = np.load(f_path)
    f_ns = np.load(f_ns_path)
    scene_ids = np.load(s_path, allow_pickle=True) if s_path.exists() else None

    print(f"[SHAPE] x_data: {x.shape}")
    print(f"[SHAPE] y_data: {y.shape}")
    print(f"[SHAPE] features_final: {f.shape}")
    print(f"[SHAPE] x_data_nosmooth: {x_ns.shape}")
    print(f"[SHAPE] features_final_nosmooth: {f_ns.shape}")

    if len(x) != len(y) or len(x) != len(f) or len(x_ns) != len(y) or len(f_ns) != len(y):
        print("[WARN] Sample count mismatch between x/y/features.")
        ok = False
    else:
        print("[OK] Sample counts are aligned.")
    if scene_ids is not None and len(scene_ids) != len(x):
        print("[WARN] scene_ids length mismatch with samples.")
        ok = False

    classes, counts = np.unique(y.astype(np.int32), return_counts=True)
    dist = {int(c): int(n) for c, n in zip(classes, counts)}
    print(f"[LABEL_DIST] {dist}")
    if len(dist) < 2:
        print("[WARN] Only one class present. Training quality may be poor.")
        ok = False
    return ok


def _check_models_and_reports():
    best_model = OUTPUT_MODELS / "best_bilstm_attention.keras"
    final_model = OUTPUT_MODELS / "final_bilstm_attention.keras"
    curve_png = OUTPUT_REPORTS / "training_curves.png"
    cm_png = OUTPUT_REPORTS / "confusion_matrix.png"
    history_csv = OUTPUT_REPORTS / "history.csv"
    metrics_csv = OUTPUT_REPORTS / "metrics_summary.csv"

    statuses = {
        "MODEL_BEST": best_model.exists(),
        "MODEL_FINAL": final_model.exists(),
        "REPORT_CURVES": curve_png.exists(),
        "REPORT_CM": cm_png.exists(),
        "REPORT_HISTORY": history_csv.exists(),
        "REPORT_METRICS": metrics_csv.exists(),
    }
    _print_file_status(best_model, "MODEL_BEST")
    _print_file_status(final_model, "MODEL_FINAL")
    _print_file_status(curve_png, "REPORT_CURVES")
    _print_file_status(cm_png, "REPORT_CM")
    _print_file_status(history_csv, "REPORT_HISTORY")
    _print_file_status(metrics_csv, "REPORT_METRICS")
    return all(statuses.values())


def main():
    parser = argparse.ArgumentParser(description="Sanity checks for Kaggle pipeline artifacts")
    parser.add_argument("--strict", action="store_true", help="Exit with code 1 if any check fails")
    args = parser.parse_args()

    print("=== Kaggle Sanity Check ===")
    print(f"[OUT] processed: {OUTPUT_DATA_PROCESSED}")
    print(f"[OUT] features : {OUTPUT_DATA_FEATURES}")
    print(f"[OUT] models   : {OUTPUT_MODELS}")
    print(f"[OUT] reports  : {OUTPUT_REPORTS}")
    print()

    arrays_ok = _check_arrays()
    print()
    model_report_ok = _check_models_and_reports()

    all_ok = arrays_ok and model_report_ok
    print(f"\n[RESULT] {'PASS' if all_ok else 'FAIL'}")
    print("[DONE] Sanity check complete.")
    if args.strict and not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
