"""Run fast ablation study for thesis reporting."""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import OUTPUT_DATA_FEATURES, OUTPUT_DATA_PROCESSED, OUTPUT_REPORTS
from src.eval.event_metrics import evaluate_window_and_event_metrics
from src.models.architectures import (build_bilstm_attention_model,
                                      build_bilstm_no_attention_model)
from src.training.losses import binary_focal_loss
from src.utils.data_loader import NpySequence


def _compute_cls_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    y_true_i = y_true.reshape(-1).astype(np.int32)
    y_pred_i = (y_prob.reshape(-1) >= threshold).astype(np.int32)
    tp = int(((y_true_i == 1) & (y_pred_i == 1)).sum())
    fp = int(((y_true_i == 0) & (y_pred_i == 1)).sum())
    fn = int(((y_true_i == 1) & (y_pred_i == 0)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = (2 * precision * recall) / max(precision + recall, 1e-9)
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}


def _find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    best_t = 0.5
    best_f1 = -1.0
    best_precision = -1.0
    for t in np.arange(0.10, 0.91, 0.01):
        m = _compute_cls_metrics(y_true, y_prob, float(t))
        if (
            m["f1"] > best_f1
            or (np.isclose(m["f1"], best_f1) and m["precision"] > best_precision)
            or (
                np.isclose(m["f1"], best_f1)
                and np.isclose(m["precision"], best_precision)
                and abs(t - 0.5) < abs(best_t - 0.5)
            )
        ):
            best_f1 = m["f1"]
            best_precision = m["precision"]
            best_t = float(t)
    return best_t


def _train_eval_model(model, x_train, y_train, x_test, y_test, epochs: int = 25):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
        loss=binary_focal_loss(gamma=2.0, alpha=0.25),
        metrics=["accuracy"],
    )
    train_gen = NpySequence(x_train, y_train, batch_size=32, shuffle=True, augment=True)
    classes = np.unique(y_train.astype(np.int32))
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train.astype(np.int32))
    class_weight = {int(c): float(w) for c, w in zip(classes, weights)}
    x_tr, x_val, y_tr, y_val = train_test_split(
        x_train, y_train, test_size=0.2, random_state=42, stratify=y_train.astype(np.int32)
    )
    tr_gen = NpySequence(x_tr, y_tr, batch_size=32, shuffle=True, augment=True)
    model.fit(tr_gen, epochs=epochs, verbose=0, class_weight=class_weight)
    y_val_prob = model.predict(x_val, verbose=0).reshape(-1)
    best_threshold = _find_best_threshold(y_val, y_val_prob)
    y_prob = model.predict(x_test, verbose=0).reshape(-1)
    metrics = evaluate_window_and_event_metrics(y_test, y_prob, fps=25.0, threshold=best_threshold)
    metrics["threshold"] = float(best_threshold)
    return metrics


def main():
    x_pose = np.load(OUTPUT_DATA_PROCESSED / "x_data.npy").astype(np.float32)  # [N,75,17,3]
    y = np.load(OUTPUT_DATA_PROCESSED / "y_data.npy").reshape(-1).astype(np.int32)
    print(f"[DATA] Total={len(y)}, Pos={(y==1).sum()}, Neg={(y==0).sum()}")
    x_feat = np.load(OUTPUT_DATA_FEATURES / "features_final.npy").astype(np.float32)  # [N,75,K]
    x_feat_ns = np.load(OUTPUT_DATA_FEATURES / "features_final_nosmooth.npy").astype(np.float32)
    x_raw_pose = x_pose.reshape(len(x_pose), x_pose.shape[1], -1).astype(np.float32)  # [N,75,51]

    x_train_feat, x_test_feat, y_train, y_test = train_test_split(
        x_feat, y, test_size=0.2, random_state=42, stratify=y
    )
    x_train_feat_ns, x_test_feat_ns, _, _ = train_test_split(
        x_feat_ns, y, test_size=0.2, random_state=42, stratify=y
    )
    x_train_raw, x_test_raw, _, _ = train_test_split(
        x_raw_pose, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[SPLIT] Train Pos={(y_train==1).sum()}, Train Neg={(y_train==0).sum()}")
    print(f"[SPLIT] Test Pos={(y_test==1).sum()}, Test Neg={(y_test==0).sum()}")

    rows = []

    # Ablation 1: no attention vs with attention (same feature input)
    m_attn = build_bilstm_attention_model(n_features=x_train_feat.shape[-1], learning_rate=5e-4)
    r_attn = _train_eval_model(m_attn, x_train_feat, y_train, x_test_feat, y_test)
    r_attn["ablation"] = "with_attention_pose_plus_physics"
    rows.append(r_attn)

    m_no = build_bilstm_no_attention_model(n_features=x_train_feat.shape[-1], learning_rate=5e-4)
    r_no = _train_eval_model(m_no, x_train_feat, y_train, x_test_feat, y_test)
    r_no["ablation"] = "no_attention_pose_plus_physics"
    rows.append(r_no)

    # Ablation 2: raw pose only vs pose + physical features
    m_raw = build_bilstm_attention_model(n_features=x_train_raw.shape[-1], learning_rate=5e-4)
    r_raw = _train_eval_model(m_raw, x_train_raw, y_train, x_test_raw, y_test)
    r_raw["ablation"] = "with_attention_raw_pose_only"
    rows.append(r_raw)

    # Ablation 3: with smoothing vs no smoothing (same architecture/features)
    m_sm = build_bilstm_attention_model(n_features=x_train_feat.shape[-1], learning_rate=5e-4)
    r_sm = _train_eval_model(m_sm, x_train_feat, y_train, x_test_feat, y_test)
    r_sm["ablation"] = "with_smoothing"
    rows.append(r_sm)

    m_ns = build_bilstm_attention_model(n_features=x_train_feat_ns.shape[-1], learning_rate=5e-4)
    r_ns = _train_eval_model(m_ns, x_train_feat_ns, y_train, x_test_feat_ns, y_test)
    r_ns["ablation"] = "no_smoothing"
    rows.append(r_ns)

    out = pd.DataFrame(rows)
    OUTPUT_REPORTS.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_REPORTS / "ablation_results.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    print(out)


if __name__ == "__main__":
    main()
