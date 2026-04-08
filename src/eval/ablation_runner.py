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
    model.fit(train_gen, epochs=epochs, verbose=0, class_weight=class_weight)
    y_prob = model.predict(x_test, verbose=0).reshape(-1)
    return evaluate_window_and_event_metrics(y_test, y_prob, fps=25.0, threshold=0.35)


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
