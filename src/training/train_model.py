"""
Train BiLSTM + Attention model for Le2i fall detection.

Enhancements:
- Focal loss + class weights (class imbalance handling)
- Lightweight augmentation in Sequence generator
- Optional scene-split CV (LE2I_RUN_SCENE_CV=1)
"""

import os
import sys
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import OUTPUT_DATA_FEATURES, OUTPUT_DATA_PROCESSED, OUTPUT_MODELS, OUTPUT_REPORTS, SEQ_LEN
from src.models.architectures import build_bilstm_attention_model
from src.training.losses import binary_focal_loss
from src.utils.data_loader import NpySequence, load_feature_label_arrays


RANDOM_STATE = 42
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 5e-4


def load_data(
    x_path: Path = OUTPUT_DATA_FEATURES / "features_final.npy",
    y_path: Path = OUTPUT_DATA_PROCESSED / "y_data.npy",
) -> Tuple[np.ndarray, np.ndarray]:
    x_data, y_data = load_feature_label_arrays(x_path, y_path)
    if x_data.ndim != 3:
        raise ValueError(f"Expected x_data shape [N,75,K], got {x_data.shape}")
    if x_data.shape[1] != SEQ_LEN:
        raise ValueError(f"Expected time_steps={SEQ_LEN}, got {x_data.shape[1]}")
    if len(x_data) != len(y_data):
        raise ValueError("x_data and y_data length mismatch")
    return x_data, y_data


def plot_training_curves(history, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history.get("accuracy", []), label="Train Accuracy")
    axes[0].plot(history.history.get("val_accuracy", []), label="Val Accuracy")
    axes[0].set_title("Accuracy qua cac epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history.history.get("loss", []), label="Train Loss")
    axes[1].plot(history.history.get("val_loss", []), label="Val Loss")
    axes[1].set_title("Loss qua cac epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "training_curves.png", dpi=200)
    plt.close(fig)


def plot_confusion_matrix(y_true: np.ndarray, y_prob: np.ndarray, output_dir: Path):
    y_pred = (y_prob >= 0.5).astype(np.int32)
    cm = confusion_matrix(y_true.astype(np.int32), y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["ADL", "Nga"])
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix tren tap test")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=200)
    plt.close(fig)


def compute_classification_metrics(y_true: np.ndarray, y_prob: np.ndarray):
    y_true_i = y_true.astype(np.int32).reshape(-1)
    y_pred_i = (y_prob.reshape(-1) >= 0.5).astype(np.int32)
    tp = int(((y_true_i == 1) & (y_pred_i == 1)).sum())
    tn = int(((y_true_i == 0) & (y_pred_i == 0)).sum())
    fp = int(((y_true_i == 0) & (y_pred_i == 1)).sum())
    fn = int(((y_true_i == 1) & (y_pred_i == 0)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = (2 * precision * recall) / max(precision + recall, 1e-9)
    specificity = tn / max(tn + fp, 1)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "specificity": float(specificity),
    }


def run_scene_split_cv(x_data: np.ndarray, y_data: np.ndarray, groups: np.ndarray, output_dir: Path):
    n_splits = min(5, len(np.unique(groups)))
    if n_splits < 2:
        print("[WARN] Scene CV skipped: not enough distinct scenes.")
        return
    gkf = GroupKFold(n_splits=n_splits)
    rows = []
    for fold, (tr_idx, te_idx) in enumerate(gkf.split(x_data, y_data, groups), 1):
        x_tr, y_tr = x_data[tr_idx], y_data[tr_idx]
        x_te, y_te = x_data[te_idx], y_data[te_idx]
        model = build_bilstm_attention_model(n_features=x_data.shape[-1], learning_rate=LEARNING_RATE)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss=binary_focal_loss(gamma=2.0, alpha=0.25),
            metrics=["accuracy"],
        )
        cw_cls = np.unique(y_tr.astype(np.int32))
        cw_val = compute_class_weight(class_weight="balanced", classes=cw_cls, y=y_tr.astype(np.int32))
        class_weight = {int(c): float(w) for c, w in zip(cw_cls, cw_val)}
        tr_gen = NpySequence(x_tr, y_tr, batch_size=BATCH_SIZE, shuffle=True, augment=True)
        te_gen = NpySequence(x_te, y_te, batch_size=BATCH_SIZE, shuffle=False, augment=False)
        model.fit(tr_gen, epochs=10, verbose=0, class_weight=class_weight)
        _, acc = model.evaluate(te_gen, verbose=0)
        rows.append({"fold": fold, "scene_count": int(len(np.unique(groups[te_idx]))), "acc": float(acc)})
        print(f"[SceneCV] Fold {fold}: acc={acc:.4f}")
    pd.DataFrame(rows).to_csv(output_dir / "scene_split_cv.csv", index=False)
    print(f"[SceneCV] Saved: {output_dir / 'scene_split_cv.csv'}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"TensorFlow device preference: {device}")
    if device == "cuda":
        for gpu in tf.config.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(gpu, True)

    print("Loading features from Kaggle working directory...")
    x_data, y_data = load_data()
    n_features = x_data.shape[-1]
    print(f"x_data: {x_data.shape}, y_data: {y_data.shape}, K={n_features}")
    scene_path = OUTPUT_DATA_PROCESSED / "scene_ids.npy"
    scene_ids = np.load(scene_path, allow_pickle=True) if scene_path.exists() else None

    x_train, x_test, y_train, y_test = train_test_split(
        x_data,
        y_data,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_data.astype(np.int32),
    )
    print(f"Train: {x_train.shape}, Test: {x_test.shape}")

    model = build_bilstm_attention_model(n_features=n_features, learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=binary_focal_loss(gamma=2.0, alpha=0.25),
        metrics=["accuracy"],
    )
    model.summary()

    model_dir = OUTPUT_MODELS
    report_dir = OUTPUT_REPORTS
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True, verbose=1),
        ModelCheckpoint(
            filepath=str(model_dir / "best_bilstm_attention.keras"),
            monitor="val_loss",
            save_best_only=True,
            mode="min",
            verbose=1,
        ),
    ]

    x_train_sub, x_val, y_train_sub, y_val = train_test_split(
        x_train,
        y_train,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_train.astype(np.int32),
    )
    train_gen = NpySequence(x_train_sub, y_train_sub, batch_size=BATCH_SIZE, shuffle=True, augment=True)
    val_gen = NpySequence(x_val, y_val, batch_size=BATCH_SIZE, shuffle=False, augment=False)

    cw_classes = np.unique(y_train_sub.astype(np.int32))
    cw_values = compute_class_weight(class_weight="balanced", classes=cw_classes, y=y_train_sub.astype(np.int32))
    class_weight = {int(c): float(w) for c, w in zip(cw_classes, cw_values)}
    print(f"Class weights: {class_weight}")

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1,
    )

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

    y_prob = model.predict(x_test, verbose=0).reshape(-1)
    cls_metrics = compute_classification_metrics(y_test, y_prob)
    print(
        f"Precision={cls_metrics['precision']:.4f}, Recall={cls_metrics['recall']:.4f}, "
        f"F1={cls_metrics['f1']:.4f}, Specificity={cls_metrics['specificity']:.4f}"
    )
    plot_training_curves(history, report_dir)
    plot_confusion_matrix(y_test, y_prob, report_dir)
    pd.DataFrame(history.history).to_csv(report_dir / "history.csv", index=False)
    pd.DataFrame([cls_metrics]).to_csv(report_dir / "metrics_summary.csv", index=False)

    model.save(model_dir / "final_bilstm_attention.keras")
    print(f"Saved model to: {model_dir / 'final_bilstm_attention.keras'}")
    print(f"Saved best checkpoint to: {model_dir / 'best_bilstm_attention.keras'}")
    print(
        f"Saved reports: {report_dir / 'training_curves.png'}, "
        f"{report_dir / 'confusion_matrix.png'}, {report_dir / 'history.csv'}, {report_dir / 'metrics_summary.csv'}"
    )

    if os.getenv("LE2I_RUN_SCENE_CV", "0") == "1" and scene_ids is not None and len(scene_ids) == len(x_data):
        run_scene_split_cv(x_data, y_data, np.asarray(scene_ids), report_dir)


if __name__ == "__main__":
    main()
