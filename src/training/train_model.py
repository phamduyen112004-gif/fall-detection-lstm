"""
Train BiLSTM + Attention model for Le2i fall detection.

Ghi chú:
- Le2i có FPS = 25, vì vậy cửa sổ 75 frames tương đương đúng 3.0 giây.
- Cửa sổ 3 giây giúp bao quát giai đoạn trước, trong và sau cú ngã,
  phù hợp để mô hình học ngữ cảnh động học.
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
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import OUTPUT_DATA_FEATURES, OUTPUT_DATA_PROCESSED, OUTPUT_MODELS, OUTPUT_REPORTS, SEQ_LEN
from src.models.architectures import build_bilstm_attention_model
from src.utils.data_loader import NpySequence, load_feature_label_arrays


RANDOM_STATE = 42
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 5e-4


def load_data(
    x_path: Path = OUTPUT_DATA_FEATURES / "features_final.npy",
    y_path: Path = OUTPUT_DATA_PROCESSED / "y_data.npy",
) -> Tuple[np.ndarray, np.ndarray]:
    """Load precomputed features [N, 75, K] and labels [N, 1]."""
    x_data, y_data = load_feature_label_arrays(x_path, y_path)

    if x_data.ndim != 3:
        raise ValueError(f"Expected x_data shape [N,75,K], got {x_data.shape}")
    if x_data.shape[1] != SEQ_LEN:
        raise ValueError(f"Expected time_steps={SEQ_LEN}, got {x_data.shape[1]}")
    if len(x_data) != len(y_data):
        raise ValueError("x_data and y_data length mismatch")

    return x_data, y_data


def plot_training_curves(history, output_dir: Path):
    """Plot training/validation Accuracy and Loss."""
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
    """Plot and save confusion matrix."""
    y_pred = (y_prob >= 0.5).astype(np.int32)
    cm = confusion_matrix(y_true.astype(np.int32), y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["ADL", "Nga"])

    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix tren tap test")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=200)
    plt.close(fig)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"TensorFlow device preference: {device}")
    if device == "cuda":
        for gpu in tf.config.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(gpu, True)

    # Chon chuoi 75 frame vi Le2i co 25 FPS => 3.0 giay, du de bat tron hanh vi nga.
    print("Loading features from Kaggle working directory...")
    x_data, y_data = load_data()
    n_features = x_data.shape[-1]
    print(f"x_data: {x_data.shape}, y_data: {y_data.shape}, K={n_features}")

    x_train, x_test, y_train, y_test = train_test_split(
        x_data,
        y_data,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_data.astype(np.int32),
    )
    print(f"Train: {x_train.shape}, Test: {x_test.shape}")

    model = build_bilstm_attention_model(
        n_features=n_features,
        learning_rate=LEARNING_RATE,
    )
    model.summary()

    model_dir = OUTPUT_MODELS
    report_dir = OUTPUT_REPORTS
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=15,
            restore_best_weights=True,
            verbose=1,
        ),
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
    train_gen = NpySequence(x_train_sub, y_train_sub, batch_size=BATCH_SIZE, shuffle=True)
    val_gen = NpySequence(x_val, y_val, batch_size=BATCH_SIZE, shuffle=False)

    history = model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS, callbacks=callbacks, verbose=1)

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

    y_prob = model.predict(x_test, verbose=0).reshape(-1)
    plot_training_curves(history, report_dir)
    plot_confusion_matrix(y_test, y_prob, report_dir)
    pd.DataFrame(history.history).to_csv(report_dir / "history.csv", index=False)

    model.save(model_dir / "final_bilstm_attention.keras")
    print(f"Saved model to: {model_dir / 'final_bilstm_attention.keras'}")
    print(f"Saved best checkpoint to: {model_dir / 'best_bilstm_attention.keras'}")
    print(
        f"Saved reports: {report_dir / 'training_curves.png'}, "
        f"{report_dir / 'confusion_matrix.png'}, {report_dir / 'history.csv'}"
    )


if __name__ == "__main__":
    main()
