"""Data generator utilities for memory-safe training on Kaggle."""

import math
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import tensorflow as tf

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class NpySequence(tf.keras.utils.Sequence):
    """
    Keras Sequence data generator over in-memory numpy arrays.
    Keeps training loop memory-stable and supports shuffling per epoch.
    """

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        batch_size: int = 32,
        shuffle: bool = True,
    ):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.x))
        self.on_epoch_end()

    def __len__(self) -> int:
        return int(math.ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        sl = slice(idx * self.batch_size, (idx + 1) * self.batch_size)
        batch_idx = self.indices[sl]
        return self.x[batch_idx], self.y[batch_idx]

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


def load_feature_label_arrays(
    feature_path: Path,
    label_path: Path,
) -> Tuple[np.ndarray, np.ndarray]:
    if not feature_path.exists():
        raise FileNotFoundError(f"Missing feature file: {feature_path}")
    if not label_path.exists():
        raise FileNotFoundError(f"Missing label file: {label_path}")

    x_data = np.load(feature_path).astype(np.float32)
    y_data = np.load(label_path).astype(np.float32).reshape(-1)
    return x_data, y_data
