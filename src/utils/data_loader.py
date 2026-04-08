"""Data generator utilities for memory-safe training on Kaggle."""

import math
import os
import sys
from pathlib import Path
from typing import Tuple

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
        augment: bool = False,
        jitter_std: float = 0.005,
        noise_std: float = 0.003,
        time_warp_prob: float = 0.2,
    ):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.jitter_std = jitter_std
        self.noise_std = noise_std
        self.time_warp_prob = time_warp_prob
        self.indices = np.arange(len(self.x))
        self.on_epoch_end()

    def __len__(self) -> int:
        return int(math.ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        sl = slice(idx * self.batch_size, (idx + 1) * self.batch_size)
        batch_idx = self.indices[sl]
        x_batch = self.x[batch_idx].copy()
        y_batch = self.y[batch_idx]
        if self.augment:
            x_batch = self._augment_batch(x_batch)
        return x_batch, y_batch

    def _augment_batch(self, x_batch: np.ndarray) -> np.ndarray:
        x_aug = x_batch.copy()
        x_aug += np.random.normal(0, self.jitter_std, size=x_aug.shape).astype(np.float32)
        x_aug += np.random.normal(0, self.noise_std, size=x_aug.shape).astype(np.float32)
        if np.random.rand() < self.time_warp_prob:
            shift = np.random.randint(-2, 3)
            x_aug = np.roll(x_aug, shift=shift, axis=1)
        return np.clip(x_aug, 0.0, 1.0).astype(np.float32)

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
