"""Temporal smoothing and missing-data handling for pose sequences."""

from typing import Optional

import numpy as np
from scipy.signal import savgol_filter

from src.config import (
    MAX_CONSECUTIVE_MISSING,
    N_CHANNELS,
    N_KEYPOINTS,
    SAVGOL_POLYORDER,
    SAVGOL_WINDOW,
)


def max_consecutive_missing(mask: np.ndarray) -> int:
    """Compute max run length of True (missing) in boolean mask."""
    max_run = 0
    cur = 0
    for m in mask:
        if m:
            cur += 1
            max_run = max(max_run, cur)
        else:
            cur = 0
    return max_run


def fill_and_smooth_window(window_pose: np.ndarray) -> Optional[np.ndarray]:
    """
    Handle missing data and smoothing for one sample [75, 17, 3].
    - If max consecutive missing > 10: drop sample (return None).
    - Else interpolate x,y,conf via np.interp.
    - Apply Savitzky-Golay for each keypoint channel over time.
    - Clamp to [0,1].
    """
    w = window_pose.copy().astype(np.float32)
    T = w.shape[0]

    frame_missing = (
        np.isnan(w[:, :, 0]).all(axis=1)
        | (np.nan_to_num(w[:, :, 2], nan=0.0) <= 0).all(axis=1)
    )

    if max_consecutive_missing(frame_missing) > MAX_CONSECUTIVE_MISSING:
        return None

    t_idx = np.arange(T, dtype=np.float32)
    for j in range(N_KEYPOINTS):
        for c in range(N_CHANNELS):
            series = w[:, j, c].copy()
            valid = ~np.isnan(series)

            if valid.sum() == 0:
                return None
            if valid.sum() == 1:
                series = np.full_like(series, series[valid][0])
            else:
                series = np.interp(t_idx, t_idx[valid], series[valid]).astype(np.float32)

            if T >= SAVGOL_WINDOW:
                series = savgol_filter(
                    series,
                    window_length=SAVGOL_WINDOW,
                    polyorder=SAVGOL_POLYORDER,
                    mode="interp",
                ).astype(np.float32)

            w[:, j, c] = series

    w[:, :, 0] = np.clip(w[:, :, 0], 0.0, 1.0)
    w[:, :, 1] = np.clip(w[:, :, 1], 0.0, 1.0)
    w[:, :, 2] = np.clip(w[:, :, 2], 0.0, 1.0)
    return w


def fill_without_smoothing(window_pose: np.ndarray) -> Optional[np.ndarray]:
    """
    Handle missing data without Savitzky-Golay smoothing (ablation baseline).
    - If max consecutive missing > 10: drop sample.
    - Else interpolate x,y,conf via np.interp.
    - Clamp to [0,1].
    """
    w = window_pose.copy().astype(np.float32)
    T = w.shape[0]

    frame_missing = (
        np.isnan(w[:, :, 0]).all(axis=1)
        | (np.nan_to_num(w[:, :, 2], nan=0.0) <= 0).all(axis=1)
    )
    if max_consecutive_missing(frame_missing) > MAX_CONSECUTIVE_MISSING:
        return None

    t_idx = np.arange(T, dtype=np.float32)
    for j in range(N_KEYPOINTS):
        for c in range(N_CHANNELS):
            series = w[:, j, c].copy()
            valid = ~np.isnan(series)
            if valid.sum() == 0:
                return None
            if valid.sum() == 1:
                series = np.full_like(series, series[valid][0])
            else:
                series = np.interp(t_idx, t_idx[valid], series[valid]).astype(np.float32)
            w[:, j, c] = series

    w[:, :, 0] = np.clip(w[:, :, 0], 0.0, 1.0)
    w[:, :, 1] = np.clip(w[:, :, 1], 0.0, 1.0)
    w[:, :, 2] = np.clip(w[:, :, 2], 0.0, 1.0)
    return w
