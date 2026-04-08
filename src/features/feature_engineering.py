"""
Le2i pipeline: video -> pose sequences -> advanced features.
Combines pose extraction, windowing, smoothing, and physics/kinematics features.
"""

import gc
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

try:
    from ultralytics import YOLO
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Missing dependency 'ultralytics'. Install with `pip install ultralytics` "
        "or add it to your environment requirements."
    ) from e

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import (
    ADL_STRIDE,
    COG_INDICES,
    IMPACT_INDEX,
    INPUT_ROOT,
    N_CHANNELS,
    N_KEYPOINTS,
    OUTPUT_DATA_FEATURES,
    OUTPUT_DATA_PROCESSED,
    SEQ_LEN,
)
from src.pose.pose_extraction import (collect_le2i_video_annotation_pairs,
                                      extract_pose_sequence, sync_video_and_labels)
from src.pose.smoothing import fill_and_smooth_window, fill_without_smoothing

# Epsilon for numerical stability
EPS = 1e-5


@dataclass
class VideoItem:
    video_path: Path
    annotation_path: Optional[Path]
    scene_name: str


def collect_video_items(root: Path) -> List[VideoItem]:
    items: List[VideoItem] = []
    for vpath, apath, scene_name in collect_le2i_video_annotation_pairs(root):
        items.append(VideoItem(vpath, apath, scene_name))
    return items


def parse_annotation_file(path: Optional[Path]) -> Optional[np.ndarray]:
    """
    Parse Le2i annotation text:
    - line 1: start_fall, line 2: end_fall
    - subsequent lines: frame-level annotations (2nd column is status label)
    """
    if path is None or not path.exists():
        return None

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    if len(lines) < 3:
        return None

    labels: List[int] = []
    for ln in lines[2:]:
        parts = ln.replace(",", " ").split()
        if len(parts) < 2:
            continue
        try:
            labels.append(int(parts[1]))
        except ValueError:
            continue

    if not labels:
        return None
    return np.asarray(labels, dtype=np.int32)


def find_impact_frame(status_labels: Optional[np.ndarray]) -> Optional[int]:
    """Find transition frame where label changes 8 -> 7."""
    if status_labels is None or len(status_labels) < 2:
        return None
    for i in range(1, len(status_labels)):
        if status_labels[i - 1] == 8 and status_labels[i] == 7:
            return i
    return None


def extract_fall_sample(pose_seq: np.ndarray, impact_frame: int) -> Optional[np.ndarray]:
    """Cut a 75-frame window so impact is fixed at index 40."""
    start = impact_frame - IMPACT_INDEX
    end = start + SEQ_LEN
    if start < 0 or end > len(pose_seq):
        return None
    return pose_seq[start:end]


def collect_adl_windows_from_labels(
    pose_seq: np.ndarray,
    status_labels: Optional[np.ndarray],
) -> List[np.ndarray]:
    """
    Collect ADL windows (label 0):
    - Entire video for non-annotated videos.
    - Standing-only windows (label=1) for annotated videos.
    """
    windows: List[np.ndarray] = []
    total = len(pose_seq)

    if total < SEQ_LEN:
        return windows

    if status_labels is None:
        for start in range(0, total - SEQ_LEN + 1, ADL_STRIDE):
            windows.append(pose_seq[start : start + SEQ_LEN])
        return windows

    valid_len = min(total, len(status_labels))
    status = status_labels[:valid_len]
    standing_mask = status == 1

    for start in range(0, valid_len - SEQ_LEN + 1, ADL_STRIDE):
        end = start + SEQ_LEN
        if standing_mask[start:end].all():
            windows.append(pose_seq[start:end])
    return windows


def compute_advanced_features(x_data: np.ndarray) -> np.ndarray:
    """
    Compute physics and kinematics features from pose [N, 75, 17, 3].
    Returns [N, 75, K] with K = 72 (raw pose 51 + geometric 3 + kinematic 6 + moments 4 + stats 8).
    """
    N, T, J, C = x_data.shape
    assert T == SEQ_LEN and J == N_KEYPOINTS and C == N_CHANNELS

    x = x_data[:, :, :, 0]  # [N,75,17]
    y = x_data[:, :, :, 1]
    conf = x_data[:, :, :, 2]

    # ---- 1. Raw pose flattened [N,75,51] ----
    raw_pose = x_data.reshape(N, T, -1)

    # ---- 2. Geometric features ----
    p_shoulders_x = (x[:, :, 5] + x[:, :, 6]) / 2
    p_shoulders_y = (y[:, :, 5] + y[:, :, 6]) / 2
    p_hips_x = (x[:, :, 11] + x[:, :, 12]) / 2
    p_hips_y = (y[:, :, 11] + y[:, :, 12]) / 2
    p_feet_y = (y[:, :, 15] + y[:, :, 16]) / 2

    body_angle = (p_hips_x - p_shoulders_x + EPS) / (p_hips_y - p_shoulders_y + EPS)
    hip_to_feet = p_hips_y - p_feet_y

    x_min = x.min(axis=2)
    x_max = x.max(axis=2)
    y_min = y.min(axis=2)
    y_max = y.max(axis=2)
    width = x_max - x_min + EPS
    height = y_max - y_min + EPS
    aspect_ratio = width / height

    geom = np.stack([body_angle, hip_to_feet, aspect_ratio], axis=2)

    # ---- 3. Kinematic features ----
    cog_x = x[:, :, COG_INDICES].mean(axis=2)
    cog_y = y[:, :, COG_INDICES].mean(axis=2)

    v_x = -np.diff(cog_x, axis=1)
    v_x = np.concatenate([v_x, v_x[:, -1:]], axis=1)
    v_y = -np.diff(cog_y, axis=1)
    v_y = np.concatenate([v_y, v_y[:, -1:]], axis=1)

    a_x = -np.diff(v_x, axis=1)
    a_x = np.concatenate([a_x, a_x[:, -1:]], axis=1)
    a_y = -np.diff(v_y, axis=1)
    a_y = np.concatenate([a_y, a_y[:, -1:]], axis=1)

    kinematic = np.stack([cog_x, cog_y, v_x, v_y, a_x, a_y], axis=2)

    # ---- 4. Spatial moments (per-frame) ----
    m00 = conf.sum(axis=2, keepdims=True) + EPS
    m10 = (x * conf).sum(axis=2, keepdims=True)
    m01 = (y * conf).sum(axis=2, keepdims=True)
    m20 = (x * x * conf).sum(axis=2, keepdims=True)
    m02 = (y * y * conf).sum(axis=2, keepdims=True)
    m11 = (x * y * conf).sum(axis=2, keepdims=True)

    moments = np.concatenate([m00, m11, m02, m20], axis=2)

    # ---- 5. Statistical (sample-level, tiled to 75) ----
    y_hips = p_hips_y
    mu_y = y_hips.mean(axis=1).reshape(-1, 1)
    sigma_y = y_hips.std(axis=1).reshape(-1, 1) + EPS
    c_y = y_hips - mu_y
    skew_y = (c_y**3).mean(axis=1).reshape(-1, 1) / (sigma_y**3)
    kurt_y = (c_y**4).mean(axis=1).reshape(-1, 1) / (sigma_y**4)

    mu_a = body_angle.mean(axis=1).reshape(-1, 1)
    sigma_a = body_angle.std(axis=1).reshape(-1, 1) + EPS
    c_a = body_angle - mu_a
    skew_a = (c_a**3).mean(axis=1).reshape(-1, 1) / (sigma_a**3)
    kurt_a = (c_a**4).mean(axis=1).reshape(-1, 1) / (sigma_a**4)

    stats = np.concatenate(
        [
            np.repeat(mu_y, T, axis=1)[:, :, np.newaxis],
            np.repeat(sigma_y, T, axis=1)[:, :, np.newaxis],
            np.repeat(skew_y, T, axis=1)[:, :, np.newaxis],
            np.repeat(kurt_y, T, axis=1)[:, :, np.newaxis],
            np.repeat(mu_a, T, axis=1)[:, :, np.newaxis],
            np.repeat(sigma_a, T, axis=1)[:, :, np.newaxis],
            np.repeat(skew_a, T, axis=1)[:, :, np.newaxis],
            np.repeat(kurt_a, T, axis=1)[:, :, np.newaxis],
        ],
        axis=2,
    )

    features = np.concatenate(
        [raw_pose, geom, kinematic, moments, stats],
        axis=2,
        dtype=np.float32,
    )
    return features


def minmax_scale(X: np.ndarray, axis: tuple = (0, 1)) -> np.ndarray:
    """Min-Max scaling to [0,1] along specified axes (default: over N and T)."""
    x_min = X.min(axis=axis, keepdims=True)
    x_max = X.max(axis=axis, keepdims=True)
    span = x_max - x_min + EPS
    return ((X - x_min) / span).astype(np.float32)


def main():
    preferred_device = os.getenv("LE2I_DEVICE", "auto").lower()
    if preferred_device == "cpu":
        device = "cpu"
    elif preferred_device == "cuda":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device} (LE2I_DEVICE={preferred_device})")
    print("Loading YOLO pose model...")
    try:
        model = YOLO("yolo11n-pose.pt")
    except Exception as e:
        raise RuntimeError(
            "Failed to load YOLO model. Ensure yolo11n-pose.pt is available "
            "and ultralytics package is installed."
        ) from e

    # Optional limit for Kaggle low-memory runs: set env LE2I_MAX_VIDEOS=10
    # Default 0 means process all videos.
    max_videos = int(os.getenv("LE2I_MAX_VIDEOS", "0"))

    items = collect_video_items(INPUT_ROOT)
    if not items:
        raise RuntimeError(f"No target videos found under {INPUT_ROOT}")

    if max_videos > 0 and len(items) > max_videos:
        print(f"[WARN] Limiting to {max_videos} videos (of {len(items)}) for low-memory run")
        items = items[:max_videos]

    x_samples: List[np.ndarray] = []
    x_samples_nosmooth: List[np.ndarray] = []
    y_samples: List[np.ndarray] = []
    scene_ids: List[str] = []

    print(f"Found {len(items)} videos to process.")
    for item in tqdm(items, desc="Videos"):
        try:
            pose_seq = extract_pose_sequence(model, item.video_path, device=device)
        except Exception as exc:
            print(f"[WARN] Failed video {item.video_path}: {exc}")
            continue

        status_labels = parse_annotation_file(item.annotation_path)
        pose_seq, status_labels = sync_video_and_labels(pose_seq, status_labels)

        impact = find_impact_frame(status_labels)
        if impact is not None:
            fall_window = extract_fall_sample(pose_seq, impact)
            if fall_window is not None:
                processed = fill_and_smooth_window(fall_window)
                processed_nosmooth = fill_without_smoothing(fall_window)
                if processed is not None and processed_nosmooth is not None:
                    x_samples.append(processed)
                    x_samples_nosmooth.append(processed_nosmooth)
                    y_samples.append(np.array([1], dtype=np.int32))
                    scene_ids.append(item.scene_name)

        adl_windows = collect_adl_windows_from_labels(pose_seq, status_labels)
        for w in adl_windows:
            processed = fill_and_smooth_window(w)
            processed_nosmooth = fill_without_smoothing(w)
            if processed is not None and processed_nosmooth is not None:
                x_samples.append(processed)
                x_samples_nosmooth.append(processed_nosmooth)
                y_samples.append(np.array([0], dtype=np.int32))
                scene_ids.append(item.scene_name)

        del pose_seq
        gc.collect()

    if not x_samples:
        raise RuntimeError("No valid samples created.")

    x_data = np.stack(x_samples, axis=0).astype(np.float32)
    x_data_nosmooth = np.stack(x_samples_nosmooth, axis=0).astype(np.float32)
    y_data = np.stack(y_samples, axis=0).astype(np.int32)

    np.save(OUTPUT_DATA_PROCESSED / "x_data.npy", x_data)
    np.save(OUTPUT_DATA_PROCESSED / "x_data_nosmooth.npy", x_data_nosmooth)
    np.save(OUTPUT_DATA_PROCESSED / "y_data.npy", y_data)
    np.save(OUTPUT_DATA_PROCESSED / "scene_ids.npy", np.asarray(scene_ids, dtype=object))
    print(f"Saved x_data.npy shape={x_data.shape}, y_data.npy shape={y_data.shape}")

    # Advanced features
    features = compute_advanced_features(x_data)
    features_norm = minmax_scale(features)
    features_ns = compute_advanced_features(x_data_nosmooth)
    features_ns_norm = minmax_scale(features_ns)
    np.save(OUTPUT_DATA_FEATURES / "features_final.npy", features_norm)
    np.save(OUTPUT_DATA_FEATURES / "features_final_nosmooth.npy", features_ns_norm)
    print(f"Saved features_final.npy shape={features_norm.shape}")
    print(f"Saved features_final_nosmooth.npy shape={features_ns_norm.shape}")

    del x_data, x_data_nosmooth, y_data, x_samples, x_samples_nosmooth, y_samples, features, features_norm, features_ns, features_ns_norm
    gc.collect()


if __name__ == "__main__":
    main()
