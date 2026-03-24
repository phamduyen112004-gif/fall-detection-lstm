"""Extract per-frame pose from video using YOLO pose estimation."""

import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# Enforce headless mode for Qt backend so Kaggle never opens a GUI window.
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

import cv2
import numpy as np
import torch
from tqdm import tqdm
from ultralytics import YOLO

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import FRAME_HEIGHT, FRAME_WIDTH, N_CHANNELS, N_KEYPOINTS

# Stabilize OpenCV/FFmpeg on Kaggle to reduce native decode crashes.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "threads;1")
cv2.setNumThreads(0)

DEBUG_LOG_PATH = Path(__file__).resolve().parents[2] / "debug-b9631d.log"


def _dbg(hypothesis_id: str, location: str, message: str, data: Optional[dict] = None, run_id: str = "pre-fix") -> None:
    payload = {
        "sessionId": "b9631d",
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data or {},
        "timestamp": int(time.time() * 1000),
    }
    with DEBUG_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")


def extract_pose_sequence(
    model: YOLO,
    video_path: Union[str, Path],
    show_progress: bool = True,
    device: Optional[str] = None,
) -> np.ndarray:
    """
    Extract per-frame pose [T, 17, 3] using YOLO pose.
    Keep only the most confident person for each frame.
    If no detection, fill frame with NaN and 0 conf.
    Spatial scaling: x/320, y/240 -> [0,1].
    """
    video_path = Path(video_path)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    # #region agent log
    _dbg("H1", "pose_extraction.py:54", "extract_pose_sequence_enter", {"video_path": str(video_path), "device": device})
    # #endregion
    cap = cv2.VideoCapture(str(video_path), cv2.CAP_FFMPEG)
    # #region agent log
    _dbg("H2", "pose_extraction.py:58", "cap_open_ffmpeg", {"is_opened": bool(cap.isOpened())})
    # #endregion
    if not cap.isOpened():
        # Fallback backend if FFmpeg backend is unavailable.
        cap = cv2.VideoCapture(str(video_path), cv2.CAP_FFMPEG)
        # #region agent log
        _dbg("H2", "pose_extraction.py:63", "cap_open_fallback", {"is_opened": bool(cap.isOpened())})
        # #endregion
    if not cap.isOpened():
        # #region agent log
        _dbg("H2", "pose_extraction.py:67", "cap_open_failed", {"video_path": str(video_path)})
        # #endregion
        raise RuntimeError(f"Cannot open video: {video_path}")

    frames_pose: list[np.ndarray] = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    pbar = (
        tqdm(total=total_frames if total_frames > 0 else None, desc=f"Pose {video_path.name}")
        if show_progress
        else None
    )
    bad_reads = 0
    frame_idx = 0
    try:
        while True:
            try:
                ret, frame = cap.read()
            except Exception as exc:
                # Some broken AVI files can trigger decoder exceptions.
                if "Header missing" in str(exc):
                    print(f"[WARN] Header missing or broken video {video_path}: {exc}")
                    _dbg("H3", "pose_extraction.py:89", "header_missing", {"video_path": str(video_path), "error": str(exc)})
                    return np.zeros((0, N_KEYPOINTS, N_CHANNELS), dtype=np.float32)
                bad_reads += 1
                if bad_reads >= 3:
                    # #region agent log
                    _dbg("H3", "pose_extraction.py:91", "bad_read_exception_break", {"video_path": str(video_path), "bad_reads": bad_reads, "frame_idx": frame_idx})
                    # #endregion
                    break
                continue
            if not ret:
                # #region agent log
                _dbg("H3", "pose_extraction.py:96", "read_ret_false_break", {"video_path": str(video_path), "frame_idx": frame_idx})
                # #endregion
                break
            if frame is None or frame.size == 0:
                bad_reads += 1
                if bad_reads >= 3:
                    # #region agent log
                    _dbg("H3", "pose_extraction.py:103", "empty_frame_break", {"video_path": str(video_path), "bad_reads": bad_reads, "frame_idx": frame_idx})
                    # #endregion
                    break
                continue
            bad_reads = 0
            frame_idx += 1

            try:
                # Force non-GUI inference mode to avoid display-related crashes on Kaggle.
                result = model.predict(
                    frame,
                    verbose=False,
                    imgsz=320,
                    conf=0.15,
                    device=device,
                    show=False,
                )[0]
            except Exception:
                # Skip problematic frame-level inference and continue stream.
                # #region agent log
                _dbg("H4", "pose_extraction.py:122", "yolo_predict_exception", {"video_path": str(video_path), "frame_idx": frame_idx})
                # #endregion
                continue

            pose = np.full((N_KEYPOINTS, N_CHANNELS), np.nan, dtype=np.float32)
            pose[:, 2] = 0.0

            if result.keypoints is not None and len(result.keypoints) > 0:
                boxes = result.boxes
                if boxes is not None and boxes.conf is not None and len(boxes.conf) > 0:
                    person_idx = int(np.argmax(boxes.conf.cpu().numpy()))
                else:
                    person_idx = 0

                xy = result.keypoints.xy[person_idx].cpu().numpy()
                conf_tensor = result.keypoints.conf
                if conf_tensor is not None:
                    conf = conf_tensor[person_idx].cpu().numpy()
                else:
                    conf = np.ones((N_KEYPOINTS,), dtype=np.float32)

                pose[:, 0] = xy[:, 0] / FRAME_WIDTH
                pose[:, 1] = xy[:, 1] / FRAME_HEIGHT
                pose[:, 2] = conf

            frames_pose.append(pose)
            if pbar:
                pbar.update(1)
            # Periodic cleanup for long videos on limited Kaggle RAM.
            if len(frames_pose) % 200 == 0:
                gc.collect()
                # #region agent log
                _dbg("H5", "pose_extraction.py:151", "periodic_gc", {"video_path": str(video_path), "frames_pose": len(frames_pose), "frame_idx": frame_idx})
                # #endregion
    finally:
        if pbar:
            pbar.close()
        cap.release()
        cv2.destroyAllWindows()
        gc.collect()
        # #region agent log
        _dbg("H1", "pose_extraction.py:159", "extract_pose_sequence_exit", {"video_path": str(video_path), "frames_pose": len(frames_pose), "frame_idx": frame_idx})
        # #endregion

    if not frames_pose:
        return np.zeros((0, N_KEYPOINTS, N_CHANNELS), dtype=np.float32)
    return np.stack(frames_pose, axis=0).astype(np.float32)


def collect_le2i_video_annotation_pairs(root_dir: Union[str, Path]) -> List[Tuple[Path, Optional[Path], str]]:
    """
    Scan nested Le2i structure on Kaggle:
    [Room]/[Room]/Videos/video (n).avi
    [Room]/[Room]/Annotation_files/video (n).txt
    """
    root = Path(root_dir)
    pairs: List[Tuple[Path, Optional[Path], str]] = []
    if not root.exists():
        return pairs

    for videos_dir in root.rglob("Videos"):
        if not videos_dir.is_dir():
            continue
        scene_dir = videos_dir.parent
        scene_name = scene_dir.name
        ann_dir = scene_dir / "Annotation_files"

        ann_map: Dict[str, Path] = {}
        if ann_dir.exists():
            for ap in ann_dir.glob("*.txt"):
                ann_map[ap.stem] = ap

        for vp in sorted(videos_dir.glob("*.avi")):
            pairs.append((vp, ann_map.get(vp.stem), scene_name))

    uniq = {str(v): (v, a, s) for v, a, s in pairs}
    return sorted(uniq.values(), key=lambda x: str(x[0]))


def sync_video_and_labels(pose_seq: np.ndarray, status_labels: Optional[np.ndarray]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Synchronize sequence and labels by trimming extra tail if frame counts mismatch.
    """
    if status_labels is None:
        return pose_seq, None
    valid_len = min(len(pose_seq), len(status_labels))
    return pose_seq[:valid_len], status_labels[:valid_len]
