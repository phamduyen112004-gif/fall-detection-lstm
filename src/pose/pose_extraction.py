"""Extract per-frame pose from video using YOLO pose estimation."""

import gc
import hashlib
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from tqdm import tqdm
from ultralytics import YOLO

from src.config import FRAME_HEIGHT, FRAME_WIDTH, N_CHANNELS, N_KEYPOINTS

# Stabilize OpenCV/FFmpeg on Kaggle to reduce native decode crashes.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "threads;1")
cv2.setNumThreads(0)

TRANSCODE_FIRST = os.getenv("LE2I_TRANSCODE_FIRST", "1") == "1"


def _transcode_video_for_safe_decode(video_path: Path) -> Optional[Path]:
    """
    Transcode problematic AVI to a clean MP4 to avoid OpenCV/FFmpeg native crashes.
    Returns transcoded path if successful, else None.
    """
    cache_dir = Path("/kaggle/working") / "video_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    vid_hash = hashlib.md5(str(video_path).encode("utf-8")).hexdigest()[:12]
    out_path = cache_dir / f"{video_path.stem}_{vid_hash}.mp4"
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path

    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-crf",
        "24",
        str(out_path),
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if res.returncode == 0 and out_path.exists() and out_path.stat().st_size > 0:
            return out_path
    except Exception:
        pass
    return None


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
    read_path = video_path
    if TRANSCODE_FIRST and video_path.suffix.lower() == ".avi":
        fixed_path = _transcode_video_for_safe_decode(video_path)
        if fixed_path is None:
            print(f"[WARN] Skip video (transcode failed): {video_path}")
            return np.zeros((0, N_KEYPOINTS, N_CHANNELS), dtype=np.float32)
        read_path = fixed_path

    cap = cv2.VideoCapture(str(read_path), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        # Fallback backend if FFmpeg backend is unavailable.
        cap = cv2.VideoCapture(str(read_path), cv2.CAP_ANY)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {read_path}")

    frames_pose: list[np.ndarray] = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    pbar = (
        tqdm(total=total_frames if total_frames > 0 else None, desc=f"Pose {video_path.name}")
        if show_progress
        else None
    )
    bad_reads = 0
    try:
        while True:
            try:
                ret, frame = cap.read()
            except Exception as exc:
                err_str = str(exc)
                if "Header missing" in err_str or "corrupted double-linked list" in err_str:
                    print(f"[WARN] Corrupt video skipped: {read_path} ({err_str})")
                    break
                bad_reads += 1
                if bad_reads >= 3:
                    break
                continue

            if not ret:
                break
            if frame is None or frame.size == 0:
                bad_reads += 1
                if bad_reads >= 3:
                    break
                continue

            bad_reads = 0

            try:
                result = model.predict(
                    frame,
                    verbose=False,
                    imgsz=320,
                    conf=0.15,
                    device=device,
                    show=False,
                )[0]
            except Exception as exc:
                print(f"[WARN] YOLO predict failed on {video_path}: {exc}")
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

            if len(frames_pose) % 200 == 0:
                gc.collect()

    except Exception as exc:
        print(f"[WARN] Unexpected processing error {video_path}: {exc}")
        frames_pose = []

    finally:
        if pbar:
            pbar.close()
        cap.release()
        gc.collect()

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
