"""Shared constants and Kaggle-aware paths for Le2i pipeline."""

import os
from pathlib import Path
from typing import Iterable

# Le2i video metadata
FPS = 25
FRAME_WIDTH = 320.0
FRAME_HEIGHT = 240.0

# Sequence config
SEQ_LEN = 75
IMPACT_INDEX = 40
N_KEYPOINTS = 17
N_CHANNELS = 3  # x, y, conf

# Keypoint indices (COCO/YOLO 17-keypoint)
IDX_NOSE = 0
IDX_LEFT_SHOULDER, IDX_RIGHT_SHOULDER = 5, 6
IDX_LEFT_HIP, IDX_RIGHT_HIP = 11, 12
IDX_LEFT_KNEE, IDX_RIGHT_KNEE = 13, 14
IDX_LEFT_ANKLE, IDX_RIGHT_ANKLE = 15, 16

# CoG keypoints: nose, shoulders, hips, knees
COG_INDICES = (0, 5, 6, 11, 12, 13, 14)

# Missing-data & smoothing
MAX_CONSECUTIVE_MISSING = 10
SAVGOL_WINDOW = 5
SAVGOL_POLYORDER = 2
ADL_STRIDE = 100

def _first_existing_path(candidates: Iterable[Path]) -> Path:
    for p in candidates:
        if p.exists():
            return p
    return next(iter(candidates))


# Global path management (Kaggle-first, with fallback auto-detection)
_env_input_root = os.getenv("LE2I_INPUT_ROOT", "").strip()
if _env_input_root:
    INPUT_ROOT = Path(_env_input_root)
else:
    candidates = [
        Path("/kaggle/input/datasets/tuyenldvn/falldataset-imvia"),
        Path("/kaggle/input/le2i-fall-dataset"),
        Path("/kaggle/input/falldataset-imvia"),
    ]
    existing = [p for p in candidates if p.exists()]
    if existing:
        INPUT_ROOT = existing[0]
    else:
        # scanning fallback for any subfolder containing key name
        parent = Path("/kaggle/input")
        INPUT_ROOT = None
        if parent.exists():
            for candidate in parent.rglob("*falldataset*imvia*"):
                if candidate.is_dir():
                    INPUT_ROOT = candidate
                    break
        if INPUT_ROOT is None:
            INPUT_ROOT = candidates[0]

OUTPUT_ROOT = Path(os.getenv("LE2I_OUTPUT_ROOT", "/kaggle/working"))

# Output sub-directories
OUTPUT_DATA_PROCESSED = OUTPUT_ROOT / "data" / "processed"
OUTPUT_DATA_FEATURES = OUTPUT_ROOT / "data" / "features"
OUTPUT_MODELS = OUTPUT_ROOT / "models"
OUTPUT_REPORTS = OUTPUT_ROOT / "reports"

for p in (OUTPUT_ROOT, OUTPUT_DATA_PROCESSED, OUTPUT_DATA_FEATURES, OUTPUT_MODELS, OUTPUT_REPORTS):
    os.makedirs(p, exist_ok=True)
