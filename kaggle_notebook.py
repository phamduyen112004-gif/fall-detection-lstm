"""
Kaggle Notebook Runner - Run Fall Detection LSTM on Kaggle
Execute this in a Kaggle notebook cell
"""

import os
import sys
from pathlib import Path
from subprocess import run, PIPE

# ============================================================================
# Setup Kaggle environment variables
# ============================================================================
print("[INFO] Setting up Kaggle environment...")
os.environ["LE2I_INPUT_ROOT"] = "/kaggle/input/datasets/tuyenldvn/falldataset-imvia"
os.environ["LE2I_OUTPUT_ROOT"] = "/kaggle/working"

INPUT_ROOT = Path("/kaggle/input/datasets/tuyenldvn/falldataset-imvia")
OUTPUT_ROOT = Path("/kaggle/working")

print(f"[INFO] INPUT_ROOT: {INPUT_ROOT}")
print(f"[INFO] OUTPUT_ROOT: {OUTPUT_ROOT}")

# ============================================================================
# Install dependencies
# ============================================================================
print("[INFO] Installing dependencies...")

dependencies = [
    "pip",
    "opencv-python-headless",
    "numpy pandas scikit-learn",
    "torch torchvision torchaudio",
    "mediapipe",
    "pyyaml",
    "scipy",
    "tqdm",
]

for dep in dependencies:
    print(f"[INFO] Installing {dep}...")
    os.system(f"pip install -q {dep}")

# Ensure gym compatibility
print("[INFO] Installing gym with compatibility...")
os.system("pip install -q 'gym<=0.25.2'")

# ============================================================================
# Create output directories
# ============================================================================
print("[INFO] Creating output directories...")
output_dirs = [
    OUTPUT_ROOT,
    OUTPUT_ROOT / "data" / "processed",
    OUTPUT_ROOT / "data" / "features",
    OUTPUT_ROOT / "models",
    OUTPUT_ROOT / "reports",
]

for d in output_dirs:
    d.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Created: {d}")

# ============================================================================
# Run the pipeline
# ============================================================================
print("\n" + "="*80)
print("[STEP] Running Fall Detection LSTM Pipeline on Kaggle")
print("="*80 + "\n")

# Add current directory to Python path so imports work
sys.path.insert(0, "/kaggle/working")

try:
    from src.kaggle_pipeline import main as run_pipeline
    
    # Parse arguments - you can modify these flags
    sys.argv = ["kaggle_notebook.py", "--skip-sanity"]
    
    print("[INFO] Starting pipeline with flags: --skip-sanity")
    run_pipeline()
    
except Exception as e:
    print(f"[ERROR] Pipeline failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*80)
print("[DONE] Pipeline completed successfully!")
print("="*80)
print(f"\nOutput files saved to: {OUTPUT_ROOT}")
print(f"  - Models: {OUTPUT_ROOT / 'models'}")
print(f"  - Features: {OUTPUT_ROOT / 'data' / 'features'}")
print(f"  - Reports: {OUTPUT_ROOT / 'reports'}")
