#!/bin/bash
set -euo pipefail

# Root folder of cloned repo
ROOT_DIR="$(pwd)"

# Ensure Kaggle path variables (optionally override)
export LE2I_INPUT_ROOT="${LE2I_INPUT_ROOT:-/kaggle/input/datasets/tuyenldvn/falldataset-imvia}"
export LE2I_OUTPUT_ROOT="${LE2I_OUTPUT_ROOT:-/kaggle/working}"

echo "[INFO] ROOT_DIR=${ROOT_DIR}"
echo "[INFO] LE2I_INPUT_ROOT=${LE2I_INPUT_ROOT}"
echo "[INFO] LE2I_OUTPUT_ROOT=${LE2I_OUTPUT_ROOT}"

# Install dependencies
echo "[INFO] Upgrading pip..."
python -m pip install --quiet --upgrade pip

# Ensure no conflicting non-headless OpenCV builds are installed
echo "[INFO] Removing conflicting OpenCV versions..."
python -m pip uninstall -y opencv-python opencv-contrib-python || true

# Install requirements
echo "[INFO] Installing requirements from requirements.txt..."
python -m pip install --quiet -r requirements.txt

# Ensure only headless OpenCV is installed (required for Kaggle)
echo "[INFO] Installing opencv-python-headless..."
python -m pip install --quiet --upgrade opencv-python-headless

# Lock gym version to avoid dependency conflicts
echo "[INFO] Installing gym compatibility..."
python -m pip install --quiet "gym<=0.25.2"

# Create output dirs
echo "[INFO] Creating output directories..."
mkdir -p "$LE2I_OUTPUT_ROOT" 
mkdir -p "$LE2I_OUTPUT_ROOT/data/processed" 
mkdir -p "$LE2I_OUTPUT_ROOT/data/features" 
mkdir -p "$LE2I_OUTPUT_ROOT/models" 
mkdir -p "$LE2I_OUTPUT_ROOT/reports"

echo "[INFO] Output directories created:"
ls -la "$LE2I_OUTPUT_ROOT"

# Run Kaggle pipeline
echo ""
echo "=========================================="
echo "[STEP] Running Kaggle Pipeline"
echo "=========================================="
echo ""
python -m src.kaggle_pipeline --skip-sanity

echo ""
echo "=========================================="
echo "[DONE] run_on_kaggle.sh completed."
echo "=========================================="
echo ""
echo "Results saved to: $LE2I_OUTPUT_ROOT"

