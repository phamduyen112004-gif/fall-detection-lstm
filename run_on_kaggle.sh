#!/bin/bash
set -euo pipefail

# Root folder of cloned repo
ROOT_DIR="$(pwd)"

# Ensure Kaggle path variables (optionally override)
export LE2I_INPUT_ROOT="${LE2I_INPUT_ROOT:-/kaggle/input/falldataset-imvia}"
export LE2I_OUTPUT_ROOT="${LE2I_OUTPUT_ROOT:-/kaggle/working}"

echo "[INFO] ROOT_DIR=${ROOT_DIR}"
echo "[INFO] LE2I_INPUT_ROOT=${LE2I_INPUT_ROOT}"
echo "[INFO] LE2I_OUTPUT_ROOT=${LE2I_OUTPUT_ROOT}"

# Install dependencies
python -m pip install --upgrade pip
# Ensure no conflicting non-headless OpenCV builds are installed.
python -m pip uninstall -y opencv-python opencv-contrib-python || true
python -m pip install -r requirements.txt
# Ensure only headless OpenCV is installed. This is safe for Kaggle and avoids GUI backend.
python -m pip install --upgrade opencv-python-headless

# Create output dirs
mkdir -p "$LE2I_OUTPUT_ROOT" 
mkdir -p "$LE2I_OUTPUT_ROOT/data/processed" 
mkdir -p "$LE2I_OUTPUT_ROOT/data/features" 
mkdir -p "$LE2I_OUTPUT_ROOT/models" 
mkdir -p "$LE2I_OUTPUT_ROOT/reports"

# Run Kaggle pipeline
python -m src.kaggle_pipeline --skip-sanity

# Optional: train-only (uncomment if you want to skip extraction in reruns)
# python -m src.kaggle_pipeline --skip-extract

echo "[DONE] run_on_kaggle.sh completed."
