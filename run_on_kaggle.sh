#!/bin/bash
set -euo pipefail

ROOT_DIR="$(pwd)"

export LE2I_INPUT_ROOT="${LE2I_INPUT_ROOT:-/kaggle/input/datasets/tuyenldvn/falldataset-imvia}"
export LE2I_OUTPUT_ROOT="${LE2I_OUTPUT_ROOT:-/kaggle/working}"

echo "[INFO] ROOT_DIR=${ROOT_DIR}"
echo "[INFO] LE2I_INPUT_ROOT=${LE2I_INPUT_ROOT}"
echo "[INFO] LE2I_OUTPUT_ROOT=${LE2I_OUTPUT_ROOT}"

echo "[INFO] Installing requirements from requirements.txt..."
python -m pip install --quiet -r requirements.txt

echo ""
echo "=========================================="
echo "[STEP] Running Kaggle Pipeline"
echo "=========================================="
echo ""
python -m src.kaggle_pipeline --strict

echo ""
echo "=========================================="
echo "[DONE] run_on_kaggle.sh completed."
echo "=========================================="
echo ""
echo "Results saved to: $LE2I_OUTPUT_ROOT"

