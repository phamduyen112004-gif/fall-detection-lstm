# fall-detection-lstm
A deep learning-based fall detection system using LSTM and pose/keypoint data

## Run on Kaggle

```bash
pip install -r /kaggle/working/fall-detection-lstm/requirements.txt
python -m src.kaggle_pipeline
```

Useful flags:

```bash
python -m src.kaggle_pipeline --skip-train
python -m src.kaggle_pipeline --skip-extract
python -m src.kaggle_pipeline --extract-only
python -m src.kaggle_pipeline --train-only
python -m src.kaggle_pipeline --strict
python -m src.kaggle_pipeline --skip-sanity
```

Optional overrides:

```bash
export LE2I_INPUT_ROOT=/kaggle/input/datasets/tuyenldvn/falldataset-imvia
export LE2I_OUTPUT_ROOT=/kaggle/working
```

Artifacts are saved to:
- `/kaggle/working/data/processed`
- `/kaggle/working/data/features`
- `/kaggle/working/models`
- `/kaggle/working/reports`

## Analysis and Inference

After training the model, run detailed analysis:

```bash
python analysis_and_inference.py --analyze
```

This will:
- Load test data and model
- Compute precision, recall, F1-score per class
- Analyze errors (false positives/negatives)
- Compare with baseline (Logistic Regression on mean features)
- Analyze training history and create detailed plots

For inference on new videos:

```bash
python analysis_and_inference.py --inference --video_path /path/to/video.mp4
```

This will:
- Extract pose keypoints from video
- Compute features using sliding windows
- Run real-time inference with post-processing
- Measure performance (speed, accuracy)
- Save results to CSV
