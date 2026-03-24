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
```

Artifacts are saved to:
- `/kaggle/working/data/processed`
- `/kaggle/working/data/features`
- `/kaggle/working/models`
- `/kaggle/working/reports`
