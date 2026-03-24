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

Sanity check after pipeline run:

```bash
python -m src.kaggle_sanity
python -m src.kaggle_sanity --strict
```
