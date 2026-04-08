# fall-detection-lstm

AI-based fall detection system using pose sequences + BiLSTM/Attention.

## Quick Start (Kaggle)

1. Clone this repository to `/kaggle/working`.
2. Add Le2i dataset in Kaggle Input.
3. Run:

```bash
pip install -r /kaggle/working/fall-detection-lstm/requirements.txt
python -m src.kaggle_pipeline --strict
```

## Pipeline Commands

```bash
# full pipeline
python -m src.kaggle_pipeline --strict

# only extract + feature engineering
python -m src.kaggle_pipeline --extract-only

# only training (requires extracted features)
python -m src.kaggle_pipeline --train-only

# final artifact checks
python -m src.kaggle_sanity --strict

# ablation study report
python -m src.eval.ablation_runner
```

## Optional Environment Variables

```bash
export LE2I_INPUT_ROOT=/kaggle/input/datasets/tuyenldvn/falldataset-imvia
export LE2I_OUTPUT_ROOT=/kaggle/working
export LE2I_RUN_SCENE_CV=1
```

## Output Artifacts

- `/kaggle/working/data/processed`
- `/kaggle/working/data/features`
- `/kaggle/working/models`
- `/kaggle/working/reports`
