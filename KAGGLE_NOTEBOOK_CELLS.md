# Kaggle Notebook Cells (Copy/Paste)

## Cell 1: Clone repository
```python
GIT_URL = "https://github.com/<username>/<repo>.git"  # change this
REPO = GIT_URL.rstrip("/").split("/")[-1].replace(".git", "")

%cd /kaggle/working
!rm -rf "/kaggle/working/{REPO}"
!git clone "{GIT_URL}"
%cd "/kaggle/working/{REPO}"
print("Cloned:", REPO)
```

## Cell 2: Install dependencies
```python
%cd "/kaggle/working/{REPO}"
!pip -q install -r requirements.txt
```

## Cell 3: Optional env override (if dataset path differs)
```python
import os
# os.environ["LE2I_INPUT_ROOT"] = "/kaggle/input/datasets/tuyenldvn/falldataset-imvia"
os.environ["LE2I_OUTPUT_ROOT"] = "/kaggle/working"
print("LE2I_INPUT_ROOT =", os.getenv("LE2I_INPUT_ROOT", "(auto from config.py)"))
print("LE2I_OUTPUT_ROOT =", os.environ["LE2I_OUTPUT_ROOT"])
```

## Cell 4: Run full pipeline
```python
%cd "/kaggle/working/{REPO}"
!python -m src.kaggle_pipeline --strict
```

## Cell 5: Sanity check outputs
```python
!python -m src.kaggle_sanity --strict
```

## Cell 6: Optional ablation study
```python
!python -m src.eval.ablation_runner
```
