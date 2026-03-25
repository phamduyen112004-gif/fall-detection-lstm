# Hướng dẫn chạy trên Kaggle

## Cách 1: Chạy từ Kaggle Notebook (Khuyên dùng)

1. Tạo Kaggle Notebook mới
2. Copy code từ `kaggle_notebook.py` hoặc dán đoạn code dưới đây:

```python
!git clone https://github.com/your-username/fall-detection-lstm.git
cd fall-detection-lstm

import os
os.environ["LE2I_INPUT_ROOT"] = "/kaggle/input/datasets/tuyenldvn/falldataset-imvia"
os.environ["LE2I_OUTPUT_ROOT"] = "/kaggle/working"

# Install dependencies
!pip install -q opencv-python-headless numpy pandas scikit-learn mediapipe pyyaml scipy tqdm
!pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
!pip install -q 'gym<=0.25.2'

# Create directories
import sys
from pathlib import Path

output_dirs = [
    Path("/kaggle/working"),
    Path("/kaggle/working/data/processed"),
    Path("/kaggle/working/data/features"),
    Path("/kaggle/working/models"),
    Path("/kaggle/working/reports"),
]

for d in output_dirs:
    d.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, "/kaggle/working")

# Run pipeline
from src.kaggle_pipeline import main
sys.argv = ["notebook", "--skip-sanity"]
main()
```

## Cách 2: Chạy từ Bash Script

Nếu bạn upload repo lên Kaggle Notebook:

```bash
#!/bin/bash
cd /kaggle/working/fall-detection-lstm
bash run_on_kaggle.sh
```

## Cách 3: Chạy file đã chuẩn bị

```bash
cd /kaggle/working
python kaggle_notebook.py
```

## Tùy chọn chạy Pipeline

Thêm các flag vào lệnh chạy:

- `--skip-extract`: Bỏ qua bước trích xuất pose (chỉ huấn luyện)
- `--skip-train`: Bỏ qua bước huấn luyện (chỉ trích xuất)
- `--extract-only`: Chỉ trích xuất features
- `--train-only`: Chỉ huấn luyện mô hình
- `--skip-sanity`: Bỏ qua sanity checks
- `--strict`: Fail nếu sanity check không đạo

Ví dụ:
```python
sys.argv = ["notebook", "--train-only", "--skip-sanity"]
main()
```

## Cách setup trên Kaggle

1. **Upload dataset**: Attach "falldataset-imvia" vào Kaggle Input
2. **Clone repo** (nếu cần): 
   ```bash
   !git clone https://github.com/your-repo/fall-detection-lstm.git
   %cd fall-detection-lstm
   ```
3. **Chạy code**: Dùng một trong 3 cách ở trên

## Environment Variables

Có thể custom paths bằng environment variables:

```bash
export LE2I_INPUT_ROOT="/kaggle/input/datasets/tuyenldvn/falldataset-imvia"
export LE2I_OUTPUT_ROOT="/kaggle/working"
```

## Output

Sau khi chạy xong, kết quả sẽ lưu tại:
- `/kaggle/working/models/` - Mô hình đã huấn luyện
- `/kaggle/working/data/features/` - Features đã trích xuất
- `/kaggle/working/reports/` - Reports và metrics
