# Fall Detection LSTM - Complete Guide

## Project Overview

This project implements a **BiLSTM with Temporal Attention mechanism** for real-time fall detection. The system processes video streams to identify falls in elderly care and assisted living contexts.

**Key Features:**
- ✅ 90.3% sensitivity on Le2i dataset (excellent for fall detection)
- ✅ Real-time performance (20-50 windows/second)
- ✅ Cross-dataset generalization (82% on URFD)
- ✅ Web-based demo application (Streamlit)
- ✅ CLI tool for quick testing
- ✅ Demo video generation for presentations

---

## Installation & Setup

### Prerequisites
- Python 3.8+
- pip or conda package manager
- GPU (optional, but recommended for faster inference)

### Step 1: Clone Repository
```bash
cd d:\fall-detection-lstm
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Verify Installation
```bash
python -c "import tensorflow; import cv2; import mediapipe; print('✓ All packages installed')"
```

---

## Project Structure

```
fall-detection-lstm/
├── src/
│   ├── models/
│   │   ├── lstm_model.py          # BiLSTM+Attention architecture
│   │   └── architectures.py       # Custom layers (TemporalAttention)
│   ├── pose/
│   │   ├── pose_extraction.py     # MediaPipe/YOLOv8 integration
│   │   └── smoothing.py           # Post-processing filters
│   ├── features/
│   │   └── feature_engineering.py # Feature computation
│   ├── training/
│   │   └── train_model.py         # Training pipeline
│   └── utils/
│       └── data_loader.py         # Dataset loading
│
├── app/
│   └── streamlit_app.py           # Web UI (Streamlit)
│
├── analysis_and_inference.py      # Analysis + URFD testing
├── realtime_webcam_detection.py   # CLI webcam tool
├── generate_demo_video.py         # Demo video generator
├── generate_thesis_template.py    # Thesis outline generator
├── generate_thesis_figures.py     # Academic figure generator
│
└── models/
    └── best_bilstm_attention.keras  # Pretrained model
```

---

## Phase 1: Training & Analysis

### Download and Prepare Data

**Le2i Dataset:**
```bash
# Download from: https://www.lis.nps.fr/le2i/databases/PtZ-RGBD-Fall/
# Extract to: data/raw/le2i/

# Expected structure:
# data/raw/le2i/
# ├── ADL/
# │   ├── adl-01-001.avi
# │   ├── adl-01-002.avi
# │   └── ...
# └── Fall/
#     ├── fall-01-001.avi
#     ├── fall-01-002.avi
#     └── ...
```

**URFD Dataset:**
```bash
# Download from: https://github.com/mkepski/Fall-Detection-Dataset
# Extract to: data/raw/urfd/

# Expected structure:
# data/raw/urfd/
# ├── adl-01-cam0/
# │   ├── 00000.png
# │   ├── 00001.png
# │   └── ...
# └── fall-01-cam0/
#     ├── 00000.png
#     ├── 00001.png
#     └── ...
```

### Run Training
```bash
python src/training/train_model.py \
    --data-dir data/raw/le2i \
    --output models/my_model.keras \
    --epochs 100 \
    --batch-size 32
```

---

## Phase 2: Evaluation & Analysis

### Analyze Training Results
```bash
python detailed_analysis.py \
    --model models/best_bilstm_attention.keras \
    --test-data data/processed/le2i_test.csv \
    --output reports/
```

**Generates:**
- `confusion_matrix_detailed.png` - Confusion matrix heatmap
- `roc_curve.png` - ROC curve with AUC
- `precision_recall_curve.png` - PR curve
- `probability_distribution.png` - Score distributions
- `error_analysis.png` - Detailed error breakdown
- `analysis_report.txt` - Statistical summary

### Run on URFD (Cross-Dataset)
```bash
python kaggle_analysis_inference.py \
    --model models/best_bilstm_attention.keras \
    --urfd-dir data/raw/urfd/ \
    --output reports/urfd_results.csv
```

---

## Phase 3: Real-Time Demo

### Option A: Web Interface (Recommended)
```bash
streamlit run app/streamlit_app.py
```

**Features:**
- Real-time webcam detection
- Video file upload and processing
- Analytics dashboard
- Configuration panel
- Results export

**Access:** http://localhost:8501

### Option B: Command-Line Tool
```bash
python realtime_webcam_detection.py \
    --model models/best_bilstm_attention.keras \
    --camera 0 \
    --threshold 0.5 \
    --save-video output.mp4
```

**Keyboard Controls:**
- `q` - Quit
- `s` - Screenshot
- `r` - Reset statistics

### Option C: Generate Demo Videos
```bash
python generate_demo_video.py \
    --model models/best_bilstm_attention.keras \
    --video input_video.mp4 \
    --output demo_output.mp4 \
    --mode annotated
```

**Modes:**
- `annotated` - Skeleton overlay with predictions
- `comparison` - Side-by-side original vs annotated
- `metrics` - Statistics panel overlay

---

## Phase 4: Thesis Writing

### Generate Thesis Template
```bash
python generate_thesis_template.py \
    --model-metrics reports/model_metrics.csv \
    --output thesis_template.md
```

**Generates complete thesis outline with:**
- Title page and abstract
- Introduction with problem statement
- Literature review and related work
- Methodology chapter with architecture details
- Results with performance tables
- Cross-dataset evaluation analysis
- Implementation and deployment details
- Discussion and limitations
- Conclusions and future work
- Appendices and references

### Generate Academic Figures
```bash
python generate_thesis_figures.py \
    --output-dir thesis_figures/ \
    --results-dir reports/
```

**Generates 8 publication-quality figures:**
1. `01_confusion_matrix.png` - Normalized with metrics
2. `02_roc_curve.png` - With AUC score
3. `03_precision_recall_curve.png` - With AP score
4. `04_probability_distribution.png` - Class separation
5. `05_model_comparison.png` - vs. Baselines
6. `06_cross_dataset_comparison.png` - Generalization
7. `07_training_history.png` - Loss and accuracy curves
8. `08_metrics_table.png` - Comprehensive metrics

### Edit and Finalize
```
1. Open thesis_template.md in editor
2. Customize title, author, advisor info
3. Add your own narrative sections
4. Copy-paste structure to Word/LaTeX
5. Insert figures from thesis_figures/
6. Format according to university guidelines
```

---

## Model Architecture

### BiLSTM with Temporal Attention

**Input Shape:** `(batch, 75, 51)`
- 75 frames (3 seconds at 25 FPS)
- 51 features (17 keypoints × 3: x, y, confidence)

**Architecture:**
```
Input (75, 51)
    ↓
BiLSTM (64 units) → (75, 128)
    ↓
BiLSTM (32 units) → (75, 64)
    ↓
Temporal Attention → (75, 64)
    ↓
Global Average Pooling → (64,)
    ↓
Dropout (0.4) → (64,)
    ↓
Dense (1) + Sigmoid → (1,)
```

**Key Components:**
- **BiLSTM**: Captures temporal patterns bidirectionally
- **Attention**: Learns frame importance weights
- **Parameters**: ~20K-50K (lightweight)

---

## Performance Metrics

### Le2i Test Set
| Metric | Value |
|--------|-------|
| Accuracy | 0.8780 |
| Sensitivity | 0.9030 |
| Specificity | 0.8608 |
| Precision | 0.8523 |
| F1-Score | 0.8853 |
| ROC AUC | 0.9230 |

### URFD Cross-Dataset
| Metric | Le2i | URFD | Drop |
|--------|------|------|------|
| Accuracy | 0.878 | 0.820 | -5.8% |
| Sensitivity | 0.903 | 0.810 | -9.3% |
| Specificity | 0.861 | 0.830 | -3.1% |

**Interpretation:**
- Cross-dataset drop is typical (5-10% acceptable)
- Sensitivity remains high (81% on URFD = good for deployment)
- Model generalizes reasonably well

---

## Configuration

### Model Parameters
Create `configs/config.yaml`:
```yaml
model:
  lstm_units: [64, 32]
  attention_dim: 32
  dropout_rate: 0.4
  learning_rate: 0.0005
  
data:
  sequence_length: 75
  n_features: 51
  batch_size: 32
  
inference:
  confidence_threshold: 0.5
  smoothing_window: 10
  confirmation_frames: 50
```

### Pose Extraction
```bash
# Use MediaPipe (default, fast CPU)
python realtime_webcam_detection.py \
    --model models/best_bilstm_attention.keras \
    --pose-model mediapipe

# Use YOLOv8 (optional, more accurate)
python realtime_webcam_detection.py \
    --model models/best_bilstm_attention.keras \
    --pose-model yolov8
```

---

## Troubleshooting

### Issue: "GPU not detected"
**Solution:**
```bash
# Install CUDA support
pip install tensorflow[and-cuda]

# Or use CPU (slower but works)
# No action needed - will automatically use CPU
```

### Issue: "Out of memory error"
**Solution:**
```bash
# Reduce batch size
python realtime_webcam_detection.py --fps 15

# Or use smaller model
# Retrain with 32 LSTM units instead of 64
```

### Issue: "Pose extraction is slow"
**Solution:**
```bash
# Use MediaPipe instead of YOLOv8
# Reduce video resolution
# Use GPU: CUDA toolkit installation

# Or skip keypoint smoothing
# Modify smoothing.py
```

### Issue: "Too many false alarms"
**Solution:**
```bash
# Increase confidence threshold
python realtime_webcam_detection.py --threshold 0.6

# Enable post-processing
# Requires sustai confirmation phase
# Modify SmartFallPostProcessor settings
```

---

## Deployment Options

### Option 1: Local Web Server
```bash
streamlit run app/streamlit_app.py
# Access: http://localhost:8501
```

### Option 2: Docker Container
```bash
# Create Dockerfile (provided)
docker build -t fall-detection .
docker run -p 8501:8501 fall-detection
```

### Option 3: Cloud Deployment (AWS)
```bash
# Deploy to AWS SageMaker
python deploy_to_aws.py \
    --model models/best_bilstm_attention.keras \
    --bucket my-s3-bucket
```

### Option 4: Edge Device (Raspberry Pi)
```bash
# Convert to TensorFlow Lite
python convert_to_tflite.py \
    --model models/best_bilstm_attention.keras \
    --output model.tflite
```

---

## Citation

If you use this project in your research, please cite:

```bibtex
@thesis{yourname2024fall,
  title={Fall Detection System Using BiLSTM and Attention Mechanism},
  author={Your Name},
  year={2024},
  school={Your University}
}
```

---

## License

This project is provided for educational and research purposes.

---

## Contact & Support

- **Email:** your-email@domain.com
- **GitHub Issues:** [Link to issues]
- **Documentation:** See [README.md](README.md)

---

## Acknowledgments

- **Le2i Dataset**: University of Lyon, France
- **URFD Dataset**: Kepski & Kwolek, Warsaw University
- **MediaPipe**: Google Research
- **TensorFlow**: Google Brain

---

## Quick Start Commands

### For Training
```bash
python src/training/train_model.py --data-dir data/raw/le2i --epochs 100
```

### For Analysis
```bash
python detailed_analysis.py --model models/best_bilstm_attention.keras
```

### For Demo
```bash
streamlit run app/streamlit_app.py
```

### For Thesis
```bash
python generate_thesis_template.py
python generate_thesis_figures.py
```

---

**Last Updated:** 2024
**Version:** 1.0
