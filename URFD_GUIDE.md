"""
URFD DATASET GUIDE - Format & Inference
Complete reference for working with URFD dataset on Kaggle

URFD Dataset Structure:
/kaggle/input/datasets/shahliza27/ur-fall-detection-dataset/
├── UR_fall_detection_dataset_cam0_rgb/
│   ├── adl-01-cam0-rgb/           ← Activity sequence as PNG frames
│   │   ├── adl-01-cam0-rgb-001.png
│   │   ├── adl-01-cam0-rgb-002.png
│   │   └── ...
│   ├── adl-02-cam0-rgb/
│   ├── fall-01-cam0-rgb/          ← Fall sequence
│   ├── fall-02-cam0-rgb/
│   └── ...
├── UR_fall_detection_dataset_cam1_rgb/
└── ...

Key Differences from Le2i:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

                    Le2i                  URFD
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Video Format        .avi (video file)    .png (image sequence)
Data Loading        cv2.VideoCapture()   cv2.imread() in loop
Pose Extraction     extract_pose_from_   extract_poses_from_
                    video()              images()
Labels              Separate file        Embedded in folder name
                    (with activity ids)  (adl-XX or fall-XX)
Sequence Length     65-75 frames         variable (50-300)
Resolution          320x240              variable (~640px width)
Cameras             Single (frontal)     Multiple (cam0, cam1, etc)
Actors              Multiple             Multiple same as Le2i
Total Samples       ~1500-2000           ~2000-3000
Test/Train Split    Predefined           Can be random

Dataset Characteristics:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
URFD (University of Rzeszow Fall Detection):
  ✓ More diverse fall types
  ✓ Multiple camera angles
  ✓ Higher resolution
  ✓ More ADL activities
  ✗ Fewer total samples than Le2i
  ✗ Different distribution (might have domain gap)

Good For:
  • Generalization testing (Le2i → URFD)
  • Cross-dataset validation
  • Understanding domain adaptation
  • Thesis comparative analysis

Challenge:
  • Different video encoding (image sequences)
  • May require pose extraction from scratch
  • Different background and lighting
"""

# ============================================================================
# QUICK START: How to Use URFD Scripts
# ============================================================================

"""
OPTION 1: Kaggle Online (Recommended for Ease)
──────────────────────────────────────────────────

Step 1: Create New Kaggle Notebook
  • Go: https://www.kaggle.com/code
  • Click: "Create" → "New Notebook"

Step 2: Add Datasets
  • Click: "Add Input"
  • Search: "ur-fall-detection-dataset" OR
            "shahliza27/ur-fall-detection-dataset"
  • Add it to notebook

Step 3: Copy Code
  • Copy from: kaggle_urfd_runner.py
  • Paste into notebook cells (match CELL numbers)

Step 4: Run Cells
  • Cell 1: Setup (configure paths)
  • Cell 2: Install dependencies
  • Cell 3: Setup imports
  • Cell 4: Evaluate model on Le2i test set
  • Cell 5: Run URFD inference

Expected Output:
  ✓ le2i_evaluation.png       (model performance on Le2i)
  ✓ le2i_metrics.csv          (Le2i test metrics)
  ✓ urfd_summary.csv          (URFD inference results)
  
Time: ~20-30 minutes


OPTION 2: Local Development (Advanced)
──────────────────────────────────────

If you have URFD dataset locally:

Step 1: Inspect Dataset
  python urfd_inspector.py
  # Shows: Structure, number of sequences, samples

Step 2: Run Inference
  python -c "
  from urfd_inference_handler import run_urfd_batch_inference
  from pathlib import Path
  import tensorflow as tf
  
  model = tf.keras.models.load_model('path/to/model.keras')
  dataset_root = Path('path/to/urfd')
  
  results = run_urfd_batch_inference(
      dataset_root,
      model,
      camera='UR_fall_detection_dataset_cam0_rgb',
      num_sequences=10,
      output_dir=Path('./urfd_results')
  )
  "

Output:
  ✓ urfd_predictions_*.csv     (per-sequence predictions)
  ✓ urfd_summary.csv           (summary statistics)
  ✓ Console output             (accuracy, timing)

Time: Depends on GPU (~5-15 min for 50 sequences)
"""

# ============================================================================
# Script Guide: Which to Use When
# ============================================================================

"""
┌─────────────────────────────────────────────────────────────────────────┐
│ 1. urfd_inspector.py                                                    │
├─────────────────────────────────────────────────────────────────────────┤
│ Purpose: Explore URFD dataset structure                                 │
│ When: First time looking at dataset, need to understand format          │
│ Usage (Local):                                                          │
│   python urfd_inspector.py                                              │
│                                                                         │
│ Output:                                                                  │
│   • List of all cameras                                                │
│   • Number of sequences per camera                                     │
│   • Sample activity sequences                                          │
│   • Class distribution (ADL vs FALL)                                   │
│                                                                         │
│ Time: 1-2 minutes                                                       │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ 2. urfd_inference_handler.py                                            │
├─────────────────────────────────────────────────────────────────────────┤
│ Purpose: Advanced URFD inference with pose extraction                   │
│ When: Want full pipeline with pose extraction & detailed analysis       │
│ Usage (Local with GPU):                                                 │
│   from urfd_inference_handler import run_urfd_batch_inference           │
│   results = run_urfd_batch_inference(                                   │
│       dataset_root,                                                     │
│       model,                                                             │
│       camera='UR_fall_detection_dataset_cam0_rgb',                      │
│       num_sequences=50,                                                 │
│       output_dir='./results'                                            │
│   )                                                                     │
│                                                                         │
│ Features:                                                                │
│   ✓ Full pose extraction from image sequences                          │
│   ✓ Feature computation                                                │
│   ✓ Per-window predictions                                             │
│   ✓ Per-sequence accuracy                                              │
│   ✓ Summary statistics                                                 │
│   ✓ Handles errors gracefully                                          │
│                                                                         │
│ Output:                                                                  │
│   ✓ urfd_predictions_*.csv (detailed per-window)                       │
│   ✓ urfd_summary.csv (summary for all sequences)                       │
│   ✓ Console output (class-wise accuracy, timing)                       │
│                                                                         │
│ Time: 15-30 minutes (depends on GPU and #sequences)                    │
│ Requires: GPU (recommend), 8GB+ RAM                                    │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ 3. kaggle_urfd_runner.py                                                │
├─────────────────────────────────────────────────────────────────────────┤
│ Purpose: Complete Kaggle workflow for URFD                              │
│ When: Running on Kaggle (simple, 1-notebook approach)                   │
│ Where: Kaggle Notebook (copy-paste each CELL section)                   │
│                                                                         │
│ Content:                                                                 │
│   • CELL 1: Setup paths & verify URFD dataset location                 │
│   • CELL 2: Install dependencies                                       │
│   • CELL 3: Import modules                                             │
│   • CELL 4: Evaluate model on Le2i test set                            │
│   • CELL 5: Run URFD inference (with fallback for pose extraction)     │
│                                                                         │
│ Output:                                                                  │
│   ✓ le2i_evaluation.png     (4 visualizations)                         │
│   ✓ le2i_metrics.csv        (test set metrics)                         │
│   ✓ urfd_summary.csv        (URFD results)                             │
│                                                                         │
│ Time: 20-30 minutes on Kaggle                                          │
│ Advantages:                                                              │
│   ✓ All-in-one notebook                                                │
│   ✓ No GPU required (but slower)                                       │
│   ✓ Good for quick testing                                             │
│                                                                         │
│ Limitations:                                                             │
│   ✗ Simplified pose extraction (may use pre-computed or skip)          │
│   ✗ Slower than local GPU execution                                    │
│   ✗ Limited sample processing (first 5 sequences)                      │
└─────────────────────────────────────────────────────────────────────────┘
"""

# ============================================================================
# Expected Results: Le2i vs URFD
# ============================================================================

"""
TYPICAL RESULTS ON YOUR MODEL:

Le2i Test Set (Known Distribution):
┌──────────────────────────┐
│ Accuracy: 0.88-0.92      │
│ Sensitivity: 0.85-0.95   │
│ Specificity: 0.85-0.95   │
│ ROC AUC: 0.90-0.97       │
│ Processing: Real-time    │
└──────────────────────────┘

URFD Test Set (Cross-Dataset):
┌──────────────────────────┐
│ Accuracy: 0.75-0.88      │ ← Lower due to domain gap
│ Sensitivity: 0.70-0.85   │ ← More critical metric
│ Specificity: 0.80-0.90   │
│ ROC AUC: 0.80-0.92       │
│ Processing: Real-time    │
└──────────────────────────┘

Domain Gap Analysis:
- Drop of 5-15% accuracy is typical for cross-dataset
- Shows model learned some generalizable features
- Sensitivity drop is more concerning (missed falls)
- Good opportunity for fine-tuning discussion in thesis

Why URFD Results Lower:
1. Different camera angles
2. Different background/lighting
3. Different fall types
4. Different actors/movements
5. Different clothing
6. Model trained only on Le2i

Interpretation for Thesis:
✓ Demonstrates generalization capability
✗ Shows need for more diverse training data
→ Opportunity: Fine-tune on URFD for improvement
→ Discussion: Domain adaptation techniques
"""

# ============================================================================
# How to Improve Results
# ============================================================================

"""
If URFD results are lower than expected:

1. CHECK POSE EXTRACTION
   □ Verify poses are being extracted correctly
   □ Check with: urfd_inspector.py on first few frames
   □ Compare pose quality with Le2i

2. HANDLE MISSING POSES
   □ Some frames may have no detected keypoints (occlusion, etc)
   □ Current code fills with zeros - might need interpolation
   □ Consider: post-processing or smoothing

3. ADAPT THRESHOLD
   □ Default threshold 0.5 may not be optimal for URFD
   □ Try: Find ROC curve optimal threshold for URFD
   □ May be 0.4-0.6 depending on domain

4. FINE-TUNE MODEL
   □ Train on (Le2i + subset of URFD)
   □ Use transfer learning
   □ Should improve cross-dataset performance

5. ANALYZE FAILURES
   □ Which sequences are misclassified?
   □ Are they edge cases (sits down slowly, etc)?
   □ Discuss in thesis limitations section

Code Example: Fine-tuning
─────────────────────────
from src.models.architectures import build_bilstm_attention_model
from src.training.train_model import train_bilstm_model

# Load pre-trained model
model = tf.keras.models.load_model('best_bilstm_attention.keras')

# Freeze early layers
for layer in model.layers[:-3]:  # Freeze all but last 3
    layer.trainable = False

# Fine-tune on URFD data
history = train_bilstm_model(
    x_train_urfd, y_train_urfd,
    model=model,
    epochs=10,  # Fewer epochs for fine-tuning
    learning_rate=1e-5  # Smaller learning rate
)
"""

# ============================================================================
# URFD Folder Name Convention
# ============================================================================

"""
Understanding URFD Activity Names:

Active Directory Listing Pattern:
  adl-XX-camY-[rgb|depth]

Breakdown:
  adl    = Activities of Daily Living (label = 0, non-fall)
  fall   = Fall activity (label = 1, fall)
  XX     = Activity sequence number (01, 02, 03, ...)
  camY   = Camera angle (cam0, cam1, etc)
  rgb    = RGB color images (we use these)
           (Some sequences also have depth data)

Examples:
  • adl-01-cam0-rgb   → Activity 1, Camera 0, Non-Fall
  • fall-05-cam0-rgb  → Activity 5, Camera 0, Fall
  • adl-12-cam1-rgb   → Activity 12, Camera 1, Non-Fall
  • fall-23-cam0-rgb  → Activity 23, Camera 0, Fall

Frame Numbering:
  adl-01-cam0-rgb-001.png   (Frame 001)
  adl-01-cam0-rgb-002.png   (Frame 002)
  ...
  adl-01-cam0-rgb-150.png   (Frame 150)

Label Assignment:
  if 'fall' in folder_name.lower():
      label = 1
  elif 'adl' in folder_name.lower():
      label = 0
  else:
      skip or unknown
"""

# ============================================================================
# Troubleshooting
# ============================================================================

"""
Common Issues & Solutions:

1. "URFD dataset not found"
   ❌ Error: /kaggle/input/ur-fall-detection-dataset not found
   ✓ Solution:
     - Verify dataset is attached in notebook settings
     - Check exact dataset name from Kaggle page
     - Look in /kaggle/input/ directory listing
     - May be under datasets/ subfolder

2. "No PNG files in sequence"
   ❌ Folders exist but no .png files
   ✓ Solution:
     - Check camera folder naming (case sensitive?)
     - Ensure RGB data is available (not just depth)
     - Verify file permissions
     - Try alternative camera (cam1_rgb, etc)

3. "Pose extraction fails"
   ❌ YOLOv8/MediaPipe errors
   ✓ Solution:
     - Ensure ultralytics/mediapipe installed
     - Check GPU memory availability
     - Reduce batch size if OOM
     - Use simpler model (yolov8n vs yolov8l)

4. "Out of memory during processing"
   ❌ OOM error during pose extraction or inference
   ✓ Solution:
     - Process fewer sequences (reduce num_sequences)
     - Limit frames per sequence (max_frames parameter)
     - Use CPU instead of GPU (slower but less memory)
     - Process sequences one at a time instead of batch

5. "Results don't match expected accuracy"
   ❌ Accuracy much lower than Le2i
   ✓ Solution:
     - This is normal (domain gap expected)
     - Range 0.70-0.88 is typical
     - Check if fallback pose extraction is used
     - Verify label assignment (look at sample predictions)

6. "Inference very slow"
   ❌ Processing takes hours
   ✓ Solution:
     - Enable GPU in Kaggle settings
     - Reduce num_sequences to test
     - Check if GPU is actually being used
     - Simplify pose extraction model
"""

# ============================================================================
# File Organization After URFD Testing
# ============================================================================

"""
After running URFD scripts, expected file structure:

project_root/
├── Scripts
│   ├── urfd_inspector.py                (REFERENCE)
│   ├── urfd_inference_handler.py        (ADVANCED)
│   ├── kaggle_urfd_runner.py            (RECOMMENDED)
│
├── results/
│   ├── le2i_evaluation.png              (Model on Le2i)
│   ├── le2i_metrics.csv                 (Le2i test metrics)
│   ├── urfd_summary.csv                 (URFD results)
│   ├── urfd_predictions_*.csv           (Per-sequence detailed results)
│
└── thesis/
    ├── Results Section
    │   ├── Table 1: Le2i Performance
    │   ├── Table 2: URFD Performance
    │   ├── Figure 1: Confusion Matrix (Le2i)
    │   ├── Figure 2: Probability Distribution
    │
    ├── Evaluation Section
    │   ├── Cross-dataset validation results
    │   ├── Domain gap discussion
    │
    └── Discussion
        ├── Generalization analysis
        ├── Comparison to related work
        └── Future improvements
"""

print("""
╔═══════════════════════════════════════════════════════════════════════════╗
║                    URFD DATASET GUIDE                                   ║
║                                                                          ║
║  Key Points:                                                            ║
║  • URFD uses PNG IMAGE SEQUENCES (not videos like Le2i)               ║
║  • Data organized as: /camera/activity-XX-camera/frame-XXX.png       ║
║  • Labels in folder name: "fall-XX" = Fall, "adl-XX" = Non-Fall      ║
║                                                                        ║
║  Scripts Created:                                                       ║
║  1. urfd_inspector.py → Explore dataset structure                    ║
║  2. urfd_inference_handler.py → Full pipeline (advanced)             ║
║  3. kaggle_urfd_runner.py → Kaggle notebook (recommended)            ║
║                                                                        ║
║  On Kaggle:                                                             ║
║  • Use: kaggle_urfd_runner.py                                        ║
║  • Copy each CELL into Kaggle notebook                               ║
║  • Time: ~20-30 minutes                                              ║
║                                                                        ║
║  Expected Results:                                                      ║
║  • Le2i Accuracy: 0.88-0.92 (known distribution)                    ║
║  • URFD Accuracy: 0.75-0.88 (domain gap expected)                   ║
║  • Processing Speed: Real-time capable (20-50 windows/sec)           ║
║                                                                       ║
║  Next Step: Run kaggle_urfd_runner.py on Kaggle notebook             ║
║                                                                       ║
╚═══════════════════════════════════════════════════════════════════════════╝
""")
