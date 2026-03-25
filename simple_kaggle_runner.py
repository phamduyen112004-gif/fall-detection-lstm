"""
Simplified Kaggle Runner - Copy & Paste vào Kaggle Notebook Cell
Chỉ cần chạy từng cell một trên Kaggle
"""

# ============================================================================
# CELL 1: Setup Environment (Chạy cái này trước)
# ============================================================================

import os
import sys
from pathlib import Path

# Set Kaggle paths
os.environ["LE2I_INPUT_ROOT"] = "/kaggle/input/datasets/tuyenldvn/falldataset-imvia"
os.environ["LE2I_OUTPUT_ROOT"] = "/kaggle/working"

INPUT_ROOT = Path("/kaggle/input/datasets/tuyenldvn/falldataset-imvia")
OUTPUT_ROOT = Path("/kaggle/working")

print(f"✓ INPUT_ROOT: {INPUT_ROOT}")
print(f"✓ OUTPUT_ROOT: {OUTPUT_ROOT}")

# ============================================================================
# CELL 2: Install Dependencies
# ============================================================================

print("📦 Installing dependencies... (có thể mất vài phút)")

import subprocess
import sys

# Upgrade pip
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--upgrade", "pip"], check=False)

# Install core dependencies
packages = [
    "ultralytics",
    "opencv-python-headless",
    "numpy",
    "pandas",
    "scikit-learn",
    "scipy",
    "tqdm",
    "pyyaml",
    "mediapipe",
    "torch",
    "torchvision",
]

for pkg in packages:
    print(f"  ⏳ Installing {pkg}...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=False)

# Gym compatibility
print("  ⏳ Installing gym...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gym<=0.25.2"], check=False)

print("✓ Dependencies installed!")

# ============================================================================
# CELL 3: Create Output Directories
# ============================================================================

print("📁 Creating output directories...")

output_dirs = [
    OUTPUT_ROOT,
    OUTPUT_ROOT / "data" / "processed",
    OUTPUT_ROOT / "data" / "features",
    OUTPUT_ROOT / "models",
    OUTPUT_ROOT / "reports",
]

for d in output_dirs:
    d.mkdir(parents=True, exist_ok=True)
    print(f"  ✓ {d}")

print("✓ Directories ready!")

# ============================================================================
# CELL 4: Verify Kaggle Input Dataset
# ============================================================================

print("🔍 Checking dataset...")

if INPUT_ROOT.exists():
    # Count videos
    video_files = list(INPUT_ROOT.rglob("*.avi"))
    print(f"✓ Dataset found: {len(video_files)} video files")
    print(f"✓ Location: {INPUT_ROOT}")
    
    # Show structure
    print("\nDataset structure:")
    for item in sorted(INPUT_ROOT.iterdir())[:5]:
        print(f"  - {item.name}/")
else:
    print(f"❌ ERROR: Dataset not found at {INPUT_ROOT}")
    print("⚠️  Ensure 'falldataset-imvia' is attached to Kaggle Input")

# ============================================================================
# CELL 5: Run the Pipeline
# ============================================================================

print("\n" + "="*60)
print("🚀 Running Fall Detection LSTM Pipeline")
print("="*60 + "\n")

# Add to path
sys.path.insert(0, "/kaggle/working")

try:
    # Import pipeline
    from src.kaggle_pipeline import main
    
    # Setup arguments
    # Tùy chọn: 
    # - "--skip-extract": Chỉ huấn luyện (bỏ trích xuất pose)
    # - "--skip-train": Chỉ trích xuất pose (bỏ huấn luyện)  
    # - "--skip-sanity": Bỏ qua sanity checks
    # - "--train-only": Chỉ huấn luyện
    # - "--extract-only": Chỉ trích xuất
    
    sys.argv = ["kaggle", "--skip-sanity"]
    
    # Run
    main()
    
    print("\n" + "="*60)
    print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    # Show output location
    print(f"\n📊 Results saved to {OUTPUT_ROOT}:")
    print(f"  - Models: {OUTPUT_ROOT / 'models'}")
    print(f"  - Features: {OUTPUT_ROOT / 'data' / 'features'}")
    print(f"  - Reports: {OUTPUT_ROOT / 'reports'}")
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# CELL 6: View Results (Optional)
# ============================================================================

import os

print("\n📂 Final output structure:")
for root, dirs, files in os.walk(OUTPUT_ROOT):
    level = root.replace(str(OUTPUT_ROOT), "").count(os.sep)
    indent = " " * 2 * level
    print(f"{indent}📁 {os.path.basename(root)}/")
    sub_indent = " " * 2 * (level + 1)
    for file in files[:5]:  # Show first 5 files in each dir
        print(f"{sub_indent}📄 {file}")
    if len(files) > 5:
        print(f"{sub_indent}... và {len(files) - 5} file khác")
