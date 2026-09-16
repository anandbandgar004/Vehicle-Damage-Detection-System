# setup.py
import os
import subprocess
import sys
from pathlib import Path

def setup_environment():
    """Setup the project environment"""
    
    print("="*60)
    print("VEHICLE DAMAGE DETECTION PROJECT SETUP")
    print("="*60)
    
    # Create directory structure
    directories = [
        'datasets',
        'sample_images',
        'inference_results',
        'output',
        'runs/train',
        'runs/val'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")
    
    # Check if CarDD dataset exists
    if Path("CarDD_release").exists():
        print(f"\nFound CarDD dataset at: CarDD_release")
    else:
        print(f"\nWarning: CarDD dataset not found!")
        print("Please download the CarDD dataset and place it in the current directory")
        print("Expected structure:")
        print("  CarDD_release/")
        print("    ├── CarDD_COCO/")
        print("    │   ├── annotations/")
        print("    │   ├── train2017/")
        print("    │   ├── val2017/")
        print("    │   └── test2017/")
        print("    ├── CarDD-TR/")
        print("    ├── CarDD-VAL/")
        print("    └── CarDD-TE/")
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print(f"\nWarning: Python 3.8+ required. Current: {python_version.major}.{python_version.minor}")
    else:
        print(f"\nPython version: {python_version.major}.{python_version.minor}.{python_version.micro} ✓")
    
    # Check for GPU
    try:
        import torch
        if torch.cuda.is_available():
            print(f"GPU available: {torch.cuda.get_device_name(0)} ✓")
        else:
            print("GPU not available - will use CPU (training will be slower)")
    except:
        print("Torch not installed yet")
    
    print("\n" + "="*60)
    print("SETUP COMPLETE")
    print("="*60)
    print("\nNext steps:")
    print("1. Install dependencies:")
    print("   pip install -r requirements.txt")
    print("\n2. Prepare dataset:")
    print("   python prepare_dataset.py")
    print("\n3. Train model:")
    print("   python train_model.py")
    print("\n4. Run inference:")
    print("   python inference.py --model runs/train/damage_detection/weights/best.pt --images sample_images/")
    print("\n5. Or test with sample:")
    print("   python damage_detector.py")

if __name__ == "__main__":
    setup_environment()