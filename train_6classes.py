import os
import sys
import argparse
import shutil

# ==============================================================================
# FIX FOR PYTORCH 2.6: "Weights only load failed"
# ==============================================================================
# PyTorch 2.6 defaults to weights_only=True, which blocks loading Ultralytics
# models (which contain architecture classes). We monkey-patch torch.load to 
# force weights_only=False globally for this script.
# ==============================================================================
import torch

if hasattr(torch, 'load'):
    _original_torch_load = torch.load
    def _patched_load(*args, **kwargs):
        # Force weights_only=False to bypass the strict PyTorch 2.6 check
        # This is safe for .pt files from trusted sources like Ultralytics
        kwargs['weights_only'] = False
        return _original_torch_load(*args, **kwargs)
    torch.load = _patched_load
    
    # Also patch torch.serialization.load if it's used internally
    if hasattr(torch, 'serialization') and hasattr(torch.serialization, 'load'):
        torch.serialization.load = _patched_load
# ==============================================================================

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from ultralytics import YOLO

def main():
    print("="*60)
    print("STARTING TRAINING - HIGH ACCURACY MODE")
    print("="*60)

    # 1. Setup Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='datasets/cardd_6classes/dataset.yaml')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch', type=int, default=16)  # Matches your command
    args = parser.parse_args()

    # 2. Check Dataset
    if not os.path.exists(args.data):
        print(f"❌ Dataset not found at {args.data}")
        print("   -> Please run 'python prepare_dataset_6classes.py' first!")
        return

    # 3. Device Selection
    device = '0' if torch.cuda.is_available() else 'cpu'
    print(f"Using Device: {device.upper()}")

    # 4. Load Pretrained Model (Transfer Learning)
    print("📦 Loading YOLOv8n pretrained weights...")
    try:
        # This will now use the patched torch.load
        model = YOLO('yolov8n.pt')
    except Exception as e:
        print(f"❌ Failed to load pretrained weights: {e}")
        return

    # 5. Training Arguments
    train_args = {
        'data': args.data,
        'epochs': args.epochs,
        'batch': args.batch,
        'imgsz': 640,
        'device': device,
        'workers': 4 if device != 'cpu' else 0,
        
        # Accuracy Optimizations
        'pretrained': True,       # Start from yolov8n.pt
        'optimizer': 'AdamW',     # Better for small/complex datasets
        'lr0': 0.001,             # Learning rate
        'lrf': 0.01,              # Final LR factor
        'cos_lr': True,           # Cosine decay
        'close_mosaic': 10,       # Stop mosaic in last 10 epochs
        
        # Project Settings
        'project': 'runs/train',
        'name': 'damage_6classes_v2',
        'exist_ok': True,
        'val': True,
        'plots': True,
        'save': True,
        'verbose': True
    }

    print(f"\n🚀 Starting training for {args.epochs} epochs...")
    try:
        results = model.train(**train_args)
        
        # Save best model to root folder for easy access
        best_path = 'runs/train/damage_6classes_v2/weights/best.pt'
        if os.path.exists(best_path):
            shutil.copy(best_path, 'best_damage_model.pt')
            print(f"\n✅ Training Complete! Best model saved as 'best_damage_model.pt'")
        else:
            print("\n⚠️ Training finished but 'best.pt' not found.")
            
    except Exception as e:
        print(f"\n❌ Training failed: {e}")

if __name__ == "__main__":
    main()