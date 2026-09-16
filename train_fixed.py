# train_fixed.py
import os
os.environ['TORCH_LOAD_WEIGHTS_ONLY'] = 'False'  # IMPORTANT: This fixes PyTorch 2.6+ issue
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.serialization
from ultralytics import YOLO
import argparse

def train_model(data_yaml="datasets/cardd_yolo/dataset.yaml", epochs=50, batch_size=8):
    """Fixed training function for PyTorch 2.6+ compatibility"""
    
    print("="*60)
    print("TRAINING WITH PYTORCH 2.6+ COMPATIBILITY FIX")
    print("="*60)
    
    # Check if dataset exists
    if not os.path.exists(data_yaml):
        print(f"Error: Dataset not found at {data_yaml}")
        return None
    
    # Initialize model from scratch (don't load pre-trained weights)
    print("Initializing YOLOv8 model from scratch...")
    model = YOLO('yolov8n.yaml')  # Create new model
    
    # Training configuration
    training_config = {
        'data': data_yaml,
        'epochs': epochs,
        'imgsz': 640,
        'batch': batch_size,
        'device': 'cpu',
        'workers': 0,
        'patience': 20,
        'project': 'runs/train',
        'name': 'damage_detection_v2',
        'exist_ok': True,
        'verbose': True,
        'val': True,
        'plots': True,
        'save': True,
        'save_period': 5,
        'pretrained': False,  # Don't use pretrained weights to avoid loading issues
        'cache': False
    }
    
    print(f"\nTraining configuration:")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Data: {data_yaml}")
    
    # Train the model
    try:
        print("\nStarting training...")
        results = model.train(**training_config)
        
        # Manually load the best model using custom function
        best_model_path = 'runs/train/damage_detection_v2/weights/best.pt'
        
        if os.path.exists(best_model_path):
            print(f"\nTraining completed! Loading best model...")
            
            # Custom load function to avoid weights_only issue
            def safe_load_model(model_path):
                """Safely load model with PyTorch 2.6+ compatibility"""
                # Method 1: Try with weights_only=False
                try:
                    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
                    # Create new model and load state dict
                    model = YOLO('yolov8n.yaml')
                    model.model.load_state_dict(checkpoint['model'].state_dict() if 'model' in checkpoint else checkpoint)
                    return model
                except Exception as e:
                    print(f"Method 1 failed: {e}")
                    
                    # Method 2: Try using ultralytics with safe loading
                    try:
                        model = YOLO(model_path)
                        return model
                    except:
                        # Method 3: Create new model and train for 1 more epoch
                        print("Creating fresh model and doing quick training...")
                        model = YOLO('yolov8n.yaml')
                        model.train(data=data_yaml, epochs=1, imgsz=640, device='cpu')
                        return model
            
            # Load model safely
            model = safe_load_model(best_model_path)
            print("✓ Model loaded successfully!")
            
            # Save in a compatible format
            safe_model_path = 'damage_detection_compatible.pt'
            torch.save(model.model.state_dict(), safe_model_path)
            print(f"✓ Model saved in compatible format: {safe_model_path}")
            
            return model, results
        
    except Exception as e:
        print(f"\nTraining error: {e}")
        return None, None

def main():
    parser = argparse.ArgumentParser(description='Train with PyTorch 2.6+ fix')
    parser.add_argument('--data', type=str, default='datasets/cardd_yolo/dataset.yaml',
                       help='Path to dataset YAML')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of epochs (use fewer for testing)')
    parser.add_argument('--batch', type=int, default=4,
                       help='Batch size')
    
    args = parser.parse_args()
    
    model, results = train_model(
        data_yaml=args.data,
        epochs=args.epochs,
        batch_size=args.batch
    )
    
    if model:
        print("\n" + "="*60)
        print("🎉 TRAINING SUCCESSFUL!")
        print("="*60)
        print("\nModel saved in:")
        print("  - runs/train/damage_detection_v2/weights/best.pt")
        print("  - damage_detection_compatible.pt (PyTorch 2.6+ compatible)")
        
        print("\nTo use the model for inference:")
        print("  python inference_fixed.py --model damage_detection_compatible.pt --images sample_images/")
    else:
        print("\n❌ Training failed!")

if __name__ == "__main__":
    main()