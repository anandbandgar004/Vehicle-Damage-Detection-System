# inference_fixed.py
import torch
from ultralytics import YOLO
import argparse
import os
from pathlib import Path
import json
from datetime import datetime
import cv2
import numpy as np

def analyze_image(model, image_path, confidence=0.25):
    """Analyze single image"""
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    # Run inference
    results = model(img, conf=confidence, device='cpu')
    
    # Process results
    detections = []
    for result in results:
        if result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                
                # Calculate damage area
                area = (x2 - x1) * (y2 - y1)
                total_area = img.shape[0] * img.shape[1]
                severity_score = min(conf * 0.7 + (area / total_area) * 0.3, 1.0)
                
                # Determine severity
                if severity_score < 0.3:
                    severity = 'low'
                elif severity_score < 0.7:
                    severity = 'medium'
                else:
                    severity = 'high'
                
                detections.append({
                    'class': cls,
                    'class_name': ['scratch', 'dent', 'crack', 'glass_break', 
                                  'lamp_break', 'tire_flat', 'deformation', 'chip'][cls],
                    'confidence': conf,
                    'severity': severity,
                    'severity_score': float(severity_score),
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'area': float(area)
                })
    
    return detections

def main():
    parser = argparse.ArgumentParser(description='Inference with PyTorch 2.6+ fix')
    parser.add_argument('--model', type=str, default='damage_detection_compatible.pt',
                       help='Path to model weights')
    parser.add_argument('--images', type=str, default='sample_images',
                       help='Path to image or directory')
    parser.add_argument('--output', type=str, default='results',
                       help='Output directory')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold')
    
    args = parser.parse_args()
    
    print("="*60)
    print("VEHICLE DAMAGE DETECTION - INFERENCE")
    print("="*60)
    
    # Check model
    if not os.path.exists(args.model):
        print(f"\nError: Model not found: {args.model}")
        print("\nPlease train the model first:")
        print("  python train_fixed.py")
        return
    
    # Load model safely
    print(f"\nLoading model: {args.model}")
    try:
        # Load state dict and create model
        if args.model.endswith('.pt'):
            # Check if it's a full model or just state dict
            checkpoint = torch.load(args.model, map_location='cpu', weights_only=False)
            
            # Create model
            model = YOLO('yolov8n.yaml')
            
            # Load weights
            if isinstance(checkpoint, dict) and 'model' in checkpoint:
                model.model.load_state_dict(checkpoint['model'].state_dict())
            else:
                model.model.load_state_dict(checkpoint)
            
            print("✓ Model loaded from state dict")
        else:
            # Try loading directly
            model = YOLO(args.model)
            print("✓ Model loaded directly")
            
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Get images
    if os.path.isdir(args.images):
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            image_files.extend(Path(args.images).glob(f'*{ext}'))
            image_files.extend(Path(args.images).glob(f'*{ext.upper()}'))
        image_paths = [str(f) for f in image_files]
    else:
        image_paths = [args.images]
    
    # Create sample images if none exist
    if not image_paths or args.images == 'sample_images':
        print("\nCreating sample images...")
        sample_dir = Path('sample_images')
        sample_dir.mkdir(exist_ok=True)
        
        for i in range(3):
            img_path = sample_dir / f'car_{i+1}.jpg'
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(img, f"Test Car {i+1}", (50, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
            
            # Add some simulated damage
            if i == 0:
                cv2.rectangle(img, (100, 100), (200, 150), (0, 0, 255), 2)  # Red damage
            elif i == 1:
                cv2.rectangle(img, (300, 200), (400, 250), (0, 255, 255), 2)  # Yellow damage
            
            cv2.imwrite(str(img_path), img)
            print(f"  Created: {img_path}")
        
        # Update image paths
        image_paths = [str(sample_dir / f'car_{i+1}.jpg') for i in range(3)]
    
    print(f"\nFound {len(image_paths)} images")
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Analyze images
    all_results = {}
    damage_counts = {}
    severity_counts = {'low': 0, 'medium': 0, 'high': 0}
    
    print("\nAnalyzing images...")
    for img_path in image_paths:
        print(f"  Processing: {Path(img_path).name}")
        detections = analyze_image(model, img_path, args.conf)
        
        if detections:
            all_results[img_path] = detections
            
            # Count damages
            for det in detections:
                cls_name = det['class_name']
                damage_counts[cls_name] = damage_counts.get(cls_name, 0) + 1
                severity_counts[det['severity']] += 1
    
    # Generate report
    report = {
        'summary': {
            'total_images': len(image_paths),
            'images_with_damage': len(all_results),
            'total_damages': sum(damage_counts.values()),
            'severity_distribution': severity_counts,
            'timestamp': datetime.now().isoformat()
        },
        'damage_counts': damage_counts,
        'detailed_results': all_results
    }
    
    # Save report
    report_file = Path(args.output) / 'damage_report.json'
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    
    print(f"\n📊 Images analyzed: {len(image_paths)}")
    print(f"📊 Images with damage: {len(all_results)}")
    print(f"📊 Total damages detected: {sum(damage_counts.values())}")
    
    if damage_counts:
        print("\n🔍 Damage types found:")
        for damage_type, count in damage_counts.items():
            print(f"  {damage_type}: {count}")
        
        print("\n⚠️  Severity distribution:")
        for severity, count in severity_counts.items():
            print(f"  {severity}: {count}")
        
        # Generate recommendations
        print("\n💡 RECOMMENDATIONS:")
        if severity_counts['high'] > 0:
            print("  • Immediate repair required for high severity damages")
        if severity_counts['medium'] > 2:
            print("  • Schedule repairs for medium severity damages")
        if sum(damage_counts.values()) > 0:
            print(f"  • Total repair estimate: ${sum(damage_counts.values()) * 150:.2f} (approx)")
    else:
        print("\n✅ No damages detected - vehicle appears to be in good condition!")
    
    print(f"\n📄 Report saved to: {report_file}")
    print(f"\n✅ Analysis complete!")

if __name__ == "__main__":
    main()