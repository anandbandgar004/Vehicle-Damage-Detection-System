# analyze_6classes.py
import os
import argparse
from pathlib import Path
import torch
from ultralytics import YOLO
import cv2
import numpy as np
import json
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(description='Analyze vehicle images with 6-class model')
    parser.add_argument('--images', type=str, required=True,
                       help='Path to image or directory')
    parser.add_argument('--model', type=str, default='damage_6classes.pt',
                       help='Path to trained model')
    parser.add_argument('--output', type=str, default='results_6classes',
                       help='Output directory')
    parser.add_argument('--conf', type=float, default=0.15,
                       help='Confidence threshold (use lower for new model)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("VEHICLE DAMAGE ANALYSIS - 6 CLASSES")
    print("="*60)
    
    # Check model
    if not os.path.exists(args.model):
        print(f"❌ Model not found: {args.model}")
        print("\nPlease train the model first:")
        print("  python train_6classes.py")
        return
    
    # Load model
    print(f"\nLoading model: {args.model}")
    try:
        # Try to load with weights_only=False for compatibility
        model = YOLO(args.model)
        print("✅ Model loaded")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Get images
    if os.path.isdir(args.images):
        image_paths = []
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            image_paths.extend(Path(args.images).glob(f'*{ext}'))
        image_paths = [str(p) for p in image_paths]
    else:
        image_paths = [args.images]
    
    if not image_paths:
        print(f"❌ No images found at: {args.images}")
        return
    
    print(f"\nFound {len(image_paths)} images")
    
    # 6 damage classes
    damage_classes = [
        'dent',
        'scratch',
        'crack',
        'glass_shatter',
        'lamp_broken',
        'tire_flat'
    ]
    
    print(f"\nDamage classes ({len(damage_classes)}): {damage_classes}")
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Analyze images
    all_results = {}
    
    print(f"\nAnalyzing images (confidence: {args.conf})...")
    
    for img_path in image_paths:
        print(f"  Processing: {Path(img_path).name}")
        
        # Read image
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        # Run inference
        results = model(img, conf=args.conf, device='cpu')
        
        detections = []
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    
                    # Calculate severity
                    area = (x2 - x1) * (y2 - y1)
                    total_area = img.shape[0] * img.shape[1]
                    severity_score = min(conf * 0.7 + (area / total_area) * 0.3, 1.0)
                    
                    if severity_score < 0.3:
                        severity = 'low'
                    elif severity_score < 0.7:
                        severity = 'medium'
                    else:
                        severity = 'high'
                    
                    # Get class name
                    class_name = damage_classes[cls] if cls < len(damage_classes) else f'class_{cls}'
                    
                    detection = {
                        'class': cls,
                        'class_name': class_name,
                        'confidence': conf,
                        'severity': severity,
                        'severity_score': float(severity_score),
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'area': float(area)
                    }
                    
                    detections.append(detection)
        
        all_results[img_path] = detections
        
        # Save visualization if detections found
        if detections:
            # Draw bounding boxes
            for det in detections:
                x1, y1, x2, y2 = map(int, det['bbox'])
                
                # Color based on severity
                if det['severity'] == 'low':
                    color = (0, 255, 0)  # Green
                elif det['severity'] == 'medium':
                    color = (0, 255, 255)  # Yellow
                else:
                    color = (0, 0, 255)  # Red
                
                # Draw rectangle
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                
                # Draw label
                label = f"{det['class_name']} {det['confidence']:.2f}"
                cv2.putText(img, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Save visualization
            viz_path = os.path.join(args.output, f"viz_{Path(img_path).name}")
            cv2.imwrite(viz_path, img)
            print(f"    ✅ Found {len(detections)} damage(s)")
        else:
            print(f"    ⭕ No damages detected")
    
    # Generate report
    total_damages = sum(len(dets) for dets in all_results.values())
    damage_counts = {}
    severity_counts = {'low': 0, 'medium': 0, 'high': 0}
    
    for img_path, detections in all_results.items():
        for det in detections:
            class_name = det['class_name']
            damage_counts[class_name] = damage_counts.get(class_name, 0) + 1
            severity_counts[det['severity']] += 1
    
    # Save report
    report = {
        'summary': {
            'analysis_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'total_images': len(image_paths),
            'images_with_damage': sum(1 for dets in all_results.values() if dets),
            'total_damages': total_damages,
            'damage_counts': damage_counts,
            'severity_counts': severity_counts,
            'model_used': args.model,
            'confidence_threshold': args.conf
        },
        'detailed_results': all_results
    }
    
    report_file = os.path.join(args.output, 'damage_report.json')
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)
    
    print(f"\n📊 Images analyzed: {len(image_paths)}")
    print(f"📊 Damages detected: {total_damages}")
    
    if damage_counts:
        print(f"\n🔍 Damage types:")
        for class_name, count in damage_counts.items():
            print(f"  {class_name}: {count}")
        
        print(f"\n⚠️  Severity:")
        for severity, count in severity_counts.items():
            print(f"  {severity}: {count}")
    else:
        print("\n✅ No damages detected")
        print("\nPossible reasons:")
        print("  1. Model needs more training (try more epochs)")
        print("  2. Images don't have visible damages")
        print("  3. Try lower confidence threshold (--conf 0.1)")
        print("  4. Model might need better training data")
    
    print(f"\n📁 Results saved to: {args.output}/")
    print(f"📄 Report: {report_file}")

if __name__ == "__main__":
    main()