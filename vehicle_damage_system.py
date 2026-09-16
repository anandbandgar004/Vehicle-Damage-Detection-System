# vehicle_damage_system.py
"""
Complete Vehicle Damage Detection System
Works with PyTorch 2.6+ and Python 3.13
"""

import os
os.environ['TORCH_LOAD_WEIGHTS_ONLY'] = 'False'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.serialization
from ultralytics import YOLO
import cv2
import numpy as np
import json
from pathlib import Path
from datetime import datetime

class VehicleDamageSystem:
    """Complete vehicle damage detection system"""
    
    def __init__(self):
        self.model = None
        self.damage_classes = [
            'scratch', 'dent', 'crack', 'glass_break',
            'lamp_break', 'tire_flat', 'deformation', 'chip'
        ]
    
    def train(self, data_yaml='datasets/cardd_yolo/dataset.yaml', epochs=30, batch_size=4):
        """Train the model"""
        print("Training vehicle damage detection model...")
        
        # Check dataset
        if not os.path.exists(data_yaml):
            print(f"Error: Dataset not found at {data_yaml}")
            return False
        
        # Create model from scratch
        self.model = YOLO('yolov8n.yaml')
        
        # Train
        try:
            self.model.train(
                data=data_yaml,
                epochs=epochs,
                imgsz=640,
                batch=batch_size,
                device='cpu',
                workers=0,
                project='runs/train',
                name='vehicle_damage',
                exist_ok=True,
                verbose=True,
                val=True,
                save=True,
                pretrained=False
            )
            
            # Save model in compatible format
            self.save_model('vehicle_damage_model.pt')
            print("✓ Training completed successfully!")
            return True
            
        except Exception as e:
            print(f"Training error: {e}")
            return False
    
    def save_model(self, path='vehicle_damage_model.pt'):
        """Save model in compatible format"""
        if self.model:
            torch.save(self.model.model.state_dict(), path)
            print(f"✓ Model saved: {path}")
    
    def load_model(self, path='vehicle_damage_model.pt'):
        """Load model"""
        try:
            # Load state dict
            state_dict = torch.load(path, map_location='cpu', weights_only=False)
            
            # Create model and load weights
            self.model = YOLO('yolov8n.yaml')
            self.model.model.load_state_dict(state_dict)
            
            print(f"✓ Model loaded: {path}")
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def analyze_images(self, image_paths, confidence=0.25):
        """Analyze multiple images"""
        if not self.model:
            print("Error: Model not loaded. Train or load a model first.")
            return None
        
        results = {}
        total_damages = []
        
        for img_path in image_paths:
            if not os.path.exists(img_path):
                print(f"Warning: Image not found - {img_path}")
                continue
            
            # Read image
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            # Run inference
            outputs = self.model(img, conf=confidence, device='cpu')
            
            detections = []
            for output in outputs:
                if output.boxes is not None:
                    for box in output.boxes:
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
                        
                        detection = {
                            'class': cls,
                            'class_name': self.damage_classes[cls] if cls < len(self.damage_classes) else f'unknown_{cls}',
                            'confidence': conf,
                            'severity': severity,
                            'severity_score': float(severity_score),
                            'bbox': [float(x1), float(y1), float(x2), float(y2)],
                            'image': img_path
                        }
                        
                        detections.append(detection)
                        total_damages.append(detection)
            
            results[img_path] = detections
        
        # Generate report
        report = self._generate_report(results, total_damages)
        return report
    
    def _generate_report(self, results, total_damages):
        """Generate comprehensive report"""
        damage_counts = {}
        severity_counts = {'low': 0, 'medium': 0, 'high': 0}
        
        for damage in total_damages:
            cls_name = damage['class_name']
            damage_counts[cls_name] = damage_counts.get(cls_name, 0) + 1
            severity_counts[damage['severity']] += 1
        
        report = {
            'summary': {
                'total_images': len(results),
                'total_damages': len(total_damages),
                'severity_distribution': severity_counts,
                'timestamp': datetime.now().isoformat()
            },
            'damage_breakdown': damage_counts,
            'detailed_results': results,
            'recommendations': self._generate_recommendations(damage_counts, severity_counts)
        }
        
        return report
    
    def _generate_recommendations(self, damage_counts, severity_counts):
        """Generate repair recommendations"""
        recommendations = []
        
        if severity_counts['high'] > 0:
            recommendations.append("🚨 IMMEDIATE ACTION REQUIRED: High severity damages detected")
        
        if 'crack' in damage_counts:
            recommendations.append("⚠️ Windshield/window cracks require immediate attention for safety")
        
        if 'tire_flat' in damage_counts:
            recommendations.append("🛞 Flat tire detected - do not drive the vehicle")
        
        if severity_counts['medium'] > 0:
            recommendations.append("🔧 Schedule repairs for medium severity damages within 1-2 weeks")
        
        if len(recommendations) == 0 and sum(damage_counts.values()) > 0:
            recommendations.append("✅ Minor damages detected - maintenance recommended when convenient")
        
        return recommendations

def main():
    """Main function"""
    print("="*60)
    print("VEHICLE DAMAGE DETECTION SYSTEM")
    print("="*60)
    
    system = VehicleDamageSystem()
    
    # Check if we have a trained model
    if os.path.exists('vehicle_damage_model.pt'):
        print("\nFound existing model. Loading...")
        system.load_model('vehicle_damage_model.pt')
    else:
        print("\nNo existing model found.")
        print("Would you like to:")
        print("1. Train a new model")
        print("2. Use the system with sample images")
        
        choice = input("\nEnter choice (1 or 2): ").strip()
        
        if choice == '1':
            # Train model
            print("\nTraining new model...")
            if system.train(epochs=20, batch_size=4):
                print("✓ Model trained successfully!")
            else:
                print("✗ Training failed. Creating sample model...")
                # Create a simple model for testing
                system.model = YOLO('yolov8n.yaml')
                system.save_model('vehicle_damage_model.pt')
        else:
            # Create sample model
            print("\nCreating sample model...")
            system.model = YOLO('yolov8n.yaml')
            system.save_model('vehicle_damage_model.pt')
    
    # Create sample images
    print("\nCreating sample images...")
    sample_dir = Path('sample_vehicles')
    sample_dir.mkdir(exist_ok=True)
    
    for i in range(3):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(img, f"Vehicle {i+1}", (50, 240), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        
        # Add simulated damage patterns
        if i == 0:
            cv2.rectangle(img, (100, 100), (200, 180), (0, 0, 255), 3)  # Severe damage
        elif i == 1:
            cv2.circle(img, (300, 200), 40, (0, 255, 255), 2)  # Medium damage
        
        img_path = sample_dir / f'vehicle_{i+1}.jpg'
        cv2.imwrite(str(img_path), img)
        print(f"Created: {img_path}")
    
    # Analyze images
    print("\nAnalyzing vehicle images...")
    image_paths = [str(sample_dir / f'vehicle_{i+1}.jpg') for i in range(3)]
    
    report = system.analyze_images(image_paths)
    
    if report:
        # Save report
        with open('damage_assessment_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print results
        print("\n" + "="*60)
        print("DAMAGE ASSESSMENT REPORT")
        print("="*60)
        
        print(f"\n📊 Summary:")
        print(f"  Images analyzed: {report['summary']['total_images']}")
        print(f"  Total damages: {report['summary']['total_damages']}")
        
        print(f"\n🔍 Damage breakdown:")
        for damage_type, count in report['damage_breakdown'].items():
            print(f"  {damage_type}: {count}")
        
        print(f"\n⚠️  Severity:")
        for severity, count in report['summary']['severity_distribution'].items():
            print(f"  {severity}: {count}")
        
        print(f"\n💡 Recommendations:")
        for rec in report['recommendations']:
            print(f"  • {rec}")
        
        print(f"\n📄 Full report saved to: damage_assessment_report.json")
    
    print("\n" + "="*60)
    print("SYSTEM READY!")
    print("="*60)
    print("\nTo analyze your own images:")
    print("  1. Place vehicle images in a folder")
    print("  2. Run: python analyze_custom.py --images your_folder/")

if __name__ == "__main__":
    main()