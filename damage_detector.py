# damage_detector.py
import os
os.environ['TORCH_LOAD_WEIGHTS_ONLY'] = 'False'  # FIX for PyTorch 2.6

import cv2
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json
from datetime import datetime
import matplotlib.pyplot as plt
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import yaml

class VehicleDamageDetector:
    """
    Multi-Image Vehicle Damage Detection System using YOLOv8
    """
    
    def __init__(self, model_path: str = None, device: str = 'cuda'):
        """
        Initialize the damage detector
        """
        self.device = 'cpu'  # Force CPU for now to avoid GPU issues
        print(f"Using device: {self.device}")
        
        # Initialize YOLO model
        if model_path and os.path.exists(model_path):
            try:
                # Load with weights_only=False for PyTorch 2.6 compatibility
                self.model = YOLO(model_path)
                print(f"✓ Loaded pre-trained model from {model_path}")
            except Exception as e:
                print(f"Error loading model: {e}")
                print("Initializing with base YOLOv8 model...")
                self.model = YOLO('yolov8n.yaml')
        else:
            # Initialize with YOLOv8n architecture
            self.model = YOLO('yolov8n.yaml')
            print("✓ Initialized base YOLOv8 model architecture")
        
        # Damage classes based on CarDD dataset
        self.damage_classes = {
            0: 'scratch',
            1: 'dent',
            2: 'crack',
            3: 'glass_break',
            4: 'lamp_break',
            5: 'tire_flat',
            6: 'deformation',
            7: 'chip'
        }
        
        # Severity mapping
        self.severity_levels = {
            'low': {'score_range': (0.0, 0.3), 'color': (0, 255, 0)},
            'medium': {'score_range': (0.3, 0.7), 'color': (0, 255, 255)},
            'high': {'score_range': (0.7, 1.0), 'color': (0, 0, 255)}
        }
        
        # Store results
        self.results_history = []
    
    def train_model(self, data_yaml: str = "datasets/cardd_yolo/dataset.yaml", 
                   epochs: int = 50, batch_size: int = 8):
        """
        Train YOLO model on CarDD dataset
        """
        print(f"Starting training with {epochs} epochs...")
        
        # Check if dataset exists
        if not os.path.exists(data_yaml):
            print(f"Error: Dataset YAML not found at {data_yaml}")
            print("Please run prepare_dataset.py first")
            return None
        
        # Simple training arguments
        training_args = {
            'data': data_yaml,
            'epochs': epochs,
            'imgsz': 640,
            'batch': batch_size,
            'device': self.device,
            'workers': 2,
            'patience': 20,
            'project': 'runs/train',
            'name': 'damage_detection',
            'exist_ok': True,
            'verbose': True,
            'seed': 42,
            'val': True,
            'plots': True
        }
        
        # Train the model
        try:
            results = self.model.train(**training_args)
            print("Training completed successfully!")
            
            # Load best model
            best_model_path = 'runs/train/damage_detection/weights/best.pt'
            if os.path.exists(best_model_path):
                self.model = YOLO(best_model_path)
                print(f"✓ Loaded best model from {best_model_path}")
            
            return results
        except Exception as e:
            print(f"Error during training: {e}")
            return None
    
    def analyze_multiple_images(self, image_paths: List[str], confidence_threshold: float = 0.25) -> Dict:
        """
        Analyze multiple images of a vehicle
        """
        all_results = []
        total_damages = []
        
        for img_path in image_paths:
            if not os.path.exists(img_path):
                print(f"Warning: Image not found - {img_path}")
                continue
            
            # Run inference
            img_result = self._analyze_single_image(img_path, confidence_threshold)
            
            if img_result:
                all_results.append(img_result)
                
                # Extract damage information
                for detection in img_result.get('detections', []):
                    total_damages.append({
                        'type': detection['class'],
                        'confidence': detection['confidence'],
                        'severity': detection['severity'],
                        'location': detection['location'],
                        'image': img_path
                    })
        
        # Generate comprehensive report
        comprehensive_report = self._generate_comprehensive_report(all_results, total_damages)
        
        # Store in history
        self.results_history.append({
            'timestamp': datetime.now().isoformat(),
            'images': image_paths,
            'report': comprehensive_report
        })
        
        return comprehensive_report
    
    def _analyze_single_image(self, image_path: str, confidence_threshold: float) -> Dict:
        """
        Analyze single image for damages
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        # Run YOLO inference
        try:
            results = self.model(image, conf=confidence_threshold, iou=0.45, device=self.device)
        except Exception as e:
            print(f"Error during inference: {e}")
            return None
        
        # Process results
        detections = []
        damage_counts = {damage_type: 0 for damage_type in self.damage_classes.values()}
        total_damage_score = 0
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Extract detection information
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    
                    # Calculate severity based on confidence and area
                    area = (x2 - x1) * (y2 - y1)
                    normalized_area = area / (image.shape[0] * image.shape[1])
                    severity_score = min(confidence * 0.7 + normalized_area * 0.3, 1.0)
                    
                    # Determine severity level
                    severity = self._get_severity_level(severity_score)
                    
                    # Store detection
                    detection = {
                        'class': self.damage_classes.get(class_id, f'unknown_{class_id}'),
                        'class_id': class_id,
                        'confidence': confidence,
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'area': area,
                        'severity': severity,
                        'severity_score': severity_score,
                        'location': self._get_damage_location(x1, y1, image.shape)
                    }
                    
                    detections.append(detection)
                    
                    # Update counters
                    damage_counts[detection['class']] += 1
                    total_damage_score += severity_score
        
        # Calculate image-level metrics
        num_damages = len(detections)
        avg_severity = total_damage_score / max(num_damages, 1)
        overall_severity = self._get_severity_level(avg_severity)
        
        return {
            'image_path': image_path,
            'detections': detections,
            'damage_counts': damage_counts,
            'num_damages': num_damages,
            'average_severity': avg_severity,
            'overall_severity': overall_severity,
            'image_dimensions': image.shape
        }
    
    def _get_severity_level(self, score: float) -> str:
        """Convert severity score to level"""
        if score < 0.3:
            return 'low'
        elif score < 0.7:
            return 'medium'
        else:
            return 'high'
    
    def _get_damage_location(self, x1: float, y1: float, img_shape: Tuple) -> str:
        """Determine approximate location of damage on vehicle"""
        height, width, _ = img_shape
        
        # Divide image into regions
        if x1 < width / 3:
            horizontal = 'left'
        elif x1 < 2 * width / 3:
            horizontal = 'center'
        else:
            horizontal = 'right'
            
        if y1 < height / 3:
            vertical = 'top'
        elif y1 < 2 * height / 3:
            vertical = 'middle'
        else:
            vertical = 'bottom'
        
        # Map to vehicle parts
        location_map = {
            ('left', 'top'): 'front-left',
            ('center', 'top'): 'front',
            ('right', 'top'): 'front-right',
            ('left', 'middle'): 'side-left',
            ('center', 'middle'): 'center',
            ('right', 'middle'): 'side-right',
            ('left', 'bottom'): 'rear-left',
            ('center', 'bottom'): 'rear',
            ('right', 'bottom'): 'rear-right'
        }
        
        return location_map.get((horizontal, vertical), 'unknown')
    
    def _generate_comprehensive_report(self, all_results: List, total_damages: List) -> Dict:
        """Generate comprehensive report from multiple image analyses"""
        
        # Aggregate statistics
        total_images = len(all_results)
        total_detections = len(total_damages)
        
        # Count damage types across all images
        damage_type_counts = {}
        severity_distribution = {'low': 0, 'medium': 0, 'high': 0}
        
        for damage in total_damages:
            damage_type = damage['type']
            damage_type_counts[damage_type] = damage_type_counts.get(damage_type, 0) + 1
            severity_distribution[damage['severity']] += 1
        
        # Calculate severity scores
        avg_severity_scores = []
        for result in all_results:
            avg_severity_scores.append(result.get('average_severity', 0))
        
        overall_avg_severity = np.mean(avg_severity_scores) if avg_severity_scores else 0
        
        # Generate report
        report = {
            'summary': {
                'total_images_analyzed': total_images,
                'total_damages_detected': total_detections,
                'overall_severity': self._get_severity_level(overall_avg_severity),
                'overall_severity_score': float(overall_avg_severity),
                'analysis_timestamp': datetime.now().isoformat()
            },
            'damage_breakdown': {
                'by_type': damage_type_counts,
                'by_severity': severity_distribution
            },
            'recommendations': self._generate_recommendations(damage_type_counts, severity_distribution)
        }
        
        return report
    
    def _generate_recommendations(self, type_counts: Dict, severity_dist: Dict) -> List[str]:
        """Generate recommendations based on damage analysis"""
        recommendations = []
        
        # High severity damages
        if severity_dist.get('high', 0) > 0:
            recommendations.append("Immediate repair required for high severity damages")
        
        # Specific damage type recommendations
        if type_counts.get('crack', 0) > 0:
            recommendations.append("Windshield/window cracks require immediate attention for safety")
        
        if type_counts.get('tire_flat', 0) > 0:
            recommendations.append("Flat tire detected - vehicle should not be driven")
        
        if type_counts.get('glass_break', 0) > 0:
            recommendations.append("Broken glass requires replacement for safety and weather protection")
        
        # General recommendations
        total_damages = sum(type_counts.values())
        if total_damages > 5:
            recommendations.append("Multiple damages detected - comprehensive inspection recommended")
        
        if severity_dist.get('medium', 0) > 2:
            recommendations.append("Schedule repair for medium severity damages within 1-2 weeks")
        
        if len(recommendations) == 0 and total_damages > 0:
            recommendations.append("Minor damages detected - maintenance recommended when convenient")
        
        return recommendations

# Simple test
def test():
    """Simple test function"""
    print("Testing Vehicle Damage Detector...")
    
    # Create sample image
    sample_dir = "sample_images"
    os.makedirs(sample_dir, exist_ok=True)
    
    img_path = f"{sample_dir}/test_car.jpg"
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(img, "Test Car Image", (50, 240), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imwrite(img_path, img)
    
    # Test detector
    detector = VehicleDamageDetector()
    report = detector.analyze_multiple_images([img_path])
    
    print(f"\nImages analyzed: {report['summary']['total_images_analyzed']}")
    print(f"Damages found: {report['summary']['total_damages_detected']}")

if __name__ == "__main__":
    test()