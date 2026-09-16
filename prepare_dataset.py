# prepare_dataset.py
import os
import json
import shutil
from pathlib import Path
import yaml
import argparse
from tqdm import tqdm

def convert_cardd_to_yolo_fixed(dataset_root="CarDD_release", output_path="datasets/cardd_yolo"):
    """
    Fixed conversion for CarDD dataset with proper type handling
    """
    
    dataset_root = Path(dataset_root)
    
    # Adjust path - based on your output, the actual path is CarDD_release\CarDD_release\
    if (dataset_root / "CarDD_release").exists():
        dataset_root = dataset_root / "CarDD_release"
    
    output_dir = Path(output_path)
    
    # Create output directories
    for split in ['train', 'val', 'test']:
        (output_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    print(f"Dataset root: {dataset_root}")
    print(f"Output directory: {output_dir}")
    
    # Check if COCO directory exists
    coco_dir = dataset_root / "CarDD_COCO"
    if not coco_dir.exists():
        print("Error: CarDD_COCO directory not found!")
        print("Looking for:", coco_dir)
        return None
    
    print(f"Found COCO directory: {coco_dir}")
    
    # Annotation files
    annotation_files = {
        'train': coco_dir / "annotations" / "instances_train2017.json",
        'val': coco_dir / "annotations" / "instances_val2017.json",
        'test': coco_dir / "annotations" / "instances_test2017.json"
    }
    
    # Image directories
    image_dirs = {
        'train': coco_dir / "train2017",
        'val': coco_dir / "val2017",
        'test': coco_dir / "test2017"
    }
    
    total_processed = 0
    
    # Process each split
    for split, ann_file in annotation_files.items():
        if not ann_file.exists():
            print(f"Warning: Annotation file not found - {ann_file}")
            continue
        
        print(f"\nProcessing {split} split...")
        print(f"Annotation file: {ann_file}")
        print(f"Image directory: {image_dirs[split]}")
        
        # Load COCO annotations
        try:
            with open(ann_file, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error loading JSON: {e}")
            continue
        
        # Check structure
        if 'images' not in data or 'annotations' not in data:
            print(f"Invalid COCO format in {ann_file}")
            continue
        
        # Map image IDs to image info
        images = {}
        for img in data['images']:
            # Convert string IDs to integers if needed
            img_id = int(img['id']) if isinstance(img['id'], str) else img['id']
            # Convert dimensions to integers
            img['width'] = int(img['width']) if isinstance(img['width'], str) else img['width']
            img['height'] = int(img['height']) if isinstance(img['height'], str) else img['height']
            images[img_id] = img
        
        # Group annotations by image
        annotations_by_image = {}
        for ann in data['annotations']:
            img_id = int(ann['image_id']) if isinstance(ann['image_id'], str) else ann['image_id']
            if img_id not in annotations_by_image:
                annotations_by_image[img_id] = []
            
            # Convert bbox values to floats
            if 'bbox' in ann and isinstance(ann['bbox'], list):
                ann['bbox'] = [float(x) for x in ann['bbox']]
            
            # Convert category_id to int
            if 'category_id' in ann:
                ann['category_id'] = int(ann['category_id'])
            
            annotations_by_image[img_id].append(ann)
        
        # Process each image
        processed_count = 0
        skipped_count = 0
        
        for img_id, anns in tqdm(annotations_by_image.items(), desc=f"Processing {split}"):
            if img_id not in images:
                skipped_count += 1
                continue
            
            img_info = images[img_id]
            img_filename = img_info['file_name']
            
            # Source image path
            src_img_path = image_dirs[split] / img_filename
            
            if not src_img_path.exists():
                print(f"  Warning: Image not found - {src_img_path}")
                skipped_count += 1
                continue
            
            # Destination paths
            dest_img_path = output_dir / 'images' / split / img_filename
            dest_label_path = output_dir / 'labels' / split / f"{Path(img_filename).stem}.txt"
            
            # Copy image
            try:
                shutil.copy2(src_img_path, dest_img_path)
            except Exception as e:
                print(f"  Error copying {src_img_path}: {e}")
                skipped_count += 1
                continue
            
            # Get image dimensions
            img_width = int(img_info['width'])
            img_height = int(img_info['height'])
            
            # Create YOLO format labels
            with open(dest_label_path, 'w') as f:
                for ann in anns:
                    # Get category ID (CarDD uses 1-indexed categories)
                    category_id = ann['category_id']
                    
                    # CarDD categories: 1-8, convert to 0-7 for YOLO
                    yolo_class = int(category_id) - 1
                    
                    # Skip if class ID is out of range (0-7)
                    if yolo_class < 0 or yolo_class > 7:
                        continue
                    
                    # Get bounding box [x, y, width, height]
                    bbox = ann['bbox']
                    
                    # Ensure bbox has 4 values
                    if len(bbox) != 4:
                        continue
                    
                    # Convert to float
                    x, y, w, h = map(float, bbox)
                    
                    # Convert to YOLO format (normalized)
                    x_center = (x + w / 2) / img_width
                    y_center = (y + h / 2) / img_height
                    width = w / img_width
                    height = h / img_height
                    
                    # Ensure values are within [0, 1]
                    x_center = max(0.0, min(1.0, x_center))
                    y_center = max(0.0, min(1.0, y_center))
                    width = max(0.0, min(1.0, width))
                    height = max(0.0, min(1.0, height))
                    
                    # Skip if invalid
                    if width <= 0 or height <= 0:
                        continue
                    
                    # Write to label file
                    f.write(f"{yolo_class} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
            
            processed_count += 1
        
        total_processed += processed_count
        print(f"  Processed: {processed_count} images")
        print(f"  Skipped: {skipped_count} images")
    
    # Create dataset.yaml file
    if total_processed > 0:
        create_dataset_yaml(output_dir)
        print(f"\n✓ Dataset preparation successful!")
        print(f"  Total images processed: {total_processed}")
        print(f"  Output directory: {output_dir}")
        return str(output_dir / 'dataset.yaml')
    else:
        print("\n✗ No images were processed!")
        return None

def create_dataset_yaml(output_dir: Path):
    """Create dataset YAML file for YOLO"""
    
    # CarDD damage categories
    damage_categories = [
        'scratch',      # 0
        'dent',         # 1
        'crack',        # 2
        'glass_break',  # 3
        'lamp_break',   # 4
        'tire_flat',    # 5
        'deformation',  # 6
        'chip'          # 7
    ]
    
    yaml_content = {
        'path': str(output_dir.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': len(damage_categories),
        'names': damage_categories
    }
    
    yaml_file = output_dir / 'dataset.yaml'
    with open(yaml_file, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    
    print(f"Created dataset.yaml at {yaml_file}")
    return yaml_file

def check_dataset(output_path="datasets/cardd_yolo"):
    """Check the prepared dataset"""
    
    output_dir = Path(output_path)
    
    print("\n" + "="*60)
    print("DATASET CHECK")
    print("="*60)
    
    for split in ['train', 'val', 'test']:
        images_dir = output_dir / 'images' / split
        labels_dir = output_dir / 'labels' / split
        
        if not images_dir.exists():
            print(f"\n{split.upper()} images directory not found!")
            continue
        
        # Count images
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_count = 0
        for ext in image_extensions:
            image_count += len(list(images_dir.glob(ext)))
        
        # Count labels
        label_count = len(list(labels_dir.glob('*.txt')))
        
        print(f"\n{split.upper()} Split:")
        print(f"  Images: {image_count}")
        print(f"  Labels: {label_count}")
        
        # Count annotations per class
        if label_count > 0:
            class_counts = {}
            for label_file in labels_dir.glob('*.txt'):
                try:
                    with open(label_file, 'r') as f:
                        for line in f:
                            if line.strip():
                                parts = line.strip().split()
                                if len(parts) >= 1:
                                    class_id = int(parts[0])
                                    class_counts[class_id] = class_counts.get(class_id, 0) + 1
                except:
                    continue
            
            if class_counts:
                print(f"  Total annotations: {sum(class_counts.values())}")
                print(f"  Class distribution:")
                for class_id in sorted(class_counts.keys()):
                    class_name = ['scratch', 'dent', 'crack', 'glass_break', 
                                 'lamp_break', 'tire_flat', 'deformation', 'chip'][class_id]
                    print(f"    Class {class_id} ({class_name}): {class_counts[class_id]}")
            else:
                print(f"  No annotations found in label files")
        else:
            print(f"  No label files found")

def main():
    parser = argparse.ArgumentParser(description='Prepare CarDD dataset for YOLO training')
    parser.add_argument('--input', type=str, default='CarDD_release',
                       help='Path to CarDD dataset root directory')
    parser.add_argument('--output', type=str, default='datasets/cardd_yolo',
                       help='Output directory for YOLO format')
    parser.add_argument('--check', action='store_true',
                       help='Check dataset only (do not convert)')
    
    args = parser.parse_args()
    
    if args.check:
        check_dataset(args.output)
    else:
        print("="*60)
        print("CARDD DATASET CONVERSION TO YOLO FORMAT")
        print("="*60)
        
        dataset_yaml = convert_cardd_to_yolo_fixed(args.input, args.output)
        
        if dataset_yaml:
            check_dataset(args.output)
            print(f"\n✓ Conversion successful!")
            print(f"Dataset YAML: {dataset_yaml}")
            print(f"\nNext step: Train the model with:")
            print(f"  python train_model.py")
        else:
            print("\n✗ Conversion failed!")

if __name__ == "__main__":
    main()