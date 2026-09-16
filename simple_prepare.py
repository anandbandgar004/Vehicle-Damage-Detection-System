# simple_prepare.py
import os
import json
import shutil
from pathlib import Path
import yaml

def main():
    # Fixed paths based on your structure
    dataset_root = Path("CarDD_release") / "CarDD_release"  # Adjust if needed
    output_dir = Path("datasets/cardd_yolo")
    
    print(f"Dataset root: {dataset_root}")
    print(f"Output directory: {output_dir}")
    
    if not dataset_root.exists():
        print("Error: Dataset root not found!")
        return
    
    # Create output directories
    for split in ['train', 'val', 'test']:
        (output_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    # Process each split
    for split in ['train', 'val', 'test']:
        print(f"\nProcessing {split} split...")
        
        # Annotation file
        ann_file = dataset_root / "CarDD_COCO" / "annotations" / f"instances_{split}2017.json"
        
        if not ann_file.exists():
            print(f"  Warning: Annotation file not found - {ann_file}")
            continue
        
        # Load annotations
        with open(ann_file, 'r') as f:
            data = json.load(f)
        
        # Image directory
        img_dir = dataset_root / "CarDD_COCO" / f"{split}2017"
        
        if not img_dir.exists():
            print(f"  Warning: Image directory not found - {img_dir}")
            continue
        
        # Process images
        processed = 0
        for img_info in data['images']:
            img_filename = img_info['file_name']
            src_img_path = img_dir / img_filename
            
            if not src_img_path.exists():
                print(f"  Warning: Image not found - {src_img_path}")
                continue
            
            # Copy image
            dest_img_path = output_dir / 'images' / split / img_filename
            shutil.copy2(src_img_path, dest_img_path)
            
            # Get annotations for this image
            img_anns = [a for a in data['annotations'] if a['image_id'] == img_info['id']]
            
            # Create label file
            label_path = output_dir / 'labels' / split / f"{Path(img_filename).stem}.txt"
            
            with open(label_path, 'w') as f_label:
                for ann in img_anns:
                    # Class ID (1-indexed to 0-indexed)
                    class_id = int(ann['category_id']) - 1
                    
                    # Bounding box
                    bbox = [float(x) for x in ann['bbox']]
                    
                    # Image dimensions
                    img_width = float(img_info['width'])
                    img_height = float(img_info['height'])
                    
                    # Convert to YOLO format
                    x_center = (bbox[0] + bbox[2] / 2) / img_width
                    y_center = (bbox[1] + bbox[3] / 2) / img_height
                    width = bbox[2] / img_width
                    height = bbox[3] / img_height
                    
                    # Write to file
                    f_label.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
            
            processed += 1
        
        print(f"  Processed {processed} images")
    
    # Create dataset.yaml
    damage_categories = [
        'scratch', 'dent', 'crack', 'glass_break',
        'lamp_break', 'tire_flat', 'deformation', 'chip'
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
    
    print(f"\n✓ Created dataset.yaml at {yaml_file}")
    print(f"\nDataset preparation complete!")
    print(f"Output directory: {output_dir}")

if __name__ == "__main__":
    main()