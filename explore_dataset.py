# explore_dataset.py
import os
import json
from pathlib import Path

def explore_car_dd_structure(dataset_root="CarDD_release"):
    """Explore the actual structure of your CarDD dataset"""
    
    print("="*60)
    print("EXPLORING CarDD DATASET STRUCTURE")
    print("="*60)
    
    dataset_root = Path(dataset_root)
    
    if not dataset_root.exists():
        print(f"Error: Dataset root not found: {dataset_root}")
        return
    
    # Walk through the directory
    print("\nDirectory structure:")
    for root, dirs, files in os.walk(dataset_root):
        level = root.replace(str(dataset_root), '').count(os.sep)
        indent = ' ' * 2 * level
        print(f'{indent}{os.path.basename(root)}/')
        subindent = ' ' * 2 * (level + 1)
        for file in files[:5]:  # Show first 5 files
            print(f'{subindent}{file}')
        if len(files) > 5:
            print(f'{subindent}... and {len(files)-5} more files')
    
    # Look for annotation files
    print("\n\nSearching for annotation files...")
    annotation_files = []
    for root, dirs, files in os.walk(dataset_root):
        for file in files:
            if file.endswith('.json'):
                annotation_files.append(Path(root) / file)
    
    print(f"Found {len(annotation_files)} JSON files:")
    for ann_file in annotation_files[:10]:  # Show first 10
        print(f"  - {ann_file}")
    
    # Look for images
    print("\n\nSearching for image directories...")
    image_dirs = []
    for root, dirs, files in os.walk(dataset_root):
        if any(f.lower().endswith(('.jpg', '.jpeg', '.png')) for f in files):
            image_dirs.append(root)
    
    print(f"Found {len(image_dirs)} directories with images:")
    for img_dir in image_dirs[:10]:  # Show first 10
        img_count = len([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        print(f"  - {img_dir} ({img_count} images)")
    
    # If we found annotation files, let's examine one
    if annotation_files:
        print("\n\nExamining first annotation file...")
        try:
            with open(annotation_files[0], 'r') as f:
                data = json.load(f)
            
            print(f"Keys in JSON: {list(data.keys())}")
            
            if 'images' in data:
                print(f"Number of images: {len(data['images'])}")
                if data['images']:
                    print(f"First image info: {data['images'][0]}")
            
            if 'annotations' in data:
                print(f"Number of annotations: {len(data['annotations'])}")
                if data['annotations']:
                    print(f"First annotation: {data['annotations'][0]}")
            
            if 'categories' in data:
                print(f"Categories: {data['categories']}")
                
        except Exception as e:
            print(f"Error reading annotation file: {e}")

if __name__ == "__main__":
    explore_car_dd_structure()