import os, json, shutil, yaml
from pathlib import Path

def main():
    print("="*60)
    print("PREPARING CARDD DATASET - 6 DAMAGE CLASSES")
    print("="*60)

    dataset_root = None
    for path in [Path("CarDD_release/CarDD_release"), Path("CarDD_release"), Path(".")]:
        if (path / "CarDD_COCO").exists():
            dataset_root = path
            print(f"✓ Found dataset at: {dataset_root}")
            break
    if not dataset_root:
        print("❌ Error: Could not find CarDD dataset!")
        return

    # Exactly 6 classes matching your requirement
    damage_categories = ['dent', 'scratch', 'crack', 'glass_shatter', 'lamp_broken', 'tire_flat']
    print(f"\nTarget classes ({len(damage_categories)}): {damage_categories}")

    output_dir = Path("datasets/cardd_6classes")
    for split in ['train', 'val', 'test']:
        (output_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)

    coco_dir = dataset_root / "CarDD_COCO"
    
    # CarDD COCO category mapping (1-8) -> Your 6 classes
    # 1:dent, 2:scratch, 3:crack, 4:glass_shatter, 5:lamp_broken, 6:tire_flat, 7:deformation, 8:chip
    category_mapping = {1:0, 2:1, 3:2, 4:3, 5:4, 6:5, 7:0, 8:1} 

    for split in ['train', 'val', 'test']:
        print(f"\nProcessing {split} split...")
        ann_file = coco_dir / "annotations" / f"instances_{split}2017.json"
        img_dir = coco_dir / f"{split}2017"
        if not ann_file.exists() or not img_dir.exists():
            print(f"⚠️ Skipping {split} (missing files)")
            continue

        with open(ann_file, 'r') as f:
            data = json.load(f)

        processed = 0
        for img_info in data['images']:
            src_img = img_dir / img_info['file_name']
            if not src_img.exists(): continue

            dest_img = output_dir / 'images' / split / img_info['file_name']
            shutil.copy2(src_img, dest_img)

            img_anns = [a for a in data['annotations'] if a['image_id'] == img_info['id']]
            label_path = output_dir / 'labels' / split / f"{Path(img_info['file_name']).stem}.txt"

            with open(label_path, 'w') as f_label:
                for ann in img_anns:
                    cat_id = ann['category_id']
                    if cat_id not in category_mapping:
                        continue  # Skip unmapped categories
                    
                    yolo_cls = category_mapping[cat_id]
                    x, y, w, h = ann['bbox']
                    img_w, img_h = img_info['width'], img_info['height']
                    
                    xc = (x + w/2) / img_w
                    yc = (y + h/2) / img_h
                    wn = w / img_w
                    hn = h / img_h
                    
                    if wn > 0 and hn > 0:
                        f_label.write(f"{yolo_cls} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}\n")
            processed += 1
        print(f"✓ Processed {processed} images")

    yaml_path = output_dir / 'dataset.yaml'
    yaml_data = {
        'path': str(output_dir.absolute()),
        'train': 'images/train', 'val': 'images/val', 'test': 'images/test',
        'nc': 6,
        'names': damage_categories
    }
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_data, f, default_flow_style=False)
    print(f"\n✅ Dataset ready! Config saved to: {yaml_path}")

if __name__ == "__main__":
    main()