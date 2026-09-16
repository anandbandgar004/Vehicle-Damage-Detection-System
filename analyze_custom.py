import os, json, cv2, numpy as np
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO
import argparse

def load_model(model_path='damage_6classes.pt'):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Model not found: {model_path}")
    model = YOLO(model_path)
    classes = model.names  # ✅ Dynamically read exact trained classes
    return model, classes

def calculate_severity(conf, bbox, img_shape):
    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    total = img_shape[0] * img_shape[1]
    score = min(conf * 0.7 + (area / total) * 0.3, 1.0)
    if score < 0.3: return 'low', score
    if score < 0.7: return 'medium', score
    return 'high', score

def calculate_location(bbox, img_shape):
    h, w = img_shape[:2]
    cx, cy = (bbox[0] + bbox[2])/2, (bbox[1] + bbox[3])/2
    
    h_pos = 'left' if cx < w/3 else ('right' if cx > 2*w/3 else 'center')
    v_pos = 'top' if cy < h/3 else ('bottom' if cy > 2*h/3 else 'middle')
    return f"{v_pos}-{h_pos}"

def analyze_images(model, classes, image_paths, conf=0.25):
    print(f"\n📷 Processing {len(image_paths)} images...")
    all_results = []
    
    # ✅ NATIVE BATCH INFERENCE
    results = model.predict(source=image_paths, conf=conf, device='0' if torch.cuda.is_available() else 'cpu', verbose=False)
    
    for img_path, res in zip(image_paths, results):
        detections = []
        if res.boxes is not None:
            for box in res.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf_val = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = classes[cls_id] if cls_id < len(classes) else 'unknown'
                
                img_shape = res.orig_shape
                severity, sev_score = calculate_severity(conf_val, [x1,y1,x2,y2], img_shape)
                location = calculate_location([x1,y1,x2,y2], img_shape)
                
                detections.append({
                    'class_name': cls_name, 'confidence': conf_val,
                    'severity': severity, 'severity_score': sev_score,
                    'location': location, 'bbox': [float(x1), float(y1), float(x2), float(y2)]
                })
        all_results.append({'image': os.path.basename(img_path), 'detections': detections})
    return all_results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', type=str, required=True, help='Path to image(s) or directory')
    parser.add_argument('--model', type=str, default='damage_6classes.pt')
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--output', type=str, default='analysis_results')
    args = parser.parse_args()

    model, classes = load_model(args.model)
    print(f"✅ Model loaded. Classes: {list(classes.values())}")

    # Collect image paths
    img_paths = []
    p = Path(args.images)
    if p.is_file():
        img_paths = [str(p)]
    elif p.is_dir():
        exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        for ext in exts:
            img_paths.extend([str(f) for f in p.glob(ext)])
    
    if not img_paths:
        print("❌ No valid images found.")
        return

    results = analyze_images(model, classes, img_paths, args.conf)
    
    # Save JSON report
    os.makedirs(args.output, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = {'timestamp': ts, 'results': results}
    out_file = os.path.join(args.output, f'report_{ts}.json')
    with open(out_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Save visualizations
    viz_dir = os.path.join(args.output, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    color_map = {'low': (0,255,0), 'medium': (0,255,255), 'high': (0,0,255)}
    
    for img_path, res in zip(img_paths, results):
        img = cv2.imread(img_path)
        for det in res['detections']:
            x1,y1,x2,y2 = map(int, det['bbox'])
            clr = color_map[det['severity']]
            cv2.rectangle(img, (x1,y1), (x2,y2), clr, 2)
            label = f"{det['class_name']} {det['confidence']:.2f} ({det['severity']})"
            cv2.putText(img, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, clr, 2)
        out_path = os.path.join(viz_dir, f"viz_{os.path.basename(img_path)}")
        cv2.imwrite(out_path, img)

    print(f"\n✅ Analysis complete!")
    print(f"📄 Report: {out_file}")
    print(f"🖼️ Visualizations: {viz_dir}")

if __name__ == "__main__":
    import torch
    main()