"""
AI Vehicle Damage Assessment System (Damage-Only Focus)
Integrates YOLOv8 ONNX Detection + Trained RandomForest Cost Model
Multi-image support | High-visibility annotations | CPU/GPU ready
Aggressive duplicate removal | Formal Insurance-Ready PDF Report
"""
import os
import sys
import json
import cv2
import numpy as np
import pandas as pd
import logging
import tempfile
from datetime import datetime
from pathlib import Path
from PIL import Image

# =============================================================================
# ⚠️ CRITICAL: PyTorch 2.6+ Compatibility Patch
# =============================================================================
import os, importlib, torch, torch.serialization
os.environ['TORCH_FORCE_WEIGHTS_ONLY_LOAD'] = '0'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def _apply_pytorch_patch():
    importlib.reload(torch.serialization)
    original_load = torch.serialization.load
    def safe_load(*args, **kwargs):
        kwargs['weights_only'] = False
        return original_load(*args, **kwargs)
    torch.load = safe_load
    torch.serialization.load = safe_load

_apply_pytorch_patch()

import streamlit as st
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    ort = None

from sklearn.preprocessing import LabelEncoder
import joblib

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# STREAMLIT CONFIG
# =============================================================================
st.set_page_config(
    page_title="🚗 AI Vehicle Damage Assessment",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CONFIGURATION & CONSTANTS
# =============================================================================
DAMAGE_CLASSES = ['dent', 'scratch', 'crack', 'glass_shatter', 'lamp_broken', 'tire_flat']
NUM_CLASSES = len(DAMAGE_CLASSES)

BASE_COSTS = {
    'dent': {'part': 200, 'labor': 150, 'paint': 100},
    'scratch': {'part': 100, 'labor': 100, 'paint': 150},
    'crack': {'part': 300, 'labor': 200, 'paint': 100},
    'glass_shatter': {'part': 500, 'labor': 150, 'paint': 50},
    'lamp_broken': {'part': 150, 'labor': 100, 'paint': 50},
    'tire_flat': {'part': 100, 'labor': 50, 'paint': 0}
}

# =============================================================================
# MODEL LOADING
# =============================================================================
@st.cache_resource
def load_onnx_model(model_path: str = 'runs/detect/runs/train/damage_detect_v2/weights/best.onnx'):
    if not ONNX_AVAILABLE:
        return None, {"status": "error", "message": "onnxruntime not installed", "details": "pip install onnxruntime"}
    if not os.path.exists(model_path):
        return None, {"status": "error", "message": "Model file missing", "details": f"Expected: {model_path}"}
    
    try:
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.intra_op_num_threads = 4
        providers = ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
        
        session = ort.InferenceSession(model_path, sess_options=session_options, providers=providers)
        active = session.get_providers()[0]
        out_shape = session.get_outputs()[0].shape
        out_type = session.get_outputs()[0].type
        
        status = "warning" if active == 'CPUExecutionProvider' else "success"
        msg = f"Loaded on CPU (GPU not available)" if active == 'CPUExecutionProvider' else f"Loaded on {active.replace('ExecutionProvider','')}"
        details = "Install CUDA 12 + cuDNN 9 for GPU acceleration" if active == 'CPUExecutionProvider' else "GPU acceleration active"
        
        if 'float16' in out_type.lower() and active == 'CPUExecutionProvider':
            msg += " | ⚠️ FP16 on CPU"
            details += " | FP16 models run best on GPU"
            
        return session, {"status": status, "message": msg, "provider": active, "output_shape": out_shape, 
                         "input_name": session.get_inputs()[0].name, "details": details}
    except Exception as e:
        return None, {"status": "error", "message": "Runtime error", "details": str(e)[:150]}

@st.cache_resource
def load_cost_model(base_path: str = 'cost_model.pkl'):
    required = [base_path, 'model_encoder.pkl', 'part_encoder.pkl']
    if not all(os.path.exists(f) for f in required):
        return None, None, None
    try:
        model = joblib.load(base_path)
        model_enc = joblib.load('model_encoder.pkl')
        part_enc = joblib.load('part_encoder.pkl')
        logger.info("✅ Cost model & encoders loaded")
        return model, model_enc, part_enc
    except Exception as e:
        logger.error(f"❌ Cost model load failed: {e}")
        return None, None, None

# =============================================================================
# INFERENCE UTILITIES
# =============================================================================
def preprocess_image(image_bgr: np.ndarray, img_size: int = 640):
    h, w = image_bgr.shape[:2]
    scale = img_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    padded = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    padded[:new_h, :new_w] = resized
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.transpose(rgb, (2, 0, 1))[None, ...], scale, (h, w)

def postprocess_yolov8_onnx(output, orig_shape, scale, conf_thresh, iou_thresh, num_classes=NUM_CLASSES):
    out = np.squeeze(output, axis=0)
    total_ch = out.shape[0]
    use_nc = min(total_ch - 4, num_classes) if total_ch > 4 else total_ch - 4
    
    detections = []
    img_h, img_w = orig_shape
    for i in range(out.shape[1]):
        box = out[:, i]
        x_c, y_c, w, h = box[:4]
        probs = box[4:4+use_nc]
        if len(probs) == 0: continue
        cls_id = np.argmax(probs)
        conf = float(probs[cls_id])
        if conf < conf_thresh or cls_id >= len(DAMAGE_CLASSES): continue
        
        x1 = ((x_c - w/2) * img_w) / scale
        y1 = ((y_c - h/2) * img_h) / scale
        x2 = ((x_c + w/2) * img_w) / scale
        y2 = ((y_c + h/2) * img_h) / scale
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_w, x2), min(img_h, y2)
        
        detections.append({'bbox': [float(x1), float(y1), float(x2), float(y2)], 
                           'confidence': conf, 'class_id': cls_id, 'class_name': DAMAGE_CLASSES[cls_id]})
    return nms(detections, iou_thresh) if detections else []

def nms(detections, iou_thresh):
    """Standard Non-Maximum Suppression"""
    if not detections:
        return []
    detections.sort(key=lambda x: x['confidence'], reverse=True)
    keep = []
    while detections:
        best = detections.pop(0)
        keep.append(best)
        detections = [d for d in detections if _iou(best['bbox'], d['bbox']) < iou_thresh]
    return keep

def _iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0: return 0.0
    area1 = (box1[2]-box1[0])*(box1[3]-box1[1])
    area2 = (box2[2]-box2[0])*(box2[3]-box2[1])
    return inter / (area1 + area2 - inter)

def merge_overlapping_detections(detections, merge_iou=0.65, max_per_class=2):
    """
    ✅ AGGRESSIVE DUPLICATE REMOVAL:
    - Merge highly overlapping boxes of SAME damage type
    - Limit max detections per class to prevent inflation
    - Keep only highest confidence detection per merged group
    """
    if not detections:
        return detections, []
    
    # Group by class first
    by_class = {}
    for det in detections:
        cls = det['class_name']
        if cls not in by_class:
            by_class[cls] = []
        by_class[cls].append(det)
    
    merged = []
    raw_count = len(detections)
    
    for cls_name, class_dets in by_class.items():
        # Sort by confidence descending
        class_dets.sort(key=lambda x: x['confidence'], reverse=True)
        used = [False] * len(class_dets)
        
        for i in range(len(class_dets)):
            if used[i]:
                continue
                
            current = class_dets[i]
            same_class_group = [current]
            used[i] = True
            
            # Find overlapping detections of same class
            for j in range(i + 1, len(class_dets)):
                if used[j]:
                    continue
                if _iou(current['bbox'], class_dets[j]['bbox']) > merge_iou:
                    same_class_group.append(class_dets[j])
                    used[j] = True
            
            # Merge group
            if len(same_class_group) > 1:
                avg_x1 = np.mean([d['bbox'][0] for d in same_class_group])
                avg_y1 = np.mean([d['bbox'][1] for d in same_class_group])
                avg_x2 = np.mean([d['bbox'][2] for d in same_class_group])
                avg_y2 = np.mean([d['bbox'][3] for d in same_class_group])
                max_conf = max(d['confidence'] for d in same_class_group)
                
                # Keep best detection's metadata
                best_det = max(same_class_group, key=lambda x: x['confidence'])
                merged_det = best_det.copy()
                merged_det['bbox'] = [avg_x1, avg_y1, avg_x2, avg_y2]
                merged_det['confidence'] = max_conf
                merged_det['merged_count'] = len(same_class_group)
                merged_det['was_merged'] = True
                merged.append(merged_det)
            else:
                current['merged_count'] = 1
                current['was_merged'] = False
                merged.append(current)
    
    # ✅ LIMIT: Keep only top-N detections per class by confidence
    final = []
    by_class_final = {}
    for det in merged:
        cls = det['class_name']
        if cls not in by_class_final:
            by_class_final[cls] = []
        by_class_final[cls].append(det)
    
    for cls_name, class_dets in by_class_final.items():
        # Sort by confidence and keep only top max_per_class
        class_dets.sort(key=lambda x: x['confidence'], reverse=True)
        final.extend(class_dets[:max_per_class])
    
    # Sort final list by confidence for consistent ordering
    final.sort(key=lambda x: x['confidence'], reverse=True)
    
    return final, raw_count

def calculate_severity(confidence, bbox, img_shape):
    area = (bbox[2]-bbox[0])*(bbox[3]-bbox[1])
    ratio = min(area / (img_shape[0]*img_shape[1]), 1.0)
    score = min(confidence * 0.7 + ratio * 0.3, 1.0)
    if score < 0.25: return 'low', round(score, 3)
    if score < 0.60: return 'medium', round(score, 3)
    return 'high', round(score, 3)

def safe_transform(encoder, value, fallback_idx=0):
    try:
        return encoder.transform([str(value).lower().strip()])[0]
    except Exception:
        known = encoder.classes_
        return encoder.transform([known[fallback_idx]])[0] if len(known) > fallback_idx else 0

def map_damage_to_part_heuristic(bbox, img_shape, damage_type):
    h, w = img_shape[:2]
    cx = (bbox[0] + bbox[2]) / 2 / w
    cy = (bbox[1] + bbox[3]) / 2 / h
    
    if damage_type == 'glass_shatter': return 'windshield'
    if damage_type == 'lamp_broken': return 'left_headlight' if cx < 0.5 else 'right_headlight'
    if damage_type == 'tire_flat': return 'left_wheel' if cx < 0.5 else 'right_wheel'
    if damage_type in ['dent', 'scratch', 'crack']:
        if cy < 0.3: return 'hood'
        if cy > 0.7: return 'rear_bumper'
        if cx < 0.3: return 'left_fender'
        if cx > 0.7: return 'right_fender'
        return 'front_bumper'
    return 'unknown_part'

def predict_cost(det, vehicle_info, cost_model, model_enc, part_enc):
    dmg_type = det.get('class_name', 'dent')
    sev_score = det.get('severity_score', 0.5)
    part_name = det.get('estimated_part', 'unknown_part')
    
    if cost_model and model_enc and part_enc:
        try:
            m_enc = safe_transform(model_enc, vehicle_info.get('model', 'unknown'))
            p_enc = safe_transform(part_enc, part_name)
            features = np.array([[m_enc, p_enc, sev_score]])
            pred = float(cost_model.predict(features)[0])
            return {
                'total_cost': round(max(pred, 0), 2),
                'method': 'ml_model',
                'breakdown': _cost_breakdown(pred, dmg_type)
            }
        except Exception as e:
            logger.warning(f"ML cost prediction failed: {e}")
    
    base = BASE_COSTS.get(dmg_type, BASE_COSTS['dent'])
    severity_label = det.get('severity', 'medium')
    mult = {'low': 0.8, 'medium': 1.0, 'high': 1.5}.get(severity_label, 1.0)
    total = (base['part'] + base['labor'] + base['paint']) * mult
    return {
        'total_cost': round(total, 2),
        'method': 'rule_based',
        'breakdown': {
            'part_cost': round(base['part'] * mult, 2),
            'labor_cost': round(base['labor'] * mult, 2),
            'paint_cost': round(base['paint'] * mult, 2)
        }
    }

def _cost_breakdown(total, dmg_type):
    ratios = {
        'glass_shatter': (0.7, 0.2, 0.1), 'tire_flat': (0.8, 0.2, 0.0),
        'lamp_broken': (0.6, 0.3, 0.1), 'crack': (0.4, 0.4, 0.2),
        'dent': (0.3, 0.5, 0.2), 'scratch': (0.2, 0.3, 0.5)
    }.get(dmg_type, (0.4, 0.4, 0.2))
    return {
        'part_cost': round(total * ratios[0], 2),
        'labor_cost': round(total * ratios[1], 2),
        'paint_cost': round(total * ratios[2], 2)
    }

def draw_detections(image_bgr, detections):
    img = image_bgr.copy()
    colors = {'low': (0, 200, 0), 'medium': (0, 165, 255), 'high': (0, 0, 255)}
    
    for det in detections:
        x1, y1, x2, y2 = map(int, det['bbox'])
        sev = det.get('severity', 'medium')
        color = colors.get(sev, (255, 255, 255))
        
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        
        label_lines = [
            f"{det['class_name'].replace('_', ' ').title()}",
            f"{det['confidence']:.0%} | {sev.upper()}"
        ]
        if 'cost_estimate' in det:
            label_lines.append(f"Rs.{det['cost_estimate']['total_cost']:,.0f}")
        if det.get('was_merged', False):
            label_lines.append(f"({det.get('merged_count',1)} merged)")
            
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale, thickness = 0.6, 2
        (tw, th), _ = cv2.getTextSize(label_lines[0], font, scale, thickness)
        label_h = th * len(label_lines) + 10
        
        cv2.rectangle(img, (x1, y1-label_h), (x1+tw+15, y1), color, -1)
        for i, line in enumerate(label_lines):
            cv2.putText(img, line, (x1+5, y1-5-(len(label_lines)-1-i)*th), 
                       font, scale, (0,0,0), thickness, cv2.LINE_AA)
    return img

def generate_recommendations(detections):
    recs = []
    if not detections: return ["✨ No damages detected - vehicle appears to be in good condition!"]
    if any(d.get('severity')=='high' for d in detections): recs.append("🚨 IMMEDIATE ACTION: High severity requires urgent repair")
    if any(d['class_name']=='crack' for d in detections): recs.append("⚠️ Cracks can propagate - repair immediately")
    if any(d['class_name']=='glass_shatter' for d in detections): recs.append("🪟 Broken glass requires replacement for safety")
    if any(d['class_name']=='tire_flat' for d in detections): recs.append("🛞 Flat tire detected - do not drive until replaced")
    if len([d for d in detections if d.get('severity')=='medium'])>=2: recs.append("🔧 Schedule medium severity repairs within 1-2 weeks")
    if all(d.get('severity')=='low' for d in detections): recs.append("✅ Minor damages - maintenance recommended when convenient")
    return recs

# =============================================================================
# PDF REPORT GENERATION
# =============================================================================
def generate_formal_pdf(batch_results, vehicle_info):
    try:
        from fpdf import FPDF
    except ImportError:
        raise ImportError("fpdf2 library is required. Run: pip install fpdf2")

    if not batch_results:
        raise ValueError("No batch results provided to PDF generator. Please analyze images first.")
    
    total_detections = sum(len(r.get('detections', [])) for r in batch_results)
    
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 20)
    pdf.ln(20)
    pdf.cell(0, 10, "Vehicle Damage Assessment Report", align="C")
    pdf.set_font("Helvetica", "I", 12)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 10, "Automated AI Analysis & Cost Estimation", align="C")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(15)
    
    pdf.set_fill_color(240, 240, 240)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, " Report Details ", ln=True, fill=True)
    pdf.set_font("Helvetica", "", 11)
    
    report_id = f"VDA-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    pdf.cell(40, 6, "Report ID:")
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 6, report_id, ln=True)
    
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(40, 6, "Date Generated:")
    pdf.cell(0, 6, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), ln=True)
    pdf.ln(5)
    
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, " Vehicle Information ", ln=True, fill=True)
    pdf.set_font("Helvetica", "", 11)
    
    pdf.cell(40, 6, "Brand:")
    pdf.cell(0, 6, str(vehicle_info.get('brand', 'N/A')), ln=True)
    pdf.cell(40, 6, "Model:")
    pdf.cell(0, 6, str(vehicle_info.get('model', 'N/A')), ln=True)
    pdf.cell(40, 6, "Year:")
    pdf.cell(0, 6, str(vehicle_info.get('year', 'N/A')), ln=True)
    pdf.cell(40, 6, "Mileage:")
    pdf.cell(0, 6, f"{vehicle_info.get('mileage', 0):,} km", ln=True)
    pdf.ln(10)
    
    pdf.set_fill_color(230, 245, 250)
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, " Financial Summary ", ln=True, fill=True)
    
    total_cost = 0
    for r in batch_results:
        for d in r.get('detections', []):
            ce = d.get('cost_estimate', {})
            total_cost += ce.get('total_cost', 0)
    
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(0, 120, 0)
    pdf.cell(0, 15, f"Total Estimated Cost: Rs. {total_cost:,.2f}", align="C", ln=True)
    pdf.set_text_color(0, 0, 0)
    
    pdf.set_font("Helvetica", "", 12)
    pdf.cell(0, 8, f"Total Unique Detections: {total_detections}", align="C", ln=True)
    
    if total_detections == 0:
        pdf.ln(10)
        pdf.set_font("Helvetica", "I", 11)
        pdf.set_text_color(150, 150, 150)
        pdf.multi_cell(0, 6, "Note: No damages were detected in the analyzed images. This report serves as documentation of the AI analysis attempt.")
        pdf.set_text_color(0, 0, 0)
    
    pdf.ln(10)
    
    pdf.set_fill_color(240, 240, 240)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, " Damage Breakdown ", ln=True, fill=True)
    
    col_widths = [35, 20, 25, 25, 30, 40]
    headers = ["Type", "Severity", "Conf.", "Method", "Cost (Rs.)", "Image"]
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_fill_color(220, 220, 220)
    for i, h in enumerate(headers):
        pdf.cell(col_widths[i], 7, h, border=1, fill=True, align="C")
    pdf.ln()
    
    pdf.set_font("Helvetica", "", 8)
    row_count = 0
    
    for r_idx, res in enumerate(batch_results):
        detections = res.get('detections', [])
        if not detections:
            continue
            
        for d in detections:
            if pdf.get_y() > 260:
                pdf.add_page()
                pdf.set_font("Helvetica", "B", 9)
                pdf.set_fill_color(220, 220, 220)
                for i, h in enumerate(headers):
                    pdf.cell(col_widths[i], 7, h, border=1, fill=True, align="C")
                pdf.ln()
                pdf.set_font("Helvetica", "", 8)
            
            ce = d.get('cost_estimate', {})
            data = [
                d.get('class_name', 'unknown').replace('_', ' ').title(),
                d.get('severity', 'N/A').upper(),
                f"{d.get('confidence', 0):.0%}",
                ce.get('method', 'N/A')[:8] if ce else 'N/A',
                f"{ce.get('total_cost', 0):,.2f}" if ce else '0.00',
                res.get('filename', 'unknown')[:20]
            ]
            for i, val in enumerate(data):
                pdf.cell(col_widths[i], 6, str(val), border=1, align="C")
            pdf.ln()
            row_count += 1
    
    if row_count == 0:
        pdf.cell(0, 10, "No damage detections to display in table", align="C", ln=True)
            
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Visual Evidence (Annotated Images)", align="C", ln=True)
    pdf.ln(5)
    
    temp_files = []
    img_count = 0
    
    for res in batch_results:
        detections = res.get('detections', [])
        
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 8, f"Image File: {res.get('filename', 'unknown')}", ln=True)
        pdf.set_font("Helvetica", "", 9)
        
        if detections:
            pdf.cell(0, 6, f"Detected {len(detections)} damage(s).", ln=True)
        else:
            pdf.set_text_color(150, 150, 150)
            pdf.cell(0, 6, "No damages detected in this image.", ln=True)
            pdf.set_text_color(0, 0, 0)
        
        img_array = res.get('processed')
        if img_array is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                try:
                    pil_img = Image.fromarray(img_array)
                    pil_img.save(tmp.name, format="JPEG")
                    temp_files.append(tmp.name)
                    pdf.image(tmp.name, w=180)
                    img_count += 1
                except Exception as e:
                    pdf.cell(0, 10, f"[Image embedding error: {str(e)}]", ln=True)
                    logger.error(f"Failed to embed image {res.get('filename')}: {e}")
        else:
            pdf.cell(0, 10, "[No processed image available]", ln=True)
            
        pdf.ln(5)
        
        if pdf.get_y() > 150:
            pdf.add_page()
            pdf.set_font("Helvetica", "B", 16)
            pdf.cell(0, 10, "Visual Evidence (Continued)", align="C", ln=True)
            pdf.ln(5)

    for tf in temp_files:
        try: os.remove(tf)
        except: pass

    pdf.add_page()
    pdf.set_font("Helvetica", "I", 10)
    disclaimer = (
        "DISCLAIMER: This report is generated by an automated AI system based on computer vision analysis. "
        "The estimated costs are indicative and may vary based on actual physical inspection, "
        "parts availability, local labor rates, and vehicle condition. "
        "This report should be verified by a certified vehicle inspector or insurance adjuster "
        "before processing any claims."
    )
    pdf.multi_cell(0, 6, disclaimer)
    
    pdf.ln(15)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 10, "Generated by AI Vehicle Damage Assessment System", align="C", ln=True)
    pdf.cell(0, 6, f"Report ID: {report_id} | {datetime.now().strftime('%Y-%m-%d')}", align="C", ln=True)
    
    logger.info(f"PDF generated: {row_count} table rows, {img_count} images, {total_detections} total detections")
    
    return pdf.output(dest='S').encode('latin-1')

# =============================================================================
# STREAMLIT UI
# =============================================================================
def main():
    st.title("🚗 AI Vehicle Damage Assessment System")
    st.markdown("*Upload vehicle images to detect damage, estimate repair costs, and generate insurance-ready reports*")
    
    if 'batch_results' not in st.session_state: 
        st.session_state.batch_results = []
    if 'model_status' not in st.session_state: 
        st.session_state.model_status = "error"
    
    onnx_session, model_info = load_onnx_model('runs/detect/runs/train/damage_detect_v2/weights/best.onnx')
    cost_model, model_enc, part_enc = load_cost_model('cost_model.pkl')
    
    if model_info and model_info["status"] in ["success", "warning"]:
        st.session_state.model_status = "ready"
    else:
        st.session_state.model_status = "error"
    
    with st.sidebar:
        st.header("⚙️ System Status")
        if st.session_state.model_status == "ready":
            st.success(f"✅ Detection Model: {model_info['message']}")
            st.caption(f"Provider: {model_info['provider']}")
            st.info(model_info.get('details', ''))
        else:
            st.error("❌ Detection Model: Failed")
            st.code(model_info.get('details', '') if model_info else 'Unknown error')
            
        st.divider()
        if cost_model:
            st.success("✅ Cost Model: 3-Feature ML Loaded")
        else:
            st.warning("⚠️ Cost Model: Rule-Based Fallback Active")
            
        st.divider()
        st.header("🎛️ Settings")
        
        # Detection thresholds
        conf_thresh = st.slider("Confidence Threshold", 0.1, 0.9, 0.35, 0.05,
                               help="Higher = fewer false positives (recommended: 0.30-0.40)")
        iou_thresh = st.slider("NMS IoU Threshold", 0.3, 0.8, 0.60, 0.05,
                              help="Higher = more aggressive duplicate removal (recommended: 0.55-0.65)")
        
        # ✅ NEW: Aggressive duplicate removal controls
        st.subheader("🔒 Duplicate Prevention")
        merge_iou = st.slider("Merge Overlap Threshold", 0.5, 0.9, 0.65, 0.05,
                             help="Boxes with IoU > this value (same class) will be merged")
        max_per_class = st.slider("Max Detections per Damage Type", 1, 5, 2, 1,
                                 help="Limit: max N detections of same type per image")
        
        st.info(f"💡 **Recommended for minimal duplicates:** Conf=0.35, NMS=0.60, Merge=0.65, Max=2")
        
        st.divider()
        st.markdown("### 📊 Quick Stats")
        st.metric("Damage Classes", len(DAMAGE_CLASSES))
        st.metric("Inference Engine", "ONNX Runtime")
        st.metric("Cost Engine", "ML" if cost_model else "Rule-Based")
        st.metric("Deduplication", "Aggressive Mode")
        
        st.metric("Images Analyzed", len(st.session_state.batch_results))
        total_dets = sum(len(r.get('detections', [])) for r in st.session_state.batch_results)
        st.metric("Total Detections", total_dets)
        
        if model_info and model_info.get('provider') == 'CPUExecutionProvider':
            with st.expander("🚀 Enable GPU Acceleration"):
                st.markdown("1. Install CUDA 12 + cuDNN 9\n2. Add to PATH\n3. Restart app")
                st.code("pip install onnxruntime-gpu")
    
    tab1, tab2, tab3 = st.tabs(["📤 Upload & Analyze", "📋 Results", "📄 Report"])
    
    with tab1:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("📷 Upload Vehicle Images")
            uploaded = st.file_uploader("Choose images...", type=['jpg', 'jpeg', 'png', 'bmp'], accept_multiple_files=True)
            
            if uploaded:
                st.markdown(f"📂 **{len(uploaded)}** images selected")
                with st.expander("🚙 Vehicle Info (Applies to all images)", expanded=True):
                    c1, c2 = st.columns(2)
                    brand = c1.selectbox("Brand", ['Maruti Suzuki', 'Hyundai', 'Tata', 'Honda', 'Mahindra', 'Toyota', 'Other'])
                    model = c2.text_input("Model", placeholder="e.g., Swift, Creta", key="model_input")
                    c3, c4 = st.columns(2)
                    year = c3.number_input("Year", 2010, 2025, 2020)
                    mileage = c4.number_input("Mileage (km)", 0, 500000, 50000)
                
                if st.button("🔍 Analyze All Images", type="primary", use_container_width=True, disabled=(st.session_state.model_status != "ready")):
                    if not model.strip():
                        st.error("⚠️ Enter vehicle model for cost estimation")
                    else:
                        st.session_state.batch_results = []
                        progress = st.progress(0, text="🔎 Processing images...")
                        
                        for i, file in enumerate(uploaded):
                            try:
                                img = Image.open(file).convert('RGB')
                                img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                                tensor, scale, orig = preprocess_image(img_bgr)
                                outs = onnx_session.run(None, {model_info['input_name']: tensor})
                                
                                # Get raw detections
                                raw_dets = postprocess_yolov8_onnx(outs[0], orig, scale, conf_thresh, iou_thresh)
                                
                                # ✅ AGGRESSIVE DEDUPLICATION
                                dets, raw_count = merge_overlapping_detections(
                                    raw_dets, 
                                    merge_iou=merge_iou, 
                                    max_per_class=max_per_class
                                )
                                
                                # Enrich detections
                                for det in dets:
                                    sev, score = calculate_severity(det['confidence'], det['bbox'], orig)
                                    det['severity'] = sev
                                    det['severity_score'] = score
                                    det['estimated_part'] = map_damage_to_part_heuristic(det['bbox'], orig, det['class_name'])
                                    det['cost_estimate'] = predict_cost(det, {'brand': brand, 'model': model}, cost_model, model_enc, part_enc)
                                
                                vis = cv2.cvtColor(draw_detections(img_bgr, dets), cv2.COLOR_BGR2RGB)
                                
                                st.session_state.batch_results.append({
                                    'filename': file.name,
                                    'original': img,
                                    'processed': vis,
                                    'detections': dets,
                                    'raw_detection_count': raw_count,
                                    'deduped_detection_count': len(dets),
                                    'duplicates_removed': raw_count - len(dets),
                                    'vehicle_info': {'brand': brand, 'model': model, 'year': year, 'mileage': mileage}
                                })
                            except Exception as e:
                                logger.error(f"Failed to process {file.name}: {e}")
                                st.warning(f"⚠️ Failed to process {file.name}")
                            
                            progress.progress((i+1)/len(uploaded), text=f"✅ Processed {i+1}/{len(uploaded)} images")
                            
                        total_detected = sum(len(r['detections']) for r in st.session_state.batch_results)
                        total_removed = sum(r['duplicates_removed'] for r in st.session_state.batch_results)
                        st.success(f"🎉 Analysis Complete! {len(st.session_state.batch_results)} images, {total_detected} genuine damages detected.")
                        if total_removed > 0:
                            st.info(f"🔒 Removed {total_removed} duplicate detections to prevent cost inflation")
                        st.rerun()
                        
        with col2:
            st.subheader("💡 Tips")
            st.markdown("- 📸 Good lighting & focus\n- 🎯 Capture damage prominently\n- 📐 Include context\n- 🔄 Try different angles")
            st.divider()
            st.subheader("🔧 Supported")
            for c in DAMAGE_CLASSES: st.markdown(f"- `{c.replace('_', ' ').title()}`")
            st.divider()
            st.subheader("📈 Severity")
            st.markdown("- 🟢 Low | 🟡 Medium | 🔴 High")
            st.divider()
            st.success("🔒 **Aggressive Deduplication Active:** Overlapping same-class detections merged + limited to prevent cost inflation")
    
    with tab2:
        if not st.session_state.batch_results:
            st.info("👆 Upload and analyze images first")
        else:
            filenames = [r['filename'] for r in st.session_state.batch_results]
            selected_idx = st.selectbox("Select Image to View", range(len(filenames)), format_func=lambda i: filenames[i])
            res = st.session_state.batch_results[selected_idx]
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.image(res['processed'], caption=f"🖼️ {res['filename']} (🟢Low 🟡Medium 🔴High)", use_column_width=True)
            with col2:
                dets = res['detections']
                raw_count = res.get('raw_detection_count', len(dets))
                dedup_count = res.get('deduped_detection_count', len(dets))
                dups_removed = res.get('duplicates_removed', 0)
                
                st.metric("Genuine Damages", dedup_count)
                if dups_removed > 0:
                    st.metric("Duplicates Removed", dups_removed, delta_color="inverse")
                    st.success(f"✅ Prevented cost inflation: {raw_count} → {dedup_count}")
                if dets:
                    st.metric("High Severity", sum(1 for d in dets if d['severity']=='high'), delta_color="inverse")
                    total = sum(d['cost_estimate']['total_cost'] for d in dets)
                    st.metric("Est. Total Cost", f"Rs.{total:,.0f}")
                    st.metric("Avg Confidence", f"{np.mean([d['confidence'] for d in dets]):.2%}")
            
            if res['detections']:
                st.divider()
                st.subheader("🔍 Detection Details")
                table_data = []
                for i, det in enumerate(dets, 1):
                    row = {
                        '#': i, 'Type': det['class_name'].replace('_', ' ').title(),
                        'Confidence': f"{det['confidence']:.2%}", 'Severity': det['severity'].upper(),
                        'Score': det['severity_score'], 'Est. Cost': f"Rs.{det['cost_estimate']['total_cost']:,.0f}",
                        'Method': det['cost_estimate']['method']
                    }
                    if det.get('was_merged', False):
                        row['Merged'] = f"✅ ({det.get('merged_count',1)}→1)"
                    else:
                        row['Merged'] = "-"
                    table_data.append(row)
                    
                st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)
                
                st.subheader("📋 Breakdown")
                for i, det in enumerate(dets, 1):
                    ce = det['cost_estimate']
                    merge_info = f" | Merged from {det.get('merged_count',1)} overlapping detections" if det.get('was_merged') else ""
                    with st.expander(f"#{i} {det['class_name'].title()} | {det['severity'].upper()} | Rs.{ce['total_cost']:,.0f}{merge_info}"):
                        st.markdown(f"**Confidence**: `{det['confidence']:.2%}` | **Severity Score**: `{det['severity_score']}`\n**Box**: `[{' '.join(f'{c:.0f}' for c in det['bbox'])}]`\n**Est. Part**: `{det.get('estimated_part', 'N/A')}`\n**Cost**: Parts Rs.{ce['breakdown'].get('part_cost',0):,.0f} | Labor Rs.{ce['breakdown'].get('labor_cost',0):,.0f} | Paint Rs.{ce['breakdown'].get('paint_cost',0):,.0f}")
    
    with tab3:
        if not st.session_state.batch_results:
            st.info("👆 Complete analysis first")
        else:
            st.header("📄 Export Reports")
            
            total_images = len(st.session_state.batch_results)
            total_detections = sum(len(r.get('detections', [])) for r in st.session_state.batch_results)
            total_cost = sum(
                sum(d.get('cost_estimate', {}).get('total_cost', 0) for d in r.get('detections', [])) 
                for r in st.session_state.batch_results
            )
            
            st.info(f"📊 **Ready to export:** {total_images} images, {total_detections} genuine damages, Rs.{total_cost:,.0f} total cost")
            
            export_mode = st.radio("Export Scope", ["All Images", "Selected Image Only"], horizontal=True)
            
            if export_mode == "Selected Image Only":
                filtered_results = [st.session_state.batch_results[selected_idx]]
            else:
                filtered_results = st.session_state.batch_results
                
            vehicle_info = filtered_results[0]['vehicle_info'] if filtered_results else {}
            
            c1, c2, c3 = st.columns(3)
            
            with c1:
                json_data = [{'filename': r['filename'], 'vehicle': r['vehicle_info'], 'detections': len(r.get('detections', []))} for r in filtered_results]
                st.download_button("📄 Download JSON", json.dumps(json_data, indent=2, default=str), 
                                   f"report_{datetime.now().strftime('%Y%m%d')}.json", "application/json", use_container_width=True)
            
            with c2:
                csv_rows = [{'File': r['filename'], 'Type': d['class_name'], 'Severity': d['severity'], 'Cost': d.get('cost_estimate', {}).get('total_cost', 0)} 
                            for r in filtered_results for d in r.get('detections', [])]
                if csv_rows:
                    st.download_button("📊 Download CSV", pd.DataFrame(csv_rows).to_csv(index=False), 
                                       f"details_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv", use_container_width=True)
                else:
                    st.info("No detections to export as CSV")
            
            with c3:
                has_detections = any(len(r.get('detections', [])) > 0 for r in filtered_results)
                
                if not has_detections and total_detections == 0:
                    st.warning("⚠️ No damages detected in selected images. PDF will show 'No damages' report.")
                    st.info("💡 Try lowering confidence threshold or uploading clearer images")
                
                try:
                    pdf_bytes = generate_formal_pdf(filtered_results, vehicle_info)
                    if pdf_bytes and len(pdf_bytes) > 100:
                        st.download_button("📕 Download Formal PDF", pdf_bytes, 
                                           f"Damage_Report_{datetime.now().strftime('%Y%m%d')}.pdf", "application/pdf", use_container_width=True)
                        st.success("✅ PDF ready for download")
                    else:
                        st.error("❌ PDF generation returned empty or invalid data")
                        st.info("💡 Check logs for details or try re-analyzing images")
                except ImportError:
                    st.error("📦 PDF library missing. Run in terminal:\n`pip install fpdf2`")
                except ValueError as ve:
                    st.error(f"❌ Data Error: {ve}")
                    st.info("💡 Make sure you've analyzed images first")
                except Exception as e:
                    st.error(f"❌ PDF Generation failed: {type(e).__name__}: {e}")
                    logger.error(f"PDF generation error: {e}", exc_info=True)
                    st.code("Troubleshooting:\n1. Check fpdf2 is installed: pip install fpdf2\n2. Ensure images were analyzed\n3. Check console logs for details")

            st.divider()
            st.subheader("📋 Summary Preview")
            for r in filtered_results:
                dup_info = f" | Duplicates removed: {r.get('duplicates_removed', 0)}" if r.get('duplicates_removed', 0) > 0 else ""
                total = sum(d.get('cost_estimate', {}).get('total_cost', 0) for d in r.get('detections', []))
                st.markdown(f"**{r['filename']}** | Genuine Damages: `{len(r.get('detections', []))}`{dup_info} | Est. Cost: `Rs.{total:,.0f}`")
                for rec in generate_recommendations(r.get('detections', [])): 
                    st.markdown(f"- {rec}")

if __name__ == "__main__":
    logger.info(f"🚀 App started | Python: {sys.executable}")
    logger.info(f"ONNX Runtime: {ort.__version__ if ONNX_AVAILABLE else 'not installed'}")
    if ONNX_AVAILABLE:
        logger.info(f"Available providers: {ort.get_available_providers()}")
    main()