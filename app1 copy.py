"""
🚗 AI Vehicle Damage Assessment System
Integrates: YOLOv8 ONNX Detection + RandomForest Cost Estimation
Features: Multi-image batch processing • High-visibility annotations • Insurance-ready reports
"""
import os
import sys
import json
import cv2
import numpy as np
import pandas as pd
import logging
from datetime import datetime
from pathlib import Path
from PIL import Image
import streamlit as st

# =============================================================================
# ⚠️ PyTorch 2.6+ Compatibility Patch (Must run before torch imports)
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

# =============================================================================
# IMPORTS (After patch)
# =============================================================================
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    ort = None

from sklearn.preprocessing import LabelEncoder
import joblib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# =============================================================================
# STREAMLIT CONFIG (Must be first Streamlit command)
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

# Damage → Part mapping for cost model compatibility
DAMAGE_TO_PART = {
    'dent': 'body_panel',
    'scratch': 'paint_surface', 
    'crack': 'windshield',
    'glass_shatter': 'windshield',
    'lamp_broken': 'headlight',
    'tire_flat': 'tire'
}

# Base costs for rule-based fallback
BASE_COSTS = {
    'dent': {'part': 200, 'labor': 150, 'paint': 100},
    'scratch': {'part': 100, 'labor': 100, 'paint': 150},
    'crack': {'part': 300, 'labor': 200, 'paint': 100},
    'glass_shatter': {'part': 500, 'labor': 150, 'paint': 50},
    'lamp_broken': {'part': 150, 'labor': 100, 'paint': 50},
    'tire_flat': {'part': 100, 'labor': 50, 'paint': 0}
}

SEVERITY_MULTIPLIERS = {'low': 0.8, 'medium': 1.0, 'high': 1.5}

# =============================================================================
# 📦 MODEL LOADING FUNCTIONS
# =============================================================================
@st.cache_resource
def load_onnx_model(model_path: str = 'runs/detect/train/damage_detect_v2/weights/best.onnx'):
    """Load ONNX model with CPU/GPU fallback and structured status"""
    if not ONNX_AVAILABLE:
        return None, {
            "status": "error",
            "message": "onnxruntime not installed",
            "provider": None,
            "details": "Run: pip install onnxruntime"
        }
    
    if not os.path.exists(model_path):
        return None, {
            "status": "error",
            "message": "Model file not found",
            "provider": None,
            "details": f"Expected: {model_path}"
        }
    
    try:
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.intra_op_num_threads = 4
        
        providers = ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
        session = ort.InferenceSession(model_path, sess_options=session_options, providers=providers)
        
        inputs = session.get_inputs()
        outputs = session.get_outputs()
        
        if not inputs or not outputs:
            return None, {"status": "error", "message": "Invalid model structure", "provider": None, "details": "No valid I/O"}
        
        active_provider = session.get_providers()[0]
        output_shape = outputs[0].shape
        
        # Determine status messaging
        if active_provider == 'CPUExecutionProvider':
            status_level, message, details = "warning", "Loaded on CPU", "Install CUDA 12 + cuDNN 9 for GPU acceleration"
        elif 'Tensorrt' in active_provider:
            status_level, message, details = "success", "Loaded on TensorRT (Fastest)", "GPU acceleration active"
        else:
            status_level, message, details = "success", f"Loaded on CUDA", "GPU acceleration active"
        
        logger.info(f"✅ ONNX model loaded | Provider: {active_provider}")
        return session, {
            "status": status_level, "message": message, "provider": active_provider,
            "output_shape": output_shape, "input_name": inputs[0].name, "details": details,
            "num_classes": output_shape[1] - 4 if len(output_shape) == 3 else NUM_CLASSES
        }
        
    except Exception as e:
        err = str(e)
        logger.error(f"❌ ONNX load failed: {err}")
        if 'half' in err.lower() or 'float16' in err.lower():
            details = "Re-export model with half=False for CPU: yolo export ... half=False"
        elif 'shape' in err.lower():
            details = "Check preprocessing matches model input (1,3,640,640)"
        else:
            details = err[:200]
        return None, {"status": "error", "message": type(e).__name__, "provider": None, "details": details}


@st.cache_resource
def load_cost_models(base_path: str = 'cost_model.pkl'):
    """Load RandomForest cost model + encoders with safe fallback"""
    required = [base_path, 'model_encoder.pkl', 'part_encoder.pkl']
    if not all(os.path.exists(f) for f in required):
        logger.warning("⚠️ Cost model files missing. Using rule-based fallback.")
        return None, None, None
    
    try:
        model = joblib.load(base_path)
        model_enc = joblib.load('model_encoder.pkl')  # Vehicle model encoder
        part_enc = joblib.load('part_encoder.pkl')     # Part/Damage type encoder
        logger.info("✅ Cost models loaded successfully")
        return model, model_enc, part_enc
    except Exception as e:
        logger.error(f"❌ Cost model load failed: {e}")
        return None, None, None

# =============================================================================
# 🔧 INFERENCE UTILITIES
# =============================================================================
def preprocess_image(image_bgr: np.ndarray, img_size: int = 640):
    """Letterbox resize + normalize + NCHW for YOLOv8 ONNX"""
    h, w = image_bgr.shape[:2]
    scale = img_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    
    resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    padded = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    padded[:new_h, :new_w] = resized
    
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = np.transpose(rgb, (2, 0, 1))[None, ...]  # (1, 3, H, W)
    return tensor, scale, (h, w)


def postprocess_yolov8_onnx(output, orig_shape, scale, conf_thresh, iou_thresh, num_classes=NUM_CLASSES):
    """Parse YOLOv8 ONNX output [1, 4+nc, N] → detections list"""
    out = np.squeeze(output, axis=0)  # (4+nc, N)
    total_channels = out.shape[0]
    use_nc = min(total_channels - 4, num_classes) if total_channels >= 4 + num_classes else total_channels - 4
    
    detections = []
    img_h, img_w = orig_shape
    
    for i in range(out.shape[1]):
        box = out[:, i]
        x_c, y_c, w, h = box[:4]
        class_probs = box[4:4+use_nc] if use_nc > 0 else []
        
        if len(class_probs) == 0:
            continue
            
        class_id = int(np.argmax(class_probs))
        confidence = float(class_probs[class_id]) if len(class_probs) > class_id else 0
        
        if confidence < conf_thresh or class_id >= len(DAMAGE_CLASSES):
            continue
        
        # Convert to corner coordinates in original image space
        x1 = ((x_c - w/2) * img_w) / scale
        y1 = ((y_c - h/2) * img_h) / scale
        x2 = ((x_c + w/2) * img_w) / scale
        y2 = ((y_c + h/2) * img_h) / scale
        
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_w, x2), min(img_h, y2)
        
        detections.append({
            'bbox': [float(x1), float(y1), float(x2), float(y2)],
            'confidence': confidence,
            'class_id': class_id,
            'class_name': DAMAGE_CLASSES[class_id]
        })
    
    return nms(detections, iou_thresh) if detections else []


def nms(detections, iou_thresh):
    """Greedy Non-Maximum Suppression"""
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
    x1, y1, x2, y2 = max(box1[0], box2[0]), max(box1[1], box2[1]), min(box1[2], box2[2]), min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / (area1 + area2 - inter)


def calculate_severity(confidence, bbox, img_shape):
    """Calculate damage severity from confidence + relative area"""
    x1, y1, x2, y2 = bbox
    area = (x2 - x1) * (y2 - y1)
    total = img_shape[0] * img_shape[1]
    ratio = min(area / total, 1.0)
    score = min(confidence * 0.7 + ratio * 0.3, 1.0)
    
    if score < 0.25:
        return 'low', round(score, 3)
    elif score < 0.60:
        return 'medium', round(score, 3)
    return 'high', round(score, 3)


def safe_transform(encoder, value, default=0):
    """Safely encode categorical value with fallback"""
    try:
        return encoder.transform([str(value).lower().strip()])[0]
    except Exception:
        return default


def predict_cost(det, vehicle_info, cost_model, model_enc, part_enc):
    """
    Predict repair cost using ML model (if available) or rule-based fallback.
    Maps damage_type → part for model compatibility.
    """
    dmg_type = det.get('class_name', 'dent')
    severity = det.get('severity', 'medium')
    sev_score = det.get('severity_score', 0.5)
    
    # Try ML model first
    if cost_model and model_enc and part_enc:
        try:
            # Encode vehicle model
            model_val = vehicle_info.get('model', 'unknown')
            model_encoded = safe_transform(model_enc, model_val)
            
            # Map damage type to part for encoding (since model was trained on parts)
            mapped_part = DAMAGE_TO_PART.get(dmg_type, 'body_panel')
            part_encoded = safe_transform(part_enc, mapped_part)
            
            # Prepare features: [Model_encoded, Part_encoded, severity_score]
            features = np.array([[model_encoded, part_encoded, sev_score]])
            pred = float(cost_model.predict(features)[0])
            
            return {
                'total_cost': round(max(pred, 0), 2),
                'method': 'ml_model',
                'breakdown': _ml_cost_breakdown(max(pred, 0), dmg_type)
            }
        except Exception as e:
            logger.warning(f"⚠️ ML cost prediction failed: {e}. Falling back to rule-based.")
    
    # Rule-based fallback
    base = BASE_COSTS.get(dmg_type, BASE_COSTS['dent'])
    mult = SEVERITY_MULTIPLIERS.get(severity, 1.0)
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


def _ml_cost_breakdown(total, dmg_type):
    """Generate component breakdown for ML predictions using heuristic ratios"""
    ratios = {
        'glass_shatter': (0.7, 0.2, 0.1),
        'tire_flat': (0.8, 0.2, 0.0),
        'lamp_broken': (0.6, 0.3, 0.1),
        'crack': (0.4, 0.4, 0.2),
        'dent': (0.3, 0.5, 0.2),
        'scratch': (0.2, 0.3, 0.5)
    }.get(dmg_type, (0.4, 0.4, 0.2))
    
    return {
        'part_cost': round(total * ratios[0], 2),
        'labor_cost': round(total * ratios[1], 2),
        'paint_cost': round(total * ratios[2], 2)
    }


def draw_detections(image_bgr, detections):
    """Draw high-visibility annotations with multi-line labels"""
    img = image_bgr.copy()
    colors = {'low': (0, 200, 0), 'medium': (0, 165, 255), 'high': (0, 0, 255)}
    
    for det in detections:
        x1, y1, x2, y2 = map(int, det['bbox'])
        sev = det.get('severity', 'medium')
        color = colors.get(sev, (255, 255, 255))
        
        # Thick bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        
        # Multi-line label
        label_lines = [
            f"{det['class_name'].replace('_', ' ').title()}",
            f"{det['confidence']:.0%} | {sev.upper()}"
        ]
        if 'cost_estimate' in det:
            label_lines.append(f"₹{det['cost_estimate']['total_cost']:,.0f}")
        
        # Label background for readability
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale, thickness = 0.6, 2
        (tw, th), _ = cv2.getTextSize(label_lines[0], font, scale, thickness)
        label_h = th * len(label_lines) + 10
        
        cv2.rectangle(img, (x1, y1 - label_h), (x1 + tw + 15, y1), color, -1)
        
        for i, line in enumerate(label_lines):
            cv2.putText(img, line, (x1 + 5, y1 - 5 - (len(label_lines) - 1 - i) * th),
                       font, scale, (0, 0, 0), thickness, cv2.LINE_AA)
    
    return img


def generate_recommendations(detections):
    """Generate actionable repair recommendations"""
    if not detections:
        return ["✨ No damages detected - vehicle appears to be in good condition!"]
    
    recs = []
    if any(d.get('severity') == 'high' for d in detections):
        recs.append("🚨 IMMEDIATE ACTION: High severity damages require urgent repair")
    if any(d['class_name'] == 'crack' for d in detections):
        recs.append("⚠️ Cracks can propagate - repair immediately to prevent worsening")
    if any(d['class_name'] == 'glass_shatter' for d in detections):
        recs.append("🪟 Broken glass requires replacement for safety and visibility")
    if any(d['class_name'] == 'tire_flat' for d in detections):
        recs.append("🛞 Flat tire detected - do not drive until replaced or repaired")
    if len([d for d in detections if d.get('severity') == 'medium']) >= 2:
        recs.append("🔧 Schedule medium severity repairs within 1-2 weeks")
    if all(d.get('severity') == 'low' for d in detections):
        recs.append("✅ Minor cosmetic damages - maintenance recommended when convenient")
    return recs

# =============================================================================
# 🖥️ STREAMLIT UI
# =============================================================================
def main():
    st.title("🚗 AI Vehicle Damage Assessment System")
    st.markdown("*Upload vehicle images to detect damage, estimate repair costs, and generate insurance-ready reports*")
    
    # Session state for batch processing
    if 'batch_results' not in st.session_state:
        st.session_state.batch_results = []
    
    # Load models
    model_path = 'runs/detect/train/damage_detect_v2/weights/best.onnx'
    onnx_session, model_info = load_onnx_model(model_path)
    cost_model, model_enc, part_enc = load_cost_models('cost_model.pkl')
    
    # Sidebar: System Status & Settings
    with st.sidebar:
        st.header("⚙️ System Status")
        
        # Detection model status
        if model_info and model_info["status"] in ["success", "warning"]:
            if model_info["status"] == "success":
                st.success(f"✅ Detection: {model_info['message']}")
            else:
                st.warning(f"⚠️ Detection: {model_info['message']}")
            st.caption(f"Provider: {model_info['provider']}")
            if model_info.get('output_shape'):
                st.caption(f"Output shape: {model_info['output_shape']}")
        else:
            st.error("❌ Detection Model: Failed to Load")
            if model_info:
                st.code(f"{model_info.get('message', '')}\n{model_info.get('details', '')}")
        
        st.divider()
        
        # Cost model status
        if cost_model and model_enc and part_enc:
            st.success("✅ Cost Model: ML Ready")
        else:
            st.warning("⚠️ Cost Model: Rule-Based Fallback")
            st.caption("Estimates based on damage type + severity multipliers")
        
        st.divider()
        st.header("🎛️ Inference Settings")
        conf_thresh = st.slider("Confidence Threshold", 0.1, 0.9, 0.25, 0.05,
                               help="Higher = fewer false positives, more precision")
        iou_thresh = st.slider("NMS IoU Threshold", 0.3, 0.7, 0.45, 0.05,
                              help="Higher = keep more overlapping detections")
        
        st.divider()
        st.header("📊 System Info")
        st.metric("Damage Classes", len(DAMAGE_CLASSES))
        st.metric("Inference Engine", "ONNX Runtime" if ONNX_AVAILABLE else "Not Available")
        st.metric("Cost Engine", "ML Model" if cost_model else "Rule-Based")
        
        # GPU acceleration hint
        if model_info and model_info.get('provider') == 'CPUExecutionProvider':
            with st.expander("🚀 Enable GPU Acceleration"):
                st.markdown("""
                **For 5-50x faster inference:**
                1. Install CUDA 12: https://developer.nvidia.com/cuda-12-0-download-archive
                2. Install cuDNN 9: https://developer.nvidia.com/cudnn
                3. Add to PATH:
                   ```
                   C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.0\\bin
                   C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.0\\libnvvp
                   ```
                4. Restart Streamlit app
                """)
                st.code("pip install onnxruntime-gpu  # Replace onnxruntime")
    
    # Main Tabs
    tab1, tab2, tab3 = st.tabs(["📤 Upload & Analyze", "📋 Results", "📄 Report"])
    
    # ==================== TAB 1: UPLOAD & ANALYZE ====================
    with tab1:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📷 Upload Vehicle Images")
            uploaded = st.file_uploader(
                "Choose images (JPG/PNG/BMP)...",
                type=['jpg', 'jpeg', 'png', 'bmp'],
                accept_multiple_files=True,
                key="uploader"
            )
            
            if uploaded:
                st.markdown(f"📂 **{len(uploaded)}** images selected for batch processing")
                
                # Vehicle info (applies to all images)
                with st.expander("🚙 Vehicle Information", expanded=True):
                    c1, c2 = st.columns(2)
                    brand = c1.selectbox("Brand", ['Maruti Suzuki', 'Hyundai', 'Tata', 'Honda', 'Mahindra', 'Toyota', 'Other'], key="brand_sel")
                    model = c2.text_input("Model", placeholder="e.g., Swift, Creta, Nexon", key="model_input")
                    c3, c4 = st.columns(2)
                    year = c3.number_input("Year", 2010, 2025, 2020, key="year_input")
                    mileage = c4.number_input("Mileage (km)", 0, 500000, 50000, key="mileage_input")
                
                # Analyze button
                if st.button("🔍 Analyze All Images", type="primary", use_container_width=True, 
                            disabled=(model_info is None or model_info.get("status") == "error")):
                    if not model.strip():
                        st.error("⚠️ Please enter vehicle model for accurate cost estimation")
                    else:
                        # Clear previous results
                        st.session_state.batch_results = []
                        progress_bar = st.progress(0, text="🔎 Initializing...")
                        status_text = st.empty()
                        
                        for i, file in enumerate(uploaded):
                            try:
                                status_text.text(f"🔄 Processing {file.name} ({i+1}/{len(uploaded)})...")
                                
                                # Load & preprocess
                                img = Image.open(file).convert('RGB')
                                img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                                tensor, scale, orig_shape = preprocess_image(img_bgr)
                                
                                # Inference
                                input_name = model_info['input_name']
                                outputs = onnx_session.run(None, {input_name: tensor})
                                
                                # Postprocess detections
                                detections = postprocess_yolov8_onnx(
                                    outputs[0], orig_shape, scale, conf_thresh, iou_thresh
                                )
                                
                                # Enrich with severity + cost
                                vehicle_info = {'brand': brand, 'model': model.strip(), 'year': year, 'mileage': mileage}
                                for det in detections:
                                    sev, sev_score = calculate_severity(det['confidence'], det['bbox'], orig_shape)
                                    det['severity'] = sev
                                    det['severity_score'] = sev_score
                                    det['cost_estimate'] = predict_cost(det, vehicle_info, cost_model, model_enc, part_enc)
                                
                                # Draw annotations
                                vis_img = draw_detections(img_bgr, detections)
                                vis_rgb = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
                                
                                # Store results
                                st.session_state.batch_results.append({
                                    'filename': file.name,
                                    'original': img,
                                    'processed': vis_rgb,
                                    'detections': detections,
                                    'vehicle_info': vehicle_info,
                                    'orig_shape': orig_shape,
                                    'settings': {'conf': conf_thresh, 'iou': iou_thresh}
                                })
                                
                            except Exception as e:
                                logger.error(f"❌ Failed to process {file.name}: {e}", exc_info=True)
                                st.warning(f"⚠️ Error processing {file.name}: {str(e)[:100]}...")
                            
                            # Update progress
                            progress_bar.progress((i + 1) / len(uploaded), text=f"✅ Processed {i+1}/{len(uploaded)}")
                        
                        progress_bar.empty()
                        status_text.empty()
                        st.success(f"🎉 Analysis Complete! {len(st.session_state.batch_results)} images processed.")
                        st.rerun()
        
        with col2:
            st.subheader("💡 Tips for Best Results")
            st.markdown("""
            - 📸 Use **good lighting** and **steady hands**
            - 🎯 Capture the **damaged area prominently** in frame
            - 📐 Include **some context** (helps model understand scale)
            - 🔄 Try **different angles** if detection is uncertain
            - ✂️ **Crop to damaged area** for faster processing
            - 🚫 Avoid **extreme close-ups** or **blurry images**
            """)
            
            st.divider()
            st.subheader("🔧 Supported Damage Types")
            for idx, cls in enumerate(DAMAGE_CLASSES, 1):
                st.markdown(f"{idx}. `{cls.replace('_', ' ').title()}`")
            
            st.divider()
            st.subheader("📈 Severity Levels")
            st.markdown("""
            - 🟢 **Low** (0.0-0.25): Minor cosmetic issues, no structural impact
            - 🟡 **Medium** (0.25-0.60): Noticeable damage, repair recommended within weeks
            - 🔴 **High** (0.60-1.0): Structural/safety concern, urgent repair required
            """)
    
    # ==================== TAB 2: RESULTS ====================
    with tab2:
        if not st.session_state.batch_results:
            st.info("👆 Upload and analyze images in Tab 1 to see results here")
        else:
            # Image selector for multi-image navigation
            filenames = [r['filename'] for r in st.session_state.batch_results]
            selected_idx = st.selectbox(
                "Select Image to View Details",
                range(len(filenames)),
                format_func=lambda i: f"📷 {filenames[i]}",
                key="result_selector"
            )
            res = st.session_state.batch_results[selected_idx]
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.image(
                    res['processed'],
                    caption=f"🖼️ {res['filename']} | 🟢Low 🟡Medium 🔴High",
                    use_column_width=True
                )
            with col2:
                detections = res['detections']
                st.metric("🔍 Damages Detected", len(detections))
                if detections:
                    high_count = sum(1 for d in detections if d['severity'] == 'high')
                    st.metric("🚨 High Severity", high_count, delta_color="inverse")
                    total_cost = sum(d['cost_estimate']['total_cost'] for d in detections)
                    st.metric("💰 Est. Total Cost", f"₹{total_cost:,.0f}")
                    avg_conf = np.mean([d['confidence'] for d in detections])
                    st.metric("🎯 Avg. Confidence", f"{avg_conf:.2%}")
            
            if res['detections']:
                st.divider()
                st.subheader("🔍 Detection Details")
                
                # Build table
                table_data = []
                for i, det in enumerate(res['detections'], 1):
                    ce = det['cost_estimate']
                    table_data.append({
                        '#': i,
                        'Type': det['class_name'].replace('_', ' ').title(),
                        'Confidence': f"{det['confidence']:.2%}",
                        'Severity': det['severity'].upper(),
                        'Score': det['severity_score'],
                        'Est. Cost': f"₹{ce['total_cost']:,.0f}",
                        'Method': ce['method'].replace('_', ' ').title()
                    })
                
                df = pd.DataFrame(table_data)
                st.dataframe(df, use_container_width=True, hide_index=True)
                
                # Detailed breakdown per detection
                st.subheader("📋 Damage Breakdown")
                for i, det in enumerate(res['detections'], 1):
                    ce = det['cost_estimate']
                    bd = ce['breakdown']
                    with st.expander(
                        f"#{i}: {det['class_name'].replace('_', ' ').title()} | {det['severity'].upper()} | ₹{ce['total_cost']:,.0f}",
                        expanded=(i == 1)
                    ):
                        st.markdown(f"""
                        **Detection Details**
                        - Confidence: `{det['confidence']:.2%}`
                        - Severity Score: `{det['severity_score']:.3f}`
                        - Bounding Box: `[{' '.join(f'{c:.1f}' for c in det['bbox'])}]`
                        
                        **Cost Estimate** (`{ce['method']}`)
                        - 💰 **Total: ₹{ce['total_cost']:,.0f}**
                        - 🔧 Parts: ₹{bd.get('part_cost', 0):,.0f}
                        - 👷 Labor: ₹{bd.get('labor_cost', 0):,.0f}
                        - 🎨 Paint: ₹{bd.get('paint_cost', 0):,.0f}
                        """)
    
    # ==================== TAB 3: REPORT ====================
    with tab3:
        if not st.session_state.batch_results:
            st.info("👆 Complete an analysis in Tab 1 to generate reports")
        else:
            st.header("📄 Export Reports")
            export_mode = st.radio(
                "Export Scope",
                ["All Images", "Selected Image Only"],
                horizontal=True,
                key="export_mode"
            )
            
            def build_report(results):
                """Build structured report dict(s) for export"""
                reports = []
                for r in results:
                    total = sum(d['cost_estimate']['total_cost'] for d in r['detections'])
                    reports.append({
                        'report_id': f"VDA-{datetime.now().strftime('%Y%m%d%H%M%S')}-{Path(r['filename']).stem}",
                        'filename': r['filename'],
                        'timestamp': datetime.now().isoformat(),
                        'vehicle': r['vehicle_info'],
                        'settings': r['settings'],
                        'summary': {
                            'damages_count': len(r['detections']),
                            'total_cost': round(total, 2),
                            'severity_distribution': {
                                'low': sum(1 for d in r['detections'] if d.get('severity') == 'low'),
                                'medium': sum(1 for d in r['detections'] if d.get('severity') == 'medium'),
                                'high': sum(1 for d in r['detections'] if d.get('severity') == 'high')
                            },
                            'damage_types': list(set(d['class_name'] for d in r['detections']))
                        },
                        'damages': r['detections'],
                        'recommendations': generate_recommendations(r['detections'])
                    })
                return reports

            # Build reports
            if export_mode == "Selected Image Only":
                reports = build_report([st.session_state.batch_results[selected_idx]])
            else:
                reports = build_report(st.session_state.batch_results)
            
            # Export buttons
            c1, c2, c3 = st.columns(3)
            with c1:
                json_str = json.dumps(reports, indent=2, default=str)
                st.download_button(
                    label="📄 Download JSON",
                    data=json_str,
                    file_name=f"damage_report_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                    mime="application/json",
                    use_container_width=True
                )
            with c2:
                csv_rows = []
                for r in reports:
                    for d in r['damages']:
                        ce = d['cost_estimate']
                        bd = ce['breakdown']
                        csv_rows.append({
                            'Report_ID': r['report_id'],
                            'Filename': r['filename'],
                            'Damage_Type': d['class_name'],
                            'Confidence': round(d['confidence'], 4),
                            'Severity': d['severity'],
                            'Severity_Score': d['severity_score'],
                            'Estimated_Cost': ce['total_cost'],
                            'Cost_Method': ce['method'],
                            'Part_Cost': bd.get('part_cost', 0),
                            'Labor_Cost': bd.get('labor_cost', 0),
                            'Paint_Cost': bd.get('paint_cost', 0),
                            'BBox_X1': round(d['bbox'][0], 2),
                            'BBox_Y1': round(d['bbox'][1], 2),
                            'BBox_X2': round(d['bbox'][2], 2),
                            'BBox_Y2': round(d['bbox'][3], 2)
                        })
                if csv_rows:
                    csv_data = pd.DataFrame(csv_rows).to_csv(index=False)
                    st.download_button(
                        label="📊 Download CSV",
                        data=csv_data,
                        file_name=f"damage_details_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                else:
                    st.info("ℹ️ No detections to export")
            with c3:
                st.button("📕 Generate PDF", disabled=True, help="Coming soon: pip install reportlab")
                st.caption("*PDF export requires additional dependencies*")
            
            st.divider()
            st.subheader("📋 Summary Preview")
            for r in reports:
                with st.expander(f"📄 {r['filename']}", expanded=True):
                    st.markdown(f"""
                    **Summary**
                    - Damages: `{r['summary']['damages_count']}`
                    - Est. Total Cost: `₹{r['summary']['total_cost']:,.0f}`
                    - Types: `{', '.join(r['summary']['damage_types'])}`
                    - Severity: 🟢{r['summary']['severity_distribution']['low']} 🟡{r['summary']['severity_distribution']['medium']} 🔴{r['summary']['severity_distribution']['high']}
                    
                    **Recommendations**
                    {chr(10).join(f"- {rec}" for rec in r['recommendations'])}
                    """)
            
            st.info("""
            🔗 **Insurance Integration Ready**  
            This report contains structured data for automated claim processing:
            - ✅ Damage classifications with confidence scores
            - ✅ Severity assessments with quantitative metrics  
            - ✅ Itemized cost estimates with component breakdowns
            - ✅ Spatial damage mapping via bounding box coordinates
            - ✅ Vehicle metadata for policy matching
            """)

# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    logger.info("🚀 Starting Vehicle Damage Assessment App")
    logger.info(f"ONNX Runtime: {ort.__version__ if ONNX_AVAILABLE else 'not installed'}")
    if ONNX_AVAILABLE:
        logger.info(f"Available providers: {ort.get_available_providers()}")
    main()