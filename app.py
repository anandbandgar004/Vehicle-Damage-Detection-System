"""
AI Vehicle Damage Assessment System (Damage-Only Focus)
Compatible with YOLOv8 ONNX exports | Streamlit | CPU/GPU ready
Multi-image support | High-visibility annotations
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

# =============================================================================
# ⚠️ CRITICAL: PyTorch 2.6+ Compatibility Patch (Must run before any torch imports)
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

# Now safe to import Streamlit & ONNX
import streamlit as st
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    ort = None

from sklearn.preprocessing import LabelEncoder
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
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

BASE_COSTS = {
    'dent': {'part': 200, 'labor': 150, 'paint': 100},
    'scratch': {'part': 100, 'labor': 100, 'paint': 150},
    'crack': {'part': 300, 'labor': 200, 'paint': 100},
    'glass_shatter': {'part': 500, 'labor': 150, 'paint': 50},
    'lamp_broken': {'part': 150, 'labor': 100, 'paint': 50},
    'tire_flat': {'part': 100, 'labor': 50, 'paint': 0}
}

# =============================================================================
# MODEL LOADING FUNCTIONS (Enhanced with structured status)
# =============================================================================
@st.cache_resource
def load_onnx_model(model_path: str = 'runs/detect/runs/train/damage_detect_v2/weights/best.onnx'):
    """Load ONNX model with CPU fallback support and structured status reporting"""
    if not ONNX_AVAILABLE:
        logger.error("❌ onnxruntime not installed")
        return None, {
            "status": "error",
            "message": "onnxruntime not installed",
            "provider": None,
            "details": "Run: pip install onnxruntime"
        }
    
    if not os.path.exists(model_path):
        logger.error(f"❌ Model not found: {model_path}")
        return None, {
            "status": "error",
            "message": f"File not found",
            "provider": None,
            "details": f"Expected: {model_path}"
        }
    
    try:
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.intra_op_num_threads = 4
        
        # Provider priority: TensorRT > CUDA > CPU (graceful fallback)
        providers = ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
        
        logger.info(f"Loading ONNX model: {model_path}")
        session = ort.InferenceSession(model_path, sess_options=session_options, providers=providers)
        
        # Validate I/O structure
        inputs = session.get_inputs()
        outputs = session.get_outputs()
        
        if not inputs or not outputs:
            return None, {
                "status": "error",
                "message": "Invalid model structure",
                "provider": None,
                "details": "Model has no valid inputs/outputs"
            }
        
        active_provider = session.get_providers()[0]
        output_shape = outputs[0].shape
        output_type = outputs[0].type
        
        # Determine status level and message
        if active_provider == 'CPUExecutionProvider':
            status_level = "warning"
            message = "Loaded on CPU (GPU not available)"
            details = "Inference will be slower. Install CUDA 12 + cuDNN 9 for GPU acceleration."
        elif 'Tensorrt' in active_provider:
            status_level = "success"
            message = "Loaded on TensorRT (Fastest)"
            details = "GPU acceleration active with TensorRT optimization"
        else:  # CUDA
            status_level = "success"
            message = f"Loaded on CUDA"
            details = "GPU acceleration active"
        
        # Check for FP16
        if 'float16' in output_type.lower() and active_provider == 'CPUExecutionProvider':
            message += " | ⚠️ FP16 on CPU"
            details += " | FP16 models run best on GPU"
        
        logger.info(f"✅ Model loaded | Provider: {active_provider} | Output: {output_shape}")
        
        return session, {
            "status": status_level,
            "message": message,
            "provider": active_provider,
            "output_shape": output_shape,
            "input_name": inputs[0].name,
            "details": details,
            "num_classes": output_shape[1] - 4 if len(output_shape) == 3 else NUM_CLASSES
        }
        
    except ort.OrtException as e:
        err_msg = str(e)
        logger.error(f"❌ ORT exception: {err_msg}")
        
        if 'half' in err_msg.lower() or 'float16' in err_msg.lower():
            return None, {
                "status": "error",
                "message": "FP16 compatibility issue",
                "provider": None,
                "details": "Re-export model with half=False for CPU: yolo export ... half=False"
            }
        elif 'shape' in err_msg.lower() or 'dimension' in err_msg.lower():
            return None, {
                "status": "error", 
                "message": "Shape mismatch",
                "provider": None,
                "details": "Check preprocessing matches model input (1,3,640,640)"
            }
        else:
            return None, {
                "status": "error",
                "message": "Runtime error",
                "provider": None,
                "details": err_msg[:200]
            }
            
    except Exception as e:
        logger.error(f"❌ Unexpected error: {type(e).__name__}: {e}", exc_info=True)
        return None, {
            "status": "error",
            "message": f"{type(e).__name__}",
            "provider": None,
            "details": str(e)[:200]
        }

@st.cache_resource
def load_cost_model(base_path: str = 'cost_model.pkl'):
    """Load cost model with safe fallback"""
    required = [base_path, 'model_encoder.pkl', 'part_encoder.pkl']
    if not all(os.path.exists(f) for f in required):
        logger.warning("⚠️ Cost model files missing. Using rule-based fallback.")
        return None, None, None
    
    try:
        model = joblib.load(base_path)
        model_enc = joblib.load('model_encoder.pkl')
        part_enc = joblib.load('part_encoder.pkl')
        logger.info("✅ Cost model loaded")
        return model, model_enc, part_enc
    except Exception as e:
        logger.error(f"❌ Cost model load failed: {e}")
        return None, None, None

# =============================================================================
# INFERENCE UTILITIES
# =============================================================================
def preprocess_image(image_bgr: np.ndarray, img_size: int = 640):
    """Letterbox resize + normalize + NCHW for YOLOv8 ONNX"""
    h, w = image_bgr.shape[:2]
    scale = img_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    
    resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Pad to square
    padded = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    padded[:new_h, :new_w] = resized
    
    # BGR -> RGB, normalize, transpose to NCHW
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = np.transpose(rgb, (2, 0, 1))[None, ...]  # (1, 3, H, W)
    
    return tensor, scale, (h, w)

def postprocess_yolov8_onnx(output, orig_shape, scale, conf_thresh, iou_thresh, num_classes=NUM_CLASSES):
    """
    Parse YOLOv8 ONNX output: shape [1, 4+nc, N]
    Handles both standard and padded exports safely.
    """
    # Squeeze batch dim: (4+nc, N)
    out = np.squeeze(output, axis=0)
    
    # YOLOv8 format: [x_c, y_c, w, h, class_probs...]
    # Automatically detect class channels
    total_channels = out.shape[0]
    actual_nc = total_channels - 4
    
    # Fallback to expected classes if mismatched
    use_nc = min(actual_nc, num_classes) if actual_nc >= num_classes else actual_nc
    
    detections = []
    img_h, img_w = orig_shape
    
    for i in range(out.shape[1]):
        box = out[:, i]
        x_c, y_c, w, h = box[:4]
        class_probs = box[4:4+use_nc]
        
        if len(class_probs) == 0:
            continue
            
        class_id = np.argmax(class_probs)
        confidence = float(class_probs[class_id])
        
        if confidence < conf_thresh or class_id >= len(DAMAGE_CLASSES):
            continue
        
        # Convert to pixel coords & corner format
        x1 = ((x_c - w/2) * img_w) / scale
        y1 = ((y_c - h/2) * img_h) / scale
        x2 = ((x_c + w/2) * img_w) / scale
        y2 = ((y_c + h/2) * img_h) / scale
        
        # Clip bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_w, x2), min(img_h, y2)
        
        detections.append({
            'bbox': [float(x1), float(y1), float(x2), float(y2)],
            'confidence': confidence,
            'class_id': int(class_id),
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
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0: return 0.0
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union

def calculate_severity(confidence, bbox, img_shape):
    x1, y1, x2, y2 = bbox
    area = (x2 - x1) * (y2 - y1)
    total = img_shape[0] * img_shape[1]
    ratio = min(area / total, 1.0)
    score = min(confidence * 0.7 + ratio * 0.3, 1.0)
    
    if score < 0.25: return 'low', round(score, 3)
    if score < 0.60: return 'medium', round(score, 3)
    return 'high', round(score, 3)

def safe_transform(encoder, value, default=0):
    try:
        return encoder.transform([str(value).lower().strip()])[0]
    except Exception:
        return default

def predict_cost(det, vehicle_info, cost_model, model_enc, part_enc):
    """Damage-focused cost prediction - FIXED: consistent breakdown keys"""
    dmg_type = det.get('class_name', 'dent')
    severity = det.get('severity', 'medium')
    sev_score = det.get('severity_score', 0.5)
    
    if cost_model and model_enc:
        try:
            model_enc_val = safe_transform(model_enc, vehicle_info.get('model', 'unknown'))
            features = np.array([[model_enc_val, sev_score]])
            pred = float(cost_model.predict(features)[0])
            return {
                'total_cost': round(max(pred, 0), 2),
                'method': 'ml_model',
                'breakdown': _cost_breakdown(pred, dmg_type)
            }
        except Exception as e:
            logger.warning(f"ML cost failed: {e}")
    
    # Fallback: rule-based estimation
    base = BASE_COSTS.get(dmg_type, BASE_COSTS['dent'])
    mult = {'low': 0.8, 'medium': 1.0, 'high': 1.5}.get(severity, 1.0)
    total = (base['part'] + base['labor'] + base['paint']) * mult
    
    # ✅ FIXED: Return breakdown with _cost suffix keys to match UI expectations
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
    """Helper for ML model breakdown - returns correct key format"""
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
    """High-visibility drawing with thick boxes, backgrounds, and multi-line labels"""
    img = image_bgr.copy()
    colors = {'low': (0, 200, 0), 'medium': (0, 165, 255), 'high': (0, 0, 255)}
    
    for det in detections:
        x1, y1, x2, y2 = map(int, det['bbox'])
        sev = det.get('severity', 'medium')
        color = colors.get(sev, (255, 255, 255))
        
        # Thick bounding box for better visibility
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        
        # Build multi-line label
        label_lines = [
            f"{det['class_name'].replace('_', ' ').title()}",
            f"{det['confidence']:.0%} | {sev.upper()}"
        ]
        if 'cost_estimate' in det:
            label_lines.append(f"₹{det['cost_estimate']['total_cost']:,.0f}")
            
        # Calculate label size for background
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.6
        thickness = 2
        (tw, th), _ = cv2.getTextSize(label_lines[0], font, scale, thickness)
        label_h = th * len(label_lines) + 10
        
        # Draw solid background for readability on any image
        cv2.rectangle(img, (x1, y1-label_h), (x1+tw+15, y1), color, -1)
        
        # Draw each line of text with anti-aliasing
        for i, line in enumerate(label_lines):
            cv2.putText(img, line, (x1+5, y1-5-(len(label_lines)-1-i)*th), 
                       font, scale, (0,0,0), thickness, cv2.LINE_AA)
            
    return img

def generate_recommendations(detections):
    recs = []
    if not detections:
        return ["✨ No damages detected - vehicle appears to be in good condition!"]
    
    if any(d.get('severity') == 'high' for d in detections):
        recs.append("🚨 IMMEDIATE ACTION: High severity damages require urgent repair")
    if any(d['class_name'] == 'crack' for d in detections):
        recs.append("⚠️ Cracks can propagate - repair immediately")
    if any(d['class_name'] == 'glass_shatter' for d in detections):
        recs.append("🪟 Broken glass requires replacement for safety")
    if any(d['class_name'] == 'tire_flat' for d in detections):
        recs.append("🛞 Flat tire detected - do not drive until replaced")
    if len([d for d in detections if d.get('severity') == 'medium']) >= 2:
        recs.append("🔧 Schedule medium severity repairs within 1-2 weeks")
    if all(d.get('severity') == 'low' for d in detections):
        recs.append("✅ Minor damages - maintenance recommended when convenient")
    return recs

# =============================================================================
# STREAMLIT UI
# =============================================================================
def main():
    st.title("🚗 AI Vehicle Damage Assessment System")
    st.markdown("*Upload vehicle images to detect damage, estimate repair costs, and generate insurance-ready reports*")
    
    # Session State - Support multi-image batch processing
    if 'batch_results' not in st.session_state:
        st.session_state.batch_results = []
    if 'model_status' not in st.session_state:
        st.session_state.model_status = "not_loaded"
    
    # Load Models
    model_path = 'runs/detect/runs/train/damage_detect_v2/weights/best.onnx'
    onnx_session, model_info = load_onnx_model(model_path)
    cost_model, model_enc, part_enc = load_cost_model('cost_model.pkl')
    
    # Update model status based on load result
    if model_info and model_info["status"] in ["success", "warning"]:
        st.session_state.model_status = "ready"
    else:
        st.session_state.model_status = "error"
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ System Status")
        
        # Handle structured model info
        if model_info and model_info["status"] == "success":
            st.success(f"✅ Detection Model: {model_info['message']}")
            st.caption(f"Provider: {model_info['provider']}")
            if model_info.get('output_shape'):
                st.caption(f"Output: {model_info['output_shape']}")
            st.session_state.model_status = "ready"
            
        elif model_info and model_info["status"] == "warning":
            st.warning(f"⚠️ Detection Model: {model_info['message']}")
            st.caption(f"Provider: {model_info['provider']}")
            st.info(f"💡 {model_info.get('details', '')}")
            st.session_state.model_status = "ready"  # CPU still works!
            
        else:  # error or not loaded
            st.error("❌ Detection Model: Failed to Load")
            if model_info:
                st.code(f"{model_info.get('message', '')}\n{model_info.get('details', '')}")
            st.session_state.model_status = "error"
        
        st.divider()
        
        # Cost model status
        if cost_model:
            st.success("✅ Cost Model: ML Ready")
        else:
            st.warning("⚠️ Cost Model: Rule-Based Fallback Active")
            st.caption("Estimates based on damage type + severity")
            
        st.divider()
        st.header("🎛️ Settings")
        conf_thresh = st.slider("Confidence Threshold", 0.1, 0.9, 0.25, 0.05,
                               help="Higher = fewer false positives")
        iou_thresh = st.slider("NMS IoU Threshold", 0.3, 0.7, 0.45, 0.05,
                              help="Higher = keep more overlapping boxes")
        
        st.divider()
        st.markdown("### 📊 Quick Stats")
        st.metric("Damage Classes", len(DAMAGE_CLASSES))
        st.metric("Inference Engine", "ONNX Runtime")
        st.metric("Focus", "Damage Detection Only")
        
        # GPU hint
        if model_info and model_info.get('provider') == 'CPUExecutionProvider':
            with st.expander("🚀 Enable GPU Acceleration"):
                st.markdown("""
                **For faster inference:**
                1. Install CUDA 12: https://developer.nvidia.com/cuda-12-0-download-archive
                2. Install cuDNN 9: https://developer.nvidia.com/cudnn
                3. Add to PATH:
                   - `C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.0\\bin`
                4. Restart app
                """)
                st.code("pip install onnxruntime-gpu  # If using GPU")
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["📤 Upload & Analyze", "📋 Results", "📄 Report"])
    
    # ==================== TAB 1: UPLOAD & ANALYZE ====================
    with tab1:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📷 Upload Vehicle Images")
            # ✅ Multi-image upload support
            uploaded = st.file_uploader("Choose images...", type=['jpg', 'jpeg', 'png', 'bmp'], accept_multiple_files=True)
            
            if uploaded:
                st.markdown(f"📂 **{len(uploaded)}** images selected")
                
                # Vehicle info (applies to all images in batch)
                with st.expander("🚙 Vehicle Info (Applies to all images)", expanded=True):
                    c1, c2 = st.columns(2)
                    brand = c1.selectbox("Brand", ['Maruti Suzuki', 'Hyundai', 'Tata', 'Honda', 'Mahindra', 'Toyota', 'Other'])
                    model = c2.text_input("Model", placeholder="e.g., Swift, Creta", key="model_input")
                    c3, c4 = st.columns(2)
                    year = c3.number_input("Year", 2010, 2025, 2020)
                    mileage = c4.number_input("Mileage (km)", 0, 500000, 50000)
                
                # ✅ Batch analyze button with progress
                if st.button("🔍 Analyze All Images", type="primary", use_container_width=True, disabled=(st.session_state.model_status != "ready")):
                    if not model.strip():
                        st.error("⚠️ Enter vehicle model for cost estimation")
                    else:
                        # Clear previous results
                        st.session_state.batch_results = []
                        progress = st.progress(0, text="🔎 Processing images...")
                        
                        for i, file in enumerate(uploaded):
                            try:
                                # Load and preprocess image
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
                                
                                # Enrich detections with severity and cost
                                for det in detections:
                                    sev, sev_score = calculate_severity(
                                        det['confidence'], det['bbox'], orig_shape
                                    )
                                    det['severity'] = sev
                                    det['severity_score'] = sev_score
                                    
                                    # Cost estimation
                                    vehicle_info = {'brand': brand, 'model': model.strip()}
                                    det['cost_estimate'] = predict_cost(
                                        det, vehicle_info, cost_model, model_enc, part_enc
                                    )
                                
                                # Draw high-visibility annotations
                                vis_img = draw_detections(img_bgr, detections)
                                vis_rgb = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
                                
                                # Store results for this image
                                st.session_state.batch_results.append({
                                    'filename': file.name,
                                    'original': img,
                                    'processed': vis_rgb,
                                    'detections': detections,
                                    'vehicle_info': {
                                        'brand': brand,
                                        'model': model.strip(),
                                        'year': year,
                                        'mileage': mileage
                                    },
                                    'orig_shape': orig_shape,
                                    'settings': {'conf': conf_thresh, 'iou': iou_thresh}
                                })
                                
                            except Exception as e:
                                logger.error(f"Failed to process {file.name}: {e}", exc_info=True)
                                st.warning(f"⚠️ Failed to process {file.name}: {e}")
                            
                            # Update progress bar
                            progress.progress((i+1)/len(uploaded), text=f"✅ Processed {i+1}/{len(uploaded)} images")
                        
                        st.success(f"🎉 Analysis Complete! {len(st.session_state.batch_results)} images processed.")
                        st.rerun()
        
        with col2:
            st.subheader("💡 Tips for Best Results")
            st.markdown("""
            - 📸 Use good lighting and clear focus
            - 🎯 Capture the damaged area prominently
            - 📐 Include some context (helps with scale)
            - 🔄 Try different angles if detection is uncertain
            - ✂️ Crop to damaged area for faster processing
            """)
            
            st.divider()
            st.subheader("🔧 Supported Damage Types")
            for i, cls in enumerate(DAMAGE_CLASSES, 1):
                st.markdown(f"{i}. `{cls.replace('_', ' ').title()}`")
            
            st.divider()
            st.subheader("📈 Severity Levels")
            st.markdown("""
            - 🟢 **Low**: Minor cosmetic issues
            - 🟡 **Medium**: Noticeable damage, repair recommended  
            - 🔴 **High**: Structural/safety concern, urgent repair
            """)
    
    # ==================== TAB 2: RESULTS ====================
    with tab2:
        if not st.session_state.batch_results:
            st.info("👆 Upload and analyze images first to see results")
        else:
            # ✅ Image selector for multi-image navigation
            filenames = [r['filename'] for r in st.session_state.batch_results]
            selected_idx = st.selectbox("Select Image to View", range(len(filenames)), format_func=lambda i: filenames[i])
            res = st.session_state.batch_results[selected_idx]
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.image(res['processed'], 
                        caption=f"🖼️ {res['filename']} (🟢Low 🟡Medium 🔴High)",
                        use_column_width=True)
            with col2:
                detections = res['detections']
                st.metric("Damages", len(detections))
                if detections:
                    high_count = sum(1 for d in detections if d['severity'] == 'high')
                    st.metric("High Severity", high_count, delta_color="inverse")
                    total_cost = sum(d['cost_estimate']['total_cost'] for d in detections)
                    st.metric("Est. Total Cost", f"₹{total_cost:,.0f}")
                    avg_conf = np.mean([d['confidence'] for d in detections])
                    st.metric("Avg. Confidence", f"{avg_conf:.2%}")
            
            if res['detections']:
                st.divider()
                st.subheader("🔍 Detection Details")
                
                # Build table data
                table_data = []
                for i, det in enumerate(res['detections'], 1):
                    table_data.append({
                        '#': i,
                        'Type': det['class_name'].replace('_', ' ').title(),
                        'Confidence': f"{det['confidence']:.2%}",
                        'Severity': det['severity'].upper(),
                        'Score': det['severity_score'],
                        'Est. Cost': f"₹{det['cost_estimate']['total_cost']:,.0f}",
                        'Method': det['cost_estimate']['method']
                    })
                
                df = pd.DataFrame(table_data)
                st.dataframe(df, use_container_width=True, hide_index=True)
                
                # Detailed breakdown per detection
                st.subheader("📋 Damage Breakdown")
                for i, det in enumerate(res['detections'], 1):
                    with st.expander(
                        f"Damage #{i}: {det['class_name'].replace('_', ' ').title()} | {det['severity'].upper()} | ₹{det['cost_estimate']['total_cost']:,.0f}",
                        expanded=(i==1)
                    ):
                        ce = det['cost_estimate']
                        # ✅ Safe access to breakdown keys (handles both formats)
                        part_cost = ce['breakdown'].get('part_cost') or ce['breakdown'].get('part', 0)
                        labor_cost = ce['breakdown'].get('labor_cost') or ce['breakdown'].get('labor', 0)
                        paint_cost = ce['breakdown'].get('paint_cost') or ce['breakdown'].get('paint', 0)
                        
                        st.markdown(f"""
                        **Detection Details:**
                        - Confidence: `{det['confidence']:.2%}`
                        - Severity Score: `{det['severity_score']:.3f}`
                        - Bounding Box: `[{' '.join(f'{c:.1f}' for c in det['bbox'])}]`
                        
                        **Cost Estimate:**
                        - 💰 Total: **₹{ce['total_cost']:,.0f}** (`{ce['method']}`)
                        - 🔧 Parts: ₹{part_cost:,.0f}
                        - 👷 Labor: ₹{labor_cost:,.0f}  
                        - 🎨 Paint: ₹{paint_cost:,.0f}
                        """)
    
    # ==================== TAB 3: REPORT ====================
    with tab3:
        if not st.session_state.batch_results:
            st.info("👆 Complete an analysis first to generate a report")
        else:
            # Export options for all or selected image
            st.header("📄 Export Reports")
            export_mode = st.radio("Export Scope", ["All Images", "Selected Image Only"], horizontal=True)
            
            def build_report(results):
                """Build report dict(s) for export"""
                reports = []
                for r in results:
                    total = sum(d['cost_estimate']['total_cost'] for d in r['detections'])
                    reports.append({
                        'report_id': f"VDA-{datetime.now().strftime('%Y%m%d%H%M%S')}-{r['filename'].split('.')[0]}",
                        'filename': r['filename'],
                        'vehicle': r['vehicle_info'],
                        'summary': {
                            'damages': len(r['detections']),
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

            # Build reports based on export mode
            if export_mode == "Selected Image Only":
                reports = build_report([st.session_state.batch_results[selected_idx]])
            else:
                reports = build_report(st.session_state.batch_results)
            
            c1, c2, c3 = st.columns(3)
            with c1:
                json_str = json.dumps(reports, indent=2, default=str)
                st.download_button(
                    label="📄 Download JSON",
                    data=json_str,
                    file_name=f"damage_report_{datetime.now().strftime('%Y%m%d')}.json",
                    mime="application/json",
                    use_container_width=True
                )
            with c2:
                csv_data = []
                for r in reports:
                    for d in r['damages']:
                        csv_data.append({
                            'File': r['filename'],
                            'Damage_Type': d['class_name'],
                            'Confidence': d['confidence'],
                            'Severity': d['severity'],
                            'Severity_Score': d['severity_score'],
                            'Estimated_Cost': d['cost_estimate']['total_cost'],
                            'Cost_Method': d['cost_estimate']['method'],
                            'Part_Cost': d['cost_estimate']['breakdown'].get('part_cost') or d['cost_estimate']['breakdown'].get('part', 0),
                            'Labor_Cost': d['cost_estimate']['breakdown'].get('labor_cost') or d['cost_estimate']['breakdown'].get('labor', 0),
                            'Paint_Cost': d['cost_estimate']['breakdown'].get('paint_cost') or d['cost_estimate']['breakdown'].get('paint', 0),
                            'BBox_X1': d['bbox'][0],
                            'BBox_Y1': d['bbox'][1],
                            'BBox_X2': d['bbox'][2],
                            'BBox_Y2': d['bbox'][3]
                        })
                if csv_data:
                    st.download_button(
                        label="📊 Download CSV",
                        data=pd.DataFrame(csv_data).to_csv(index=False),
                        file_name=f"damage_details_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                else:
                    st.info("No detections to export")
            with c3:
                st.button("📕 Generate PDF", disabled=True, 
                         help="PDF export requires: pip install reportlab")
                st.caption("*Coming soon*")
            
            st.divider()
            st.subheader("📋 Summary Preview")
            for r in reports:
                st.markdown(f"**{r['filename']}** | Damages: `{r['summary']['damages']}` | Est. Cost: `₹{r['summary']['total_cost']:,.0f}`")
                st.markdown(f"Types: `{', '.join(r['summary']['damage_types'])}`")
                for rec in r['recommendations']:
                    st.markdown(f"- {rec}")
            
            st.info("""
            🔗 **Insurance Integration Ready**  
            This report contains structured data for automated claim processing:
            - ✅ Damage classifications with confidence scores
            - ✅ Severity assessments with quantitative metrics
            - ✅ Itemized cost estimates with component breakdowns
            - ✅ Spatial damage mapping via bounding box coordinates
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