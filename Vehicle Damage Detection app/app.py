"""
AI Vehicle Damage Detection System
ONNX-based inference for damage detection only
"""
import os
import json
import cv2
import numpy as np
from PIL import Image
from datetime import datetime
from pathlib import Path

# =============================================================================
# ⚠️ CRITICAL: Import streamlit AND set_page_config FIRST
# =============================================================================
import streamlit as st

# THIS MUST BE THE FIRST STREAMLIT COMMAND - NO EXCEPTIONS
st.set_page_config(
    page_title="🚗 Vehicle Damage Detection",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)
# =============================================================================

# Now safe to use other Streamlit commands
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    # Show warning LATER in sidebar, not at module level

# =============================================================================
# CONFIGURATION & CONSTANTS
# =============================================================================

# Damage classes (must match your training)
DAMAGE_CLASSES = ['dent', 'scratch', 'crack', 'glass_shatter', 'lamp_broken', 'tire_flat']

# Color map for severity visualization
SEVERITY_COLORS = {
    'low': (0, 255, 0),      # Green
    'medium': (0, 255, 255), # Yellow/Cyan
    'high': (0, 0, 255)      # Red
}

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

@st.cache_resource
def load_onnx_model(model_path: str = 'best_damage_model.onnx'):
    """Load ONNX model with session options"""
    if not ONNX_AVAILABLE:
        return None
    if not os.path.exists(model_path):
        return None
    
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_options.intra_op_num_threads = 4
    
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    
    try:
        session = ort.InferenceSession(model_path, sess_options=session_options, providers=providers)
        return session
    except Exception as e:
        st.error(f"❌ Failed to load ONNX model: {e}")
        return None

def preprocess_image(image: np.ndarray, img_size: int = 640):
    """Preprocess image for ONNX model inference"""
    h, w = image.shape[:2]
    scale = img_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    padded = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    padded[:new_h, :new_w] = resized
    padded = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    padded = np.transpose(padded, (2, 0, 1))
    padded = np.expand_dims(padded, axis=0)
    
    return padded, scale, (h, w)

def postprocess_onnx_output(output, original_shape, scale, conf_threshold=0.25, iou_threshold=0.45):
    """Postprocess ONNX model output to get detections"""
    output = output[0]
    detections = []
    
    for box in output:
        cx, cy, w, h, obj_conf = box[:5]
        class_probs = box[5:5+len(DAMAGE_CLASSES)]
        
        if obj_conf < conf_threshold:
            continue
        
        class_id = np.argmax(class_probs)
        class_conf = class_probs[class_id]
        confidence = obj_conf * class_conf
        
        if confidence < conf_threshold:
            continue
        
        x1 = (cx - w/2) / scale
        y1 = (cy - h/2) / scale
        x2 = (cx + w/2) / scale
        y2 = (cy + h/2) / scale
        
        h_orig, w_orig = original_shape
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_orig, x2), min(h_orig, y2)
        
        detections.append({
            'bbox': [float(x1), float(y1), float(x2), float(y2)],
            'confidence': float(confidence),
            'class_id': int(class_id),
            'class_name': DAMAGE_CLASSES[class_id] if class_id < len(DAMAGE_CLASSES) else 'unknown'
        })
    
    if detections:
        detections = nms(detections, iou_threshold)
    
    return detections

def nms(detections, iou_threshold):
    """Simple Non-Maximum Suppression"""
    if not detections:
        return []
    
    detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
    keep = []
    
    while detections:
        best = detections.pop(0)
        keep.append(best)
        detections = [d for d in detections if calculate_iou(best['bbox'], d['bbox']) < iou_threshold]
    
    return keep

def calculate_iou(box1, box2):
    """Calculate Intersection over Union"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    if inter_area == 0:
        return 0.0
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area

def calculate_severity(confidence, bbox, img_shape):
    """Calculate damage severity score"""
    x1, y1, x2, y2 = bbox
    area = (x2 - x1) * (y2 - y1)
    total_area = img_shape[0] * img_shape[1]
    area_ratio = min(area / total_area, 1.0)
    score = min(confidence * 0.7 + area_ratio * 0.3, 1.0)
    
    if score < 0.25:
        return 'low', round(score, 3)
    elif score < 0.60:
        return 'medium', round(score, 3)
    else:
        return 'high', round(score, 3)

def calculate_location(bbox, img_shape):
    """Determine damage location on image"""
    h, w = img_shape[:2]
    cx, cy = (bbox[0] + bbox[2])/2, (bbox[1] + bbox[3])/2
    
    h_pos = 'left' if cx < w/3 else ('right' if cx > 2*w/3 else 'center')
    v_pos = 'top' if cy < h/3 else ('bottom' if cy > 2*h/3 else 'middle')
    return f"{v_pos}-{h_pos}"

def draw_detections(image: np.ndarray, detections: list, show_labels=True):
    """Draw bounding boxes and labels on image"""
    img_copy = image.copy()
    
    for det in detections:
        x1, y1, x2, y2 = map(int, det['bbox'])
        severity = det.get('severity', 'medium')
        color = SEVERITY_COLORS.get(severity, (255, 255, 255))
        
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)
        
        if show_labels:
            label = f"{det['class_name']} ({det['confidence']:.2f})"
            if 'severity' in det:
                label += f" [{det['severity']}]"
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
            
            cv2.rectangle(img_copy, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
            cv2.putText(img_copy, label, (x1, y1 - 5), font, font_scale, (0, 0, 0), thickness)
    
    return img_copy

def generate_report(detections: list, image_name: str):
    """Generate comprehensive assessment report"""
    return {
        'report_id': f"VDD-{datetime.now().strftime('%Y%m%d%H%M%S')}",
        'timestamp': datetime.now().isoformat(),
        'image_name': image_name,
        'summary': {
            'total_damages': len(detections),
            'severity_distribution': {
                'low': sum(1 for d in detections if d.get('severity') == 'low'),
                'medium': sum(1 for d in detections if d.get('severity') == 'medium'),
                'high': sum(1 for d in detections if d.get('severity') == 'high')
            }
        },
        'damages': detections,
        'recommendations': generate_recommendations(detections)
    }

def generate_recommendations(detections: list):
    """Generate repair recommendations"""
    recommendations = []
    
    high_severity = [d for d in detections if d.get('severity') == 'high']
    if high_severity:
        recommendations.append("🚨 IMMEDIATE ACTION: High severity damages require urgent repair")
    
    if any(d['class_name'] == 'crack' for d in detections):
        recommendations.append("⚠️ Windshield cracks compromise structural integrity - repair immediately")
    
    if any(d['class_name'] == 'glass_shatter' for d in detections):
        recommendations.append("🪟 Broken glass requires replacement for safety")
    
    if any(d['class_name'] == 'tire_flat' for d in detections):
        recommendations.append("🛞 Flat tire detected - do not drive vehicle until replaced")
    
    medium_severity = [d for d in detections if d.get('severity') == 'medium']
    if len(medium_severity) >= 2:
        recommendations.append("🔧 Schedule repairs for medium severity damages within 1-2 weeks")
    
    if not recommendations and detections:
        recommendations.append("✅ Minor damages detected - maintenance recommended when convenient")
    
    if not detections:
        recommendations.append("✨ No damages detected - vehicle appears to be in good condition!")
    
    return recommendations

# =============================================================================
# STREAMLIT UI
# =============================================================================

def main():
    # Header
    st.title("🚗 AI Vehicle Damage Detection")
    st.markdown("*Upload vehicle images to detect damage using ONNX-based YOLOv8 model*")
    
    # Sidebar - Model Status & Settings
    with st.sidebar:
        st.header("⚙️ System Status")
        
        # Check ONNX model
        onnx_model = load_onnx_model('best_damage_model.onnx')
        if onnx_model:
            st.success("✅ Detection Model: Loaded (ONNX)")
            st.caption(f"Providers: {onnx_model.get_providers()}")
        else:
            st.error("❌ Detection Model: Not Found")
            st.caption("Expected: `best_damage_model.onnx` in current directory")
            if not ONNX_AVAILABLE:
                st.warning("⚠️ Install onnxruntime: `pip install onnxruntime`")
            if st.button("🔄 Export Model Now"):
                st.info("Run this in terminal:\n```bash\nyolo export model=best.pt format=onnx opset=12 simplify=True\n```")
        
        st.divider()
        
        # Settings
        st.header("🎛️ Detection Settings")
        conf_threshold = st.slider("Confidence Threshold", 0.1, 0.9, 0.25, 0.05)
        iou_threshold = st.slider("NMS IoU Threshold", 0.3, 0.7, 0.45, 0.05)
        
        st.divider()
        st.markdown("### 📊 Quick Stats")
        st.metric("Damage Classes", len(DAMAGE_CLASSES))
        st.metric("Input Size", "640x640")
        st.metric("Avg. Inference Time", "~150ms", "on CPU")
    
    # Main Content Tabs
    tab1, tab2, tab3 = st.tabs(["📤 Upload & Analyze", "📋 Results", "📄 Report"])
    
    # Session state
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'processed_image' not in st.session_state:
        st.session_state.processed_image = None
    
    # ==================== TAB 1: UPLOAD & ANALYZE ====================
    with tab1:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📷 Upload Vehicle Image")
            uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png', 'bmp'])
            
            if uploaded_file:
                image = Image.open(uploaded_file).convert('RGB')
                image_np = np.array(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))
                
                st.image(image, caption="Uploaded Image", use_column_width=True)
                
                if st.button("🔍 Detect Damage", type="primary", use_container_width=True):
                    if not onnx_model:
                        st.error("❌ Detection model not loaded. Please export your model to ONNX format first.")
                    else:
                        with st.spinner("🔎 Analyzing image..."):
                            input_tensor, scale, orig_shape = preprocess_image(image_np)
                            input_name = onnx_model.get_inputs()[0].name
                            outputs = onnx_model.run(None, {input_name: input_tensor})
                            detections = postprocess_onnx_output(outputs[0], orig_shape, scale, conf_threshold, iou_threshold)
                            
                            for det in detections:
                                severity, sev_score = calculate_severity(det['confidence'], det['bbox'], orig_shape)
                                det['severity'] = severity
                                det['severity_score'] = sev_score
                                det['location'] = calculate_location(det['bbox'], orig_shape)
                            
                            result_image = draw_detections(image_np, detections)
                            result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
                            
                            st.session_state.analysis_results = {
                                'detections': detections,
                                'original_shape': orig_shape,
                                'image_name': uploaded_file.name
                            }
                            st.session_state.processed_image = result_image_rgb
                            
                            st.success(f"✅ Analysis Complete! Found {len(detections)} damage(s)")
                            st.rerun()
        
        with col2:
            st.subheader("💡 Tips for Best Results")
            st.markdown("""
            - 📸 Use good lighting and clear focus
            - 🎯 Capture the damaged area prominently
            - 📐 Include context (whole vehicle helps detection)
            - 🔄 Try different angles if detection is uncertain
            """)
            
            st.divider()
            st.subheader("🔧 Supported Damage Types")
            for cls in DAMAGE_CLASSES:
                st.markdown(f"- `{cls.replace('_', ' ').title()}`")
            
            st.divider()
            st.subheader("🎨 Severity Colors")
            st.markdown("🟢 **Low**: Minor damage")
            st.markdown("🟡 **Medium**: Moderate damage")
            st.markdown("🔴 **High**: Severe damage")
    
    # ==================== TAB 2: RESULTS ====================
    with tab2:
        if st.session_state.processed_image is None:
            st.info("👆 Upload and analyze an image first to see results")
        else:
            st.image(st.session_state.processed_image, 
                    caption="Detected Damages (Color: 🟢Low 🟡Medium 🔴High)",
                    use_column_width=True)
            
            results = st.session_state.analysis_results
            detections = results['detections']
            
            if not detections:
                st.success("✨ No damages detected in this image!")
            else:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Damages", len(detections))
                with col2:
                    high_count = sum(1 for d in detections if d['severity'] == 'high')
                    st.metric("High Severity", high_count, delta_color="inverse")
                with col3:
                    st.metric("Avg. Confidence", f"{np.mean([d['confidence'] for d in detections]):.2%}")
                
                st.divider()
                st.subheader("🔍 Damage Details")
                
                table_data = []
                for i, det in enumerate(detections, 1):
                    table_data.append({
                        '#': i,
                        'Type': det['class_name'].replace('_', ' ').title(),
                        'Confidence': f"{det['confidence']:.2%}",
                        'Severity': det['severity'].upper(),
                        'Location': det['location'],
                        'BBox': f"[{det['bbox'][0]:.0f}, {det['bbox'][1]:.0f}, {det['bbox'][2]:.0f}, {det['bbox'][3]:.0f}]"
                    })
                
                import pandas as pd
                df = pd.DataFrame(table_data)
                st.dataframe(df, use_container_width=True, hide_index=True)
                
                st.subheader("📋 Damage Breakdown")
                for i, det in enumerate(detections, 1):
                    with st.expander(f"Damage #{i}: {det['class_name'].title()}", expanded=(i==1)):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown(f"""
                            **Details:**
                            - Confidence: `{det['confidence']:.2%}`
                            - Severity Score: `{det['severity_score']:.3f}`
                            - Location: `{det['location']}`
                            """)
                        with col2:
                            st.markdown(f"""
                            **Bounding Box:**
                            - X1: `{det['bbox'][0]:.1f}`
                            - Y1: `{det['bbox'][1]:.1f}`
                            - X2: `{det['bbox'][2]:.1f}`
                            - Y2: `{det['bbox'][3]:.1f}`
                            """)
    
    # ==================== TAB 3: REPORT ====================
    with tab3:
        if st.session_state.analysis_results is None:
            st.info("👆 Complete an analysis first to generate a report")
        else:
            results = st.session_state.analysis_results
            detections = results['detections']
            image_name = results.get('image_name', 'unknown')
            report = generate_report(detections, image_name)
            
            st.header("📄 Detection Report")
            st.markdown(f"""
            **Report ID:** `{report['report_id']}`  
            **Generated:** `{report['timestamp']}`  
            **Image:** `{image_name}`
            """)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Damages Found", report['summary']['total_damages'])
            with col2:
                high = report['summary']['severity_distribution']['high']
                st.metric("High Severity Items", high, delta_color="inverse")
            
            st.subheader("💡 Recommendations")
            for rec in report['recommendations']:
                st.markdown(f"- {rec}")
            
            st.divider()
            st.subheader("💾 Export Report")
            
            col1, col2 = st.columns(2)
            
            with col1:
                json_str = json.dumps(report, indent=2)
                st.download_button(
                    label="📄 Download JSON",
                    data=json_str,
                    file_name=f"{report['report_id']}.json",
                    mime="application/json",
                    use_container_width=True
                )
            
            with col2:
                if detections:
                    import pandas as pd
                    csv_data = pd.DataFrame([{
                        'Damage_Type': d['class_name'],
                        'Confidence': d['confidence'],
                        'Severity': d['severity'],
                        'Severity_Score': d['severity_score'],
                        'Location': d['location'],
                        'BBox_X1': d['bbox'][0],
                        'BBox_Y1': d['bbox'][1],
                        'BBox_X2': d['bbox'][2],
                        'BBox_Y2': d['bbox'][3]
                    } for d in detections])
                    csv_str = csv_data.to_csv(index=False)
                    st.download_button(
                        label="📊 Download CSV",
                        data=csv_str,
                        file_name=f"{report['report_id']}_details.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
            
            st.info("🔜 **Coming Soon**: Vehicle part identification, repair cost estimation, and insurance integration")

# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()