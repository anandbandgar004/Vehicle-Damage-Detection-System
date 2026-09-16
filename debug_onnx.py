# debug_onnx.py
import os, sys, numpy as np, logging, traceback

logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

MODEL_PATH = 'runs/detect/runs/train/damage_detect_v2/weights/best.onnx'

print("🔍 ONNX Model Diagnostic")
print("="*70)

# 1. File checks
print(f"1️⃣ File exists: {os.path.exists(MODEL_PATH)}")
if not os.path.exists(MODEL_PATH):
    print("   ❌ File not found at path")
    sys.exit(1)

size_mb = os.path.getsize(MODEL_PATH) / (1024*1024)
print(f"2️⃣ File size: {size_mb:.2f} MB")
if size_mb < 5:
    print("   ⚠️  Suspiciously small — export may have failed")

# 2. Try loading with onnxruntime
try:
    import onnxruntime as ort
    print(f"3️⃣ onnxruntime version: {ort.__version__}")
    print(f"   Available providers: {ort.get_available_providers()}")
    
    # Try loading with explicit provider order
    providers = ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
    
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    print("4️⃣ Attempting to load model...")
    session = ort.InferenceSession(MODEL_PATH, sess_options=session_options, providers=providers)
    
    print("✅ Model loaded successfully!")
    
    # Inspect I/O
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    print(f"5️⃣ Input: name='{inputs[0].name}', shape={inputs[0].shape}, type={inputs[0].type}")
    print(f"6️⃣ Output: name='{outputs[0].name}', shape={outputs[0].shape}, type={outputs[0].type}")
    
    # Check active provider
    print(f"7️⃣ Active provider: {session.get_providers()[0]}")
    
    # Test inference with dummy input
    print("8️⃣ Testing inference with dummy input...")
    dummy = np.random.randn(1, 3, 640, 640).astype(np.float32)
    result = session.run(None, {inputs[0].name: dummy})
    print(f"✅ Inference successful! Output shape: {result[0].shape}")
    
    # Check for FP16
    if 'float16' in outputs[0].type.lower():
        print("⚠️  Model uses FP16 — ensure GPU provider is active for best performance")
    
    print("\n🎉 ALL CHECKS PASSED — Model is valid!")
    
except ImportError:
    print("❌ onnxruntime not installed")
except Exception as e:
    print(f"❌ Load failed: {type(e).__name__}: {e}")
    print("\n📋 Full traceback:")
    traceback.print_exc()
    
    # Common fixes based on error type
    err = str(e).lower()
    if 'half' in err or 'float16' in err:
        print("\n💡 FIX: Re-export model with half=False for CPU compatibility")
        print("   yolo export model=best.pt format=onnx half=False")
    elif 'shape' in err or 'dimension' in err:
        print("\n💡 FIX: Check preprocessing matches model input shape")
    elif 'node' in err or 'operator' in err or 'op' in err:
        print("\n💡 FIX: Upgrade onnxruntime or export with lower opset")
        print("   pip install --upgrade onnxruntime")
        print("   yolo export model=best.pt format=onnx opset=11")
    elif 'cuda' in err or 'tensorrt' in err:
        print("\n💡 FIX: Try CPU-only provider")
        print("   In code: providers=['CPUExecutionProvider']")
    elif 'parse' in err or 'protobuf' in err:
        print("\n💡 FIX: Model file may be corrupted — re-export")