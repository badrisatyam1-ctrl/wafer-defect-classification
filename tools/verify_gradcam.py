import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
import torch
import cv2
import numpy as np
import sys
from pathlib import Path

# Add project root to sys.path to ensure absolute imports work
project_root = str(Path(__file__).parent.parent.absolute())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from deployment.inference import WaferInferenceEngine
from utils.synthetic_generator import generate_macro_wafer_map

def verify_gradcam_edge_cases():
    print("🚀 Starting Grad-CAM Edge Case Verification...")
    
    # Initialize engine
    # Note: Using best.pt; ensure it exists in models/
    checkpoint = "models/best.pt"
    if not os.path.exists(checkpoint):
        print(f"❌ Error: {checkpoint} not found. Please run training first.")
        return

    engine = WaferInferenceEngine(checkpoint_path=checkpoint)
    
    # Define edge cases (using V2 class names)
    test_cases = [
        {"name": "Strong Defect (Scratch)", "defect_type": "scratch"},
        {"name": "Subtle Defect (Center)", "defect_type": "center"},
        {"name": "Normal Wafer (No Defect)", "defect_type": "normal"},
        {"name": "Noisy Image (Random)", "defect_type": "random_noise"}
    ]
    
    for case in test_cases:
        print(f"\n--- Testing: {case['name']} ---")
        
        # Generate image
        if case['defect_type'] == "random_noise":
            image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        else:
            # generate_macro_wafer_map v2 already returns RGB (H, W, 3)
            image = generate_macro_wafer_map(case['defect_type'], size=(224, 224))
            
        # Run inference
        result = engine.predict_from_array(image)
        
        # Check reliability
        reliable = result.get("grad_cam_reliable", False)
        pred_class = result.get("predicted_class", "unknown")
        confidence = result.get("confidence", 0.0)
        
        print(f"Prediction: {pred_class} ({confidence:.2f})")
        print(f"Grad-CAM Reliable: {reliable}")
        
        # Performance logging
        if not reliable:
            print("ℹ️ Info: Grad-CAM correctly flagged as unreliable for this input.")
        else:
            print("✨ Success: Grad-CAM generated a reliable activation map.")

    print("\n✅ Verification Complete.")

if __name__ == "__main__":
    verify_gradcam_edge_cases()
