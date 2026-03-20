import cv2
import numpy as np
import os

# -------------------------------
# YOLO LOAD (auto-update from training)
# -------------------------------
# Priority: live training weights > project checkpoint
_TRAINING_WEIGHTS = os.path.expanduser("~/runs/detect/runs/yolo_wafer10k/weights/best.pt")
_PROJECT_WEIGHTS = "models/checkpoints/yolo_wafer_best.pt"

try:
    from ultralytics import YOLO
    if os.path.exists(_TRAINING_WEIGHTS):
        yolo_model = YOLO(_TRAINING_WEIGHTS)
    elif os.path.exists(_PROJECT_WEIGHTS):
        yolo_model = YOLO(_PROJECT_WEIGHTS)
    else:
        yolo_model = YOLO("yolov8s.pt")
    YOLO_AVAILABLE = True
except:
    YOLO_AVAILABLE = False


# -------------------------------
# FALLBACK DETECTOR (WORKING)
# -------------------------------
def fallback_wafer_check(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # smooth
    gray = cv2.GaussianBlur(gray, (9, 9), 2)

    # detect circle
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=150,
        param1=100,
        param2=30,
        minRadius=100,
        maxRadius=400
    )

    if circles is None:
        return False, None

    # take first circle
    x, y, r = circles[0][0]
    x, y, r = int(x), int(y), int(r)

    # Ensure box bounds are valid preventing array slice errors
    h, w = image.shape[:2]
    x1, y1, x2, y2 = max(0, x - r), max(0, y - r), min(w, x + r), min(h, y + r)

    return True, (x1, y1, x2, y2)


# -------------------------------
# YOLO DETECTOR
# -------------------------------
def detect_wafer_yolo(image):
    results = yolo_model(image, conf=0.1, verbose=False)

    for r in results:
        if r.boxes is not None and len(r.boxes) > 0:
            box = r.boxes.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, box)
            return True, (x1, y1, x2, y2)

    return False, None


# -------------------------------
# FINAL DETECTION FUNCTION
# -------------------------------
def detect_wafer(image):
    # Try YOLO first
    if YOLO_AVAILABLE:
        detected, box = detect_wafer_yolo(image)
        if detected:
            return True, box

    # fallback
    detected, box = fallback_wafer_check(image)
    if detected:
        return True, box

    return False, None
