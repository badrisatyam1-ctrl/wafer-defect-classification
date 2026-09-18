import urllib.request
import json
import os

def test_system_info():
    req = urllib.request.urlopen("http://127.0.0.1:8000/api/system_info")
    data = json.loads(req.read().decode())
    print("[TEST 1] System info:", data["model_architecture"], "Val acc:", data["val_accuracy"], "Macro F1:", data["macro_f1"], "Gating:", data.get("camera_gating"))

def test_predict(filepath, desc):
    with open(filepath, "rb") as f:
        file_bytes = f.read()
    
    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
    fname = os.path.basename(filepath)
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{fname}"\r\n'
        f"Content-Type: image/png\r\n\r\n"
    ).encode("utf-8") + file_bytes + f"\r\n--{boundary}--\r\n".encode("utf-8")
    
    req = urllib.request.Request(
        "http://127.0.0.1:8000/api/predict",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
    )
    resp = urllib.request.urlopen(req)
    res = json.loads(resp.read().decode())
    print(f"[TEST 2: {desc}] -> status: {res.get('status')}, class: {res.get('class')}, conf: {res.get('confidence')}, err: {res.get('error_type')}")

def test_synthetic(cls):
    data = json.dumps({"defect_class": cls}).encode("utf-8")
    req = urllib.request.Request("http://127.0.0.1:8000/api/synthetic", data=data, headers={"Content-Type": "application/json"})
    resp = urllib.request.urlopen(req)
    res = json.loads(resp.read().decode())
    print(f"[TEST 3: Synthetic {cls}] -> status: {res.get('status')}, class: {res.get('class')}, conf: {res.get('confidence')}")

if __name__ == "__main__":
    test_system_info()
    test_predict(r"C:\Users\badri\.gemini\antigravity-ide\brain\9f56b1bf-1a01-4f98-9b89-17a7ef396510\.user_uploaded\media_1789741647602.png", "Face / Webcam Screenshot")
    test_predict("real_demo_images/real_center_1.png", "Real Center Wafer")
    test_predict("real_demo_images/real_cluster_1.png", "Real Cluster Wafer")
    test_synthetic("center")
    test_synthetic("ring")
    test_synthetic("scratch")
    test_synthetic("edge_ring")
    test_synthetic("full_fail")
