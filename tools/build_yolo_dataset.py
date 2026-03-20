"""
YOLO Wafer Detection Dataset Builder (10K Scale)
Generates a massive dataset of Wafers vs Non-Wafers using multithreading.
"""

import sys
import os
import cv2
import numpy as np
import requests
import random
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "dataset"

def setup_dirs():
    dirs_to_make = [
        DATA_DIR / "images" / "train",
        DATA_DIR / "images" / "val",
        DATA_DIR / "labels" / "train",
        DATA_DIR / "labels" / "val",
    ]
    for d in dirs_to_make:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

def is_blurry(image: np.ndarray, threshold: float = 100.0) -> bool:
    if image is None: return True
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < threshold

def make_diverse_wafer(size: int = 640) -> np.ndarray:
    colors = [
        (100, 100, 100), (120, 100, 80), (80, 80, 120),
        (90, 120, 90), (70, 70, 70), (140, 130, 120),
        (110, 110, 115), (85, 90, 85)
    ]
    base_col = np.array(random.choice(colors))
    img = np.zeros((size, size, 3), dtype=np.uint8)
    
    bg_choice = random.random()
    if bg_choice < 0.25:
        bg_col = (0, 0, 0) # Pure black
    elif bg_choice < 0.50:
        bg_col = (random.randint(10, 50), random.randint(10, 50), random.randint(10, 50))
    elif bg_choice < 0.75:
        bg_col = (random.randint(150, 220), random.randint(150, 220), random.randint(150, 220))
    else:
        bg_col = (random.randint(50, 150), random.randint(50, 150), random.randint(50, 150))
        
    img[:,:] = bg_col
    
    if bg_col != (0, 0, 0):
        noise_level = random.uniform(2, 12)
        noise = np.random.normal(0, noise_level, (size, size, 3)).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    center = (size//2 + random.randint(-40, 40), size//2 + random.randint(-40, 40))
    r1 = random.randint(int(size*0.35), int(size*0.48))
    r2 = r1 if random.random() > 0.4 else int(r1 * random.uniform(0.65, 0.95))
    angle = random.randint(0, 180)
    
    wafer_mask = np.zeros((size, size), dtype=np.uint8)
    cv2.ellipse(wafer_mask, center, (r1, r2), angle, 0, 360, 255, -1)
    
    if bg_col == (0, 0, 0):
        wafer_col = np.tile(base_col, (size, size, 1)).astype(np.uint8)
    else:
        X, Y = np.meshgrid(np.linspace(0.4, 1.6, size), np.linspace(0.4, 1.6, size))
        gradient = (X + Y) / 2
        gradient = gradient.reshape(size, size, 1)
        if random.random() > 0.5: gradient = np.flipud(gradient)
        if random.random() > 0.5: gradient = np.fliplr(gradient)
        wafer_col = np.clip(base_col * gradient, 0, 255).astype(np.uint8)
    
    img[wafer_mask > 0] = wafer_col[wafer_mask > 0]
    return img

def download_non_wafer_image(seed: int, size: int = 640) -> np.ndarray:
    url = f"https://picsum.photos/seed/{seed}/{size}/{size}"
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            arr = np.frombuffer(resp.content, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            return img
    except Exception:
        pass
    return None

def build_dataset(total_wafers: int = 10000, total_non_wafers: int = 500):
    print("🧹 Cleaning and setting up dataset directories...")
    setup_dirs()
    val_split = 0.2
    
    print(f"\n🏭 Generating {total_wafers} Wafer images... (Multithreading)")
    success_wafers = 0
    
    def generate_task(i):
        img = make_diverse_wafer()
        # Drop blurry wafers
        if is_blurry(img, threshold=40.0):
            return False
        split = "val" if random.random() < val_split else "train"
        img_name = f"wafer_{i:05d}.jpg"
        cv2.imwrite(str(DATA_DIR / "images" / split / img_name), img)
        # 0.9 0.9 bounding box to prevent YOLO 100% full-frame clipping bugs
        with open(DATA_DIR / "labels" / split / f"wafer_{i:05d}.txt", "w") as f:
            f.write("0 0.500000 0.500000 0.900000 0.900000\n")
        return True

    with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
        # Pre-assign tasks so we don't hold the main thread forever, just generate 1.2x total and take first N
        futures = [executor.submit(generate_task, i) for i in range(int(total_wafers * 1.5))]
        for i, future in enumerate(futures):
            if success_wafers >= total_wafers:
                break
            if future.result():
                success_wafers += 1
                if success_wafers % 500 == 0:
                    print(f"  👉 Generated {success_wafers}/{total_wafers} wafers...")

    print(f"\n🌍 Downloading {total_non_wafers} Non-Wafer images...")
    success_nonwafers = 0
    
    def fetch_task(i):
        img = download_non_wafer_image(i + 40000)
        if img is not None and not is_blurry(img, threshold=100.0):
            split = "val" if random.random() < val_split else "train"
            img_name = f"nonwafer_{i:05d}.jpg"
            cv2.imwrite(str(DATA_DIR / "images" / split / img_name), img)
            open(DATA_DIR / "labels" / split / f"nonwafer_{i:05d}.txt", "w").close()
            return True
        return False

    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(fetch_task, i) for i in range(total_non_wafers * 3)]
        for i, future in enumerate(futures):
            if success_nonwafers >= total_non_wafers:
                break
            if future.result():
                success_nonwafers += 1
                if success_nonwafers % 50 == 0:
                    print(f"  👉 Fetched {success_nonwafers}/{total_non_wafers} background images...")
    
    print("\n✅ Dataset generation complete!")

if __name__ == "__main__":
    if len(sys.argv) > 2:
        build_dataset(int(sys.argv[1]), int(sys.argv[2]))
    else:
        build_dataset(10000, 500)
