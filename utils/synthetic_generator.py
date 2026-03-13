"""
Synthetic Wafer Generator (Segmentation Mode)
Generates (Image, Mask) pairs for U-Net training.
Updated to include Grid Textures for real-world robustness.
"""
import numpy as np
import cv2
import random

def add_grid_texture(img):
    """Adds a grid-like texture to the wafer background"""
    h, w = img.shape
    
    # 1. Grid Lines (Horizontal & Vertical)
    step = random.randint(8, 15) # Grid spacing
    color = random.randint(80, 120) # Slightly lighter/darker than background
    
    # Horizontal
    for y in range(0, h, step):
        cv2.line(img, (0, y), (w, y), color, 1)
        
    # Vertical
    for x in range(0, w, step):
        cv2.line(img, (x, 0), (x, h), color, 1)
        
    # 2. Random specialized texture (Moiré / Crosshatch)
    if random.random() > 0.5:
        # Fixed: Generate noise properly for addition
        noise = np.random.randint(0, 20, (h, w), dtype=np.uint8)
        # Ensure img is compatible
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        img = cv2.addWeighted(img, 0.9, noise, 0.1, 0)
        
    return img

def generate_wafer_mask_pair(defect_type, size=(128, 128)):
    h, w = size
    
    # 1. Base Image & Mask
    # Create a grid-textured background instead of flat gray
    img = np.full((h, w), 100, dtype=np.uint8)
    img = add_grid_texture(img)
    
    mask = np.zeros((h, w), dtype=np.float32) # 0=Background, 1=Defect
    
    # Circular Crop for Wafer Shape
    center = (w // 2, h // 2)
    radius = min(h, w) // 2 - 10
    
    # Mask out everything outside the wafer circle
    wafer_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(wafer_mask, center, radius, 255, -1)
    img = cv2.bitwise_and(img, img, mask=wafer_mask)
    
    # Defect Generation (Draw on BOTH img and mask)
    # Draw defects BRIGHTER than the grid so they stand out
    defect_color = 255
    
    if defect_type == 'Center':
        cv2.circle(img, center, radius // 4, defect_color, -1)
        cv2.circle(mask, center, radius // 4, 1.0, -1)
        
    elif defect_type == 'Donut':
        cv2.circle(img, center, radius // 2, defect_color, 15)
        cv2.circle(mask, center, radius // 2, 1.0, 15)
        
    elif defect_type == 'Edge-Loc':
        angle = random.uniform(0, 2*np.pi)
        r = radius - 20
        x = int(center[0] + r * np.cos(angle))
        y = int(center[1] + r * np.sin(angle))
        cv2.circle(img, (x, y), 30, defect_color, -1)
        cv2.circle(mask, (x, y), 30, 1.0, -1)
        
    elif defect_type == 'Edge-Ring':
        cv2.circle(img, center, radius - 10, defect_color, 10)
        cv2.circle(mask, center, radius - 10, 1.0, 10)
        
    elif defect_type == 'Loc':
        x = random.randint(center[0]-radius//2, center[0]+radius//2)
        y = random.randint(center[1]-radius//2, center[1]+radius//2)
        cv2.circle(img, (x, y), 25, defect_color, -1)
        cv2.circle(mask, (x, y), 25, 1.0, -1)
        
    elif defect_type == 'Random':
        for _ in range(50):
            rx, ry = random.randint(0, w-1), random.randint(0, h-1)
            if (rx-center[0])**2 + (ry-center[1])**2 < radius**2:
                cv2.circle(img, (rx, ry), 3, defect_color, -1)
                cv2.circle(mask, (rx, ry), 3, 1.0, -1)
                
    elif defect_type == 'Scratch':
        p1 = (random.randint(20, w-20), random.randint(20, h-20))
        p2 = (random.randint(20, w-20), random.randint(20, h-20))
        cv2.line(img, p1, p2, defect_color, 3)
        cv2.line(mask, p1, p2, 1.0, 3)
        
    elif defect_type == 'Near-full':
        cv2.circle(img, center, radius - 10, defect_color, -1)
        cv2.circle(mask, center, radius - 10, 1.0, -1)

    # --- Augmentations (Apply ONLY to Image) ---
    scale = random.uniform(0.8, 1.1)
    img_aug = np.clip(img.astype(float) * scale, 0, 255).astype(np.uint8)
    
    if random.random() > 0.5:
        img_aug = cv2.GaussianBlur(img_aug, (3, 3), 0)
        
    noise = np.random.normal(0, 5, (h, w)).astype(np.float32) # Increased noise
    img_aug = np.clip(img_aug.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    
    # Convert Image to RGB (MobileNet/U-Net expects 3 channels)
    img_rgb = cv2.cvtColor(img_aug, cv2.COLOR_GRAY2RGB)
    
    # Mask is single channel (H, W, 1)
    mask = np.expand_dims(mask, axis=-1)
    
    return img_rgb, mask

def create_segmentation_dataset(n_samples=2000):
    classes = ['none', 'Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc', 'Random', 'Scratch', 'Near-full']
    x_data, y_data = [], []
    
    for _ in range(n_samples):
        # 80% chance of defect, 20% none (better balance for segmentation training)
        if random.random() > 0.2:
            cls = random.choice(classes[1:]) # Skip 'none'
        else:
            cls = 'none'
            
        img, mask = generate_wafer_mask_pair(cls)
        x_data.append(img)
        y_data.append(mask)
        
    return np.array(x_data, dtype=np.float32), np.array(y_data, dtype=np.float32)
