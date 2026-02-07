"""
OpenCV-based Wafer Image Preprocessing Pipeline
"""

import cv2
import numpy as np
from typing import Tuple

class WaferImagePreprocessor:
    """Complete preprocessing pipeline for wafer defect images."""
    
    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        normalize: bool = True,
        denoise: bool = True,
        enhance_contrast: bool = True
    ):
        self.target_size = target_size
        self.normalize = normalize
        self.denoise = denoise
        self.enhance_contrast = enhance_contrast
        self.imagenet_mean = np.array([0.485, 0.456, 0.406])
        self.imagenet_std = np.array([0.229, 0.224, 0.225])
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        image = self.resize(image)
        if self.denoise:
            image = self.reduce_noise(image)
        if self.enhance_contrast:
            image = self.enhance_contrast_clahe(image)
        if self.normalize:
            image = self.normalize_image(image)
        return image
    
    def resize(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        target_h, target_w = self.target_size
        interp = cv2.INTER_AREA if h > target_h or w > target_w else cv2.INTER_CUBIC
        return cv2.resize(image, (target_w, target_h), interpolation=interp)
    
    def reduce_noise(self, image: np.ndarray) -> np.ndarray:
        return cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
    
    def enhance_contrast_clahe(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l_enhanced = clahe.apply(l)
            lab_enhanced = cv2.merge([l_enhanced, a, b])
            return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        else:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            return clahe.apply(image)
    
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        image = image.astype(np.float32) / 255.0
        return image # Simple normalization for demo
