"""
Semiconductor Wafer Presence Detector & Camera Gating Engine.

Verifies that an image or camera frame contains a circular semiconductor wafer
before running defect classification.

Rejects:
  - Faces, people, rooms, desks, hands, background clutter
  - Blank frames, random noise
  - Rectangular objects, non-circular shapes
  - Human skin tones (faces, hands, fingers)

Accepts:
  - Real WM-811K fab wafer maps
  - Camera-captured circular wafers / reticles
  - Synthetic wafer maps
"""
import cv2
import numpy as np


def is_wafer_image(image: np.ndarray) -> bool:
    """
    High-precision circular semiconductor wafer detector.
    Uses multi-stage contour geometry, morphological opening (to prevent
    thin defect scratches from distorting circular bounds), minimum enclosing
    circle fill, convex hull solidity, aspect ratio, and skin-tone gating.

    Rejects:
      - Faces, people, rooms, desks, hands, random objects
      - Blank frames, uniform noise
      - Rectangular objects, non-circular shapes

    Accepts:
      - Real WM-811K fab wafer maps
      - Scratches and linear defect patterns cutting across wafer boundary
      - Camera-captured circular wafers / reticles
      - Synthetic wafer maps
    """
    if image is None or image.size == 0:
        return False

    h, w = image.shape[:2]
    if h < 32 or w < 32:
        return False

    # 1. Skin-tone rejection (rejects faces, hands, skin holding objects)
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        cr = ycrcb[:, :, 1]
        cb = ycrcb[:, :, 2]
        skin_mask = (cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127)
        if float(np.mean(skin_mask)) > 0.28:
            return False
    else:
        gray = image.copy()

    # 2. Reject uniform / blank / low-contrast frames
    std_val = float(np.std(gray))
    if std_val < 6.0:
        return False

    total_area = h * w
    blurred = cv2.GaussianBlur(gray, (7, 7), 1.5)

    # 3. Multi-threshold search (both dark background and bright background)
    for invert in [False, True]:
        thresholds = [15, 30, 45, 60, 80, 110] if not invert else [140, 180, 210]
        mode = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
        for thresh_val in thresholds:
            _, thresh = cv2.threshold(blurred, thresh_val, 255, mode)

            # Evaluate base threshold and morphologically opened versions.
            # Morphological opening detaches thin scratch lines that reach or slightly
            # extend past the circular wafer edge, allowing the true circular disc to be evaluated.
            for ksize in [1, 9, 13]:
                if ksize > 1:
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
                    m = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
                else:
                    m = thresh

                cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not cnts:
                    continue

                c = max(cnts, key=cv2.contourArea)
                area = cv2.contourArea(c)
                if area < 0.10 * total_area or area > 0.98 * total_area:
                    continue

                hull = cv2.convexHull(c)
                hull_area = cv2.contourArea(hull)
                if hull_area <= 0:
                    continue

                solidity = area / hull_area
                peri = cv2.arcLength(c, True)
                if peri <= 0:
                    continue

                circ = 4 * np.pi * (area / (peri * peri))
                (cx, cy), radius = cv2.minEnclosingCircle(c)
                if radius <= 0:
                    continue

                fill = area / (np.pi * radius * radius)
                x, y, bw, bh = cv2.boundingRect(c)
                aspect = min(bw, bh) / max(bw, bh) if max(bw, bh) > 0 else 0

                # Genuine semiconductor wafer geometry:
                # - Circular aspect ratio >= 0.80
                # - Circularity >= 0.50
                # - Convex hull solidity >= 0.88
                # - Minimum enclosing circle fill >= 0.76
                if fill >= 0.76 and solidity >= 0.88 and aspect >= 0.80 and circ >= 0.50:
                    return True

    return False
