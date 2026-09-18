"""
YOLO Bounding Box Labeling Tool for Wafer Detection

Usage:
    python tools/label_wafer.py --image-dir dataset/images/train --label-dir dataset/labels/train

Controls:
    - Click and drag to draw bounding box
    - Press 's' to save label and move to next image
    - Press 'r' to reset current bounding box
    - Press 'q' to quit
    - Press 'n' to skip image (no label)
"""

import argparse
import cv2
import os
import glob
from pathlib import Path

# --- Global State ---
drawing = False
ix, iy = -1, -1
fx, fy = -1, -1
bbox_drawn = False
current_image = None
clone = None

WAFER_CLASS_ID = 0


def mouse_callback(event, x, y, flags, param):
    global drawing, ix, iy, fx, fy, bbox_drawn, current_image, clone

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        bbox_drawn = False

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            current_image = clone.copy()
            cv2.rectangle(current_image, (ix, iy), (x, y), (0, 255, 0), 2)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        fx, fy = x, y
        bbox_drawn = True
        current_image = clone.copy()
        cv2.rectangle(current_image, (ix, iy), (fx, fy), (0, 255, 0), 2)

        # Show label preview
        h, w = current_image.shape[:2]
        x_center = ((ix + fx) / 2) / w
        y_center = ((iy + fy) / 2) / h
        bw = abs(fx - ix) / w
        bh = abs(fy - iy) / h
        label_text = f"{WAFER_CLASS_ID} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}"
        cv2.putText(current_image, label_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)


def save_yolo_label(label_path, img_shape):
    """Save bounding box in YOLO format: class x_center y_center width height"""
    h, w = img_shape[:2]

    x_center = ((ix + fx) / 2) / w
    y_center = ((iy + fy) / 2) / h
    bw = abs(fx - ix) / w
    bh = abs(fy - iy) / h

    with open(label_path, 'w') as f:
        f.write(f"{WAFER_CLASS_ID} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}\n")

    print(f"  Saved: {label_path}")


def main():
    global current_image, clone, bbox_drawn, ix, iy, fx, fy

    parser = argparse.ArgumentParser(description="YOLO Bounding Box Labeler for Wafer Images")
    parser.add_argument("--image-dir", type=str, default="dataset/images/train",
                        help="Directory containing images to label")
    parser.add_argument("--label-dir", type=str, default="dataset/labels/train",
                        help="Directory to save YOLO labels")
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    label_dir = Path(args.label_dir)
    label_dir.mkdir(parents=True, exist_ok=True)

    # Find all images
    extensions = ["*.png", "*.jpg", "*.jpeg", "*.bmp"]
    image_files = []
    for ext in extensions:
        image_files.extend(sorted(image_dir.glob(ext)))

    if not image_files:
        print(f"No images found in {image_dir}")
        return

    print(f"Found {len(image_files)} images in {image_dir}")
    print("Controls: drag=draw box | s=save | r=reset | n=skip | q=quit")
    print("-" * 60)

    cv2.namedWindow("YOLO Wafer Labeler")
    cv2.setMouseCallback("YOLO Wafer Labeler", mouse_callback)

    for idx, img_path in enumerate(image_files):
        label_path = label_dir / (img_path.stem + ".txt")

        # Skip already labeled images
        if label_path.exists():
            print(f"[{idx+1}/{len(image_files)}] SKIP (already labeled): {img_path.name}")
            continue

        print(f"[{idx+1}/{len(image_files)}] Labeling: {img_path.name}")

        clone = cv2.imread(str(img_path))
        if clone is None:
            print(f"  Could not read image, skipping.")
            continue

        current_image = clone.copy()
        bbox_drawn = False

        while True:
            cv2.imshow("YOLO Wafer Labeler", current_image)
            key = cv2.waitKey(20) & 0xFF

            if key == ord('s'):
                if bbox_drawn:
                    save_yolo_label(label_path, clone.shape)
                    break
                else:
                    print("  No bounding box drawn yet. Draw a box first.")

            elif key == ord('r'):
                current_image = clone.copy()
                bbox_drawn = False
                print("  Reset bounding box.")

            elif key == ord('n'):
                # Create empty label file (negative sample)
                with open(label_path, 'w') as f:
                    pass
                print(f"  Skipped (empty label): {label_path}")
                break

            elif key == ord('q'):
                print("Exiting labeler.")
                cv2.destroyAllWindows()
                return

    cv2.destroyAllWindows()
    print("Done! All images processed.")


if __name__ == "__main__":
    main()
