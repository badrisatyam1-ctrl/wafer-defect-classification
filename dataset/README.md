# YOLO Wafer Detection Dataset

## Folder Structure
```
dataset/
 ├── dataset.yaml          # YOLO config file
 ├── images/
 │    ├── train/            # Training images (80%)
 │    ├── val/              # Validation images (20%)
 ├── labels/
 │    ├── train/            # Training labels (YOLO format)
 │    ├── val/              # Validation labels (YOLO format)
```

## Where to Put Images

### Wafer Images (Positive Samples)
Place full-frame images **containing wafers** into `images/train/` and `images/val/`.
Each image must have a corresponding `.txt` label file in `labels/train/` or `labels/val/`.

### Non-Wafer Images (Negative Samples / Background)
Place images **without wafers** into the same folders.
For negative samples, create an **empty** `.txt` label file (0 bytes) with the same filename.

## Label Format (YOLO)
Each `.txt` file contains one line per object:
```
class x_center y_center width height
```
All values are **normalized** (0.0 to 1.0) relative to image dimensions.

**Class IDs:**
- `0` = wafer

### Example
For an image `wafer_001.png` (640x480) with a wafer bounding box at pixel coords (100, 50, 500, 430):
```
# labels/train/wafer_001.txt
0 0.46875 0.5 0.625 0.791667
```

Calculation:
- x_center = (100 + 500) / 2 / 640 = 0.46875
- y_center = (50 + 430) / 2 / 480 = 0.5
- width = (500 - 100) / 640 = 0.625
- height = (430 - 50) / 480 = 0.791667

## Labeling Tool
Use `tools/label_wafer.py` to interactively draw bounding boxes and auto-save YOLO labels.

## Training
```bash
yolo detect train data=dataset/dataset.yaml model=yolov8n.pt epochs=50 imgsz=640
```
