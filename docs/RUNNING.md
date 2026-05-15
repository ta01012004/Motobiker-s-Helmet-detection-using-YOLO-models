# Running Guide

This guide describes the recommended way to run the public version of the project. The original root-level scripts are kept for traceability, but the `src/` entrypoints are cleaner and do not require editing hard-coded local paths.

## 1. Create the Environment

```bash
cd /path/to/Motobiker-s-Helmet-detection-using-YOLO-models
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python scripts/check_environment.py
```

If your CUDA/PyTorch versions need a specific wheel, install PyTorch from the official selector first, then run `pip install -r requirements.txt`.

## 2. Prepare Data

Prepare YOLO-format folders like this:

```text
dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

Each label file should contain:

```text
class_id x_center y_center width height
```

with normalized coordinates and class ids:

```text
0 person
1 motorcycle
2 helmet
```

Copy `configs/data.yaml` and replace `<DATASET_ROOT>` with your prepared dataset path.

## 3. Train the Detector

```bash
python src/train_yolov8m.py \
  --data configs/data.yaml \
  --weights yolov8m.pt \
  --epochs 120 \
  --batch 16 \
  --imgsz 640 \
  --device 0 \
  --project runs \
  --name helmet_yolov8m_person-bike-helmet
```

Use `--device cpu` only for debugging; full training should use a GPU.

## 4. Run Image or Folder Inference

```bash
python src/infer_riders.py \
  --weights runs/helmet_yolov8m_person-bike-helmet/weights/best.pt \
  --source path/to/image_or_folder \
  --out-dir outputs/rider_inference \
  --device 0
```

The output images contain only the final rider-level labels:

```text
rider_helmet
rider_nohelmet
```

## 5. Run Video Inference

```bash
python src/infer_video.py \
  --weights runs/helmet_yolov8m_person-bike-helmet/weights/best.pt \
  --source path/to/video.mp4 \
  --out outputs/video/video_rider_helmet.mp4 \
  --device 0
```

For a quick smoke test on the first 100 frames:

```bash
python src/infer_video.py \
  --weights path/to/best.pt \
  --source path/to/video.mp4 \
  --max-frames 100 \
  --device 0
```

## 6. Reproduce the Paper Figures

The selected paper run is `runs/helmet_yolov8m_person-bike-helmet3`. The lightweight assets copied into this repository are:

```text
assets/results/selected_run_curves.png
assets/results/selected_run_confusion_matrix_normalized.png
assets/results/selected_run_validation_predictions.jpg
assets/results/rider_level_demo.jpg
```

The most appropriate single image for Fig. 4 is:

```text
assets/results/selected_run_curves.png
```

## 7. Common Issues

- If `ultralytics` downloads weights automatically, ensure the machine has internet access or place `yolov8m.pt` locally and pass it with `--weights`.
- If CUDA is unavailable, use `--device cpu` for inference or install a CUDA-compatible PyTorch build.
- If class names look wrong, verify that your training YAML uses exactly `person`, `motorcycle`, `helmet` in that order.
- Do not commit datasets, videos, `.pt` checkpoints, or `runs/` outputs to normal Git. Use GitHub Releases, cloud storage, DVC, or Git LFS for large artifacts.
