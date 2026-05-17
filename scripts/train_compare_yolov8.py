"""Train a YOLOv8 comparison model and evaluate its test split.

This runner is intended for fair YOLOv8n/YOLOv8s comparisons against the
selected YOLOv8m experiment. It mirrors the selected run configuration and then
evaluates the trained best checkpoint on `split=test`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, help="Initial YOLO weights, e.g. models/yolov8n.pt")
    parser.add_argument("--data", default="data.yaml")
    parser.add_argument("--name", required=True, help="Run name under runs_compare/")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.weights)
    model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        optimizer="SGD",
        lr0=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        device=args.device,
        workers=args.workers,
        project="runs_compare",
        name=args.name,
        exist_ok=True,
        plots=True,
        close_mosaic=15,
        hsv_h=0.015,
        hsv_s=0.50,
        hsv_v=0.20,
        degrees=2.0,
        translate=0.05,
        scale=0.20,
        shear=1.0,
        perspective=0.0,
        fliplr=0.5,
        flipud=0.0,
        mosaic=0.5,
        mixup=0.05,
        copy_paste=0.0,
        erasing=0.4,
    )

    best = Path("runs_compare") / args.name / "weights" / "best.pt"
    if not best.is_file():
        raise FileNotFoundError(f"Expected checkpoint not found: {best}")

    YOLO(str(best)).val(
        data=args.data,
        split="test",
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project="runs_compare_test",
        name=f"{args.name}_test",
        exist_ok=True,
        plots=True,
    )


if __name__ == "__main__":
    main()
