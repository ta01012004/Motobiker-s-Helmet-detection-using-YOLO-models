"""Train the three-class YOLOv8m detector."""

from __future__ import annotations

import argparse

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="YOLO data.yaml path")
    parser.add_argument("--weights", default="yolov8m.pt", help="Initial YOLOv8m weights")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default="runs")
    parser.add_argument("--name", default="helmet_yolov8m_person-bike-helmet")
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
        workers=8,
        project=args.project,
        name=args.name,
        hsv_h=0.015,
        hsv_s=0.50,
        hsv_v=0.20,
        degrees=2.0,
        translate=0.05,
        scale=0.20,
        shear=1.0,
        fliplr=0.5,
        flipud=0.0,
        mosaic=0.5,
        mixup=0.05,
        copy_paste=0.0,
    )


if __name__ == "__main__":
    main()
