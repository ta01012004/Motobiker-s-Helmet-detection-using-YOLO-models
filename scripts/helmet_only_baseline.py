"""Helmet-only heuristic baseline.

This baseline uses only helmet detections from a trained three-class YOLO model.
It does not associate persons, motorcycles, and helmets, so the output is a
simple image-level heuristic rather than a true rider-level decision.
"""

from __future__ import annotations

import argparse
import csv
import glob
from pathlib import Path

import cv2
from ultralytics import YOLO


def iter_images(source: str):
    src = Path(source)
    if src.is_dir():
        patterns = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.JPG", "*.JPEG", "*.PNG", "*.BMP"]
        paths = []
        for pattern in patterns:
            paths.extend(glob.glob(str(src / pattern)))
        return sorted(set(paths))
    if src.is_file():
        return [str(src)]
    return sorted(glob.glob(source))


def draw_helmet_boxes(image, helmets):
    draw = image.copy()
    for x1, y1, x2, y2, conf in helmets:
        label = f"helmet {conf:.2f}"
        cv2.rectangle(draw, (x1, y1), (x2, y2), (0, 200, 255), 2)
        cv2.putText(
            draw,
            label,
            (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 200, 255),
            2,
            cv2.LINE_AA,
        )
    return draw


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, help="Path to trained YOLO .pt weights")
    parser.add_argument("--source", required=True, help="Image path, image folder, or glob")
    parser.add_argument("--out-dir", default="outputs/helmet_only_baseline")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--device", default="0")
    parser.add_argument("--helmet-class", type=int, default=2, help="Class id for helmet")
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)
    rows = []

    image_paths = iter_images(args.source)
    if not image_paths:
        raise FileNotFoundError(f"No images found for source: {args.source}")

    for index, image_path in enumerate(image_paths, 1):
        image = cv2.imread(image_path)
        if image is None:
            print(f"[WARN] Cannot read {image_path}; skipped")
            continue

        result = model.predict(
            image,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            verbose=False,
        )[0]

        helmets = []
        if result.boxes is not None:
            for box in result.boxes:
                if int(box.cls) != args.helmet_class:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                helmets.append((x1, y1, x2, y2, float(box.conf)))

        label = "helmet_detected" if helmets else "no_helmet_detected"
        out_image = out_dir / Path(image_path).name
        cv2.imwrite(str(out_image), draw_helmet_boxes(image, helmets))

        rows.append(
            {
                "image": image_path,
                "num_helmets": len(helmets),
                "label": label,
                "output_image": str(out_image),
            }
        )
        print(f"[{index}/{len(image_paths)}] {Path(image_path).name}: {label} ({len(helmets)} helmets)")

    csv_path = out_dir / "helmet_only_baseline.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "num_helmets", "label", "output_image"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved summary: {csv_path}")


if __name__ == "__main__":
    main()
