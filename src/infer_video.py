"""Run rider-level helmet-use inference on a video file."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

from infer_riders import (
    CONF_THR_BIKE,
    CONF_THR_HELMET,
    CONF_THR_PERSON,
    MIN_FRAC_AREA_BIKE,
    MIN_FRAC_AREA_HELMET,
    MIN_FRAC_AREA_PERSON,
    draw_labels,
    filter_by_conf_area,
    greedy_assign_helmets,
    is_rider,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, help="Path to fine-tuned YOLO .pt weights")
    parser.add_argument("--source", required=True, help="Input video path")
    parser.add_argument("--out", default=None, help="Output MP4 path")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0", help="CUDA device id or 'cpu'")
    parser.add_argument("--max-frames", type=int, default=0, help="Optional frame limit for quick tests")
    return parser.parse_args()


def run(args):
    source = Path(args.source)
    if not source.is_file():
        raise FileNotFoundError(f"Video not found: {source}")

    out_path = Path(args.out) if args.out else Path("outputs/video") / f"{source.stem}_rider_helmet.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)
    class_map = {0: "person", 1: "motorcycle", 2: "helmet"}

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    frame_idx = 0
    base_conf = min(CONF_THR_PERSON, CONF_THR_BIKE, CONF_THR_HELMET)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if args.max_frames and frame_idx > args.max_frames:
            break

        result = model.predict(frame, imgsz=args.imgsz, conf=base_conf, device=args.device, verbose=False)[0]
        persons, bikes, helmets = [], [], []
        if result.boxes is not None:
            for box in result.boxes:
                cls_id = int(box.cls)
                conf = float(box.conf)
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                name = class_map.get(cls_id)
                if name == "person":
                    persons.append((x1, y1, x2, y2, conf))
                elif name == "motorcycle":
                    bikes.append((x1, y1, x2, y2, conf))
                elif name == "helmet":
                    helmets.append((x1, y1, x2, y2, conf))

        persons = filter_by_conf_area(persons, CONF_THR_PERSON, MIN_FRAC_AREA_PERSON, height, width)
        bikes = filter_by_conf_area(bikes, CONF_THR_BIKE, MIN_FRAC_AREA_BIKE, height, width)
        helmets = filter_by_conf_area(helmets, CONF_THR_HELMET, MIN_FRAC_AREA_HELMET, height, width)

        rider_idxs = []
        for person_idx, person_box in enumerate(persons):
            if any(is_rider(person_box, bike_box) for bike_box in bikes):
                rider_idxs.append(person_idx)

        rider_persons = [persons[i] for i in sorted(set(rider_idxs))]
        person_with_helmet = greedy_assign_helmets(rider_persons, helmets)
        rider_helmet, rider_nohelmet = [], []
        for local_idx, person_box in enumerate(rider_persons):
            if local_idx in person_with_helmet:
                rider_helmet.append(person_box)
            else:
                rider_nohelmet.append(person_box)

        writer.write(draw_labels(frame, rider_helmet, rider_nohelmet))
        if frame_idx % 30 == 0:
            print(f"Processed {frame_idx} frames -> {out_path}")

    cap.release()
    writer.release()
    print(f"Saved video: {out_path}")


if __name__ == "__main__":
    run(parse_args())
