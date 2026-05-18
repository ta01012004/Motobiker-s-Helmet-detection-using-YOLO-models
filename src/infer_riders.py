"""Run rider-level helmet-use inference from a three-class YOLO detector.

The detector is expected to output three primitive classes:
0=person, 1=motorcycle/motobike, 2=helmet.
"""

from __future__ import annotations

import argparse
import glob
import math
import os
from pathlib import Path

import cv2
from ultralytics import YOLO


CONF_THR_PERSON = 0.30
CONF_THR_BIKE = 0.30
CONF_THR_HELMET = 0.35

MIN_FRAC_AREA_PERSON = 0.0005
MIN_FRAC_AREA_BIKE = 0.0005
MIN_FRAC_AREA_HELMET = 0.0002

IOU_RIDER_MIN = 0.05
EXPAND_X = 0.12
EXPAND_Y_BOTTOM = 0.25
BIKE_MUST_BE_LOWER = True

HEAD_FRAC_MAX = 0.60
ALLOW_HEAD_ABOVE = 0.30
HEAD_DIST_NORM_THR_FB = 0.60
PH_IOU_NORM_DENOM = 0.30
AFFINITY_THR = 0.50
FALLBACK_IOU_PH_MIN = 0.06

W_CIP = 0.35
W_IOU = 0.15
W_DIST = 0.30
W_YPROX = 0.10
W_CONF = 0.10

COLOR_RIDER_HELMET = (0, 255, 0)
COLOR_RIDER_NOHELMET = (0, 0, 255)


def iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1b, y1b, x2b, y2b = box2
    xi1, yi1 = max(x1, x1b), max(y1, y1b)
    xi2, yi2 = min(x2, x2b), min(y2, y2b)
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    area1 = max(0, x2 - x1) * max(0, y2 - y1)
    area2 = max(0, x2b - x1b) * max(0, y2b - y1b)
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def center(box):
    x1, y1, x2, y2 = box
    return (x1 + x2) * 0.5, (y1 + y2) * 0.5


def size(box):
    x1, y1, x2, y2 = box
    return max(1, x2 - x1), max(1, y2 - y1)


def point_in_expanded_person(px1, py1, px2, py2, cx, cy):
    pw, ph = size((px1, py1, px2, py2))
    ex1 = px1 - int(pw * EXPAND_X)
    ex2 = px2 + int(pw * EXPAND_X)
    ey1 = py1 - int(ph * 0.05)
    ey2 = py2 + int(ph * EXPAND_Y_BOTTOM)
    return ex1 <= cx <= ex2 and ey1 <= cy <= ey2


def is_rider(person_box, bike_box):
    person = person_box[:4]
    bike = bike_box[:4]
    if iou(person, bike) >= IOU_RIDER_MIN:
        candidate = True
    else:
        bcx, bcy = center(bike)
        candidate = point_in_expanded_person(*person, bcx, bcy)
    if not candidate:
        return False
    if BIKE_MUST_BE_LOWER:
        _, pcy = center(person)
        _, bcy = center(bike)
        if bcy < pcy - 0.10 * max(1, person[3] - person[1]):
            return False
    return True


def affinity_person_helmet(person_box, helmet_box):
    x1, y1, x2, y2, _ = person_box
    hx1, hy1, hx2, hy2, hconf = helmet_box
    _, ph = size((x1, y1, x2, y2))

    head_anchor = ((x1 + x2) * 0.5, y1)
    head_top = y1 - ALLOW_HEAD_ABOVE * ph
    head_bottom = y1 + HEAD_FRAC_MAX * ph
    hcx, hcy = center((hx1, hy1, hx2, hy2))

    cip = 1.0 if point_in_expanded_person(x1, y1, x2, y2, hcx, hcy) else 0.0
    iou_ph = iou((x1, y1, x2, y2), (hx1, hy1, hx2, hy2))
    iou_norm = min(1.0, iou_ph / PH_IOU_NORM_DENOM)
    d_norm = math.hypot(hcx - head_anchor[0], hcy - head_anchor[1]) / max(1.0, ph)
    dist_term = max(0.0, 1.0 - d_norm)

    if head_top <= hcy <= head_bottom:
        yprox = 1.0
    else:
        band = 0.15 * ph
        if hcy < head_top:
            yprox = max(0.0, 1.0 - (head_top - hcy) / max(1.0, band))
        else:
            yprox = max(0.0, 1.0 - (hcy - head_bottom) / max(1.0, band))

    score = (
        W_CIP * cip
        + W_IOU * iou_norm
        + W_DIST * dist_term
        + W_YPROX * yprox
        + W_CONF * max(0.0, min(1.0, hconf))
    )
    return float(max(0.0, min(1.0, score))), iou_ph, d_norm


def greedy_assign_helmets(persons, helmets):
    candidates = []
    for i, person in enumerate(persons):
        for j, helmet in enumerate(helmets):
            score, iou_ph, d_norm = affinity_person_helmet(person, helmet)
            if score >= AFFINITY_THR or (
                iou_ph > FALLBACK_IOU_PH_MIN and d_norm <= HEAD_DIST_NORM_THR_FB
            ):
                candidates.append((score, i, j))
    candidates.sort(reverse=True)

    used_persons, used_helmets, person_with_helmet = set(), set(), set()
    for _, person_idx, helmet_idx in candidates:
        if person_idx in used_persons or helmet_idx in used_helmets:
            continue
        person_with_helmet.add(person_idx)
        used_persons.add(person_idx)
        used_helmets.add(helmet_idx)
    return person_with_helmet


def filter_by_conf_area(boxes, conf_thr, min_frac_area, height, width):
    out = []
    min_area = min_frac_area * height * width
    for x1, y1, x2, y2, conf in boxes:
        if conf < conf_thr:
            continue
        area = max(0, x2 - x1) * max(0, y2 - y1)
        if area >= min_area:
            out.append((x1, y1, x2, y2, conf))
    return out


def draw_labels(img, rider_helmet, rider_nohelmet):
    font, font_scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
    draw = img.copy()
    for boxes, label, color in (
        (rider_helmet, "rider_helmet", COLOR_RIDER_HELMET),
        (rider_nohelmet, "rider_nohelmet", COLOR_RIDER_NOHELMET),
    ):
        for x1, y1, x2, y2, conf in boxes:
            text = f"{label} {conf:.2f}"
            cv2.rectangle(draw, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
            tx, ty = x1, max(0, y1 - 5)
            if ty - th < 0:
                ty = y1 + th + 5
            cv2.rectangle(draw, (tx, ty - th - 2), (tx + tw + 2, ty + 2), (0, 0, 0), cv2.FILLED)
            cv2.putText(draw, text, (tx, ty), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return draw


def iter_images(source):
    src = Path(source)
    if src.is_file() and src.suffix.lower() == ".txt":
        return [
            line.strip()
            for line in src.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    if src.is_dir():
        patterns = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.JPG", "*.JPEG", "*.PNG", "*.BMP"]
        paths = []
        for pattern in patterns:
            paths.extend(glob.glob(str(src / pattern)))
        return sorted(set(paths))
    return [str(src)] if src.is_file() else sorted(glob.glob(source))


def run(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.weights)
    class_map = {0: "person", 1: "motorcycle", 2: "helmet"}

    image_paths = iter_images(args.source)
    if not image_paths:
        raise FileNotFoundError(f"No images found for source: {args.source}")

    for idx, image_path in enumerate(image_paths, 1):
        img = cv2.imread(image_path)
        if img is None:
            print(f"[WARN] Cannot read {image_path}; skipped")
            continue

        height, width = img.shape[:2]
        base_conf = min(CONF_THR_PERSON, CONF_THR_BIKE, CONF_THR_HELMET)
        result = model.predict(img, imgsz=args.imgsz, conf=base_conf, device=args.device, verbose=False)[0]

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

        draw = draw_labels(img, rider_helmet, rider_nohelmet)
        out_path = out_dir / Path(image_path).name
        cv2.imwrite(str(out_path), draw)
        print(
            f"[{idx}/{len(image_paths)}] {Path(image_path).name}: "
            f"person={len(persons)}, motorcycle={len(bikes)}, helmet={len(helmets)}, "
            f"rider_helmet={len(rider_helmet)}, rider_nohelmet={len(rider_nohelmet)} -> {out_path}"
        )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, help="Path to fine-tuned YOLO .pt weights")
    parser.add_argument("--source", required=True, help="Image path, directory, or glob")
    parser.add_argument("--out-dir", default="outputs/rider_inference", help="Directory for visualized outputs")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0", help="CUDA device id or 'cpu'")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
