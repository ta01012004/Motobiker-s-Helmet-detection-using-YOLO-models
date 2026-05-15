# -*- coding: utf-8 -*-
import os, cv2, math
import numpy as np
from ultralytics import YOLO

# ================== PATHS ==================
BEST_PT    = "/home/22011107/TA/Deep/Deep/runs/helmet_yolov8m/weights/best.pt"
INPUT_VID  = "/home/22011107/TA/Deep/Deep/Data video/IMG_8106.MOV"
OUT_DIR    = "/home/22011107/TA/Deep/Deep/runs_test/viz_video"
OUT_VIDEO  = None  # nếu None sẽ tự tạo theo tên video đầu vào
IMGSZ      = 640
DEVICE     = 0

os.makedirs(OUT_DIR, exist_ok=True)
if OUT_VIDEO is None:
    base = os.path.splitext(os.path.basename(INPUT_VID))[0]
    OUT_VIDEO = os.path.join(OUT_DIR, f"{base}_rider_helmet.mp4")

# ================== NGƯỠNG & THAM SỐ ==================
# Lọc conf theo lớp
CONF_THR_PERSON  = 0.30
CONF_THR_BIKE    = 0.30
CONF_THR_HELMET  = 0.35

# Loại bbox quá nhỏ (tỉ lệ diện tích ảnh)
MIN_FRAC_AREA_PERSON = 0.0005
MIN_FRAC_AREA_BIKE   = 0.0005
MIN_FRAC_AREA_HELMET = 0.0002

# ---- RIDER: person <-> bike (khoan dung để không bỏ người) ----
IOU_RIDER_MIN      = 0.05
EXPAND_X           = 0.12
EXPAND_Y_BOTTOM    = 0.25
BIKE_MUST_BE_LOWER = True

# ---- HELMET: affinity score ----
HEAD_FRAC_MAX        = 0.60
ALLOW_HEAD_ABOVE     = 0.30
HEAD_DIST_NORM_THR_FB= 0.60
HELMET_MIN_AREA_FRAC = 0.004
HELMET_MAX_AREA_FRAC = 0.22
HELMET_ASPECT_MIN    = 0.40
HELMET_ASPECT_MAX    = 2.50
PH_IOU_NORM_DENOM    = 0.30
AFFINITY_THR         = 0.50
FALLBACK_IOU_PH_MIN  = 0.06

# Trọng số affinity
W_CIP   = 0.35
W_IOU   = 0.15
W_DIST  = 0.30
W_YPROX = 0.10
W_CONF  = 0.10

# ================== MÀU VẼ ==================
COLOR_RIDER_HELMET   = (0, 255, 0)
COLOR_RIDER_NOHELMET = (0, 0, 255)

# ================== HÀM TIỆN ÍCH ==================
def iou(box1, box2):
    x1, y1, x2, y2 = box1; X1, Y1, X2, Y2 = box2
    xi1, yi1 = max(x1, X1), max(y1, Y1)
    xi2, yi2 = min(x2, X2), min(y2, Y2)
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    a1 = max(0, x2 - x1) * max(0, y2 - y1)
    a2 = max(0, X2 - X1) * max(0, Y2 - Y1)
    u = a1 + a2 - inter
    return inter / u if u > 0 else 0.0

def center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)

def size(box):
    x1, y1, x2, y2 = box
    return max(1, x2 - x1), max(1, y2 - y1)

def point_in_expanded_person(px1, py1, px2, py2, cx, cy):
    pw, ph = size((px1, py1, px2, py2))
    ex1 = px1 - int(pw * EXPAND_X)
    ex2 = px2 + int(pw * EXPAND_X)
    ey1 = py1 - int(ph * 0.05)
    ey2 = py2 + int(ph * EXPAND_Y_BOTTOM)
    return (ex1 <= cx <= ex2) and (ey1 <= cy <= ey2)

def is_rider(person_box, bike_box):
    pb = person_box[:4]; bb = bike_box[:4]
    if iou(pb, bb) >= IOU_RIDER_MIN:
        candidate = True
    else:
        bcx, bcy = center(bb)
        candidate = point_in_expanded_person(*pb, bcx, bcy)
    if not candidate:
        return False
    if BIKE_MUST_BE_LOWER:
        _, pcy = center(pb)
        _, bcy = center(bb)
        if bcy < pcy - 0.10 * max(1, pb[3] - pb[1]):
            return False
    return True

def affinity_person_helmet(person_box, helmet_box):
    x1, y1, x2, y2, _ = person_box
    hx1, hy1, hx2, hy2, hconf = helmet_box

    pw, ph = size((x1, y1, x2, y2))
    head_anchor = ((x1 + x2) * 0.5, y1)
    head_top    = y1 - ALLOW_HEAD_ABOVE * ph
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

    conf_term = max(0.0, min(1.0, hconf))
    score = (W_CIP*cip + W_IOU*iou_norm + W_DIST*dist_term + W_YPROX*yprox + W_CONF*conf_term)
    return float(max(0.0, min(1.0, score))), iou_ph, d_norm

def greedy_assign_helmets(persons, helmets):
    candidates = []
    for i, p in enumerate(persons):
        for j, h in enumerate(helmets):
            score, iou_ph, d_norm = affinity_person_helmet(p, h)
            if score >= AFFINITY_THR or (iou_ph > FALLBACK_IOU_PH_MIN and d_norm <= HEAD_DIST_NORM_THR_FB):
                candidates.append((score, i, j))
    candidates.sort(reverse=True)
    used_p, used_h = set(), set()
    person_with_helmet = set()
    for s, i, j in candidates:
        if i in used_p or j in used_h: continue
        person_with_helmet.add(i)
        used_p.add(i); used_h.add(j)
    return person_with_helmet

def filter_by_conf_area(boxes, conf_thr, min_frac_area, H, W):
    out = []
    min_area = min_frac_area * (H * W)
    for (x1, y1, x2, y2, conf) in boxes:
        if conf < conf_thr: 
            continue
        area = max(0, x2 - x1) * max(0, y2 - y1)
        if area < min_area:
            continue
        out.append((x1, y1, x2, y2, conf))
    return out

def draw_labels(img, rider_helmet, rider_nohelmet):
    font, fs, th = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
    draw = img.copy()
    for (x1, y1, x2, y2, conf) in rider_helmet:
        label = f"rider_helmet {conf:.2f}"
        cv2.rectangle(draw, (x1, y1), (x2, y2), COLOR_RIDER_HELMET, 2)
        (tw, th_text), _ = cv2.getTextSize(label, font, fs, th)
        tx, ty = x1, max(0, y1 - 5)
        if ty - th_text < 0: ty = y1 + th_text + 5
        cv2.rectangle(draw, (tx, ty - th_text - 2), (tx + tw + 2, ty + 2), (0,0,0), cv2.FILLED)
        cv2.putText(draw, label, (tx, ty), font, fs, (255,255,255), th, cv2.LINE_AA)
    for (x1, y1, x2, y2, conf) in rider_nohelmet:
        label = f"rider_nohelmet {conf:.2f}"
        cv2.rectangle(draw, (x1, y1), (x2, y2), COLOR_RIDER_NOHELMET, 2)
        (tw, th_text), _ = cv2.getTextSize(label, font, fs, th)
        tx, ty = x1, max(0, y1 - 5)
        if ty - th_text < 0: ty = y1 + th_text + 5
        cv2.rectangle(draw, (tx, ty - th_text - 2), (tx + tw + 2, ty + 2), (0,0,0), cv2.FILLED)
        cv2.putText(draw, label, (tx, ty), font, fs, (255,255,255), th, cv2.LINE_AA)
    return draw

# ================== LOAD MODEL ==================
model = YOLO(BEST_PT)
print("=== KIỂM TRA NHÃN TRONG MÔ HÌNH ===")
if hasattr(model, 'model') and hasattr(model.model, 'names'):
    for cid, cname in model.model.names.items():
        print(f"  {cid}: {cname}")
else:
    print("Không thể lấy tên lớp từ mô hình")
mapping = {0: "person", 1: "motobike", 2: "helmet"}

# ================== VIDEO I/O ==================
cap = cv2.VideoCapture(INPUT_VID)
if not cap.isOpened():
    raise RuntimeError(f"Không mở được video: {INPUT_VID}")

fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(OUT_VIDEO, fourcc, fps, (w, h))
print(f"Bắt đầu xử lý video → {OUT_VIDEO} | {w}x{h}@{fps:.2f}fps")

frame_idx = 0
log_every = 30  # log mỗi 30 frame

while True:
    ret, frame = cap.read()
    if not ret:
        break
    H, W = frame.shape[:2]

    base_conf = min(CONF_THR_PERSON, CONF_THR_BIKE, CONF_THR_HELMET)
    res = model.predict(frame, imgsz=IMGSZ, conf=base_conf, device=DEVICE, verbose=False)[0]

    persons, bikes, helmets = [], [], []
    if res.boxes is not None:
        for b in res.boxes:
            cls = int(b.cls); conf = float(b.conf)
            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
            name = mapping.get(cls)
            if name == "person":
                persons.append((x1, y1, x2, y2, conf))
            elif name == "motobike":
                bikes.append((x1, y1, x2, y2, conf))
            elif name == "helmet":
                helmets.append((x1, y1, x2, y2, conf))

    persons = filter_by_conf_area(persons, CONF_THR_PERSON, MIN_FRAC_AREA_PERSON, H, W)
    bikes   = filter_by_conf_area(bikes,   CONF_THR_BIKE,   MIN_FRAC_AREA_BIKE,   H, W)
    helmets = filter_by_conf_area(helmets, CONF_THR_HELMET, MIN_FRAC_AREA_HELMET, H, W)

    # Chọn các person là rider (ít ràng buộc để không bỏ người)
    rider_idxs = []
    for i, pb in enumerate(persons):
        for bb in bikes:
            if is_rider(pb, bb):
                rider_idxs.append(i); break
    rider_idxs = sorted(set(rider_idxs))

    # Gán mũ (greedy trên affinity)
    rider_persons = [persons[i] for i in rider_idxs]
    person_with_helmet_local = greedy_assign_helmets(rider_persons, helmets)

    rider_helmet, rider_nohelmet = [], []
    for local_i, pb in enumerate(rider_persons):
        if local_i in person_with_helmet_local:
            rider_helmet.append(pb)
        else:
            rider_nohelmet.append(pb)

    drawn = draw_labels(frame, rider_helmet, rider_nohelmet)
    writer.write(drawn)

    frame_idx += 1
    if frame_idx % log_every == 0:
        print(f"  • Frame {frame_idx}: rider_helmet={len(rider_helmet)} | rider_nohelmet={len(rider_nohelmet)}")

cap.release()
writer.release()
print(f"Hoàn tất! Video đã lưu tại: {OUT_VIDEO}")
