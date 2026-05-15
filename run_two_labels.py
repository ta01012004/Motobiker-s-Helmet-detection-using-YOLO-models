# -*- coding: utf-8 -*-
import os, glob, cv2, math, random
import numpy as np
from ultralytics import YOLO
# ================== PATHS ==================
BEST_PT   = "/home/22011107/TA/Deep/Deep/runs/helmet_yolov8m/weights/best.pt"
TEST_GLOB = "/home/22011107/TA/Deep/Deep/dataloc/images/*.jpg"
OUT_DIR   = "/home/22011107/TA/Deep/Deep/runs_test/viz2_py"
IMGSZ     = 640
DEVICE    = 0 

os.makedirs(OUT_DIR, exist_ok=True)

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
IOU_RIDER_MIN    = 0.05     # IoU rất thấp cũng chấp nhận
EXPAND_X         = 0.12     # nở ngang person ±12% khi kiểm tra tâm xe
EXPAND_Y_BOTTOM  = 0.25     # nở xuống dưới +25%
BIKE_MUST_BE_LOWER = True   # tâm xe nên thấp hơn tâm người (hợp cảnh)

# ---- HELMET: affinity score (không dùng IoU toàn thân cứng) ----
# CHỈNH: nới nhẹ để giảm miss mũ sáng/nhỏ
HEAD_FRAC_MAX        = 0.60   # vùng đầu tới 60% chiều cao (tăng từ 0.55)
ALLOW_HEAD_ABOVE     = 0.30   # cho phép mũ cao hơn đỉnh tối đa 30% chiều cao (tăng từ 0.25)
HEAD_DIST_NORM_THR_FB= 0.60   # fallback: khoảng cách chuẩn hoá tối đa (giữ nguyên)
HELMET_MIN_AREA_FRAC = 0.004  # 0.4% diện tích person (giữ)
HELMET_MAX_AREA_FRAC = 0.22   # 22% diện tích person (giữ)
HELMET_ASPECT_MIN    = 0.40   # w/h (giữ)
HELMET_ASPECT_MAX    = 2.50   # w/h (giữ)
PH_IOU_NORM_DENOM    = 0.30   # chuẩn hoá IoU (IoU/0.30 -> [0..1] cắt max 1)
AFFINITY_THR         = 0.50   # CHỈNH: hạ ngưỡng affinity từ 0.55 -> 0.50 để bớt bỏ sót
FALLBACK_IOU_PH_MIN  = 0.06   # fallback: IoU nhỏ giữa helmet và person

# Trọng số affinity (tổng ~1.0)
W_CIP   = 0.35   # center-in-person (nở)
W_IOU   = 0.15   # IoU_norm (helmet-person)
W_DIST  = 0.30   # 1 - khoảng cách tới top-center (chuẩn hoá theo ph)
W_YPROX = 0.10   # proximity theo trục Y trong dải [head_top, head_bottom]
W_CONF  = 0.10   # trọng số theo conf của helmet

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
    """Kiểm tra điểm (cx,cy) nằm trong bbox person nở rộng."""
    pw, ph = size((px1, py1, px2, py2))
    ex1 = px1 - int(pw * EXPAND_X)
    ex2 = px2 + int(pw * EXPAND_X)
    ey1 = py1 - int(ph * 0.05)                 # nới nhẹ phía trên
    ey2 = py2 + int(ph * EXPAND_Y_BOTTOM)      # nới mạnh phía dưới
    return (ex1 <= cx <= ex2) and (ey1 <= cy <= ey2)

def is_rider(person_box, bike_box):
    """Nhận rider nếu: IoU thấp hoặc tâm xe nằm trong person (nở); xe nên thấp hơn người."""
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
        if bcy < pcy - 0.10 * max(1, pb[3]-pb[1]):  # cho phép gần ngang
            return False
    return True

def affinity_person_helmet(person_box, helmet_box):
    """
    Điểm affinity ∈ [0..1] cho cặp (person, helmet):
      - CIP: tâm mũ nằm trong person (nở) -> 0/1
      - IoU_norm: min(1, IoU / PH_IOU_NORM_DENOM)
      - DIST: 1 - ( khoảng cách tới top-center / ph )
      - YPROX: 1 nếu tâm mũ nằm trong dải [head_top, head_bottom], 0 nếu lệch xa
      - CONF: conf_helmet ∈ [0..1]
    """
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
    dist_term = max(0.0, 1.0 - d_norm)  # càng gần top-center càng tốt

    # proximity theo trục Y
    if head_top <= hcy <= head_bottom:
        yprox = 1.0
    else:
        # giảm tuyến tính trong 0.15*ph ra ngoài dải
        band = 0.15 * ph
        if hcy < head_top:
            yprox = max(0.0, 1.0 - (head_top - hcy) / max(1.0, band))
        else:
            yprox = max(0.0, 1.0 - (hcy - head_bottom) / max(1.0, band))

    conf_term = max(0.0, min(1.0, hconf))

    score = (W_CIP*cip + W_IOU*iou_norm + W_DIST*dist_term + W_YPROX*yprox + W_CONF*conf_term)
    return float(max(0.0, min(1.0, score))), iou_ph, d_norm

def greedy_assign_helmets(persons, helmets):
    """
    Trả về set index person có mũ theo greedy assignment trên affinity.
    Mỗi helmet chỉ gán cho 1 person; mỗi person tối đa 1 helmet.
    """
    candidates = []
    for i, p in enumerate(persons):
        for j, h in enumerate(helmets):
            score, iou_ph, d_norm = affinity_person_helmet(p, h)
            if score >= AFFINITY_THR or (iou_ph > FALLBACK_IOU_PH_MIN and d_norm <= HEAD_DIST_NORM_THR_FB):
                candidates.append((score, i, j))
    candidates.sort(reverse=True)  # điểm cao trước

    used_p, used_h = set(), set()
    person_with_helmet = set()
    for s, i, j in candidates:
        if i in used_p or j in used_h:
            continue
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

# ánh xạ kỳ vọng
mapping = {0: "person", 1: "motobike", 2: "helmet"}

# ================== TEST TOÀN BỘ ẢNH (CHỈNH: bỏ random 10) ==================
img_paths = sorted(glob.glob(TEST_GLOB))
if not img_paths:
    raise FileNotFoundError(f"Không tìm thấy ảnh: {TEST_GLOB}")
print(f"\n=== BẮT ĐẦU XỬ LÝ {len(img_paths)} ẢNH ===")

# ================== LOOP ==================
for p in img_paths:
    img = cv2.imread(p)
    if img is None:
        print(f"[WARN] Không đọc được ảnh: {p}, bỏ qua.")
        continue
    H, W = img.shape[:2]

    base_conf = min(CONF_THR_PERSON, CONF_THR_BIKE, CONF_THR_HELMET)
    res = model.predict(img, imgsz=IMGSZ, conf=base_conf, device=DEVICE, verbose=False)[0]

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

    # Lọc theo conf + kích thước
    persons = filter_by_conf_area(persons, CONF_THR_PERSON, MIN_FRAC_AREA_PERSON, H, W)
    bikes   = filter_by_conf_area(bikes,   CONF_THR_BIKE,   MIN_FRAC_AREA_BIKE,   H, W)
    helmets = filter_by_conf_area(helmets, CONF_THR_HELMET, MIN_FRAC_AREA_HELMET, H, W)

    # === CHỌN RIDER (cho phép nhiều rider/xe nếu hợp lý) ===
    rider_idxs = []
    for i, pb in enumerate(persons):
        for bb in bikes:
            if is_rider(pb, bb):
                rider_idxs.append(i); break
    rider_idxs = sorted(set(rider_idxs))

    # === GÁN HELMET (greedy trên affinity) ===
    rider_persons = [persons[i] for i in rider_idxs]
    person_with_helmet_local = greedy_assign_helmets(rider_persons, helmets)

    rider_helmet, rider_nohelmet = [], []
    for local_i, pb in enumerate(rider_persons):
        if local_i in person_with_helmet_local:
            rider_helmet.append(pb)
        else:
            rider_nohelmet.append(pb)

    print(f"\n==> {os.path.basename(p)}")
    print(f"  Person: {len(persons)} | Motobike: {len(bikes)} | Helmet: {len(helmets)}")
    print(f"  → rider_helmet: {len(rider_helmet)} | rider_nohelmet: {len(rider_nohelmet)}")

    draw = draw_labels(img, rider_helmet, rider_nohelmet)
    out_path = os.path.join(OUT_DIR, os.path.basename(p))
    cv2.imwrite(out_path, draw)
    print(f"  -> Đã lưu: {out_path}")

print(f"\n=== HOÀN THÀNH ===")
print(f"Ảnh kết quả lưu tại: {OUT_DIR}")
