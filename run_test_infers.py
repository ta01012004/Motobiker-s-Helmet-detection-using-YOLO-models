# -*- coding: utf-8 -*-
import os, glob, cv2
import numpy as np
from ultralytics import YOLO

BEST_PT = "/home/22011107/TA/Deep/Deep/runs/helmet_yolov8m/weights/best.pt"
# Chọn vài ảnh để test (đổi lại đường dẫn/glob tuỳ ý)
TEST_GLOB = "/home/22011107/TA/Deep/Deep/dataloc/images/*.jpg"
OUT_DIR   = "/home/22011107/TA/Deep/Deep/runs_test/viz3_py"

os.makedirs(OUT_DIR, exist_ok=True)

# Load model
model = YOLO(BEST_PT)

# Màu theo lớp (0=rider,1=helmet,2=person)
COLORS = {
    0: (0,   0, 255),  # rider -> đỏ
    1: (0, 255,   0),  # helmet -> xanh lá
    2: (255, 0,   0),  # person -> xanh dương
}

# Lấy tên lớp từ model (nếu file names có sẵn)
names = model.model.names if hasattr(model.model, "names") else {0:"person",1:"motobike",2:"helmet"}
# Đảm bảo thứ tự 3 lớp
names = {0:"person", 1:"motobike", 2:"helmet"}

# Lấy danh sách file ảnh
img_paths = sorted(glob.glob(TEST_GLOB))
if len(img_paths) == 0:
    raise FileNotFoundError(f"Không tìm thấy ảnh với glob: {TEST_GLOB}")

for p in img_paths:
    img = cv2.imread(p)
    if img is None:
        print(f"[WARN] Không đọc được ảnh: {p}, bỏ qua.")
        continue

    # Inference 1 ảnh
    res = model.predict(img, imgsz=640, conf=0.25, device=0, verbose=False)[0]

    # Gom kết quả theo lớp để in gọn
    per_class = {0:[], 1:[], 2:[]}
    if res.boxes is not None:
        for box in res.boxes:
            cls_id = int(box.cls)
            conf   = float(box.conf)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            per_class.setdefault(cls_id, []).append((x1,y1,x2,y2,conf))

    # In ra console tóm tắt
    print(f"\n==> {os.path.basename(p)}")
    for cid in [0,1,2]:
        items = per_class.get(cid, [])
        print(f"  {names[cid]}: {len(items)}")
        # in tối đa 5 box đầu
        for (x1,y1,x2,y2,conf) in items[:5]:
            print(f"     - box=({x1},{y1},{x2},{y2}) conf={conf:.2f}")

    # Vẽ bbox và nhãn
    draw = img.copy()
    font, fs, th = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
    for cid, items in per_class.items():
        color = COLORS.get(cid, (255,255,255))
        label_name = names.get(cid, str(cid))
        for (x1,y1,x2,y2,conf) in items:
            cv2.rectangle(draw, (x1,y1), (x2,y2), color, 2)
            text = f"{label_name} {conf:.2f}"
            (tw, th_text), base = cv2.getTextSize(text, font, fs, th)
            tx, ty = x1, max(0, y1-5)
            if ty - th_text < 0:  # nếu đụng mép trên
                ty = y1 + th_text + 5
            cv2.rectangle(draw, (tx, ty - th_text - 2), (tx + tw + 2, ty + 2), (0,0,0), cv2.FILLED)
            cv2.putText(draw, text, (tx, ty), font, fs, (255,255,255), th, cv2.LINE_AA)

    # Lưu ảnh đã vẽ
    out_path = os.path.join(OUT_DIR, os.path.basename(p))
    cv2.imwrite(out_path, draw)
    print(f"  -> Saved: {out_path}")
