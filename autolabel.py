# Auto-labeler: Gán nhãn YOLO (3 lớp: person, motorcycle, helmet)
# Dùng cho suy luận logic rider_helmet vs rider_no_helmet sau huấn luyện

import cv2, os
from pathlib import Path
from ultralytics import YOLO
import numpy as np
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener
register_heif_opener()

# ==== Cấu hình ====
COCO_MODEL = "E:/Deep/models/yolov8m.pt"
HELMET_MODEL = "E:/Deep/models/hemletYoloV8_100epochs.pt"
IN_DIR = "E:/Deep/Data image/JPG"
VIDEO_DIR = "E:/Deep/Data video"
OUT_IMG = "E:/Deep/data_autolabel/images"
OUT_LBL = "E:/Deep/data_autolabel/labels"
OUT_VIZ = "E:/Deep/data_autolabel/visuals"
CONF_THRES = 0.4
TARGET_FRAMES_PER_VIDEO = 30
CLASS_NAMES = ["person", "motorcycle", "helmet"]

# ==== Load models ====
print("🚀 Đang load mô hình...")
det_coco = YOLO(COCO_MODEL)
det_helm = YOLO(HELMET_MODEL)
print("✅ Mô hình đã sẵn sàng.")

Path(OUT_IMG).mkdir(parents=True, exist_ok=True)
Path(OUT_LBL).mkdir(parents=True, exist_ok=True)
Path(OUT_VIZ).mkdir(parents=True, exist_ok=True)

# ==== Chuyển ảnh JPG ====
print("🖼️ Đang xử lý ảnh JPG...")
for img in Path(IN_DIR).rglob("*.jpg"):
    try:
        im = Image.open(img)
        im = ImageOps.exif_transpose(im).convert("RGB")
        im.save(Path(OUT_IMG)/img.name)
    except Exception as e:
        print(f"Lỗi ảnh: {img} - {e}")
print("✅ Hoàn tất xử lý ảnh JPG.")

# ==== Trích frame video ====
print("🎞️ Đang trích frame từ video...")
for vid in Path(VIDEO_DIR).glob("*"):
    if vid.suffix.lower() not in ['.mp4', '.mov', '.avi']:
        continue
    print(f"▶️  Video: {vid.name}")
    cap = cv2.VideoCapture(str(vid))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total_frames // TARGET_FRAMES_PER_VIDEO)
    i = 0
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if i % step == 0:
            name = f"{vid.stem}_{i:05d}.jpg"
            cv2.imwrite(str(Path(OUT_IMG)/name), frame)
            count += 1
        i += 1
    cap.release()
    print(f"    → Đã trích {count} khung hình")
print("✅ Hoàn tất trích video.")

# ==== Gán nhãn + visualize ====
imgs = list(Path(OUT_IMG).glob("*.jpg"))
print(f"📌 Tổng số ảnh cần gán nhãn: {len(imgs)}")

for idx, img_path in enumerate(imgs):
    print(f"🧠 [{idx+1}/{len(imgs)}] Gán nhãn: {img_path.name}")
    im = cv2.imread(str(img_path))
    H, W = im.shape[:2]
    out = []

    coco = det_coco.predict(source=im, conf=CONF_THRES, verbose=False)[0]
    boxes = coco.boxes.xyxy.cpu().numpy()
    clss = coco.boxes.cls.cpu().numpy().astype(int)
    persons = [boxes[i] for i,c in enumerate(clss) if c==0]
    motos   = [boxes[i] for i,c in enumerate(clss) if c==3]

    for p in persons:
        x1,y1,x2,y2 = p
        x = (x1+x2)/2/W; y=(y1+y2)/2/H; w=(x2-x1)/W; h=(y2-y1)/H
        out.append((0,x,y,w,h))  # person

    for m in motos:
        x1,y1,x2,y2 = m
        x = (x1+x2)/2/W; y=(y1+y2)/2/H; w=(x2-x1)/W; h=(y2-y1)/H
        out.append((1,x,y,w,h))  # motorcycle

    # detect helmet riêng
    helmets = det_helm.predict(source=im, conf=CONF_THRES, verbose=False)[0]
    for hbox in helmets.boxes.xyxy.cpu().numpy():
        x1,y1,x2,y2 = hbox
        x = (x1+x2)/2/W; y=(y1+y2)/2/H; w=(x2-x1)/W; h=(y2-y1)/H
        out.append((2,x,y,w,h))  # helmet

    with open(Path(OUT_LBL)/(img_path.stem + ".txt"),"w") as f:
        for cls,x,y,w,h in out:
            f.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

    # visualize
    for cls,x,y,w,h in out:
        x1 = int((x - w/2)*W); y1 = int((y - h/2)*H)
        x2 = int((x + w/2)*W); y2 = int((y + h/2)*H)
        color = (0,255,0) if cls==0 else (255,165,0) if cls==1 else (0,200,255)
        label = CLASS_NAMES[cls]
        cv2.rectangle(im, (x1,y1), (x2,y2), color, 2)
        cv2.putText(im, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imwrite(str(Path(OUT_VIZ)/(img_path.stem + "_viz.jpg")), im)

print("\n✅ Đã gán nhãn và visualize xong - YOLO 3 lớp: person, motorcycle, helmet")