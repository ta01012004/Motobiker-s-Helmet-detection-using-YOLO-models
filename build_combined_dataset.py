# build_combined_dataset.py
# Gộp 2 bộ dữ liệu: helmetdata (VOC XML) + data_autolabel (YOLO)
# - Chuyển VOC XML -> YOLO TXT (giữ class: helmet)
# - Pseudo-label person/motorcycle cho ảnh helmetdata bằng yolov8m (COCO)
# - Tách train/val và tạo YAML

import os, random, shutil, xml.etree.ElementTree as ET
from pathlib import Path
from ultralytics import YOLO

# ====== Đường dẫn nguồn ======
HELMET_IMG_DIR = Path(r"E:/Deep/helmetdata/images")
HELMET_XML_DIR = Path(r"E:/Deep/helmetdata/annotations")  # VOC XML

AUTO_IMG_DIR   = Path(r"E:/Deep/data_autolabel/images")
AUTO_LBL_DIR   = Path(r"E:/Deep/data_autolabel/labels")   # YOLO 3 lớp sẵn

# ====== Đường dẫn đích ======
OUT_ROOT   = Path(r"E:/Deep/dataset_combined")
IM_TRAIN   = OUT_ROOT/"images/train"
IM_VAL     = OUT_ROOT/"images/val"
LB_TRAIN   = OUT_ROOT/"labels/train"
LB_VAL     = OUT_ROOT/"labels/val"
for p in [IM_TRAIN, IM_VAL, LB_TRAIN, LB_VAL]:
    p.mkdir(parents=True, exist_ok=True)

# ====== Model dùng pseudo-label ======
# COCO: 0=person, 3=motorcycle
COCO_MODEL = r"E:/Deep/models/yolov8m.pt"
det_coco = YOLO(COCO_MODEL)
CONF_PERSON = 0.40
CONF_MOTO   = 0.40

# ====== Class mapping YOLO đích ======
# 0: person, 1: motorcycle, 2: helmet
DEST_CLASSES = ["person", "motorcycle", "helmet"]

def voc_to_yolo_box(bbox, W, H):
    xmin, ymin, xmax, ymax = map(float, bbox)
    x = ((xmin + xmax) / 2) / W
    y = ((ymin + ymax) / 2) / H
    w = (xmax - xmin) / W
    h = (ymax - ymin) / H
    return x, y, w, h

def read_voc_objects(xml_path):
    """Trả về list [(name, (xmin,ymin,xmax,ymax)), ...]"""
    objs = []
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find("size")
    W = float(size.find("width").text)
    H = float(size.find("height").text)
    for obj in root.findall("object"):
        name = obj.find("name").text.strip().lower()
        bb = obj.find("bndbox")
        xmin = float(bb.find("xmin").text)
        ymin = float(bb.find("ymin").text)
        xmax = float(bb.find("xmax").text)
        ymax = float(bb.find("ymax").text)
        objs.append((name, (xmin, ymin, xmax, ymax), W, H))
    return objs

def write_yolo_label(lbl_path, rows):
    lbl_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lbl_path, "w", encoding="utf-8") as f:
        for (cls, x, y, w, h) in rows:
            f.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

def iou(boxA, boxB):
    # box: (xmin, ymin, xmax, ymax)
    ax1, ay1, ax2, ay2 = boxA
    bx1, by1, bx2, by2 = boxB
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = a + b - inter
    return inter / union if union > 0 else 0.0

# ====== 1) Chuẩn hoá bộ HELMET (VOC) thành YOLO 3 lớp ======
helmet_items = []  # (img_path, label_rows)
print("🛠  Đang chuyển VOC XML -> YOLO cho helmetdata ...")
for img_path in HELMET_IMG_DIR.glob("*"):
    if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
        continue
    xml_path = HELMET_XML_DIR / f"{img_path.stem}.xml"
    if not xml_path.exists():
        # nếu thiếu nhãn, vẫn dùng pseudo-label sau để có person/moto/helmet dự đoán
        objs = []
        W = H = None
    else:
        objs_raw = read_voc_objects(xml_path)  # list(name, bbox, W, H)
        if len(objs_raw) > 0:
            W = objs_raw[0][2]; H = objs_raw[0][3]
        objs = []
        for name, (xmin, ymin, xmax, ymax), W, H in objs_raw:
            n = name.replace("-", "_").lower()
            if n in ["helmet", "with_helmet", "helmets"]:
                x, y, w, h = voc_to_yolo_box((xmin, ymin, xmax, ymax), W, H)
                objs.append((2, x, y, w, h))  # class 2: helmet
            # bỏ qua no-helmet vì ta suy luận logic sau

    # ===== pseudo-label person & motorcycle bằng COCO model =====
    imp = str(img_path)
    pred = det_coco.predict(source=imp, conf=0.25, verbose=False)[0]
    boxes = pred.boxes.xyxy.cpu().numpy()
    clss  = pred.boxes.cls.cpu().numpy().astype(int)
    Wp = pred.orig_shape[1]; Hp = pred.orig_shape[0]

    # thu thập bbox thô (để lọc trùng với helmet GT)
    gt_helm_xyxy = []
    for row in objs:
        if row[0] == 2:
            # chuyển từ yolo -> xyxy để so iou
            x,y,w,h = row[1:]
            xmin = (x - w/2) * W
            ymin = (y - h/2) * H
            xmax = (x + w/2) * W
            ymax = (y + h/2) * H
            gt_helm_xyxy.append((xmin,ymin,xmax,ymax))

    # add person & motorcycle
    for b, c in zip(boxes, clss):
        if c == 0 and pred.boxes.conf[0] >= CONF_PERSON:  # person
            x1,y1,x2,y2 = b
            x = (x1 + x2)/2 / Wp; y = (y1 + y2)/2 / Hp
            w = (x2 - x1)/Wp;    h = (y2 - y1)/Hp
            objs.append((0, x, y, w, h))
        if c == 3 and pred.boxes.conf[0] >= CONF_MOTO:   # motorcycle
            x1,y1,x2,y2 = b
            x = (x1 + x2)/2 / Wp; y = (y1 + y2)/2 / Hp
            w = (x2 - x1)/Wp;    h = (y2 - y1)/Hp
            objs.append((1, x, y, w, h))

    # thêm vào danh sách
    helmet_items.append((img_path, objs))

print(f"✅ Đã xử lý {len(helmet_items)} ảnh từ helmetdata.")

# ====== 2) Gom bộ data_autolabel (đã là YOLO 3 lớp) ======
auto_items = []  # (img_path, lbl_path)
print("📦 Gom ảnh & nhãn từ data_autolabel ...")
for img_path in AUTO_IMG_DIR.glob("*"):
    if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
        continue
    lbl = AUTO_LBL_DIR / f"{img_path.stem}.txt"
    if lbl.exists():
        auto_items.append((img_path, lbl))
print(f"✅ Tìm thấy {len(auto_items)} ảnh có nhãn từ data_autolabel.")

# ====== 3) Chia train/val & sao chép về dataset_combined ======
random.seed(123)
pack = []

# từ helmetdata (nhãn được build ở bước 1)
for img_path, rows in helmet_items:
    pack.append(("helmet", img_path, rows))

# từ data_autolabel (nhãn đã có sẵn file)
for img_path, lbl in auto_items:
    pack.append(("auto", img_path, lbl))

random.shuffle(pack)
VAL_RATIO = 0.2
val_count = int(len(pack) * VAL_RATIO)
val_set = set(range(val_count))

def copy_and_write(idx, kind, img_path, meta):
    # meta: rows(list) nếu kind="helmet", hoặc lbl_path nếu kind="auto"
    dest_im = (IM_VAL if idx in val_set else IM_TRAIN) / img_path.name
    shutil.copy2(img_path, dest_im)

    if kind == "helmet":
        # ghi file nhãn YOLO từ rows
        dest_lb = (LB_VAL if idx in val_set else LB_TRAIN) / (img_path.stem + ".txt")
        write_yolo_label(dest_lb, meta)
    else:
        # copy file nhãn có sẵn
        src_lb = meta
        dest_lb = (LB_VAL if idx in val_set else LB_TRAIN) / src_lb.name
        shutil.copy2(src_lb, dest_lb)

for i, (kind, img_path, meta) in enumerate(pack):
    copy_and_write(i, kind, img_path, meta)

print("🎯 Hoàn tất gộp & tách train/val vào:", OUT_ROOT)

# ====== 4) Tạo YAML ======
yaml_path = Path(r"E:/Deep/dataset_combined.yaml")
yaml_text = f"""# dataset_combined.yaml
train: {IM_TRAIN.as_posix()}
val: {IM_VAL.as_posix()}
nc: 3
names: ["person","motorcycle","helmet"]
"""
yaml_path.write_text(yaml_text, encoding="utf-8")
print("📝 Đã tạo YAML:", yaml_path)
