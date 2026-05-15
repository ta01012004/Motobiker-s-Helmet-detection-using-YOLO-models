# -*- coding: utf-8 -*-
"""
Train YOLOv8-m trên 3 lớp {0: person, 1: motobike, 2: helmet} + suy luận realtime.
- Nguồn dữ liệu:
    + Pascal VOC XML ở /helmetdata  -> convert sang YOLO (map: rider->person, helmet->helmet, person->person)
    + YOLO (TXT) ở /dataloc        -> đã đúng thứ tự nhãn {0 person, 1 motobike, 2 helmet}
- Huấn luyện: YOLOv8-m (Ultralytics) + SGD, imgsz=640, batch=16, epochs=150, augmentation "nhẹ tay".
- Xuất .pt (ưu tiên best.pt; fallback last.pt).
- Hậu xử lý realtime:
    + Ghép person + motobike gần nhau => rider
    + Helmet nằm trong vùng đầu (top 33% bbox person) => rider_helmet, ngược lại rider_nohelmet
    + Chỉ vẽ 2 nhãn rider_helmet / rider_nohelmet (không vẽ person/motobike riêng).
- Cài đặt: pip install ultralytics opencv-python (hoặc opencv-python-headless).
"""

import os
import glob
import random
import cv2
from ultralytics import YOLO
import xml.etree.ElementTree as ET

# -------------------- CẤU HÌNH ĐƯỜNG DẪN --------------------
# VOC
helmetdata_images_dir = "/home/22011107/TA/Deep/Deep/helmetdata/images"
helmetdata_ann_dir    = "/home/22011107/TA/Deep/Deep/helmetdata/annotations"
# YOLO
dataloc_images_dir    = "/home/22011107/TA/Deep/Deep/dataloc/images"
dataloc_labels_dir    = "/home/22011107/TA/Deep/Deep/dataloc/labels"

# PRETRAIN (đường dẫn tuyệt đối để tránh tải mạng)
yolov8m_pt_path = "/home/22011107/TA/Deep/Deep/models/yolov8m.pt"

# THƯ MỤC LÀM VIỆC
base_dir = "/home/22011107/TA/Deep/Deep"

# -------------------- KIỂM TRA TỒN TẠI --------------------
if not os.path.isdir(helmetdata_images_dir) or not os.path.isdir(helmetdata_ann_dir):
    raise FileNotFoundError("Thiếu thư mục helmetdata (images/annotations).")
if not os.path.isdir(dataloc_images_dir) or not os.path.isdir(dataloc_labels_dir):
    raise FileNotFoundError("Thiếu thư mục dataloc (images/labels).")
if not os.path.isfile(yolov8m_pt_path):
    raise FileNotFoundError(f"Không tìm thấy trọng số pretrain: {yolov8m_pt_path}")

# -------------------- B1: CONVERT VOC -> YOLO (HELMETDATA) --------------------
print("B1) Chuyển đổi nhãn VOC XML -> YOLO cho /helmetdata ...")
helmetdata_labels_dir = os.path.join(os.path.dirname(helmetdata_images_dir), "labels")
os.makedirs(helmetdata_labels_dir, exist_ok=True)

# Map về bộ nhãn {0: person, 1: motobike, 2: helmet}
# Lưu ý: dữ liệu helmetdata có 'rider' => map sang 'person' (0); không thấy 'motobike' trong VOC này.
voc_to_yolo_map = {
    "rider": 0,     # coi rider là person
    "person": 0,
    "helmet": 2,
    # các biến thể no-helmet bỏ qua
    "no-helmet": None, "no_helmet": None, "nohelmet": None
}

helmet_images_list = []
for xml_file in glob.glob(os.path.join(helmetdata_ann_dir, "*.xml")):
    try:
        tree = ET.parse(xml_file)
    except Exception as e:
        print(f"  [Cảnh báo] Không parse được {xml_file}: {e}")
        continue

    root = tree.getroot()
    filename_tag = root.find('filename')
    if filename_tag is None or not filename_tag.text:
        print(f"  [Cảnh báo] XML thiếu <filename>: {xml_file} -> bỏ qua")
        continue
    filename = filename_tag.text

    # tìm ảnh tương ứng
    img_path = os.path.join(helmetdata_images_dir, filename)
    if not os.path.exists(img_path):
        name_no_ext = os.path.splitext(filename)[0]
        found = False
        for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
            alt_path = os.path.join(helmetdata_images_dir, name_no_ext + ext)
            if os.path.exists(alt_path):
                img_path = alt_path
                found = True
                break
        if not found:
            print(f"  [Cảnh báo] Không tìm thấy ảnh cho {xml_file} -> bỏ qua")
            continue

    # lấy kích thước ảnh
    size_tag = root.find('size')
    if (size_tag is not None and
        size_tag.find('width') is not None and
        size_tag.find('height') is not None):
        img_w = int(size_tag.find('width').text)
        img_h = int(size_tag.find('height').text)
    else:
        img = cv2.imread(img_path)
        if img is None:
            print(f"  [Cảnh báo] Không đọc được ảnh {img_path} -> bỏ qua")
            continue
        img_h, img_w = img.shape[:2]

    # ghi file YOLO label
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    yolo_label_path = os.path.join(helmetdata_labels_dir, base_name + ".txt")
    with open(yolo_label_path, 'w', encoding='utf-8') as f_txt:
        for obj in root.findall('object'):
            name_tag = obj.find('name')
            bndbox = obj.find('bndbox')
            if name_tag is None or bndbox is None:
                continue
            name = (name_tag.text or "").strip()
            if name not in voc_to_yolo_map:
                continue
            class_id = voc_to_yolo_map[name]
            if class_id is None:
                continue  # bỏ qua no-helmet

            xmin = float(bndbox.find('xmin').text)
            ymin = float(bndbox.find('ymin').text)
            xmax = float(bndbox.find('xmax').text)
            ymax = float(bndbox.find('ymax').text)
            # clip
            xmin = max(0.0, min(xmin, img_w - 1))
            ymin = max(0.0, min(ymin, img_h - 1))
            xmax = max(0.0, min(xmax, img_w - 1))
            ymax = max(0.0, min(ymax, img_h - 1))
            if xmax <= xmin or ymax <= ymin:
                continue

            cx = (xmin + xmax) / 2.0 / img_w
            cy = (ymin + ymax) / 2.0 / img_h
            bw = (xmax - xmin) / img_w
            bh = (ymax - ymin) / img_h
            f_txt.write(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
    helmet_images_list.append(img_path)

print(f"   -> Đã convert {len(helmet_images_list)} ảnh từ /helmetdata.")

# -------------------- B2: THU THẬP DATALOC (YOLO) --------------------
print("B2) Thu thập ảnh & nhãn từ /dataloc (YOLO {0 person,1 motobike,2 helmet}) ...")
dataloc_images_list = []
for img_file in glob.glob(os.path.join(dataloc_images_dir, "*.*")):
    ext = os.path.splitext(img_file)[1].lower()
    if ext not in [".jpg", ".jpeg", ".png", ".bmp"]:
        continue
    base_name = os.path.splitext(os.path.basename(img_file))[0]
    lbl_file = os.path.join(dataloc_labels_dir, base_name + ".txt")
    if not os.path.isfile(lbl_file):
        # tạo nhãn trống để YOLO vẫn load được
        open(lbl_file, 'w').close()
        print(f"  [Chú ý] Ảnh {img_file} không có nhãn, tạo file nhãn trống.")
    dataloc_images_list.append(img_file)

print(f"   -> Thu thập {len(dataloc_images_list)} ảnh từ /dataloc.")

# -------------------- B3: GỘP & CHIA TRAIN/VAL/TEST --------------------
all_images = helmet_images_list + dataloc_images_list
total_images = len(all_images)
if total_images == 0:
    raise RuntimeError("Không tìm thấy ảnh nào để huấn luyện.")

print("B3) Chia train/val/test (80/10/10) ...")
random.seed(42)
random.shuffle(all_images)

train_count = int(0.8 * total_images)
val_count   = int(0.1 * total_images)
test_count  = total_images - train_count - val_count

train_images = all_images[:train_count]
val_images   = all_images[train_count:train_count + val_count]
test_images  = all_images[train_count + val_count:]

print(f"   -> Tổng: {total_images} | Train: {len(train_images)} | Val: {len(val_images)} | Test: {len(test_images)}")

# -------------------- B4: GHI LIST ẢNH --------------------
print("B4) Ghi train.txt / val.txt / test.txt ...")
train_list_path = os.path.join(base_dir, "train.txt")
val_list_path   = os.path.join(base_dir, "val.txt")
test_list_path  = os.path.join(base_dir, "test.txt")

with open(train_list_path, 'w', encoding='utf-8') as f:
    for p in train_images:
        f.write(p + "\n")
with open(val_list_path, 'w', encoding='utf-8') as f:
    for p in val_images:
        f.write(p + "\n")
with open(test_list_path, 'w', encoding='utf-8') as f:
    for p in test_images:
        f.write(p + "\n")

print(f"   -> Đã tạo: {train_list_path}, {val_list_path}, {test_list_path}")

# -------------------- B5: TẠO data.yaml (đúng thứ tự nhãn) --------------------
print("B5) Tạo data.yaml với names: {0: person, 1: motobike, 2: helmet} ...")
data_yaml_path = os.path.join(base_dir, "data.yaml")
with open(data_yaml_path, 'w', encoding='utf-8') as f:
    f.write(f"path: {base_dir}\n")
    f.write(f"train: {train_list_path}\n")
    f.write(f"val: {val_list_path}\n")
    f.write(f"test: {test_list_path}\n")
    f.write("names:\n")
    f.write("  0: person\n")
    f.write("  1: motobike\n")
    f.write("  2: helmet\n")
print(f"   -> Đã tạo: {data_yaml_path}")

# -------------------- B6: TRAIN YOLOv8-m (SGD + AUG "NHẸ TAY", EPOCHS=150) --------------------
print("B6) Bắt đầu huấn luyện YOLOv8-m (SGD + augmentation nhẹ) ...")
model = YOLO(yolov8m_pt_path)

results = model.train(
    data=data_yaml_path,
    epochs=120,
    batch=16,
    imgsz=640,
    optimizer="SGD",
    lr0=0.01,
    momentum=0.937,
    weight_decay=0.0005,

    device=0,
    workers=8,
    project="runs",
    name="helmet_yolov8m_person-bike-helmet",  # thư mục sẽ tự tăng số nếu trùng

    # ---- Augmentation "nhẹ tay" phù hợp vật thể nhỏ (helmet) ----
    hsv_h=0.015,
    hsv_s=0.50,
    hsv_v=0.20,

    degrees=2.0,
    translate=0.05,
    scale=0.20,
    shear=1.0,
    perspective=0.0,

    fliplr=0.5,
    flipud=0.0,     # không thực tế -> tắt
    mosaic=0.50,    # giảm cường độ
    mixup=0.05,

    copy_paste=0.0,
    close_mosaic=15,  # tắt mosaic ở 15 epoch cuối để “khóa” hình thật
)
print("   -> Huấn luyện hoàn tất!")

# -------------------- B7: CHỌN TRỌNG SỐ (best.pt ưu tiên, fallback last.pt) --------------------
print("B7) Tìm trọng số để suy luận ...")
weights_dir = os.path.join("runs", "helmet_yolov8m_person-bike-helmet", "weights")
best_weights_path = os.path.join(weights_dir, "best.pt")
last_weights_path = os.path.join(weights_dir, "last.pt")

def find_any_best_last():
    found_best, found_last = None, None
    for root, _, files in os.walk("runs"):
        if "weights" in root:
            if "best.pt" in files:
                found_best = os.path.join(root, "best.pt")
            if "last.pt" in files:
                found_last = os.path.join(root, "last.pt")
    return found_best, found_last

if not os.path.isfile(best_weights_path):
    fb_best, fb_last = find_any_best_last()
    if fb_best:
        best_weights_path = fb_best
    elif os.path.isfile(last_weights_path):
        best_weights_path = last_weights_path
    elif fb_last:
        best_weights_path = fb_last
    else:
        raise FileNotFoundError("Không tìm thấy best.pt hoặc last.pt trong runs/.")

print(f"   -> Dùng trọng số: {best_weights_path}")

# -------------------- B8: SUY LUẬN REALTIME (chỉ vẽ rider_helmet / rider_nohelmet) --------------------
print("B8) Tải mô hình và chuẩn bị suy luận realtime ...")
trained_model = YOLO(best_weights_path)

# 0 = webcam; có thể thay bằng đường dẫn video
source = 0

def iou(boxA, boxB):
    ax1, ay1, ax2, ay2 = boxA
    bx1, by1, bx2, by2 = boxB
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    areaA = max(0, (ax2 - ax1)) * max(0, (ay2 - ay1))
    areaB = max(0, (bx2 - bx1)) * max(0, (by2 - by1))
    union = areaA + areaB - inter + 1e-9
    return inter / union

# Nếu máy không có GUI (HPC), bỏ qua realtime.
if os.environ.get('DISPLAY', '') == '':
    print("Không có DISPLAY. Bỏ qua suy luận realtime. Hãy chạy trên máy có màn hình (VD: laptop).")
else:
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Không thể mở nguồn video/webcam {source}.")
    else:
        print("Bắt đầu realtime... Nhấn 'q' để dừng.")
        color_rider_nohelmet = (0, 0, 255)   # đỏ
        color_rider_helmet  = (0, 255, 0)    # xanh lá
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Hạ conf để tăng Recall cho helmet
            results = trained_model.predict(frame, imgsz=640, conf=0.15, verbose=False)
            result = results[0]
            boxes = result.boxes

            person_boxes, bike_boxes, helmet_boxes = [], [], []
            for box in boxes:
                cls_id = int(box.cls)
                conf = float(box.conf)
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                if cls_id == 0:      # person
                    person_boxes.append((x1, y1, x2, y2, conf))
                elif cls_id == 1:    # motobike
                    bike_boxes.append((x1, y1, x2, y2, conf))
                elif cls_id == 2:    # helmet
                    helmet_boxes.append((x1, y1, x2, y2, conf))

            # Xác định rider: person gắn với motobike gần (IoU > 0.10 hoặc tâm xe nằm trong person)
            riders = []  # (person_bbox, has_helmet_bool)
            for (px1, py1, px2, py2, pconf) in person_boxes:
                person_bb = (px1, py1, px2, py2)
                is_rider = False
                for (bx1, by1, bx2, by2, bconf) in bike_boxes:
                    bike_bb = (bx1, by1, bx2, by2)
                    iou_pb = iou(person_bb, bike_bb)
                    bike_cx = (bx1 + bx2) / 2.0
                    bike_cy = (by1 + by2) / 2.0
                    center_in_person = (px1 <= bike_cx <= px2) and (py1 <= bike_cy <= py2)
                    if iou_pb > 0.10 or center_in_person:
                        is_rider = True
                        break
                if not is_rider:
                    continue

                # Kiểm tra helmet trong vùng đầu (top 33% của bbox person)
                ph = max(1, py2 - py1)
                head_y2 = py1 + int(ph * 0.33)
                has_helmet = False
                for (hx1, hy1, hx2, hy2, hconf) in helmet_boxes:
                    hcx = (hx1 + hx2) / 2.0
                    hcy = (hy1 + hy2) / 2.0
                    inside_person = (px1 <= hcx <= px2) and (py1 <= hcy <= py2)
                    in_head_region = (hy1 >= py1) and (hy2 <= head_y2)
                    in_head_center = (py1 <= hcy <= head_y2)  # nới lỏng nếu bbox helmet hơi lệch
                    if inside_person and (in_head_region or in_head_center):
                        has_helmet = True
                        break
                riders.append(((px1, py1, px2, py2), has_helmet))

            # Vẽ: chỉ rider_helmet / rider_nohelmet
            for (px1, py1, px2, py2), has_helmet in riders:
                if has_helmet:
                    label = "rider_helmet"
                    color = color_rider_helmet
                else:
                    label = "rider_nohelmet"
                    color = color_rider_nohelmet

                cv2.rectangle(frame, (px1, py1), (px2, py2), color, thickness)
                (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
                tx, ty = px1, py1 - 5
                if ty - th < 0:
                    ty = py1 + th + 5
                cv2.rectangle(frame, (tx, ty - th - 2), (tx + tw + 2, ty + 2), (0, 0, 0), cv2.FILLED)
                cv2.putText(frame, label, (tx, ty), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

            cv2.imshow("YOLOv8m - Rider Helmet Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
