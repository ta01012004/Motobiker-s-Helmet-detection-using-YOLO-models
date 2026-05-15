# -*- coding: utf-8 -*-
"""
Huấn luyện YOLOv8-m trên 3 lớp [person(0), motobike(1), helmet(2)] và suy luận realtime.
Yêu cầu:
 - Dữ liệu YOLO (TXT) ở /dataloc (đã đúng thứ tự nhãn: 0 person, 1 motobike, 2 helmet).
 - Huấn luyện YOLOv8-m (Ultralytics) với SGD, imgsz=640, batch=16, epochs=120, có augmentation.
 - Xuất file .pt sau huấn luyện.
 - Hậu xử lý: ghép person + motobike => rider; kiểm tra helmet trong vùng đầu của person
   => rider_helmet / rider_nohelmet. Chỉ vẽ 2 nhãn này.
 - Cài đặt: ultralytics, opencv-python (hoặc opencv-python-headless trên máy không GUI).
"""

import os
import glob
import random
import cv2
from ultralytics import YOLO

# -------------------- CẤU HÌNH ĐƯỜNG DẪN --------------------
# DỮ LIỆU CHỈ DÙNG /dataloc
dataloc_images_dir = "/home/22011107/TA/Deep/Deep/dataloc/images"
dataloc_labels_dir = "/home/22011107/TA/Deep/Deep/dataloc/labels"

# TRỌNG SỐ PRETRAIN (DÙNG ĐƯỜNG DẪN TUYỆT ĐỐI → KHÔNG TẢI MẠNG)
yolov8m_pt_path = "/home/22011107/TA/Deep/Deep/models/yolov8m.pt"

# THƯ MỤC LÀM VIỆC/LƯU CẤU HÌNH
base_dir = "/home/22011107/TA/Deep/Deep"

# -------------------- KIỂM TRA TỒN TẠI --------------------
if not os.path.isdir(dataloc_images_dir) or not os.path.isdir(dataloc_labels_dir):
    raise FileNotFoundError("Thư mục dữ liệu dataloc không tồn tại. Kiểm tra lại đường dẫn.")
if not os.path.isfile(yolov8m_pt_path):
    raise FileNotFoundError(f"Không tìm thấy trọng số pretrain: {yolov8m_pt_path}.")

# -------------------- B1: THU THẬP ẢNH /dataloc --------------------
print("Đang thu thập ảnh & nhãn từ tập dataloc (YOLO format, 0:person,1:motobike,2:helmet)...")
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

total_images = len(dataloc_images_list)
if total_images == 0:
    raise RuntimeError("Không tìm thấy ảnh nào trong dataloc.")

print(f"Đã thu thập {total_images} ảnh từ dataloc.")

# -------------------- B2: CHIA TRAIN/VAL/TEST --------------------
print("Đang chia train/val/test (80/10/10)...")
random.seed(42)
random.shuffle(dataloc_images_list)

train_count = int(0.8 * total_images)
val_count   = int(0.1 * total_images)
test_count  = total_images - train_count - val_count

train_images = dataloc_images_list[:train_count]
val_images   = dataloc_images_list[train_count:train_count + val_count]
test_images  = dataloc_images_list[train_count + val_count:]

print(f"Tổng số ảnh: {total_images} -> Train: {len(train_images)}, Val: {len(val_images)}, Test: {len(test_images)}")

# -------------------- B3: GHI LIST ẢNH --------------------
print("Đang ghi đường dẫn ảnh cho train/val/test vào .txt...")
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

print(f"Đã tạo {train_list_path}, {val_list_path}, {test_list_path}.")

# -------------------- B4: TẠO data.yaml (đúng thứ tự nhãn) --------------------
print("Đang tạo data.yaml với names: {0:person, 1:motobike, 2:helmet} ...")
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
print(f"Đã tạo file cấu hình dữ liệu: {data_yaml_path}")

# -------------------- B5: TRAIN YOLOv8-m (SGD + AUGMENTATION) --------------------
print("Bắt đầu huấn luyện YOLOv8-m (SGD + augmentation)...")
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
    name="helmet_yolov8m_person-bike-helmet",

    # ---------------- Augmentation (tập trung tăng Recall cho helmet) ----------------
    # Màu sắc (HSV)
    hsv_h=0.015,     # đổi hue nhẹ
    hsv_s=0.70,      # tăng bão hoà
    hsv_v=0.40,      # thay đổi độ sáng

    # Hình học
    degrees=5.0,     # xoay ±5°
    translate=0.10,  # tịnh tiến ±10%
    scale=0.50,      # co giãn
    shear=2.0,       # shear nhẹ
    perspective=0.0005,  # biến dạng phối cảnh rất nhẹ

    # Lật & ghép ảnh
    fliplr=0.5,      # lật ngang 50%
    flipud=0.5,      # lật dọc 50%
    mosaic=1.0,      # bật mosaic
    mixup=0.10,      # mixup nhẹ

    # Khác
    copy_paste=0.0,  # tránh sinh object ảo cho bài này
    # erasing=0.0,   # bật 0.1 nếu phiên bản Ultralytics hỗ trợ và muốn Random Erasing
)
print("Huấn luyện hoàn tất!")

# -------------------- B6: CHỌN TRỌNG SỐ (best.pt ưu tiên, fallback last.pt) --------------------
weights_dir = os.path.join("runs", "helmet_yolov8m_person-bike-helmet", "weights")
best_weights_path = os.path.join(weights_dir, "best.pt")
last_weights_path = os.path.join(weights_dir, "last.pt")

if not os.path.isfile(best_weights_path):
    # nếu session bị thêm số, tìm trong toàn bộ runs
    found_best, found_last = None, None
    for root, dirs, files in os.walk("runs"):
        if "best.pt" in files and root.endswith("weights"):
            found_best = os.path.join(root, "best.pt")
        if "last.pt" in files and root.endswith("weights"):
            found_last = os.path.join(root, "last.pt")
    if found_best:
        best_weights_path = found_best
    elif os.path.isfile(last_weights_path):
        best_weights_path = last_weights_path
    elif found_last:
        best_weights_path = found_last
    else:
        raise FileNotFoundError("Không tìm thấy best.pt hoặc last.pt sau huấn luyện. Kiểm tra thư mục runs/.")

if os.path.basename(best_weights_path) == "best.pt":
    print(f"Trọng số được dùng cho suy luận: {best_weights_path} (best.pt)")
else:
    print(f"Không có best.pt, dùng tạm: {best_weights_path}")

# -------------------- B7: SUY LUẬN REALTIME (chỉ vẽ rider_helmet / rider_nohelmet) --------------------
print("Đang tải mô hình và chuẩn bị suy luận realtime...")
trained_model = YOLO(best_weights_path)

# 0 = webcam; hoặc thay bằng đường dẫn video
source = 0

def iou(boxA, boxB):
    # box: (x1,y1,x2,y2)
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
# (Lưu ý: trên Windows DISPLAY có thể rỗng; nếu bạn có màn hình mà vẫn bị bỏ qua, hãy comment khối if này.)
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

            # Xác định rider: person gắn với motobike gần (IoU > 0.1 hoặc tâm xe nằm trong person)
            riders = []  # mỗi phần tử: (person_bbox, has_helmet_bool)
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
                    continue  # chỉ quan tâm person thật sự là rider

                # kiểm tra helmet trong vùng đầu (top 33% chiều cao person)
                ph = max(1, py2 - py1)
                head_y2 = py1 + int(ph * 0.33)
                has_helmet = False
                for (hx1, hy1, hx2, hy2, hconf) in helmet_boxes:
                    hcx = (hx1 + hx2) / 2.0
                    hcy = (hy1 + hy2) / 2.0
                    inside_person = (px1 <= hcx <= px2) and (py1 <= hcy <= py2)
                    # chấp nhận hoặc bbox helmet nằm trong vùng đầu, hoặc ít nhất tâm mũ trong vùng đầu
                    in_head_region = (hy1 >= py1) and (hy2 <= head_y2)
                    in_head_center = (py1 <= hcy <= head_y2)
                    if inside_person and (in_head_region or in_head_center):
                        has_helmet = True
                        break
                riders.append(((px1, py1, px2, py2), has_helmet))

            # VẼ KẾT QUẢ: CHỈ 2 NHÃN rider_helmet / rider_nohelmet
            for (px1, py1, px2, py2), has_helmet in riders:
                if has_helmet:
                    label = "rider_helmet"
                    color = color_rider_helmet
                else:
                    label = "rider_nohelmet"
                    color = color_rider_nohelmet

                cv2.rectangle(frame, (px1, py1), (px2, py2), color, thickness)
                (tw, th), base = cv2.getTextSize(label, font, font_scale, thickness)
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

