import os
import glob
import cv2
import xml.etree.ElementTree as ET
from tqdm import tqdm

# ==== CẤU HÌNH ====
IMAGES_DIR = r"E:\Deep\helmetdata\images"
ANN_DIR    = r"E:\Deep\helmetdata\annotations"
OUT_DIR    = r"E:\Deep\helmetdata\viz_xml"
os.makedirs(OUT_DIR, exist_ok=True)

# Nếu dữ liệu bạn còn "With Helmet"/"Without Helmet", map sang tên chuẩn để hiển thị
NAME_MAP = {
    "With Helmet": "helmet",
    "with helmet": "helmet",
    "Helmet": "helmet",
    "helmet": "helmet",
    "Without Helmet": "no-helmet",
    "without helmet": "no-helmet",
    "No Helmet": "no-helmet",
    "no helmet": "no-helmet",
    "person-motorbike": "rider",
    "rider": "rider",
}

# Màu hiển thị theo lớp (BGR)
COLOR = {
    "rider":      (36, 255, 12),   # xanh lá
    "helmet":     (0, 165, 255),   # cam
    "no-helmet":  (0, 0, 255),     # đỏ
}

def voc_xml_to_boxes(xml_path):
    boxes = []
    if not os.path.isfile(xml_path):
        return boxes
    root = ET.parse(xml_path).getroot()
    for obj in root.findall("object"):
        name_tag = obj.find("name")
        if name_tag is None or not name_tag.text:
            continue
        raw_name = name_tag.text.strip()
        cls = NAME_MAP.get(raw_name, raw_name)  # chuẩn hoá tên để hiển thị

        bnd = obj.find("bndbox")
        if bnd is None: 
            continue
        try:
            x1 = int(float(bnd.findtext("xmin", "0")))
            y1 = int(float(bnd.findtext("ymin", "0")))
            x2 = int(float(bnd.findtext("xmax", "0")))
            y2 = int(float(bnd.findtext("ymax", "0")))
        except:
            continue

        boxes.append((cls, (x1, y1, x2, y2)))
    return boxes

def draw_boxes(img, boxes):
    for cls, (x1, y1, x2, y2) in boxes:
        color = COLOR.get(cls, (255, 255, 255))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        label = str(cls)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 6, y1), color, -1)
        cv2.putText(img, label, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return img

# Quét ảnh (mọi định dạng phổ biến)
image_exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
image_list = []
for ext in image_exts:
    image_list.extend(glob.glob(os.path.join(IMAGES_DIR, ext)))

if not image_list:
    print("⚠️ Không tìm thấy ảnh trong:", IMAGES_DIR)
else:
    for img_path in tqdm(image_list, desc="Visualizing VOC"):
        img = cv2.imread(img_path)
        if img is None:
            continue

        stem = os.path.splitext(os.path.basename(img_path))[0]
        xml_path = os.path.join(ANN_DIR, stem + ".xml")

        boxes = voc_xml_to_boxes(xml_path)
        out = img.copy()
        if boxes:
            out = draw_boxes(out, boxes)
        # nếu không có nhãn, vẫn lưu ảnh gốc để bạn biết file nào thiếu annotation

        out_path = os.path.join(OUT_DIR, os.path.basename(img_path))
        cv2.imwrite(out_path, out)

    print(f"✅ Xong! Ảnh đã vẽ box nằm ở: {OUT_DIR}")
