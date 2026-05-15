import cv2
from pathlib import Path

IMG_DIR = "E:/Deep/data_autolabel/images"
LBL_DIR = "E:/Deep/data_autolabel/labels"
CLASS_NAMES = ["person", "motorcycle", "helmet", "no-helmet"]

for img_path in Path(IMG_DIR).glob("*.jpg"):
    lbl_path = Path(LBL_DIR) / (img_path.stem + ".txt")
    if not lbl_path.exists():
        continue
    img = cv2.imread(str(img_path))
    H, W = img.shape[:2]

    with open(lbl_path) as f:
        for line in f:
            cls, x, y, w, h = map(float, line.strip().split())
            cls = int(cls)
            x1 = int((x - w/2) * W)
            y1 = int((y - h/2) * H)
            x2 = int((x + w/2) * W)
            y2 = int((y + h/2) * H)
            color = (0,255,0) if cls==0 else (255,165,0) if cls==1 else (0,200,255) if cls==2 else (0,0,255)
            label = CLASS_NAMES[cls]
            cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
            cv2.putText(img, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Gán nhãn", img)
    if cv2.waitKey(0) == 27:  # nhấn ESC để thoát
        break
