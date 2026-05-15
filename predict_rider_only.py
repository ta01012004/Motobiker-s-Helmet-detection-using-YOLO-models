# predict_rider_only.py
# Suy luận logic để chỉ xuất 2 nhãn: rider_helmet (0) và rider_no_helmet (1)
# Đầu vào model 3 lớp: person, motorcycle, helmet

import argparse
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

# ====== cấu hình mặc định ======
DEFAULT_WEIGHTS = "/home/22011107/TA/Deep/Deep/runs_combo/train_v8m/weights/best.pt"
DEFAULT_SOURCE  = "/home/22011107/TA/Deep/Deep/dataset_combined/images/val"
DEFAULT_OUTDIR  = "/home/22011107/TA/Deep/Deep/pred_rider_only"
# Thứ tự lớp của mô hình đầu vào (phải đúng theo lúc train)
SRC_CLASS_NAMES = ["person", "motorcycle", "helmet"]

# Màu vẽ 2 lớp cuối
COLOR_RH  = (255, 0, 255)  # rider_helmet - tím
COLOR_RNH = (0, 0, 255)    # rider_no_helmet - đỏ

def iou_xyxy(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    areaA = max(0, a[2]-a[0]) * max(0, a[3]-a[1])
    areaB = max(0, b[2]-b[0]) * max(0, b[3]-b[1])
    union = areaA + areaB - inter
    return inter/union if union > 0 else 0.0

def center_in(box, x, y):
    return (box[0] <= x <= box[2]) and (box[1] <= y <= box[3])

def box_center(box):
    return ( (box[0]+box[2])/2.0, (box[1]+box[3])/2.0 )

def draw_box(im, box, color, label):
    x1,y1,x2,y2 = map(int, box)
    cv2.rectangle(im, (x1,y1), (x2,y2), color, 2)
    cv2.putText(im, label, (x1, max(0, y1-7)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

def normalize_xyxy(box, W, H):
    x1,y1,x2,y2 = box
    x = ((x1+x2)/2)/W; y = ((y1+y2)/2)/H
    w = (x2-x1)/W;     h = (y2-y1)/H
    return x,y,w,h

def infer_riders_only(model, img_bgr, conf=0.4, iou=0.45, device=0):
    """Trả về 2 list: riders_helmet, riders_no_helmet; mỗi phần tử: dict(box, head, conf)"""
    H, W = img_bgr.shape[:2]
    r = model.predict(source=img_bgr, conf=conf, iou=iou, device=device, verbose=False)[0]

    xyxy = r.boxes.xyxy.cpu().numpy()
    cls  = r.boxes.cls.cpu().numpy().astype(int)
    confs= r.boxes.conf.cpu().numpy()

    persons, motos, helmets = [], [], []
    for b, c, cf in zip(xyxy, cls, confs):
        name = SRC_CLASS_NAMES[c] if 0 <= c < len(SRC_CLASS_NAMES) else f"id{c}"
        box = [float(b[0]), float(b[1]), float(b[2]), float(b[3])]
        if name == "person":
            persons.append({"box": box, "conf": float(cf)})
        elif name == "motorcycle":
            motos.append({"box": box, "conf": float(cf)})
        elif name == "helmet":
            helmets.append({"box": box, "conf": float(cf)})

    riders_h, riders_no = [], []
    for p in persons:
        pbox = p["box"]
        pcx, pcy = box_center(pbox)
        # rider nếu có moto "liên quan" (tâm người ở trên tâm moto hoặc IoU > ngưỡng nhỏ)
        related_to_moto = False
        best_i = 0.0
        for m in motos:
            mbox = m["box"]
            mcx, mcy = box_center(mbox)
            i = iou_xyxy(pbox, mbox)
            if pcy < mcy or i > 0.05:
                related_to_moto = True
                best_i = max(best_i, i)
        if not related_to_moto:
            continue

        # kiểm tra helmet trong vùng đầu (top ~ 35% người)
        head_top = pbox[1]
        head_bottom = pbox[1] + (pbox[3]-pbox[1]) * 0.35
        head_box = [pbox[0], head_top, pbox[2], head_bottom]

        has_helmet = False
        for h in helmets:
            hcx, hcy = box_center(h["box"])
            if center_in(head_box, hcx, hcy):
                has_helmet = True
                break

        item = {"box": pbox, "head": head_box, "conf": p["conf"], "note": f"IoU_moto≈{best_i:.2f}"}
        if has_helmet:
            riders_h.append(item)
        else:
            riders_no.append(item)

    return riders_h, riders_no

def main(weights, source, outdir, conf, iou, device, save_txt):
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    model = YOLO(weights)

    # lấy danh sách input (ảnh hoặc folder)
    src = Path(source)
    if src.is_dir():
        image_paths = [p for p in sorted(src.iterdir()) if p.suffix.lower() in [".jpg",".jpeg",".png",".bmp"]]
    else:
        image_paths = [src] if src.exists() else []
    if not image_paths:
        print(f"❌ Không tìm thấy ảnh trong: {source}")
        return

    total_rh = total_rnh = 0

    for i, img_path in enumerate(image_paths, 1):
        im = cv2.imread(str(img_path))
        if im is None:
            print(f"⚠️ Bỏ qua {img_path} (không đọc được)")
            continue
        H, W = im.shape[:2]
        riders_h, riders_no = infer_riders_only(model, im, conf=conf, iou=iou, device=device)

        # visualize chỉ 2 lớp cuối
        vis = im.copy()
        for r in riders_h:
            draw_box(vis, r["box"], COLOR_RH, "rider_helmet")
            # vẽ thêm vùng đầu
            draw_box(vis, r["head"], COLOR_RH, None)
        for r in riders_no:
            draw_box(vis, r["box"], COLOR_RNH, "rider_no_helmet")
            draw_box(vis, r["head"], COLOR_RNH, None)

        out_img = outdir / f"{img_path.stem}_rider.jpg"
        cv2.imwrite(str(out_img), vis)

        # (tuỳ chọn) lưu txt 2 lớp (0=rider_helmet, 1=rider_no_helmet)
        if save_txt:
            out_txt = outdir / f"{img_path.stem}.txt"
            with open(out_txt, "w") as f:
                for r in riders_h:
                    x,y,w,h = normalize_xyxy(r["box"], W, H)
                    f.write(f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
                for r in riders_no:
                    x,y,w,h = normalize_xyxy(r["box"], W, H)
                    f.write(f"1 {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

        # in kết quả từng ảnh
        print(f"[{i}/{len(image_paths)}] {img_path.name}: rider_helmet={len(riders_h)}, rider_no_helmet={len(riders_no)} → {out_img}")
        total_rh  += len(riders_h)
        total_rnh += len(riders_no)

    print(f"\n📊 Tổng kết: rider_helmet={total_rh}, rider_no_helmet={total_rnh}")
    print(f"🖼  Ảnh visualize lưu ở: {outdir}")
    if save_txt:
        print(f"📝 Đã lưu nhãn YOLO 2 lớp (0=rider_helmet, 1=rider_no_helmet).")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, default=DEFAULT_WEIGHTS, help="đường dẫn .pt đã fine-tune (3 lớp)")
    ap.add_argument("--source",  type=str, default=DEFAULT_SOURCE,  help="ảnh hoặc thư mục ảnh")
    ap.add_argument("--outdir",  type=str, default=DEFAULT_OUTDIR)
    ap.add_argument("--conf",    type=float, default=0.40)
    ap.add_argument("--iou",     type=float, default=0.45)
    ap.add_argument("--device",  type=str, default="0")  # "0" GPU0, "cpu" nếu không có GPU
    ap.add_argument("--save_txt", action="store_true", help="lưu nhãn YOLO 2 lớp cho rider_*")
    args = ap.parse_args()

    main(
        weights=args.weights,
        source=args.source,
        outdir=args.outdir,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        save_txt=args.save_txt
    )
