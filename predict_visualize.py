# predict_visualize.py
# Dùng YOLO (Ultralytics) để infer trên ẢNH hoặc THƯ MỤC ảnh và visualize kết quả
# Tùy chọn hậu xử lý: suy luận rider_helmet / rider_no_helmet từ 3 lớp (person, motorcycle, helmet)

import argparse
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

# ================== cấu hình mặc định ==================
DEFAULT_WEIGHTS = "/home/22011107/TA/Deep/Deep/runs_combo/train_v8m/weights/best.pt"  # sửa theo checkpoint bạn muốn test
DEFAULT_SOURCE  = "/home/22011107/TA/Deep/Deep/dataset_combined/images/val"          # Ảnh hoặc folder
DEFAULT_OUTDIR  = "/home/22011107/TA/Deep/Deep/pred_vis"
CLASS_NAMES = ["person", "motorcycle", "helmet"]  # thứ tự đúng với dataset bạn train

# Màu vẽ bbox
COLORS = {
    "person":      (0, 255, 0),     # xanh lá
    "motorcycle":  (255, 165, 0),   # cam
    "helmet":      (0, 200, 255),   # vàng-xanh
    "rider_helmet":    (255, 0, 255),  # tím
    "rider_no_helmet": (0, 0, 255),    # đỏ
}

def box_iou_xyxy(a, b):
    # a, b: [x1,y1,x2,y2]
    xA = max(a[0], b[0]); yA = max(a[1], b[1])
    xB = min(a[2], b[2]); yB = min(a[3], b[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = max(0, a[2]-a[0]) * max(0, a[3]-a[1])
    areaB = max(0, b[2]-b[0]) * max(0, b[3]-b[1])
    union = areaA + areaB - inter
    return inter/union if union>0 else 0.0

def center_in_box(cx, cy, box):
    return (box[0] <= cx <= box[2]) and (box[1] <= cy <= box[3])

def draw_box(im, box, color, label=None):
    x1,y1,x2,y2 = map(int, box)
    cv2.rectangle(im, (x1,y1), (x2,y2), color, 2)
    if label:
        cv2.putText(im, label, (x1, max(0, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

def to_xyxy_normed(box_xyxy, W, H):
    x1,y1,x2,y2 = box_xyxy
    x = ((x1+x2)/2)/W; y = ((y1+y2)/2)/H
    w = (x2 - x1)/W;   h = (y2 - y1)/H
    return x,y,w,h

def postproc_riders(persons, motos, helmets):
    """
    persons/motos/helmets: list of dict {box: [x1,y1,x2,y2], conf: float}
    Trả về list rider_helmet & rider_no_helmet: mỗi phần tử là dict với key: box, note
    Logic:
      - một person được coi là rider nếu trọng tâm người nằm "trên" (y nhỏ hơn) tâm bbox moto gần nhất (hoặc IoU > ngưỡng),
      - có helmet nằm bên trong vùng trên của person -> rider_helmet, ngược lại rider_no_helmet.
    """
    riders_h = []
    riders_no = []

    for p in persons:
        pbox = p["box"]
        # tìm moto gần nhất theo khoảng cách tâm hoặc IoU
        px = (pbox[0]+pbox[2])/2; py = (pbox[1]+pbox[3])/2
        is_rider = False
        best_iou = 0
        for m in motos:
            mbox = m["box"]
            mx = (mbox[0]+mbox[2])/2; my = (mbox[1]+mbox[3])/2
            # điều kiện đơn giản: tâm người ở phía trên tâm xe + IoU > một ngưỡng nho nhỏ
            iou = box_iou_xyxy(pbox, mbox)
            if py < my or iou > 0.05:
                is_rider = True
                best_iou = max(best_iou, iou)
        if not is_rider:
            continue

        # kiểm tra có helmet trong vùng đầu (1/3 trên của person) hay không
        has_helmet = False
        head_top = pbox[1]
        head_bottom = pbox[1] + (pbox[3] - pbox[1]) * 0.35
        head_box = [pbox[0], head_top, pbox[2], head_bottom]

        for h in helmets:
            hbox = h["box"]
            hx = (hbox[0]+hbox[2])/2; hy = (hbox[1]+hbox[3])/2
            if center_in_box(hx, hy, head_box):
                has_helmet = True
                break

        if has_helmet:
            riders_h.append({"box": pbox, "head": head_box, "note": f"rider_helmet (IoU_moto≈{best_iou:.2f})"})
        else:
            riders_no.append({"box": pbox, "head": head_box, "note": f"rider_no_helmet (IoU_moto≈{best_iou:.2f})"})

    return riders_h, riders_no

def run(weights, source, outdir, conf=0.4, iou=0.45, device=0, do_rider_logic=True, save_txt=False):
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)

    model = YOLO(weights)
    # hỗ trợ: source là 1 ảnh hoặc một thư mục
    sources = []
    src = Path(source)
    if src.is_dir():
        for p in sorted(src.iterdir()):
            if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
                sources.append(p)
    else:
        if src.exists() and src.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
            sources = [src]

    for idx, img_path in enumerate(sources, 1):
        im0 = cv2.imread(str(img_path))
        if im0 is None:
            print(f"⚠️  Bỏ qua {img_path} (không đọc được).")
            continue
        H, W = im0.shape[:2]

        # infer
        res = model.predict(source=im0, conf=conf, iou=iou, device=device, verbose=False)[0]
        xyxy = res.boxes.xyxy.cpu().numpy()
        cls  = res.boxes.cls.cpu().numpy().astype(int)
        confs= res.boxes.conf.cpu().numpy()

        persons, motos, helmets = [], [], []
        for b, c, cf in zip(xyxy, cls, confs):
            box = [float(b[0]), float(b[1]), float(b[2]), float(b[3])]
            name = CLASS_NAMES[c] if 0 <= c < len(CLASS_NAMES) else f"id{c}"
            if name == "person":
                persons.append({"box": box, "conf": float(cf)})
            elif name == "motorcycle":
                motos.append({"box": box, "conf": float(cf)})
            elif name == "helmet":
                helmets.append({"box": box, "conf": float(cf)})

        # vẽ bbox 3 lớp
        vis = im0.copy()
        for p in persons:
            draw_box(vis, p["box"], COLORS["person"], f"person {p['conf']:.2f}")
        for m in motos:
            draw_box(vis, m["box"], COLORS["motorcycle"], f"motorcycle {m['conf']:.2f}")
        for h in helmets:
            draw_box(vis, h["box"], COLORS["helmet"], f"helmet {h['conf']:.2f}")

        # hậu xử lý: rider_helmet / rider_no_helmet
        if do_rider_logic:
            riders_h, riders_no = postproc_riders(persons, motos, helmets)
            for r in riders_h:
                draw_box(vis, r["box"], COLORS["rider_helmet"], "rider_helmet")
                # vẽ thêm box vùng đầu cho đẹp
                draw_box(vis, r["head"], COLORS["rider_helmet"])
            for r in riders_no:
                draw_box(vis, r["box"], COLORS["rider_no_helmet"], "rider_no_helmet")
                draw_box(vis, r["head"], COLORS["rider_no_helmet"])

        # lưu ảnh visualize
        out_img = outdir / f"{img_path.stem}_vis.jpg"
        cv2.imwrite(str(out_img), vis)

        # (tùy chọn) export txt detections theo YOLO format
        if save_txt:
            out_lbl = outdir / f"{img_path.stem}.txt"
            with open(out_lbl, "w") as f:
                # ghi 3 lớp gốc
                for p in persons:
                    x,y,w,h = to_xyxy_normed(p["box"], W, H)
                    f.write(f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
                for m in motos:
                    x,y,w,h = to_xyxy_normed(m["box"], W, H)
                    f.write(f"1 {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
                for hbox in helmets:
                    x,y,w,h = to_xyxy_normed(hbox["box"], W, H)
                    f.write(f"2 {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

        print(f"[{idx}/{len(sources)}] ✅ {img_path.name} → {out_img}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, default=DEFAULT_WEIGHTS, help="đường dẫn .pt đã fine-tune")
    ap.add_argument("--source",  type=str, default=DEFAULT_SOURCE,  help="ảnh hoặc thư mục ảnh")
    ap.add_argument("--outdir",  type=str, default=DEFAULT_OUTDIR,  help="nơi lưu ảnh visualize")
    ap.add_argument("--conf",    type=float, default=0.40)
    ap.add_argument("--iou",     type=float, default=0.45)
    ap.add_argument("--device",  type=str, default="0")  # "0" GPU0, "cpu" nếu không có GPU
    ap.add_argument("--rider_logic", action="store_true", help="bật suy luận rider_helmet / rider_no_helmet")
    ap.add_argument("--save_txt", action="store_true", help="lưu thêm txt YOLO kết quả 3 lớp")
    args = ap.parse_args()

    run(
        weights=args.weights,
        source=args.source,
        outdir=args.outdir,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        do_rider_logic=args.rider_logic,
        save_txt=args.save_txt
    )
