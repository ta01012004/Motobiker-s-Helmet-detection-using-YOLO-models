# save_best_model.py
import shutil, os
from pathlib import Path

src1 = Path("/home/22011107/TA/Deep/Deep/runs_combo/train_v8m3/weights/best.pt")
src2 = Path("/home/22011107/TA/Deep/Deep/runs_combo/train_helmet_pre/weights/best.pt")
dst_dir = Path("/home/22011107/TA/Deep/Deep/models_trained")
dst_dir.mkdir(parents=True, exist_ok=True)

if src1.exists():
    shutil.copy2(src1, dst_dir / "yolov8m_best.pt")
    print("✅ Đã lưu yolov8m_best.pt")

if src2.exists():
    shutil.copy2(src2, dst_dir / "helmet_best.pt")
    print("✅ Đã lưu helmet_best.pt")

print("📁 Mô hình đã được lưu tại:", dst_dir)
