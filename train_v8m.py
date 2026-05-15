from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO(r"E:/Deep/models/yolov8m.pt")
    model.train(
        data=r"E:/Deep/dataset_combined.yaml",
        epochs=60,
        batch=16,
        imgsz=640,
        device=0,       # GPU 3050
        workers=4
    )
