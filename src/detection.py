from pathlib import Path

import numpy as np
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_model = None


def load_model(model_path=None):
    """โหลด best.pt ครั้งเดียว แล้วคืนโมเดลให้ฟังก์ชันตรวจภาพ"""
    global _model

    if model_path is None:
        candidates = [
            PROJECT_ROOT / "weights" / "best.pt",
            PROJECT_ROOT / "best.pt",
            PROJECT_ROOT / "runs" / "helmet_yolo11n_v1" / "weights" / "best.pt",
        ]
        model_path = next((path for path in candidates if path.is_file()), None)

    if model_path is None or not Path(model_path).is_file():
        raise FileNotFoundError("หา best.pt ไม่เจอ กรุณาวางไว้ใน weights/ หรือโฟลเดอร์โปรเจกต์")

    _model = YOLO(str(model_path))
    return _model


def predict_image(image_rgb, confidence=0.25):
    """รับภาพ RGB แล้วคืนสิ่งที่พบ จำนวนแต่ละคลาส และภาพที่วาดกรอบแล้ว"""
    global _model

    if not 0 <= confidence <= 1:
        raise ValueError("confidence ต้องอยู่ระหว่าง 0 ถึง 1")

    if _model is None:
        load_model()

    image_rgb = np.asarray(image_rgb)

    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("ภาพต้องเป็น RGB 3 ช่องสี")
    if image_rgb.dtype != np.uint8:
        raise TypeError("ภาพต้องเป็น uint8 ค่าสี 0–255")

    # NumPy ที่ส่งให้ Ultralytics ใช้ BGR จึงแปลงจาก RGB ก่อน
    image_bgr = np.ascontiguousarray(image_rgb[:, :, ::-1])
    result = _model.predict(
        source=image_bgr,
        conf=confidence,
        verbose=False,
    )[0]

    detections = []
    counts = {}
    names = _model.names

    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls[0].item())

            if isinstance(names, dict):
                class_name = names.get(class_id, names.get(str(class_id), str(class_id)))
            else:
                class_name = names[class_id]

            detections.append({
                "class_name": class_name,
                "confidence": float(box.conf[0].item()),
                "bbox_xyxy": [
                    int(round(value))
                    for value in box.xyxy[0].detach().cpu().tolist()
                ],
            })
            counts[class_name] = counts.get(class_name, 0) + 1

    # plot() คืนภาพ BGR; แปลงกลับเป็น RGB ตามข้อตกลงของทีม
    annotated_rgb = result.plot()[:, :, ::-1].copy()

    return {
        "detections": detections,
        "counts": counts,
        "annotated_rgb": annotated_rgb,
    }
