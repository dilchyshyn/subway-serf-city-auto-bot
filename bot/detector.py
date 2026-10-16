"""Детектор об'єктів на основі навченої YOLO-моделі (ultralytics)."""
from pathlib import Path

from .scene import Detection


class YoloDetector:
    def __init__(self, model_path, conf=0.35, imgsz=640):
        from ultralytics import YOLO  # імпорт тут: евристичний режим працює й без ultralytics

        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Модель {model_path} не знайдено. Спочатку натренуй її: python tools/train.py "
                "(або запусти з --vision heuristic)")
        self.model = YOLO(model_path, task="detect")
        self.conf = conf
        self.imgsz = imgsz

    def detect(self, frame):
        res = self.model.predict(frame, conf=self.conf, imgsz=self.imgsz, verbose=False)[0]
        names = res.names
        return [Detection(names[int(c)], float(p), *map(float, xyxy))
                for xyxy, c, p in zip(res.boxes.xyxy.tolist(), res.boxes.cls.tolist(),
                                      res.boxes.conf.tolist())]
