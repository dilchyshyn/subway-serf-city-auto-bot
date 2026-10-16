"""Евристичний детектор (OpenCV, без навчання).

Потрібен для двох речей:
  * авторозмітка: чорнові мітки для YOLO-датасету (tools/autolabel.py),
    які потім лише виправляються вручну;
  * запасний режим, поки модель ще не натренована (--vision heuristic).
Предмети — невеликі плями характерного кольору (монети, Energy, пауер-апи),
перешкоди — великі яскраві об'єкти на дорозі. Класи, які за кольором не
розрізнити (board_coin, stomp_target, star, board_ready), евристика не
знаходить: їх розмічають вручну.
"""
import cv2
import numpy as np

from .scene import Detection


def hsv_mask(hsv, ranges):
    mask = np.zeros(hsv.shape[:2], np.uint8)
    for r in ranges:
        mask |= cv2.inRange(hsv, np.array(r["lo"]), np.array(r["hi"]))
    return mask


class HeuristicDetector:
    def __init__(self, cfg, geo):
        self.cfg = cfg["heuristic"]
        self.zones = cfg["zones"]
        self.geo = geo
        self._road_shape = None
        self._road = None

    def _road_mask(self, shape):
        """Дорога = об'єднання трапецій усіх смуг від дальнього до ближнього краю зон."""
        if self._road_shape != shape[:2]:
            h, w = shape[:2]
            y0 = min(z[0] for z in self.zones.values())
            y1 = max(z[1] for z in self.zones.values())
            self._road = np.zeros((h, w), np.uint8)
            for lane in range(3):
                cv2.fillPoly(self._road, [self.geo.polygon(lane, y0, y1, w, h)], 255)
            self._road_shape = shape[:2]
        return self._road

    def _pickups(self, hsv):
        h, w = hsv.shape[:2]
        taken = np.zeros((h, w), np.uint8)
        found = []
        for cat in self.cfg["pickups"]:
            mask = hsv_mask(hsv, cat["hsv"]) & ~taken
            n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            for i in range(1, n):
                x, y, bw, bh, area = stats[i]
                # Пляма, більша за max_area, — вже не предмет (наприклад, жовтий вагон)
                if not cat["min_area"] <= area / (h * w) <= cat["max_area"]:
                    continue
                # Предмети компактні: відкидаємо довгі смуги (рейки, краї вагонів)
                if max(bw, bh) > cat.get("max_aspect", 3.0) * max(min(bw, bh), 1):
                    continue
                found.append(Detection(cat["name"], 0.5, x, y, x + bw, y + bh))
                taken[labels == i] = 255
        return found, taken

    def _obstacles(self, hsv, exclude):
        h, w = hsv.shape[:2]
        mask = hsv_mask(hsv, self.cfg["obstacle_hsv"]) & ~exclude & self._road_mask(hsv.shape)
        k = self.cfg["morph_kernel"]
        if k > 1:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        found = []
        for c in contours:
            if cv2.contourArea(c) < self.cfg["min_obstacle_area"] * h * w:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            cls = "obstacle_tall" if bh >= self.cfg["tall_aspect"] * bw else "obstacle_low"
            found.append(Detection(cls, 0.5, x, y, x + bw, y + bh))
        return found

    def detect(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        pickups, taken = self._pickups(hsv)
        return pickups + self._obstacles(hsv, taken)
