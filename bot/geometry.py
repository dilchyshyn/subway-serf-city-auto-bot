"""Перспективна геометрія трьох смуг.

Гравець біжить унизу по центру, смуги сходяться до точки сходу
(vanishing point). Усі координати нормовані (0..1), тому від роздільної
здатності телефону нічого не залежить.
"""
import numpy as np


class LaneGeometry:
    def __init__(self, cfg):
        self.cfg = cfg

    def _t(self, y):
        # t = 1 на рівні гравця, 0 у точці сходу; ширина й зсув смуги лінійні по t
        lc = self.cfg["lanes"]
        return (y - lc["vanishing_point"][1]) / (lc["near_y"] - lc["vanishing_point"][1])

    def center_x(self, lane, y):
        lc = self.cfg["lanes"]
        vx = lc["vanishing_point"][0]
        return vx + (lc["centers_near"][lane] - vx) * self._t(y)

    def half_width(self, y):
        return self.cfg["lanes"]["width_near"] / 2 * self._t(y)

    def polygon(self, lane, y0, y1, w, h):
        """Трапеція смуги між y0 (дальній край) і y1 (ближній), у пікселях."""
        pts = []
        for y, side in ((y0, -1), (y0, 1), (y1, 1), (y1, -1)):
            x = self.center_x(lane, y) + side * self.half_width(y)
            pts.append((int(x * w), int(y * h)))
        return np.array(pts, dtype=np.int32)

    def lanes_spanning(self, x1, x2, y):
        """Смуги, центр яких на висоті y потрапляє в [x1, x2]. Широка перешкода займає кілька смуг."""
        lanes = [l for l in range(3) if x1 <= self.center_x(l, y) <= x2]
        if not lanes:
            lane = self.nearest_lane((x1 + x2) / 2, y)
            lanes = [lane] if lane is not None else []
        return lanes

    def nearest_lane(self, x, y, tolerance=1.5):
        """Найближча смуга до точки або None, якщо точка далеко поза дорогою."""
        dist = [abs(x - self.center_x(l, y)) for l in range(3)]
        lane = int(np.argmin(dist))
        return lane if dist[lane] <= self.half_width(y) * tolerance else None

    def zone_of(self, y, names=("near", "far")):
        for name in names:
            y0, y1 = self.cfg["zones"][name]
            if y0 <= y <= y1:
                return name
        return None
