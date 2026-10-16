"""Бібліотека маршрутів City Tour і впізнавання рівня за карткою.

Порядок рівнів у турі змінюється, тож маршрут шукається не за назвою, а за
блоком завдань на картці рівня: його відбиток порівнюється з картками,
збереженими разом із маршрутами (routes/cards/*.png).
"""
import json
from dataclasses import dataclass
from pathlib import Path

import cv2

from . import route, video


@dataclass
class Match:
    path: Path        # файл маршруту
    name: str
    score: float      # подібність найкращої картки
    margin: float     # відрив від другої найкращої
    crew: Path | None  # кадр FEATURED CREW з відео: яким персонажем грали


class RouteLibrary:
    def __init__(self, routes_dir):
        self.dir = Path(routes_dir)
        self.entries = []  # (шлях маршруту, назва, відбиток, crew)
        for path in sorted(self.dir.glob("*.json")):
            if path.name == "index.json":
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            card = data.get("card")
            if not card or not (self.dir / card).exists():
                continue
            band = cv2.imread(str(self.dir / card))
            crew = self.dir / data["crew"] if data.get("crew") else None
            self.entries.append((path, data["level"], video.band_fingerprint(band), crew))

    def __len__(self):
        return len(self.entries)

    def match(self, frame):
        """Найкращий маршрут для картки рівня на кадрі або None, якщо картки немає."""
        play = video.find_play_button(frame)
        if play is None or not self.entries:
            return None, play
        fp = video.card_fingerprint(frame, play)
        scored = sorted(((video.card_similarity(fp, e[2]), e) for e in self.entries),
                        key=lambda x: x[0], reverse=True)
        best, (path, name, _, crew) = scored[0]
        second = scored[1][0] if len(scored) > 1 else 0.0
        return Match(path, name, best, best - second, crew), play

    @staticmethod
    def confident(m, cfg):
        return m is not None and m.score >= cfg["tour"]["min_score"] and m.margin >= cfg["tour"]["min_margin"]


def load_route(path):
    return route.load(path)
