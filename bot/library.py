"""Бібліотека маршрутів City Tour і впізнавання рівня за карткою.

Порядок рівнів у турі змінюється, тож маршрут шукається не за назвою, а за
текстом завдань на картці (OCR): рівні з однаковими типами завдань
відрізняються лише числами. Завдання зберігаються між спробами, тому у відео
рівень часто пройдено кількома спробами: кожна стартує з іншого стану картки
(які завдання вже виконані). Бот бере ту спробу, що стартувала з такого ж стану.
"""
import json
from dataclasses import dataclass
from pathlib import Path

import cv2

from . import route, video

SAME_LEVEL = 0.55  # подібність тексту завдань, з якої це той самий рівень
MIN_COMMON_NUMBERS = 2  # для впевненого збігу потрібні щонайменше 2 однакові числа завдань


@dataclass
class Entry:
    path: Path
    name: str
    text: str
    state: tuple
    crew: Path | None


@dataclass
class Match:
    path: Path        # файл маршруту (обрана спроба)
    name: str
    score: float      # подібність тексту найкращого рівня
    margin: float     # відрив від найкращого іншого рівня
    common: int       # скільки чисел завдань збіглося точно
    text: str         # що OCR прочитав на картці (для діагностики)
    state: tuple      # стан картки на телефоні
    crew: Path | None  # кадр FEATURED CREW з відео: яким персонажем грали


def entry_text_state(rdir, data):
    """Текст і стан картки маршруту: з JSON, а якщо їх там немає — OCR вирізки."""
    if "card_text" in data and "card_state" in data:
        return data["card_text"], tuple(data["card_state"])
    band = cv2.imread(str(rdir / data["card"]))
    return video.card_text(band), video.card_state(band)


class RouteLibrary:
    def __init__(self, routes_dir):
        self.dir = Path(routes_dir)
        self.entries = []
        for path in sorted(self.dir.glob("*.json")):
            if path.name == "index.json":
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            if not data.get("card") or not (self.dir / data["card"]).exists():
                continue
            text, state = entry_text_state(self.dir, data)
            crew = self.dir / data["crew"] if data.get("crew") else None
            self.entries.append(Entry(path, data["level"], text, state, crew))

    def __len__(self):
        return len(self.entries)

    def match_band(self, band):
        """Найкраща спроба для вирізки картки: спершу рівень за текстом, потім спроба за станом."""
        text, state = video.card_text(band), video.card_state(band)
        scored = sorted(((video.text_similarity(text, e.text), e) for e in self.entries),
                        key=lambda x: x[0], reverse=True)
        if not scored:
            return None
        best, top = scored[0]
        # спроби того самого рівня: схожі на найкращий маршрут і майже так само схожі на картку
        same = [e for s, e in scored if s >= best - 0.1 and video.text_similarity(e.text, top.text) >= SAME_LEVEL]
        if top not in same:
            same.insert(0, top)
        others = [s for s, e in scored if e not in same]
        margin = best - (others[0] if others else 0.0)
        # спроба з тим самим станом; інакше — з найбільшим збігом виконаних завдань,
        # серед рівних — найраніша у відео (назви маршрутів закінчуються номером забігу)
        attempts = sorted(same, key=lambda e: e.path.name)
        pick = max(enumerate(attempts),
                   key=lambda ie: (ie[1].state == state, sum(a == b for a, b in zip(ie[1].state, state)), -ie[0]))[1]
        return Match(pick.path, pick.name, best, margin, video.common_numbers(text, top.text),
                     text, state, pick.crew)

    def match(self, frame):
        """(Match або None, рамка кнопки PLAY або None) для кадру з карткою рівня."""
        play = video.find_play_button(frame)
        if play is None or not self.entries:
            return None, play
        return self.match_band(video._band(frame, play, video.OBJECTIVES_DY)), play

    @staticmethod
    def confident(m, cfg):
        return (m is not None and m.score >= cfg["tour"]["min_score"] and m.margin >= cfg["tour"]["min_margin"]
                and m.common >= MIN_COMMON_NUMBERS)


def load_route(path):
    return route.load(path)
