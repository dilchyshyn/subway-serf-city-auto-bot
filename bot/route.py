"""Запис і відтворення маршрутів рівнів City Tour.

Рівні City Tour мають фіксовану розкладку, тому прохід, у якому виконано
всі 3 завдання, можна записати й повторити. Запис читає дотики з
`adb shell getevent -lt` і перетворює їх на жести з таймінгом відносно
першого дотику (тапу «Грати» на екрані рівня). Відтворення повторює ті
самі жести з тими самими інтервалами.
"""
import json
import re
import subprocess
from dataclasses import asdict, dataclass

from .controls import adb_cmd

EVENT_RE = re.compile(r"\[\s*([\d.]+)\]\s+(\S+):\s+(\S+)\s+(\S+)\s+(\S+)")


@dataclass
class Gesture:
    t: float          # секунди від першого жесту маршруту
    action: str       # tap / left / right / jump / roll
    x1: float         # нормовані координати 0..1
    y1: float
    x2: float
    y2: float
    duration_ms: int


def classify(x1, y1, x2, y2, min_dist=0.04):
    dx, dy = x2 - x1, y2 - y1
    if max(abs(dx), abs(dy)) < min_dist:
        return "tap"
    if abs(dx) > abs(dy):
        return "right" if dx > 0 else "left"
    return "roll" if dy > 0 else "jump"


def parse_getevent(lines, x_max, y_max):
    """Генерує Gesture (з абсолютним t) з рядків `getevent -lt`. Лише перший палець."""
    down_t = start = None
    pos = [None, None]
    for line in lines:
        m = EVENT_RE.match(line.strip())
        if not m:
            continue
        t, _, _, code, value = m.groups()
        t = float(t)
        if code == "ABS_MT_TRACKING_ID":
            if value == "ffffffff":  # палець відпущено
                if down_t is not None and start and None not in pos:
                    x1, y1 = start
                    x2, y2 = pos[0] / x_max, pos[1] / y_max
                    yield Gesture(down_t, classify(x1, y1, x2, y2), x1, y1, x2, y2,
                                  max(int((t - down_t) * 1000), 1))
                down_t, start, pos = None, None, [None, None]
            elif down_t is None:
                down_t = t
        elif code in ("ABS_MT_POSITION_X", "ABS_MT_POSITION_Y") and down_t is not None:
            pos[code.endswith("Y")] = int(value, 16)
            if start is None and None not in pos:
                start = (pos[0] / x_max, pos[1] / y_max)


def touch_range(serial=None):
    """Максимальні значення ABS_MT_POSITION_X/Y сенсорного екрана (`getevent -lp`)."""
    out = subprocess.run([*adb_cmd(serial), "shell", "getevent", "-lp"],
                         capture_output=True, text=True, check=True).stdout
    xs = re.findall(r"ABS_MT_POSITION_X\s*:.*?max (\d+)", out)
    ys = re.findall(r"ABS_MT_POSITION_Y\s*:.*?max (\d+)", out)
    if not xs or not ys:
        raise RuntimeError("Не знайдено сенсорний екран у `getevent -lp`")
    return int(xs[0]), int(ys[0])


def record(path, level, serial=None, on_gesture=print):
    """Записує жести до Ctrl+C і зберігає маршрут."""
    x_max, y_max = touch_range(serial)
    proc = subprocess.Popen([*adb_cmd(serial), "shell", "getevent", "-lt"],
                            stdout=subprocess.PIPE, text=True)
    gestures = []
    try:
        for g in parse_getevent(proc.stdout, x_max, y_max):
            gestures.append(g)
            on_gesture(f"{g.t - gestures[0].t:7.2f}s  {g.action}")
    except KeyboardInterrupt:
        pass
    finally:
        proc.terminate()
    save(path, level, gestures)
    return gestures


def save(path, level, gestures):
    t0 = gestures[0].t if gestures else 0
    data = {"level": level,
            "gestures": [asdict(g) | {"t": round(g.t - t0, 3)} for g in gestures]}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1, ensure_ascii=False)


def load(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data["level"], [Gesture(**g) for g in data["gestures"]]


class RoutePlayer:
    """Видає жести, час яких настав. offset > 0 зсуває весь маршрут пізніше
    (якщо телефон завантажує рівень довше, ніж під час запису)."""

    def __init__(self, gestures, start_time, offset=0.0):
        self.gestures = gestures
        self.start = start_time
        self.offset = offset
        self.i = 0

    def due(self, now):
        out = []
        while self.i < len(self.gestures):
            g = self.gestures[self.i]
            # перший жест (тап «Грати») — одразу, решта зсуваються на offset
            t = g.t + (self.offset if self.i > 0 else 0)
            if self.start + t > now:
                break
            out.append(g)
            self.i += 1
        return out

    def next_in(self, now):
        """Через скільки секунд наступний жест (inf, якщо маршрут скінчився)."""
        if self.i >= len(self.gestures):
            return float("inf")
        return self.start + self.gestures[self.i].t + self.offset - now

    @property
    def finished(self):
        return self.i >= len(self.gestures)
