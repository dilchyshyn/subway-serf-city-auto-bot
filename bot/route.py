"""Запис і відтворення маршрутів рівнів City Tour.

Рівні City Tour мають фіксовану розкладку, тому прохід, у якому виконано
всі 3 завдання, можна записати й повторити. Є два джерела маршруту:
  * запис власного проходу: дотики з `adb shell getevent -lt`, таймінг
    відносно першого дотику (тапу «Грати»), start = "tap";
  * відео чужого проходу: дії відновлюються з руху персонажа
    (actions_from_track), таймінг відносно старту забігу, start = "manual".
"""
import json
import re
import subprocess
from dataclasses import asdict, dataclass

import numpy as np

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


def save(path, level, gestures, start="tap", rebase=True, meta=None):
    """start: "tap" — маршрут починається з тапу «Грати» (свій запис);
    "hud" — t=0 у момент появи HUD забігу (з відео, бот синхронізується сам);
    "manual" — t=0 за натисканням Enter."""
    t0 = gestures[0].t if gestures and rebase else 0
    data = {"level": level, "start": start, **(meta or {}),
            "gestures": [asdict(g) | {"t": round(g.t - t0, 3)} for g in gestures]}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1, ensure_ascii=False)


def load(path):
    """Повертає (назва рівня, жести, режим старту)."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data["level"], [Gesture(**g) for g in data["gestures"]], data.get("start", "tap")


# ---------- маршрут з відео ----------

def merge_boxes(boxes):
    """Об'єднує рамки персонажа, що перекриваються з найвпевненішою, в одну.

    Детектор інколи дає на персонажа кілька рамок (усе тіло, лише торс), і
    якщо брати лише найвпевненішу, її низ «стрибає» від кадру до кадру.
    boxes — список (conf, (x1, y1, x2, y2)).
    """
    if not boxes:
        return None
    _, best = max(boxes, key=lambda b: b[0])
    group = [b for _, b in boxes if b[0] < best[2] and best[0] < b[2] and b[1] < best[3] and best[1] < b[3]]
    return (min(b[0] for b in group), min(b[1] for b in group),
            max(b[2] for b in group), max(b[3] for b in group))


def _lane_position(xs, iters=20, merge=0.06, min_share=0.1):
    """Функція x -> положення в смугах (дробове: 0.5 = посередині між смугами 0 і 1).

    1D k-means на 3 кластери. Кластери з малою часткою кадрів (проміжні
    положення під час переходу) відкидаються, близькі центри зливаються:
    якщо гравець бігав лише у двох смугах, третій кластер інакше розрізав би
    одну смугу навпіл. Відлік від крайнього центру з кроком між смугами.
    """
    xs = np.asarray(xs)
    centers = np.quantile(xs, [0.1, 0.5, 0.9])
    for _ in range(iters):
        labels = np.argmin(np.abs(xs[:, None] - centers[None, :]), axis=1)
        centers = np.array([xs[labels == k].mean() if (labels == k).any() else centers[k] for k in range(3)])
    labels = np.argmin(np.abs(xs[:, None] - centers[None, :]), axis=1)
    major = [c for k, c in enumerate(centers) if (labels == k).mean() >= min_share]
    merged = []
    for c in np.sort(major):
        if merged and c - merged[-1] < merge:
            merged[-1] = (merged[-1] + c) / 2
        else:
            merged.append(c)
    if len(merged) == 1:
        return lambda x: 0.0
    step = min(np.diff(merged))
    return lambda x: (x - merged[0]) / step


def _median3(values):
    """Медіанний фільтр на 3 точки: прибирає одиночні викиди детектора."""
    v = np.asarray(values, dtype=float)
    if len(v) < 3:
        return v
    padded = np.concatenate([v[:1], v, v[-1:]])
    return np.median(np.stack([padded[:-2], padded[1:-1], padded[2:]]), axis=0)


def with_board_taps(gestures, board_cfg):
    """Тапи по капсулі ховерборда (праворуч під лічильником монет) на старті й
    далі з інтервалом: гравці у відео зазвичай вмикають дошку одразу, а вона
    рятує від одного зіткнення й міняє стрибки — без неї маршрут розходиться.
    Якщо дошки немає, тап у порожнє місце нічого не робить. Тапи не ставимо
    ближче ніж gap до свайпів, щоб не зірвати жест."""
    if not gestures:
        return gestures
    x, y = board_cfg["tap"]
    gap, end = board_cfg["gap"], gestures[-1].t
    swipes = [g.t for g in gestures if g.action != "tap"]
    taps, t = [], board_cfg["start"]
    while t <= end:
        shifted = t
        while any(abs(shifted - s) < gap for s in swipes):
            shifted += 0.1
        taps.append(Gesture(round(shifted, 3), "tap", x, y, x, y, 40))
        t += board_cfg["every"]
    return sorted(gestures + taps, key=lambda g: g.t)


def _swipe(action, t, swipe_cfg):
    dx, dy = {"left": (-1, 0), "right": (1, 0), "jump": (0, -1), "roll": (0, 1)}[action]
    ox, oy = swipe_cfg["origin"]
    length = swipe_cfg["length"]
    return Gesture(round(max(t, 0.0), 3), action, ox, oy, ox + dx * length, oy + dy * length,
                   swipe_cfg["duration_ms"])


def actions_from_track(track, swipe_cfg, lead=0.1, stable=2, jump_rise=0.08, roll_shrink=0.7,
                       snap=0.3, lanes=True, stomp_max_air=None, stomp_fall=3.0, stomp_shrink=0.6,
                       ground_window=1.5, jumps_log=None):
    """Відновлює свайпи з траєкторії персонажа у відео.

    track — список (t, (x1, y1, x2, y2) у частках кадру або None, якщо персонажа не видно).
    Зміна смуги — коли персонаж `stable` кадрів поспіль ближче ніж `snap` кроку
    до центру нової смуги (гістерезис: проміжні положення під час руху не рахуються);
    стрибок — низ рамки різко піднявся; пірнання — рамка стала нижчою.
    lead віднімається від моменту, коли рух помітно на кадрі: свайп був трохи раніше.
    lanes=False — лише стрибки й пірнання (у City смуги надійніше визначати за
    зсувом рейок, див. bot/video.py, бо камера йде за персонажем вбік).
    stomp_max_air — stomp (свайп униз у повітрі) визнається, лише якщо політ
    коротший за стільки секунд І падіння різке (stomp_fall, частки кадру/с) або
    рамка «згорнулась» (stomp_shrink від звичної висоти). Тоді "roll" на вершині.
    jumps_log — список, куди дописуються (старт, приземлення) кожного стрибка.
    """
    seen = [(t, b) for t, b in track if b is not None]
    if len(seen) < stable + 1:
        return []
    ts = [t for t, _ in seen]
    cxs = _median3([(b[0] + b[2]) / 2 for _, b in seen])
    bottoms = _median3([b[3] for _, b in seen])
    heights = _median3([b[3] - b[1] for _, b in seen])
    pos_of = _lane_position(cxs)

    out = []
    lane = cand = None
    cand_len = 0
    move_start = None   # коли персонаж почав зсуватися зі смуги
    in_air = rolling = False
    jump_t = apex_t = None
    apex_bottom = 1.0
    hist = []  # (t, bottom, height, у повітрі) за останні ground_window с

    for t, cx, bottom, height in zip(ts, cxs, bottoms, heights):
        # --- смуга ---
        pos = pos_of(cx) if lanes else 0.0
        near = int(round(pos))
        if lane is not None and abs(pos - lane) > snap and move_start is None:
            move_start = t
        if abs(pos - near) <= snap:
            cand_len = cand_len + 1 if near == cand else 1
            cand = near
            if cand_len == stable:
                if lane is not None and near != lane:
                    step = "right" if near > lane else "left"
                    t0 = move_start if move_start is not None else t
                    for k in range(abs(near - lane)):  # перескок через смугу = два свайпи
                        out.append(_swipe(step, t0 - lead + k * 0.1, swipe_cfg))
                lane, move_start = near, None
            elif near == lane:
                move_start = None

        # --- стрибок, stomp і пірнання ---
        # Земля — 85-й перцентиль низу рамки за останні ground_window с: стійко до
        # пропусків детектора й підлаштовується під дах вагона.
        hist = [h for h in hist if t - h[0] <= ground_window]
        hist.append((t, bottom, height, in_air))
        ground = float(np.percentile([h[1] for h in hist], 85))
        elev = ground - bottom
        if not in_air:
            if elev > jump_rise:
                out.append(_swipe("jump", t - lead - 0.05, swipe_cfg))
                in_air, jump_t, apex_t, apex_bottom = True, t, t, bottom
                fall_v, min_h, prev = 0.0, height, (t, bottom)
            else:
                usual_h = float(np.median([h[2] for h in hist if not h[3]] or [height]))
                if not rolling and height < roll_shrink * usual_h:
                    out.append(_swipe("roll", t - lead, swipe_cfg))
                    rolling = True
                elif rolling and height > 0.85 * usual_h:
                    rolling = False
        else:
            if bottom < apex_bottom:
                apex_t, apex_bottom = t, bottom          # ще набирає висоту
            else:                                         # падає: швидкість і «згортання»
                if t > prev[0]:
                    fall_v = max(fall_v, (bottom - prev[1]) / (t - prev[0]))
                min_h = min(min_h, height)
            prev = (t, bottom)
            if elev < jump_rise * 0.4 or t - jump_t > 2.5:
                in_air = False
                if jumps_log is not None:
                    jumps_log.append((jump_t, t))
                usual_h = float(np.median([h[2] for h in hist if not h[3]] or [height]))
                # stomp вмикаємо лише за кількох ознак разом: хибний удар униз посеред
                # стрибка через вагон коштує всього рівня, а пропущений — однієї зірки
                if stomp_max_air and t - jump_t < stomp_max_air and (
                        fall_v >= stomp_fall or min_h <= stomp_shrink * usual_h):
                    out.append(_swipe("roll", apex_t - lead, swipe_cfg))

    return sorted(out, key=lambda g: g.t)


class RoutePlayer:
    """Видає жести, час яких настав. offset > 0 зсуває весь маршрут пізніше
    (якщо телефон завантажує рівень довше, ніж під час запису).
    first_immediate: перший жест (тап «Грати») виконати одразу, без offset."""

    def __init__(self, gestures, start_time, offset=0.0, first_immediate=True):
        self.gestures = gestures
        self.start = start_time
        self.offset = offset
        self.first_immediate = first_immediate
        self.i = 0

    def due(self, now):
        out = []
        while self.i < len(self.gestures):
            g = self.gestures[self.i]
            immediate = self.first_immediate and self.i == 0
            t = g.t + (0 if immediate else self.offset)
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
