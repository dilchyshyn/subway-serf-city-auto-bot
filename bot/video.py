"""Аналіз відео проходжень Subway Surfers City без нейромережі.

* hud_visible — чи йде забіг: жовта кнопка паузи з двома смужками в лівому
  верхньому куті є лише під час забігу (у меню, на картці рівня й екрані результатів її немає).
  Той самий тест бот застосовує до свого екрана, щоб синхронізувати маршрут.
* segment_runs — межі забігів (рівнів) у відео за HUD.
* rail_profile / profile_shift / lane_events — зміни смуги. Камера йде за
  персонажем вбік, тому персонаж завжди біля центру, а зміна смуги видна як
  горизонтальний зсув рейок під ним. Профіль вертикальних країв (Sobel-x)
  не чутливий до руху вперед, бо шпали горизонтальні.
"""
import cv2
import numpy as np

# де шукати кнопку паузи: лівий верхній кут, частки кадру (x1, y1).
# Відступ зверху буває різним (статус-бар, обрізка запису), тож зона з запасом.
PAUSE_SEARCH = (0.15, 0.13)
RAIL_BAND = (0.80, 0.97)


def hud_visible(frame):
    """Чи видно кнопку паузи: жовтий майже квадратний блок з темними смужками всередині."""
    h, w = frame.shape[:2]
    roi = frame[:int(PAUSE_SEARCH[1] * h), :int(PAUSE_SEARCH[0] * w)]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    yellow = cv2.inRange(hsv, (18, 140, 170), (35, 255, 255))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(yellow)
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if not 0.06 * w <= bw <= 0.16 * w or not 0.6 <= bw / max(bh, 1) <= 1.6:
            continue
        if area < 0.35 * bw * bh:
            continue
        inner = hsv[y + bh // 4:y + 3 * bh // 4, x + bw // 4:x + 3 * bw // 4]
        dark = cv2.inRange(inner, (0, 0, 0), (180, 255, 90)).mean() / 255
        if 0.15 <= dark <= 0.7:   # дві темні смужки «паузи»
            return True
    return False


def segment_runs(times, flags, min_len=5.0, max_gap=4.0):
    """Безперервні проміжки з HUD довші за min_len секунд.

    Дірки до max_gap склеюються: посеред забігу HUD буває на 2-3 с закритий
    ефектами, а між рівнями (результати, картка, завантаження) минає 15+ с.
    """
    runs, start, last = [], None, None
    for t, on in zip(times, flags):
        if on:
            if start is None:
                start = t
            elif t - last > max_gap:
                if last - start >= min_len:
                    runs.append((start, last))
                start = t
            last = t
    if start is not None and last - start >= min_len:
        runs.append((start, last))
    return runs


def rail_profile(frame):
    h = frame.shape[0]
    band = cv2.cvtColor(frame[int(RAIL_BAND[0] * h):int(RAIL_BAND[1] * h)], cv2.COLOR_BGR2GRAY)
    gx = np.abs(cv2.Sobel(band.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)).mean(axis=0)
    gx -= gx.mean()
    return gx / (np.linalg.norm(gx) + 1e-6)


def profile_shift(prev, cur, max_shift=25, margin=30):
    """Зсув (пікселі) профілю cur відносно prev; додатний = рейки поїхали вправо."""
    a = prev[margin:-margin]
    return max(range(-max_shift, max_shift + 1),
               key=lambda s: float(np.dot(np.roll(cur, -s)[margin:-margin], a)))


def motion_bursts(times, shifts, min_speed=3, max_gap=2):
    """Сплески бічного руху: (t_start, t_end, сумарний зсув)."""
    out, i, n = [], 0, len(shifts)
    while i < n:
        if abs(shifts[i]) < min_speed:
            i += 1
            continue
        j, k, gap = i, i, 0
        while k < n and gap <= max_gap:
            if abs(shifts[k]) >= min_speed:
                j, gap = k, 0
            else:
                gap += 1
            k += 1
        out.append((times[i], times[j], float(np.sum(shifts[i:j + 1]))))
        i = j + 1
    return out


def lane_events(bursts, width, start_lane=1, min_frac=0.25):
    """Сплески -> свайпи [(t, "left"/"right")] і моменти відкинутих сплесків.

    Крок «одна смуга» калібрується по самому відео: медіана модулів великих
    сплесків. Рейки їдуть вправо, коли гравець іде вліво.

    Деякі сплески хибні (стрибок на дах вагона, Bubblegum, ефекти). Гравець
    завжди лишається в смугах 0..2, тому динамічним програмуванням обираємо
    найдешевший набір відкинутих сплесків, з яким маршрут можливий. Ціна
    відкидання — впевненість сплеску: наскільки його зсув близький до цілого
    числа смуг.
    """
    big = [abs(s) for _, _, s in bursts if abs(s) >= min_frac * width]
    if not big:
        return [], []
    step = float(np.median(big))
    events = []  # (t, кількість смуг зі знаком: + вправо, впевненість)
    for t0, _, s in bursts:
        ratio = abs(s) / step
        n = int(round(ratio))
        if n == 0:
            continue
        conf = max(0.05, 1 - 2 * abs(ratio - n))
        events.append((t0, -n if s > 0 else n, conf))

    # dp[lane] = (ціна, список рішень keep/drop)
    inf = float("inf")
    dp = {lane: (inf, []) for lane in range(3)}
    dp[start_lane] = (0.0, [])
    for _, d, conf in events:
        nxt = {lane: (inf, []) for lane in range(3)}
        for lane, (cost, path) in dp.items():
            if cost == inf:
                continue
            if cost + conf < nxt[lane][0]:                       # відкинути сплеск
                nxt[lane] = (cost + conf, path + [False])
            if 0 <= lane + d <= 2 and cost < nxt[lane + d][0]:   # прийняти
                nxt[lane + d] = (cost, path + [True])
        dp = nxt
    _, best = min(dp.values(), key=lambda v: v[0])

    out, dropped = [], []
    for (t0, d, _), keep in zip(events, best):
        if not keep:
            dropped.append(t0)
            continue
        for k in range(abs(d)):
            out.append((t0 + k * 0.1, "right" if d > 0 else "left"))
    return out, dropped


# ---------- картка рівня ----------
# Відносно центру кнопки PLAY (частки висоти кадру): завдання рівня й обраний
# персонаж з FEATURED CREW. Номер рівня не беремо: після перемішування рівнів
# у турі він може змінитися, а завдання лишаються.
OBJECTIVES_DY = (-0.42, -0.25)
CREW_DY = (-0.17, -0.05)


def find_play_button(frame):
    """Рамка жовтої кнопки PLAY (x1, y1, x2, y2 у частках кадру) або None."""
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (15, 150, 190), (32, 255, 255))
    mask[:int(0.5 * h)] = 0
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    best = None
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if 0.5 * w <= bw <= 0.8 * w and 0.03 * h <= bh <= 0.08 * h and area >= 0.6 * bw * bh:
            if best is None or y > best[1]:  # найнижча з придатних
                best = (x, y, bw, bh)
    if best is None:
        return None
    x, y, bw, bh = best
    return x / w, y / h, (x + bw) / w, (y + bh) / h


def _band(frame, play, dy, x_range=(0.06, 0.94)):
    h, w = frame.shape[:2]
    cy = (play[1] + play[3]) / 2
    y0, y1 = max(int((cy + dy[0]) * h), 0), max(int((cy + dy[1]) * h), 1)
    return frame[y0:y1, int(x_range[0] * w):int(x_range[1] * w)]


def card_fingerprint(frame, play):
    """Нормований сірий «відбиток» блоку завдань: однаковий для того самого рівня
    незалежно від роздільної здатності телефону."""
    band = cv2.cvtColor(_band(frame, play, OBJECTIVES_DY), cv2.COLOR_BGR2GRAY)
    v = cv2.resize(band, (128, 40), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    v -= v.mean()
    return v / (np.linalg.norm(v) + 1e-6)


def card_similarity(a, b):
    return float(np.dot(a, b))


def crew_crop(frame, play):
    """Ряд FEATURED CREW: обраний персонаж підсвічений рамкою."""
    return _band(frame, play, CREW_DY, (0.0, 1.0))
