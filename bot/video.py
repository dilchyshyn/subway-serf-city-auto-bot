"""Аналіз відео проходжень Subway Surfers City без нейромережі.

* hud_visible — чи йде забіг: жовта кнопка паузи в лівому верхньому куті
  є лише під час забігу (у меню, на картці рівня й екрані результатів її немає).
  Той самий тест бот застосовує до свого екрана, щоб синхронізувати маршрут.
* segment_runs — межі забігів (рівнів) у відео за HUD.
* rail_profile / profile_shift / lane_events — зміни смуги. Камера йде за
  персонажем вбік, тому персонаж завжди біля центру, а зміна смуги видна як
  горизонтальний зсув рейок під ним. Профіль вертикальних країв (Sobel-x)
  не чутливий до руху вперед, бо шпали горизонтальні.
"""
import cv2
import numpy as np

# зона кнопки паузи, частки кадру (x0, x1, y0, y1)
PAUSE_ROI = (0.02, 0.09, 0.012, 0.055)
RAIL_BAND = (0.80, 0.97)


def hud_visible(frame, min_yellow=0.05, max_dark=0.04):
    h, w = frame.shape[:2]
    x0, x1, y0, y1 = PAUSE_ROI
    roi = cv2.cvtColor(frame[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)], cv2.COLOR_BGR2HSV)
    yellow = cv2.inRange(roi, (18, 140, 170), (35, 255, 255)).mean() / 255
    dark = cv2.inRange(roi, (0, 0, 0), (180, 255, 70)).mean() / 255
    return yellow >= min_yellow and dark <= max_dark


def segment_runs(times, flags, min_len=5.0, max_gap=1.5):
    """Безперервні проміжки з HUD довші за min_len секунд; дірки до max_gap склеюються."""
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
