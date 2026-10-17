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
    side = min(h, w)  # розмір кнопки задається коротшою стороною: однаково у вертикальному й горизонтальному режимі
    roi = frame[:int(PAUSE_SEARCH[1] * h), :int(PAUSE_SEARCH[0] * w)]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    yellow = cv2.inRange(hsv, (18, 140, 170), (35, 255, 255))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(yellow)
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if not 0.045 * side <= bw <= 0.16 * side or not 0.6 <= bw / max(bh, 1) <= 1.6:
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


# Аналіз рейок і пікселні пороги (зсув за кадр тощо) підібрані під коротшу
# сторону 288 px (вертикальні відео Zero Mode); інші кадри зводимо до цього масштабу.
ANALYSIS_SIDE = 288


def analysis_scale(frame):
    h, w = frame.shape[:2]
    if min(h, w) == ANALYSIS_SIDE:
        return frame
    k = ANALYSIS_SIDE / min(h, w)
    return cv2.resize(frame, (round(w * k), round(h * k)), interpolation=cv2.INTER_AREA)


def is_landscape(frame):
    return frame.shape[1] > frame.shape[0]


def rail_profile(frame):
    frame = analysis_scale(frame)
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
OBJECTIVES_DY = (-0.46, -0.22)  # із запасом: макет картки трохи зсувається після виконання завдань
CREW_DY = (-0.17, -0.05)


# Горизонтальний режим (оновлений тур, відео інших каналів): картки перед забігом
# немає — PLAY тиснуть на головному екрані. Ті самі три завдання видно на екрані
# результатів після забігу (синій фон, жовта кнопка CONTINUE, плитки завдань у ряд).
LANDSCAPE_OBJECTIVES_DY = (-0.41, -0.20)
LANDSCAPE_OBJECTIVES_X = (0.36, 0.64)


def find_play_button(frame):
    """Рамка жовтої кнопки PLAY (x1, y1, x2, y2 у частках кадру) або None.

    У горизонтальному кадрі — кнопка CONTINUE екрана результатів (див. вище)."""
    h, w = frame.shape[:2]
    if w > h:
        return _find_results_button(frame)
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


def _find_results_button(frame):
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    if cv2.inRange(hsv[int(0.45 * h):int(0.75 * h)], (95, 120, 120), (115, 255, 255)).mean() / 255 < 0.5:
        return None   # не синій фон екрана результатів
    mask = cv2.inRange(hsv, (15, 150, 190), (32, 255, 255))
    mask[:int(0.6 * h)] = 0
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if 0.1 * w <= bw <= 0.3 * w and 0.04 * h <= bh <= 0.1 * h and area >= 0.6 * bw * bh:
            return x / w, y / h, (x + bw) / w, (y + bh) / h
    return None


def objectives_band(frame, play):
    """Блок трьох завдань рівня відносно знайденої кнопки (вертикальний чи горизонтальний кадр)."""
    if is_landscape(frame):
        return _band(frame, play, LANDSCAPE_OBJECTIVES_DY, LANDSCAPE_OBJECTIVES_X)
    return _band(frame, play, OBJECTIVES_DY)


def _band(frame, play, dy, x_range=(0.06, 0.94)):
    h, w = frame.shape[:2]
    cy = (play[1] + play[3]) / 2
    y0, y1 = max(int((cy + dy[0]) * h), 0), max(int((cy + dy[1]) * h), 1)
    return frame[y0:y1, int(x_range[0] * w):int(x_range[1] * w)]


def card_fingerprint(frame, play):
    """Нормований сірий «відбиток» блоку завдань: однаковий для того самого рівня
    незалежно від роздільної здатності телефону."""
    return band_fingerprint(objectives_band(frame, play))


def band_fingerprint(band):
    """Відбиток уже вирізаного блоку завдань (так зберігаються картки маршрутів)."""
    band = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
    v = cv2.resize(band, (128, 40), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    v -= v.mean()
    return v / (np.linalg.norm(v) + 1e-6)


def card_similarity(a, b):
    return float(np.dot(a, b))


def crew_crop(frame, play):
    """Ряд FEATURED CREW: обраний персонаж підсвічений рамкою."""
    return _band(frame, play, CREW_DY, (0.0, 1.0))


# ---------- текст і стан картки ----------
# Рівні з однаковими типами завдань відрізняються лише числами («Pop 19 toxic
# bubbles» / «Pop 13 toxic bubbles»), тож рівень упізнається за текстом (OCR).
# Кільця завдань між спробами зеленіють, а текст лишається тим самим.
_ocr = None
_CONFUSABLE = str.maketrans({"o": "0", "l": "1", "i": "1", "s": "5", "b": "8", "z": "2", "g": "9"})


def _get_ocr():
    global _ocr
    if _ocr is None:
        from rapidocr_onnxruntime import RapidOCR  # потрібен лише для режиму туру
        _ocr = RapidOCR()
    return _ocr


def tiles_text(band):
    """Текст завдань горизонтального екрана результатів: три плитки в ряд,
    кожну читаємо окремо (інакше OCR змішує колонки), лише нижню частину з
    підписом — без чисел на значках і лічильників."""
    import re
    h, w = band.shape[:2]
    words = []
    for k in range(3):
        tile = band[int(0.6 * h):, int(k * w / 3):int((k + 1) * w / 3)]
        scale = max(1.0, 360 / max(tile.shape[1], 1))
        tile = cv2.resize(tile, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        result, _ = _get_ocr()(tile)
        for r in sorted(result or [], key=lambda r: (r[0][0][1] // 40, r[0][0][0])):
            words.append(re.sub(r"(?<![A-Za-z])[Il|](?![A-Za-z])", "1", r[1]))  # окреме «I» у шрифті гри — це 1
    keep = [w.lower() for w in words if "/" not in w and not any(j in _norm_word(w) for j in _JUNK)]
    return "".join(ch for w in keep for ch in w if ch.isalnum())


def card_text(band):
    """Текст завдань з вирізки картки (без лічильників «0/8»), нормалізований."""
    _ocr = _get_ocr()
    scale = max(1.0, 900 / band.shape[1])  # дрібний текст з відео збільшуємо
    img = cv2.resize(band, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    result, _ = _ocr(img)
    words = [r[1] for r in sorted(result or [], key=lambda r: (r[0][0][1] // 40, r[0][0][0]))]
    # лишаємо тільки завдання: без лічильників «0/8», назви району, «Recommended
    # surfer level» тощо — ці рядки потрапляють у вирізку не на кожній спробі
    keep = [w.lower() for w in words if "/" not in w and not any(j in _norm_word(w) for j in _JUNK)]
    return "".join(ch for w in keep for ch in w if ch.isalnum())


# фрагменти, стійкі до помилок OCR («rocommended», «rewaros», «suferlevel»)
_JUNK = ("comm", "mend", "sufer", "surfer", "level", "rewar", "blvd", "southl", "docks", "delor",
         "undert", "phant", "spook", "normal", "hard", "featur", "crew")


def _norm_word(w):
    return "".join(ch for ch in w.lower() if ch.isalpha())


def is_level_card_text(text):
    """Схоже на завдання рівня: є числа й достатньо тексту (а не екран нагород)."""
    return len(text) >= 15 and sum(ch.isdigit() for ch in text) >= 2


def _numbers(s):
    """Множина чисел з тексту OCR. Літери, які OCR плутає з цифрами, поруч із
    цифрою читаються як цифри: 16ooo → 16000, pop1g → 19, b8000 → 88000,
    triggeri0 → trigger10 (у шрифті гри «1» схожа на «I»)."""
    import re
    fix = {"o": "0", "g": "9", "b": "8", "i": "1", "l": "1"}
    prev = None
    while prev != s:
        prev = s
        s = re.sub(r"(?<=\d)[og]|[obil](?=\d)", lambda m: fix[m.group()], s)
    return {str(int(n)) for n in re.findall(r"\d+", s) if int(n) > 0}  # «scoro12650» → 012650 → 12650


def common_numbers(a, b):
    """Скільки чисел завдань збігаються точно. Допуск на «одну іншу цифру» не
    годиться: тоді 12000, 14000 і 16000 «однакові», і бот грає чужий рівень."""
    return len(_numbers(a) & _numbers(b))


def _grams(s):
    s = "".join(ch for ch in s if ch.isalpha())
    return {s[i:i + 3] for i in range(len(s) - 2)}


def text_similarity(a, b):
    """Подібність текстів завдань, нечутлива до порядку рядків (OCR читає їх по-різному).

    Головне — числа завдань (рівні з однаковими типами завдань відрізняються саме
    ними): частка точно збіжних чисел від більшої множини. Збіг трибуквених
    фрагментів слів лише уточнює оцінку.
    """
    if not a or not b:
        return 0.0
    ga, gb = _grams(a), _grams(b)
    letters = 2 * len(ga & gb) / (len(ga) + len(gb)) if ga and gb else 0.0
    na, nb = _numbers(a), _numbers(b)
    if not na or not nb:
        return 0.3 * letters
    nums = len(na & nb) / max(len(na), len(nb))
    return 0.7 * nums + 0.3 * letters


def card_state(band):
    """Які з трьох завдань уже виконані: зелене кільце над текстом завдання."""
    h, w = band.shape[:2]
    hsv = cv2.cvtColor(band[: int(0.7 * h)], cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, (40, 110, 110), (85, 255, 255))
    return tuple(int(green[:, int(k * w / 3):int((k + 1) * w / 3)].mean() / 255 > 0.04) for k in range(3))


# ---------- відскік на Bubblegum ----------
# Свайп униз у стрибку з Bubblegum дає різкий рожевий спалах навколо персонажа
# і відскік угору. Рожевого в City багато (бульбашки, Phantom Park), але воно
# наростає поступово, а спалах — за 0,1 с.

def magenta_ratio(frame):
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame[int(0.2 * h):int(0.9 * h), int(0.15 * w):int(0.85 * w)], cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, (140, 90, 170), (172, 255, 255)).mean() / 255


def bounce_times(times, ratios, jump=0.12, window=0.1, refractory=0.6):
    """Моменти різкого рожевого спалаху (приріст ≥ jump за window секунд)."""
    out, j = [], 0
    for i, (t, r) in enumerate(zip(times, ratios)):
        while times[j] < t - window:
            j += 1
        if r - min(ratios[j:i + 1]) >= jump and (not out or t - out[-1] > refractory):
            out.append(t)
    return out
