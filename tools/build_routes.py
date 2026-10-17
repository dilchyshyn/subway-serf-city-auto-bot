"""Маршрути рівнів City Tour з відео проходжень (без участі людини).

  python tools/build_routes.py "https://www.youtube.com/watch?v=..."      # одне відео
  python tools/build_routes.py --channel https://www.youtube.com/@zero_mode_yt   # увесь канал
  python tools/build_routes.py rec/videos/docks.mp4 --limit-runs 1         # локальний файл

Для кожного відео за один прохід:
  * межі забігів — за кнопкою паузи в HUD (bot/video.py);
  * зміни смуги — за зсувом рейок, з фільтром неможливих переходів;
  * стрибки й пірнання — за рамкою персонажа (модель з класом player);
  * картка рівня перед забігом — щоб бот упізнав рівень, хоч би де той
    стояв у турі, і кадр FEATURED CREW — яким персонажем грали.
Результат: routes/<назва>.json + routes/cards/<назва>.png + routes/crew/<назва>.png.
Аналіз кожного відео кешується (rec/videos/*.analysis.npz), тож після
покращення алгоритмів маршрути перебудовуються за секунди.

Обмеження: stomp (свайп униз у повітрі) з відео поки не відновлюється.
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import av
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from bot import load_config, route, video  # noqa: E402

CITY_TITLE = re.compile(r"subway surfers city|southline|sunrise blvd|undertr?acks|the docks|delorean|phantom park|spooky",
                        re.IGNORECASE)


def slug(text):
    text = re.sub(r"subway surfers city", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return text[:60] or "video"


def list_channel(url):
    out = subprocess.run([sys.executable, "-m", "yt_dlp", "--flat-playlist", "--print", "%(id)s\t%(title)s",
                          url.rstrip("/") + "/videos"], capture_output=True, text=True, check=True).stdout
    items = [line.split("\t", 1) for line in out.splitlines() if "\t" in line]
    return [(f"https://www.youtube.com/watch?v={vid}", title) for vid, title in items if CITY_TITLE.search(title)]


def download(url, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    vid = url.rsplit("=", 1)[-1]
    existing = list(out_dir.glob(f"{vid}.*"))
    if existing:
        return existing[0]
    # горизонтальні відео беремо не ширше 854 px: для аналізу досить, а трафіку вдвічі менше
    subprocess.run([sys.executable, "-m", "yt_dlp", "-q", "--no-warnings", "-f",
                    "bv*[height<=720][width<=900][ext=mp4]/bv*[height<=720][width<=900]/bv*[height<=720]", "-o", str(out_dir / "%(id)s.%(ext)s"), url],
                   check=True)
    return next(out_dir.glob(f"{vid}.*"))


def video_info(url):
    """(назва, дата завантаження YYYYMMDD) відео YouTube."""
    try:
        out = subprocess.run([sys.executable, "-m", "yt_dlp", "--no-warnings", "--skip-download", "--print",
                              "%(upload_date)s\t%(title)s", url], capture_output=True, text=True, check=True).stdout
        date, title = out.strip().split("\t", 1)
        return title or url, date if date.isdigit() else ""
    except (subprocess.CalledProcessError, ValueError):
        return url, ""


def hires_frame(url, t, videos_dir):
    """Кадр у 1080p на момент t: дрібний текст завдань у 480p читається з
    помилками, тож для кожного екрана результатів докачуємо лише секунду відео."""
    vid = url.rsplit("=", 1)[-1]
    cache = videos_dir / "hires" / f"{vid}_{t:.1f}.png"
    if cache.exists():
        return cv2.imread(str(cache))
    cache.parent.mkdir(parents=True, exist_ok=True)
    clip = cache.with_suffix(".mp4")
    try:
        subprocess.run([sys.executable, "-m", "yt_dlp", "-q", "--no-warnings", "-f",
                        "bv*[height<=1080][ext=mp4]/bv*[height<=1080]", "--download-sections",
                        f"*{t:.2f}-{t + 0.5:.2f}", "-o", str(clip), url], check=True, timeout=300)
        img = None
        with av.open(str(clip)) as c:
            for f in c.decode(video=0):  # кліп починається з ключового кадру до t — беремо останній
                img = f.to_ndarray(format="bgr24")
    except (subprocess.SubprocessError, OSError, av.FFmpegError):
        return None
    finally:
        clip.unlink(missing_ok=True)
    if img is not None:
        cv2.imwrite(str(cache), img)
    return img


def player_box(detector, img):
    """Рамка персонажа (частки кадру). У горизонтальному кадрі шукаємо лише в
    центральній смузі: персонаж завжди біля центру, а краї кадру дають хибні
    рамки. x нормуємо на ширину смуги, y — на висоту кадру, як у вертикальному."""
    h, w = img.shape[:2]
    x0, x1 = 0, w
    if w > h:
        x0, x1 = int(w / 2 - 0.3 * h), int(w / 2 + 0.3 * h)
    cw = x1 - x0
    return route.merge_boxes([(d.conf, (d.x1 / cw, d.y1 / h, d.x2 / cw, d.y2 / h))
                              for d in detector.detect(img[:, x0:x1]) if d.cls == "player"])


def analyse(path, detector, det_every):
    """Один прохід по відео: HUD, зсув рейок, рамка персонажа, картки рівнів.

    Картки зберігаються по одній на кожен показ (не частіше ніж раз на 2 с).
    """
    times, hud, shifts, boxes, cards = [], [], [], [], []
    prev_profile = None
    with av.open(str(path)) as container:
        for i, frame in enumerate(container.decode(video=0)):
            if frame.time is None:
                continue
            img = frame.to_ndarray(format="bgr24")
            h, w = img.shape[:2]
            on = video.hud_visible(img)
            times.append(frame.time)
            hud.append(on)
            shift, box = 0, None
            if on:
                profile = video.rail_profile(img)
                if prev_profile is not None:
                    shift = video.profile_shift(prev_profile, profile)
                prev_profile = profile
                if i % det_every == 0:
                    box = player_box(detector, img)
            else:
                prev_profile = None
                play = video.find_play_button(img) if not cards or frame.time - cards[-1][0] > 2 else None
                if play is not None:
                    cards.append((frame.time, img, play))
            shifts.append(shift)
            boxes.append(box)
    return np.array(times), hud, np.array(shifts, float), boxes, cards


def analyse_cached(path, detector, det_every):
    """analyse() з кешем поруч із відео: повторна побудова маршрутів без декодування."""
    cache = Path(path).with_suffix(".analysis.npz")
    if cache.exists():
        d = np.load(cache, allow_pickle=True)
        cards = [(t, img, tuple(play)) for t, img, play in d["cards"]]
        return d["times"], list(d["hud"]), d["shifts"], list(d["boxes"]), cards
    times, hud, shifts, boxes, cards = analyse(path, detector, det_every)
    np.savez_compressed(cache, times=times, hud=np.array(hud), shifts=shifts,
                        boxes=np.array(boxes, dtype=object),
                        cards=np.array([(t, img, play) for t, img, play in cards], dtype=object))
    return times, hud, shifts, boxes, cards


def bounces_cached(path):
    """Рожевість кадру для пошуку відскоків на Bubblegum; окремий кеш, без детектора."""
    cache = Path(path).with_suffix(".bounce.npz")
    if cache.exists():
        d = np.load(cache)
        return d["times"], d["ratios"]
    times, ratios = [], []
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            if frame.time is not None:
                times.append(frame.time)
                ratios.append(video.magenta_ratio(frame.to_ndarray(format="bgr24")))
    np.savez_compressed(cache, times=np.array(times), ratios=np.array(ratios))
    return np.array(times), np.array(ratios)


def apply_bounces(gestures, bounces, cfg, lead=0.12):
    """Свайп униз перед кожним відскоком; «стрибок» одразу після спалаху — це сам відскік."""
    out = [g for g in gestures
           if not (g.action == "jump" and any(-0.15 <= g.t - b <= 0.25 for b in bounces))]
    for b in bounces:
        if not any(g.action == "roll" and abs(g.t - (b - lead)) < 0.3 for g in out):
            out.append(route._swipe("roll", b - lead, cfg["swipe"]))
    return sorted(out, key=lambda g: g.t)


def build(path, title, out_dir, detector, cfg, det_every=2, source=None, uploaded=""):
    videos_dir = Path(path).parent
    times, hud, shifts, boxes, cards = analyse_cached(path, detector, det_every)
    b_times, b_ratios = bounces_cached(path)
    runs = video.segment_runs(times, hud)
    with av.open(str(path)) as c:
        width, height = c.streams.video[0].width, c.streams.video[0].height
    landscape = width > height
    # зсуви рейок рахуються в масштабі video.ANALYSIS_SIDE по коротшій стороні
    width = video.ANALYSIS_SIDE
    results = []
    for k, (t0, t1) in enumerate(runs):
        sel = (times >= t0) & (times <= t1)
        rt, rs = times[sel], shifts[sel]
        bursts = video.motion_bursts(list(rt), rs)
        lane_swipes, dropped = video.lane_events(bursts, width)
        gestures = [route._swipe(a, t - t0 - 0.1, cfg["swipe"]) for t, a in lane_swipes]
        track = [(t - t0, b) for t, b, s in zip(times, boxes, sel) if s and b is not None]
        gestures += route.actions_from_track(track, cfg["swipe"], lanes=False,
                                             stomp_max_air=cfg["video"]["stomp_max_air"],
                                             stomp_fall=cfg["video"]["stomp_fall"],
                                             stomp_shrink=cfg["video"]["stomp_shrink"])
        sel_b = (b_times >= t0) & (b_times <= t1)
        bounces = [t - t0 for t in video.bounce_times(list(b_times[sel_b]), list(b_ratios[sel_b]))]
        gestures = apply_bounces(gestures, bounces, cfg)

        name = f"{slug(title)}_{k + 1}"
        card, card_text = None, None
        # вертикальне відео: остання справжня картка рівня перед забігом;
        # горизонтальне: перший екран результатів після забігу (там ті самі завдання)
        candidates = ([c for c in cards if t1 < c[0] <= t1 + 30] if landscape else
                      [c for c in reversed(cards) if t0 - 90 <= c[0] < t0])
        for c in candidates:
            if landscape:
                hi = hires_frame(source, c[0], videos_dir) if source and source.startswith("http") else None
                play = video.find_play_button(hi) if hi is not None else None
                if play is not None:
                    c = (c[0], hi, play)
                text = video.tiles_text(video.objectives_band(c[1], c[2]))
            else:
                text = video.card_text(video.objectives_band(c[1], c[2]))
            if video.is_level_card_text(text):  # екран нагород теж має жовту кнопку
                card, card_text = c, text
                break
        meta = {"source": source, "title": title, "uploaded": uploaded, "run": [round(t0, 2), round(t1, 2)],
                "dropped_lane_events": [round(t - t0, 2) for t in dropped],
                "bounces": [round(t, 2) for t in bounces],
                "player_frames": len(track)}
        if card:
            _, img, play = card
            (out_dir / "cards").mkdir(parents=True, exist_ok=True)
            (out_dir / "crew").mkdir(parents=True, exist_ok=True)
            band = video.objectives_band(img, play)
            cv2.imwrite(str(out_dir / "cards" / f"{name}.jpg"), band, [cv2.IMWRITE_JPEG_QUALITY, 85])
            meta["card_text"] = card_text
            meta["card_state"] = list(video.card_state(band))
            # для впізнавання рівня потрібен лише блок завдань; повні кадри в репозиторій не кладемо
            meta["card"] = f"cards/{name}.jpg"
            if not landscape:  # у горизонтальному режимі FEATURED CREW перед забігом не показують
                cv2.imwrite(str(out_dir / "crew" / f"{name}.jpg"), video.crew_crop(img, play),
                            [cv2.IMWRITE_JPEG_QUALITY, 85])
                meta["crew"] = f"crew/{name}.jpg"
        route.save(out_dir / f"{name}.json", name, gestures, start="hud", rebase=False, meta=meta)
        results.append((name, len(gestures), t1 - t0, len(dropped), card is not None))
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("sources", nargs="*", help="посилання на відео або локальні файли")
    p.add_argument("--channel", help="канал YouTube: обробити всі відео Subway Surfers City")
    p.add_argument("--model", help="YOLO-модель з класом player (за замовчуванням models/player.pt)")
    p.add_argument("--out", default=str(ROOT / "routes"))
    p.add_argument("--videos", default=str(ROOT / "rec" / "videos"), help="куди качати відео")
    p.add_argument("--det-every", type=int, default=2, help="шукати персонажа на кожному N-му кадрі")
    args = p.parse_args()

    from bot.detector import YoloDetector
    cfg = load_config()
    v = cfg["vision"]
    detector = YoloDetector(args.model or str(ROOT / "models" / "player.pt"), v["conf"], cfg["video"]["imgsz"])

    items = [(s, *(video_info(s) if s.startswith("http") else (Path(s).stem, ""))) for s in args.sources]
    if args.channel:
        items += [(url, title, "") for url, title in list_channel(args.channel)]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = []
    for i, (src, title, uploaded) in enumerate(items, 1):
        path = download(src, Path(args.videos)) if src.startswith("http") else Path(src)
        print(f"[{i}/{len(items)}] {title}")
        for name, n, dur, dropped, has_card in build(path, title, out, detector, cfg, args.det_every, src, uploaded):
            flag = "" if has_card else "  ⚠️ без картки"
            flag += f"  ⚠️ відкинуто подій смуги: {dropped}" if dropped else ""
            print(f"    {name}: {n} жестів, {dur:.0f} с{flag}")
            summary.append(name)
    (out / "index.json").write_text(json.dumps(sorted(summary), indent=1, ensure_ascii=False))
    print(f"Готово: {len(summary)} маршрутів у {out}")


if __name__ == "__main__":
    main()
