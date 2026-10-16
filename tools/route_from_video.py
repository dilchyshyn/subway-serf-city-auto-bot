"""Маршрут рівня City Tour з відео чужого проходження (експериментально).

  python tools/route_from_video.py video.mp4 docks_07 --start 4.2 --end 71.5
  python tools/route_from_video.py "https://youtube.com/watch?v=..." docks_07 --start 4.2 --end 71.5 --crop 0.33,0,0.34,1

--start — секунда відео, коли персонаж починає бігти; --end — фініш.
Модель (з класом player) знаходить персонажа на кожному кадрі, а з його
руху відновлюються свайпи: зміна смуги, стрибок, пірнання. Тапи (ховерборд)
і stomp у повітрі з відео не відновлюються.

Бери відео, де виконано всі 3 завдання рівня: бот повторить саме цей прохід.
Відтворення: python main.py --route routes/docks_07.json
"""
import argparse
import sys
import tempfile
from pathlib import Path

import av

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from bot import load_config, route  # noqa: E402
from bot.detector import YoloDetector  # noqa: E402
from extract_frames import download  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("source", help="файл відео або посилання")
    p.add_argument("level", help="назва рівня (стане назвою файлу маршруту)")
    p.add_argument("--start", type=float, required=True, help="секунда старту забігу у відео")
    p.add_argument("--end", type=float, help="секунда фінішу")
    p.add_argument("--crop", help="x,y,w,h у частках кадру, щоб лишити тільки екран гри")
    p.add_argument("--model", help="YOLO-модель з класом player (за замовчуванням з config.json)")
    p.add_argument("--lead", type=float, default=0.1, help="на скільки секунд свайп передує видимому руху")
    p.add_argument("--out-dir", default=str(ROOT / "routes"))
    args = p.parse_args()

    cfg = load_config()
    v = cfg["vision"]
    detector = YoloDetector(args.model or str(ROOT / v["model"]), v["conf"], v["imgsz"])
    crop = [float(x) for x in args.crop.split(",")] if args.crop else None

    track, missing = [], 0
    with tempfile.TemporaryDirectory() as tmp:
        src = download(args.source, tmp) if args.source.startswith("http") else Path(args.source)
        with av.open(str(src)) as container:
            for frame in container.decode(video=0):
                t = frame.time
                if t is None or t < args.start:
                    continue
                if args.end is not None and t > args.end:
                    break
                img = frame.to_ndarray(format="bgr24")
                if crop:
                    h, w = img.shape[:2]
                    x, y, cw, ch = crop
                    img = img[int(y * h):int((y + ch) * h), int(x * w):int((x + cw) * w)]
                h, w = img.shape[:2]
                box = route.merge_boxes([(d.conf, (d.x1 / w, d.y1 / h, d.x2 / w, d.y2 / h))
                                         for d in detector.detect(img) if d.cls == "player"])
                track.append((t - args.start, box))
                missing += box is None
                if len(track) % 300 == 0:
                    print(f"  оброблено {t:6.1f} с")

    if not track:
        sys.exit("У заданому проміжку немає кадрів")
    found = len(track) - missing
    print(f"Персонажа знайдено на {found}/{len(track)} кадрах")
    if found < 0.7 * len(track):
        print("⚠️  Мало кадрів з персонажем: дорозміть клас player на кадрах цього відео й донавчи модель.")

    gestures = route.actions_from_track(track, cfg["swipe"], lead=args.lead)
    out = Path(args.out_dir) / f"{args.level}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    route.save(out, args.level, gestures, start="manual", rebase=False)
    counts = {a: sum(g.action == a for g in gestures) for a in ("left", "right", "jump", "roll")}
    print(f"Збережено {len(gestures)} жестів {counts} → {out}")


if __name__ == "__main__":
    main()
