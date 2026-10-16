"""Кадри з відео (запис екрана або проходження з YouTube) для датасету.

  python tools/extract_frames.py video.mp4 rec/video1 --fps 2
  python tools/extract_frames.py "https://youtube.com/watch?v=..." rec/yt1 --fps 2   # потрібен yt-dlp

Відео проходжень з YouTube — швидкий спосіб набрати різноманітні кадри всіх
районів. Якщо гравець записаний разом з рамкою чи вебкамерою, задай --crop,
щоб лишити тільки екран гри. Такі кадри — лише для власного навчання
моделі, не викладай їх у репозиторій.
"""
import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import av
import cv2


def download(url, tmp):
    if not shutil.which("yt-dlp"):
        sys.exit("Для посилань потрібен yt-dlp: .venv/bin/pip install yt-dlp")
    out = Path(tmp) / "video.%(ext)s"
    subprocess.run(["yt-dlp", "-f", "bv*[height<=1080][ext=mp4]/bv*[height<=1080]/b",
                    "-o", str(out), url], check=True)
    return next(Path(tmp).glob("video.*"))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("source", help="файл відео або посилання")
    p.add_argument("out_dir")
    p.add_argument("--fps", type=float, default=2.0, help="скільки кадрів на секунду зберігати")
    p.add_argument("--crop", help="x,y,w,h у частках кадру, напр. 0.33,0,0.34,1 для вертикальної гри в 16:9 відео")
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    crop = [float(v) for v in args.crop.split(",")] if args.crop else None

    with tempfile.TemporaryDirectory() as tmp:
        src = download(args.source, tmp) if args.source.startswith("http") else Path(args.source)
        saved, next_t = 0, 0.0
        with av.open(str(src)) as container:
            for frame in container.decode(video=0):
                if frame.time is None or frame.time < next_t:
                    continue
                next_t = frame.time + 1 / args.fps
                img = frame.to_ndarray(format="bgr24")
                if crop:
                    h, w = img.shape[:2]
                    x, y, cw, ch = crop
                    img = img[int(y * h):int((y + ch) * h), int(x * w):int((x + cw) * w)]
                cv2.imwrite(str(out / f"{src.stem}_{saved:05d}.jpg"), img)
                saved += 1
    print(f"Збережено {saved} кадрів у {out}")


if __name__ == "__main__":
    main()
