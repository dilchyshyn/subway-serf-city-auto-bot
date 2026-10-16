"""Створює YOLO-датасет із записаних кадрів, з чорновими мітками від евристики.

  python tools/autolabel.py rec/docks rec/southline rec/yt1 --zip

Результат — папка dataset/ у форматі YOLO (images/, labels/, data.yaml,
classes.txt). Евристика розмічає монети, Energy, пауер-апи й перешкоди;
board_coin, stomp_target, star, board_ready та тип перешкоди (low/high)
доведеться позначити або виправити вручну в редакторі розмітки (див. README).
--zip пакує датасет для завантаження в Roboflow чи Google Colab.
"""
import argparse
import hashlib
import shutil
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from bot import load_config  # noqa: E402
from bot.classes import CLASSES  # noqa: E402
from bot.geometry import LaneGeometry  # noqa: E402
from bot.heuristic import HeuristicDetector  # noqa: E402


def split_of(name, val_fraction):
    # Детермінований поділ: той самий кадр завжди потрапляє в ту саму частину
    return "val" if int(hashlib.md5(name.encode()).hexdigest(), 16) % 1000 < val_fraction * 1000 else "train"


def yolo_line(d, w, h):
    cx, cy = (d.x1 + d.x2) / 2 / w, (d.y1 + d.y2) / 2 / h
    bw, bh = (d.x2 - d.x1) / w, (d.y2 - d.y1) / h
    return f"{CLASSES.index(d.cls)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def write_data_yaml(out):
    lines = [f"path: {out.resolve()}", "train: images/train", "val: images/val", "names:"]
    lines += [f"  {i}: {name}" for i, name in enumerate(CLASSES)]
    (out / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "classes.txt").write_text("\n".join(CLASSES) + "\n", encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("inputs", nargs="+", help="папки з кадрами")
    p.add_argument("--out", default=str(ROOT / "dataset"))
    p.add_argument("--profile", help="профіль району для евристики")
    p.add_argument("--val", type=float, default=0.15, help="частка кадрів для валідації")
    p.add_argument("--empty", action="store_true", help="не ставити чорнових міток, лише порожні файли")
    p.add_argument("--zip", action="store_true", help="запакувати датасет у dataset.zip")
    args = p.parse_args()

    cfg = load_config(profile=args.profile)
    detector = HeuristicDetector(cfg, LaneGeometry(cfg))
    out = Path(args.out)
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    total = boxes = 0
    for src_dir in map(Path, args.inputs):
        for img_path in sorted([*src_dir.glob("*.jpg"), *src_dir.glob("*.png")]):
            name = f"{src_dir.name}_{img_path.stem}"
            label_path = out / "labels" / split_of(name, args.val) / f"{name}.txt"
            if label_path.exists():
                continue  # не затирати вже виправлені вручну мітки
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            dets = [] if args.empty else detector.detect(img)
            shutil.copy(img_path, out / "images" / split_of(name, args.val) / f"{name}{img_path.suffix}")
            label_path.write_text("".join(yolo_line(d, w, h) + "\n" for d in dets))
            total += 1
            boxes += len(dets)

    write_data_yaml(out)
    print(f"Додано {total} кадрів, {boxes} чорнових рамок → {out}")
    if args.zip:
        archive = shutil.make_archive(str(out), "zip", out)
        print(f"Архів: {archive}")


if __name__ == "__main__":
    main()
