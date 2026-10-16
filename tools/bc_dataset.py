"""Датасет для навчання імітацією (behavior cloning) з відео проходжень.

  python tools/bc_dataset.py              # усі відео з routes/ → rec/bc/<відео>.npz
  python tools/bc_dataset.py --limit 5    # пробний прогін на 5 відео

Для кожного забігу з маршрутів беремо кадри (fps на секунду), зменшені до
SIZE, і мітку — що гравець зробив за мить після кадру: 0 нічого, 1 вліво,
2 вправо, 3 стрибок, 4 униз. «За мить» = lead секунд: бот бачить кадр,
вирішує, а свайп через adb доходить із затримкою — модель має вчитися
реагувати трохи наперед, як і гравець.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import av
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
# та сама підготовка кадру, що й у грі: відео й телефон мають виглядати для моделі однаково
from bot.policy import preprocess  # noqa: E402

ACTIONS = ["none", "left", "right", "jump", "roll"]


def label_for(t, gestures, lead, tol):
    """Дія, свайп якої припадає на [t + lead - tol, t + lead + tol], або 0."""
    for g in gestures:
        if abs(g["t"] - (t + lead)) <= tol:
            return ACTIONS.index(g["action"]) if g["action"] in ACTIONS else 0
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--routes", default=str(ROOT / "routes"))
    p.add_argument("--videos", default=str(ROOT / "rec" / "videos"))
    p.add_argument("--out", default=str(ROOT / "rec" / "bc"))
    p.add_argument("--fps", type=float, default=10)
    p.add_argument("--lead", type=float, default=0.15)
    p.add_argument("--tol", type=float, default=0.1)
    p.add_argument("--limit", type=int, help="лише перші N відео")
    args = p.parse_args()

    by_video = defaultdict(list)
    for path in sorted(Path(args.routes).glob("*.json")):
        if path.name == "index.json":
            continue
        d = json.loads(path.read_text(encoding="utf-8"))
        if d.get("source"):
            by_video[d["source"].rsplit("=", 1)[-1]].append(d)
    vids = sorted(by_video)[: args.limit] if args.limit else sorted(by_video)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for i, vid in enumerate(vids, 1):
        dst = out / f"{vid}.npz"
        if dst.exists():
            continue
        src = next(Path(args.videos).glob(f"{vid}.*mp4"), None)
        if src is None:
            print(f"[{i}/{len(vids)}] {vid}: відео немає, пропускаю")
            continue
        runs = sorted(by_video[vid], key=lambda d: d["run"][0])
        frames, prev, labels, run_ids = [], [], [], []
        last_img, last_run, next_t = None, None, 0.0
        with av.open(str(src)) as c:
            for f in c.decode(video=0):
                t = f.time
                if t is None or t < next_t:
                    continue
                k = next((j for j, d in enumerate(runs) if d["run"][0] <= t <= d["run"][1]), None)
                if k is None:
                    continue
                next_t = t + 1 / args.fps
                img = preprocess(f.to_ndarray(format="bgr24"))
                # другий вхід моделі — попередній кадр того самого забігу (0,1 с тому): видно рух
                frames.append(img)
                prev.append(last_img if last_run == k else img)
                labels.append(label_for(t - runs[k]["run"][0], runs[k]["gestures"], args.lead, args.tol))
                run_ids.append(k)
                last_img, last_run = img, k
        if frames:
            np.savez_compressed(dst, x=np.stack(frames), prev=np.stack(prev),
                                y=np.array(labels, np.uint8), run=np.array(run_ids, np.int16))
            counts = np.bincount(labels, minlength=len(ACTIONS))
            print(f"[{i}/{len(vids)}] {vid}: кадрів {len(frames)}, мітки {dict(zip(ACTIONS, counts.tolist()))}")


if __name__ == "__main__":
    main()
