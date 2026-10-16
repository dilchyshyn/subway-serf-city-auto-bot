"""Автоматична перевірка маршрутів, зібраних build_routes.py (без перегляду кадрів).

  python tools/check_routes.py            # звіт
  python tools/check_routes.py --fix      # відкинути невдалі спроби в routes/rejected/

Перевіряє для кожного відео:
  * кількість забігів проти кількості рівнів з назви («Levels 11–15» → 5,
    «(1.1–1.3)» → 3, «12/12 Stars» → 4);
  * повтори рівня: якщо картка рівня трапилась двічі, гравець програв і
    перепройшов — лишається остання спроба;
  * маршрути без картки (бот не зможе їх упізнати) і з підозрілими даними.
"""
import argparse
import json
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from bot import video  # noqa: E402

SAME_LEVEL = 0.97  # подібність карток, з якої це той самий рівень


def expected_levels(title):
    t = title.replace("–", "-").replace("—", "-")
    m = re.search(r"\((\d+)\.(\d+)\s*-\s*(\d+)\.(\d+)\)", t)            # (1.1-1.3)
    if m and m.group(1) == m.group(3):
        return int(m.group(4)) - int(m.group(2)) + 1
    m = re.search(r"levels?\s+(\d+)\s*(?:-|to)\s*(\d+)", t, re.IGNORECASE)  # level 28 - 32
    if m:
        return int(m.group(2)) - int(m.group(1)) + 1
    m = re.search(r"\((\d+)\s+levels?\)", t, re.IGNORECASE)                # (5 Levels)
    if m:
        return int(m.group(1))
    m = re.search(r"(\d+)\s*/\s*(\d+)\s*stars", t, re.IGNORECASE)          # 12/12 Stars
    if m and int(m.group(2)) % 3 == 0:
        return int(m.group(2)) // 3
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--routes", default=str(ROOT / "routes"))
    p.add_argument("--fix", action="store_true", help="перенести невдалі спроби в routes/rejected/")
    args = p.parse_args()
    rdir = Path(args.routes)

    by_video = defaultdict(list)
    for path in sorted(rdir.glob("*.json")):
        if path.name == "index.json":
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        data["_path"] = path
        by_video[data.get("source") or path.stem.rsplit("_", 1)[0]].append(data)

    rejected, warnings, ok_levels = [], 0, 0
    for source, runs in by_video.items():
        runs.sort(key=lambda d: d["run"][0])
        title = runs[0].get("title", source)
        fps = []
        for d in runs:
            card = rdir / d["card"] if d.get("card") else None
            fps.append(video.band_fingerprint(cv2.imread(str(card))) if card and card.exists() else None)

        # повтори: та сама картка пізніше у відео → ранні спроби невдалі
        keep = [True] * len(runs)
        for i in range(len(runs)):
            for j in range(i + 1, len(runs)):
                if fps[i] is not None and fps[j] is not None and video.card_similarity(fps[i], fps[j]) >= SAME_LEVEL:
                    keep[i] = False
        kept = [d for d, k in zip(runs, keep) if k]
        rejected += [d for d, k in zip(runs, keep) if not k]

        exp = expected_levels(title)
        problems = []
        if exp is not None and exp != len(kept):
            problems.append(f"рівнів у назві {exp}, знайдено {len(kept)}")
        for d in kept:
            dur = d["run"][1] - d["run"][0]
            n = len(d["gestures"])
            if not d.get("card"):
                problems.append(f"{d['level']}: немає картки — бот не впізнає рівень")
            if len(d.get("dropped_lane_events", [])) > 3:
                problems.append(f"{d['level']}: відкинуто подій смуги {len(d['dropped_lane_events'])}")
            if n < dur / 6:
                problems.append(f"{d['level']}: лише {n} жестів за {dur:.0f} с")
        ok_levels += len(kept)
        warnings += len(problems)
        mark = "⚠️ " if problems else "✅"
        print(f"{mark} {title[:70]}: {len(kept)} рівн." + (f", відкинуто повторів {len(runs) - len(kept)}" if len(runs) > len(kept) else ""))
        for pr in problems:
            print(f"     {pr}")

    print(f"\nРівнів з маршрутами: {ok_levels}, попереджень: {warnings}, невдалих спроб: {len(rejected)}")
    if args.fix and rejected:
        (rdir / "rejected").mkdir(exist_ok=True)
        for d in rejected:
            shutil.move(str(d["_path"]), rdir / "rejected" / d["_path"].name)
        print(f"Перенесено в {rdir / 'rejected'}: {len(rejected)}")


if __name__ == "__main__":
    main()
