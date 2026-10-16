"""Автоматична перевірка маршрутів, зібраних build_routes.py (без перегляду кадрів).

  python tools/check_routes.py            # звіт

Перевіряє для кожного відео:
  * кількість забігів проти кількості рівнів з назви («Levels 11–15» → 5,
    «(1.1–1.3)» → 3, «12/12 Stars» → 4);
  * спроби рівня: завдання зберігаються між спробами, тож рівень у відео
    часто пройдено кількома забігами — їх групуємо за текстом завдань (OCR);
  * маршрути без картки (бот не зможе їх упізнати) і з підозрілими даними.
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from bot import video  # noqa: E402
from bot.library import SAME_LEVEL, entry_text_state  # noqa: E402


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
    args = p.parse_args()
    rdir = Path(args.routes)

    by_video = defaultdict(list)
    for path in sorted(rdir.glob("*.json")):
        if path.name == "index.json":
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        data["_path"] = path
        by_video[data.get("source") or path.stem.rsplit("_", 1)[0]].append(data)

    warnings, total_levels, total_runs = 0, 0, 0
    for source, runs in by_video.items():
        runs.sort(key=lambda d: d["run"][0])
        title = runs[0].get("title", source)
        # групуємо послідовні спроби того самого рівня за текстом завдань
        levels = []
        for d in runs:
            if d.get("card"):
                d["_text"], d["_state"] = entry_text_state(rdir, d)
            if levels and d.get("_text") and levels[-1][-1].get("_text") and \
                    video.text_similarity(d["_text"], levels[-1][-1]["_text"]) >= SAME_LEVEL:
                levels[-1].append(d)
            else:
                levels.append([d])

        exp = expected_levels(title)
        problems = []
        if exp is not None and exp != len(levels):
            problems.append(f"рівнів у назві {exp}, знайдено {len(levels)} (забігів {len(runs)})")
        for d in runs:
            dur = d["run"][1] - d["run"][0]
            n = len(d["gestures"])
            if not d.get("card"):
                problems.append(f"{d['level']}: немає картки — бот не впізнає рівень")
            if len(d.get("dropped_lane_events", [])) > 3:
                problems.append(f"{d['level']}: відкинуто подій смуги {len(d['dropped_lane_events'])}")
            if n < dur / 6:
                problems.append(f"{d['level']}: лише {n} жестів за {dur:.0f} с")
        total_levels += len(levels)
        total_runs += len(runs)
        warnings += len(problems)
        mark = "⚠️ " if problems else "✅"
        print(f"{mark} {title[:70]}: рівнів {len(levels)}, забігів {len(runs)}")
        for pr in problems:
            print(f"     {pr}")

    print(f"\nРівнів: {total_levels}, забігів (спроб): {total_runs}, попереджень: {warnings}")

if __name__ == "__main__":
    main()
