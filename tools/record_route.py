"""Запис маршруту рівня City Tour.

  python tools/record_route.py docks_07

1. Відкрий на телефоні екран рівня з кнопкою «Грати».
2. Запусти скрипт, натисни «Грати» й пройди рівень, виконавши всі 3 завдання.
3. Після фінішу натисни Ctrl+C. Маршрут збережеться в routes/docks_07.json.

Якщо не всі завдання виконано, просто запиши рівень ще раз: файл перезапишеться.
Відтворення: python main.py --route routes/docks_07.json
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from bot import route  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("level", help="назва рівня, напр. docks_07 (стане назвою файлу)")
    p.add_argument("--serial")
    p.add_argument("--out-dir", default=str(ROOT / "routes"))
    args = p.parse_args()

    out = Path(args.out_dir) / f"{args.level}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    print("Запис почався. Натисни «Грати» на телефоні й пройди рівень. Ctrl+C — зберегти.")
    gestures = route.record(out, args.level, args.serial)
    if gestures:
        print(f"Збережено {len(gestures)} жестів ({gestures[-1].t - gestures[0].t:.1f} с) → {out}")
    else:
        print("Жодного жесту не записано, файл порожній.")


if __name__ == "__main__":
    main()
