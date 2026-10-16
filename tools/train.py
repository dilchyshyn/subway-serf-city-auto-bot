"""Навчання YOLO-детектора для Subway Surfers City.

  python tools/train.py                       # CPU, ~1-2 год на 400 кадрах
  python tools/train.py --device 0            # NVIDIA GPU
  python tools/train.py --export openvino     # + прискорена модель для CPU Intel/AMD

Береться маленька попередньо навчена модель (yolo11n) і донавчається на
dataset/. Найкраща модель копіюється в models/city.pt, яку main.py бере
за замовчуванням.
"""
import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=str(ROOT / "dataset" / "data.yaml"))
    p.add_argument("--model", default="yolo11n.pt", help="базова модель (завантажиться автоматично)")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default="cpu", help="cpu або номер GPU")
    p.add_argument("--out", default=str(ROOT / "models" / "city.pt"))
    p.add_argument("--export", choices=["openvino", "onnx"], help="додатково експортувати модель")
    args = p.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.model)
    results = model.train(
        data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, device=args.device,
        project=str(ROOT / "runs"), name="city", exist_ok=True,
        patience=20,         # зупинитися раніше, якщо метрики не ростуть
        fliplr=0.5,          # гра симетрична: дзеркальні кадри — безкоштовні нові приклади
        degrees=0, shear=0,  # перспектива в грі фіксована, тож обертати кадри не треба
        workers=2,
    )
    best = Path(results.save_dir) / "weights" / "best.pt"
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(best, out)
    print(f"Модель збережено: {out}")

    if args.export:
        exported = YOLO(str(out)).export(format=args.export, imgsz=args.imgsz)
        print(f"Експортовано: {exported}\nЩоб використати: python main.py --model {exported}")


if __name__ == "__main__":
    main()
