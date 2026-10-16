"""Інтерактивне калібрування геометрії смуг і HSV-порогів евристики.

  python tools/calibrate.py                                # кадр з телефону (screencap)
  python tools/calibrate.py --backend image --image rec/   # записані кадри

Повзунками підганяєш трапеції під дорогу так, щоб near-зона була трохи
попереду персонажа (геометрія потрібна і для YOLO, і для евристики).
HSV-повзунки й піпетка впливають лише на евристику (авторозмітку).
Клік мишею по пікселю друкує його HSV.
Клавіші: n — новий кадр, s — зберегти в config.json, q — вихід.
"""
import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bot import load_config, save_config  # noqa: E402
from bot.capture import BACKENDS, open_capture  # noqa: E402
from bot.geometry import LaneGeometry  # noqa: E402
from bot.heuristic import HeuristicDetector  # noqa: E402
from bot.overlay import draw_overlay  # noqa: E402
from bot.scene import build_scene  # noqa: E402

WIN = "calibrate"


def _set(path):
    """Сеттер для вкладеного ключа: _set(("lanes", "near_y"))(cfg, value)."""
    def setter(cfg, v):
        node = cfg
        for k in path[:-1]:
            node = node[k]
        node[path[-1]] = v
    return setter


def _get(path):
    def getter(cfg):
        node = cfg
        for k in path:
            node = node[k]
        return node
    return getter


def _set_near_top(cfg, v):
    cfg["zones"]["near"][0] = v
    cfg["zones"]["far"][1] = v


# назва -> (геттер, сеттер, максимум); значення в тисячних частках
SLIDERS = {
    "vp_y": (_get(("lanes", "vanishing_point", 1)), _set(("lanes", "vanishing_point", 1)), 1000),
    "near_y": (_get(("lanes", "near_y")), _set(("lanes", "near_y")), 1000),
    "lane_off": (lambda c: 0.5 - c["lanes"]["centers_near"][0],
                 lambda c, v: c["lanes"].__setitem__("centers_near", [0.5 - v, 0.5, 0.5 + v]), 500),
    "lane_w": (_get(("lanes", "width_near")), _set(("lanes", "width_near")), 500),
    "far_top": (_get(("zones", "far", 0)), _set(("zones", "far", 0)), 1000),
    "near_top": (_get(("zones", "near", 0)), _set_near_top, 1000),
    "near_bot": (_get(("zones", "near", 1)), _set(("zones", "near", 1)), 1000),
}
# HSV та ядро евристики — цілі числа, без масштабу
INT_SLIDERS = {
    "S_min": (_get(("heuristic", "obstacle_hsv", 0, "lo", 1)), _set(("heuristic", "obstacle_hsv", 0, "lo", 1)), 255),
    "V_min": (_get(("heuristic", "obstacle_hsv", 0, "lo", 2)), _set(("heuristic", "obstacle_hsv", 0, "lo", 2)), 255),
    "kernel": (_get(("heuristic", "morph_kernel")),
               lambda c, v: c["heuristic"].__setitem__("morph_kernel", max(v, 1)), 31),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config")
    p.add_argument("--profile", help="район; s збереже налаштування в profiles[район]")
    p.add_argument("--backend", default="screencap", choices=BACKENDS)
    p.add_argument("--device", default="/dev/video10")
    p.add_argument("--image")
    p.add_argument("--serial")
    args = p.parse_args()

    cfg = load_config(args.config, args.profile)
    cap = open_capture(args.backend, args.device, args.serial, args.image)
    frame = None
    while frame is None:
        frame = cap.read()

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    view = {"scale": 1.0}

    def pick_hsv(event, x, y, *_):
        # Клік по монеті/предмету друкує HSV пікселя — так підбираються діапазони в config.json
        if event == cv2.EVENT_LBUTTONDOWN:
            fx, fy = int(x / view["scale"]), int(y / view["scale"])
            if 0 <= fy < frame.shape[0] and 0 <= fx < frame.shape[1]:
                h, s, v = cv2.cvtColor(frame[fy:fy + 1, fx:fx + 1], cv2.COLOR_BGR2HSV)[0, 0]
                print(f"({fx}, {fy})  HSV = [{h}, {s}, {v}]")

    cv2.setMouseCallback(WIN, pick_hsv)
    for name, (get, _, mx) in SLIDERS.items():
        cv2.createTrackbar(name, WIN, int(get(cfg) * 1000), mx, lambda _: None)
    for name, (get, _, mx) in INT_SLIDERS.items():
        cv2.createTrackbar(name, WIN, int(get(cfg)), mx, lambda _: None)

    while True:
        for name, (_, set_, _) in SLIDERS.items():
            set_(cfg, cv2.getTrackbarPos(name, WIN) / 1000)
        for name, (_, set_, _) in INT_SLIDERS.items():
            set_(cfg, cv2.getTrackbarPos(name, WIN))

        geo = LaneGeometry(cfg)
        h, w = frame.shape[:2]
        scene = build_scene(HeuristicDetector(cfg, geo).detect(frame), w, h, geo, cfg)
        vis = draw_overlay(frame, geo, scene)
        view["scale"] = 800 / vis.shape[0]
        cv2.imshow(WIN, cv2.resize(vis, None, fx=view["scale"], fy=view["scale"]))

        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            break
        if key == ord("n"):
            new = cap.read()
            if new is not None:
                frame = new
        if key == ord("s"):
            save_config(cfg, args.config, args.profile)
            print(f"Збережено в config.json{' (профіль ' + args.profile + ')' if args.profile else ''}")

    cap.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
