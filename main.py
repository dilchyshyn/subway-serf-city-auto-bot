"""Бот для Subway Surfers City на Android-телефоні (root не потрібен).

Приклади:
  python main.py --show                          # грати (YOLO-модель models/city.pt)
  python main.py --show --dry-run                # лише дивитися рішення, без свайпів
  python main.py --vision heuristic --show       # без моделі, на кольорових евристиках
  python main.py --backend screenrecord --show   # захоплення без v4l2loopback
  python main.py --tour                          # рівень City Tour: бот упізнає його за карткою
  python main.py --route routes/docks_07.json    # пройти рівень City Tour за конкретним маршрутом
  python main.py --dry-run --record rec/ --vision heuristic   # записувати кадри, поки граєш сам

Клавіші у вікні (--show): r — скинути смугу на середню (на початку забігу),
p — пауза, q — вихід.
"""
import argparse
import time
from pathlib import Path

import cv2

from bot import load_config
from bot import route as routes
from bot.brain import LEFT, RIGHT, Brain
from bot.capture import BACKENDS, open_capture
from bot.controls import AdbController, DryRunController
from bot.geometry import LaneGeometry
from bot.heuristic import HeuristicDetector
from bot.overlay import draw_overlay
from bot.scene import build_scene

ROOT = Path(__file__).resolve().parent


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", help="шлях до config.json")
    p.add_argument("--profile", help="район для евристик: docks, southline, sunrise, delorean, undertracks")
    p.add_argument("--backend", choices=BACKENDS, help="джерело кадрів")
    p.add_argument("--device", help="v4l2-пристрій, напр. /dev/video10")
    p.add_argument("--image", help="картинка або папка (для --backend image)")
    p.add_argument("--serial", help="серійний номер телефону, якщо підключено кілька")
    p.add_argument("--vision", choices=["yolo", "heuristic"], help="чим розпізнавати об'єкти")
    p.add_argument("--model", help="шлях до YOLO-моделі (.pt або експортованої)")
    p.add_argument("--tour", action="store_true",
                   help="відкрита картка рівня City Tour: упізнати рівень, натиснути PLAY і пройти за маршрутом")
    p.add_argument("--route", help="маршрут рівня City Tour (routes/*.json)")
    p.add_argument("--route-offset", type=float, default=0.0,
                   help="зсув маршруту в секундах, якщо рівень вантажиться довше/коротше, ніж під час запису")
    p.add_argument("--route-guard", action="store_true",
                   help="між жестами маршруту стрибати/пірнати перед несподіваними перешкодами")
    p.add_argument("--diag", action="store_true",
                   help="записати кадри й журнал дій у rec/diag/<час>/ для розбору помилок")
    p.add_argument("--dry-run", action="store_true", help="не надсилати жести на телефон")
    p.add_argument("--show", action="store_true", help="показувати вікно налагодження")
    p.add_argument("--record", help="зберігати сирі кадри в цю папку (для датасету)")
    p.add_argument("--record-every", type=int, default=5, help="зберігати кожен N-й кадр")
    return p.parse_args()


def make_detector(kind, cfg, geo, model):
    v = cfg["vision"]
    path = model or v["model"]
    if kind == "yolo" and not (ROOT / path).exists() and not Path(path).exists():
        print(f"YOLO-модель {path} не знайдено — звичайний режим на евристиках (слабкий).")
        kind = "heuristic"
    if kind == "heuristic":
        return HeuristicDetector(cfg, geo)
    from bot.detector import YoloDetector
    return YoloDetector(path, v["conf"], v["imgsz"])


def wait_for_run(cap, timeout):
    """Чекає, доки на екрані з'явиться HUD забігу (2 кадри поспіль)."""
    from bot.video import hud_visible
    deadline, streak = time.monotonic() + timeout, 0
    while time.monotonic() < deadline:
        frame = cap.read()
        if frame is None:
            time.sleep(0.005)
            continue
        streak = streak + 1 if hud_visible(frame) else 0
        if streak >= 2:
            return True
    return False


def pick_tour_route(cap, cfg, timeout=10):
    """Чекає картку рівня на екрані й шукає для неї маршрут у бібліотеці."""
    from bot.library import RouteLibrary
    lib = RouteLibrary(ROOT / cfg["tour"]["routes"])
    print(f"Маршрутів у бібліотеці: {len(lib)}")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        frame = cap.read()
        if frame is None:
            time.sleep(0.01)
            continue
        match, play = lib.match(frame)
        if play is None:
            continue
        if lib.confident(match, cfg):
            print(f"Рівень упізнано: {match.name} (подібність {match.score:.2f}, відрив {match.margin:.2f}, "
                  f"спільних чисел {match.common})")
            if match.crew:
                print(f"Персонаж з відео (обраний у FEATURED CREW): {match.crew}")
            return str(match.path), play
        best = f"{match.name} ({match.score:.2f}, спільних чисел {match.common})" if match else "—"
        print(f"Маршруту для цього рівня немає (найближчий: {best}). Гратиму у звичайному режимі.")
        return None, play
    raise SystemExit("Картку рівня з кнопкою PLAY не знайдено. Відкрий її на телефоні й запусти знову.")


def main():
    args = parse_args()
    cfg = load_config(args.config, args.profile)
    backend = args.backend or cfg["capture"]["backend"]
    geo = LaneGeometry(cfg)
    player = None
    cap = None
    tour_play = None
    if args.tour:
        cap = open_capture(backend, args.device or cfg["capture"]["device"], args.serial, args.image)
        args.route, tour_play = pick_tour_route(cap, cfg)
    if args.route:
        level, gestures, start_mode = routes.load(args.route)
        if not gestures:
            raise SystemExit(f"Маршрут {args.route} порожній")
    need_vision = not args.route or args.route_guard or args.show or args.record
    # маршрут з відео стартує за появою HUD на екрані — для цього потрібні кадри
    need_capture = need_vision or args.diag or (args.route and start_mode == "hud")

    detector = make_detector(args.vision or cfg["vision"]["backend"], cfg, geo, args.model) if need_vision else None
    if cap is None and need_capture:
        cap = open_capture(backend, args.device or cfg["capture"]["device"], args.serial, args.image)
    ctl = DryRunController() if args.dry_run else AdbController(cfg, args.serial)
    brain = Brain(cfg)
    play_xy = ((tour_play[0] + tour_play[2]) / 2, (tour_play[1] + tour_play[3]) / 2) if tour_play else cfg["play_button"]

    if args.tour and not args.route:
        ctl.tap(*play_xy)
        if not wait_for_run(cap, cfg["route_start_timeout"]):
            raise SystemExit("Забіг не почався: HUD так і не з'явився.")

    if args.route:
        print(f"Маршрут «{level}»: {len(gestures)} жестів, {gestures[-1].t:.1f} с.")
        if start_mode == "hud":
            # маршрут з відео: t=0 — поява HUD забігу, як і у відео
            print("Екран рівня з кнопкою PLAY має бути відкритий. Бот натисне її й чекатиме старту забігу.")
            ctl.tap(*play_xy)
            if not wait_for_run(cap, cfg["route_start_timeout"]):
                raise SystemExit("Забіг не почався: HUD так і не з'явився. Чи відкрито екран рівня?")
        elif start_mode == "manual":
            input("Натисни «Грати» на телефоні, потім Enter тут у момент, коли персонаж почне бігти...")
        else:
            print("Екран рівня з кнопкою «Грати» має бути відкритий, бот натисне її сам.")
        player = routes.RoutePlayer(gestures, time.monotonic(), args.route_offset,
                                    first_immediate=start_mode == "tap")

    diag = None
    if args.diag:
        diag_dir = ROOT / "rec" / "diag" / time.strftime("%Y%m%d_%H%M%S")
        diag_dir.mkdir(parents=True, exist_ok=True)
        diag = {"dir": diag_dir, "t0": time.monotonic(), "log": open(diag_dir / "log.txt", "w", encoding="utf-8"),
                "last": 0.0}
        diag["log"].write(f"route={args.route} offset={args.route_offset}\n")
        print(f"Діагностика пишеться в {diag_dir}")

    def diag_event(text):
        if diag:
            diag["log"].write(f"{time.monotonic() - diag['t0']:8.3f} {text}\n")
            diag["log"].flush()

    rec_dir = Path(args.record) if args.record else None
    if rec_dir:
        rec_dir.mkdir(parents=True, exist_ok=True)

    paused = False
    frames, saved, t_fps, last_action = 0, 0, time.monotonic(), None
    print("Бот запущено. Ctrl+C — вихід.")
    try:
        while True:
            now = time.monotonic()
            if player and not paused:
                for g in player.due(now):
                    ctl.gesture(g)
                    diag_event(f"route {g.t:.2f} {g.action}")
                    last_action = f"route {g.action}"
                    if g.action in (LEFT, RIGHT):
                        brain.lane = min(max(brain.lane + (1 if g.action == RIGHT else -1), 0), 2)
                if player.finished:
                    print("Маршрут завершено.")
                    player = None
                    if not args.show:
                        break

            if diag and now - diag["last"] >= 0.1:  # кадр кожні 0,1 с
                frame = cap.read()
                if frame is not None:
                    diag["last"] = now
                    cv2.imwrite(str(diag["dir"] / f"{now - diag['t0']:08.3f}.jpg"),
                                cv2.resize(frame, (288, int(288 * frame.shape[0] / frame.shape[1]))))
            if not need_vision:
                time.sleep(0.005)
                continue
            frame = cap.read()
            if frame is None:
                time.sleep(0.002)
                continue
            h, w = frame.shape[:2]
            scene = build_scene(detector.detect(frame), w, h, geo, cfg)

            action = None
            if not paused:
                if not args.route:
                    action = brain.decide(scene, now)
                elif player and args.route_guard and player.next_in(now) > cfg["action_cooldown"]:
                    # не міняємо смугу, щоб не зламати маршрут, лише стрибаємо/пірнаємо
                    action = brain.emergency(scene, now)
            if action:
                ctl.act(action)
                diag_event(f"act {action}")
                last_action = action
                print(f"{action:>6}  lane={brain.lane}  " + "  ".join(
                    f"{s.near or '-'}/{s.far or '-'}+{s.value:g}" for s in scene.lanes) +
                    ("  board" if scene.board_ready else ""))

            if rec_dir and frames % args.record_every == 0:
                cv2.imwrite(str(rec_dir / f"{int(time.time() * 1000)}.jpg"), frame)
                saved += 1
            frames += 1
            if now - t_fps >= 5:
                print(f"FPS: {frames / (now - t_fps):.1f}" + (f", збережено кадрів: {saved}" if rec_dir else ""))
                frames, t_fps = 0, now

            if args.show:
                vis = draw_overlay(frame, geo, scene, brain.lane, "PAUSED" if paused else last_action)
                scale = 720 / vis.shape[0]
                cv2.imshow("subway bot", cv2.resize(vis, None, fx=scale, fy=scale))
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("r"):
                    brain.reset()
                    print("Смугу скинуто на середню")
                if key == ord("p"):
                    paused = not paused
            if backend == "image":
                time.sleep(1 / 30)  # імітуємо реальну частоту кадрів
    except KeyboardInterrupt:
        pass
    finally:
        if cap:
            cap.close()
        ctl.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
