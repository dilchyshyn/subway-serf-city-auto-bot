"""Перевірка, чи все готово до запуску бота (root не потрібен).

  python tools/doctor.py

Перевіряє adb, підключення телефону, дозвіл на свайпи (на Xiaomi/POCO/Redmi
його блокує HyperOS), читання дотиків для запису маршрутів, захоплення
екрана, наявність гри й YOLO-моделі. На кожну проблему підказує, що зробити.
"""
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

GAME_PACKAGE = "com.sybogames.subway.surfers.game"
OK, FAIL, WARN = "✅", "❌", "⚠️ "
problems = 0


def report(ok, msg, fix=None, warn=False):
    global problems
    print(f"{OK if ok else (WARN if warn else FAIL)} {msg}")
    if not ok and fix:
        for line in fix.strip().splitlines():
            print(f"     {line}")
    if not ok and not warn:
        problems += 1
    return ok


def adb(*args, timeout=10):
    r = subprocess.run(["adb", *args], capture_output=True, text=True, timeout=timeout)
    return r.returncode, r.stdout + r.stderr


def main():
    print("Перевірка середовища для Subway Surfers City Bot\n")

    if not report(shutil.which("adb") is not None, "adb встановлено",
                  "sudo dnf install android-tools"):
        return

    _, out = adb("devices")
    lines = [l.split() for l in out.splitlines()[1:] if l.strip()]
    states = [l[1] for l in lines if len(l) >= 2]
    if not states:
        report(False, "Телефон підключено", """
На телефоні: Налаштування → Про телефон → 7 разів тапни «Версія ОС»,
потім Додаткові налаштування → Для розробників → Налагодження USB.
Підключи кабель і обери режим «Передача файлів».""")
        return
    if "unauthorized" in states:
        report(False, "Телефон дозволив налагодження", "Підтверди запит «Дозволити налагодження USB?» на екрані телефону")
        return
    if len(states) > 1:
        report(False, "Підключено один телефон", "Підключено кілька пристроїв: додавай --serial <номер> з `adb devices`",
               warn=True)
    report(True, "Телефон підключено й авторизовано")

    _, model = adb("shell", "getprop", "ro.product.model")
    _, android = adb("shell", "getprop", "ro.build.version.release")
    _, size = adb("shell", "wm", "size")
    print(f"     {model.strip()}, Android {android.strip()}, {size.strip().splitlines()[-1]}")

    # Тест дозволу на ін'єкцію подій: KEYCODE_UNKNOWN (0) нічого не робить
    _, out = adb("shell", "input", "keyevent", "0")
    report("INJECT_EVENTS" not in out and "SecurityException" not in out,
           "Бот може надсилати свайпи (input)", """
На Xiaomi / POCO / Redmi (HyperOS, MIUI) увімкни:
  Для розробників → «Налагодження USB (налаштування безпеки)»
Цей перемикач вимагає увійти в Mi-акаунт, вставлену SIM-картку та інтернет.
Після ввімкнення перепідключи кабель.""")

    _, out = adb("shell", "pm", "list", "packages", GAME_PACKAGE)
    report(GAME_PACKAGE in out, "Subway Surfers City встановлено", "Встанови гру з Google Play", warn=True)

    try:
        from bot.route import touch_range
        x_max, y_max = touch_range()
        report(True, f"Читання дотиків для запису маршрутів (getevent, сенсор {x_max}x{y_max})")
    except Exception as e:  # noqa: BLE001
        report(False, f"Читання дотиків для запису маршрутів: {e}", """
Без root деякі прошивки ховають /dev/input. Тоді запис маршрутів не працює,
але звичайна гра бота — працює.""", warn=True)

    try:
        from bot.capture import ScreencapCapture
        frame = ScreencapCapture().read()
        report(frame is not None, f"Скріншот екрана (screencap) {frame.shape[1]}x{frame.shape[0]}")
    except Exception as e:  # noqa: BLE001
        report(False, f"Скріншот екрана: {e}", "Розблокуй телефон і повтори")

    try:
        import av  # noqa: F401
        from bot.capture import ScreenrecordCapture
        cap = ScreenrecordCapture()
        n, shape, t0 = 0, None, time.time()
        while time.time() - t0 < 4:
            f = cap.read()
            if f is not None:
                n, shape = n + 1, f.shape
            time.sleep(0.005)
        cap.close()
        report(n > 0, f"Відеопотік (screenrecord): {n / 4:.0f} FPS" + (f", {shape[1]}x{shape[0]}" if shape else ""),
               """Відео не йде. Розблокуй телефон, увімкни на ньому будь-що рухоме й повтори.
Запасний варіант: scrcpy + v4l2loopback (--backend v4l2) або --backend screencap.""")
        if 0 < n < 40:
            print("     Мало кадрів: це нормально для статичного екрана (кадри йдуть лише при зміні зображення).")
    except ImportError:
        report(False, "PyAV встановлено", ".venv/bin/pip install -r requirements.txt")

    try:
        import ultralytics  # noqa: F401
        report(True, "ultralytics встановлено")
    except ImportError:
        report(False, "ultralytics встановлено", ".venv/bin/pip install -r requirements.txt")

    from bot import load_config
    model = ROOT / load_config()["vision"]["model"]
    report(model.exists(), f"YOLO-модель {model.relative_to(ROOT)}",
           "Модель ще не натренована, див. README «Навчання моделі». "
           "Поки можна грати з --vision heuristic.", warn=True)

    print()
    print("Усе готово, можна запускати main.py" if problems == 0
          else f"Знайдено проблем: {problems}. Виправ їх і запусти перевірку ще раз.")


if __name__ == "__main__":
    main()
