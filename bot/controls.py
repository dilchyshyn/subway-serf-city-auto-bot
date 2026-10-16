"""Керування телефоном через adb (свайпи й тапи)."""
import re
import subprocess


def adb_cmd(serial=None):
    return ["adb", "-s", serial] if serial else ["adb"]


class AdbController:
    # напрямок свайпу (dx, dy) для кожної дії
    DIRECTIONS = {"left": (-1, 0), "right": (1, 0), "jump": (0, -1), "roll": (0, 1)}

    def __init__(self, cfg, serial=None):
        self.adb = adb_cmd(serial)
        self.cfg = cfg
        self.width, self.height = self._screen_size()
        # Один постійний `adb shell`: не платимо за нове з'єднання на кожен свайп.
        self.shell = subprocess.Popen(
            [*self.adb, "shell"], stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True)

    def _screen_size(self):
        out = subprocess.run([*self.adb, "shell", "wm", "size"],
                             capture_output=True, text=True, check=True).stdout
        # "Override size" (якщо є) іде після "Physical size", тому беремо останній збіг
        w, h = re.findall(r"(\d+)x(\d+)", out)[-1]
        return int(w), int(h)

    def _swipe(self, action):
        sw = self.cfg["swipe"]
        dx, dy = self.DIRECTIONS[action]
        x1, y1 = int(sw["origin"][0] * self.width), int(sw["origin"][1] * self.height)
        x2 = int(x1 + dx * sw["length"] * self.width)
        y2 = int(y1 + dy * sw["length"] * self.height)
        return f"input swipe {x1} {y1} {x2} {y2} {sw['duration_ms']}"

    def _command(self, action):
        if action == "board":
            x, y = self.cfg["board"]["tap"]
            return f"input tap {int(x * self.width)} {int(y * self.height)}"
        if action == "stomp":
            # стрибок, пауза в повітрі, удар вниз — усе виконує телефон, бот не чекає
            return f"{self._swipe('jump')}; sleep {self.cfg['stomp_delay']}; {self._swipe('roll')}"
        return self._swipe(action)

    def act(self, action):
        # `( ... ) &` — не чекаємо завершення `input`, щоб не блокувати shell
        self.shell.stdin.write(f"({self._command(action)}) &\n")
        self.shell.stdin.flush()

    def gesture(self, g):
        """Точне відтворення записаного жесту (route.Gesture) з його координатами."""
        x1, y1 = int(g.x1 * self.width), int(g.y1 * self.height)
        x2, y2 = int(g.x2 * self.width), int(g.y2 * self.height)
        cmd = (f"input tap {x1} {y1}" if g.action == "tap"
               else f"input swipe {x1} {y1} {x2} {y2} {g.duration_ms}")
        self.shell.stdin.write(f"({cmd}) &\n")
        self.shell.stdin.flush()

    def close(self):
        if self.shell.poll() is None:
            self.shell.stdin.close()
            self.shell.terminate()


class DryRunController:
    """Нічого не надсилає на телефон, лише друкує дії."""

    def act(self, action):
        print(f"[dry-run] {action}")

    def gesture(self, g):
        print(f"[dry-run] route {g.t:.2f}s {g.action}")

    def close(self):
        pass
