"""Джерела кадрів з екрана телефону. Жодне не потребує root.

screenrecord — `adb exec-out screenrecord` віддає H.264-потік, PyAV його
               декодує. Потрібен лише adb (за замовчуванням).
v4l2         — scrcpy пише відео у віртуальну камеру (v4l2loopback). Трохи
               швидше, але потрібні scrcpy, модуль ядра й sudo.
screencap    — `adb exec-out screencap -p`. Повільно (2-5 FPS), але надійно;
               підходить для калібрування.
image        — статична картинка або папка з картинками, для офлайн-налагодження.
"""
import re
import subprocess
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from .controls import adb_cmd

BACKENDS = ("screenrecord", "v4l2", "screencap", "image")


class _LatestFrame:
    """Окремий потік читає кадри й тримає лише останній, щоб бот не відставав через буфер."""

    def __init__(self):
        self._frame = None
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._reader, daemon=True)

    def _put(self, frame):
        with self._lock:
            self._frame = frame

    def read(self):
        with self._lock:
            frame, self._frame = self._frame, None
        return frame


class ScreenrecordCapture(_LatestFrame):
    def __init__(self, serial=None, width=544, cmd=None):
        super().__init__()
        self.adb = adb_cmd(serial)
        self.cmd = cmd or self._command(width)
        self.proc = None
        self._thread.start()

    def _command(self, width):
        out = subprocess.run([*self.adb, "shell", "wm", "size"],
                             capture_output=True, text=True, check=True).stdout
        w, h = map(int, re.findall(r"(\d+)x(\d+)", out)[-1])
        # Апаратні кодеки люблять розміри, кратні 16; зменшення кадру прискорює і кодек, і бота
        height = round(width * h / w / 16) * 16
        return [*self.adb, "exec-out", "screenrecord", "--output-format=h264",
                f"--size={width}x{height}", "--bit-rate=6000000", "-"]

    def _reader(self):
        import av  # потрібен лише для цього бекенду

        while self._running:
            self.proc = subprocess.Popen(self.cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            try:
                with av.open(self.proc.stdout, format="h264",
                             options={"flags": "low_delay"}) as container:
                    for frame in container.decode(video=0):
                        if not self._running:
                            break
                        self._put(frame.to_ndarray(format="bgr24"))
            except (av.error.FFmpegError, OSError):
                pass
            finally:
                self.proc.kill()
            # screenrecord сам зупиняється через 3 хвилини — перезапускаємо
            if self._running:
                time.sleep(0.2)

    def close(self):
        self._running = False
        if self.proc:
            self.proc.kill()
        self._thread.join(timeout=2)


class V4L2Capture(_LatestFrame):
    def __init__(self, device):
        super().__init__()
        self.cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError(f"Не вдалося відкрити {device}. Чи запущено scrcpy з --v4l2-sink? "
                               "Або спробуй --backend screenrecord")
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._thread.start()

    def _reader(self):
        while self._running:
            ok, frame = self.cap.read()
            if ok:
                self._put(frame)

    def close(self):
        self._running = False
        self._thread.join(timeout=1)
        self.cap.release()


class ScreencapCapture:
    def __init__(self, serial=None):
        self.adb = adb_cmd(serial)

    def read(self):
        png = subprocess.run([*self.adb, "exec-out", "screencap", "-p"],
                             capture_output=True, check=True).stdout
        return cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)

    def close(self):
        pass


class ImageCapture:
    def __init__(self, path):
        p = Path(path)
        self.files = sorted(p.glob("*.png")) + sorted(p.glob("*.jpg")) if p.is_dir() else [p]
        if not self.files:
            raise RuntimeError(f"Немає картинок у {path}")
        self.i = 0

    def read(self):
        frame = cv2.imread(str(self.files[self.i % len(self.files)]))
        self.i += 1
        return frame

    def close(self):
        pass


def open_capture(backend, device=None, serial=None, image=None):
    if backend == "screenrecord":
        return ScreenrecordCapture(serial)
    if backend == "v4l2":
        return V4L2Capture(device)
    if backend == "screencap":
        return ScreencapCapture(serial)
    if backend == "image":
        return ImageCapture(image)
    raise ValueError(f"Невідомий backend: {backend}")
