"""Політика, навчена імітацією на відео проходжень (behavior cloning).

Модель дивиться на поточний і попередній кадр і вибирає дію: нічого,
вліво, вправо, стрибок, униз. На відміну від маршрутів, вона не прив'язана
до конкретного рівня, тож працює і в звичайному забігу, і на нових рівнях туру.
"""
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

SIZE = (48, 96)    # ширина, висота входу моделі
CROP_TOP = 0.12    # верх кадру з HUD відрізаємо: у відео й на телефоні він різний


def preprocess(frame):
    """Кадр будь-якої роздільності → uint8 (96, 48, 3): однаково для відео й телефона."""
    h = frame.shape[0]
    return cv2.resize(frame[int(CROP_TOP * h):], SIZE, interpolation=cv2.INTER_AREA)


def to_tensor(x, prev):
    """(N, 96, 48, 3) uint8 ×2 → (N, 6, 96, 48) float."""
    a = np.concatenate([x, prev], axis=-1).astype(np.float32) / 255.0
    return torch.from_numpy(np.ascontiguousarray(a.transpose(0, 3, 1, 2)))


class PolicyNet(nn.Module):
    def __init__(self, n_actions=5):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(),
                                 nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(),
                                 nn.MaxPool2d(2))

        self.features = nn.Sequential(block(6, 24), block(24, 48), block(48, 64), block(64, 96))
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(96 * 6 * 3, 128), nn.ReLU(),
                                  nn.Dropout(0.3), nn.Linear(128, n_actions))

    def forward(self, x):
        return self.head(self.features(x))


class Policy:
    """Обгортка для гри: кадр → дія (або None), з порогом упевненості й паузою між діями."""

    def __init__(self, path, threshold=0.6, cooldown=0.3, prev_gap=0.1):
        ckpt = torch.load(Path(path), map_location="cpu")
        self.actions = ckpt["actions"]
        self.net = PolicyNet(len(self.actions))
        self.net.load_state_dict(ckpt["state"])
        self.net.eval()
        self.threshold, self.cooldown, self.prev_gap = threshold, cooldown, prev_gap
        self.history = []          # (час, кадр) — щоб узяти кадр ~0,1 с тому
        self.busy_until = 0.0

    def decide(self, frame, now=None):
        now = time.monotonic() if now is None else now
        img = preprocess(frame)
        self.history = [(t, f) for t, f in self.history if now - t <= 0.5] + [(now, img)]
        prev = min(self.history, key=lambda h: abs(now - h[0] - self.prev_gap))[1]
        with torch.no_grad():
            probs = torch.softmax(self.net(to_tensor(img[None], prev[None])), 1)[0].numpy()
        k = int(probs.argmax())
        if k == 0 or probs[k] < self.threshold or now < self.busy_until:
            return None, probs
        self.busy_until = now + self.cooldown
        return self.actions[k], probs
