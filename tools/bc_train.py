"""Навчання політики (behavior cloning) на датасеті з rec/bc/.

  python tools/bc_train.py                 # тренування, модель → models/policy.pt
  python tools/bc_train.py --epochs 3      # швидкий пробний прогін

Вхід моделі — поточний і попередній кадр (6 каналів, 96×48), вихід — одна з
дій: нічого, вліво, вправо, стрибок, униз. Кілька відео відкладаються для
перевірки (інші рівні, ніж у навчанні), щоб чесно бачити узагальнення.
Метрика подій: чи модель «стріляє» потрібною дією в межах ±0,2 с від
справжньої і як часто стріляє без потреби.
"""
import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from bc_dataset import ACTIONS  # noqa: E402
from bot.policy import PolicyNet, to_tensor  # noqa: E402


def load(files):
    xs, ps, ys, runs = [], [], [], []
    for i, f in enumerate(files):
        d = np.load(f)
        xs.append(d["x"]), ps.append(d["prev"]), ys.append(d["y"])
        runs.append(d["run"].astype(np.int32) + 1000 * i)  # унікальний номер забігу
    return np.concatenate(xs), np.concatenate(ps), np.concatenate(ys), np.concatenate(runs)


def augment(xb, pb, rng):
    """Відео з YouTube і екран телефона виглядають по-різному (стиснення, різкість,
    кольори), тож кадри випадково псуються однаково для обох входів: модель
    має вчитися на сцені, а не на артефактах конкретного відео."""
    import cv2
    xb, pb = xb.astype(np.float32), pb.astype(np.float32)
    n = len(xb)
    gain = rng.uniform(0.8, 1.2, (n, 1, 1, 1)) * rng.uniform(0.9, 1.1, (n, 1, 1, 3))  # яскравість і баланс кольорів
    bias = rng.uniform(-15, 15, (n, 1, 1, 1))
    xb, pb = xb * gain + bias, pb * gain + bias
    noise = rng.normal(0, rng.uniform(0, 6), (n, 1, 1, 1))
    xb = xb + noise * rng.standard_normal(xb.shape)
    xb, pb = np.clip(xb, 0, 255).astype(np.uint8), np.clip(pb, 0, 255).astype(np.uint8)
    for i in range(n):
        if rng.random() < 0.4:  # розмиття, як у стиснутого відео
            k = int(rng.choice([3, 5]))
            xb[i], pb[i] = cv2.GaussianBlur(xb[i], (k, k), 0), cv2.GaussianBlur(pb[i], (k, k), 0)
        if rng.random() < 0.5:  # артефакти JPEG/H.264 різної якості
            q = [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(30, 90))]
            xb[i] = cv2.imdecode(cv2.imencode(".jpg", xb[i], q)[1], cv2.IMREAD_COLOR)
            pb[i] = cv2.imdecode(cv2.imencode(".jpg", pb[i], q)[1], cv2.IMREAD_COLOR)
    return xb, pb


def event_metrics(probs, y, runs, thr, window=2):
    """Подійні метрики: влучання дією в ±window кадрів і хибні спрацювання."""
    pred = probs.argmax(1)
    conf = probs.max(1)
    fire = (pred > 0) & (conf >= thr)
    hits, total, false = 0, 0, 0
    for i in np.flatnonzero(y > 0):
        if i > 0 and y[i - 1] == y[i] and runs[i - 1] == runs[i]:
            continue  # подія триває кілька кадрів — рахуємо раз
        total += 1
        lo, hi = max(0, i - window), min(len(y), i + window + 2)
        if any(fire[j] and pred[j] == y[i] for j in range(lo, hi) if runs[j] == runs[i]):
            hits += 1
    for j in np.flatnonzero(fire):
        lo, hi = max(0, j - window - 1), min(len(y), j + window + 1)
        if not any(y[k] == pred[j] for k in range(lo, hi)):
            false += 1
    return hits / max(total, 1), false / max(len(y), 1) * 10  # хибних на секунду (10 кадрів/с)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=str(ROOT / "rec" / "bc"))
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--val-videos", type=int, default=5)
    p.add_argument("--out", default=str(ROOT / "models" / "policy.pt"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--none-keep", type=float, default=0.25,
                   help="частка кадрів «нічого не робити» в навчанні (їх ~87%%, решта лише сповільнює)")
    args = p.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    files = sorted(Path(args.data).glob("*.npz"))
    random.shuffle(files)
    val_files, train_files = files[: args.val_videos], files[args.val_videos:]
    x, px, y, _ = load(train_files)
    keep = (y > 0) | (np.random.rand(len(y)) < args.none_keep)
    x, px, y = x[keep], px[keep], y[keep]
    vx, vpx, vy, vruns = load(val_files)
    print(f"навчання: {len(y)} кадрів з {len(train_files)} відео; перевірка: {len(vy)} кадрів з {len(val_files)} відео")
    counts = np.bincount(y, minlength=len(ACTIONS))
    print("мітки:", dict(zip(ACTIONS, counts.tolist())))

    # рідкісні дії важать більше, інакше модель завжди «нічого не робить»
    weights = torch.tensor((counts.sum() / np.maximum(counts, 1)) ** 0.5, dtype=torch.float32)
    weights /= weights.mean()
    model = PolicyNet(len(ACTIONS))
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=args.epochs * (len(y) // args.batch + 1))
    lossf = nn.CrossEntropyLoss(weight=weights)
    torch.set_num_threads(4)

    def predict(xx, pp):
        model.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(xx), 512):
                out.append(torch.softmax(model(to_tensor(xx[i:i + 512], pp[i:i + 512])), 1).numpy())
        return np.concatenate(out)

    best = (-1, None)
    for epoch in range(args.epochs):
        model.train()
        order = np.random.permutation(len(y))
        total = 0.0
        for i in range(0, len(order), args.batch):
            idx = order[i:i + args.batch]
            xb, pb = augment(x[idx], px[idx], rng)
            if random.random() < 0.5:  # гра симетрична: дзеркало міняє вліво ↔ вправо
                xb, pb = xb[:, :, ::-1].copy(), pb[:, :, ::-1].copy()
                yb = np.where(y[idx] == 1, 2, np.where(y[idx] == 2, 1, y[idx]))
            else:
                yb = y[idx]
            loss = lossf(model(to_tensor(xb, pb)), torch.from_numpy(yb.astype(np.int64)))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(idx)
        probs = predict(vx, vpx)
        hit, false = event_metrics(probs, vy, vruns, thr=0.5)
        print(f"епоха {epoch + 1}: loss {total / len(y):.3f}; перевірка: влучань {hit:.0%}, хибних {false:.2f}/с")
        score = hit - 0.5 * false
        if score > best[0]:
            best = (score, {k: v.clone() for k, v in model.state_dict().items()})

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": best[1], "actions": ACTIONS}, args.out)
    model.load_state_dict(best[1])
    probs = predict(vx, vpx)
    for thr in (0.4, 0.5, 0.6, 0.7, 0.8):
        hit, false = event_metrics(probs, vy, vruns, thr)
        print(f"  поріг {thr}: влучань {hit:.0%}, хибних {false:.2f}/с")
    print(f"Модель збережено: {args.out}")


if __name__ == "__main__":
    main()
