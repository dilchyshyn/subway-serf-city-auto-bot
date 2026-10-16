import copy
import json
from pathlib import Path

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config.json"

# Секції, які залежать від району і тому зберігаються в профілі
PROFILE_KEYS = ("lanes", "zones", "heuristic")


def _merge(base, override):
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _merge(base[k], v)
        else:
            base[k] = copy.deepcopy(v)
    return base


def load_config(path=None, profile=None):
    """Завантажує конфіг; profile (напр. "southline") накладає налаштування району."""
    with open(path or DEFAULT_CONFIG, encoding="utf-8") as f:
        cfg = json.load(f)
    if profile:
        profiles = cfg.get("profiles", {})
        if profile not in profiles:
            raise KeyError(f"Профіль '{profile}' не знайдено. Є: {', '.join(profiles) or 'жодного'}")
        _merge(cfg, profiles[profile])
    return cfg


def save_config(cfg, path=None, profile=None):
    """Зберігає конфіг. З profile записує району-залежні секції в profiles[profile]."""
    path = path or DEFAULT_CONFIG
    if profile:
        with open(path, encoding="utf-8") as f:
            on_disk = json.load(f)
        on_disk.setdefault("profiles", {})[profile] = {k: cfg[k] for k in PROFILE_KEYS}
        cfg = on_disk
    else:
        cfg = {k: v for k, v in cfg.items()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
        f.write("\n")
