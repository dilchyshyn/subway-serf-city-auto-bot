"""Перетворення списку детекцій на стан трьох смуг, з яким працює Brain."""
from dataclasses import dataclass, field

from .classes import HUD_CLASSES, OBSTACLE_KIND


@dataclass
class Detection:
    cls: str
    conf: float
    x1: float  # пікселі
    y1: float
    x2: float
    y2: float


@dataclass
class LaneState:
    near: str | None = None   # тип перешкоди в near-зоні (tall/low/high) або None
    far: str | None = None    # те саме у far-зоні
    value: float = 0.0        # сумарна цінність предметів у смузі
    items: dict = field(default_factory=dict)
    stomp: bool = False       # stomp-ціль у near-зоні: час стрибати й бити вниз


@dataclass
class Scene:
    lanes: list
    board_ready: bool = False
    detections: list = field(default_factory=list)
    player_lane: int | None = None  # смуга, де модель бачить персонажа


def build_scene(detections, w, h, geo, cfg):
    weights = cfg["weights"]
    lanes = [LaneState() for _ in range(3)]
    closest = [{"near": -1.0, "far": -1.0} for _ in range(3)]  # y низу найближчої перешкоди
    board_ready = False
    player_lane, player_conf = None, 0.0

    for d in detections:
        if d.cls == "player":
            if d.conf > player_conf:
                player_conf = d.conf
                player_lane = geo.nearest_lane((d.x1 + d.x2) / 2 / w, d.y2 / h, tolerance=2.0)
            continue
        if d.cls in HUD_CLASSES:
            board_ready = board_ready or d.cls == "board_ready"
            continue
        x1, x2 = d.x1 / w, d.x2 / w
        bottom = d.y2 / h  # низ рамки = точка контакту із землею = глибина

        if d.cls in OBSTACLE_KIND:
            zone = geo.zone_of(bottom)
            if zone is None:
                continue
            for lane in geo.lanes_spanning(x1, x2, bottom):
                # якщо в зоні кілька перешкод, важить найближча (найнижча на екрані)
                if bottom > closest[lane][zone]:
                    closest[lane][zone] = bottom
                    setattr(lanes[lane], zone, OBSTACLE_KIND[d.cls])
            continue

        cx, cy = (x1 + x2) / 2, (d.y1 + d.y2) / 2 / h
        lane = geo.nearest_lane(cx, bottom if d.cls == "stomp_target" else cy)
        if lane is None:
            continue
        if d.cls == "stomp_target" and geo.zone_of(bottom, ("near",)):
            lanes[lane].stomp = True
        if geo.zone_of(cy, ("pickup",)) and d.cls in weights:
            st = lanes[lane]
            st.items[d.cls] = st.items.get(d.cls, 0) + 1
            st.value += weights[d.cls]

    return Scene(lanes, board_ready, detections, player_lane)
