"""Візуалізація для налагодження: смуги, зони, детекції, рішення."""
import cv2

COLORS = {
    "obstacle_tall": (0, 0, 255), "obstacle_low": (0, 128, 255), "obstacle_high": (0, 255, 255),
    "coin": (0, 215, 255), "board_coin": (180, 180, 255), "energy": (255, 160, 0),
    "powerup": (255, 0, 255), "star": (255, 255, 255), "stomp_target": (0, 255, 0),
    "board_ready": (255, 255, 0), "player": (255, 128, 0),
}


def draw_overlay(frame, geo, scene, lane=None, action=None):
    out = frame.copy()
    h, w = out.shape[:2]
    for d in scene.detections:
        color = COLORS.get(d.cls, (200, 200, 200))
        cv2.rectangle(out, (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2)), color, 2)
        cv2.putText(out, f"{d.cls} {d.conf:.2f}", (int(d.x1), int(d.y1) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    for i, st in enumerate(scene.lanes):
        for zone in ("near", "far"):
            y0, y1 = geo.cfg["zones"][zone]
            blocked = getattr(st, zone) is not None
            color = (0, 0, 255) if blocked else (0, 255, 0)
            cv2.polylines(out, [geo.polygon(i, y0, y1, w, h)], True, color, 3 if i == lane else 1)
        x, y = geo.polygon(i, *geo.cfg["zones"]["near"], w, h)[3]
        label = f"{st.near or '-'}/{st.far or '-'} +{st.value:g}" + (" STOMP" if st.stomp else "")
        cv2.putText(out, label, (x, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    status = action.upper() if action else ""
    if scene.board_ready:
        status += "  [BOARD READY]"
    if status:
        cv2.putText(out, status, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
    return out
