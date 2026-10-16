"""Класи об'єктів Subway Surfers City. Порядок = індекси класів у YOLO-датасеті."""

CLASSES = [
    "obstacle_tall",  # вагон, контейнер, навантажувач — тільки обходити
    "obstacle_low",   # низька перешкода — перестрибнути
    "obstacle_high",  # планка зверху — пірнути (свайп вниз)
    "coin",           # звичайна монета
    "board_coin",     # монета для бордів
    "energy",         # синя Board Energy, заряджає ховерборд
    "powerup",        # Magnet, Bubblegum, Batteries, Coin doubler, Board charge
    "star",           # Secret Star
    "stomp_target",   # об'єкт, на який стрибають і б'ють stomp, щоб випали монети
    "board_ready",    # HUD: шкала ховерборда повна, можна тапнути
]

TALL, LOW, HIGH = "tall", "low", "high"
OBSTACLE_KIND = {"obstacle_tall": TALL, "obstacle_low": LOW, "obstacle_high": HIGH}
HUD_CLASSES = {"board_ready"}
