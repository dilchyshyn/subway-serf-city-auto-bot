"""Логіка прийняття рішень.

Пріоритети:
1. Вижити. Якщо попереду перешкода, перейти у вільну сусідню смугу (серед
   вільних обрати ту, де більше предметів). Якщо обидві зайняті, стрибнути
   або пірнути.
2. Заздалегідь піти зі смуги, де далі стоїть перешкода, якщо сусідня чиста.
3. Увімкнути ховерборд, щойно шкала повна (він ще й працює як друге життя).
4. Stomp-ціль просто попереду: стрибнути й ударити вниз, щоб випали монети.
5. Збирати предмети. Якщо поточна смуга безпечна, а в чистій сусідній
   предметів суттєво більше, перейти туди. Ваги: пауер-ап > Energy >
   звичайні монети > монети для бордів.
"""
from .classes import HIGH

LEFT, RIGHT, JUMP, ROLL, STOMP, BOARD = "left", "right", "jump", "roll", "stomp", "board"


class Brain:
    def __init__(self, cfg, lane=1):
        self.cfg = cfg
        self.lane = lane  # 0 = ліва, 1 = середня, 2 = права
        self._busy_until = float("-inf")
        self._board_until = float("-inf")

    def reset(self, lane=1):
        self.__init__(self.cfg, lane)

    def _cost(self, st):
        # Менше = краще: небезпека мінус бонус за предмети
        danger = (st.near is not None) + self.cfg["far_weight"] * (st.far is not None)
        return danger - self.cfg["pickup_bias"] * st.value

    def _move_towards(self, target):
        action = LEFT if target < self.lane else RIGHT
        self.lane = target
        return action

    def emergency(self, scene, now):
        """Лише стрибок/пірнання перед перешкодою, без зміни смуги (для режиму маршруту)."""
        cur = scene.lanes[self.lane]
        if now < self._busy_until or cur.near is None:
            return None
        self._busy_until = now + self.cfg["action_cooldown"]
        return ROLL if cur.near == HIGH else JUMP

    def decide(self, scene, now):
        if now < self._busy_until:
            return None
        lanes = scene.lanes
        cur = lanes[self.lane]
        neighbours = [l for l in (self.lane - 1, self.lane + 1) if 0 <= l <= 2]
        clear = [l for l in neighbours if lanes[l].near is None and lanes[l].far is None]
        cooldown = self.cfg["action_cooldown"]

        action = None
        if cur.near is not None:
            free = [l for l in neighbours if lanes[l].near is None]
            if free:
                action = self._move_towards(min(free, key=lambda l: self._cost(lanes[l])))
            else:
                action = ROLL if cur.near == HIGH else JUMP
        elif cur.far is not None and clear:
            action = self._move_towards(min(clear, key=lambda l: self._cost(lanes[l])))
        elif scene.board_ready and now >= self._board_until:
            action = BOARD
            self._board_until = now + self.cfg["board"]["cooldown"]
        elif cur.stomp:
            action = STOMP
            cooldown = self.cfg["stomp_cooldown"]
        elif clear:
            # Поріг min_gain не дає боту смикатися між смугами через одну-дві монети
            best = max(clear, key=lambda l: lanes[l].value)
            if lanes[best].value - cur.value >= self.cfg["pickup_min_gain"]:
                action = self._move_towards(best)

        if action:
            self._busy_until = now + cooldown
        return action
