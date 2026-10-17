"""Тести без телефону: python -m unittest discover tests"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bot import load_config, save_config  # noqa: E402
from bot import route  # noqa: E402
from bot.brain import BOARD, JUMP, LEFT, RIGHT, ROLL, STOMP, Brain  # noqa: E402
from bot.classes import HIGH, LOW, TALL  # noqa: E402
from bot.geometry import LaneGeometry  # noqa: E402
from bot.heuristic import HeuristicDetector  # noqa: E402
from bot.scene import Detection, LaneState, Scene, build_scene  # noqa: E402

W, H = 540, 1200
GROUND = (90, 100, 110)   # тьмяна дорога, BGR
COIN = (0, 215, 255)      # золото
ENERGY = (255, 120, 0)    # синій
POWERUP = (200, 0, 230)   # рожево-пурпуровий


def box_at(geo, cls, lane, y, bw=0.06, bh=0.05, bottom=True):
    """Детекція по центру смуги; y — низ рамки (bottom=True) або її центр."""
    cx = geo.center_x(lane, y)
    y2 = y if bottom else y + bh / 2
    return Detection(cls, 0.9, (cx - bw / 2) * W, (y2 - bh) * H, (cx + bw / 2) * W, y2 * H)


class SceneTest(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config()
        self.geo = LaneGeometry(self.cfg)

    def scene(self, dets):
        return build_scene(dets, W, H, self.geo, self.cfg)

    def test_obstacle_zones(self):
        s = self.scene([box_at(self.geo, "obstacle_tall", 1, 0.65),
                        box_at(self.geo, "obstacle_low", 0, 0.50)])
        self.assertEqual(s.lanes[1].near, TALL)
        self.assertIsNone(s.lanes[1].far)
        self.assertEqual(s.lanes[0].far, LOW)
        self.assertIsNone(s.lanes[2].near)

    def test_wide_obstacle_blocks_several_lanes(self):
        y = 0.65
        x1, x2 = self.geo.center_x(0, y) - 0.02, self.geo.center_x(1, y) + 0.02
        s = self.scene([Detection("obstacle_high", 0.9, x1 * W, (y - 0.03) * H, x2 * W, y * H)])
        self.assertEqual([l.near for l in s.lanes], [HIGH, HIGH, None])

    def test_closest_obstacle_wins(self):
        s = self.scene([box_at(self.geo, "obstacle_low", 1, 0.58),
                        box_at(self.geo, "obstacle_high", 1, 0.70)])
        self.assertEqual(s.lanes[1].near, HIGH)

    def test_pickup_values(self):
        w = self.cfg["weights"]
        s = self.scene([box_at(self.geo, "coin", 0, 0.6, 0.02, 0.02, bottom=False),
                        box_at(self.geo, "coin", 0, 0.65, 0.02, 0.02, bottom=False),
                        box_at(self.geo, "board_coin", 1, 0.6, 0.02, 0.02, bottom=False),
                        box_at(self.geo, "powerup", 2, 0.6, 0.04, 0.04, bottom=False)])
        self.assertEqual(s.lanes[0].items, {"coin": 2})
        self.assertEqual(s.lanes[0].value, 2 * w["coin"])
        self.assertEqual(s.lanes[1].value, w["board_coin"])
        self.assertGreater(w["coin"], w["board_coin"])  # звичайні монети в пріоритеті
        self.assertEqual(s.lanes[2].value, w["powerup"])

    def test_stomp_and_board(self):
        s = self.scene([box_at(self.geo, "stomp_target", 2, 0.66),
                        Detection("board_ready", 0.9, 10, 10, 60, 60)])
        self.assertTrue(s.lanes[2].stomp)
        self.assertTrue(s.board_ready)
        far = self.scene([box_at(self.geo, "stomp_target", 2, 0.45)])
        self.assertFalse(far.lanes[2].stomp)
        self.assertGreater(far.lanes[2].value, 0)  # але в смугу з нею варто перейти

    def test_player_is_not_an_obstacle(self):
        s = self.scene([box_at(self.geo, "player", 1, 0.80, 0.08, 0.12)])
        self.assertTrue(all(l.value == 0 and l.near is None for l in s.lanes))

    def test_drone_and_bubble_are_jump_targets(self):
        off = self.cfg["float_offset"]

        def floating(cls, lane, depth, bw, bh):
            # висить на off вище за точку на дорозі з глибиною depth
            cx = self.geo.center_x(lane, depth)
            y2 = depth - off
            return Detection(cls, 0.9, (cx - bw / 2) * W, (y2 - bh) * H, (cx + bw / 2) * W, y2 * H)

        s = self.scene([floating("drone", 1, 0.62, 0.10, 0.04),
                        floating("bounce_bubble", 2, 0.45, 0.06, 0.06)])
        self.assertTrue(s.lanes[1].jump_target)
        self.assertFalse(s.lanes[2].jump_target)          # бабл ще далеко
        self.assertEqual(s.lanes[2].value, self.cfg["weights"]["bounce_bubble"])

    def test_off_road_pickup_ignored(self):
        s = self.scene([Detection("coin", 0.9, 0, 0.6 * H, 10, 0.6 * H + 10)])
        self.assertTrue(all(l.value == 0 for l in s.lanes))


def sc(*lanes, board=False):
    return Scene(list(lanes), board)


def st(near=None, far=None, value=0.0, stomp=False):
    return LaneState(near, far, value, {}, stomp)


class BrainTest(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config()

    def test_switch_to_free_neighbour(self):
        b = Brain(self.cfg, lane=1)
        self.assertEqual(b.decide(sc(st(TALL, TALL), st(TALL, TALL), st()), now=0), RIGHT)
        self.assertEqual(b.lane, 2)

    def test_prefer_less_dangerous_neighbour(self):
        b = Brain(self.cfg, lane=1)
        self.assertEqual(b.decide(sc(st(far=TALL), st(TALL), st()), now=0), RIGHT)

    def test_dodge_towards_coins(self):
        b = Brain(self.cfg, lane=1)
        self.assertEqual(b.decide(sc(st(value=6), st(TALL), st()), now=0), LEFT)

    def test_jump_or_roll_when_boxed_in(self):
        # у лівій смузі: зліва стіна (екранна смуга 0 ігнорується), справа вагон
        b = Brain(self.cfg, lane=0)
        self.assertEqual(b.decide(sc(st(), st(LOW), st(TALL)), now=0), JUMP)
        b = Brain(self.cfg, lane=0)
        self.assertEqual(b.decide(sc(st(), st(HIGH), st(TALL)), now=0), ROLL)

    def test_never_swipe_into_wall(self):
        b = Brain(self.cfg, lane=2)
        # справа від крайньої правої смуги — стіна, навіть якщо там «порожньо»
        self.assertEqual(b.decide(sc(st(TALL), st(TALL), st()), now=0), JUMP)
        b = Brain(self.cfg, lane=2)
        self.assertEqual(b.decide(sc(st(), st(TALL), st()), now=0), LEFT)
        self.assertEqual(b.lane, 1)

    def test_lanes_are_relative_to_player(self):
        # після переходу вліво екранна «права» смуга — це колишня поточна
        b = Brain(self.cfg, lane=1)
        self.assertEqual(b.decide(sc(st(), st(TALL), st(TALL)), now=0), LEFT)
        self.assertEqual(b.decide(sc(st(), st(), st(value=10)), now=1), RIGHT)

    def test_cooldown(self):
        b = Brain(self.cfg, lane=1)
        self.assertIsNotNone(b.decide(sc(st(), st(TALL), st()), now=0))
        self.assertIsNone(b.decide(sc(st(TALL), st(), st(TALL)), now=0.1))

    def test_early_switch_only_to_clear_lane(self):
        b = Brain(self.cfg, lane=1)
        self.assertIsNone(b.decide(sc(st(far=LOW), st(far=TALL), st(far=LOW)), now=0))
        self.assertEqual(b.decide(sc(st(far=LOW), st(far=TALL), st()), now=0), RIGHT)

    def test_board_when_ready(self):
        b = Brain(self.cfg, lane=1)
        self.assertEqual(b.decide(sc(st(), st(), st(), board=True), now=0), BOARD)
        # не тапати повторно, поки триває cooldown ховерборда
        self.assertIsNone(b.decide(sc(st(), st(), st(), board=True), now=1))

    def test_survival_before_board(self):
        b = Brain(self.cfg, lane=1)
        self.assertEqual(b.decide(sc(st(), st(TALL), st(TALL), board=True), now=0), LEFT)

    def test_stomp(self):
        b = Brain(self.cfg, lane=1)
        self.assertEqual(b.decide(sc(st(), st(stomp=True), st()), now=0), STOMP)
        self.assertIsNone(b.decide(sc(st(TALL), st(TALL), st()), now=self.cfg["action_cooldown"] + 0.01))

    def test_jump_to_drone(self):
        b = Brain(self.cfg, lane=1)
        jt = LaneState(jump_target=True)
        self.assertEqual(b.decide(sc(st(), jt, st()), now=0), JUMP)

    def test_chase_powerup_over_coins(self):
        b = Brain(self.cfg, lane=1)
        self.assertEqual(b.decide(sc(st(value=4), st(value=1), st(value=10)), now=0), RIGHT)

    def test_board_coins_when_nothing_else(self):
        w = self.cfg["weights"]["board_coin"]
        b = Brain(self.cfg, lane=1)
        self.assertEqual(b.decide(sc(st(value=5 * w), st(), st()), now=0), LEFT)

    def test_ignore_small_gain(self):
        b = Brain(self.cfg, lane=1)
        self.assertIsNone(b.decide(sc(st(value=2), st(), st()), now=0))

    def test_never_chase_into_danger(self):
        b = Brain(self.cfg, lane=1)
        self.assertIsNone(b.decide(sc(st(far=LOW, value=10), st(), st(TALL, value=10)), now=0))

    def test_emergency_keeps_lane(self):
        b = Brain(self.cfg, lane=1)
        self.assertEqual(b.emergency(sc(st(), st(LOW), st()), now=0), JUMP)
        self.assertEqual(b.lane, 1)
        self.assertIsNone(b.emergency(sc(st(TALL), st(), st()), now=1))


class HeuristicTest(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config()
        self.geo = LaneGeometry(self.cfg)
        self.det = HeuristicDetector(self.cfg, self.geo)

    def frame(self):
        return np.full((H, W, 3), GROUND, np.uint8)

    def center(self, lane, y):
        return int(self.geo.center_x(lane, y) * W), int(y * H)

    def test_empty(self):
        self.assertEqual(self.det.detect(self.frame()), [])

    def test_pickups_and_train(self):
        img = self.frame()
        for y in (0.5, 0.58, 0.66):
            cv2.circle(img, self.center(0, y), 8, COIN, -1)
        cv2.circle(img, self.center(2, 0.6), 12, ENERGY, -1)
        cv2.circle(img, self.center(2, 0.5), 22, POWERUP, -1)
        cv2.fillPoly(img, [self.geo.polygon(1, 0.45, 0.70, W, H)], (0, 0, 220))
        dets = self.det.detect(img)
        names = sorted(d.cls for d in dets)
        self.assertEqual(names, ["coin"] * 3 + ["energy", "obstacle_tall", "powerup"])
        s = build_scene(dets, W, H, self.geo, self.cfg)
        self.assertEqual(s.lanes[1].near, TALL)
        self.assertEqual(s.lanes[0].items, {"coin": 3})

    def test_big_yellow_train_is_obstacle(self):
        img = self.frame()
        cv2.fillPoly(img, [self.geo.polygon(1, 0.45, 0.70, W, H)], COIN)
        self.assertEqual([d.cls for d in self.det.detect(img)], ["obstacle_tall"])


GETEVENT = """\
[   100.000000] /dev/input/event3: EV_ABS       ABS_MT_TRACKING_ID   00000001
[   100.000000] /dev/input/event3: EV_ABS       ABS_MT_POSITION_X    00000262
[   100.000000] /dev/input/event3: EV_ABS       ABS_MT_POSITION_Y    00000a00
[   100.050000] /dev/input/event3: EV_ABS       ABS_MT_TRACKING_ID   ffffffff
[   101.500000] /dev/input/event3: EV_ABS       ABS_MT_TRACKING_ID   00000002
[   101.500000] /dev/input/event3: EV_ABS       ABS_MT_POSITION_X    00000262
[   101.500000] /dev/input/event3: EV_ABS       ABS_MT_POSITION_Y    00000600
[   101.520000] /dev/input/event3: EV_ABS       ABS_MT_POSITION_X    00000100
[   101.540000] /dev/input/event3: EV_SYN       SYN_REPORT           00000000
[   101.560000] /dev/input/event3: EV_ABS       ABS_MT_TRACKING_ID   ffffffff
[   102.000000] /dev/input/event3: EV_ABS       ABS_MT_TRACKING_ID   00000003
[   102.000000] /dev/input/event3: EV_ABS       ABS_MT_POSITION_X    00000262
[   102.000000] /dev/input/event3: EV_ABS       ABS_MT_POSITION_Y    00000600
[   102.030000] /dev/input/event3: EV_ABS       ABS_MT_POSITION_Y    00000300
[   102.060000] /dev/input/event3: EV_ABS       ABS_MT_TRACKING_ID   ffffffff
"""


class RouteTest(unittest.TestCase):
    def test_parse_getevent(self):
        gestures = list(route.parse_getevent(GETEVENT.splitlines(), 1220, 2712))
        self.assertEqual([g.action for g in gestures], ["tap", "left", "jump"])
        self.assertAlmostEqual(gestures[1].t - gestures[0].t, 1.5)
        self.assertEqual(gestures[1].duration_ms, 60)
        self.assertAlmostEqual(gestures[0].x1, 0x262 / 1220)

    def test_save_load_and_play(self):
        gestures = list(route.parse_getevent(GETEVENT.splitlines(), 1220, 2712))
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "lvl.json"
            route.save(path, "docks_01", gestures)
            level, loaded, start = route.load(path)
        self.assertEqual(level, "docks_01")
        self.assertEqual(start, "tap")
        self.assertEqual([g.t for g in loaded], [0.0, 1.5, 2.0])

        player = route.RoutePlayer(loaded, start_time=10.0, offset=0.3)
        self.assertEqual([g.action for g in player.due(10.0)], ["tap"])   # тап «Грати» — одразу
        self.assertEqual(player.due(11.7), [])                             # 1.5 + 0.3 ще не настало
        self.assertEqual([g.action for g in player.due(11.8)], ["left"])
        self.assertAlmostEqual(player.next_in(11.8), 0.5)
        self.assertEqual([g.action for g in player.due(20)], ["jump"])
        self.assertTrue(player.finished)

    def test_actions_from_video_track(self):
        lanes_x = {0: 0.25, 1: 0.5, 2: 0.75}
        track = []
        for i in range(int(6 / 0.033)):
            t = i * 0.033
            lane = 1 if t < 1.0 else 0 if t < 4.0 else 1
            bottom, height = 0.85, 0.12
            if 2.0 <= t < 2.6:                       # стрибок: низ рамки піднімається
                bottom -= 0.15 * np.sin((t - 2.0) / 0.6 * np.pi)
            if 3.0 <= t < 3.5:                       # пірнання: рамка нижча
                height = 0.06
            x = lanes_x[lane]
            track.append((t, (x - 0.04, bottom - height, x + 0.04, bottom)))
            if i % 40 == 0:
                track.append((t + 0.01, None))       # персонаж іноді не знайдений
        gestures = route.actions_from_track(track, load_config()["swipe"])
        self.assertEqual([g.action for g in gestures], ["left", "jump", "roll", "right"])
        expected = [1.0, 2.0, 3.0, 4.0]
        for g, t in zip(gestures, expected):
            self.assertLess(abs(g.t - t), 0.25, (g.action, g.t))

    def test_stomp_needs_short_air_and_hard_fall(self):
        def jump_track(fall_time):
            track = []
            for i in range(int(4 / 0.033)):
                t = i * 0.033
                bottom = 0.85
                if 1.0 <= t < 1.3:                    # злёт
                    bottom -= 0.4 * (t - 1.0) / 0.3
                elif 1.3 <= t < 1.3 + fall_time:      # падіння
                    bottom -= 0.4 * (1 - (t - 1.3) / fall_time)
                track.append((t, (0.45, bottom - 0.2, 0.55, bottom)))
            return track
        cfg = load_config()
        kw = dict(lanes=False, stomp_max_air=0.65, stomp_fall=3.0, stomp_shrink=0.6)
        stomp = route.actions_from_track(jump_track(0.12), cfg["swipe"], **kw)
        normal = route.actions_from_track(jump_track(0.6), cfg["swipe"], **kw)
        self.assertEqual([g.action for g in stomp], ["jump", "roll"])
        self.assertEqual([g.action for g in normal], ["jump"])

    def test_merge_boxes(self):
        full, torso, other = (0.2, 0.7, 0.3, 0.85), (0.21, 0.72, 0.29, 0.8), (0.7, 0.7, 0.8, 0.85)
        self.assertEqual(route.merge_boxes([(0.6, full), (0.8, torso), (0.5, other)]), full)
        self.assertIsNone(route.merge_boxes([]))

    def test_manual_route_player(self):
        gestures = route.actions_from_track([], load_config()["swipe"])
        self.assertEqual(gestures, [])
        g = route.Gesture(1.0, "left", 0.5, 0.6, 0.25, 0.6, 40)
        player = route.RoutePlayer([g], start_time=0, offset=0.5, first_immediate=False)
        self.assertEqual(player.due(1.2), [])          # offset діє і на перший жест
        self.assertEqual(player.due(1.5), [g])

    def test_classify(self):
        self.assertEqual(route.classify(0.5, 0.5, 0.51, 0.5), "tap")
        self.assertEqual(route.classify(0.5, 0.5, 0.8, 0.55), "right")
        self.assertEqual(route.classify(0.5, 0.5, 0.5, 0.8), "roll")


# Сплески бічного руху, виміряні на справжньому відео (The Docks 1.1, Zero Mode)
PILOT_BURSTS = [(10.53, -196), (11.87, 176), (13.20, 7), (15.20, -199), (16.73, 404), (18.83, -406),
                (21.60, 194), (22.87, -196), (24.27, 204), (25.10, -198), (26.10, 194), (31.17, 200),
                (32.20, -205), (32.87, -55), (35.13, -145), (38.03, 55), (39.83, 179), (42.37, -204),
                (44.43, -187), (46.17, 179), (48.00, 191), (48.83, -387), (50.63, 192)]


class VideoTest(unittest.TestCase):
    def test_segment_runs(self):
        from bot.video import segment_runs
        times = [i * 0.5 for i in range(120)]
        # рівень 1 з 2-секундною діркою (HUD під ефектом), короткий шматок меню, рівень 2
        flags = [3 <= t < 12 or 14 <= t < 20 or 30 <= t < 32 or 40 <= t < 55 for t in times]
        self.assertEqual(segment_runs(times, flags), [(3.0, 19.5), (40.0, 54.5)])

    def test_lane_events_on_pilot(self):
        from bot.video import lane_events
        bursts = [(t, t + 0.4, s) for t, s in PILOT_BURSTS]
        swipes, dropped = lane_events(bursts, width=288)
        self.assertEqual(dropped, [35.13])   # стрибок Bubblegum на дах, не зміна смуги
        lane = 1
        for _, a in swipes:
            lane += 1 if a == "right" else -1
            self.assertTrue(0 <= lane <= 2)
        self.assertEqual(len(swipes), 22)    # 16 одинарних + 3 подвійних сплески

    def test_hud_visible(self):
        from bot.video import hud_visible
        img = np.full((640, 288, 3), (40, 90, 200), np.uint8)
        self.assertFalse(hud_visible(img))
        x, y, s = int(0.03 * 288), int(0.05 * 640), 30             # кнопка паузи, зсунута вниз
        cv2.rectangle(img, (x, y), (x + s, y + s), (0, 210, 255), -1)
        self.assertFalse(hud_visible(img))                         # жовтий квадрат без смужок — не пауза
        cv2.rectangle(img, (x + 8, y + 6), (x + 12, y + s - 6), (20, 20, 20), -1)
        cv2.rectangle(img, (x + 18, y + 6), (x + 22, y + s - 6), (20, 20, 20), -1)
        self.assertTrue(hud_visible(img))


# Текст завдань, прочитаний OCR з карток справжнього відео (Phantom Park 11-15)
CARD_TEXTS = {
    "stomp8_a": "stompon8scoro75000collect3burledoachosfrophles",
    "stomp8_b": "stompon8scoro75000collect3burledcachestrophles",
    "pop13_a": "pop13loxloscorebao00collnot3bubblostrophilos",
    "pop13_b": "pop13toxloscore68000collect3bubblostrophies",
    "pop19": "pop19toxicscoro57000collect3bubblosbubblogums",
    "stomp14": "bounoytarpsstompon14so0ro28800collect13colnsacks",
}


class CardTextTest(unittest.TestCase):
    def test_same_level_despite_ocr_noise(self):
        from bot.video import text_similarity
        from bot.library import SAME_LEVEL
        self.assertGreaterEqual(text_similarity(CARD_TEXTS["stomp8_a"], CARD_TEXTS["stomp8_b"]), SAME_LEVEL)
        self.assertGreaterEqual(text_similarity(CARD_TEXTS["pop13_a"], CARD_TEXTS["pop13_b"]), SAME_LEVEL)

    def test_levels_differ_by_numbers(self):
        from bot.video import text_similarity
        from bot.library import SAME_LEVEL
        for a, b in (("pop13_b", "pop19"), ("stomp8_a", "stomp14"), ("pop19", "stomp8_b")):
            self.assertLess(text_similarity(CARD_TEXTS[a], CARD_TEXTS[b]), SAME_LEVEL, (a, b))

    def test_close_scores_are_different_levels(self):
        # живий OCR з телефона проти маршруту з іншим рівнем: 14000 ≠ 12000
        from bot.video import text_similarity, common_numbers
        live = "1500collect165score14000collecticoinstrophies"
        other = "conootrocoinscon12000collectsnckstrophlas"
        self.assertEqual(common_numbers(live, other), 0)
        self.assertLess(text_similarity(live, other), 0.4)

    def test_phone_ocr_reads_one_as_i(self):
        # справжня картка Phantom Park 3 з телефона проти маршруту з відео
        from bot.video import common_numbers, text_similarity
        from bot.library import SAME_LEVEL
        phone = "triggeri0scorei2650pop2toxicbubblessteamvents"
        video_route = "triggor10scoro12650pop12toxiobubblos"
        self.assertGreaterEqual(common_numbers(phone, video_route), 2)
        self.assertGreaterEqual(text_similarity(phone, video_route), SAME_LEVEL)

    def test_card_state_reads_green_rings(self):
        from bot.video import card_state
        band = np.full((120, 300, 3), (120, 60, 20), np.uint8)       # синя картка
        cv2.circle(band, (250, 40), 25, (60, 200, 60), 6)          # третє завдання виконане
        self.assertEqual(card_state(band), (0, 0, 1))


class PolicyTest(unittest.TestCase):
    def test_same_input_for_video_and_phone(self):
        from bot.policy import preprocess
        video = np.zeros((640, 288, 3), np.uint8)
        phone = np.zeros((1216, 544, 3), np.uint8)
        self.assertEqual(preprocess(video).shape, preprocess(phone).shape)

    def test_net_outputs_one_score_per_action(self):
        from bot.policy import PolicyNet, preprocess, to_tensor
        img = preprocess(np.zeros((640, 288, 3), np.uint8))[None]
        out = PolicyNet(5)(to_tensor(np.repeat(img, 2, 0), np.repeat(img, 2, 0)))
        self.assertEqual(tuple(out.shape), (2, 5))


class ProfileTest(unittest.TestCase):
    def test_profile_overrides_and_saves(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "config.json"
            base = load_config()
            base["profiles"] = {"southline": {"heuristic": {"morph_kernel": 3}}}
            path.write_text(json.dumps(base))

            cfg = load_config(path, "southline")
            self.assertEqual(cfg["heuristic"]["morph_kernel"], 3)
            self.assertEqual(cfg["heuristic"]["pickups"], base["heuristic"]["pickups"])

            cfg["heuristic"]["morph_kernel"] = 11
            save_config(cfg, path, "southline")
            saved = json.loads(path.read_text())
            self.assertEqual(saved["profiles"]["southline"]["heuristic"]["morph_kernel"], 11)
            self.assertEqual(saved["heuristic"]["morph_kernel"], base["heuristic"]["morph_kernel"])

    def test_unknown_profile(self):
        with self.assertRaises(KeyError):
            load_config(profile="nope")


class LibraryPreferenceTest(unittest.TestCase):
    """Маршрути з відео після оновлення туру важливіші за Zero Mode, але лише для того самого рівня."""

    def _lib(self, routes):
        from bot.library import RouteLibrary
        d = Path(tempfile.mkdtemp())
        (d / "cards").mkdir()
        for name, text, state, uploaded in routes:
            cv2.imwrite(str(d / "cards" / f"{name}.jpg"), np.zeros((10, 30, 3), np.uint8))
            data = {"level": name, "card": f"cards/{name}.jpg", "card_text": text, "card_state": state}
            if uploaded:
                data["uploaded"] = uploaded
            (d / f"{name}.json").write_text(json.dumps(data))
        return RouteLibrary(d)

    def _match(self, lib, text, state=(0, 0, 0)):
        from unittest import mock
        from bot import video
        with mock.patch.object(video, "card_text", return_value=text), \
                mock.patch.object(video, "card_state", return_value=state):
            return lib.match_band(np.zeros((10, 30, 3), np.uint8))

    def test_prefers_new_video_for_same_level(self):
        text = "collect120coinsscore9000stompon12buriedcaches"
        lib = self._lib([("zero_1", text, [0, 0, 0], ""), ("new_1", text, [1, 0, 0], "20261002")])
        self.assertEqual(self._match(lib, text).name, "new_1")

    def test_falls_back_to_zero_mode_when_numbers_differ(self):
        text = "collect120coinsscore9000stompon12buriedcaches"
        lib = self._lib([("zero_1", text, [0, 0, 0], ""),
                         ("new_1", "collect120coinsscore9500stompon13buriedcaches", [0, 0, 0], "20261002")])
        self.assertEqual(self._match(lib, text).name, "zero_1")


if __name__ == "__main__":
    unittest.main()
