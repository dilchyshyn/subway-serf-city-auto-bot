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
        b = Brain(self.cfg, lane=0)
        self.assertEqual(b.decide(sc(st(LOW), st(TALL), st()), now=0), JUMP)
        b = Brain(self.cfg, lane=0)
        self.assertEqual(b.decide(sc(st(HIGH), st(TALL), st()), now=0), ROLL)

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
            level, loaded = route.load(path)
        self.assertEqual(level, "docks_01")
        self.assertEqual([g.t for g in loaded], [0.0, 1.5, 2.0])

        player = route.RoutePlayer(loaded, start_time=10.0, offset=0.3)
        self.assertEqual([g.action for g in player.due(10.0)], ["tap"])   # тап «Грати» — одразу
        self.assertEqual(player.due(11.7), [])                             # 1.5 + 0.3 ще не настало
        self.assertEqual([g.action for g in player.due(11.8)], ["left"])
        self.assertAlmostEqual(player.next_in(11.8), 0.5)
        self.assertEqual([g.action for g in player.due(20)], ["jump"])
        self.assertTrue(player.finished)

    def test_classify(self):
        self.assertEqual(route.classify(0.5, 0.5, 0.51, 0.5), "tap")
        self.assertEqual(route.classify(0.5, 0.5, 0.8, 0.55), "right")
        self.assertEqual(route.classify(0.5, 0.5, 0.5, 0.8), "roll")


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


if __name__ == "__main__":
    unittest.main()
