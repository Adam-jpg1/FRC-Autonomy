"""Tests for the navigation policy, especially invalid-pose degradation."""

import math

import pytest

from frc_perception.config import NavConfig
from frc_perception.field_map import FieldMap
from frc_perception.localization import PoseEstimate
from frc_perception.navigation import (
    STATE_ACQUIRE,
    STATE_DRIVE_TO_CENTER,
    STATE_SPIN_SEARCH,
    STATE_SPIN_SEARCH_BLIND,
    Navigator,
)
from frc_perception.world_model import Track


def make_navigator(**overrides) -> Navigator:
    field = FieldMap("/nonexistent/layout.json", 16.0, 8.0)
    return Navigator(NavConfig(**overrides), field)


def valid_pose(x=2.0, y=2.0, theta=0.0) -> PoseEstimate:
    return PoseEstimate(x, y, theta, True, "fused")


INVALID_POSE = PoseEstimate(0.0, 0.0, 0.0, False, "none")


def ball(x: float, y: float) -> Track:
    return Track("fuel", "game_piece", x, y)


def robot(x: float, y: float) -> Track:
    return Track("bumper_blue", "robot", x, y)


class TestInvalidPoseDegradation:
    """CRITICAL: a bogus pose must never produce translation commands."""

    def test_no_balls_invalid_pose_spins_in_place(self):
        cmd = make_navigator().update(INVALID_POSE, [], [])
        assert cmd.state == STATE_SPIN_SEARCH_BLIND
        assert cmd.vx == 0.0 and cmd.vy == 0.0
        assert cmd.omega > 0.0

    def test_even_with_balls_invalid_pose_spins_in_place(self):
        # Field-space ball tracks may persist briefly after the pose dies;
        # they must not be chased blind.
        cmd = make_navigator().update(INVALID_POSE, [ball(5.0, 3.0)], [])
        assert cmd.state == STATE_SPIN_SEARCH_BLIND
        assert cmd.vx == 0.0 and cmd.vy == 0.0


class TestSearch:
    def test_no_balls_drives_to_center(self):
        nav = make_navigator()
        cmd = nav.update(valid_pose(2.0, 4.0), [], [])  # center is (8, 4)
        assert cmd.state == STATE_DRIVE_TO_CENTER
        assert cmd.vx > 0.0  # toward +X
        assert cmd.vy == pytest.approx(0.0, abs=1e-9)

    def test_at_center_spins(self):
        cmd = make_navigator().update(valid_pose(8.0, 4.0), [], [])
        assert cmd.state == STATE_SPIN_SEARCH
        assert cmd.vx == 0.0 and cmd.vy == 0.0
        assert cmd.omega > 0.0


class TestAcquire:
    def test_densest_pocket_beats_nearer_loner(self):
        nav = make_navigator()
        cluster = [ball(10.0, 4.0), ball(10.3, 4.2), ball(10.1, 3.8)]
        loner = [ball(3.0, 2.0)]
        cmd = nav.update(valid_pose(2.0, 2.0), cluster + loner, [])
        assert cmd.state == STATE_ACQUIRE
        # Target is the cluster's nearest member, not the lone closer ball.
        assert cmd.target[0] > 9.0

    def test_blocked_pocket_yields_to_open_one(self):
        nav = make_navigator()
        cluster = [ball(10.0, 4.0), ball(10.3, 4.2)]
        open_ball = [ball(4.0, 7.0)]
        blocker = [robot(6.0, 3.0)]  # sits on the straight line to the cluster
        cmd = nav.update(valid_pose(2.0, 2.0), cluster + open_ball, blocker)
        assert cmd.target == pytest.approx((4.0, 7.0))

    def test_command_faces_travel_direction(self):
        nav = make_navigator()
        # Ball straight left of a robot facing +X: expect CCW rotation.
        cmd = nav.update(valid_pose(8.0, 2.0, 0.0), [ball(8.0, 6.0)], [])
        assert cmd.omega > 0.0
        assert cmd.vy > 0.0

    def test_speed_caps_respected(self):
        nav = make_navigator(max_speed=0.6, max_omega=0.5)
        cmd = nav.update(valid_pose(1.0, 1.0, math.pi), [ball(14.0, 7.0)], [])
        assert math.hypot(cmd.vx, cmd.vy) <= 0.6 + 1e-9
        assert abs(cmd.omega) <= 0.5 + 1e-9

    def test_target_clamped_off_walls(self):
        nav = make_navigator(wall_margin_m=0.6)
        cmd = nav.update(valid_pose(2.0, 2.0), [ball(0.05, 0.05)], [])
        assert cmd.target[0] >= 0.6 and cmd.target[1] >= 0.6
