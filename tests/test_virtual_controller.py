"""Tests for the virtual gamepad: safety default, mapping, slew limiting."""

import math

import pytest

from frc_perception.config import ControllerConfig
from frc_perception.localization import PoseEstimate
from frc_perception.navigation import NavCommand
from frc_perception.virtual_controller import (
    AXIS_NAMES,
    BUTTON_NAMES,
    VirtualController,
)

POSE = PoseEstimate(2.0, 2.0, 0.0, True, "fused")


def build(ctrl: VirtualController, nav: NavCommand, now: float):
    return ctrl.build(nav, POSE, [], now=now)


class TestSafetyDefault:
    def test_disabled_publishes_all_zeros(self):
        ctrl = VirtualController(ControllerConfig(enabled=False))
        state = build(ctrl, NavCommand(vx=1.0, vy=1.0, omega=1.0), now=0.0)
        assert state.enabled is False
        assert all(v == 0.0 for v in state.axes.values())
        assert all(v is False for v in state.buttons.values())

    def test_default_config_is_disabled(self):
        assert ControllerConfig().enabled is False


def enabled_config(**overrides) -> ControllerConfig:
    values = dict(enabled=True, slew_rate_per_s=1000.0, axis_deadband=0.0)
    values.update(overrides)
    return ControllerConfig(**values)


class TestMapping:
    def test_forward_command_maps_to_negative_left_y(self):
        ctrl = VirtualController(enabled_config())
        build(ctrl, NavCommand(), now=0.0)  # prime the slew limiter
        state = build(ctrl, NavCommand(vx=0.5), now=1.0)
        assert state.axes["axis_left_y"] == pytest.approx(-0.5)

    def test_inversion_flags_flip_sign(self):
        ctrl = VirtualController(enabled_config(invert_left_y=False))
        build(ctrl, NavCommand(), now=0.0)
        state = build(ctrl, NavCommand(vx=0.5), now=1.0)
        assert state.axes["axis_left_y"] == pytest.approx(0.5)

    def test_robot_oriented_rotates_by_heading(self):
        ctrl = VirtualController(enabled_config(field_oriented_drive=False))
        pose = PoseEstimate(0.0, 0.0, math.pi / 2, True, "fused")
        ctrl.build(NavCommand(), pose, [], now=0.0)
        # Field +Y command while facing +Y == drive straight forward.
        state = ctrl.build(NavCommand(vy=0.5), pose, [], now=1.0)
        assert state.axes["axis_left_y"] == pytest.approx(-0.5)
        assert state.axes["axis_left_x"] == pytest.approx(0.0, abs=1e-9)

    def test_all_contract_names_present(self):
        ctrl = VirtualController(enabled_config())
        state = build(ctrl, NavCommand(), now=0.0)
        assert set(state.axes) == set(AXIS_NAMES)
        assert set(state.buttons) == set(BUTTON_NAMES)
        assert state.pov == -1


class TestShaping:
    def test_slew_limits_step(self):
        ctrl = VirtualController(enabled_config(slew_rate_per_s=1.0))
        build(ctrl, NavCommand(), now=0.0)
        state = build(ctrl, NavCommand(vx=1.0), now=0.1)  # 0.1 s at 1.0/s
        assert abs(state.axes["axis_left_y"]) == pytest.approx(0.1)

    def test_deadband_zeroes_small_commands(self):
        ctrl = VirtualController(enabled_config(axis_deadband=0.05))
        build(ctrl, NavCommand(), now=0.0)
        state = build(ctrl, NavCommand(vx=0.01), now=1.0)
        assert state.axes["axis_left_y"] == 0.0

    def test_commands_clamped_to_unit_range(self):
        ctrl = VirtualController(enabled_config())
        build(ctrl, NavCommand(), now=0.0)
        state = build(ctrl, NavCommand(vx=5.0), now=1.0)
        assert abs(state.axes["axis_left_y"]) <= 1.0
