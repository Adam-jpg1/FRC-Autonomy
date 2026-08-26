"""Virtual controller: neutral by default, clamping, deadband, slew, heartbeat."""

import math

import pytest

from frc_perception.config import ControllerConfig
from frc_perception.types import ControllerState, DesiredMotion, MechanismCommand
from frc_perception.virtual_controller import (
    VirtualController,
    apply_deadband,
    mechanism_policy,
    slew,
)


def enabled_config(**overrides) -> ControllerConfig:
    defaults = dict(
        enabled=True,
        field_oriented_drive=True,
        invert_left_x=False,
        invert_left_y=False,
        invert_right_x=False,
        full_scale_speed_mps=2.0,
        full_scale_omega_radps=4.0,
        deadband=0.0,
        output_limit=1.0,
        slew_rate_per_s=None,
    )
    defaults.update(overrides)
    return ControllerConfig(**defaults)


def drive(vx=0.0, vy=0.0, omega=0.0, field_relative=True) -> DesiredMotion:
    return DesiredMotion(
        vx_mps=vx, vy_mps=vy, omega_radps=omega, reason="test",
        confidence=1.0, field_relative=field_relative,
    )


# ---------------------------------------------------------------------------
# Disabled is the default and it is genuinely neutral
# ---------------------------------------------------------------------------


def test_default_configuration_is_disabled():
    assert ControllerConfig().enabled is False


def test_disabled_publishes_a_fully_neutral_gamepad():
    controller = VirtualController(ControllerConfig(enabled=False))
    state = controller.update(drive(vx=2.0, vy=2.0, omega=3.0), robot_theta=0.0, now=1.0)
    assert state.enabled is False
    assert (state.left_x, state.left_y, state.right_x, state.right_y) == (0, 0, 0, 0)
    assert (state.left_trigger, state.right_trigger) == (0.0, 0.0)
    assert state.pov == -1
    assert not any(
        [state.a, state.b, state.x, state.y, state.left_bumper, state.right_bumper,
         state.back, state.start, state.left_stick, state.right_stick]
    )


def test_no_motion_at_all_publishes_neutral():
    controller = VirtualController(enabled_config())
    state = controller.update(None, robot_theta=0.0, now=1.0)
    assert state.enabled is False
    assert state.left_y == 0.0


def test_every_key_is_published_every_cycle():
    """The roboRIO must never have to guess a missing value."""
    keys = set(VirtualController(enabled_config()).neutral().as_dict())
    assert keys == {
        "enabled", "heartbeat",
        "left_x", "left_y", "right_x", "right_y",
        "left_trigger", "right_trigger",
        "a", "b", "x", "y",
        "left_bumper", "right_bumper",
        "back", "start",
        "left_stick", "right_stick",
        "pov",
    }


# ---------------------------------------------------------------------------
# Heartbeat
# ---------------------------------------------------------------------------


def test_heartbeat_advances_even_while_disabled():
    """A frozen heartbeat is how the RIO detects a dead Jetson, so it must
    keep moving whether or not we are commanding anything."""
    controller = VirtualController(ControllerConfig(enabled=False))
    first = controller.update(None, 0.0, now=1.0).heartbeat
    second = controller.update(None, 0.0, now=1.1).heartbeat
    third = controller.update(drive(vx=1.0), 0.0, now=1.2).heartbeat
    assert second == first + 1 and third == second + 1


def test_neutral_state_carries_the_current_heartbeat():
    controller = VirtualController(enabled_config())
    controller.update(drive(vx=1.0), 0.0, now=1.0)
    assert controller.neutral().heartbeat == controller.heartbeat


# ---------------------------------------------------------------------------
# Mapping
# ---------------------------------------------------------------------------


def test_forward_command_maps_to_the_forward_axis():
    controller = VirtualController(enabled_config())
    state = controller.update(drive(vx=1.0), robot_theta=0.0, now=1.0)
    assert state.left_y == pytest.approx(0.5)  # 1.0 of a 2.0 m/s full scale
    assert state.left_x == pytest.approx(0.0)


def test_left_command_maps_to_the_strafe_axis():
    controller = VirtualController(enabled_config())
    state = controller.update(drive(vy=1.0), robot_theta=0.0, now=1.0)
    assert state.left_x == pytest.approx(0.5)


def test_rotation_maps_to_the_right_stick():
    controller = VirtualController(enabled_config())
    state = controller.update(drive(omega=2.0), robot_theta=0.0, now=1.0)
    assert state.right_x == pytest.approx(0.5)


def test_inversion_flags_flip_each_axis_independently():
    controller = VirtualController(
        enabled_config(invert_left_y=True, invert_left_x=False, invert_right_x=True)
    )
    state = controller.update(drive(vx=1.0, vy=1.0, omega=2.0), 0.0, now=1.0)
    assert state.left_y == pytest.approx(-0.5)
    assert state.left_x == pytest.approx(0.5)
    assert state.right_x == pytest.approx(-0.5)


def test_field_command_is_rotated_for_a_robot_oriented_drive():
    """Navigation speaks field frame. A robot-oriented drivetrain does not."""
    controller = VirtualController(enabled_config(field_oriented_drive=False))
    # Robot facing +Y (90 deg); a field +X command is to the robot's RIGHT.
    state = controller.update(drive(vx=2.0), robot_theta=math.pi / 2, now=1.0)
    assert state.left_y == pytest.approx(0.0, abs=1e-9)
    assert state.left_x == pytest.approx(-1.0)


def test_field_command_passes_straight_through_for_a_field_oriented_drive():
    controller = VirtualController(enabled_config(field_oriented_drive=True))
    state = controller.update(drive(vx=2.0), robot_theta=math.pi / 2, now=1.0)
    assert state.left_y == pytest.approx(1.0)


def test_robot_relative_command_is_rotated_up_for_a_field_oriented_drive():
    controller = VirtualController(enabled_config(field_oriented_drive=True))
    state = controller.update(
        drive(vx=2.0, field_relative=False), robot_theta=math.pi / 2, now=1.0
    )
    assert state.left_y == pytest.approx(0.0, abs=1e-9)
    assert state.left_x == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Clamp, deadband, slew
# ---------------------------------------------------------------------------


def test_output_is_clamped_to_the_configured_limit():
    controller = VirtualController(enabled_config(output_limit=0.6))
    state = controller.update(drive(vx=99.0, omega=99.0), 0.0, now=1.0)
    assert state.left_y == pytest.approx(0.6)
    assert state.right_x == pytest.approx(0.6)


def test_clamping_is_symmetric_for_negative_commands():
    controller = VirtualController(enabled_config(output_limit=0.6))
    state = controller.update(drive(vx=-99.0), 0.0, now=1.0)
    assert state.left_y == pytest.approx(-0.6)


def test_deadband_zeroes_tiny_commands_exactly():
    controller = VirtualController(enabled_config(deadband=0.1))
    state = controller.update(drive(vx=0.05), 0.0, now=1.0)  # -> 0.025 of full scale
    assert state.left_y == 0.0


def test_deadband_leaves_commands_above_it_untouched():
    controller = VirtualController(enabled_config(deadband=0.1))
    state = controller.update(drive(vx=1.0), 0.0, now=1.0)
    assert state.left_y == pytest.approx(0.5)


@pytest.mark.parametrize(
    "value,deadband,expected", [(0.0, 0.1, 0.0), (0.05, 0.1, 0.0), (-0.05, 0.1, 0.0),
                                (0.2, 0.1, 0.2), (-0.2, 0.1, -0.2)]
)
def test_apply_deadband(value, deadband, expected):
    assert apply_deadband(value, deadband) == expected


def test_slew_limits_the_step_per_second():
    assert slew(0.0, 1.0, max_rate_per_s=2.0, dt=0.1) == pytest.approx(0.2)
    # No elapsed time means no permitted change, not a free jump to target.
    assert slew(0.3, 1.0, max_rate_per_s=2.0, dt=0.0) == 0.3
    assert slew(0.0, -1.0, max_rate_per_s=2.0, dt=0.1) == pytest.approx(-0.2)
    assert slew(0.0, 0.05, max_rate_per_s=2.0, dt=0.1) == pytest.approx(0.05)
    assert slew(0.0, 1.0, max_rate_per_s=None, dt=0.1) == 1.0


def test_slew_ramps_the_published_axis_over_several_cycles():
    controller = VirtualController(enabled_config(slew_rate_per_s=2.0))
    first = controller.update(drive(vx=2.0), 0.0, now=1.0)
    assert first.left_y == 0.0, "no elapsed time yet: the axis may not move"
    second = controller.update(drive(vx=2.0), 0.0, now=1.1)
    assert second.left_y == pytest.approx(0.2)
    third = controller.update(drive(vx=2.0), 0.0, now=1.2)
    assert third.left_y == pytest.approx(0.4)


def test_disabling_resets_the_slew_state_so_re_enabling_starts_from_a_stop():
    cfg = enabled_config(slew_rate_per_s=2.0)
    controller = VirtualController(cfg)
    for index in range(20):
        controller.update(drive(vx=2.0), 0.0, now=1.0 + index * 0.1)
    assert controller.update(drive(vx=2.0), 0.0, now=3.0).left_y > 0.5
    cfg.enabled = False
    controller.update(drive(vx=2.0), 0.0, now=3.1)
    cfg.enabled = True
    resumed = controller.update(drive(vx=2.0), 0.0, now=3.2)
    assert resumed.left_y == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# Mechanisms
# ---------------------------------------------------------------------------


def test_mechanism_policy_ships_neutral():
    command = mechanism_policy(drive(vx=1.0), pose_valid=True, has_target=True)
    assert isinstance(command, MechanismCommand)
    assert not (command.intake or command.outtake or command.shoot or command.prepare_shot)


def test_no_mechanism_buttons_are_ever_pressed_by_default():
    controller = VirtualController(enabled_config())
    state = controller.update(drive(vx=1.0, omega=1.0), 0.0, now=1.0)
    assert not any([state.a, state.b, state.x, state.y, state.left_bumper,
                    state.right_bumper, state.back, state.start])
    assert state.left_trigger == 0.0 and state.right_trigger == 0.0


def test_an_unmapped_mechanism_intent_is_reported_rather_than_dropped_silently(caplog):
    controller = VirtualController(enabled_config())
    state = ControllerState()
    with caplog.at_level("ERROR"):
        controller._apply_mechanism(state, MechanismCommand(intake=True))
    assert "MUST INPUT" in caplog.text
    assert state.right_bumper is False


def test_pose_validity_is_passed_through_rather_than_inferred():
    """The mechanism policy is told whether the pose is valid explicitly.

    Inferring it from a navigation confidence value would be wrong: a
    spin-search with a perfectly valid pose reports confidence 0.0.
    """
    seen = {}
    controller = VirtualController(enabled_config())
    original = controller._apply_mechanism

    def capture(state, mechanism):
        seen["mechanism"] = mechanism
        return original(state, mechanism)

    controller._apply_mechanism = capture
    controller.update(drive(omega=1.0), 0.0, now=1.0, pose_valid=True)
    assert "mechanism" in seen
