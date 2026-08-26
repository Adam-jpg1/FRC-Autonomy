"""Virtual gamepad adapter: desired motion -> controller axes.

The Jetson can publish a synthetic gamepad to NetworkTables so the
roboRIO can drive from it exactly as it drives from a real one.  That is
a powerful and dangerous capability, so this module is built around three
rules:

1. **Disabled by default.** ``ControllerConfig.enabled`` starts False and
   the adapter then publishes a fully neutral gamepad with
   ``enabled = false``: every axis 0.0, every trigger 0.0, every button
   False, POV -1.
2. **Everything is published every cycle.** The roboRIO never has to
   remember or infer a value that was left out.
3. **The Jetson is not the safety gate.** A heartbeat counter is
   published unconditionally so the RIO can detect a dead or wedged
   Jetson.  The RIO must require, at minimum: ``enabled == true`` AND an
   advancing heartbeat AND its own deadman held AND a robot mode that
   permits automation.  See the README.

STICK POLARITY
--------------
Command space here is unambiguous: ``forward`` (+X robot), ``left`` (+Y
robot), ``ccw`` (positive theta).  The three ``invert_*`` flags map that
onto whatever polarity your robot's teleop bindings expect.  There is no
universal convention, so they are MUST INPUT values to be confirmed with
the robot on blocks.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

from .config import ControllerConfig
from .geometry import clamp, rotate_vector
from .types import ControllerState, DesiredMotion, MechanismCommand

logger = logging.getLogger(__name__)


def apply_deadband(value: float, deadband: float) -> float:
    """Zero out tiny commands so the robot does not creep."""
    return 0.0 if abs(value) < deadband else value


def slew(previous: float, target: float, max_rate_per_s: Optional[float], dt: float) -> float:
    """Limit how fast an axis may change, in units per second.

    With ``dt <= 0`` (the very first cycle, or a repeated timestamp) no
    time has passed, so no change is permitted and the previous value is
    held.  Returning the target instead would let the first command after
    enabling jump straight to full deflection -- exactly what the rate
    limit exists to prevent.
    """
    if max_rate_per_s is None:
        return target
    if dt <= 0.0:
        return previous
    max_step = max_rate_per_s * dt
    delta = target - previous
    if delta > max_step:
        return previous + max_step
    if delta < -max_step:
        return previous - max_step
    return target


def mechanism_policy(
    motion: DesiredMotion,
    pose_valid: bool,
    has_target: bool,
) -> MechanismCommand:
    """Decide mechanism intents (intake, shoot, ...) for this cycle.

    SHIPPED NEUTRAL ON PURPOSE.  Actuating a mechanism from the Jetson is
    a robot-specific decision that depends on hardware this code has never
    seen, and a wrong guess here fires a shooter in the pit.

    # MUST INPUT: your robot's mechanism logic and the button bindings it
    # maps onto. Implement the intents below (or add your own) and then
    # wire them to buttons in `controller_state_from_motion`. Until you do,
    # every mechanism button stays False and the Jetson only ever suggests
    # driving.

    The arguments are the information a real policy would need; they are
    accepted now so that adding the logic later does not change any call
    site.
    """
    del motion, pose_valid, has_target  # unused until the policy is written
    return MechanismCommand(note="neutral: no mechanism policy configured")


class VirtualController:
    """Stateful adapter: holds slew state and the heartbeat counter."""

    def __init__(self, cfg: ControllerConfig) -> None:
        self.cfg = cfg
        self._heartbeat = 0
        self._left_x = 0.0
        self._left_y = 0.0
        self._right_x = 0.0
        self._last_update: Optional[float] = None
        self._logged_enabled: Optional[bool] = None
        self._warned_unmapped_mechanism = False

    @property
    def heartbeat(self) -> int:
        return self._heartbeat

    def neutral(self) -> ControllerState:
        """A fully neutral state carrying the current heartbeat."""
        return ControllerState(enabled=False, heartbeat=self._heartbeat)

    def command_axes(
        self, motion: DesiredMotion, robot_theta: float
    ) -> Tuple[float, float, float]:
        """Motion -> (forward, left, ccw) fractions of full scale.

        Handles the field-relative/robot-relative mismatch: navigation
        works in field coordinates, but if the robot's teleop sticks are
        robot-oriented the command must be rotated into the robot frame
        first (and vice versa).
        """
        vx, vy = motion.vx_mps, motion.vy_mps
        if motion.field_relative and not self.cfg.field_oriented_drive:
            vx, vy = rotate_vector(vx, vy, -robot_theta)
        elif not motion.field_relative and self.cfg.field_oriented_drive:
            vx, vy = rotate_vector(vx, vy, robot_theta)

        forward = vx / self.cfg.full_scale_speed_mps
        left = vy / self.cfg.full_scale_speed_mps
        ccw = motion.omega_radps / self.cfg.full_scale_omega_radps
        return (forward, left, ccw)

    def update(
        self,
        motion: Optional[DesiredMotion],
        robot_theta: float,
        now: float,
        pose_valid: bool = False,
    ) -> ControllerState:
        """Produce the gamepad state for this cycle.

        The heartbeat advances on EVERY call, enabled or not, so the RIO
        can distinguish "Jetson says do nothing" from "Jetson is gone".
        """
        self._heartbeat += 1
        dt = 0.0 if self._last_update is None else max(0.0, now - self._last_update)
        self._last_update = now

        if self._logged_enabled != self.cfg.enabled:
            if self.cfg.enabled:
                logger.warning(
                    "VIRTUAL CONTROLLER ENABLED -- the Jetson is publishing live stick "
                    "commands; the roboRIO deadman is the only remaining gate"
                )
            else:
                logger.info("virtual controller disabled (publishing neutral)")
            self._logged_enabled = self.cfg.enabled

        if not self.cfg.enabled or motion is None:
            # Reset slew memory so re-enabling always starts from a stop.
            self._left_x = self._left_y = self._right_x = 0.0
            return self.neutral()

        forward, left, ccw = self.command_axes(motion, robot_theta)

        # Command space -> the robot's stick polarity.
        left_y = -forward if self.cfg.invert_left_y else forward
        left_x = -left if self.cfg.invert_left_x else left
        right_x = -ccw if self.cfg.invert_right_x else ccw

        limit = self.cfg.output_limit
        left_y = apply_deadband(clamp(left_y, -limit, limit), self.cfg.deadband)
        left_x = apply_deadband(clamp(left_x, -limit, limit), self.cfg.deadband)
        right_x = apply_deadband(clamp(right_x, -limit, limit), self.cfg.deadband)

        self._left_y = slew(self._left_y, left_y, self.cfg.slew_rate_per_s, dt)
        self._left_x = slew(self._left_x, left_x, self.cfg.slew_rate_per_s, dt)
        self._right_x = slew(self._right_x, right_x, self.cfg.slew_rate_per_s, dt)

        state = ControllerState(
            enabled=True,
            heartbeat=self._heartbeat,
            left_x=self._left_x,
            left_y=self._left_y,
            right_x=self._right_x,
            right_y=0.0,
            left_trigger=0.0,
            right_trigger=0.0,
            pov=-1,
        )
        mechanism = mechanism_policy(
            motion, pose_valid=pose_valid, has_target=motion.target is not None
        )
        self._apply_mechanism(state, mechanism)
        return state

    def _apply_mechanism(self, state: ControllerState, mechanism: MechanismCommand) -> None:
        """Map mechanism intents onto gamepad buttons.

        # MUST INPUT: your robot's button bindings. Until they are filled in
        # here, every button stays False and an enabled virtual controller
        # can only ever drive -- which is the safe default.
        #
        # Example of what this becomes once the bindings are known:
        #     state.right_bumper = mechanism.intake
        #     state.a = mechanism.shoot

        If someone implements ``mechanism_policy`` but forgets to wire the
        bindings here, the intent would be silently dropped -- so say so
        once instead.
        """
        requested = (
            mechanism.intake or mechanism.outtake or mechanism.shoot or mechanism.prepare_shot
        )
        if requested and not self._warned_unmapped_mechanism:
            logger.error(
                "mechanism_policy requested %s but no button bindings are configured in "
                "VirtualController._apply_mechanism (MUST INPUT); the intent is being dropped",
                mechanism,
            )
            self._warned_unmapped_mechanism = True
        # `state` is intentionally left with every button False.


__all__ = [
    "VirtualController",
    "apply_deadband",
    "mechanism_policy",
    "slew",
]
