"""Virtual gamepad: turns NavCommands into stick/button values over NT.

The robot's existing teleop bindings read table "VirtualController"
instead of a physical joystick. Safety properties:

  * `enabled` is False (and every axis 0, every button False) unless
    ControllerConfig.enabled is True — the whole feature defaults OFF.
  * `heartbeat` increments on every publish even while disabled, so the
    robot code can reject a dead Jetson (require heartbeat to advance).
  * Axes are deadbanded and slew-rate limited to avoid commanding jerks.

Stick mapping follows the WPILib convention (push forward = negative Y),
with per-axis inversion flags in config because every team's bindings
differ. VERIFY ON BLOCKS.
"""

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .config import ControllerConfig
from .localization import PoseEstimate
from .navigation import NavCommand
from .world_model import Track

BUTTON_NAMES = (
    "button_a", "button_b", "button_x", "button_y",
    "button_left_bumper", "button_right_bumper",
    "button_back", "button_start",
    "button_left_stick", "button_right_stick",
)
AXIS_NAMES = (
    "axis_left_x", "axis_left_y", "axis_right_x", "axis_right_y",
    "axis_left_trigger", "axis_right_trigger",
)


@dataclass
class GamepadState:
    """Everything published under VirtualController/ (plus heartbeat)."""

    enabled: bool = False
    axes: Dict[str, float] = field(default_factory=lambda: {n: 0.0 for n in AXIS_NAMES})
    buttons: Dict[str, bool] = field(default_factory=lambda: {n: False for n in BUTTON_NAMES})
    pov: int = -1  # d-pad angle in degrees, -1 = released


def mechanism_policy(
    nav_state: str,
    pose: PoseEstimate,
    balls: List[Track],
) -> Dict[str, bool]:
    """MUST INPUT: your robot's intake/shoot button policy goes here.

    Return button overrides, e.g. hold intake while acquiring:

        if nav_state == "ACQUIRE":
            return {"button_right_bumper": True}   # if RB = run intake

    Which button does what depends entirely on YOUR teleop bindings, so
    this ships empty (no mechanisms actuate) until you fill it in.
    """
    return {}


class VirtualController:
    def __init__(self, cfg: ControllerConfig):
        self.cfg = cfg
        self._last_axes: Dict[str, float] = {n: 0.0 for n in AXIS_NAMES}
        self._last_time: Optional[float] = None

    def build(
        self,
        nav: NavCommand,
        pose: PoseEstimate,
        balls: List[Track],
        now: Optional[float] = None,
    ) -> GamepadState:
        """Convert a field-frame NavCommand into gamepad state."""
        if now is None:
            now = time.monotonic()
        state = GamepadState(enabled=self.cfg.enabled)
        if not self.cfg.enabled:
            self._last_axes = {n: 0.0 for n in AXIS_NAMES}
            self._last_time = now
            return state  # all zeros, enabled=False

        vx, vy = nav.vx, nav.vy
        if not self.cfg.field_oriented_drive:
            # Robot-oriented sticks: rotate the field command into the
            # robot frame. With an invalid pose nav only ever commands
            # rotation, so theta is not load-bearing here.
            c, s = math.cos(pose.theta), math.sin(pose.theta)
            vx, vy = c * nav.vx + s * nav.vy, -s * nav.vx + c * nav.vy

        # WPILib convention: +X drive = push stick forward = NEGATIVE Y
        # axis; +Y (left) drive = stick left = NEGATIVE X axis; CCW
        # rotation = stick left = NEGATIVE X. The invert flags flip each
        # if your bindings differ.
        raw = {
            "axis_left_y": -vx if self.cfg.invert_left_y else vx,
            "axis_left_x": -vy if self.cfg.invert_left_x else vy,
            "axis_right_x": -nav.omega if self.cfg.invert_right_x else nav.omega,
            "axis_right_y": 0.0,
            "axis_left_trigger": 0.0,
            "axis_right_trigger": 0.0,
        }

        dt = (now - self._last_time) if self._last_time is not None else 0.0
        self._last_time = now
        max_step = self.cfg.slew_rate_per_s * max(dt, 0.0)
        for name in AXIS_NAMES:
            target = max(-1.0, min(1.0, raw[name]))
            prev = self._last_axes[name]
            if max_step > 0.0:
                target = prev + max(-max_step, min(max_step, target - prev))
            if abs(target) < self.cfg.axis_deadband:
                target = 0.0
            state.axes[name] = target
            self._last_axes[name] = target

        for name, pressed in mechanism_policy(nav.state, pose, balls).items():
            if name in state.buttons:
                state.buttons[name] = bool(pressed)
        return state
