"""All NetworkTables traffic in one place (ntcore, NT4 client).

PUBLISH  Perception/          balls_field_x, balls_field_y,
                              opponents_field_x, opponents_field_y,
                              robot_x, robot_y, robot_theta, pose_valid,
                              frame_counter, ball_count, opponent_count
         VirtualController/   enabled, heartbeat, every button and axis, pov
SUBSCRIBE FMSInfo/IsRedAlliance
          Odometry/robot_x, robot_y, robot_theta, timestamp

Field values are meters/radians, WPILib blue origin (+X toward red,
+Y left). Everything degrades to defaults when the server is away —
ntcore reconnects on its own.
"""

import logging
from typing import List, Optional, Tuple

from .config import AppConfig
from .virtual_controller import AXIS_NAMES, BUTTON_NAMES, GamepadState

log = logging.getLogger(__name__)


class NTInterface:
    def __init__(self, cfg: AppConfig):
        import ntcore  # local import: robotpy native lib

        self._inst = ntcore.NetworkTableInstance.getDefault()
        self._inst.startClient4(cfg.nt_identity)
        if cfg.nt_server_override:
            self._inst.setServer(cfg.nt_server_override)
            log.info("NT client -> server %s", cfg.nt_server_override)
        elif cfg.team_number > 0:
            self._inst.setServerTeam(cfg.team_number)
            log.info("NT client -> team %d", cfg.team_number)
        else:
            self._inst.setServer("127.0.0.1")
            log.error(
                "team_number is 0 and no nt_server_override — connecting to "
                "localhost only. Set AppConfig.team_number."
            )

        perception = self._inst.getTable("Perception")
        self._pub_balls_x = perception.getDoubleArrayTopic("balls_field_x").publish()
        self._pub_balls_y = perception.getDoubleArrayTopic("balls_field_y").publish()
        self._pub_opp_x = perception.getDoubleArrayTopic("opponents_field_x").publish()
        self._pub_opp_y = perception.getDoubleArrayTopic("opponents_field_y").publish()
        self._pub_robot_x = perception.getDoubleTopic("robot_x").publish()
        self._pub_robot_y = perception.getDoubleTopic("robot_y").publish()
        self._pub_robot_theta = perception.getDoubleTopic("robot_theta").publish()
        self._pub_pose_valid = perception.getBooleanTopic("pose_valid").publish()
        self._pub_frame_counter = perception.getIntegerTopic("frame_counter").publish()
        self._pub_ball_count = perception.getIntegerTopic("ball_count").publish()
        self._pub_opp_count = perception.getIntegerTopic("opponent_count").publish()

        controller = self._inst.getTable("VirtualController")
        self._pub_ctrl_enabled = controller.getBooleanTopic("enabled").publish()
        self._pub_ctrl_heartbeat = controller.getIntegerTopic("heartbeat").publish()
        self._pub_ctrl_pov = controller.getIntegerTopic("pov").publish()
        self._pub_axes = {
            name: controller.getDoubleTopic(name).publish() for name in AXIS_NAMES
        }
        self._pub_buttons = {
            name: controller.getBooleanTopic(name).publish() for name in BUTTON_NAMES
        }
        self._ctrl_heartbeat = 0

        self._sub_is_red = (
            self._inst.getTable("FMSInfo").getBooleanTopic("IsRedAlliance").subscribe(False)
        )
        odometry = self._inst.getTable("Odometry")
        self._sub_odom_x = odometry.getDoubleTopic("robot_x").subscribe(0.0)
        self._sub_odom_y = odometry.getDoubleTopic("robot_y").subscribe(0.0)
        self._sub_odom_theta = odometry.getDoubleTopic("robot_theta").subscribe(0.0)
        self._sub_odom_ts = odometry.getDoubleTopic("timestamp").subscribe(0.0)
        self._warned_no_alliance = False

    # -- subscriptions -----------------------------------------------------------

    @property
    def connected(self) -> bool:
        return self._inst.isConnected()

    def is_red_alliance(self) -> Optional[bool]:
        """Alliance color, or None if FMSInfo has never been received."""
        value = self._sub_is_red.getAtomic()
        if value.time == 0:  # no publisher yet
            if not self._warned_no_alliance:
                self._warned_no_alliance = True
                log.warning(
                    "FMSInfo/IsRedAlliance not published yet — opponent "
                    "reporting disabled until the alliance is known"
                )
            return None
        return bool(value.value)

    def get_odometry(self) -> Optional[Tuple[float, float, float, float]]:
        """(x, y, theta, rio_timestamp) or None if never published."""
        ts = self._sub_odom_ts.getAtomic()
        if ts.time == 0:
            return None
        return (
            self._sub_odom_x.get(),
            self._sub_odom_y.get(),
            self._sub_odom_theta.get(),
            float(ts.value),
        )

    # -- publications --------------------------------------------------------------

    def publish_perception(
        self,
        balls_xy: Tuple[List[float], List[float]],
        opponents_xy: Tuple[List[float], List[float]],
        robot_x: float,
        robot_y: float,
        robot_theta: float,
        pose_valid: bool,
        frame_counter: int,
    ) -> None:
        self._pub_balls_x.set(balls_xy[0])
        self._pub_balls_y.set(balls_xy[1])
        self._pub_opp_x.set(opponents_xy[0])
        self._pub_opp_y.set(opponents_xy[1])
        self._pub_robot_x.set(robot_x)
        self._pub_robot_y.set(robot_y)
        self._pub_robot_theta.set(robot_theta)
        self._pub_pose_valid.set(pose_valid)
        self._pub_frame_counter.set(frame_counter)
        self._pub_ball_count.set(len(balls_xy[0]))
        self._pub_opp_count.set(len(opponents_xy[0]))

    def publish_controller(self, state: GamepadState) -> None:
        """Publish the full gamepad. Heartbeat advances even when disabled."""
        self._ctrl_heartbeat += 1
        self._pub_ctrl_enabled.set(state.enabled)
        self._pub_ctrl_heartbeat.set(self._ctrl_heartbeat)
        self._pub_ctrl_pov.set(state.pov)
        for name, pub in self._pub_axes.items():
            pub.set(state.axes[name])
        for name, pub in self._pub_buttons.items():
            pub.set(state.buttons[name])

    def close(self) -> None:
        self._inst.stopClient()
