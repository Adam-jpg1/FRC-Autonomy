"""NetworkTables 4 client: subscriptions, publications, and a simulator.

Everything that talks to the roboRIO goes through this module, so the
rest of the system can be tested without a robot on the bench.

SUBSCRIBED (published by the roboRIO)::

    FMSInfo/IsRedAlliance     boolean
    Odometry/robot_x          double, meters, blue-origin field frame
    Odometry/robot_y          double, meters, blue-origin field frame
    Odometry/robot_theta      double, radians, CCW from field +X
    Odometry/timestamp        double, the RIO's own clock

PUBLISHED (consumed by the roboRIO / dashboard) -- see
:meth:`NetworkTablesIO.publish_perception` for the full contract.

FRESHNESS
---------
This module reports what the table currently holds; it does NOT decide
whether that is fresh.  Odometry freshness is decided in
``localization.py`` by watching ``Odometry/timestamp`` advance, because
NT will happily serve the last value forever after the publisher dies.
Alliance data is the one exception handled here: the FMS value genuinely
does stop changing, so it is aged from the time it was last SEEN TO
CHANGE plus a generous timeout.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .config import NetworkTablesConfig
from .types import ControllerState

logger = logging.getLogger(__name__)


class NTBackend:
    """Interface implemented by the real client and by the simulator."""

    @property
    def connected(self) -> bool:
        raise NotImplementedError

    def connect(self) -> None:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError

    def read_is_red_alliance(self) -> Optional[bool]:
        """Raw ``FMSInfo/IsRedAlliance``; ``None`` when unknown or stale."""
        raise NotImplementedError

    def read_odometry(self) -> Optional[Tuple[float, float, float, float]]:
        """``(x, y, theta, rio_timestamp)`` or ``None`` if not published."""
        raise NotImplementedError

    def publish_perception(self, payload: Dict[str, Any]) -> None:
        raise NotImplementedError

    def publish_controller(self, state: ControllerState) -> None:
        raise NotImplementedError


def build_perception_payload(
    *,
    balls_field_x: Sequence[float],
    balls_field_y: Sequence[float],
    balls_field_vx: Sequence[float],
    balls_field_vy: Sequence[float],
    opponents_field_x: Sequence[float],
    opponents_field_y: Sequence[float],
    opponents_field_vx: Sequence[float],
    opponents_field_vy: Sequence[float],
    robot_x: float,
    robot_y: float,
    robot_theta: float,
    pose_valid: bool,
    pose_invalid_reason: str,
    frame_counter: int,
    camera_health: Sequence[Tuple[bool, float]],
    model_loaded: bool,
    odometry_fresh: bool,
    tag_pose_available: bool,
    tag_count: int,
    localization_source: str,
    localization_quality: float,
    translation_disagreement_m: float,
    heading_disagreement_deg: float,
    loop_fps: float,
    perception_latency_ms: float,
    inference_ms: float,
    apriltag_ms: float,
    nav_reason: str,
    nav_vx: float,
    nav_vy: float,
    nav_omega: float,
    nav_target_x: float,
    nav_target_y: float,
    nav_has_target: bool,
    virtual_controller_enabled: bool,
) -> Dict[str, Any]:
    """Assemble the ``Perception`` table payload.

    Kept as a pure function so the exact contract is visible in one place
    and testable without a NetworkTables server.  The required keys from
    the system contract are listed first; the rest are diagnostics.
    ``camera_health`` is indexed by position in ``config.cameras``, which
    is what ``camera_0_*`` / ``camera_1_*`` refer to.
    """
    payload: Dict[str, Any] = {
        # -- required contract keys ----------------------------------------
        "balls_field_x": list(balls_field_x),
        "balls_field_y": list(balls_field_y),
        "opponents_field_x": list(opponents_field_x),
        "opponents_field_y": list(opponents_field_y),
        "robot_x": float(robot_x),
        "robot_y": float(robot_y),
        "robot_theta": float(robot_theta),
        "pose_valid": bool(pose_valid),
        "frame_counter": int(frame_counter),
        "ball_count": len(balls_field_x),
        "opponent_count": len(opponents_field_x),
        # -- diagnostics ---------------------------------------------------
        "balls_field_vx": list(balls_field_vx),
        "balls_field_vy": list(balls_field_vy),
        "opponents_field_vx": list(opponents_field_vx),
        "opponents_field_vy": list(opponents_field_vy),
        "pose_invalid_reason": pose_invalid_reason,
        "model_loaded": bool(model_loaded),
        "odometry_fresh": bool(odometry_fresh),
        "tag_pose_available": bool(tag_pose_available),
        "tag_count": int(tag_count),
        "localization_source": localization_source,
        "localization_quality": float(localization_quality),
        "translation_disagreement_m": float(translation_disagreement_m),
        "heading_disagreement_deg": float(heading_disagreement_deg),
        "loop_fps": float(loop_fps),
        "perception_latency_ms": float(perception_latency_ms),
        "inference_ms": float(inference_ms),
        "apriltag_ms": float(apriltag_ms),
        "nav_reason": nav_reason,
        "nav_vx_mps": float(nav_vx),
        "nav_vy_mps": float(nav_vy),
        "nav_omega_radps": float(nav_omega),
        "nav_target_x": float(nav_target_x),
        "nav_target_y": float(nav_target_y),
        "nav_has_target": bool(nav_has_target),
        "virtual_controller_enabled": bool(virtual_controller_enabled),
    }
    for index, (healthy, fps) in enumerate(camera_health):
        payload[f"camera_{index}_healthy"] = bool(healthy)
        payload[f"camera_{index}_fps"] = float(fps)
    return payload


#: The keys the roboRIO contract guarantees.  ``run.py --check`` and the
#: unit tests assert that every one of these is present in every payload.
REQUIRED_PERCEPTION_KEYS: Tuple[str, ...] = (
    "balls_field_x",
    "balls_field_y",
    "opponents_field_x",
    "opponents_field_y",
    "robot_x",
    "robot_y",
    "robot_theta",
    "pose_valid",
    "frame_counter",
    "ball_count",
    "opponent_count",
)


class NetworkTablesIO(NTBackend):
    """Real NT4 client, built on ``ntcore`` (robotpy)."""

    def __init__(self, cfg: NetworkTablesConfig) -> None:
        self.cfg = cfg
        self._instance: Any = None
        self._perception_table: Any = None
        self._controller_table: Any = None
        self._publishers: Dict[str, Any] = {}
        self._subscribers: Dict[str, Any] = {}
        self.load_error: str = ""
        self._last_connected: Optional[bool] = None
        self._alliance_value: Optional[bool] = None
        self._alliance_changed_at: float = 0.0

    # -- lifecycle ---------------------------------------------------------

    @property
    def connected(self) -> bool:
        if self._instance is None:
            return False
        try:
            return bool(self._instance.isConnected())
        except Exception:  # noqa: BLE001 - never let a status check throw
            return False

    def connect(self) -> None:
        """Start the NT4 client.  Never raises; sets ``load_error`` instead.

        ``ntcore`` reconnects on its own, so there is no retry loop here;
        the application simply keeps publishing and the values land once
        the link comes back.
        """
        try:
            import ntcore
        except ImportError as exc:
            self.load_error = f"ntcore not installed: {exc}"
            logger.error("networktables: %s", self.load_error)
            return

        try:
            instance = ntcore.NetworkTableInstance.getDefault()
            instance.startClient4(self.cfg.client_identity)
            if self.cfg.server_override:
                instance.setServer(self.cfg.server_override)
                logger.info("networktables: connecting to %s", self.cfg.server_override)
            else:
                instance.setServerTeam(self.cfg.team_number)
                logger.info(
                    "networktables: connecting to team %d roboRIO", self.cfg.team_number
                )
            self._instance = instance
            self._perception_table = instance.getTable(self.cfg.perception_table)
            self._controller_table = instance.getTable("VirtualController")

            fms = instance.getTable(self.cfg.fms_table)
            self._subscribers["is_red"] = fms.getBooleanTopic(
                self.cfg.key_is_red_alliance
            ).subscribe(False)
            odometry = instance.getTable("Odometry")
            for key in ("robot_x", "robot_y", "robot_theta", "timestamp"):
                self._subscribers[key] = odometry.getDoubleTopic(key).subscribe(math.nan)
        except Exception as exc:  # noqa: BLE001 - NT startup must not kill the app
            self.load_error = f"NetworkTables startup failed: {exc}"
            logger.exception("networktables: %s", self.load_error)
            self._instance = None

    def close(self) -> None:
        if self._instance is None:
            return
        try:
            for publisher in self._publishers.values():
                publisher.close()
            for subscriber in self._subscribers.values():
                subscriber.close()
            self._instance.stopClient()
        except Exception as exc:  # noqa: BLE001 - shutdown is best effort
            logger.warning("networktables: unclean shutdown: %s", exc)
        finally:
            self._publishers.clear()
            self._subscribers.clear()
            self._instance = None

    def poll_connection(self) -> bool:
        """Log connect/disconnect transitions.  Returns the current state."""
        state = self.connected
        if state != self._last_connected:
            if state:
                logger.info("networktables: CONNECTED")
            elif self._last_connected is not None:
                logger.warning("networktables: DISCONNECTED (will auto-reconnect)")
            self._last_connected = state
        return state

    # -- reads -------------------------------------------------------------

    def read_is_red_alliance(self) -> Optional[bool]:
        subscriber = self._subscribers.get("is_red")
        if subscriber is None:
            return None
        try:
            if not subscriber.getTopic().exists():
                return None
            value = bool(subscriber.get())
        except Exception as exc:  # noqa: BLE001 - a read must never kill the loop
            logger.debug("networktables: alliance read failed: %s", exc)
            return None

        now = time.monotonic()
        if self._alliance_value is None or value != self._alliance_value:
            self._alliance_value = value
            self._alliance_changed_at = now
        if (now - self._alliance_changed_at) > self.cfg.alliance_stale_timeout_s:
            # The value is old but the FMS genuinely does not republish it,
            # so only treat it as unknown if the link itself is down.
            if not self.connected:
                return None
        return value

    def read_odometry(self) -> Optional[Tuple[float, float, float, float]]:
        try:
            values = [
                float(self._subscribers[key].get())
                for key in ("robot_x", "robot_y", "robot_theta", "timestamp")
            ]
        except KeyError:
            return None
        except Exception as exc:  # noqa: BLE001
            logger.debug("networktables: odometry read failed: %s", exc)
            return None
        if any(not math.isfinite(value) for value in values):
            return None
        return (values[0], values[1], values[2], values[3])

    # -- writes ------------------------------------------------------------

    def _publisher(self, table: Any, key: str, value: Any) -> Any:
        cache_key = f"{id(table)}::{key}"
        publisher = self._publishers.get(cache_key)
        if publisher is not None:
            return publisher
        if isinstance(value, bool):
            publisher = table.getBooleanTopic(key).publish()
        elif isinstance(value, int):
            publisher = table.getIntegerTopic(key).publish()
        elif isinstance(value, float):
            publisher = table.getDoubleTopic(key).publish()
        elif isinstance(value, str):
            publisher = table.getStringTopic(key).publish()
        elif isinstance(value, (list, tuple)):
            publisher = table.getDoubleArrayTopic(key).publish()
        else:
            raise TypeError(f"cannot publish {key!r} of type {type(value).__name__}")
        self._publishers[cache_key] = publisher
        return publisher

    def _publish_dict(self, table: Any, payload: Dict[str, Any]) -> None:
        if table is None:
            return
        for key, value in payload.items():
            try:
                publisher = self._publisher(table, key, value)
                if isinstance(value, (list, tuple)):
                    publisher.set([float(v) for v in value])
                else:
                    publisher.set(value)
            except Exception as exc:  # noqa: BLE001 - one bad key must not stop the rest
                logger.debug("networktables: publish of %r failed: %s", key, exc)

    def publish_perception(self, payload: Dict[str, Any]) -> None:
        self._publish_dict(self._perception_table, payload)

    def publish_controller(self, state: ControllerState) -> None:
        self._publish_dict(self._controller_table, state.as_dict())

    def flush(self) -> None:
        if self._instance is not None:
            try:
                self._instance.flush()
            except Exception:  # noqa: BLE001
                pass


class SimulatedNT(NTBackend):
    """In-memory NetworkTables for hardware-free development.

    Odometry advances on its own so the freshness logic downstream is
    genuinely exercised; set ``freeze_odometry`` to reproduce a dead RIO
    publisher and watch the pose go invalid.
    """

    def __init__(
        self,
        is_red: Optional[bool] = None,
        start_pose: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        self.is_red = is_red
        self.freeze_odometry = False
        self._pose = list(start_pose)
        self._velocity = velocity
        self._rio_time = 0.0
        self._last_wall = time.monotonic()
        self.published_perception: Dict[str, Any] = {}
        self.published_controller: Dict[str, Any] = {}
        self.publish_count = 0

    @property
    def connected(self) -> bool:
        return True

    def connect(self) -> None:
        logger.info("networktables: SIMULATED backend (no roboRIO)")

    def close(self) -> None:
        return

    def poll_connection(self) -> bool:
        return True

    def read_is_red_alliance(self) -> Optional[bool]:
        return self.is_red

    def read_odometry(self) -> Optional[Tuple[float, float, float, float]]:
        now = time.monotonic()
        dt = now - self._last_wall
        self._last_wall = now
        if not self.freeze_odometry:
            self._rio_time += dt
            self._pose[0] += self._velocity[0] * dt
            self._pose[1] += self._velocity[1] * dt
            self._pose[2] += self._velocity[2] * dt
        return (self._pose[0], self._pose[1], self._pose[2], self._rio_time)

    def publish_perception(self, payload: Dict[str, Any]) -> None:
        self.published_perception = dict(payload)
        self.publish_count += 1

    def publish_controller(self, state: ControllerState) -> None:
        self.published_controller = state.as_dict()

    def flush(self) -> None:
        return


def missing_required_keys(payload: Dict[str, Any]) -> List[str]:
    """Contract check used by the tests and by ``run.py --check``."""
    return [key for key in REQUIRED_PERCEPTION_KEYS if key not in payload]


__all__ = [
    "NTBackend",
    "NetworkTablesIO",
    "REQUIRED_PERCEPTION_KEYS",
    "SimulatedNT",
    "build_perception_payload",
    "missing_required_keys",
]
