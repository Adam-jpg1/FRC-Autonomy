"""The application: wires the layers together and runs the main loop.

Layer order, top to bottom.  Each layer only depends on the ones above
it, and a failure in a lower layer never corrupts a higher one:

    cameras -> detection -> robot-relative geometry -> localization
    -> field transform -> world model -> navigation -> virtual controller
    -> NetworkTables -> (roboRIO safety gate)

RESILIENCE
----------
Every stage runs inside :meth:`PerceptionApp._stage`, which contains the
exception, counts consecutive failures, rate-limits the log, and escalates
to an ERROR once a stage has failed repeatedly.  One dead camera, one
AprilTag hiccup, or one bad inference does not stop the others -- but a
stage that keeps failing is never silently ignored.

TWO RULES THAT MATTER MORE THAN THE REST
----------------------------------------
* Detections are only transformed into field coordinates while the pose
  is VALID.  With a bad pose those transforms are fiction, so the world
  model is cleared instead of being poisoned with phantom objects.
* Frames are pulled with :meth:`FrameSource.read_fresh`.  A frozen camera
  yields nothing rather than feeding a stale image to YOLO.
"""

from __future__ import annotations

import logging
import math
import signal
import time
from typing import Any, Callable, Dict, List, Optional, Sequence

from .alliance import alliance_from_is_red, is_obstacle, role_for_class
from .apriltag_locator import AprilTagLocalizer, to_grayscale
from .camera import CameraManager, build_camera_manager
from .config import AppConfig, CLASS_FUEL
from .detector import YoloDetector
from .field_map import try_load_field_map
from .geometry import robot_to_field
from .localization import PoseEstimator
from .logging_setup import RateLimiter, StateLogger
from .navigation import FieldBounds, Navigator, filter_usable_tracks
from .nt_io import NTBackend, NetworkTablesIO, build_perception_payload
from .types import (
    Alliance,
    DesiredMotion,
    FieldObservation,
    LoopStats,
    ObjectRole,
    Pose2d,
    RobotObservation,
    TagObservation,
)
from .virtual_controller import VirtualController
from .world_model import WorldModel

logger = logging.getLogger(__name__)


class PerceptionApp:
    """Owns every subsystem and drives one perception cycle at a time."""

    def __init__(
        self,
        cfg: AppConfig,
        nt: Optional[NTBackend] = None,
        replay_paths: Optional[Dict[str, str]] = None,
    ) -> None:
        self.cfg = cfg
        self.stats = LoopStats()
        self._running = False
        self._stop_requested = False

        self._rate_limiter = RateLimiter(cfg.logging.rate_limit_s)
        self._states = StateLogger(logger)
        self._stage_failures: Dict[str, int] = {}

        # -- layers ---------------------------------------------------------
        self.cameras: CameraManager = build_camera_manager(
            cfg.cameras, cfg.frame_freshness_s, replay_paths
        )
        self.detector = YoloDetector(cfg.detector)

        self.field_map, field_map_error = try_load_field_map(
            cfg.localization.apriltag.field_layout_path
        )
        if self.field_map is None:
            logger.error(
                "field layout unavailable (%s); AprilTag localization is DISABLED",
                field_map_error,
            )

        self.apriltags = AprilTagLocalizer(cfg.localization.apriltag, self.field_map)
        self.estimator = PoseEstimator(cfg.localization, self.field_map)
        self.world = WorldModel(cfg.world_model, self.field_map)
        self.navigator = Navigator(cfg.navigation, FieldBounds.from_field_map(self.field_map))
        self.controller = VirtualController(cfg.controller)
        self.nt: NTBackend = nt if nt is not None else NetworkTablesIO(cfg.networktables)

        self.alliance = Alliance.UNKNOWN
        self._loop_times: List[float] = []
        self._last_status_log = 0.0
        self._apriltag_cycle = 0
        self._camera_order = [cam.name for cam in cfg.cameras]
        self._camera_by_name = {cam.name: cam for cam in cfg.cameras}
        self._cameras_seen_healthy: set = set()

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        logger.info("starting perception app")
        self.cameras.start()
        self.detector.load()
        self.apriltags.load()
        self.nt.connect()
        self._running = True

    def stop(self) -> None:
        logger.info("stopping perception app")
        self._running = False
        self.cameras.stop()
        try:
            # Leave the gamepad in a known-neutral state on the way out.
            self.nt.publish_controller(self.controller.neutral())
        except Exception as exc:  # noqa: BLE001 - shutdown is best effort
            logger.warning("could not publish neutral controller state: %s", exc)
        self.nt.close()

    def request_stop(self, *_args: Any) -> None:
        self._stop_requested = True

    def install_signal_handlers(self) -> None:
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, self.request_stop)
            except (ValueError, OSError):  # pragma: no cover - non-main thread
                pass

    # -- error containment -------------------------------------------------

    def _stage(self, name: str, function: Callable[[], Any], default: Any = None) -> Any:
        """Run one pipeline stage, containing and counting its failures."""
        try:
            result = function()
        except Exception as exc:  # noqa: BLE001 - that is the entire point
            failures = self._stage_failures.get(name, 0) + 1
            self._stage_failures[name] = failures
            level = (
                logging.ERROR
                if failures >= self.cfg.stage_failure_alarm_count
                else logging.WARNING
            )
            self._rate_limiter.log(
                logger, level, f"stage:{name}",
                "stage %s failed (%d consecutive): %s", name, failures, exc,
            )
            return default
        if self._stage_failures.get(name):
            logger.info("stage %s recovered after %d failures", name, self._stage_failures[name])
            self._stage_failures[name] = 0
        return result

    # -- the loop ----------------------------------------------------------

    def run(self) -> int:
        """Run until interrupted.  Returns a process exit code."""
        self.start()
        period = 1.0 / self.cfg.loop_hz
        try:
            while not self._stop_requested:
                started = time.monotonic()
                self.step(started)
                elapsed = time.monotonic() - started
                if elapsed < period:
                    time.sleep(period - elapsed)
        except KeyboardInterrupt:
            logger.info("interrupted")
        finally:
            self.stop()
        return 0

    def step(self, now: Optional[float] = None) -> None:
        """One full perception cycle.  Safe to call from a test."""
        moment = time.monotonic() if now is None else now
        self.stats.frame_counter += 1

        frames = self._stage("capture", lambda: self.cameras.fresh_frames(moment), {}) or {}
        self._log_camera_health()

        self._stage("nt_poll", self._poll_connection)
        self._stage("alliance", lambda: self._update_alliance(moment))
        self._stage("odometry", lambda: self._update_odometry(moment))

        apriltag_started = time.monotonic()
        tag_observations = self._stage(
            "apriltags", lambda: self._detect_tags(frames, moment), []
        ) or []
        self.stats.apriltag_ms = (time.monotonic() - apriltag_started) * 1000.0
        self._stage("tag_fusion", lambda: self._fuse_tags(tag_observations, moment))

        pose_estimate = self.estimator.update(moment)

        inference_started = time.monotonic()
        robot_observations = self._stage(
            "detect", lambda: self._detect_objects(frames, moment), []
        ) or []
        self.stats.inference_ms = (time.monotonic() - inference_started) * 1000.0

        self._stage(
            "world_model",
            lambda: self._update_world(robot_observations, pose_estimate, moment),
        )

        motion = self._stage(
            "navigation", lambda: self._navigate(pose_estimate, moment),
            DesiredMotion.stopped("navigation stage failed"),
        )
        if motion is None:
            motion = DesiredMotion.stopped("navigation returned nothing")

        controller_state = self._stage(
            "controller",
            lambda: self.controller.update(
                motion, pose_estimate.pose.theta, moment, pose_estimate.valid
            ),
            self.controller.neutral(),
        )

        publish_started = time.monotonic()
        self._stage(
            "publish",
            lambda: self._publish(frames, pose_estimate, motion, controller_state, moment),
        )
        self.stats.publish_ms = (time.monotonic() - publish_started) * 1000.0

        self._update_loop_stats(moment)
        self._periodic_status(moment, pose_estimate, motion)

    # -- stages ------------------------------------------------------------

    def _poll_connection(self) -> None:
        poll = getattr(self.nt, "poll_connection", None)
        connected = poll() if callable(poll) else self.nt.connected
        self._states.log(
            "nt_connected", connected,
            "NetworkTables connected", "NetworkTables DISCONNECTED",
        )

    def _update_alliance(self, now: float) -> None:
        alliance = alliance_from_is_red(self.nt.read_is_red_alliance())
        if alliance is not self.alliance:
            logger.info("alliance: %s -> %s", self.alliance.value, alliance.value)
            self.alliance = alliance
        if alliance is Alliance.UNKNOWN:
            self._rate_limiter.log(
                logger, logging.WARNING, "alliance_unknown",
                "alliance UNKNOWN (FMSInfo/IsRedAlliance not available); every robot "
                "detection is treated as an obstacle",
                now=now,
            )

    def _update_odometry(self, now: float) -> None:
        reading = self.nt.read_odometry()
        if reading is None:
            return
        x, y, theta, rio_timestamp = reading
        self.estimator.update_odometry(Pose2d(x, y, theta), rio_timestamp, now)

    def _detect_tags(self, frames: Dict[str, Any], now: float) -> List[TagObservation]:
        if not self.apriltags.available:
            return []
        self._apriltag_cycle += 1
        every = max(1, self.cfg.localization.apriltag.detect_every_n_frames)
        if self._apriltag_cycle % every != 0:
            return []

        observations: List[TagObservation] = []
        for name, frame in frames.items():
            cam = self._camera_by_name.get(name)
            if cam is None or not cam.use_for_apriltags:
                continue
            try:
                gray = to_grayscale(frame.image)
                observations.extend(self.apriltags.detect_frame(gray, cam, frame.timestamp))
            except Exception as exc:  # noqa: BLE001 - one camera must not stop the other
                self._rate_limiter.log(
                    logger, logging.WARNING, f"apriltag:{name}",
                    "apriltag detection failed on %s: %s", name, exc, now=now,
                )
        return observations

    def _fuse_tags(self, observations: Sequence[TagObservation], now: float) -> None:
        estimate = self.apriltags.estimate(observations, now)
        if estimate is not None:
            self.estimator.add_tag_estimate(estimate, now)

    def _detect_objects(self, frames: Dict[str, Any], now: float) -> List[RobotObservation]:
        if not self.detector.loaded:
            self._rate_limiter.log(
                logger, logging.WARNING, "model_missing",
                "YOLO model not loaded (%s); no object detection",
                self.detector.load_error or "unknown reason", now=now,
            )
            return []

        observations: List[RobotObservation] = []
        for name, frame in frames.items():
            cam = self._camera_by_name.get(name)
            if cam is None or not cam.use_for_objects:
                continue
            try:
                detections = self.detector.detect(frame)
            except Exception as exc:  # noqa: BLE001 - contained per camera
                self._rate_limiter.log(
                    logger, logging.WARNING, f"infer:{name}",
                    "inference failed on %s: %s", name, exc, now=now,
                )
                continue

            age = time.monotonic() - frame.timestamp
            if age > self.cfg.max_result_age_s:
                # The answer is correct but no longer describes the present.
                self._rate_limiter.log(
                    logger, logging.WARNING, f"stale_result:{name}",
                    "dropping %s result: %.0f ms old (limit %.0f ms)",
                    name, age * 1000.0, self.cfg.max_result_age_s * 1000.0, now=now,
                )
                continue

            observations.extend(self._range_detections(detections, cam))
        return observations

    def _range_detections(self, detections, cam) -> List[RobotObservation]:
        results: List[RobotObservation] = []
        for detection in detections:
            role = role_for_class(detection.class_name, self.alliance)
            if role is ObjectRole.FRIENDLY:
                # Positively identified teammate: never an obstacle, and never
                # counted as an opponent. Dropped here on purpose.
                continue
            if role is ObjectRole.UNKNOWN and not is_obstacle(
                role, self.cfg.world_model.treat_unknown_robots_as_obstacles
            ):
                continue

            known_width = (
                self.cfg.detector.fuel_diameter_m
                if detection.class_name == CLASS_FUEL
                else self.cfg.detector.bumper_width_m
            )
            estimate = self._stage(
                "ranging",
                lambda d=detection, c=cam, w=known_width: _range_one(
                    d, c, self.cfg.ranging, w
                ),
            )
            if estimate is None or not estimate.valid:
                continue
            results.append(
                RobotObservation(
                    camera_name=detection.camera_name,
                    timestamp=detection.timestamp,
                    class_name=detection.class_name,
                    role=role,
                    confidence=detection.confidence,
                    x_robot_m=estimate.x_robot_m,
                    y_robot_m=estimate.y_robot_m,
                    range_m=estimate.range_m,
                    bearing_rad=estimate.bearing_rad,
                    quality=estimate.quality,
                )
            )
        return results

    def _update_world(self, observations, pose_estimate, now: float) -> None:
        if not pose_estimate.valid:
            # Field-space tracks recorded against an unknown pose are worse
            # than no tracks: they look authoritative and are wrong.
            self.world.clear()
            return
        pose = pose_estimate.pose
        field_observations = [
            FieldObservation(
                camera_name=observation.camera_name,
                timestamp=observation.timestamp,
                class_name=observation.class_name,
                role=observation.role,
                confidence=observation.confidence,
                x_field_m=field_x,
                y_field_m=field_y,
                range_m=observation.range_m,
                quality=observation.quality,
            )
            for observation, (field_x, field_y) in (
                (o, robot_to_field(o.x_robot_m, o.y_robot_m, pose.x, pose.y, pose.theta))
                for o in observations
            )
        ]
        self.world.update(field_observations, now)

    def _navigate(self, pose_estimate, now: float) -> DesiredMotion:
        fuel = filter_usable_tracks(
            self.world.game_pieces(now), self.world.freshness, now
        )
        opponents = filter_usable_tracks(
            self.world.obstacle_robots(now), self.world.freshness, now
        )
        return self.navigator.plan(pose_estimate, fuel, opponents, now)

    def _publish(self, frames, pose_estimate, motion, controller_state, now: float) -> None:
        fuel = self.world.game_pieces(now)
        opponents = self.world.obstacle_robots(now)
        health = self.cameras.health()
        camera_health = [
            (
                health[name].healthy if name in health else False,
                health[name].fps if name in health else 0.0,
            )
            for name in self._camera_order
        ]

        latency_ms = 0.0
        if frames:
            oldest = min(frame.timestamp for frame in frames.values())
            latency_ms = max(0.0, (time.monotonic() - oldest) * 1000.0)
        self.stats.perception_latency_ms = latency_ms

        target = motion.target or (math.nan, math.nan)
        payload = build_perception_payload(
            balls_field_x=[t.x for t in fuel],
            balls_field_y=[t.y for t in fuel],
            balls_field_vx=[t.vx for t in fuel],
            balls_field_vy=[t.vy for t in fuel],
            opponents_field_x=[t.x for t in opponents],
            opponents_field_y=[t.y for t in opponents],
            opponents_field_vx=[t.vx for t in opponents],
            opponents_field_vy=[t.vy for t in opponents],
            robot_x=pose_estimate.pose.x,
            robot_y=pose_estimate.pose.y,
            robot_theta=pose_estimate.pose.theta,
            pose_valid=pose_estimate.valid,
            pose_invalid_reason=pose_estimate.invalid_reason,
            frame_counter=self.stats.frame_counter,
            camera_health=camera_health,
            model_loaded=self.detector.loaded,
            odometry_fresh=pose_estimate.odometry_fresh,
            tag_pose_available=pose_estimate.tag_pose_available,
            tag_count=pose_estimate.tag_count,
            localization_source=pose_estimate.source.value,
            localization_quality=pose_estimate.quality,
            translation_disagreement_m=pose_estimate.translation_disagreement_m,
            heading_disagreement_deg=math.degrees(pose_estimate.heading_disagreement_rad),
            loop_fps=self.stats.loop_fps,
            perception_latency_ms=latency_ms,
            inference_ms=self.stats.inference_ms,
            apriltag_ms=self.stats.apriltag_ms,
            nav_reason=motion.reason,
            nav_vx=motion.vx_mps,
            nav_vy=motion.vy_mps,
            nav_omega=motion.omega_radps,
            nav_target_x=float(target[0]),
            nav_target_y=float(target[1]),
            nav_has_target=motion.target is not None,
            virtual_controller_enabled=self.cfg.controller.enabled,
        )
        self.nt.publish_perception(payload)
        self.nt.publish_controller(controller_state)
        flush = getattr(self.nt, "flush", None)
        if callable(flush):
            flush()

    # -- diagnostics -------------------------------------------------------

    def _log_camera_health(self) -> None:
        for name, health in self.cameras.health().items():
            # "recovered" only reads correctly the second time a camera comes
            # up; the first time it is simply starting.
            first_time = name not in self._cameras_seen_healthy
            if health.healthy:
                self._cameras_seen_healthy.add(name)
            self._states.log(
                f"camera:{name}", health.healthy,
                f"camera {name} {'streaming' if first_time else 'RECOVERED'} "
                f"({health.frames_captured} frames)",
                f"camera {name} UNHEALTHY (opened={health.opened} "
                f"age={health.last_frame_age_s:.2f}s {health.last_error})",
            )

    def _update_loop_stats(self, now: float) -> None:
        self._loop_times.append(now)
        cutoff = now - 2.0
        while self._loop_times and self._loop_times[0] < cutoff:
            self._loop_times.pop(0)
        if len(self._loop_times) >= 2:
            span = self._loop_times[-1] - self._loop_times[0]
            self.stats.loop_fps = (len(self._loop_times) - 1) / span if span > 0 else 0.0

    def _periodic_status(self, now: float, pose_estimate, motion) -> None:
        if now - self._last_status_log < self.cfg.logging.status_period_s:
            return
        self._last_status_log = now
        health = self.cameras.health()
        cameras = " ".join(
            f"{name}={'ok' if h.healthy else 'BAD'}@{h.fps:.0f}" for name, h in health.items()
        )
        logger.info(
            "status loop=%.1ffps %s alliance=%s pose=%s(%s) fuel=%d opp=%d "
            "infer=%.0fms latency=%.0fms nav=%s",
            self.stats.loop_fps, cameras, self.alliance.value,
            "VALID" if pose_estimate.valid else "INVALID", pose_estimate.source.value,
            len(self.world.game_pieces(now)), len(self.world.obstacle_robots(now)),
            self.stats.inference_ms, self.stats.perception_latency_ms, motion.reason,
        )


def _range_one(detection, cam, ranging, known_width):
    """Module-level so the lambda in ``_range_detections`` stays readable."""
    from .geometry import estimate_ground_object

    return estimate_ground_object(detection.bbox, cam, ranging, known_width)


__all__ = ["PerceptionApp"]
