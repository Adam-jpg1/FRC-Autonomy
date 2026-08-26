"""The application: wires the layers together and runs the main loop.

Layer order, top to bottom.  Each layer only depends on the ones above
it, and a failure in a lower layer never corrupts a higher one:

    cameras -> detection -> robot-relative geometry -> localization
    -> field transform -> world model -> navigation -> virtual controller
    -> NetworkTables -> (roboRIO safety gate)

Only the first two layers know what hardware this is running on.  Frames
come from USB cameras on a Raspberry Pi 5; inference runs on a Hailo-8 via
:mod:`frc_perception.detector`; and from :class:`~frc_perception.types.Detection`
downward, nothing changes if the accelerator does.

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
* Frames reach the accelerator only through
  :class:`~frc_perception.inference_scheduler.InferenceScheduler`, which
  takes each camera's NEWEST frame, refuses stale ones, and discards a
  result that finished too late to describe the present.  A frozen camera
  yields nothing rather than feeding a stale image to the detector.
* A detector failure produces NO detections -- never fabricated ones and
  never a silent CPU fallback.  Tracks then age out on their own and
  navigation degrades safely.
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
from .detector import DetectorBackend, build_detector_backend
from .field_map import try_load_field_map
from .inference_scheduler import InferenceScheduler
from .geometry import robot_to_field
from .localization import PoseEstimator
from .logging_setup import RateLimiter, StateLogger
from .navigation import FieldBounds, Navigator, filter_usable_tracks
from .nt_io import NTBackend, NetworkTablesIO, build_perception_payload
from .types import (
    Alliance,
    DesiredMotion,
    Detection,
    FieldObservation,
    InferenceResult,
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
        self.detector: DetectorBackend = build_detector_backend(cfg.detector)
        self.scheduler = InferenceScheduler(self.cameras, self.detector, cfg)

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
        if not self.detector.load():
            logger.error(
                "detector backend %r did not start: %s. The system will run "
                "WITHOUT object detection: no fuel or opponent tracks will be "
                "created, existing tracks age out, and navigation degrades "
                "safely. No CPU fallback is started on purpose.",
                self.cfg.detector.backend, self.detector.load_error,
            )
        self.apriltags.load()
        self.nt.connect()
        self._running = True

    def stop(self) -> None:
        logger.info("stopping perception app")
        self._running = False
        self.cameras.stop()
        try:
            self.detector.close()
        except Exception as exc:  # noqa: BLE001 - shutdown is best effort
            logger.warning("detector close failed: %s", exc)
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
        inference_results = self._stage(
            "detect", self.scheduler.run_cycle, []
        ) or []
        self.stats.inference_ms = (time.monotonic() - inference_started) * 1000.0
        robot_observations = self._stage(
            "ranging", lambda: self._range_results(inference_results, moment), []
        ) or []
        self._update_detector_stats()

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

    def _range_results(
        self, results: Sequence[InferenceResult], now: float
    ) -> List[RobotObservation]:
        """Turn this cycle's detections into robot-frame observations.

        The scheduler has already refused stale frames and dropped results
        that finished too late, so everything arriving here is young enough
        to act on.  Ranging is per-detection and per-camera, using that
        camera's own intrinsics and measured mount pose.
        """
        if not self.detector.healthy():
            self._rate_limiter.log(
                logger, logging.WARNING, "detector_unhealthy",
                "detector (%s) is UNHEALTHY: %s -- producing no detections",
                self.cfg.detector.backend,
                self.detector.load_error or "unknown reason", now=now,
            )
            return []

        observations: List[RobotObservation] = []
        for result in results:
            cam = self._camera_by_name.get(result.camera_id)
            if cam is None or not cam.use_for_objects:
                continue
            observations.extend(self._range_detections(result.detections, cam))
        return observations

    def _range_detections(
        self, detections: Sequence[Detection], cam
    ) -> List[RobotObservation]:
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
            estimate = _range_one(detection, cam, self.cfg.ranging, known_width)
            if estimate is None or not estimate.valid:
                continue
            results.append(
                RobotObservation(
                    camera_id=detection.camera_id,
                    timestamp=detection.capture_timestamp,
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

    def _update_detector_stats(self) -> None:
        """Copy the scheduler's and backend's counters into LoopStats."""
        self.stats.detector_latency_ms = self.detector.last_latency_ms
        self.stats.preprocess_ms = self.detector.last_preprocess_ms
        self.stats.postprocess_ms = self.detector.last_postprocess_ms
        self.stats.frames_skipped = self.scheduler.total_skipped
        self.stats.stale_frames_dropped = self.scheduler.total_stale_frames
        self.stats.stale_results_dropped = self.scheduler.total_stale_results
        health = self.detector.health()
        self._states.log(
            "detector_healthy", health.healthy,
            f"detector {health.backend} healthy ({health.device_report})",
            f"DETECTOR {health.backend.upper()} UNHEALTHY: {health.last_error} "
            "-- no new detections; tracks will age out",
        )

    def _update_world(self, observations, pose_estimate, now: float) -> None:
        if not pose_estimate.valid:
            # Field-space tracks recorded against an unknown pose are worse
            # than no tracks: they look authoritative and are wrong.
            self.world.clear()
            return
        pose = pose_estimate.pose
        field_observations = [
            FieldObservation(
                camera_id=observation.camera_id,
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
        detector_health = self.detector.health()
        camera_health = []
        for name in self._camera_order:
            camera = health.get(name)
            inference = self.scheduler.stats.get(name)
            camera_health.append(
                (
                    camera.healthy if camera else False,
                    camera.fps if camera else 0.0,
                    inference.fps(now) if inference else 0.0,
                    inference.skipped_frames if inference else 0,
                )
            )

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
            model_loaded=detector_health.loaded,
            detector_healthy=detector_health.healthy,
            detector_backend=detector_health.backend,
            detector_classes_ok=detector_health.classes_ok,
            detector_error=detector_health.last_error,
            detector_latency_ms=self.stats.detector_latency_ms,
            preprocess_ms=self.stats.preprocess_ms,
            postprocess_ms=self.stats.postprocess_ms,
            frames_skipped=self.stats.frames_skipped,
            stale_frames_dropped=self.stats.stale_frames_dropped,
            stale_results_dropped=self.stats.stale_results_dropped,
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
            f"{name}={'ok' if h.healthy else 'BAD'}@{h.fps:.0f}"
            f"/{self.scheduler.stats_for(name).fps(now):.0f}"
            for name, h in health.items()
        )
        detector = self.detector.health()
        logger.info(
            "status loop=%.1ffps cam(capture/infer)=%s det=%s%s@%.0fms alliance=%s "
            "pose=%s(%s) fuel=%d opp=%d skipped=%d stale=%d latency=%.0fms nav=%s",
            self.stats.loop_fps, cameras, detector.backend,
            "" if detector.healthy else "-DOWN", self.stats.detector_latency_ms,
            self.alliance.value,
            "VALID" if pose_estimate.valid else "INVALID", pose_estimate.source.value,
            len(self.world.game_pieces(now)), len(self.world.obstacle_robots(now)),
            self.stats.frames_skipped, self.stats.stale_results_dropped,
            self.stats.perception_latency_ms, motion.reason,
        )


def _range_one(detection: Detection, cam, ranging, known_width):
    """Range one detection.  Module level so the call site stays readable."""
    from .geometry import estimate_ground_object

    return estimate_ground_object(detection.bbox, cam, ranging, known_width)


__all__ = ["PerceptionApp"]
