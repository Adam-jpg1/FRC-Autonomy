"""Main application: wires cameras -> detector/tags -> world -> nav -> NT.

Design rule: one subsystem failing (a camera unplugged, a bad tag solve,
a model hiccup) logs and degrades — it never kills the loop. The only
things that stop the loop are SIGINT/SIGTERM.
"""

import logging
import signal
import time
from typing import List, Tuple

from .camera import CameraWorker
from .config import AppConfig, CameraConfig
from .detector import Detection, Detector
from .field_map import FieldMap
from .geometry import estimate_object_position, robot_to_field
from .localization import OdometryTracker, PoseFuser, TagLocalizer
from .navigation import Navigator
from .nt_io import NTInterface
from .virtual_controller import VirtualController
from .world_model import FieldDetection, WorldModel, positions

log = logging.getLogger(__name__)


class PerceptionApp:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.field_map = FieldMap(
            cfg.localization.field_layout_path,
            cfg.localization.fallback_field_length_m,
            cfg.localization.fallback_field_width_m,
        )
        self.nt = NTInterface(cfg)
        self.cameras = [CameraWorker(c) for c in cfg.cameras]
        self.detector = Detector(cfg.detector)
        self.tag_localizer = TagLocalizer(cfg.localization, self.field_map, cfg.cameras)
        self.odom = OdometryTracker(cfg.localization.odom_stale_s)
        self.fuser = PoseFuser(cfg.localization)
        self.world = WorldModel(cfg.world_model, self.field_map)
        self.navigator = Navigator(cfg.nav, self.field_map)
        self.controller = VirtualController(cfg.controller)

        self._running = False
        self._frame_counter = 0
        self._last_status_log = 0.0
        self._error_log_times = {}  # stage name -> last log time (rate limiting)

        if cfg.controller.enabled:
            log.warning("VIRTUAL CONTROLLER IS ENABLED — the robot may drive itself")
        else:
            log.info("virtual controller disabled (ControllerConfig.enabled=False)")

    # -- lifecycle ---------------------------------------------------------------

    def run(self) -> None:
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, self._handle_signal)
        for cam in self.cameras:
            cam.start()
        self._running = True
        period = 1.0 / self.cfg.loop_hz
        log.info("perception loop starting at %.0f Hz target", self.cfg.loop_hz)
        while self._running:
            start = time.monotonic()
            try:
                self._cycle(start)
            except Exception:
                log.exception("unhandled error in perception cycle")
            elapsed = time.monotonic() - start
            if elapsed < period:
                time.sleep(period - elapsed)
        self._shutdown()

    def _handle_signal(self, signum, _frame) -> None:
        log.info("signal %d received, shutting down", signum)
        self._running = False

    def _shutdown(self) -> None:
        # Publish a final, safe controller state before exiting.
        try:
            from .virtual_controller import GamepadState
            self.nt.publish_controller(GamepadState(enabled=False))
        except Exception:
            pass
        for cam in self.cameras:
            cam.stop()
        self.nt.close()
        log.info("shutdown complete")

    # -- one cycle -----------------------------------------------------------------

    def _cycle(self, now: float) -> None:
        # 1. Odometry in (staleness judged by RIO timestamp advancing).
        odom_sample = self.nt.get_odometry()
        if odom_sample is not None:
            x, y, theta, rio_ts = odom_sample
            self.odom.update(x, y, theta, rio_ts, now)

        # 2. Fresh frames only. A frozen camera contributes nothing.
        frames: List[Tuple[CameraConfig, object]] = []
        for cam in self.cameras:
            got = cam.get_frame(self.cfg.frame_freshness_s)
            if got is not None:
                frames.append((cam.cfg, got[0]))

        # 3. AprilTag localization (decimated if configured).
        if frames and self.tag_localizer.enabled and (
            self._frame_counter % self.cfg.apriltag_every_n_frames == 0
        ):
            self._run_tags(frames, now)

        # 4. Current best pose.
        pose = self.fuser.estimate(self.odom, now)

        # 5. Object detection + ranging, all cameras batched.
        field_detections = self._run_detection(frames, pose)

        # 6. World model: only update with a valid pose; always prune.
        if pose.valid:
            self.world.update(field_detections, now)
        else:
            self.world.prune(now)

        # 7. Role/alliance routing.
        is_red = self.nt.is_red_alliance()
        balls = self.world.game_pieces()
        opponents = self.world.opponents(is_red)

        # 8. Navigate (degrades internally on invalid pose).
        nav = self.navigator.update(pose, balls, opponents)

        # 9. Publish gamepad + perception.
        self.nt.publish_controller(self.controller.build(nav, pose, balls, now))
        self._frame_counter += 1
        self.nt.publish_perception(
            positions(balls),
            positions(opponents),
            pose.x, pose.y, pose.theta, pose.valid,
            self._frame_counter,
        )

        # 10. Periodic status line.
        if now - self._last_status_log >= self.cfg.status_log_period_s:
            self._last_status_log = now
            ages = ", ".join(
                f"{c.cfg.name}={c.age_s:.2f}s" if c.age_s != float("inf")
                else f"{c.cfg.name}=NONE"
                for c in self.cameras
            )
            log.info(
                "pose=(%.2f, %.2f, %.1f°) %s src=%s | balls=%d opp=%d | nav=%s | "
                "cams[%s] | nt=%s",
                pose.x, pose.y, pose.theta * 57.2958,
                "VALID" if pose.valid else "INVALID", pose.source,
                len(balls), len(opponents), nav.state, ages,
                "up" if self.nt.connected else "DOWN",
            )

    # -- stages ---------------------------------------------------------------------

    def _run_tags(self, frames, now: float) -> None:
        try:
            import cv2

            fixes = []
            for cam_cfg, frame in frames:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                fixes.extend(self.tag_localizer.process(gray, cam_cfg))
            combined = self.tag_localizer.combine(fixes)
            if combined is not None:
                self.fuser.update_from_tag(combined, self.odom, now)
        except Exception:
            self._log_stage_error("apriltag", now)

    def _run_detection(self, frames, pose) -> List[FieldDetection]:
        if not frames:
            return []
        try:
            batches = self.detector.detect_batch([f for _, f in frames])
        except Exception:
            self._log_stage_error("yolo", time.monotonic())
            return []
        out: List[FieldDetection] = []
        for (cam_cfg, _), detections in zip(frames, batches):
            for det in detections:
                est = self._range_detection(det, cam_cfg)
                if est is None:
                    continue
                if not pose.valid:
                    continue  # nothing to project onto without a pose
                fx, fy = robot_to_field(est.x, est.y, pose.x, pose.y, pose.theta)
                out.append(
                    FieldDetection(det.class_name, det.role, fx, fy, det.confidence)
                )
        return out

    def _range_detection(self, det: Detection, cam_cfg: CameraConfig):
        known_width = (
            self.cfg.detector.fuel_diameter_m
            if det.class_name == "fuel"
            else self.cfg.detector.bumper_width_m
        )
        est = estimate_object_position(det.bbox_xyxy, cam_cfg, known_width)
        if est is not None and not est.cross_check_ok:
            log.debug(
                "%s on %s: width range %.2f m disagrees with ground range %.2f m",
                det.class_name, cam_cfg.name,
                est.width_distance_m or -1.0, est.ground_distance_m,
            )
        return est

    def _log_stage_error(self, stage: str, now: float) -> None:
        """Log one traceback per stage per 5 s so a hard failure can't spam."""
        if now - self._error_log_times.get(stage, 0.0) >= 5.0:
            self._error_log_times[stage] = now
            log.exception("stage '%s' failed (rate-limited log)", stage)
