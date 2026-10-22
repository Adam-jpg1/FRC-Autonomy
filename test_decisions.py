#!/usr/bin/env python3
"""Deterministic decision-model test harness for the FRC 2026 autonomy stack.

    python test_decisions.py --scenario all        # run every built-in scenario
    python test_decisions.py --scenario valid_fuel_ahead --verbose
    python test_decisions.py --list                # list scenarios
    python test_decisions.py --json my_tests.json  # run scenarios from a file
    python test_decisions.py --interactive         # type values, see decisions
    python test_decisions.py --write-example-json example_scenarios.json
    python test_decisions.py --gpu-check           # the ONLY GPU-dependent test

WHAT THIS IS
------------
It drives the REAL decision pipeline.  Camera capture and GPU inference are
the only things replaced -- everything that makes a decision is the
production code:

    ScriptedFrameSource  ->  REAL InferenceScheduler        (frc_perception.inference_scheduler)
    ScriptedDetector     ->  REAL Detection type            (frc_perception.types)
                             REAL alliance routing          (frc_perception.alliance)
                             REAL ray-to-ground ranging     (frc_perception.geometry)
                             REAL pose estimator + validity (frc_perception.localization)
                             REAL robot->field transform    (frc_perception.geometry)
                             REAL world model + dedup       (frc_perception.world_model)
                             REAL navigation + avoidance    (frc_perception.navigation)
                             REAL controller mapping        (frc_perception.virtual_controller)
                             REAL NetworkTables payload     (frc_perception.nt_io)

There is no second, simplified decision model anywhere in this file.  The
scenarios call ``PerceptionApp.step()``, the same method the robot runs.

SAFETY
------
* The NetworkTables backend is ALWAYS an in-memory :class:`SimulatedNT`.
  :func:`assert_offline` refuses to run otherwise, and nothing in this file
  imports ``NetworkTablesIO``, so no socket is ever opened and no roboRIO
  can be reached.
* ``controller.enabled`` is forced False on every scenario and asserted
  neutral afterwards, so the virtual gamepad publishes zeros.
* Nothing here commands a motor. The stack never could: it only recommends
  motion, and the roboRIO is the gate.

OUTPUT LABELS -- read these carefully
-------------------------------------
    INPUT    a value this harness fed into the pipeline
    CODE     a value read back OUT of the real implementation
    NOTE     the harness explaining something; NOT produced by the code
    PASS/FAIL an assertion result

Anything on a CODE line came from the production objects.  Anything on a
NOTE line is commentary and proves nothing.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

from frc_perception.app import PerceptionApp  # noqa: E402
from frc_perception.camera import CameraManager, FrameSource  # noqa: E402
from frc_perception.config import (  # noqa: E402
    AppConfig,
    CameraConfig,
    NetworkTablesConfig,
)
from frc_perception.geometry import (  # noqa: E402
    camera_to_robot_transform,
    invert_transform,
)
from frc_perception.detector import YoloDetector  # noqa: E402
from frc_perception.inference_scheduler import InferenceScheduler  # noqa: E402
from frc_perception.nt_io import SimulatedNT, missing_required_keys  # noqa: E402
from frc_perception.types import (  # noqa: E402
    CameraHealth,
    Detection,
    Frame,
    Pose2d,
    TagPoseEstimate,
)

logger = logging.getLogger("decision_harness")


# ===========================================================================
# Safety
# ===========================================================================

SAFETY_BANNER = """\
+----------------------------------------------------------------------+
|  OFFLINE DECISION HARNESS                                            |
|  NetworkTables : in-memory simulation only (no socket, no roboRIO)   |
|  Cameras       : synthetic frames (no V4L2 device opened)            |
|  Inference     : scripted detections (no GPU, no model file)         |
|  Controller    : force-disabled, asserted neutral every scenario     |
|  Motors        : never commanded. This stack only recommends motion. |
+----------------------------------------------------------------------+"""


def assert_offline(app: PerceptionApp) -> None:
    """Refuse to run against anything that could reach a real robot.

    Checked before every scenario.  This is belt-and-braces: the harness
    only ever constructs :class:`SimulatedNT`, but an accidental edit that
    swapped in the real client would otherwise publish live stick values
    onto a robot's network.
    """
    backend_type = type(app.nt).__name__
    if not isinstance(app.nt, SimulatedNT):
        raise SystemExit(
            f"REFUSING TO RUN: NetworkTables backend is {backend_type}, not "
            "SimulatedNT. This harness must never touch a real robot network."
        )
    if app.cfg.controller.enabled:
        raise SystemExit(
            "REFUSING TO RUN: controller.enabled is True. The harness must "
            "run with the virtual gamepad disabled."
        )
    for source in app.cameras.sources:
        if not isinstance(source, ScriptedFrameSource):
            raise SystemExit(
                f"REFUSING TO RUN: camera {source.name} is a "
                f"{type(source).__name__}, not a synthetic source."
            )


# ===========================================================================
# Deterministic clock
# ===========================================================================


class FakeClock:
    """A clock the harness owns, so every decision is reproducible.

    The scheduler takes its clock by injection, and ``app.step(now=...)``
    takes the cycle time, so parking both on this clock makes frame
    freshness, result age and track aging exactly repeatable.
    """

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def set(self, value: float) -> None:
        self.now = value

    def advance(self, seconds: float) -> None:
        self.now += seconds


# ===========================================================================
# Fixture math: robot-relative position -> pixel box
# ===========================================================================
#
# So a scenario can say "a fuel 3 m ahead and 1 m left" instead of hand
# computing pixels.  This is the INVERSE of the production ranging path; a
# built-in scenario (`projection_round_trip`) feeds a projected box back
# through the real `estimate_ground_object` and asserts the position comes
# back out, so this fixture math cannot silently lie about where an object
# was placed.


def project_ground_point(
    cam: CameraConfig, x_robot: float, y_robot: float, z_robot: float = 0.0
) -> Optional[Tuple[float, float]]:
    """Project a robot-frame point into pixel coordinates for ``cam``.

    Returns ``(u, v)`` or ``None`` when the point is behind the camera or
    outside the image.
    """
    robot_from_camera = camera_to_robot_transform(cam)
    camera_from_robot = invert_transform(robot_from_camera)
    point_camera = camera_from_robot @ np.array([x_robot, y_robot, z_robot, 1.0])
    x_c, y_c, z_c = point_camera[0], point_camera[1], point_camera[2]
    if z_c <= 1e-6:
        return None  # behind the lens
    fx, fy, cx, cy = cam.effective_intrinsics()
    u = fx * (x_c / z_c) + cx
    v = fy * (y_c / z_c) + cy
    if not (0.0 <= u < cam.width and 0.0 <= v < cam.height):
        return None  # outside the sensor
    return (float(u), float(v))


def bbox_for_ground_point(
    cam: CameraConfig,
    x_robot: float,
    y_robot: float,
    pixel_height: float = 40.0,
    pixel_width: float = 40.0,
) -> Optional[Tuple[float, float, float, float]]:
    """A bbox whose BOTTOM-CENTER lands on a robot-frame ground point.

    Bottom-center is what the production ranging path uses as the ground
    contact point, so this is the correct way to stage an object at a known
    position.
    """
    projected = project_ground_point(cam, x_robot, y_robot, 0.0)
    if projected is None:
        return None
    u, v = projected
    return (u - pixel_width / 2.0, v - pixel_height, u + pixel_width / 2.0, v)


# ===========================================================================
# Injected input sources (the ONLY things replacing production code)
# ===========================================================================


class ScriptedFrameSource(FrameSource):
    """A camera whose frames the scenario supplies by hand.

    Implements the production :class:`FrameSource` interface, so the real
    :class:`CameraManager` and :class:`InferenceScheduler` treat it exactly
    like a USB camera -- including the freshness and sequence-number rules.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._frame: Optional[Frame] = None
        self.pushed: List[Tuple[int, float]] = []
        self.frames_captured = 0

    def push(self, seq: int, timestamp: float, image: Any = None) -> Frame:
        """Make ``seq`` this camera's latest frame, captured at ``timestamp``."""
        frame = Frame(
            camera_id=self.name,
            seq=seq,
            timestamp=timestamp,
            image=image if image is not None else _blank_image(),
        )
        self._frame = frame
        self.frames_captured += 1
        self.pushed.append((seq, timestamp))
        return frame

    # -- production FrameSource interface ---------------------------------

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def read_latest(self) -> Optional[Frame]:
        return self._frame

    def health(self) -> CameraHealth:
        return CameraHealth(
            name=self.name,
            opened=True,
            healthy=self._frame is not None,
            frames_captured=self.frames_captured,
        )


_BLANK: Optional[Any] = None


def _blank_image() -> Any:
    """A tiny shared black image.  Nothing reads pixels in these scenarios."""
    global _BLANK
    if _BLANK is None:
        _BLANK = np.zeros((8, 8, 3), dtype=np.uint8)
    return _BLANK


class _FakeScalar:
    """Mimics a torch scalar tensor's ``.item()``."""

    def __init__(self, value: Any) -> None:
        self._value = value

    def item(self) -> Any:
        return self._value


class _FakeList:
    """Mimics a torch row tensor's ``.tolist()``."""

    def __init__(self, values: Sequence[float]) -> None:
        self._values = list(values)

    def tolist(self) -> List[float]:
        return self._values


class _FakeBox:
    """One synthetic Ultralytics box."""

    def __init__(self, class_index: int, confidence: float, xyxy: Sequence[float]) -> None:
        self.cls = _FakeScalar(class_index)
        self.conf = _FakeScalar(confidence)
        self.xyxy = [_FakeList(xyxy)]


class _FakeResult:
    """One synthetic Ultralytics Results object."""

    def __init__(self, boxes: List[_FakeBox], names: Dict[int, str]) -> None:
        self.boxes = boxes
        self.names = names
        self.speed = {"preprocess": 0.0, "inference": 0.0, "postprocess": 0.0}


class ScriptedDetector(YoloDetector):
    """Returns the scenario's detections instead of running YOLO.

    Subclasses the PRODUCTION :class:`~frc_perception.detector.YoloDetector`
    and hands synthetic Ultralytics-shaped results to the real
    ``_convert()``.  That matters: the per-class confidence gate, the
    class-name allow-list and the :class:`Detection` construction are all
    production code, so a scenario cannot accidentally test a more
    permissive detector than the robot runs.

    Only ``load()`` and the model call itself are replaced.
    """

    name = "scripted"

    def __init__(self, cfg: Any) -> None:
        super().__init__(cfg)
        #: (camera_id, frame_seq) -> list of detection specs
        self.script: Dict[Tuple[str, int], List["DetectionSpec"]] = {}
        self.cameras: Dict[str, CameraConfig] = {}
        self.healthy_flag = True
        self.fail_message = ""
        self.seen: List[Tuple[str, int]] = []
        self._class_index = {
            name: index for index, name in enumerate(cfg.class_names)
        }
        self._names = {index: name for name, index in self._class_index.items()}

    # -- lifecycle (replaced: no weights file, no GPU) ---------------------

    def load(self) -> bool:
        self._health.loaded = self.healthy_flag
        self._health.healthy = self.healthy_flag
        self._health.classes_ok = True
        self._health.class_report = "scripted detector: production class gate in use"
        self._health.device_report = "scripted (no GPU)"
        self._health.last_error = "" if self.healthy_flag else self.fail_message
        return self.healthy_flag

    def healthy(self) -> bool:
        return self.healthy_flag

    @property
    def loaded(self) -> bool:
        return self.healthy_flag

    @property
    def load_error(self) -> str:
        return self.fail_message

    def close(self) -> None:
        self.healthy_flag = False
        self._health.healthy = False
        self._health.loaded = False

    def set_healthy(self, healthy: bool, message: str = "") -> None:
        self.healthy_flag = healthy
        self.fail_message = message if not healthy else ""
        self._health.healthy = healthy
        self._health.loaded = healthy
        self._health.last_error = self.fail_message

    # -- inference (the model call is replaced; conversion is NOT) ---------

    def predict(
        self, frame: Frame, camera_id: str = "", capture_timestamp: float = 0.0
    ) -> List[Detection]:
        from frc_perception.detector import DetectionError

        self.seen.append((frame.camera_id, frame.seq))
        if not self.healthy_flag:
            self._record_failure(self.fail_message or "scripted detector is down")
            raise DetectionError(self.fail_message or "scripted detector is down")

        specs = self.script.get((frame.camera_id, frame.seq), [])
        cam = self.cameras.get(frame.camera_id)
        boxes: List[_FakeBox] = []
        for spec in specs:
            bbox = spec.resolve_bbox(cam)
            if bbox is None:
                # Outside this camera's field of view or behind the lens.
                continue
            index = self._class_index.get(spec.class_name, -1)
            boxes.append(_FakeBox(index, spec.confidence, bbox))

        self._record_success(0.0)
        # REAL conversion: per-class confidence gate, class-name allow-list,
        # Detection construction.
        return self._convert(
            [_FakeResult(boxes, self._names)],
            camera_id or frame.camera_id,
            frame.seq,
            capture_timestamp or frame.timestamp,
        )


class ScriptedNT(SimulatedNT):
    """:class:`SimulatedNT` with odometry driven by the harness clock.

    Subclassed rather than replaced so the alliance read and the whole
    publish path stay production code; only the odometry timestamp becomes
    deterministic, which is what makes the staleness scenarios exact.
    """

    def __init__(self, clock: FakeClock, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._clock = clock
        self._rio_time = 0.0
        self._last_read = clock()

    def read_odometry(self) -> Optional[Tuple[float, float, float, float]]:
        now = self._clock()
        dt = max(0.0, now - self._last_read)
        self._last_read = now
        if not self.freeze_odometry:
            self._rio_time += dt
            self._pose[0] += self._velocity[0] * dt
            self._pose[1] += self._velocity[1] * dt
            self._pose[2] += self._velocity[2] * dt
        return (self._pose[0], self._pose[1], self._pose[2], self._rio_time)


# ===========================================================================
# Scenario specification (declarative, JSON-serializable)
# ===========================================================================


@dataclass
class DetectionSpec:
    """One synthetic detection.

    Give EITHER a robot-relative position (``x_robot_m``/``y_robot_m``,
    which the harness projects to a pixel box through the camera's real
    intrinsics and mount) OR an explicit ``bbox``.  Position is usually
    what you want; an explicit bbox is for testing geometry rejections such
    as a box clipped by the frame edge.
    """

    class_name: str = "fuel"
    confidence: float = 0.90
    x_robot_m: Optional[float] = None
    y_robot_m: Optional[float] = None
    bbox: Optional[Tuple[float, float, float, float]] = None
    pixel_width: float = 40.0
    pixel_height: float = 40.0

    def resolve_bbox(
        self, cam: Optional[CameraConfig]
    ) -> Optional[Tuple[float, float, float, float]]:
        if self.bbox is not None:
            return tuple(self.bbox)  # type: ignore[return-value]
        if cam is None or self.x_robot_m is None or self.y_robot_m is None:
            return None
        return bbox_for_ground_point(
            cam, self.x_robot_m, self.y_robot_m, self.pixel_height, self.pixel_width
        )

    @staticmethod
    def from_dict(payload: Dict[str, Any]) -> "DetectionSpec":
        return DetectionSpec(
            class_name=payload.get("class_name", "fuel"),
            confidence=float(payload.get("confidence", 0.90)),
            x_robot_m=_opt_float(payload.get("x_robot_m")),
            y_robot_m=_opt_float(payload.get("y_robot_m")),
            bbox=tuple(payload["bbox"]) if payload.get("bbox") else None,
            pixel_width=float(payload.get("pixel_width", 40.0)),
            pixel_height=float(payload.get("pixel_height", 40.0)),
        )


@dataclass
class FrameSpec:
    """One synthetic camera frame delivered on a given cycle."""

    camera_id: str = "cam0"
    seq: int = 1
    #: Capture time, as an offset from the cycle's clock value. Negative
    #: means the frame was grabbed in the past -- that is how you stage a
    #: stale frame.
    capture_offset_s: float = 0.0
    detections: List[DetectionSpec] = field(default_factory=list)

    @staticmethod
    def from_dict(payload: Dict[str, Any]) -> "FrameSpec":
        return FrameSpec(
            camera_id=payload.get("camera_id", "cam0"),
            seq=int(payload.get("seq", 1)),
            capture_offset_s=float(payload.get("capture_offset_s", 0.0)),
            detections=[
                DetectionSpec.from_dict(d) for d in payload.get("detections", [])
            ],
        )


@dataclass
class CycleSpec:
    """One iteration of the main loop."""

    #: Seconds to advance the harness clock BEFORE this cycle runs.
    advance_s: float = 0.033
    frames: List[FrameSpec] = field(default_factory=list)
    #: Flip detector health at the start of this cycle (None = leave as is).
    detector_healthy: Optional[bool] = None
    #: Freeze/unfreeze the simulated roboRIO odometry publisher.
    freeze_odometry: Optional[bool] = None
    #: Inject an AprilTag pose fix, as ``[x, y, theta_deg]``, through the
    #: REAL PoseEstimator.add_tag_estimate().
    inject_tag_pose: Optional[Tuple[float, float, float]] = None
    note: str = ""

    @staticmethod
    def from_dict(payload: Dict[str, Any]) -> "CycleSpec":
        tag = payload.get("inject_tag_pose")
        return CycleSpec(
            advance_s=float(payload.get("advance_s", 0.033)),
            frames=[FrameSpec.from_dict(f) for f in payload.get("frames", [])],
            detector_healthy=payload.get("detector_healthy"),
            freeze_odometry=payload.get("freeze_odometry"),
            inject_tag_pose=tuple(tag) if tag else None,
            note=payload.get("note", ""),
        )


@dataclass
class ScenarioSpec:
    """A complete, self-contained decision test."""

    name: str
    description: str = ""
    #: "odometry_only"   pose valid from odometry alone (require_tag_init off)
    #: "tag_initialized" pose valid after a real injected tag fix
    #: "invalid"         pose never becomes valid (no tag fix, tag init required)
    pose_mode: str = "odometry_only"
    robot_pose: Tuple[float, float, float] = (5.0, 3.0, 0.0)
    #: True = we are RED, False = we are BLUE, None = FMS data unavailable
    is_red_alliance: Optional[bool] = True
    cameras: List[str] = field(default_factory=lambda: ["cam0", "cam1"])
    cycles: List[CycleSpec] = field(default_factory=list)
    expect: Dict[str, Any] = field(default_factory=dict)
    #: Config overrides applied before the app is built, e.g.
    #: {"navigation.search_point_field": null, "world_model.min_hits_to_confirm": 1}
    config: Dict[str, Any] = field(default_factory=dict)
    requires_gpu: bool = False

    @staticmethod
    def from_dict(payload: Dict[str, Any]) -> "ScenarioSpec":
        if "name" not in payload:
            raise ValueError("every scenario needs a 'name'")
        pose = payload.get("robot_pose", [5.0, 3.0, 0.0])
        return ScenarioSpec(
            name=payload["name"],
            description=payload.get("description", ""),
            pose_mode=payload.get("pose_mode", "odometry_only"),
            robot_pose=(float(pose[0]), float(pose[1]), float(pose[2])),
            is_red_alliance=payload.get("is_red_alliance", True),
            cameras=list(payload.get("cameras", ["cam0", "cam1"])),
            cycles=[CycleSpec.from_dict(c) for c in payload.get("cycles", [])],
            expect=dict(payload.get("expect", {})),
            config=dict(payload.get("config", {})),
            requires_gpu=bool(payload.get("requires_gpu", False)),
        )


def _opt_float(value: Any) -> Optional[float]:
    return None if value is None else float(value)


# ===========================================================================
# Synthetic camera geometry
# ===========================================================================
#
# Exact, hand-chosen values so expected positions can be reasoned about on
# paper.  These are NOT measurements of a real robot -- a real bring-up
# fills these in from scripts/calibrate_intrinsics.py and a tape measure.

SYNTHETIC_CAMERAS: Dict[str, Dict[str, Any]] = {
    # Forward-looking, 0.6 m up, pitched 20 deg down.
    "cam0": dict(
        width=640, height=480, fx=500.0, fy=500.0, cx=320.0, cy=240.0,
        mount_x_m=0.30, mount_y_m=0.0, mount_z_m=0.60,
        pitch_deg=20.0, yaw_deg=0.0, roll_deg=0.0,
    ),
    # Same optics, mounted facing 30 deg to the robot's left, so the two
    # cameras overlap and cross-camera de-duplication gets exercised.
    "cam1": dict(
        width=640, height=480, fx=500.0, fy=500.0, cx=320.0, cy=240.0,
        mount_x_m=0.30, mount_y_m=0.0, mount_z_m=0.60,
        pitch_deg=20.0, yaw_deg=30.0, roll_deg=0.0,
    ),
}


def make_camera(name: str) -> CameraConfig:
    if name not in SYNTHETIC_CAMERAS:
        raise ValueError(
            f"unknown synthetic camera {name!r}; have {sorted(SYNTHETIC_CAMERAS)}"
        )
    return CameraConfig(name=name, **SYNTHETIC_CAMERAS[name])


# ===========================================================================
# Per-cycle trace of what the real code decided
# ===========================================================================


@dataclass
class CycleTrace:
    """Everything observable after one real ``app.step()``.

    Every field here is read back OUT of the production objects.
    """

    index: int
    clock: float
    note: str = ""

    frames_pushed: List[Tuple[str, int, float]] = field(default_factory=list)
    detector_healthy: bool = True
    #: Detections the scenario asked for that this camera physically cannot
    #: see. Surfaced loudly: a silently dropped object makes a test lie.
    staging_warnings: List[str] = field(default_factory=list)

    # scheduler
    inferred: List[Tuple[str, int]] = field(default_factory=list)
    frames_skipped: int = 0
    stale_frames_dropped: int = 0
    stale_results_dropped: int = 0

    # ranging (robot frame)
    observations: List[Dict[str, Any]] = field(default_factory=list)

    # localization
    pose_valid: bool = False
    pose: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    pose_source: str = "none"
    pose_invalid_reason: str = ""

    # world model (field frame)
    fuel_tracks: List[Dict[str, Any]] = field(default_factory=list)
    opponent_tracks: List[Dict[str, Any]] = field(default_factory=list)
    all_track_count: int = 0

    # navigation
    nav_vx: float = 0.0
    nav_vy: float = 0.0
    nav_omega: float = 0.0
    nav_reason: str = ""
    nav_target: Optional[Tuple[float, float]] = None
    nav_confidence: float = 0.0
    nav_field_relative: bool = True
    pockets: List[Dict[str, Any]] = field(default_factory=list)

    # controller
    controller_enabled: bool = False
    controller_axes: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    controller_heartbeat: int = 0
    controller_buttons_any: bool = False

    # published contract
    published: Dict[str, Any] = field(default_factory=dict)

    @property
    def translation_magnitude(self) -> float:
        return math.hypot(self.nav_vx, self.nav_vy)


# ===========================================================================
# The harness
# ===========================================================================


class DecisionHarness:
    """Builds a real :class:`PerceptionApp` wired to synthetic inputs."""

    def __init__(self, spec: ScenarioSpec) -> None:
        self.spec = spec
        self.clock = FakeClock()
        self.cfg = self._build_config(spec)
        self.sources = {name: ScriptedFrameSource(name) for name in spec.cameras}
        self.detector = ScriptedDetector(self.cfg.detector)
        self.detector.cameras = {c.name: c for c in self.cfg.cameras}

        self.nt = ScriptedNT(
            self.clock,
            is_red=spec.is_red_alliance,
            start_pose=spec.robot_pose,
        )

        self.app = PerceptionApp(self.cfg, nt=self.nt)
        # Replace ONLY the two hardware-coupled inputs. Everything that
        # makes a decision stays production code.
        self.app.cameras = CameraManager(
            list(self.sources.values()), self.cfg.frame_freshness_s
        )
        self.app.detector = self.detector
        self.app.scheduler = InferenceScheduler(
            self.app.cameras, self.detector, self.cfg, clock=self.clock
        )

        # Record the REAL ranging stage's output rather than re-deriving it,
        # so the trace can never show an observation the pipeline rejected.
        self._observed: List[Any] = []
        _original_range = self.app._range_results

        def _recording_range(results: Any, now: float) -> Any:
            produced = _original_range(results, now)
            self._observed = list(produced)
            return produced

        self.app._range_results = _recording_range  # type: ignore[method-assign]

        assert_offline(self.app)
        self.traces: List[CycleTrace] = []

    # -- configuration -----------------------------------------------------

    def _build_config(self, spec: ScenarioSpec) -> AppConfig:
        cfg = AppConfig(
            # An explicit server string keeps the config valid without a team
            # number; it is never used because the backend is simulated.
            networktables=NetworkTablesConfig(server_override="127.0.0.1"),
        )
        cfg.cameras = [make_camera(name) for name in spec.cameras]

        # Harness defaults that make single-cycle scenarios legible.
        cfg.world_model.min_hits_to_confirm = 1
        cfg.detector.require_cuda = False  # scripted detector, no GPU involved
        cfg.controller.enabled = False     # SAFETY: never enabled here
        cfg.localization.require_tag_initialization = spec.pose_mode != "odometry_only"

        for dotted, value in spec.config.items():
            _set_dotted(cfg, dotted, value)

        # SAFETY: re-assert after overrides, so a scenario cannot enable it.
        cfg.controller.enabled = False
        return cfg

    # -- execution ---------------------------------------------------------

    def run(self) -> List[CycleTrace]:
        cycles = self.spec.cycles or [CycleSpec()]
        self.app.start()
        try:
            for index, cycle in enumerate(cycles):
                self.traces.append(self._run_cycle(index, cycle))
        finally:
            self.app.stop()
        return self.traces

    def _run_cycle(self, index: int, cycle: CycleSpec) -> CycleTrace:
        self.clock.advance(cycle.advance_s)
        now = self.clock()

        if cycle.detector_healthy is not None:
            self.detector.set_healthy(
                cycle.detector_healthy,
                "" if cycle.detector_healthy else "scripted detector failure",
            )

        if cycle.freeze_odometry is not None:
            self.nt.freeze_odometry = cycle.freeze_odometry

        trace = CycleTrace(index=index, clock=now, note=cycle.note)

        # Deliver this cycle's frames and register their detections.
        for frame_spec in cycle.frames:
            source = self.sources.get(frame_spec.camera_id)
            if source is None:
                raise ValueError(
                    f"scenario {self.spec.name!r} pushes a frame for unknown camera "
                    f"{frame_spec.camera_id!r}; cameras are {sorted(self.sources)}"
                )
            capture_time = now + frame_spec.capture_offset_s
            source.push(frame_spec.seq, capture_time)
            self.detector.script[(frame_spec.camera_id, frame_spec.seq)] = (
                frame_spec.detections
            )
            cam = self.detector.cameras.get(frame_spec.camera_id)
            for spec in frame_spec.detections:
                if spec.resolve_bbox(cam) is None:
                    trace.staging_warnings.append(
                        f"{frame_spec.camera_id} CANNOT SEE {spec.class_name} at "
                        f"x={spec.x_robot_m} y={spec.y_robot_m} "
                        "(outside its field of view, or behind the lens) -- "
                        "this detection was NOT staged"
                    )
            trace.frames_pushed.append(
                (frame_spec.camera_id, frame_spec.seq, frame_spec.capture_offset_s)
            )

        # Optional AprilTag fix, through the REAL fusion entry point.
        if cycle.inject_tag_pose is not None:
            x, y, theta_deg = cycle.inject_tag_pose
            self.app.estimator.add_tag_estimate(
                TagPoseEstimate(
                    pose=Pose2d(x, y, math.radians(theta_deg)),
                    timestamp=now,
                    tag_ids=[1],
                    mean_distance_m=2.0,
                    quality=1.0,
                ),
                now,
            )

        seen_before = len(self.detector.seen)
        self._observed = []

        # ---- the real main loop ----
        self.app.step(now=now)

        self._capture_trace(trace, now, seen_before)
        return trace

    def _capture_trace(self, trace: CycleTrace, now: float, seen_before: int) -> None:
        """Read decisions back out of the production objects."""
        app = self.app

        trace.detector_healthy = self.detector.healthy()
        trace.inferred = self.detector.seen[seen_before:]
        trace.frames_skipped = app.scheduler.total_skipped
        trace.stale_frames_dropped = app.scheduler.total_stale_frames
        trace.stale_results_dropped = app.scheduler.total_stale_results

        estimate = app.estimator.last_estimate
        trace.pose_valid = estimate.valid
        trace.pose = (estimate.pose.x, estimate.pose.y, estimate.pose.theta)
        trace.pose_source = estimate.source.value
        trace.pose_invalid_reason = estimate.invalid_reason

        trace.fuel_tracks = [_track_dict(t) for t in app.world.game_pieces(now)]
        trace.opponent_tracks = [_track_dict(t) for t in app.world.obstacle_robots(now)]
        trace.all_track_count = len(app.world.all_tracks)

        trace.pockets = [
            {
                "x": round(p.x, 3), "y": round(p.y, 3), "count": p.count,
                "distance_m": round(p.distance_m, 3),
                "blocked": p.blocked, "score": round(p.score, 3),
                "clearance_m": (
                    None if not math.isfinite(p.path_clearance_m)
                    else round(p.path_clearance_m, 3)
                ),
            }
            for p in app.navigator.last_pockets
        ]

        published = dict(self.nt.published_perception)
        trace.published = published
        trace.nav_vx = float(published.get("nav_vx_mps", 0.0))
        trace.nav_vy = float(published.get("nav_vy_mps", 0.0))
        trace.nav_omega = float(published.get("nav_omega_radps", 0.0))
        trace.nav_reason = str(published.get("nav_reason", ""))
        if published.get("nav_has_target"):
            trace.nav_target = (
                float(published["nav_target_x"]), float(published["nav_target_y"])
            )

        controller = dict(self.nt.published_controller)
        trace.controller_enabled = bool(controller.get("enabled", False))
        trace.controller_axes = (
            float(controller.get("left_x", 0.0)),
            float(controller.get("left_y", 0.0)),
            float(controller.get("right_x", 0.0)),
        )
        trace.controller_heartbeat = int(controller.get("heartbeat", 0))
        trace.controller_buttons_any = any(
            bool(controller.get(key, False))
            for key in ("a", "b", "x", "y", "left_bumper", "right_bumper",
                        "back", "start", "left_stick", "right_stick")
        )

        # Robot-frame observations exactly as the real ranging stage emitted
        # them this cycle -- the same list the world model was handed.
        trace.observations = [
            {
                "camera_id": observation.camera_id,
                "class_name": observation.class_name,
                "role": observation.role.value,
                "confidence": round(observation.confidence, 3),
                "x_robot_m": round(observation.x_robot_m, 3),
                "y_robot_m": round(observation.y_robot_m, 3),
                "range_m": round(observation.range_m, 3),
                "bearing_deg": round(math.degrees(observation.bearing_rad), 2),
            }
            for observation in self._observed
        ]

def _track_dict(track: Any) -> Dict[str, Any]:
    return {
        "id": track.track_id,
        "class_name": track.class_name,
        "role": track.role.value,
        "x": round(track.x, 3),
        "y": round(track.y, 3),
        "vx": round(track.vx, 3),
        "vy": round(track.vy, 3),
        "confidence": round(track.confidence, 3),
        "hits": track.hit_count,
    }


def _set_dotted(cfg: AppConfig, dotted: str, value: Any) -> None:
    """Apply ``{"navigation.max_speed_mps": 1.0}`` style config overrides."""
    target: Any = cfg
    parts = dotted.split(".")
    for part in parts[:-1]:
        if not hasattr(target, part):
            raise ValueError(f"unknown config section {part!r} in {dotted!r}")
        target = getattr(target, part)
    leaf = parts[-1]
    if not hasattr(target, leaf):
        raise ValueError(f"unknown config field {dotted!r}")
    if isinstance(value, list):
        value = tuple(value)
    setattr(target, leaf, value)


# ===========================================================================
# Assertions
# ===========================================================================
#
# Every scenario declares its expected behavior and is PASSED or FAILED
# against the final cycle's trace.  These are real assertions, not prints:
# the process exit code is non-zero if any check fails.


@dataclass
class CheckResult:
    key: str
    passed: bool
    detail: str


#: Documentation for every supported expectation key, printed by --help-expect
#: and used to reject typos in a JSON file.
EXPECTATION_KEYS: Dict[str, str] = {
    "pose_valid": "bool -- the pose-validity gate",
    "pose_source": "str -- 'odometry_only' | 'odometry_tag_corrected' | 'tag_only' | 'none'",
    "pose_near": "[x, y, tol] -- fused pose within tol metres",
    "translation_zero": "bool -- True asserts vx == vy == 0.0 exactly",
    "translation_nonzero": "bool -- True asserts some commanded translation",
    "nav_vx_sign": "+1 / -1 / 0 -- sign of the field-frame X command",
    "nav_vy_sign": "+1 / -1 / 0 -- sign of the field-frame Y command",
    "omega_zero": "bool -- True asserts no commanded rotation",
    "omega_nonzero": "bool -- True asserts some commanded rotation",
    "nav_reason_contains": "str -- substring of the navigation reason",
    "has_target": "bool -- navigation selected a field target",
    "target_near": "[x, y, tol] -- selected target within tol metres",
    "fuel_tracks": "int -- exact reported fuel track count",
    "min_fuel_tracks": "int",
    "max_fuel_tracks": "int",
    "opponent_tracks": "int -- exact reported obstacle-robot count",
    "all_tracks": "int -- exact live track count, confirmed or not",
    "fuel_track_near": "[x, y, tol] -- some fuel track within tol metres",
    "observations": "int -- exact ranged robot-frame observation count",
    "min_observations": "int",
    "inferred_frames": "[[camera_id, seq], ...] -- exactly these frames were inferred on",
    "frames_skipped": "int -- exact scheduler skip count",
    "min_frames_skipped": "int",
    "stale_frames_dropped": "int -- frames refused for being too old",
    "stale_results_dropped": "int -- results that finished too late",
    "detector_healthy": "bool",
    "controller_enabled": "bool -- MUST be False in this harness",
    "controller_axes_zero": "bool -- all published stick axes exactly 0.0",
    "controller_buttons_none": "bool -- no gamepad button pressed",
    "heartbeat_advanced": "bool -- heartbeat moved across the scenario",
    "contract_complete": "bool -- every required Perception key published",
    "target_pocket_blocked": "bool -- the selected pocket's path is obstructed",
    "target_pocket_count": "int -- pieces in the selected pocket",
    "max_target_pocket_score": "float -- selected pocket score at or below this",
    "allow_unstaged_detections": (
        "bool -- set True when a scenario deliberately places an object "
        "outside a camera's field of view"
    ),
}


def check_expectations(spec: ScenarioSpec, traces: List[CycleTrace]) -> List[CheckResult]:
    """Evaluate a scenario's declared expectations against the final cycle."""
    unknown = sorted(set(spec.expect) - set(EXPECTATION_KEYS))
    if unknown:
        return [
            CheckResult(
                key=key, passed=False,
                detail="unknown expectation key (see --help-expect)",
            )
            for key in unknown
        ]

    final = traces[-1]
    results: List[CheckResult] = []

    # A detection the camera cannot see is almost always a scenario mistake,
    # and a silent drop makes the test assert the wrong thing. Fail loudly
    # unless the scenario says it is intentional.
    warnings = [w for trace in traces for w in trace.staging_warnings]
    if warnings and not spec.expect.get("allow_unstaged_detections"):
        results.append(CheckResult(
            "staging", False,
            f"{len(warnings)} detection(s) could not be staged: {warnings[0]}"
            + (" (and more)" if len(warnings) > 1 else ""),
        ))

    def record(key: str, passed: bool, detail: str) -> None:
        results.append(CheckResult(key, passed, detail))

    for key, want in spec.expect.items():
        if key == "pose_valid":
            record(key, final.pose_valid is bool(want),
                   f"want {want}, got {final.pose_valid} "
                   f"({final.pose_invalid_reason or 'valid'})")
        elif key == "pose_source":
            record(key, final.pose_source == want,
                   f"want {want!r}, got {final.pose_source!r}")
        elif key == "pose_near":
            x, y, tol = want
            distance = math.hypot(final.pose[0] - x, final.pose[1] - y)
            record(key, distance <= tol,
                   f"want within {tol} m of ({x}, {y}), got "
                   f"({final.pose[0]:.3f}, {final.pose[1]:.3f}), off by {distance:.3f} m")
        elif key == "translation_zero":
            is_zero = final.nav_vx == 0.0 and final.nav_vy == 0.0
            record(key, is_zero is bool(want),
                   f"vx={final.nav_vx:.4f} vy={final.nav_vy:.4f}")
        elif key == "translation_nonzero":
            nonzero = final.translation_magnitude > 1e-9
            record(key, nonzero is bool(want),
                   f"|v|={final.translation_magnitude:.4f} m/s")
        elif key == "nav_vx_sign":
            record(key, _sign(final.nav_vx) == int(want),
                   f"want sign {want}, got {_sign(final.nav_vx)} (vx={final.nav_vx:.4f})")
        elif key == "nav_vy_sign":
            record(key, _sign(final.nav_vy) == int(want),
                   f"want sign {want}, got {_sign(final.nav_vy)} (vy={final.nav_vy:.4f})")
        elif key == "omega_zero":
            is_zero = final.nav_omega == 0.0
            record(key, is_zero is bool(want), f"omega={final.nav_omega:.4f} rad/s")
        elif key == "omega_nonzero":
            nonzero = abs(final.nav_omega) > 1e-9
            record(key, nonzero is bool(want), f"omega={final.nav_omega:.4f} rad/s")
        elif key == "nav_reason_contains":
            record(key, str(want).lower() in final.nav_reason.lower(),
                   f"want {want!r} in reason, got {final.nav_reason!r}")
        elif key == "has_target":
            record(key, (final.nav_target is not None) is bool(want),
                   f"target={final.nav_target}")
        elif key == "target_near":
            x, y, tol = want
            if final.nav_target is None:
                record(key, False, f"want within {tol} m of ({x}, {y}), got no target")
            else:
                distance = math.hypot(final.nav_target[0] - x, final.nav_target[1] - y)
                record(key, distance <= tol,
                       f"want within {tol} m of ({x}, {y}), got "
                       f"({final.nav_target[0]:.3f}, {final.nav_target[1]:.3f}), "
                       f"off by {distance:.3f} m")
        elif key == "fuel_tracks":
            record(key, len(final.fuel_tracks) == int(want),
                   f"want {want}, got {len(final.fuel_tracks)}")
        elif key == "min_fuel_tracks":
            record(key, len(final.fuel_tracks) >= int(want),
                   f"want >= {want}, got {len(final.fuel_tracks)}")
        elif key == "max_fuel_tracks":
            record(key, len(final.fuel_tracks) <= int(want),
                   f"want <= {want}, got {len(final.fuel_tracks)}")
        elif key == "opponent_tracks":
            record(key, len(final.opponent_tracks) == int(want),
                   f"want {want}, got {len(final.opponent_tracks)}")
        elif key == "all_tracks":
            record(key, final.all_track_count == int(want),
                   f"want {want}, got {final.all_track_count}")
        elif key == "fuel_track_near":
            x, y, tol = want
            best = min(
                (math.hypot(t["x"] - x, t["y"] - y) for t in final.fuel_tracks),
                default=float("inf"),
            )
            record(key, best <= tol,
                   f"want a fuel track within {tol} m of ({x}, {y}); "
                   f"closest was {best:.3f} m away "
                   f"(tracks={[(t['x'], t['y']) for t in final.fuel_tracks]})")
        elif key == "observations":
            record(key, len(final.observations) == int(want),
                   f"want {want}, got {len(final.observations)}")
        elif key == "min_observations":
            record(key, len(final.observations) >= int(want),
                   f"want >= {want}, got {len(final.observations)}")
        elif key == "inferred_frames":
            expected = [tuple(pair) for pair in want]
            actual = [tuple(pair) for pair in final.inferred]
            record(key, actual == expected, f"want {expected}, got {actual}")
        elif key == "frames_skipped":
            record(key, final.frames_skipped == int(want),
                   f"want {want}, got {final.frames_skipped}")
        elif key == "min_frames_skipped":
            record(key, final.frames_skipped >= int(want),
                   f"want >= {want}, got {final.frames_skipped}")
        elif key == "stale_frames_dropped":
            record(key, final.stale_frames_dropped == int(want),
                   f"want {want}, got {final.stale_frames_dropped}")
        elif key == "stale_results_dropped":
            record(key, final.stale_results_dropped == int(want),
                   f"want {want}, got {final.stale_results_dropped}")
        elif key == "detector_healthy":
            record(key, final.detector_healthy is bool(want),
                   f"want {want}, got {final.detector_healthy}")
        elif key == "controller_enabled":
            record(key, final.controller_enabled is bool(want),
                   f"want {want}, got {final.controller_enabled}")
        elif key == "controller_axes_zero":
            is_zero = all(axis == 0.0 for axis in final.controller_axes)
            record(key, is_zero is bool(want), f"axes={final.controller_axes}")
        elif key == "controller_buttons_none":
            record(key, (not final.controller_buttons_any) is bool(want),
                   f"any button pressed = {final.controller_buttons_any}")
        elif key == "heartbeat_advanced":
            advanced = final.controller_heartbeat > traces[0].controller_heartbeat or (
                len(traces) == 1 and final.controller_heartbeat > 0
            )
            record(key, advanced is bool(want),
                   f"heartbeat {traces[0].controller_heartbeat} -> "
                   f"{final.controller_heartbeat}")
        elif key == "target_pocket_blocked":
            if not final.pockets:
                record(key, False, "no pocket was scored")
            else:
                record(key, final.pockets[0]["blocked"] is bool(want),
                       f"best pocket blocked={final.pockets[0]['blocked']} "
                       f"clearance={final.pockets[0]['clearance_m']} m")
        elif key == "target_pocket_count":
            if not final.pockets:
                record(key, False, "no pocket was scored")
            else:
                record(key, final.pockets[0]["count"] == int(want),
                       f"want {want}, got {final.pockets[0]['count']}")
        elif key == "max_target_pocket_score":
            if not final.pockets:
                record(key, False, "no pocket was scored")
            else:
                record(key, final.pockets[0]["score"] <= float(want),
                       f"want score <= {want}, got {final.pockets[0]['score']}")
        elif key == "allow_unstaged_detections":
            record(key, True, "scenario permits out-of-view placements")
        elif key == "contract_complete":
            missing = missing_required_keys(final.published)
            record(key, (not missing) is bool(want), f"missing keys: {missing}")

    return results


#: Safety checks applied to EVERY scenario, whether it declares them or not.
def safety_checks(traces: List[CycleTrace]) -> List[CheckResult]:
    """Non-negotiable invariants.  Added to every scenario automatically."""
    results: List[CheckResult] = []
    for trace in traces:
        if trace.controller_enabled:
            results.append(CheckResult(
                "SAFETY:controller_disabled", False,
                f"cycle {trace.index}: virtual controller reported ENABLED",
            ))
        if any(axis != 0.0 for axis in trace.controller_axes):
            results.append(CheckResult(
                "SAFETY:axes_neutral", False,
                f"cycle {trace.index}: non-zero stick axes {trace.controller_axes}",
            ))
        if trace.controller_buttons_any:
            results.append(CheckResult(
                "SAFETY:buttons_neutral", False,
                f"cycle {trace.index}: a gamepad button was pressed",
            ))
        if not trace.pose_valid and trace.translation_magnitude > 0.0:
            results.append(CheckResult(
                "SAFETY:no_translation_without_pose", False,
                f"cycle {trace.index}: pose INVALID but translation "
                f"|v|={trace.translation_magnitude:.4f} m/s was commanded",
            ))
    if not results:
        results.append(CheckResult(
            "SAFETY", True,
            "controller disabled and neutral every cycle; no translation "
            "without a valid pose",
        ))
    return results


def _sign(value: float, tolerance: float = 1e-9) -> int:
    if value > tolerance:
        return 1
    if value < -tolerance:
        return -1
    return 0


# ===========================================================================
# Trace printing
# ===========================================================================

LABEL_INPUT = "INPUT "
LABEL_CODE = "CODE  "
LABEL_NOTE = "NOTE  "


def print_scenario_header(spec: ScenarioSpec) -> None:
    print()
    print("=" * 76)
    print(f"SCENARIO: {spec.name}")
    print("=" * 76)
    if spec.description:
        for line in spec.description.strip().splitlines():
            print(f"  {LABEL_NOTE}| {line.strip()}")
    alliance = (
        "RED" if spec.is_red_alliance is True
        else "BLUE" if spec.is_red_alliance is False
        else "UNKNOWN (no FMS data)"
    )
    print(f"  {LABEL_INPUT}| alliance            = {alliance}")
    print(f"  {LABEL_INPUT}| robot_pose          = {spec.robot_pose}")
    print(f"  {LABEL_INPUT}| pose_mode           = {spec.pose_mode}")
    print(f"  {LABEL_INPUT}| cameras             = {spec.cameras}")
    for dotted, value in sorted(spec.config.items()):
        print(f"  {LABEL_INPUT}| config {dotted} = {value}")


def print_cycle(trace: CycleTrace, verbose: bool) -> None:
    print()
    print(f"  --- cycle {trace.index}  (t = {trace.clock:.3f}s) ---")
    if trace.note:
        print(f"  {LABEL_NOTE}| {trace.note}")

    for camera_id, seq, offset in trace.frames_pushed:
        age = "" if offset == 0.0 else f"  captured {abs(offset) * 1000:.0f} ms ago"
        print(f"  {LABEL_INPUT}| frame {camera_id} seq={seq}{age}")
    if not trace.frames_pushed:
        print(f"  {LABEL_INPUT}| (no new frames this cycle)")
    print(f"  {LABEL_INPUT}| detector_healthy    = {trace.detector_healthy}")
    for warning in trace.staging_warnings:
        print(f"  {LABEL_NOTE}| WARNING: {warning}")

    print(f"  {LABEL_CODE}| scheduler inferred  = {trace.inferred or 'nothing'}")
    print(
        f"  {LABEL_CODE}| skipped={trace.frames_skipped} "
        f"stale_frames={trace.stale_frames_dropped} "
        f"stale_results={trace.stale_results_dropped}"
    )

    if verbose or trace.observations:
        if trace.observations:
            print(f"  {LABEL_CODE}| ranged observations (ROBOT frame):")
            for observation in trace.observations:
                print(
                    f"  {LABEL_CODE}|   {observation['camera_id']} "
                    f"{observation['class_name']:<12} role={observation['role']:<10} "
                    f"conf={observation['confidence']:.2f} "
                    f"x={observation['x_robot_m']:+.3f} y={observation['y_robot_m']:+.3f} "
                    f"range={observation['range_m']:.3f} "
                    f"bearing={observation['bearing_deg']:+.1f}deg"
                )
        else:
            print(f"  {LABEL_CODE}| ranged observations = none")

    pose_state = "VALID" if trace.pose_valid else "INVALID"
    print(
        f"  {LABEL_CODE}| pose {pose_state:<7} ({trace.pose_source}) "
        f"x={trace.pose[0]:.3f} y={trace.pose[1]:.3f} "
        f"theta={math.degrees(trace.pose[2]):+.2f}deg"
    )
    if not trace.pose_valid:
        print(f"  {LABEL_CODE}| pose_invalid_reason = {trace.pose_invalid_reason}")

    if trace.fuel_tracks or verbose:
        print(f"  {LABEL_CODE}| fuel tracks (FIELD frame) = {len(trace.fuel_tracks)}")
        for track in trace.fuel_tracks:
            print(
                f"  {LABEL_CODE}|   id={track['id']} x={track['x']:.3f} y={track['y']:.3f} "
                f"v=({track['vx']:+.2f},{track['vy']:+.2f}) "
                f"conf={track['confidence']:.2f} hits={track['hits']}"
            )
    if trace.opponent_tracks or verbose:
        print(f"  {LABEL_CODE}| obstacle robots = {len(trace.opponent_tracks)}")
        for track in trace.opponent_tracks:
            print(
                f"  {LABEL_CODE}|   id={track['id']} {track['class_name']} "
                f"role={track['role']} x={track['x']:.3f} y={track['y']:.3f}"
            )

    if verbose and trace.pockets:
        print(f"  {LABEL_CODE}| navigation scoring (intermediate reasoning):")
        for pocket in trace.pockets:
            print(
                f"  {LABEL_CODE}|   pocket ({pocket['x']:.2f},{pocket['y']:.2f}) "
                f"count={pocket['count']} dist={pocket['distance_m']:.2f} "
                f"blocked={pocket['blocked']} clearance={pocket['clearance_m']} "
                f"score={pocket['score']:+.3f}"
            )

    print(
        f"  {LABEL_CODE}| DECISION vx={trace.nav_vx:+.4f} vy={trace.nav_vy:+.4f} "
        f"omega={trace.nav_omega:+.4f}  target={trace.nav_target}"
    )
    print(f"  {LABEL_CODE}| DECISION reason     = {trace.nav_reason}")
    print(
        f"  {LABEL_CODE}| controller enabled={trace.controller_enabled} "
        f"axes={tuple(round(a, 4) for a in trace.controller_axes)} "
        f"heartbeat={trace.controller_heartbeat}"
    )


def print_checks(results: List[CheckResult]) -> bool:
    print()
    all_passed = True
    for result in results:
        marker = "PASS" if result.passed else "FAIL"
        if not result.passed:
            all_passed = False
        print(f"  {marker}  {result.key:<38} {result.detail}")
    return all_passed


# ===========================================================================
# Built-in scenarios
# ===========================================================================
#
# Each one states its expected behavior declaratively. The robot sits at
# field (5, 3) facing +X unless stated, so a fuel "3 m ahead" is field
# (8, 3) and "1 m to the left" is field (5, 4).


def builtin_scenarios() -> List[ScenarioSpec]:
    return [
        # -- 0. fixture self-check ----------------------------------------
        ScenarioSpec(
            name="projection_round_trip",
            description="""
                Proves the harness's position->pixel fixture math agrees with
                the production ranging path. A fuel is staged 3.0 m ahead and
                0.5 m left; the real estimate_ground_object() must return the
                same robot-relative position. If this fails, every other
                scenario's placement is suspect.
            """,
            cameras=["cam0"],
            cycles=[CycleSpec(frames=[FrameSpec("cam0", 1, detections=[
                DetectionSpec("fuel", 0.95, x_robot_m=3.0, y_robot_m=0.5)])])],
            expect={
                "observations": 1,
                "min_fuel_tracks": 1,
                # robot at (5,3) facing +X, object 3.0 ahead / 0.5 left
                "fuel_track_near": [8.0, 3.5, 0.05],
                "pose_valid": True,
            },
        ),

        # -- 1. valid detections -------------------------------------------
        ScenarioSpec(
            name="valid_fuel_ahead",
            description="""
                A single confident fuel 3 m directly ahead. Expect: ranged,
                tracked in field coordinates at (8, 3), selected as the
                target, and a forward (+X) translation command.
            """,
            cameras=["cam0"],
            cycles=[CycleSpec(frames=[FrameSpec("cam0", 1, detections=[
                DetectionSpec("fuel", 0.92, x_robot_m=3.0, y_robot_m=0.0)])])],
            expect={
                "pose_valid": True,
                "observations": 1,
                "fuel_tracks": 1,
                "fuel_track_near": [8.0, 3.0, 0.05],
                "has_target": True,
                "target_near": [8.0, 3.0, 0.1],
                "translation_nonzero": True,
                "nav_vx_sign": 1,
                "nav_vy_sign": 0,
                "nav_reason_contains": "approach fuel",
                "contract_complete": True,
                "controller_enabled": False,
                "controller_axes_zero": True,
            },
        ),

        # -- 2. no detections ----------------------------------------------
        ScenarioSpec(
            name="no_detections",
            description="""
                Cameras are healthy and delivering frames, but nothing is in
                view. Pose is valid, so the robot may search. With no search
                point configured it must spin in place: zero translation,
                non-zero rotation.
            """,
            cameras=["cam0", "cam1"],
            config={"navigation.search_point_field": None},
            cycles=[CycleSpec(frames=[FrameSpec("cam0", 1), FrameSpec("cam1", 1)])],
            expect={
                "pose_valid": True,
                "observations": 0,
                "fuel_tracks": 0,
                "opponent_tracks": 0,
                "translation_zero": True,
                "omega_nonzero": True,
                "has_target": False,
                "nav_reason_contains": "no fuel known",
            },
        ),

        # -- 3. low-confidence detections ----------------------------------
        ScenarioSpec(
            name="low_confidence_rejected",
            description="""
                A fuel detected at 0.20 confidence, below the configured
                per-class gate of 0.40. The detector's own confidence filter
                must drop it, so no observation, no track, and no target.
            """,
            cameras=["cam0"],
            config={"navigation.search_point_field": None},
            cycles=[CycleSpec(frames=[FrameSpec("cam0", 1, detections=[
                DetectionSpec("fuel", 0.20, x_robot_m=3.0, y_robot_m=0.0)])])],
            expect={
                "pose_valid": True,
                "fuel_tracks": 0,
                "translation_zero": True,
                "has_target": False,
            },
        ),
        ScenarioSpec(
            name="low_confidence_boundary_accepted",
            description="""
                The same fuel at 0.45 confidence, just above the 0.40 gate.
                Must be accepted and tracked -- confirming the gate is a
                threshold and not a blanket rejection.
            """,
            cameras=["cam0"],
            cycles=[CycleSpec(frames=[FrameSpec("cam0", 1, detections=[
                DetectionSpec("fuel", 0.45, x_robot_m=3.0, y_robot_m=0.0)])])],
            expect={"fuel_tracks": 1, "has_target": True},
        ),

        # -- 4. stale frames -----------------------------------------------
        ScenarioSpec(
            name="stale_frame_refused",
            description="""
                A frozen camera: the frame was captured 600 ms ago, well past
                frame_freshness_s (250 ms). The scheduler must refuse it
                BEFORE inference -- a stale image would produce confident,
                current-looking, wrong detections. Expect zero inferences and
                one stale-frame drop.
            """,
            cameras=["cam0"],
            config={"navigation.search_point_field": None},
            cycles=[CycleSpec(frames=[FrameSpec(
                "cam0", 1, capture_offset_s=-0.600,
                detections=[DetectionSpec("fuel", 0.95, x_robot_m=3.0, y_robot_m=0.0)],
            )])],
            expect={
                "inferred_frames": [],
                "stale_frames_dropped": 1,
                "fuel_tracks": 0,
                "translation_zero": True,
            },
        ),
        ScenarioSpec(
            name="stale_result_dropped_after_inference",
            description="""
                The frame is fresh when selected, but max_detection_result_age_s
                is tightened to 1 ms so the finished result is already too old
                to describe the present. Inference DOES run, and the result is
                then discarded rather than injected into the world model.
            """,
            cameras=["cam0"],
            config={
                "max_detection_result_age_s": 0.001,
                "navigation.search_point_field": None,
            },
            cycles=[CycleSpec(frames=[FrameSpec("cam0", 1, capture_offset_s=-0.010,
                detections=[DetectionSpec("fuel", 0.95, x_robot_m=3.0, y_robot_m=0.0)],
            )])],
            expect={
                "inferred_frames": [["cam0", 1]],
                "stale_results_dropped": 1,
                "fuel_tracks": 0,
                "translation_zero": True,
            },
        ),

        # -- 5. out-of-order frames ----------------------------------------
        ScenarioSpec(
            name="out_of_order_frame_ignored",
            description="""
                Cycle 0 delivers seq 10. Cycle 1 delivers seq 7 -- an older
                frame arriving late. The scheduler tracks the last sequence
                per camera and must ignore anything at or below it, so cycle 1
                runs no inference.
            """,
            cameras=["cam0"],
            cycles=[
                CycleSpec(note="newest frame so far: seq 10",
                          frames=[FrameSpec("cam0", 10, detections=[
                              DetectionSpec("fuel", 0.95, x_robot_m=3.0, y_robot_m=0.0)])]),
                # The late frame claims a DIFFERENT position, so if it were
                # wrongly processed the track would move and the assertion
                # below would catch it.
                CycleSpec(note="stale sequence 7 arrives late",
                          frames=[FrameSpec("cam0", 7, detections=[
                              DetectionSpec("fuel", 0.95, x_robot_m=1.5, y_robot_m=0.5)])]),
            ],
            expect={
                "inferred_frames": [],
                "fuel_track_near": [8.0, 3.0, 0.10],
                "fuel_tracks": 1,
            },
        ),
        ScenarioSpec(
            name="skipped_frames_counted",
            description="""
                Cycle 0 infers seq 100. While the GPU was busy, seqs 101-103
                arrived; cycle 1 sees only the newest. The scheduler must
                process seq 103 and report 2 skipped frames, with no queue
                anywhere.
            """,
            cameras=["cam0"],
            cycles=[
                CycleSpec(frames=[FrameSpec("cam0", 100)]),
                CycleSpec(note="101 and 102 were superseded by 103",
                          frames=[FrameSpec("cam0", 101), FrameSpec("cam0", 102),
                                  FrameSpec("cam0", 103)]),
            ],
            expect={"inferred_frames": [["cam0", 103]], "frames_skipped": 2},
        ),

        # -- 6. detector failures ------------------------------------------
        ScenarioSpec(
            name="detector_failure_no_fabrication",
            description="""
                The detector goes unhealthy mid-scenario. It must produce NO
                detections -- never fabricated, never the frozen last result.
                The track established in cycle 0 then ages out, and navigation
                falls back to searching with zero translation.
            """,
            cameras=["cam0"],
            config={"navigation.search_point_field": None},
            cycles=[
                CycleSpec(note="healthy: establish a track",
                          frames=[FrameSpec("cam0", 1, detections=[
                              DetectionSpec("fuel", 0.95, x_robot_m=3.0, y_robot_m=0.0)])]),
                CycleSpec(note="detector dies", detector_healthy=False, advance_s=0.5,
                          frames=[FrameSpec("cam0", 2, detections=[
                              DetectionSpec("fuel", 0.95, x_robot_m=3.0, y_robot_m=0.0)])]),
                CycleSpec(note="tracks must age out, not freeze", advance_s=2.5,
                          frames=[FrameSpec("cam0", 3)]),
            ],
            expect={
                "detector_healthy": False,
                "observations": 0,
                "fuel_tracks": 0,
                "all_tracks": 0,
                "translation_zero": True,
                "pose_valid": True,
                "nav_reason_contains": "no fuel known",
            },
        ),

        # -- 7. conflicting results from two cameras -----------------------
        ScenarioSpec(
            name="two_cameras_same_object_deduped",
            description="""
                Both cameras see the SAME fuel. cam0 (facing forward) and cam1
                (yawed 30 deg left) each report it at the position that camera
                would see, with slightly different confidence. Cross-camera
                de-duplication must produce ONE field track, not two.
            """,
            cameras=["cam0", "cam1"],
            cycles=[CycleSpec(frames=[
                FrameSpec("cam0", 1, detections=[
                    DetectionSpec("fuel", 0.93, x_robot_m=2.50, y_robot_m=1.00)]),
                FrameSpec("cam1", 1, detections=[
                    DetectionSpec("fuel", 0.71, x_robot_m=2.58, y_robot_m=1.05)]),
            ])],
            expect={
                "min_observations": 2,
                "fuel_tracks": 1,
                "fuel_track_near": [7.5, 4.0, 0.2],
            },
        ),
        ScenarioSpec(
            name="two_cameras_different_objects_stay_separate",
            description="""
                The cameras see two genuinely different fuels, 3 m apart. They
                must remain two tracks -- de-duplication must not merge
                distinct objects.
            """,
            cameras=["cam0", "cam1"],
            cycles=[CycleSpec(frames=[
                # cam0 looks straight ahead; cam1 is yawed 30 deg left. These
                # two positions are each inside exactly one camera's view.
                FrameSpec("cam0", 1, detections=[
                    DetectionSpec("fuel", 0.90, x_robot_m=2.5, y_robot_m=0.0)]),
                FrameSpec("cam1", 1, detections=[
                    DetectionSpec("fuel", 0.90, x_robot_m=2.0, y_robot_m=2.0)]),
            ])],
            expect={
                "fuel_tracks": 2,
                "fuel_track_near": [7.5, 3.0, 0.1],
                # the two objects are 2.06 m apart, far beyond the merge radius
                "allow_unstaged_detections": True,
            },
        ),

        # -- 8. objects at different positions ------------------------------
        ScenarioSpec(
            name="object_positions_map_correctly",
            description="""
                Four fuels at known robot-relative positions, all in cam0's
                view. Verifies the robot->field transform for each quadrant:
                ahead-left, ahead-right, near, and far.
            """,
            cameras=["cam0"],
            cycles=[CycleSpec(frames=[FrameSpec("cam0", 1, detections=[
                DetectionSpec("fuel", 0.9, x_robot_m=2.0, y_robot_m=1.0),
                DetectionSpec("fuel", 0.9, x_robot_m=2.0, y_robot_m=-1.0),
                DetectionSpec("fuel", 0.9, x_robot_m=1.2, y_robot_m=0.0),
                DetectionSpec("fuel", 0.9, x_robot_m=4.0, y_robot_m=0.0),
            ])])],
            expect={
                "observations": 4,
                "fuel_tracks": 4,
                "fuel_track_near": [7.0, 4.0, 0.1],   # ahead-left  -> field +Y
                "min_fuel_tracks": 4,
            },
        ),
        ScenarioSpec(
            name="rotated_robot_transforms_correctly",
            description="""
                The robot is rotated 90 deg CCW at (5, 3), so its forward axis
                points along field +Y. A fuel 3 m straight ahead must land at
                field (5, 6), not (8, 3). This is the robot->field rotation
                under test.
            """,
            robot_pose=(5.0, 3.0, math.pi / 2),
            cameras=["cam0"],
            cycles=[CycleSpec(frames=[FrameSpec("cam0", 1, detections=[
                DetectionSpec("fuel", 0.95, x_robot_m=3.0, y_robot_m=0.0)])])],
            expect={
                "fuel_tracks": 1,
                "fuel_track_near": [5.0, 6.0, 0.05],
                "nav_vy_sign": 1,
                "nav_vx_sign": 0,
            },
        ),

        # -- 9. THE safety rule ---------------------------------------------
        ScenarioSpec(
            name="pose_invalid_never_translates",
            description="""
                THE most important test. Pose is invalid (tag initialization
                required, no tag ever seen) while a confident fuel sits 3 m
                ahead. Every field coordinate is meaningless without a pose,
                so translation MUST be exactly zero. Only an in-place search
                rotation is permitted.
            """,
            pose_mode="invalid",
            cameras=["cam0"],
            cycles=[CycleSpec(frames=[FrameSpec("cam0", 1, detections=[
                DetectionSpec("fuel", 0.99, x_robot_m=3.0, y_robot_m=0.0)])])],
            expect={
                "pose_valid": False,
                "translation_zero": True,
                "omega_nonzero": True,
                "has_target": False,
                "all_tracks": 0,
                "nav_reason_contains": "pose invalid",
                "controller_enabled": False,
                "controller_axes_zero": True,
            },
        ),
        ScenarioSpec(
            name="pose_invalid_fully_neutral_when_configured",
            description="""
                Same as above but with allow_search_rotation_when_pose_invalid
                turned off: output must be completely neutral, including
                rotation.
            """,
            pose_mode="invalid",
            cameras=["cam0"],
            config={"navigation.allow_search_rotation_when_pose_invalid": False},
            cycles=[CycleSpec(frames=[FrameSpec("cam0", 1, detections=[
                DetectionSpec("fuel", 0.99, x_robot_m=3.0, y_robot_m=0.0)])])],
            expect={
                "pose_valid": False,
                "translation_zero": True,
                "omega_zero": True,
            },
        ),
        ScenarioSpec(
            name="odometry_goes_stale_invalidates_pose",
            description="""
                The roboRIO stops republishing Odometry/timestamp. Freshness
                is decided by that value ADVANCING, not by us re-reading it,
                so the pose must go invalid and translation must stop -- even
                though a fuel is still in view.
            """,
            pose_mode="odometry_only",
            cameras=["cam0"],
            cycles=[
                CycleSpec(note="odometry alive",
                          frames=[FrameSpec("cam0", 1, detections=[
                              DetectionSpec("fuel", 0.95, x_robot_m=3.0, y_robot_m=0.0)])]),
                CycleSpec(note="RIO publisher dies", freeze_odometry=True, advance_s=1.0,
                          frames=[FrameSpec("cam0", 2, detections=[
                              DetectionSpec("fuel", 0.95, x_robot_m=3.0, y_robot_m=0.0)])]),
            ],
            expect={
                "pose_valid": False,
                "translation_zero": True,
                "all_tracks": 0,
                "nav_reason_contains": "pose invalid",
            },
        ),
        ScenarioSpec(
            name="tag_fix_makes_pose_valid",
            description="""
                Tag initialization is required. Cycle 0 has no tag, so the
                pose is invalid and nothing translates. Cycle 1 injects a real
                AprilTag fix through PoseEstimator.add_tag_estimate(), the
                pose becomes valid, and the fuel target is acted on.
            """,
            pose_mode="tag_initialized",
            cameras=["cam0"],
            cycles=[
                CycleSpec(note="no tag yet -- pose invalid",
                          frames=[FrameSpec("cam0", 1, detections=[
                              DetectionSpec("fuel", 0.95, x_robot_m=3.0, y_robot_m=0.0)])]),
                CycleSpec(note="AprilTag fix arrives",
                          inject_tag_pose=(5.0, 3.0, 0.0),
                          frames=[FrameSpec("cam0", 2, detections=[
                              DetectionSpec("fuel", 0.95, x_robot_m=3.0, y_robot_m=0.0)])]),
            ],
            expect={
                "pose_valid": True,
                "pose_source": "odometry_tag_corrected",
                "pose_near": [5.0, 3.0, 0.05],
                "fuel_tracks": 1,
                "translation_nonzero": True,
            },
        ),

        # -- 10. alliance routing -------------------------------------------
        ScenarioSpec(
            name="red_alliance_blue_bumper_is_opponent",
            description="""
                We are RED. A blue bumper 3 m ahead must become an OPPONENT
                obstacle track.
            """,
            is_red_alliance=True,
            cameras=["cam0"],
            config={"navigation.search_point_field": None},
            cycles=[CycleSpec(frames=[FrameSpec("cam0", 1, detections=[
                DetectionSpec("bumper_blue", 0.90, x_robot_m=3.0, y_robot_m=0.0)])])],
            expect={"opponent_tracks": 1, "fuel_tracks": 0},
        ),
        ScenarioSpec(
            name="red_alliance_red_bumper_is_friendly",
            description="""
                We are RED. A red bumper is a TEAMMATE and must never enter
                the obstacle list -- it is dropped at the alliance-routing
                stage, before ranging.
            """,
            is_red_alliance=True,
            cameras=["cam0"],
            config={"navigation.search_point_field": None},
            cycles=[CycleSpec(frames=[FrameSpec("cam0", 1, detections=[
                DetectionSpec("bumper_red", 0.95, x_robot_m=3.0, y_robot_m=0.0)])])],
            expect={"opponent_tracks": 0, "observations": 0, "all_tracks": 0},
        ),
        ScenarioSpec(
            name="unknown_alliance_treats_robots_as_obstacles",
            description="""
                No FMS data, so neither bumper color can be classified. The
                conservative rule applies: an unidentifiable robot is an
                obstacle, because refusing to drive through a robot we cannot
                identify is always safe.
            """,
            is_red_alliance=None,
            cameras=["cam0"],
            config={"navigation.search_point_field": None},
            cycles=[CycleSpec(frames=[FrameSpec("cam0", 1, detections=[
                DetectionSpec("bumper_blue", 0.90, x_robot_m=3.0, y_robot_m=0.5),
                DetectionSpec("bumper_red", 0.90, x_robot_m=3.0, y_robot_m=-1.5),
            ])])],
            expect={"opponent_tracks": 2, "fuel_tracks": 0},
        ),

        # -- 11. navigation decision quality --------------------------------
        ScenarioSpec(
            name="dense_cluster_beats_isolated_piece",
            description="""
                One isolated fuel 2 m ahead-right versus a three-piece cluster
                4 m ahead-left. Pieces are spaced beyond the 0.40 m merge
                radius so all four stay distinct. Pocket scoring weights
                density over distance, so the three-piece cluster must win.
            """,
            cameras=["cam0"],
            cycles=[CycleSpec(frames=[FrameSpec("cam0", 1, detections=[
                DetectionSpec("fuel", 0.9, x_robot_m=2.0, y_robot_m=-1.0),
                DetectionSpec("fuel", 0.9, x_robot_m=4.0, y_robot_m=0.0),
                DetectionSpec("fuel", 0.9, x_robot_m=4.5, y_robot_m=0.5),
                DetectionSpec("fuel", 0.9, x_robot_m=4.0, y_robot_m=0.9),
            ])])],
            expect={
                "observations": 4,
                "fuel_tracks": 4,
                "has_target": True,
                "target_pocket_count": 3,
                # cluster centroid: robot-relative (4.167, 0.467) -> field
                "target_near": [9.167, 3.467, 0.2],
                "nav_reason_contains": "3 piece",
            },
        ),
        ScenarioSpec(
            name="adjacent_objects_merge_within_radius",
            description="""
                DOCUMENTS CURRENT BEHAVIOR -- see the findings section of
                TESTING_DECISIONS.md. Three fuels from ONE camera, two of them
                0.32 m apart (inside merge_distance_game_piece_m = 0.40).
                merge_observations() merges them even though they came from the
                same camera, so three real pieces become two tracks and the
                pocket under-counts. This test pins the behavior rather than
                asserting it is correct; the production code is unchanged.
            """,
            cameras=["cam0"],
            cycles=[CycleSpec(frames=[FrameSpec("cam0", 1, detections=[
                DetectionSpec("fuel", 0.9, x_robot_m=4.0, y_robot_m=0.8),
                DetectionSpec("fuel", 0.9, x_robot_m=4.1, y_robot_m=0.5),
                DetectionSpec("fuel", 0.9, x_robot_m=4.3, y_robot_m=1.1),
            ])])],
            expect={
                "observations": 3,
                # Three detections, two tracks: the 0.32 m pair was merged.
                "fuel_tracks": 2,
                "target_pocket_count": 2,
            },
        ),
        ScenarioSpec(
            name="opponent_blocking_path_is_penalized",
            description="""
                A fuel 4 m ahead with an opponent sitting on the straight-line
                path, 0.35 m off it. The pocket must be flagged BLOCKED and
                heavily penalized. Note it is still selected -- it is the only
                pocket -- which is correct: the penalty decides between
                options, it does not forbid the only one.
            """,
            is_red_alliance=True,
            cameras=["cam0"],
            cycles=[CycleSpec(frames=[FrameSpec("cam0", 1, detections=[
                DetectionSpec("fuel", 0.9, x_robot_m=4.0, y_robot_m=0.0),
                DetectionSpec("bumper_blue", 0.9, x_robot_m=1.6, y_robot_m=0.35),
            ])])],
            expect={
                "fuel_tracks": 1,
                "opponent_tracks": 1,
                "target_pocket_blocked": True,
                "max_target_pocket_score": 0.0,
            },
        ),
        ScenarioSpec(
            name="opponent_near_robot_pushes_command_sideways",
            description="""
                The same geometry but the opponent is 1.27 m from the robot,
                inside the 1.45 m safe distance, so the repulsion term fires.
                The commanded direction must be pushed away from the opponent:
                it sits to the robot's LEFT, so the command gains a -Y
                component. This is the half of avoidance that depends on
                proximity to the ROBOT, not to the path.
            """,
            is_red_alliance=True,
            cameras=["cam0"],
            cycles=[CycleSpec(frames=[FrameSpec("cam0", 1, detections=[
                DetectionSpec("fuel", 0.9, x_robot_m=4.0, y_robot_m=0.0),
                DetectionSpec("bumper_blue", 0.9, x_robot_m=1.2, y_robot_m=0.4),
            ])])],
            expect={
                "fuel_tracks": 1,
                "opponent_tracks": 1,
                "nav_reason_contains": "avoidance",
                "nav_vy_sign": -1,
                "nav_vx_sign": 1,
            },
        ),

        # -- 12. contract + heartbeat ----------------------------------------
        ScenarioSpec(
            name="contract_and_heartbeat_hold",
            description="""
                Across several cycles the full Perception key contract must be
                published every time, and the virtual controller heartbeat
                must keep advancing even while disabled -- that is how the
                roboRIO tells "say nothing" from "the coprocessor is gone".
            """,
            cameras=["cam0", "cam1"],
            cycles=[
                CycleSpec(frames=[FrameSpec("cam0", 1), FrameSpec("cam1", 1)]),
                CycleSpec(frames=[FrameSpec("cam0", 2), FrameSpec("cam1", 2)]),
                CycleSpec(frames=[FrameSpec("cam0", 3), FrameSpec("cam1", 3)]),
            ],
            expect={
                "contract_complete": True,
                "heartbeat_advanced": True,
                "controller_enabled": False,
                "controller_buttons_none": True,
            },
        ),
    ]


def scenarios_by_name() -> Dict[str, ScenarioSpec]:
    return {spec.name: spec for spec in builtin_scenarios()}


# ===========================================================================
# Runner
# ===========================================================================


def run_scenario(spec: ScenarioSpec, verbose: bool = False, quiet: bool = False) -> bool:
    """Execute one scenario and return True if every assertion passed."""
    if not quiet:
        print_scenario_header(spec)

    harness = DecisionHarness(spec)
    try:
        traces = harness.run()
    except Exception as exc:  # noqa: BLE001 - a harness bug must be visible
        print(f"\n  FAIL  {'HARNESS ERROR':<38} {type(exc).__name__}: {exc}")
        if verbose:
            import traceback

            traceback.print_exc()
        return False

    if not quiet:
        for trace in traces:
            print_cycle(trace, verbose)

    results = safety_checks(traces) + check_expectations(spec, traces)
    if not spec.expect:
        results.append(CheckResult(
            "expectations", False,
            "scenario declares no expectations; a test that asserts nothing "
            "is not a test",
        ))

    if quiet:
        passed = all(r.passed for r in results)
        failures = [r for r in results if not r.passed]
        marker = "PASS" if passed else "FAIL"
        print(f"  {marker}  {spec.name}")
        for failure in failures:
            print(f"        {failure.key}: {failure.detail}")
        return passed

    passed = print_checks(results)
    print()
    print(f"  RESULT: {spec.name} -- {'PASS' if passed else 'FAIL'}")
    return passed


def run_all(
    specs: Sequence[ScenarioSpec], verbose: bool = False, quiet: bool = False
) -> int:
    """Run scenarios and return a process exit code."""
    print(SAFETY_BANNER)
    runnable = [s for s in specs if not s.requires_gpu]
    skipped = [s for s in specs if s.requires_gpu]

    results: List[Tuple[str, bool]] = []
    for spec in runnable:
        results.append((spec.name, run_scenario(spec, verbose, quiet)))

    passed = [name for name, ok in results if ok]
    failed = [name for name, ok in results if not ok]

    print()
    print("=" * 76)
    print(f"SUMMARY: {len(passed)} passed, {len(failed)} failed, "
          f"{len(skipped)} skipped (GPU-dependent)")
    print("=" * 76)
    for name in failed:
        print(f"  FAILED: {name}")
    for spec in skipped:
        print(f"  SKIPPED (needs GPU): {spec.name}  -- run with --gpu-check")
    print()
    print("  NOTE  | All of the above ran OFFLINE: synthetic frames, scripted")
    print("  NOTE  | detections, in-memory NetworkTables. No camera, GPU,")
    print("  NOTE  | roboRIO or motor was involved. See TESTING_DECISIONS.md")
    print("  NOTE  | for what still requires real hardware.")
    return 1 if failed else 0


# ===========================================================================
# Interactive mode
# ===========================================================================

INTERACTIVE_HELP = """
Interactive decision explorer. Build a scenario by answering prompts, then
see exactly what the real pipeline decides. Blank input accepts the default
shown in [brackets].

Commands at the object prompt:
    fuel  <x> <y> [conf]      stage a fuel at robot-relative (x, y) metres
    blue  <x> <y> [conf]      stage a blue bumper
    red   <x> <y> [conf]      stage a red bumper
    list                      show what is staged
    clear                     remove everything staged
    run                       execute the cycle and print the decision
    pose                      change pose mode / robot pose
    alliance                  change alliance
    detector                  toggle detector health
    save <file.json>          write the current scenario to JSON
    help                      this text
    quit                      exit

Robot-relative frame: +x is FORWARD, +y is LEFT, in metres.
"""


def _ask(prompt: str, default: str) -> str:
    try:
        answer = input(f"  {prompt} [{default}]: ").strip()
    except EOFError:
        return default
    return answer or default


def interactive() -> int:
    print(SAFETY_BANNER)
    print(INTERACTIVE_HELP)

    pose_mode = "odometry_only"
    robot_pose = (5.0, 3.0, 0.0)
    is_red: Optional[bool] = True
    detector_healthy = True
    staged: List[DetectionSpec] = []
    cameras = ["cam0", "cam1"]

    while True:
        try:
            raw = input("\ndecision> ").strip()
        except EOFError:
            print()
            return 0
        if not raw:
            continue
        parts = raw.split()
        command = parts[0].lower()

        if command in ("quit", "exit", "q"):
            return 0
        if command == "help":
            print(INTERACTIVE_HELP)
            continue
        if command == "clear":
            staged = []
            print("  staged objects cleared")
            continue
        if command == "list":
            if not staged:
                print("  nothing staged")
            for index, spec in enumerate(staged):
                print(f"    [{index}] {spec.class_name} at "
                      f"x={spec.x_robot_m} y={spec.y_robot_m} conf={spec.confidence}")
            continue
        if command == "alliance":
            answer = _ask("alliance (red/blue/unknown)",
                          "red" if is_red else "blue" if is_red is False else "unknown")
            is_red = {"red": True, "blue": False}.get(answer.lower())
            print(f"  alliance set to {answer}")
            continue
        if command == "detector":
            detector_healthy = not detector_healthy
            print(f"  detector_healthy = {detector_healthy}")
            continue
        if command == "pose":
            pose_mode = _ask("pose_mode (odometry_only/tag_initialized/invalid)", pose_mode)
            x = float(_ask("robot field x", str(robot_pose[0])))
            y = float(_ask("robot field y", str(robot_pose[1])))
            theta = float(_ask("robot heading deg", str(math.degrees(robot_pose[2]))))
            robot_pose = (x, y, math.radians(theta))
            print(f"  pose_mode={pose_mode} pose=({x}, {y}, {theta} deg)")
            continue
        if command in ("fuel", "blue", "red"):
            class_name = {"fuel": "fuel", "blue": "bumper_blue", "red": "bumper_red"}[command]
            try:
                x = float(parts[1])
                y = float(parts[2])
                confidence = float(parts[3]) if len(parts) > 3 else 0.90
            except (IndexError, ValueError):
                print(f"  usage: {command} <x_forward_m> <y_left_m> [confidence]")
                continue
            staged.append(DetectionSpec(class_name, confidence, x_robot_m=x, y_robot_m=y))
            print(f"  staged {class_name} at x={x} y={y} conf={confidence}")
            continue

        if command in ("run", "save"):
            spec = ScenarioSpec(
                name="interactive",
                description="Built interactively.",
                pose_mode=pose_mode,
                robot_pose=robot_pose,
                is_red_alliance=is_red,
                cameras=cameras,
                config={"navigation.search_point_field": None},
                cycles=[CycleSpec(
                    detector_healthy=detector_healthy,
                    frames=[FrameSpec("cam0", 1, detections=list(staged))],
                )],
                # Interactive runs are exploratory: only the safety
                # invariants are asserted, and they always are.
                expect={"controller_enabled": False, "controller_axes_zero": True},
            )
            if command == "save":
                if len(parts) < 2:
                    print("  usage: save <file.json>")
                    continue
                payload = {"scenarios": [_spec_to_dict(spec)]}
                with open(parts[1], "w", encoding="utf-8") as handle:
                    json.dump(payload, handle, indent=2)
                print(f"  wrote {parts[1]} -- edit its 'expect' block to make it a test")
                continue
            run_scenario(spec, verbose=True)
            continue

        print(f"  unknown command {command!r}; type 'help'")


def _spec_to_dict(spec: ScenarioSpec) -> Dict[str, Any]:
    return {
        "name": spec.name,
        "description": spec.description,
        "pose_mode": spec.pose_mode,
        "robot_pose": list(spec.robot_pose),
        "is_red_alliance": spec.is_red_alliance,
        "cameras": spec.cameras,
        "config": spec.config,
        "cycles": [
            {
                "advance_s": cycle.advance_s,
                "note": cycle.note,
                "detector_healthy": cycle.detector_healthy,
                "freeze_odometry": cycle.freeze_odometry,
                "inject_tag_pose": (
                    list(cycle.inject_tag_pose) if cycle.inject_tag_pose else None
                ),
                "frames": [
                    {
                        "camera_id": frame.camera_id,
                        "seq": frame.seq,
                        "capture_offset_s": frame.capture_offset_s,
                        "detections": [
                            {
                                "class_name": d.class_name,
                                "confidence": d.confidence,
                                "x_robot_m": d.x_robot_m,
                                "y_robot_m": d.y_robot_m,
                                "bbox": list(d.bbox) if d.bbox else None,
                            }
                            for d in frame.detections
                        ],
                    }
                    for frame in cycle.frames
                ],
            }
            for cycle in spec.cycles
        ],
        "expect": spec.expect,
    }


# ===========================================================================
# GPU-dependent check (run separately, on the Jetson)
# ===========================================================================


def gpu_check() -> int:
    """The ONLY test here that needs CUDA, a model file and real Ultralytics.

    Everything else in this file is deliberately GPU-free.  This loads the
    production :class:`YoloDetector` with the configured weights and runs a
    real inference on a synthetic image, reporting measured latency.
    """
    from frc_perception.config import load_config
    from frc_perception.detector import YoloDetector

    print(SAFETY_BANNER)
    print()
    print("GPU-DEPENDENT CHECK -- this one needs real hardware")
    print("=" * 76)

    cfg = load_config()
    print(f"  {LABEL_INPUT}| model_path   = {cfg.detector.model_path}")
    print(f"  {LABEL_INPUT}| device       = {cfg.detector.device}")
    print(f"  {LABEL_INPUT}| require_cuda = {cfg.detector.require_cuda}")

    failures = 0
    try:
        import torch

        available = torch.cuda.is_available()
        name = torch.cuda.get_device_name(0) if available else "none"
        print(f"  {LABEL_CODE}| torch {torch.__version__}, cuda={available}, device={name}")
        if not available:
            print(f"  FAIL  {'cuda.available':<38} torch reports no CUDA device")
            failures += 1
    except ImportError as exc:
        print(f"  FAIL  {'torch.import':<38} {exc}")
        return 1

    if not os.path.exists(cfg.detector.model_path):
        print(f"  FAIL  {'model.file':<38} not found: {cfg.detector.model_path}")
        print()
        print("  NOTE  | Train a model (training/) and copy best.pt into models/.")
        return 1

    detector = YoloDetector(cfg.detector)
    if not detector.load():
        print(f"  FAIL  {'model.load':<38} {detector.load_error}")
        return 1
    health = detector.health()
    print(f"  {LABEL_CODE}| loaded on {health.device_report}: {health.class_report}")
    if not health.classes_ok:
        print(f"  FAIL  {'model.classes':<38} {health.class_report}")
        failures += 1

    image = np.zeros((cfg.cameras[0].height, cfg.cameras[0].width, 3), dtype=np.uint8)
    latencies: List[float] = []
    for index in range(10):
        frame = Frame("gpu-check", index, time.monotonic(), image)
        started = time.monotonic()
        try:
            detections = detector.predict(frame)
        except Exception as exc:  # noqa: BLE001
            print(f"  FAIL  {'inference':<38} {exc}")
            return 1
        latencies.append((time.monotonic() - started) * 1000.0)
    detector.close()

    warm = sorted(latencies[1:]) or latencies
    print(f"  {LABEL_CODE}| 10 inferences, {len(detections)} detections on a blank frame")
    print(f"  {LABEL_CODE}| first={latencies[0]:.1f} ms  "
          f"median={warm[len(warm) // 2]:.1f} ms  max={warm[-1]:.1f} ms")
    print(f"  {LABEL_NOTE}| Measured on THIS machine just now. No frame rate is")
    print(f"  {LABEL_NOTE}| assumed anywhere in this project -- use")
    print(f"  {LABEL_NOTE}| scripts/benchmark_detector.py for a real benchmark.")
    print()
    print(f"  RESULT: gpu-check -- {'PASS' if not failures else 'FAIL'}")
    return 1 if failures else 0


# ===========================================================================
# CLI
# ===========================================================================


def load_json_scenarios(path: str) -> List[ScenarioSpec]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    raw = payload.get("scenarios", payload) if isinstance(payload, dict) else payload
    if not isinstance(raw, list):
        raise SystemExit(f"{path}: expected a list of scenarios or {{'scenarios': [...]}}")
    return [ScenarioSpec.from_dict(entry) for entry in raw]


EXAMPLE_JSON = {
    "scenarios": [
        {
            "name": "my_first_test",
            "description": "A fuel 2.5 m ahead and 1 m left should be chased.",
            "pose_mode": "odometry_only",
            "robot_pose": [5.0, 3.0, 0.0],
            "is_red_alliance": True,
            "cameras": ["cam0"],
            "config": {"navigation.search_point_field": None},
            "cycles": [
                {
                    "advance_s": 0.033,
                    "note": "one fuel in view",
                    "frames": [
                        {
                            "camera_id": "cam0",
                            "seq": 1,
                            "capture_offset_s": 0.0,
                            "detections": [
                                {
                                    "class_name": "fuel",
                                    "confidence": 0.9,
                                    "x_robot_m": 2.5,
                                    "y_robot_m": 1.0,
                                }
                            ],
                        }
                    ],
                }
            ],
            "expect": {
                "pose_valid": True,
                "fuel_tracks": 1,
                "fuel_track_near": [7.5, 4.0, 0.1],
                "translation_nonzero": True,
                "controller_enabled": False,
            },
        }
    ]
}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--scenario", default=None,
                        help="scenario name, or 'all'")
    parser.add_argument("--list", action="store_true", help="list scenarios and exit")
    parser.add_argument("--json", default="", help="run scenarios from a JSON file")
    parser.add_argument("--interactive", action="store_true",
                        help="enter values and inspect decisions")
    parser.add_argument("--write-example-json", default="",
                        help="write a starter JSON scenario file and exit")
    parser.add_argument("--gpu-check", action="store_true",
                        help="the only GPU/model-dependent test (run on the Jetson)")
    parser.add_argument("--help-expect", action="store_true",
                        help="list every supported expectation key and exit")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="show intermediate reasoning and package logs")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="one PASS/FAIL line per scenario")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="  LOG   | %(levelname)-7s %(name)s: %(message)s",
    )

    if args.help_expect:
        print("Supported expectation keys:\n")
        for key, description in EXPECTATION_KEYS.items():
            print(f"  {key:<28} {description}")
        return 0

    if args.write_example_json:
        with open(args.write_example_json, "w", encoding="utf-8") as handle:
            json.dump(EXAMPLE_JSON, handle, indent=2)
        print(f"wrote {args.write_example_json}")
        print("Run it with:  python test_decisions.py --json "
              f"{args.write_example_json}")
        return 0

    if args.gpu_check:
        return gpu_check()

    if args.interactive:
        return interactive()

    if args.json:
        specs = load_json_scenarios(args.json)
        if args.scenario and args.scenario != "all":
            specs = [s for s in specs if s.name == args.scenario]
            if not specs:
                raise SystemExit(f"no scenario named {args.scenario!r} in {args.json}")
        return run_all(specs, args.verbose, args.quiet)

    catalogue = scenarios_by_name()

    if args.list:
        print("Built-in scenarios:\n")
        for name, spec in catalogue.items():
            summary = " ".join(spec.description.split())[:90]
            print(f"  {name:<44} {summary}")
        print(f"\n  {len(catalogue)} scenarios. Run all:  "
              "python test_decisions.py --scenario all")
        print("  GPU-dependent check (separate):  python test_decisions.py --gpu-check")
        return 0

    if not args.scenario:
        parser.print_help()
        print("\nStart with:  python test_decisions.py --scenario all")
        return 0

    if args.scenario == "all":
        return run_all(list(catalogue.values()), args.verbose, args.quiet)

    if args.scenario not in catalogue:
        close = [n for n in catalogue if args.scenario in n]
        hint = f" Did you mean: {close}?" if close else ""
        raise SystemExit(
            f"no scenario named {args.scenario!r}.{hint} Use --list to see them all."
        )
    return run_all([catalogue[args.scenario]], args.verbose, args.quiet)


if __name__ == "__main__":
    sys.exit(main())
