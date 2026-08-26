"""Shared data types for every layer of the perception stack.

These are deliberately dumb containers: no I/O, no hardware, no numpy
requirement beyond plain floats.  Every layer boundary in the pipeline
(cameras -> detection -> geometry -> localization -> world model ->
navigation -> controller -> NetworkTables) passes one of these types, so
a layer can be unit tested by constructing its inputs by hand.

UNITS (everywhere in this package unless a name ends in `_deg`):
    distance  meters
    angle     radians, counter-clockwise positive
    time      seconds, from ``time.monotonic()`` unless named ``rio_*``
    speed     meters/second, radians/second

FRAMES:
    robot   +X forward, +Y left, +Z up, origin at the robot center on
            the floor.  ``theta`` is CCW-positive.
    field   WPILib blue-origin: origin at the blue-alliance right-hand
            corner, +X toward the red alliance wall, +Y left when
            standing at the blue wall looking down field, +Z up.
    camera  OpenCV: +X image right, +Y image down, +Z out of the lens.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class Alliance(Enum):
    """Which alliance WE are on, as reported by the FMS through the RIO."""

    RED = "red"
    BLUE = "blue"
    UNKNOWN = "unknown"


class ObjectRole(Enum):
    """What a detected class means to the rest of the stack.

    ``FRIENDLY`` exists so that friendly bumpers can be positively
    identified and then positively excluded from the obstacle list, which
    is safer than silently dropping them at the detector.
    """

    GAME_PIECE = "game_piece"
    OPPONENT = "opponent"
    FRIENDLY = "friendly"
    UNKNOWN = "unknown"


class LocalizationSource(Enum):
    """Where the currently published pose actually comes from."""

    NONE = "none"
    ODOMETRY_ONLY = "odometry_only"
    ODOMETRY_TAG_CORRECTED = "odometry_tag_corrected"
    TAG_ONLY = "tag_only"


class TrackFreshness(Enum):
    """How much a world-model track should be trusted right now.

    Navigation must be able to tell "I saw this 80 ms ago" from "I last
    saw this a second ago and I am extrapolating".
    """

    RECENT = "recent"
    AGING = "aging"
    EXPIRED = "expired"


class RangeRejection(Enum):
    """Why a range estimate was refused.  ``NONE`` means the range is good."""

    NONE = "none"
    ABOVE_HORIZON = "above_horizon"
    DEGENERATE_RAY = "degenerate_ray"
    BEHIND_CAMERA = "behind_camera"
    TOO_FAR = "too_far"
    TOO_NEAR = "too_near"
    OUTSIDE_USABLE_IMAGE = "outside_usable_image"
    WIDTH_DISAGREEMENT = "width_disagreement"
    BAD_INTRINSICS = "bad_intrinsics"


# ---------------------------------------------------------------------------
# Cameras and detections
# ---------------------------------------------------------------------------


@dataclass
class Frame:
    """One captured image plus everything needed to reason about its age."""

    camera_id: str
    seq: int
    #: ``time.monotonic()`` sampled immediately after the grab returned.
    timestamp: float
    #: HxWx3 BGR ``numpy.ndarray``.  Typed ``Any`` so this module stays
    #: importable without numpy.
    image: Any

    def age(self, now: float) -> float:
        return now - self.timestamp


@dataclass
class CameraHealth:
    """Snapshot of one capture thread, published for diagnostics."""

    name: str
    opened: bool = False
    healthy: bool = False
    fps: float = 0.0
    frames_captured: int = 0
    frames_dropped: int = 0
    reconnect_count: int = 0
    consecutive_failures: int = 0
    last_frame_age_s: float = float("inf")
    last_error: str = ""


@dataclass
class Detection:
    """One detected object box, in pixels.

    THIS IS THE HARDWARE BOUNDARY.  Every detector backend -- Hailo-8,
    PyTorch, or whatever accelerator comes next -- produces exactly this
    type, and nothing downstream of it knows or cares which one did.
    Geometry, the world model and navigation see only ``Detection``.

    ``camera_id`` and ``frame_sequence`` identify which frame this came
    from, and ``capture_timestamp`` is when that frame was GRABBED, not
    when inference finished.  Keeping the capture time attached is what
    lets the pipeline reject a result that became stale while it waited
    for the accelerator.
    """

    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    camera_id: str
    frame_sequence: int
    capture_timestamp: float

    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        """(x1, y1, x2, y2) in pixels, top-left origin."""
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def bottom_center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) * 0.5, self.y2)

    @property
    def pixel_width(self) -> float:
        return self.x2 - self.x1

    @property
    def pixel_height(self) -> float:
        return self.y2 - self.y1


@dataclass
class InferenceResult:
    """One camera frame's detections, with the full timing chain attached.

    The scheduler produces these; the application decides whether each is
    young enough to act on.  Every timestamp is ``time.monotonic()``.
    """

    camera_id: str
    frame_sequence: int
    capture_timestamp: float
    inference_start: float
    inference_end: float
    detections: List[Detection] = field(default_factory=list)
    #: Frames this camera produced and the scheduler skipped to get here.
    #: Skipping is correct -- an old frame is worthless on a moving robot --
    #: but the count is published so the rate is visible.
    skipped_frames: int = 0
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error

    @property
    def inference_latency_s(self) -> float:
        """Time the accelerator (plus pre/post-processing) actually took."""
        return self.inference_end - self.inference_start

    def result_age(self, now: float) -> float:
        """How long ago inference FINISHED."""
        return now - self.inference_end

    def frame_age(self, now: float) -> float:
        """End-to-end age: how old the underlying image is, right now.

        This is the number that matters for a moving robot. It includes
        capture, queueing, inference and everything since.
        """
        return now - self.capture_timestamp


@dataclass
class DetectorHealth:
    """What the detector backend is currently able to do.

    ``healthy`` False means NO new detections are being produced. It never
    means "quietly switched to something slower": on the competition
    machine a failed accelerator produces nothing, tracks age out, and
    navigation degrades safely.
    """

    backend: str = "none"
    loaded: bool = False
    healthy: bool = False
    classes_ok: bool = False
    class_report: str = "not loaded"
    model_path: str = ""
    device_report: str = ""
    last_error: str = ""
    consecutive_failures: int = 0
    inferences: int = 0
    mean_latency_ms: float = 0.0


@dataclass
class RangeResult:
    """Outcome of ranging one detection.  Always inspect ``valid`` first.

    An invalid result carries a ``rejection`` reason instead of a guessed
    position: the pipeline must never be handed a fabricated range.
    """

    valid: bool
    rejection: RangeRejection = RangeRejection.NONE
    x_robot_m: float = 0.0
    y_robot_m: float = 0.0
    range_m: float = 0.0
    bearing_rad: float = 0.0
    #: Raw camera-to-object ground distance before the calibration polynomial.
    raw_ground_distance_m: float = 0.0
    #: Independent pinhole-width distance, when the real width is known.
    width_distance_m: Optional[float] = None
    #: Relative disagreement between the two independent estimates.
    width_disagreement: Optional[float] = None
    #: 0..1 multiplier the world model applies to detection confidence.
    quality: float = 1.0


@dataclass
class RobotObservation:
    """A detection that has been successfully ranged, in the ROBOT frame."""

    camera_id: str
    timestamp: float
    class_name: str
    role: ObjectRole
    confidence: float
    x_robot_m: float
    y_robot_m: float
    range_m: float
    bearing_rad: float
    quality: float = 1.0


@dataclass
class FieldObservation:
    """A ranged detection transformed into WPILib field coordinates."""

    camera_id: str
    timestamp: float
    class_name: str
    role: ObjectRole
    confidence: float
    x_field_m: float
    y_field_m: float
    range_m: float
    quality: float = 1.0

    #: How many raw detections were merged into this observation by the
    #: cross-camera de-duplication stage (1 == seen by a single camera).
    merged_count: int = 1


# ---------------------------------------------------------------------------
# Poses and localization
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Pose2d:
    """A 2D pose.  Immutable so it can be shared between layers safely."""

    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0

    def is_finite(self) -> bool:
        return all(math.isfinite(v) for v in (self.x, self.y, self.theta))


@dataclass
class OdometryReading:
    """Raw odometry as published by the roboRIO, in blue-origin field coords.

    ``rio_timestamp`` is the RIO's own clock.  Freshness is decided by
    whether that number ADVANCES, never by when we happened to read it.
    """

    pose: Pose2d
    rio_timestamp: float
    #: ``time.monotonic()`` when ``rio_timestamp`` was last observed to change.
    local_time_of_change: float


@dataclass
class TagObservation:
    """One AprilTag seen by one camera, already solved to a robot pose."""

    camera_id: str
    timestamp: float
    tag_id: int
    #: Robot pose in field coordinates implied by this single tag.
    robot_pose: Pose2d
    #: Camera-to-tag distance in meters (the dominant error driver).
    distance_m: float
    decision_margin: float
    hamming: int
    #: pupil-apriltags ``pose_err``; lower is better.  None when unavailable.
    pose_error: Optional[float] = None
    #: Height above the floor the solve implies for the robot origin; a
    #: large value means the geometry is wrong.
    implied_height_m: float = 0.0
    #: Robot tilt implied by the solve, degrees.  Large means wrong.
    implied_tilt_deg: float = 0.0


@dataclass
class TagPoseEstimate:
    """Fused-from-tags robot pose for one processing cycle."""

    pose: Pose2d
    timestamp: float
    tag_ids: List[int] = field(default_factory=list)
    #: Max spread between individual single-tag solutions, meters.
    consistency_error_m: float = 0.0
    consistency_error_rad: float = 0.0
    mean_distance_m: float = 0.0
    #: 0..1, higher is better.  Combines tag count, distance and spread.
    quality: float = 0.0

    @property
    def tag_count(self) -> int:
        return len(self.tag_ids)


@dataclass
class PoseEstimate:
    """The pose the rest of the robot is allowed to act on.

    ``valid`` is the safety gate: False means field-coordinate navigation
    is NOT safe, regardless of whether ``pose`` contains numbers.
    """

    pose: Pose2d
    valid: bool
    source: LocalizationSource
    timestamp: float
    odometry_fresh: bool = False
    tag_pose_available: bool = False
    translation_disagreement_m: float = 0.0
    heading_disagreement_rad: float = 0.0
    tag_count: int = 0
    quality: float = 0.0
    invalid_reason: str = ""


# ---------------------------------------------------------------------------
# World model
# ---------------------------------------------------------------------------


@dataclass
class Track:
    """A persistent field-space object.  Owned by ``world_model.py``."""

    track_id: int
    class_name: str
    role: ObjectRole
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    confidence: float = 0.0
    #: ``time.monotonic()`` of the most recent associated observation.
    last_seen: float = 0.0
    #: Seconds since the track was created.
    age: float = 0.0
    hit_count: int = 0
    #: Consecutive update cycles with no associated observation.
    misses: int = 0
    velocity_valid: bool = False

    def staleness(self, now: float) -> float:
        return now - self.last_seen

    def predict(self, now: float, max_horizon_s: float) -> Tuple[float, float]:
        """Constant-velocity extrapolation, clamped to a sane horizon."""
        if not self.velocity_valid:
            return (self.x, self.y)
        dt = min(max(now - self.last_seen, 0.0), max_horizon_s)
        return (self.x + self.vx * dt, self.y + self.vy * dt)


# ---------------------------------------------------------------------------
# Navigation and control
# ---------------------------------------------------------------------------


@dataclass
class DesiredMotion:
    """High-level motion RECOMMENDATION.  Nothing here moves the robot.

    ``vx_mps``/``vy_mps`` are FIELD-frame velocities when
    ``field_relative`` is True (the normal case) and robot-frame when it
    is False.  ``virtual_controller.py`` converts to stick axes.
    """

    vx_mps: float = 0.0
    vy_mps: float = 0.0
    omega_radps: float = 0.0
    reason: str = "idle"
    target: Optional[Tuple[float, float]] = None
    confidence: float = 0.0
    field_relative: bool = True

    @staticmethod
    def stopped(reason: str) -> "DesiredMotion":
        return DesiredMotion(0.0, 0.0, 0.0, reason=reason, confidence=0.0)


@dataclass
class ControllerState:
    """Every axis and button of the virtual gamepad, published as a set.

    All fields are published every cycle so the roboRIO never has to
    infer a missing value.
    """

    enabled: bool = False
    heartbeat: int = 0

    left_x: float = 0.0
    left_y: float = 0.0
    right_x: float = 0.0
    right_y: float = 0.0

    left_trigger: float = 0.0
    right_trigger: float = 0.0

    a: bool = False
    b: bool = False
    x: bool = False
    y: bool = False

    left_bumper: bool = False
    right_bumper: bool = False

    back: bool = False
    start: bool = False

    left_stick: bool = False
    right_stick: bool = False

    #: WPILib POV convention: -1 == not pressed, otherwise degrees CW from up.
    pov: int = -1

    def as_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "heartbeat": self.heartbeat,
            "left_x": self.left_x,
            "left_y": self.left_y,
            "right_x": self.right_x,
            "right_y": self.right_y,
            "left_trigger": self.left_trigger,
            "right_trigger": self.right_trigger,
            "a": self.a,
            "b": self.b,
            "x": self.x,
            "y": self.y,
            "left_bumper": self.left_bumper,
            "right_bumper": self.right_bumper,
            "back": self.back,
            "start": self.start,
            "left_stick": self.left_stick,
            "right_stick": self.right_stick,
            "pov": self.pov,
        }


@dataclass
class MechanismCommand:
    """Neutral mechanism request produced by ``mechanism_policy``.

    Deliberately abstract: the mapping from these intents onto real
    buttons is robot specific and is NOT invented here.
    """

    intake: bool = False
    outtake: bool = False
    shoot: bool = False
    prepare_shot: bool = False
    note: str = ""


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


@dataclass
class LoopStats:
    """Rolling performance counters published for debugging.

    Every number here is MEASURED on the machine that is running. No
    expected frame rate is assumed or hard-coded anywhere in this project.
    """

    frame_counter: int = 0
    loop_fps: float = 0.0
    #: End-to-end age of the oldest frame that contributed to this cycle.
    perception_latency_ms: float = 0.0
    #: Wall time of the whole inference stage, all cameras.
    inference_ms: float = 0.0
    #: Accelerator latency of the most recent single inference.
    detector_latency_ms: float = 0.0
    preprocess_ms: float = 0.0
    postprocess_ms: float = 0.0
    apriltag_ms: float = 0.0
    publish_ms: float = 0.0
    #: Frames refused by the freshness check before reaching the detector.
    stale_frames_dropped: int = 0
    #: Results that finished too late to be worth using.
    stale_results_dropped: int = 0
    #: Frames the scheduler skipped in favor of a newer one from the same
    #: camera. Not an error -- this is the latest-frame policy working.
    frames_skipped: int = 0
