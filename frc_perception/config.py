"""Single source of truth for every tunable constant in the system.

Target platform: Raspberry Pi 5 (8 GB) + Raspberry Pi AI HAT+ (Hailo-8,
26 TOPS), two Arducam OV9782 USB cameras, roboRIO over NetworkTables 4.

Nothing else in this package hard-codes a threshold, timeout, gain, gain
weight, or physical dimension.  If you are tuning at a competition, tune
here and nowhere else.

Units follow ``frc_perception/types.py``: meters, seconds, radians --
except for fields whose name ends in ``_deg``, which humans type in
degrees and the code converts.

MUST INPUT
----------
Any value that cannot be known without the physical robot, a calibration
run, a trained model, or the official FRC 2026 game manual is marked with
a ``# MUST INPUT:`` comment saying exactly what is needed and how to get
it.  ``python run.py --check`` scans this file for those markers and
reports the ones still holding placeholder values.  Never guess them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ===========================================================================
# Cameras
# ===========================================================================


@dataclass
class CameraConfig:
    """One physical camera: capture settings, lens model, mounting pose.

    The mount pose describes where the camera sits in the ROBOT frame
    (+X forward, +Y left, +Z up, origin on the floor at the robot center).

    Mount rotation is applied yaw -> pitch -> roll (extrinsic Z, then Y,
    then X) to a camera whose optical axis starts pointing straight
    forward:

        R_robot_from_camera = Rz(yaw) @ Ry(pitch) @ Rx(roll) @ CV_TO_FORWARD

    ``mount_pitch_deg`` is POSITIVE when the camera is tilted DOWN toward
    the floor.  ``mount_yaw_deg`` is POSITIVE counter-clockwise (to the
    robot's left).  ``mount_roll_deg`` is rotation about the optical axis.
    """

    name: str
    # MUST INPUT: a PERSISTENT device path for this camera. On Raspberry Pi
    # OS, `ls -l /dev/v4l/by-id/` gives paths that survive reboots and
    # re-plugging; `/dev/videoN` numbers do not, and two identical OV9782
    # modules will happily swap places between boots. Each USB camera also
    # claims two /dev/video nodes (the second is metadata), so pick the one
    # `v4l2-ctl -d <path> --list-formats-ext` shows real formats for.
    #
    # If the two cameras are the same model with the same USB serial, by-id
    # will not distinguish them -- use /dev/v4l/by-path/ instead, which is
    # keyed on the physical USB port, and always plug each camera into the
    # same port.
    device_path: str = ""
    #: Fallback only, used when device_path is empty. Do NOT ship with this.
    device_index: int = 0

    # OV9782 native sensor format is 1280x800 global shutter.
    width: int = 1280
    height: int = 800
    fps: int = 60
    #: OV9782 needs MJPG to reach full resolution/rate over USB.
    fourcc: str = "MJPG"
    #: Fixed exposure keeps YOLO input consistent and kills motion blur.
    #: None = leave the driver on auto. Values are driver units, not seconds.
    manual_exposure: Optional[int] = None  # MUST INPUT (recommended): pick with scripts/camera_preview.py under real field lighting
    manual_gain: Optional[int] = None      # MUST INPUT (recommended): pick with scripts/camera_preview.py under real field lighting

    # -- Intrinsics, in pixels, from scripts/calibrate_intrinsics.py --------
    # Left as None until calibrated. Uncalibrated cameras fall back to an
    # ideal pinhole derived from horizontal_fov_deg, which is good enough
    # for bring-up but NOT good enough for AprilTag pose solving.
    fx: Optional[float] = None  # MUST INPUT: run scripts/calibrate_intrinsics.py
    fy: Optional[float] = None  # MUST INPUT: run scripts/calibrate_intrinsics.py
    cx: Optional[float] = None  # MUST INPUT: run scripts/calibrate_intrinsics.py
    cy: Optional[float] = None  # MUST INPUT: run scripts/calibrate_intrinsics.py
    #: OpenCV (k1, k2, p1, p2, k3). All zeros == ideal lens.
    dist_coeffs: Tuple[float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0)  # MUST INPUT: run scripts/calibrate_intrinsics.py
    #: Reprojection error reported by the calibration, for the record.
    calibration_rms_px: Optional[float] = None

    # MUST INPUT: horizontal field of view of the lens actually fitted to
    # your OV9782 module. Arducam sells several variants; read the product
    # page or measure it. Only used as a fallback before calibration.
    horizontal_fov_deg: float = 70.0

    # -- Mounting pose, measured on the real robot -------------------------
    mount_x_m: float = 0.0     # MUST INPUT: lens center distance FORWARD of robot center, meters
    mount_y_m: float = 0.0     # MUST INPUT: lens center distance LEFT of robot center, meters
    mount_z_m: float = 0.0     # MUST INPUT: lens center height ABOVE THE FLOOR, meters (dominant range-error term)
    pitch_deg: float = 0.0     # MUST INPUT: downward tilt of the optical axis, degrees, DOWN is positive
    yaw_deg: float = 0.0       # MUST INPUT: pan of the optical axis, degrees, CCW/left is positive
    roll_deg: float = 0.0      # MUST INPUT: rotation about the optical axis, degrees (usually 0)

    # -- Empirical range correction ---------------------------------------
    #: Polynomial in numpy.polyval order (highest power first) mapping the
    #: raw geometric ground distance to the measured true distance.
    #: (1.0, 0.0) is the identity. Produced by scripts/calibrate_distance.py.
    distance_poly: Tuple[float, ...] = (1.0, 0.0)  # MUST INPUT: run scripts/calibrate_distance.py once the camera is mounted
    #: Range beyond which this camera's polynomial was never fitted; the
    #: correction is clamped there instead of extrapolating wildly.
    distance_poly_max_m: float = 8.0

    # -- Usable image region ----------------------------------------------
    #: Detections whose bottom edge falls outside this fraction of the image
    #: height are rejected: objects cut off by the frame edge have no valid
    #: ground contact point. 0.0 == top of image, 1.0 == bottom.
    usable_bottom_min_frac: float = 0.05
    usable_bottom_max_frac: float = 0.995

    #: Set False to keep a camera configured but out of the running system.
    enabled: bool = True
    #: Run the AprilTag detector on this camera's frames.
    use_for_apriltags: bool = True
    #: Run the object detector on this camera's frames.
    use_for_objects: bool = True

    def effective_intrinsics(self) -> Tuple[float, float, float, float]:
        """Return (fx, fy, cx, cy), falling back to an FOV-derived pinhole."""
        if self.has_calibrated_intrinsics:
            assert self.fx is not None and self.fy is not None
            cx = self.cx if self.cx is not None else self.width / 2.0
            cy = self.cy if self.cy is not None else self.height / 2.0
            return (float(self.fx), float(self.fy), float(cx), float(cy))
        focal = (self.width / 2.0) / math.tan(math.radians(self.horizontal_fov_deg) / 2.0)
        return (focal, focal, self.width / 2.0, self.height / 2.0)

    @property
    def has_calibrated_intrinsics(self) -> bool:
        return bool(self.fx and self.fy and self.cx is not None and self.cy is not None)

    @property
    def source(self) -> object:
        """What to hand to ``cv2.VideoCapture``: persistent path if set."""
        return self.device_path if self.device_path else self.device_index

    @property
    def uses_persistent_path(self) -> bool:
        return bool(self.device_path)


# ===========================================================================
# Object detection (YOLO)
# ===========================================================================

#: The exact class names the model is trained with. Do not rename: the
#: training data.yaml, the detector validation, and the role routing below
#: all key off these strings.
CLASS_FUEL = "fuel"
CLASS_BUMPER_BLUE = "bumper_blue"
CLASS_BUMPER_RED = "bumper_red"

EXPECTED_CLASS_NAMES: Tuple[str, ...] = (CLASS_FUEL, CLASS_BUMPER_BLUE, CLASS_BUMPER_RED)

#: Coprocessors this project knows how to preflight.
KNOWN_PLATFORMS: Tuple[str, ...] = ("raspberry_pi_5", "generic_linux")

#: Detector backends implemented in detector.py.
KNOWN_DETECTOR_BACKENDS: Tuple[str, ...] = ("hailo", "pytorch")

#: Architectures the Ultralytics Hailo exporter accepts. The Raspberry Pi
#: AI HAT+ 26 TOPS carries a Hailo-8; the 13 TOPS board carries a Hailo-8L.
HAILO_ARCHITECTURES: Tuple[str, ...] = (
    "hailo8", "hailo8l", "hailo10h", "hailo15h", "hailo15l",
)


@dataclass
class DetectorConfig:
    """Object detection: which backend, which artifact, which thresholds.

    ``backend`` is the ONLY place inference hardware is chosen. There is no
    automatic fallback: on the robot, a Hailo failure means no detections,
    not a CPU model quietly eating the cores everything else needs.
    """

    #: "hailo"   -> Hailo-8 on the AI HAT+ (production)
    #: "pytorch" -> Ultralytics .pt (DEVELOPMENT AND TESTING ONLY)
    backend: str = "hailo"

    # -- Hailo artifact ----------------------------------------------------
    #: The Ultralytics Hailo export DIRECTORY, not just the .hef. The
    #: metadata.yaml beside the .hef carries the class names, and
    #: class-name routing has nothing to route on without it.
    # MUST INPUT: build this on an x86_64 Linux machine with the Hailo
    # Dataflow Compiler (see training/export_hailo.py) and copy the whole
    # directory here. It cannot be compiled on the Pi.
    hailo_model_path: str = "models/best_hailo_model"
    #: Accelerator this artifact was compiled for. The Raspberry Pi AI HAT+
    #: 26 TOPS carries a Hailo-8; the 13 TOPS version carries a Hailo-8L.
    #: A binary built for one does not run on the other.
    hailo_arch: str = "hailo8"

    # -- PyTorch artifact (development only) -------------------------------
    pytorch_model_path: str = "models/best.pt"
    #: Torch device for the DEVELOPMENT backend. Never used in production.
    pytorch_device: str = "auto"

    class_names: Tuple[str, ...] = EXPECTED_CLASS_NAMES

    conf_threshold: float = 0.40
    iou_threshold: float = 0.50
    #: Inference letterbox size. Must match the imgsz the HEF was compiled
    #: at -- the Hailo input shape is fixed at compile time.
    imgsz: int = 640
    #: Hard cap on boxes per frame; protects the downstream stages from a
    #: pathological frame.
    max_detections: int = 60
    #: Consecutive inference failures before the backend reports unhealthy.
    failures_before_unhealthy: int = 5

    #: Per-class minimum confidence, applied on top of conf_threshold. Robot
    #: bumpers are usually easier than small distant game pieces.
    #:
    #: NOTE for Hailo detection exports: on-chip NMS bakes the EXPORT-time
    #: confidence into nms_config.json, so the accelerator never returns a
    #: box below it. These runtime values can only filter further. Export
    #: low (see training/export_hailo.py) and tune here.
    per_class_conf: Dict[str, float] = field(
        default_factory=lambda: {
            CLASS_FUEL: 0.40,
            CLASS_BUMPER_BLUE: 0.50,
            CLASS_BUMPER_RED: 0.50,
        }
    )

    # -- Physical sizes used by the pinhole-width range CROSS-CHECK --------
    # These are cross-checks only; the primary range is ray-to-ground.
    # MUST INPUT: FUEL outside diameter in meters, from the official FRC 2026
    # game manual. There is no safe default -- leave None to disable the
    # width cross-check for fuel rather than inventing a number.
    fuel_diameter_m: Optional[float] = None
    # MUST INPUT: nominal width of an FRC robot's bumper face as it appears
    # to the camera, meters. Measure a real robot; bumper perimeter rules are
    # in the 2026 manual. Leave None to disable the cross-check for robots.
    bumper_width_m: Optional[float] = None

    @property
    def model_path(self) -> str:
        """The artifact the selected backend will actually load."""
        return (
            self.hailo_model_path
            if self.backend == "hailo"
            else self.pytorch_model_path
        )


#: Class name -> semantic role BEFORE alliance is applied. Bumper classes
#: resolve to OPPONENT/FRIENDLY in ``alliance.py``; routing is by NAME, never
#: by model class index.
CLASS_TO_KIND: Dict[str, str] = {
    CLASS_FUEL: "game_piece",
    CLASS_BUMPER_BLUE: "bumper_blue",
    CLASS_BUMPER_RED: "bumper_red",
}


# ===========================================================================
# Ranging / geometry
# ===========================================================================


@dataclass
class RangingConfig:
    #: Objects nearer/farther than these are treated as bad geometry.
    min_range_m: float = 0.25
    max_range_m: float = 8.0
    #: The ray must descend by at least this much (unitless z-component of
    #: the unit ray) before we accept a ground intersection. Guards the
    #: near-horizon singularity where range error explodes.
    min_ray_descent: float = 0.02
    #: Relative disagreement |width - geometric| / geometric above which the
    #: measurement is rejected outright.
    width_reject_ratio: float = 0.60
    #: Disagreement above which confidence is degraded but kept.
    width_warn_ratio: float = 0.30
    #: Quality multiplier applied when the width cross-check merely warns.
    width_warn_quality: float = 0.5
    #: Object center height above the floor, used as the ground plane for
    #: ray intersection. 0.0 == the box bottom touches the floor, which is
    #: the correct model for a game piece resting on the carpet.
    ground_plane_z_m: float = 0.0


# ===========================================================================
# Localization
# ===========================================================================


@dataclass
class AprilTagConfig:
    #: Official WPILib field layout JSON. Fetch with
    #: scripts/download_field_layout.py -- never hand-write this file.
    field_layout_path: str = "data/field_layout.json"  # MUST INPUT: the official FRC 2026 AprilTag layout JSON from WPILib
    tag_family: str = "tag36h11"
    # MUST INPUT: AprilTag black-square edge length in meters for FRC 2026,
    # from the official game manual / field drawings. 2024-2025 used 0.1651 m
    # (6.5 in) but that MUST be confirmed for 2026 before it is trusted.
    tag_size_m: Optional[float] = None

    # pupil-apriltags detector knobs.
    nthreads: int = 2
    quad_decimate: float = 2.0
    quad_sigma: float = 0.0
    refine_edges: int = 1
    decode_sharpening: float = 0.25

    # -- Per-tag acceptance filters ---------------------------------------
    max_tag_distance_m: float = 5.0
    min_decision_margin: float = 30.0
    max_hamming: int = 0
    #: pupil-apriltags pose_err above this is rejected (None disables).
    max_pose_error: Optional[float] = None
    #: A solve that puts the robot origin this far off the floor is wrong.
    max_implied_height_m: float = 0.35
    #: A solve that says the robot is tipped this far is wrong.
    max_implied_tilt_deg: float = 12.0
    #: Reject a solve that lands outside the field by more than this.
    field_bounds_margin_m: float = 1.0
    #: With several tags visible, discard single-tag solutions further than
    #: this from the median solution before averaging.
    multi_tag_outlier_m: float = 0.75
    #: Run the tag detector every Nth main-loop iteration (1 == every frame).
    detect_every_n_frames: int = 1


@dataclass
class LocalizationConfig:
    # -- NetworkTables key names (see nt_io.py) ----------------------------
    odometry_table: str = "Odometry"
    key_robot_x: str = "robot_x"
    key_robot_y: str = "robot_y"
    key_robot_theta: str = "robot_theta"
    key_timestamp: str = "timestamp"

    #: The RIO's odometry timestamp must ADVANCE within this many seconds or
    #: odometry is stale. Read repetition never counts as freshness.
    odometry_stale_timeout_s: float = 0.3
    #: How long a tag fix keeps being usable after it was taken.
    tag_stale_timeout_s: float = 1.0
    #: With odometry dead, a recent tag pose alone is trusted this long.
    tag_only_valid_s: float = 0.75

    # -- Correction strength ----------------------------------------------
    #: Blend weight of each accepted tag fix into the odometry correction.
    #: 0.0 ignores tags entirely; 1.0 snaps to every tag fix. This is an
    #: exponential complementary correction -- NOT a Kalman filter. See
    #: localization.py for the honest description of what it does.
    correction_alpha_translation: float = 0.15
    correction_alpha_heading: float = 0.10
    #: Scale the blend weight by the tag estimate quality (0..1).
    scale_alpha_by_quality: bool = True
    #: Never move the fused pose more than this per correction step. Stops a
    #: single bad tag solve from teleporting the robot.
    max_correction_step_m: float = 0.35
    max_correction_step_rad: float = math.radians(5.0)

    # -- Disagreement handling --------------------------------------------
    #: Log a warning above these.
    disagreement_warn_m: float = 0.75
    disagreement_warn_rad: float = math.radians(10.0)
    #: Above these, the tag fix is not blended at all (it is assumed wrong,
    #: or the odometry is wrong -- either way, do not average garbage).
    disagreement_reject_m: float = 2.0
    disagreement_reject_rad: float = math.radians(45.0)
    #: If disagreement stays above the reject threshold this long while tags
    #: are consistently visible, odometry is presumed broken and the pose is
    #: declared INVALID (we do not silently snap).
    disagreement_invalidate_after_s: float = 2.0

    # -- Validity ----------------------------------------------------------
    #: Require at least one accepted tag fix before the pose is ever valid.
    #: Leave True: without it, the fused pose is only as good as whatever
    #: the RIO happened to be initialized to.
    require_tag_initialization: bool = True
    #: Sanity bounds; a pose outside the field plus this margin is invalid.
    field_bounds_margin_m: float = 1.0

    apriltag: AprilTagConfig = field(default_factory=AprilTagConfig)

    # Fallback field size used ONLY for bounds checks when the layout JSON is
    # absent. MUST INPUT if you run without the layout: official 2026 field
    # dimensions. Tag localization is disabled entirely without the layout.
    fallback_field_length_m: Optional[float] = None
    fallback_field_width_m: Optional[float] = None


# ===========================================================================
# World model
# ===========================================================================


@dataclass
class WorldModelConfig:
    #: Two detections of the same class from DIFFERENT cameras within this
    #: distance are treated as one object (cross-camera de-duplication).
    merge_distance_game_piece_m: float = 0.40
    merge_distance_robot_m: float = 0.90

    #: Gate for associating an observation with an existing track.
    gate_game_piece_m: float = 0.60
    gate_robot_m: float = 1.20

    #: Position smoothing weight applied to each new measurement.
    position_alpha: float = 0.5

    # -- Velocity ----------------------------------------------------------
    #: Velocity is fitted by least squares over RAW measurements inside this
    #: window -- never by differencing the smoothed position, which lags and
    #: biases the estimate low.
    velocity_window_s: float = 0.60
    velocity_min_samples: int = 3
    velocity_min_span_s: float = 0.15
    max_speed_game_piece_mps: float = 6.0
    max_speed_robot_mps: float = 6.0

    # -- Aging -------------------------------------------------------------
    #: An observation must be seen this many times before it is reported.
    min_hits_to_confirm: int = 2
    #: Track is "recent" below this staleness, "aging" above it.
    fresh_timeout_game_piece_s: float = 0.5
    fresh_timeout_robot_s: float = 0.5
    #: Track is deleted above this staleness.
    expire_timeout_game_piece_s: float = 2.0
    expire_timeout_robot_s: float = 1.5
    #: Confidence decays linearly to zero across the expire timeout.
    #: Tracks below this confidence are not reported to navigation.
    min_report_confidence: float = 0.15
    #: Reject observations projected this far outside the field.
    field_margin_m: float = 0.5
    #: Cap on how far a stale track is extrapolated for collision checks.
    max_prediction_horizon_s: float = 1.0
    #: When the alliance is unknown or stale, a detected robot cannot be
    #: classified. True (the conservative default) treats it as an obstacle;
    #: it is then reported in the opponent list, because "a robot we cannot
    #: identify" is exactly what we must not drive through. Setting this
    #: False makes the robot ignore unidentifiable robots -- do not.
    treat_unknown_robots_as_obstacles: bool = True


# ===========================================================================
# Navigation
# ===========================================================================


@dataclass
class NavigationConfig:
    # -- Fuel target selection --------------------------------------------
    #: Fuel tracks within this radius of each other form one "pocket".
    cluster_radius_m: float = 1.00
    #: Score weights. Higher total score wins.
    weight_density: float = 1.0        # per additional piece in the pocket
    weight_distance: float = 0.35      # subtracted per meter to the pocket
    weight_heading: float = 0.15       # subtracted per radian of turn needed
    weight_freshness: float = 0.60     # multiplied by mean track confidence
    weight_obstacle_risk: float = 2.5  # subtracted per unit of blocking risk
    #: A pocket whose straight-line path is blocked loses this much score.
    blocked_path_penalty: float = 4.0
    #: Ignore fuel further away than this.
    max_target_distance_m: float = 6.0
    #: Never target a point closer than this to a field wall.
    wall_margin_m: float = 0.50

    # -- Approach ----------------------------------------------------------
    max_speed_mps: float = 2.0           # MUST INPUT: your drivetrain's safe assisted-driving speed cap, m/s
    max_omega_radps: float = 3.0         # MUST INPUT: your drivetrain's safe assisted-driving turn-rate cap, rad/s
    approach_kp: float = 1.6             # m/s per meter of position error
    heading_kp: float = 2.5              # rad/s per radian of heading error
    slow_radius_m: float = 1.20          # begin decelerating inside this radius
    arrive_tolerance_m: float = 0.30

    # -- Opponent avoidance ------------------------------------------------
    opponent_radius_m: float = 0.55      # MUST INPUT: half the diagonal of a bumpered FRC robot, meters (measure one)
    own_robot_radius_m: float = 0.55     # MUST INPUT: half the diagonal of YOUR bumpered robot, meters
    safety_margin_m: float = 0.35
    #: Opponent tracks are extrapolated this far ahead when checking paths.
    prediction_horizon_s: float = 0.75
    #: Strength of the sideways push away from a threatening opponent.
    avoid_gain: float = 1.5
    #: Opponents further than this are ignored for avoidance.
    avoid_consider_radius_m: float = 3.0

    # -- Search behavior ---------------------------------------------------
    #: Rotation used while searching. This is the ONLY motion permitted when
    #: the pose is invalid (see rule below) because spinning in place needs
    #: no field coordinates.
    search_rotation_radps: float = 1.0
    #: When the pose is INVALID the robot must never translate. Set False to
    #: also suppress the search rotation (fully neutral output).
    allow_search_rotation_when_pose_invalid: bool = True
    #: When the pose is VALID but no fuel is known, drive to this point.
    #: MUST INPUT: a sensible staging point in official 2026 field
    #: coordinates -- typically field center. Leave None and the robot will
    #: spin-search in place instead of driving anywhere.
    search_point_field: Optional[Tuple[float, float]] = None
    #: Also spin while driving to the search point, to sweep the cameras.
    rotate_while_searching: bool = True


# ===========================================================================
# Virtual controller
# ===========================================================================


@dataclass
class ControllerConfig:
    #: SAFETY DEFAULT. While False the adapter publishes a fully neutral
    #: gamepad and enabled=false. The roboRIO must ALSO gate on its own
    #: deadman; the coprocessor is never the last line of defense.
    enabled: bool = False

    #: True when your teleop drive code interprets the sticks as
    #: field-relative. False means robot-relative sticks, and the adapter
    #: rotates the field-frame command by -heading before mapping.
    field_oriented_drive: bool = True  # MUST INPUT: match your robot's teleop binding

    # MUST INPUT: verify all three of these with the robot ON BLOCKS.
    # Joystick polarity differs between robot codebases and there is no
    # convention safe enough to assume. WPILib's examples negate the Y axis
    # of a pushed-forward stick, hence the defaults -- confirm anyway.
    invert_left_x: bool = True   # left stick X drives strafe
    invert_left_y: bool = True   # left stick Y drives forward/back
    invert_right_x: bool = True  # right stick X drives rotation

    #: Command magnitude that maps to full stick deflection.
    full_scale_speed_mps: float = 2.0    # MUST INPUT: the speed your teleop maps to a fully deflected stick
    full_scale_omega_radps: float = 3.0  # MUST INPUT: the turn rate your teleop maps to a fully deflected stick

    #: Axis magnitudes below this publish as exactly 0.0.
    deadband: float = 0.03
    #: Hard clamp on every axis after scaling.
    output_limit: float = 0.85
    #: Maximum axis change per second (jerk limiting). None disables.
    slew_rate_per_s: Optional[float] = 3.0

    #: NetworkTables table the gamepad is published to.
    table_name: str = "VirtualController"


# ===========================================================================
# NetworkTables / application
# ===========================================================================


@dataclass
class NetworkTablesConfig:
    #: MUST INPUT: your FRC team number. The NT4 client uses it to find the
    #: roboRIO (10.TE.AM.2). 0 means "not configured" and run.py --check
    #: fails on it.
    team_number: int = 0
    #: Set to an explicit host/IP (e.g. "10.12.34.2" or "127.0.0.1") to skip
    #: team-number discovery. Useful with a desktop NT server for testing.
    server_override: str = ""
    client_identity: str = "pi-perception"
    perception_table: str = "Perception"
    fms_table: str = "FMSInfo"
    key_is_red_alliance: str = "IsRedAlliance"
    #: Alliance information older than this is treated as UNKNOWN.
    alliance_stale_timeout_s: float = 5.0
    #: Log a state change if the connection flaps; this bounds retry noise.
    reconnect_log_period_s: float = 10.0


@dataclass
class LoggingConfig:
    directory: str = "logs"
    filename: str = "perception.log"
    level: str = "INFO"
    console_level: str = "INFO"
    max_bytes: int = 5 * 1024 * 1024
    backup_count: int = 5
    #: Minimum seconds between repeats of an identical rate-limited message.
    rate_limit_s: float = 5.0
    #: Period of the one-line periodic status summary.
    status_period_s: float = 5.0


@dataclass
class AppConfig:
    """The whole configuration.  Build it with :func:`load_config`."""

    #: Coprocessor this configuration targets. Used by run.py --check to
    #: decide which platform preflight to run and to refuse a development
    #: detector backend on the competition machine.
    platform: str = "raspberry_pi_5"

    #: Main loop target rate. The loop never runs faster than this.
    loop_hz: float = 30.0
    #: A frame older than this is refused before it reaches the accelerator:
    #: a frozen camera must never feed the detector the same image forever.
    frame_freshness_s: float = 0.25
    #: End-to-end age (capture -> now) above which a finished inference
    #: result is DISCARDED instead of entering the world model. A correct
    #: answer about where objects were 400 ms ago is a wrong answer about
    #: where they are.
    max_detection_result_age_s: float = 0.40
    #: How many cameras the scheduler may run inference on per loop cycle.
    #: With one accelerator and two cameras, 2 means "both, newest frame
    #: each"; drop to 1 to halve accelerator load at the cost of per-camera
    #: rate. The scheduler is round-robin, so 1 still alternates fairly.
    max_inferences_per_cycle: int = 2
    #: Consecutive failures of one pipeline stage before it is reported as
    #: a hard error rather than a rate-limited warning.
    stage_failure_alarm_count: int = 10

    cameras: List[CameraConfig] = field(default_factory=list)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    ranging: RangingConfig = field(default_factory=RangingConfig)
    localization: LocalizationConfig = field(default_factory=LocalizationConfig)
    world_model: WorldModelConfig = field(default_factory=WorldModelConfig)
    navigation: NavigationConfig = field(default_factory=NavigationConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    networktables: NetworkTablesConfig = field(default_factory=NetworkTablesConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def camera(self, name: str) -> CameraConfig:
        for cam in self.cameras:
            if cam.name == name:
                return cam
        raise KeyError(f"no camera named {name!r}")

    @property
    def enabled_cameras(self) -> List[CameraConfig]:
        return [c for c in self.cameras if c.enabled]


# ---------------------------------------------------------------------------
# The actual configuration used at runtime
# ---------------------------------------------------------------------------


def load_config() -> AppConfig:
    """Assemble the running configuration.

    This is the ONE place the camera list is defined.  Everything else in
    the package receives config objects as arguments so it can be tested
    with a purpose-built configuration.
    """
    return AppConfig(
        networktables=NetworkTablesConfig(
            team_number=0,  # MUST INPUT: your FRC team number
        ),
        cameras=[
            CameraConfig(
                name="cam0",
                # MUST INPUT: the persistent path for this camera. Find it with
                #   ls -l /dev/v4l/by-id/
                # and confirm with
                #   v4l2-ctl -d <path> --list-formats-ext
                # Leaving this empty falls back to device_index, which is not
                # safe across reboots.
                device_path="",
                device_index=0,
                # MUST INPUT: every mount value below must be measured on the
                # real robot. They are left at zero on purpose so that an
                # unmeasured camera produces obviously wrong numbers during
                # bring-up instead of plausible-but-wrong ones.
                mount_x_m=0.0,
                mount_y_m=0.0,
                mount_z_m=0.0,
                pitch_deg=0.0,
                yaw_deg=0.0,
                roll_deg=0.0,
            ),
            CameraConfig(
                name="cam1",
                device_path="",   # MUST INPUT: see cam0
                device_index=2,   # each USB camera often claims two /dev/video nodes
                mount_x_m=0.0,    # MUST INPUT: measure on the robot
                mount_y_m=0.0,    # MUST INPUT: measure on the robot
                mount_z_m=0.0,    # MUST INPUT: measure on the robot
                pitch_deg=0.0,    # MUST INPUT: measure on the robot
                yaw_deg=0.0,      # MUST INPUT: measure on the robot
                roll_deg=0.0,     # MUST INPUT: measure on the robot
            ),
        ],
    )


# ---------------------------------------------------------------------------
# Validation (used by run.py --check and at startup)
# ---------------------------------------------------------------------------


@dataclass
class ConfigIssue:
    severity: str  # "FAIL" | "WARNING" | "MUST INPUT"
    where: str
    message: str


def validate_config(cfg: AppConfig) -> List[ConfigIssue]:
    """Structural and completeness checks that need no hardware.

    Returns a list of issues; an empty list means the configuration is
    internally consistent AND has no unfilled MUST INPUT values.
    """
    issues: List[ConfigIssue] = []

    def fail(where: str, msg: str) -> None:
        issues.append(ConfigIssue("FAIL", where, msg))

    def warn(where: str, msg: str) -> None:
        issues.append(ConfigIssue("WARNING", where, msg))

    def must(where: str, msg: str) -> None:
        issues.append(ConfigIssue("MUST INPUT", where, msg))

    # -- NetworkTables ------------------------------------------------------
    nt = cfg.networktables
    if nt.team_number <= 0 and not nt.server_override:
        must("networktables.team_number", "set your FRC team number (or server_override for bench testing)")
    if nt.team_number < 0 or nt.team_number > 99999:
        fail("networktables.team_number", "team number out of range")

    # -- Cameras ------------------------------------------------------------
    if not cfg.enabled_cameras:
        fail("cameras", "no cameras enabled")
    names = [c.name for c in cfg.cameras]
    if len(set(names)) != len(names):
        fail("cameras", f"duplicate camera names: {names}")
    sources = [c.source for c in cfg.enabled_cameras]
    if len(set(map(str, sources))) != len(sources):
        fail("cameras", f"two enabled cameras share a device: {sources}")

    for cam in cfg.enabled_cameras:
        where = f"cameras[{cam.name}]"
        if not cam.uses_persistent_path:
            must(
                where + ".device_path",
                "set a persistent path from `ls -l /dev/v4l/by-id/` (or by-path); "
                "/dev/videoN numbers change across reboots and the two cameras "
                "will eventually swap",
            )
        if cam.width <= 0 or cam.height <= 0 or cam.fps <= 0:
            fail(where, "width/height/fps must be positive")
        if not cam.has_calibrated_intrinsics:
            must(where + ".fx/fy/cx/cy", "run scripts/calibrate_intrinsics.py; uncalibrated cameras cannot solve AprilTag poses")
        else:
            fx, fy, cx, cy = cam.effective_intrinsics()
            if not (0 < cx < cam.width) or not (0 < cy < cam.height):
                fail(where, f"principal point ({cx:.1f},{cy:.1f}) outside the image")
            if fx <= 0 or fy <= 0:
                fail(where, "focal lengths must be positive")
            if abs(fx - fy) / max(fx, fy) > 0.15:
                warn(where, f"fx and fy differ by >15% ({fx:.1f} vs {fy:.1f}); re-check the calibration")
        if cam.mount_z_m <= 0.0:
            must(where + ".mount_z_m", "measure the lens height above the floor; ground ranging is impossible without it")
        if cam.pitch_deg == 0.0 and cam.yaw_deg == 0.0 and cam.roll_deg == 0.0:
            must(where + ".pitch_deg", "measure the camera mounting angles; a perfectly level camera never sees the ground plane")
        if tuple(cam.distance_poly) == (1.0, 0.0):
            warn(where + ".distance_poly", "no empirical distance correction fitted yet (run scripts/calibrate_distance.py)")
        if not (0.0 <= cam.usable_bottom_min_frac < cam.usable_bottom_max_frac <= 1.0):
            fail(where, "usable image fractions must satisfy 0 <= min < max <= 1")

    # -- Platform / detector backend ---------------------------------------
    if cfg.platform not in KNOWN_PLATFORMS:
        warn("platform", f"unrecognized platform {cfg.platform!r}; expected one of {sorted(KNOWN_PLATFORMS)}")
    if cfg.detector.backend not in KNOWN_DETECTOR_BACKENDS:
        fail("detector.backend", f"unknown backend {cfg.detector.backend!r}; expected one of {sorted(KNOWN_DETECTOR_BACKENDS)}")
    elif cfg.detector.backend == "pytorch" and cfg.platform == "raspberry_pi_5":
        warn(
            "detector.backend",
            "the PyTorch backend is selected on the Raspberry Pi. It is a "
            "DEVELOPMENT backend: running YOLO on the Pi's CPU starves capture, "
            "AprilTags and the main loop. Use backend='hailo' for competition.",
        )
    if cfg.detector.backend == "hailo" and cfg.detector.hailo_arch not in HAILO_ARCHITECTURES:
        fail(
            "detector.hailo_arch",
            f"{cfg.detector.hailo_arch!r} is not a Hailo architecture; expected one of "
            f"{sorted(HAILO_ARCHITECTURES)} (the AI HAT+ 26 TOPS is 'hailo8')",
        )
    if cfg.max_inferences_per_cycle < 1:
        fail("max_inferences_per_cycle", "must be at least 1")
    if cfg.max_detection_result_age_s <= 0:
        fail("max_detection_result_age_s", "must be positive")
    if cfg.max_detection_result_age_s < cfg.frame_freshness_s:
        warn(
            "max_detection_result_age_s",
            "is below frame_freshness_s, so a frame that passed the freshness "
            "check can never survive inference; results will always be dropped",
        )

    # -- Detector -----------------------------------------------------------
    det = cfg.detector
    if tuple(det.class_names) != EXPECTED_CLASS_NAMES:
        fail("detector.class_names", f"must be exactly {EXPECTED_CLASS_NAMES}, got {tuple(det.class_names)}")
    if not (0.0 < det.conf_threshold < 1.0):
        fail("detector.conf_threshold", "must be in (0,1)")
    if not (0.0 < det.iou_threshold < 1.0):
        fail("detector.iou_threshold", "must be in (0,1)")
    if det.imgsz % 32 != 0:
        warn("detector.imgsz", "YOLO prefers a multiple of 32")
    for name in det.per_class_conf:
        if name not in EXPECTED_CLASS_NAMES:
            fail("detector.per_class_conf", f"unknown class {name!r}")
    if det.fuel_diameter_m is None:
        must("detector.fuel_diameter_m", "FUEL diameter from the FRC 2026 game manual (width cross-check disabled until set)")
    elif det.fuel_diameter_m <= 0:
        fail("detector.fuel_diameter_m", "must be positive")
    if det.bumper_width_m is None:
        must("detector.bumper_width_m", "measured bumper face width of a real robot (width cross-check disabled until set)")

    # -- Localization -------------------------------------------------------
    loc = cfg.localization
    if loc.apriltag.tag_size_m is None:
        must("localization.apriltag.tag_size_m", "AprilTag edge length for FRC 2026, from the official game manual")
    elif loc.apriltag.tag_size_m <= 0:
        fail("localization.apriltag.tag_size_m", "must be positive")
    if not (0.0 <= loc.correction_alpha_translation <= 1.0):
        fail("localization.correction_alpha_translation", "must be in [0,1]")
    if not (0.0 <= loc.correction_alpha_heading <= 1.0):
        fail("localization.correction_alpha_heading", "must be in [0,1]")
    if loc.disagreement_reject_m <= loc.disagreement_warn_m:
        fail("localization.disagreement_reject_m", "reject threshold must exceed the warn threshold")
    if loc.disagreement_reject_rad <= loc.disagreement_warn_rad:
        fail("localization.disagreement_reject_rad", "reject threshold must exceed the warn threshold")
    if loc.odometry_stale_timeout_s <= 0:
        fail("localization.odometry_stale_timeout_s", "must be positive")
    if not loc.require_tag_initialization:
        warn("localization.require_tag_initialization", "pose can become valid without ever seeing a tag; the field origin is then whatever the RIO assumed")

    # -- World model --------------------------------------------------------
    wm = cfg.world_model
    if wm.merge_distance_game_piece_m > wm.gate_game_piece_m:
        warn("world_model", "cross-camera merge distance exceeds the track gate; duplicates may create separate tracks")
    if wm.velocity_min_samples < 2:
        fail("world_model.velocity_min_samples", "need at least 2 samples to fit a velocity")
    if wm.expire_timeout_game_piece_s <= wm.fresh_timeout_game_piece_s:
        fail("world_model", "expire timeout must exceed the fresh timeout")

    # -- Navigation ---------------------------------------------------------
    nav = cfg.navigation
    if nav.max_speed_mps <= 0 or nav.max_omega_radps <= 0:
        fail("navigation", "speed caps must be positive")
    if nav.search_point_field is None:
        must("navigation.search_point_field", "a staging point in official FRC 2026 field coordinates (None => spin-search in place, which is safe but passive)")
    if nav.opponent_radius_m <= 0 or nav.own_robot_radius_m <= 0:
        fail("navigation", "robot radii must be positive")

    # -- Controller ---------------------------------------------------------
    ctl = cfg.controller
    if ctl.enabled:
        warn("controller.enabled", "VIRTUAL CONTROLLER IS ENABLED -- the Pi will publish live stick commands")
    if not (0.0 < ctl.output_limit <= 1.0):
        fail("controller.output_limit", "must be in (0,1]")
    if not (0.0 <= ctl.deadband < 1.0):
        fail("controller.deadband", "must be in [0,1)")
    if ctl.full_scale_speed_mps <= 0 or ctl.full_scale_omega_radps <= 0:
        fail("controller", "full-scale speeds must be positive")
    if ctl.slew_rate_per_s is not None and ctl.slew_rate_per_s <= 0:
        fail("controller.slew_rate_per_s", "must be positive or None")

    # -- App ----------------------------------------------------------------
    if cfg.loop_hz <= 0:
        fail("loop_hz", "must be positive")
    if cfg.frame_freshness_s <= 0:
        fail("frame_freshness_s", "must be positive")

    return issues


def config_summary(cfg: AppConfig) -> str:
    """One-screen human summary, printed at startup and by --check."""
    lines = [
        f"platform={cfg.platform} detector={cfg.detector.backend}"
        + (f" ({cfg.detector.hailo_arch})" if cfg.detector.backend == "hailo" else ""),
        f"team={cfg.networktables.team_number or '(unset)'} "
        f"server={cfg.networktables.server_override or 'auto'} loop={cfg.loop_hz:g}Hz",
        f"model={cfg.detector.model_path} classes={tuple(cfg.detector.class_names)}",
        f"layout={cfg.localization.apriltag.field_layout_path} "
        f"tag_size={cfg.localization.apriltag.tag_size_m}",
        f"virtual_controller_enabled={cfg.controller.enabled}",
    ]
    for cam in cfg.cameras:
        fx, fy, cx, cy = cam.effective_intrinsics()
        lines.append(
            f"  camera {cam.name}: src={cam.source} {cam.width}x{cam.height}@{cam.fps} "
            f"calibrated={cam.has_calibrated_intrinsics} "
            f"fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f} "
            f"mount=({cam.mount_x_m:.3f},{cam.mount_y_m:.3f},{cam.mount_z_m:.3f}) "
            f"rpy=({cam.roll_deg:g},{cam.pitch_deg:g},{cam.yaw_deg:g})deg "
            f"enabled={cam.enabled}"
        )
    return "\n".join(lines)


__all__ = [
    "AppConfig",
    "AprilTagConfig",
    "CameraConfig",
    "CLASS_BUMPER_BLUE",
    "CLASS_BUMPER_RED",
    "CLASS_FUEL",
    "CLASS_TO_KIND",
    "ConfigIssue",
    "ControllerConfig",
    "DetectorConfig",
    "EXPECTED_CLASS_NAMES",
    "HAILO_ARCHITECTURES",
    "KNOWN_DETECTOR_BACKENDS",
    "KNOWN_PLATFORMS",
    "LocalizationConfig",
    "LoggingConfig",
    "NavigationConfig",
    "NetworkTablesConfig",
    "RangingConfig",
    "WorldModelConfig",
    "config_summary",
    "load_config",
    "validate_config",
]
