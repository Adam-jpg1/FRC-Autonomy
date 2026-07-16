"""All tunable values for the whole system live in this one file.

Nothing else in the package hardcodes a threshold, timeout, gain, or
dimension. If you are tuning at competition, tune here.

Units: meters, seconds, radians unless a field name says `_deg`.
Angles in this file that humans type are degrees (`_deg`); code converts.

Every value that cannot be known without the physical robot/camera/model
is marked `# MUST INPUT` with an explanation. Search for that string.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Cameras
# ---------------------------------------------------------------------------

@dataclass
class CameraConfig:
    """One physical camera: capture settings, lens model, and mounting pose.

    Mount pose is the camera's position/orientation in the ROBOT frame
    (+X forward, +Y left, +Z up, origin at robot center on the floor).
    `mount_pitch_deg` is positive when the camera is tilted DOWN toward
    the floor. `mount_yaw_deg` is CCW-positive (left).
    """

    name: str
    device_index: int          # /dev/videoN index  # MUST INPUT: run `ls /dev/video*` on the Jetson; order can change per boot — pin with udev rules if needed
    width: int = 1280
    height: int = 800          # OV9782 native is 1280x800
    fps: int = 60
    fourcc: str = "MJPG"       # OV9782 needs MJPG for full res/rate over USB2

    # Intrinsics (pixels). Produced by scripts/calibrate_intrinsics.py.
    # If left at None the code falls back to a pinhole model derived from
    # horizontal_fov_deg — usable for bring-up, NOT for AprilTag pose.
    fx: Optional[float] = None  # MUST INPUT: from chessboard calibration
    fy: Optional[float] = None  # MUST INPUT: from chessboard calibration
    cx: Optional[float] = None  # MUST INPUT: from chessboard calibration
    cy: Optional[float] = None  # MUST INPUT: from chessboard calibration
    # OpenCV distortion (k1, k2, p1, p2, k3). Zeros = ideal lens.
    dist_coeffs: Tuple[float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0)  # MUST INPUT: from chessboard calibration

    # Fallback lens model when intrinsics are not calibrated yet.
    horizontal_fov_deg: float = 70.0  # MUST INPUT: depends on which Arducam OV9782 lens variant you bought (70/100/120 deg versions exist)

    # Mounting pose in the robot frame.
    mount_x: float = 0.0        # MUST INPUT: measure lens center fwd of robot center
    mount_y: float = 0.0        # MUST INPUT: measure lens center left of robot center
    mount_z: float = 0.5        # MUST INPUT: measure lens height above the floor
    mount_pitch_deg: float = 15.0  # MUST INPUT: measure downward tilt (down = positive)
    mount_yaw_deg: float = 0.0     # MUST INPUT: measure pan (CCW/left = positive)
    mount_roll_deg: float = 0.0    # MUST INPUT: usually 0 unless mounted sideways

    # Range-correction polynomial, highest power first (numpy.polyval order).
    # Applied to the raw camera->object ground distance. (1.0, 0.0) = identity.
    # Produced by scripts/calibrate_distance.py.
    distance_poly: Tuple[float, ...] = (1.0, 0.0)

    def effective_intrinsics(self) -> Tuple[float, float, float, float]:
        """Return (fx, fy, cx, cy), deriving a pinhole fallback from FOV."""
        if self.fx and self.fy and self.cx is not None and self.cy is not None:
            return (self.fx, self.fy, self.cx, self.cy)
        import math
        f = (self.width / 2.0) / math.tan(math.radians(self.horizontal_fov_deg) / 2.0)
        return (f, f, self.width / 2.0, self.height / 2.0)

    @property
    def has_calibrated_intrinsics(self) -> bool:
        return bool(self.fx and self.fy)


# ---------------------------------------------------------------------------
# Object detection
# ---------------------------------------------------------------------------

@dataclass
class DetectorConfig:
    model_path: str = "models/frc2026.pt"  # MUST INPUT: your trained YOLO weights (classes below). Export to TensorRT .engine on the Jetson for speed.
    # Class names EXACTLY as trained. Do not reorder casually: role routing
    # below is by NAME, so order here does not matter, but spelling does.
    class_names: Tuple[str, ...] = ("fuel", "bumper_blue", "bumper_red")
    conf_threshold: float = 0.40
    iou_threshold: float = 0.50
    imgsz: int = 640            # inference size; 640 is a good Orin Nano default
    device: str = "auto"        # "auto" -> cuda:0 if available, else cpu
    half: bool = True           # FP16 on GPU; ignored on CPU

    # Physical sizes for the width-based range cross-check.
    fuel_diameter_m: float = 0.150  # MUST INPUT: FUEL diameter from the official 2026 game manual — verify before trusting width ranging
    bumper_width_m: float = 0.90    # nominal robot-with-bumpers width; only used as a sanity cross-check for robots


# Role routing: class name -> role. Only these two roles exist.
CLASS_ROLES = {
    "fuel": "game_piece",
    "bumper_blue": "robot",
    "bumper_red": "robot",
}


# ---------------------------------------------------------------------------
# Localization (AprilTags + roboRIO odometry fusion)
# ---------------------------------------------------------------------------

@dataclass
class LocalizationConfig:
    field_layout_path: str = "data/field_layout.json"  # official WPILib JSON; fetch with scripts/download_field_layout.py
    tag_family: str = "tag36h11"
    tag_size_m: float = 0.1651   # 6.5 in, same as 2024/2025 — MUST INPUT: verify against the 2026 game manual
    quad_decimate: float = 2.0   # apriltag detector speed/accuracy tradeoff

    # Per-tag acceptance filters.
    max_tag_distance_m: float = 4.5   # solves get noisy beyond this
    min_decision_margin: float = 30.0
    max_hamming: int = 0
    max_pose_height_error_m: float = 0.35  # reject solves that put the robot off the floor
    max_pose_tilt_deg: float = 15.0        # reject solves that say the robot is tipped

    # Fusion: fused = correction * odometry. Tag fixes update the correction.
    fusion_alpha: float = 0.15         # EMA weight of each new tag fix on the correction
    odom_stale_s: float = 0.5          # RIO timestamp must ADVANCE within this window
    tag_only_timeout_s: float = 1.0    # with odometry dead, a tag pose is trusted this long
    disagreement_warn_m: float = 1.0   # warn if tag and fused poses differ by more

    # Fallback field size only if the layout JSON is missing (tags disabled).
    fallback_field_length_m: float = 17.548  # MUST INPUT if layout JSON absent: from 2026 manual
    fallback_field_width_m: float = 8.052    # MUST INPUT if layout JSON absent: from 2026 manual


# ---------------------------------------------------------------------------
# World model (field-space tracking)
# ---------------------------------------------------------------------------

@dataclass
class WorldModelConfig:
    gate_game_piece_m: float = 0.55   # association radius for fuel
    gate_robot_m: float = 1.20        # association radius for robots
    position_alpha: float = 0.5       # EMA weight of a new measurement on position
    velocity_alpha: float = 0.3       # EMA weight on velocity estimate
    min_dt_s: float = 0.01            # ignore velocity from near-zero dt
    track_timeout_game_piece_s: float = 1.0
    track_timeout_robot_s: float = 1.5
    min_hits_to_confirm: int = 2      # sightings before a track is reported
    field_margin_m: float = 0.25      # reject projections this far outside the field
    max_speed_game_piece_mps: float = 6.0  # clamp for absurd velocity estimates


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------

@dataclass
class NavConfig:
    pocket_radius_m: float = 1.0      # balls within this radius are one "pocket"
    avoid_radius_m: float = 1.2       # keep the path this far from opponent robots
    blocked_penalty: float = 100.0    # score penalty for a blocked pocket
    distance_weight: float = 0.05     # score penalty per meter; density dominates, distance breaks ties
    max_speed: float = 0.6            # translation command cap, 0..1 stick units
    approach_kp: float = 0.8          # stick-per-meter toward the target
    slow_radius_m: float = 1.0        # start slowing inside this distance
    arrive_tolerance_m: float = 0.35
    heading_kp: float = 1.2           # stick-per-radian of heading error
    max_omega: float = 0.5            # rotation command cap, 0..1 stick units
    search_spin_omega: float = 0.3    # spin speed while searching
    wall_margin_m: float = 0.6        # never target closer than this to a wall


# ---------------------------------------------------------------------------
# Virtual controller
# ---------------------------------------------------------------------------

@dataclass
class ControllerConfig:
    # SAFETY: the virtual gamepad publishes zeros and enabled=False until
    # this is True. Flip only when you are ready for autonomous driving.
    enabled: bool = False

    # True if the robot's teleop drive interprets sticks field-oriented.
    # False = robot-oriented sticks (we rotate commands by -heading first).
    field_oriented_drive: bool = True  # MUST INPUT: match your drive code's teleop binding

    # Stick sign conventions vary between robot codebases. Defaults follow
    # the WPILib examples (push forward = negative Y = drive +X). Verify
    # with the robot on blocks before trusting.
    invert_left_x: bool = True   # MUST INPUT: verify against your bindings, robot on blocks
    invert_left_y: bool = True   # MUST INPUT: verify against your bindings, robot on blocks
    invert_right_x: bool = True  # MUST INPUT: verify against your bindings, robot on blocks

    axis_deadband: float = 0.02   # commands below this publish as exactly 0
    slew_rate_per_s: float = 3.0  # max axis change per second (jerk limiting)


# ---------------------------------------------------------------------------
# App / plumbing
# ---------------------------------------------------------------------------

@dataclass
class AppConfig:
    team_number: int = 0            # MUST INPUT: your FRC team number (used for NT server discovery)
    nt_server_override: str = ""    # set to e.g. "10.TE.AM.2" if mDNS discovery is flaky; empty = use team number
    nt_identity: str = "jetson-perception"

    loop_hz: float = 30.0
    frame_freshness_s: float = 0.25   # frames older than this are discarded (frozen camera protection)
    apriltag_every_n_frames: int = 1  # raise to 2-3 if CPU-bound

    log_dir: str = "logs"
    log_level: str = "INFO"
    status_log_period_s: float = 5.0

    cameras: List[CameraConfig] = field(default_factory=list)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    localization: LocalizationConfig = field(default_factory=LocalizationConfig)
    world_model: WorldModelConfig = field(default_factory=WorldModelConfig)
    nav: NavConfig = field(default_factory=NavConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)


def default_config() -> AppConfig:
    """The single place the running configuration is assembled."""
    return AppConfig(
        team_number=0,  # MUST INPUT: your team number
        cameras=[
            CameraConfig(
                name="front",
                device_index=0,   # MUST INPUT: verify with `ls /dev/video*`
                mount_x=0.30,     # MUST INPUT: measure on the robot
                mount_y=0.0,      # MUST INPUT: measure on the robot
                mount_z=0.55,     # MUST INPUT: measure on the robot
                mount_pitch_deg=20.0,  # MUST INPUT: measure on the robot
                mount_yaw_deg=0.0,     # MUST INPUT: measure on the robot
            ),
            CameraConfig(
                name="rear",
                device_index=2,   # MUST INPUT: verify with `ls /dev/video*` (USB cams often enumerate 2 nodes each)
                mount_x=-0.30,    # MUST INPUT: measure on the robot
                mount_y=0.0,      # MUST INPUT: measure on the robot
                mount_z=0.55,     # MUST INPUT: measure on the robot
                mount_pitch_deg=20.0,  # MUST INPUT: measure on the robot
                mount_yaw_deg=180.0,   # MUST INPUT: measure on the robot
            ),
        ],
    )
