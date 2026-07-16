"""Robot localization: AprilTag absolute fixes fused with RIO odometry.

Odometry (read over NT, RAW blue-origin) is the smooth, high-rate
backbone; AprilTag solves correct its drift. The fusion state is a 2D
"correction" transform such that

    fused_pose = correction ∘ odometry_pose

The correction starts as identity (we trust the RIO's blue-origin frame)
and is nudged toward each accepted tag fix with an EMA.

Odometry staleness is judged by whether the RIO's PUBLISHED timestamp
value actually advances — a wedged RIO loop that keeps republishing the
same numbers is treated as dead even though NT traffic continues.
"""

import logging
import math
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .config import CameraConfig, LocalizationConfig
from .field_map import FieldMap
from .geometry import (
    camera_to_robot_transform,
    invert_transform,
    make_transform,
    wrap_angle,
    yaw_from_rotation,
)

log = logging.getLogger(__name__)


@dataclass
class Pose2D:
    x: float
    y: float
    theta: float


@dataclass
class PoseEstimate:
    x: float
    y: float
    theta: float
    valid: bool
    source: str  # "fused", "tag_only", "none"


# ---------------------------------------------------------------------------
# 2D rigid transform helpers (a transform is a Pose2D acting on points)
# ---------------------------------------------------------------------------

def transform_point(t: Pose2D, x: float, y: float) -> Tuple[float, float]:
    c, s = math.cos(t.theta), math.sin(t.theta)
    return (t.x + c * x - s * y, t.y + s * x + c * y)


def compose(a: Pose2D, b: Pose2D) -> Pose2D:
    """Return a ∘ b (apply b first, then a)."""
    x, y = transform_point(a, b.x, b.y)
    return Pose2D(x, y, wrap_angle(a.theta + b.theta))


def invert(t: Pose2D) -> Pose2D:
    c, s = math.cos(t.theta), math.sin(t.theta)
    return Pose2D(-(c * t.x + s * t.y), -(-s * t.x + c * t.y), wrap_angle(-t.theta))


# ---------------------------------------------------------------------------
# Odometry freshness
# ---------------------------------------------------------------------------

class OdometryTracker:
    """Tracks RIO odometry and whether its published timestamp advances."""

    def __init__(self, stale_s: float):
        self.stale_s = stale_s
        self._pose: Optional[Pose2D] = None
        self._last_rio_timestamp: Optional[float] = None
        self._last_advance_mono: Optional[float] = None

    def update(self, x: float, y: float, theta: float,
               rio_timestamp: float, now: Optional[float] = None) -> None:
        """Feed the currently-published NT values (call every loop)."""
        if now is None:
            now = time.monotonic()
        if rio_timestamp != self._last_rio_timestamp:
            # The RIO's own clock moved -> the publisher is alive.
            self._last_rio_timestamp = rio_timestamp
            self._last_advance_mono = now
            self._pose = Pose2D(x, y, wrap_angle(theta))

    def fresh(self, now: Optional[float] = None) -> bool:
        if now is None:
            now = time.monotonic()
        return (
            self._last_advance_mono is not None
            and (now - self._last_advance_mono) <= self.stale_s
        )

    @property
    def pose(self) -> Optional[Pose2D]:
        return self._pose


# ---------------------------------------------------------------------------
# AprilTag -> robot field pose
# ---------------------------------------------------------------------------

# pupil-apriltags (AprilTag 3) tag frame, seen from the tag's front:
# +X right, +Y down, +Z INTO the tag. WPILib layout tag frame: +X OUT of
# the face, +Y tag-left, +Z up. This matrix re-expresses apriltag-frame
# coordinates in the WPILib tag frame. VERIFY ON THE FIELD with
# scripts/verify_localization.py before trusting signs.
_ATAG_TO_WPILIB_TAG = np.array(
    [
        [0.0, 0.0, -1.0],
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=float,
)


def robot_pose_from_tag(
    tag_pose_field: np.ndarray,
    tag_rotation_cam: np.ndarray,
    tag_translation_cam: Sequence[float],
    camera_in_robot: np.ndarray,
) -> Tuple[Pose2D, float, float]:
    """Solve the robot's field pose from one tag observation.

    Args:
      tag_pose_field: 4x4 tag pose from the WPILib layout (field frame).
      tag_rotation_cam / tag_translation_cam: the pupil-apriltags solve
        (tag in camera-CV frame).
      camera_in_robot: 4x4 camera mount transform from geometry.

    Returns (pose2d, robot_z, tilt_rad) — z and tilt are for sanity checks
    (a good solve puts the robot on the floor and upright).
    """
    r = np.asarray(tag_rotation_cam, dtype=float)
    t = np.asarray(tag_translation_cam, dtype=float).reshape(3)
    # Tag (WPILib frame) as seen by the camera: re-express the solve.
    cam_to_tag = make_transform(r @ _ATAG_TO_WPILIB_TAG.T, t)
    field_to_cam = tag_pose_field @ invert_transform(cam_to_tag)
    field_to_robot = field_to_cam @ invert_transform(camera_in_robot)
    rotation = field_to_robot[:3, :3]
    tilt = math.acos(max(-1.0, min(1.0, float(rotation[2, 2]))))
    return (
        Pose2D(
            float(field_to_robot[0, 3]),
            float(field_to_robot[1, 3]),
            yaw_from_rotation(rotation),
        ),
        float(field_to_robot[2, 3]),
        tilt,
    )


@dataclass
class TagFix:
    pose: Pose2D
    weight: float
    tag_id: int
    distance_m: float


class TagLocalizer:
    """Runs pupil-apriltags on grayscale frames and produces robot poses."""

    def __init__(self, cfg: LocalizationConfig, field_map: FieldMap,
                 cameras: Sequence[CameraConfig]):
        self.cfg = cfg
        self.field_map = field_map
        self._cam_transforms = {
            cam.name: camera_to_robot_transform(cam) for cam in cameras
        }
        self._detector = None
        if field_map.loaded:
            from pupil_apriltags import Detector  # local import: native lib

            self._detector = Detector(
                families=cfg.tag_family,
                nthreads=2,
                quad_decimate=cfg.quad_decimate,
            )
        for cam in cameras:
            if not cam.has_calibrated_intrinsics:
                log.warning(
                    "camera %s has no calibrated intrinsics — AprilTag poses "
                    "from it will be inaccurate (FOV fallback in use)",
                    cam.name,
                )

    @property
    def enabled(self) -> bool:
        return self._detector is not None

    def process(self, gray_frame, cam: CameraConfig) -> List[TagFix]:
        """Detect tags in one grayscale frame, return accepted robot-pose fixes."""
        if self._detector is None:
            return []
        fx, fy, cx, cy = cam.effective_intrinsics()
        detections = self._detector.detect(
            gray_frame,
            estimate_tag_pose=True,
            camera_params=(fx, fy, cx, cy),
            tag_size=self.cfg.tag_size_m,
        )
        fixes: List[TagFix] = []
        for det in detections:
            if det.hamming > self.cfg.max_hamming:
                continue
            if det.decision_margin < self.cfg.min_decision_margin:
                continue
            tag_pose_field = self.field_map.tag_pose(det.tag_id)
            if tag_pose_field is None:
                continue
            distance = float(np.linalg.norm(np.asarray(det.pose_t).reshape(3)))
            if distance > self.cfg.max_tag_distance_m:
                continue
            pose, z, tilt = robot_pose_from_tag(
                tag_pose_field, det.pose_R, det.pose_t,
                self._cam_transforms[cam.name],
            )
            if abs(z) > self.cfg.max_pose_height_error_m:
                continue
            if math.degrees(tilt) > self.cfg.max_pose_tilt_deg:
                continue
            if not self.field_map.contains(pose.x, pose.y, margin=-1.0):
                continue  # more than 1 m outside the field: garbage solve
            # Nearer tags with cleaner decodes count for more.
            weight = det.decision_margin / (1.0 + distance * distance)
            fixes.append(TagFix(pose, weight, det.tag_id, distance))
        return fixes

    @staticmethod
    def combine(fixes: List[TagFix]) -> Optional[Pose2D]:
        """Weighted mean of fixes (circular mean for heading)."""
        if not fixes:
            return None
        total = sum(f.weight for f in fixes)
        if total <= 0:
            return None
        x = sum(f.pose.x * f.weight for f in fixes) / total
        y = sum(f.pose.y * f.weight for f in fixes) / total
        sin_sum = sum(math.sin(f.pose.theta) * f.weight for f in fixes)
        cos_sum = sum(math.cos(f.pose.theta) * f.weight for f in fixes)
        return Pose2D(x, y, math.atan2(sin_sum, cos_sum))


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------

class PoseFuser:
    """Maintains fused = correction ∘ odometry, with tag-only fallback."""

    def __init__(self, cfg: LocalizationConfig):
        self.cfg = cfg
        self._correction = Pose2D(0.0, 0.0, 0.0)
        self._last_tag_pose: Optional[Pose2D] = None
        self._last_tag_mono: Optional[float] = None
        self._warned_disagreement = 0.0  # last warn time, for rate limiting

    def update_from_tag(self, tag_pose: Pose2D, odom: OdometryTracker,
                        now: Optional[float] = None) -> None:
        """Fold an accepted tag fix into the correction (or the fallback)."""
        if now is None:
            now = time.monotonic()
        self._last_tag_pose = tag_pose
        self._last_tag_mono = now
        if not (odom.fresh(now) and odom.pose is not None):
            return  # no live odometry to correct; tag-only fallback holds it
        target = compose(tag_pose, invert(odom.pose))
        a = self.cfg.fusion_alpha
        self._correction = Pose2D(
            self._correction.x + a * (target.x - self._correction.x),
            self._correction.y + a * (target.y - self._correction.y),
            wrap_angle(
                self._correction.theta
                + a * wrap_angle(target.theta - self._correction.theta)
            ),
        )
        # Warn (rate-limited) when the tag and fused frames disagree badly:
        # usually a sign the RIO is NOT publishing raw blue-origin odometry.
        fused = compose(self._correction, odom.pose)
        err = math.hypot(tag_pose.x - fused.x, tag_pose.y - fused.y)
        if err > self.cfg.disagreement_warn_m and (now - self._warned_disagreement) > 5.0:
            self._warned_disagreement = now
            log.warning(
                "tag pose (%.2f, %.2f) vs fused (%.2f, %.2f) differ by %.2f m — "
                "check that the RIO publishes RAW blue-origin odometry",
                tag_pose.x, tag_pose.y, fused.x, fused.y, err,
            )

    def estimate(self, odom: OdometryTracker,
                 now: Optional[float] = None) -> PoseEstimate:
        """Best current pose. `valid=False` means DO NOT navigate on it."""
        if now is None:
            now = time.monotonic()
        if odom.fresh(now) and odom.pose is not None:
            fused = compose(self._correction, odom.pose)
            return PoseEstimate(fused.x, fused.y, fused.theta, True, "fused")
        if (
            self._last_tag_pose is not None
            and self._last_tag_mono is not None
            and (now - self._last_tag_mono) <= self.cfg.tag_only_timeout_s
        ):
            p = self._last_tag_pose
            return PoseEstimate(p.x, p.y, p.theta, True, "tag_only")
        return PoseEstimate(0.0, 0.0, 0.0, False, "none")
