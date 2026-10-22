"""AprilTag detection and absolute pose solving.

Separated from ``localization.py`` on purpose: everything that needs the
native ``pupil_apriltags`` extension lives here, while the fusion logic
and the pose-validity rules stay in a module that imports nothing but
numpy and can therefore be unit tested anywhere.

FRAME CONVENTIONS -- read this before changing a sign
----------------------------------------------------
``pupil_apriltags`` reports ``pose_R``/``pose_t`` as the pose of the TAG
in the CAMERA frame, using the OpenCV camera convention (+X right, +Y
down, +Z out of the lens).  Its TAG frame is the "optical" one: +X right
across the tag as printed, +Y down the tag, +Z INTO the tag away from the
viewer.

WPILib's field layout uses a different tag frame: +X out of the tag face
toward a viewer, +Z up, +Y completing the right-handed set (which lands
on the viewer's right).  :data:`APRILTAG_TO_WPILIB_TAG` converts between
them, and the full chain is

    field_from_robot = field_from_tagWPI
                       @ inv(camera_from_tagWPI)
                       @ inv(robot_from_camera)

where ``camera_from_tagWPI = camera_from_tagAT @ inv(APRILTAG_TO_WPILIB_TAG)``.

This convention is the single most likely place for a sign error in the
whole project, which is why ``scripts/verify_localization.py`` exists:
push the robot to a known spot on the real field and confirm X, Y and
heading all move the way you expect before trusting any of it.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .config import AprilTagConfig, CameraConfig
from .field_map import FieldMap
from .geometry import (
    camera_to_robot_transform,
    invert_transform,
    make_transform,
    tilt_from_rotation,
    wrap_angle,
    yaw_from_rotation,
)
from .types import Pose2d, TagObservation, TagPoseEstimate

logger = logging.getLogger(__name__)


#: 4x4 transform taking points expressed in the pupil-apriltags TAG frame
#: into the WPILib TAG frame.  See the module docstring for the derivation.
APRILTAG_TO_WPILIB_TAG = make_transform(
    np.array(
        [
            [0.0, 0.0, -1.0],
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ],
        dtype=float,
    ),
    (0.0, 0.0, 0.0),
)


def robot_pose_from_tag(
    field_from_tag: np.ndarray,
    camera_from_tag_apriltag: np.ndarray,
    robot_from_camera: np.ndarray,
) -> np.ndarray:
    """Solve the robot's field pose from one tag sighting.

    All three arguments are 4x4 rigid transforms:

    ``field_from_tag``
        From the official layout (WPILib tag frame).
    ``camera_from_tag_apriltag``
        What ``pupil_apriltags`` returns (pupil tag frame, OpenCV camera).
    ``robot_from_camera``
        From the measured camera mount.

    Returns ``field_from_robot`` as a 4x4 transform.  Pure math: no
    hardware, fully unit tested.
    """
    camera_from_tag_wpilib = camera_from_tag_apriltag @ invert_transform(
        APRILTAG_TO_WPILIB_TAG
    )
    return (
        field_from_tag
        @ invert_transform(camera_from_tag_wpilib)
        @ invert_transform(robot_from_camera)
    )


def pose2d_from_transform(field_from_robot: np.ndarray) -> Pose2d:
    """Flatten a 4x4 field pose to the 2D pose the robot actually drives in."""
    return Pose2d(
        x=float(field_from_robot[0, 3]),
        y=float(field_from_robot[1, 3]),
        theta=wrap_angle(yaw_from_rotation(field_from_robot[:3, :3])),
    )


def circular_mean(angles: Sequence[float]) -> float:
    """Mean of angles that wraps correctly across +-pi."""
    if not angles:
        return 0.0
    sin_sum = sum(math.sin(a) for a in angles)
    cos_sum = sum(math.cos(a) for a in angles)
    return math.atan2(sin_sum, cos_sum)


def aggregate_tag_poses(
    observations: Sequence[TagObservation],
    timestamp: float,
    outlier_distance_m: float,
    max_tag_distance_m: float,
) -> Optional[TagPoseEstimate]:
    """Combine per-tag solutions into one estimate, discarding outliers.

    With several tags visible, the median solution is the reference and
    single-tag solutions further than ``outlier_distance_m`` from it are
    dropped -- one badly-flipped tag must not drag the average.  The
    residual spread of the survivors is reported as the consistency error
    and feeds the quality score, so navigation can tell a tight four-tag
    fix from one distant, ambiguous tag.
    """
    if not observations:
        return None

    xs = np.array([o.robot_pose.x for o in observations], dtype=float)
    ys = np.array([o.robot_pose.y for o in observations], dtype=float)
    median_x, median_y = float(np.median(xs)), float(np.median(ys))

    if len(observations) >= 3:
        keep = [
            o
            for o in observations
            if math.hypot(o.robot_pose.x - median_x, o.robot_pose.y - median_y)
            <= outlier_distance_m
        ]
        if keep:
            observations = keep

    xs = np.array([o.robot_pose.x for o in observations], dtype=float)
    ys = np.array([o.robot_pose.y for o in observations], dtype=float)
    thetas = [o.robot_pose.theta for o in observations]

    mean_x = float(np.mean(xs))
    mean_y = float(np.mean(ys))
    mean_theta = wrap_angle(circular_mean(thetas))

    spread_m = float(max(math.hypot(x - mean_x, y - mean_y) for x, y in zip(xs, ys)))
    spread_rad = float(max(abs(wrap_angle(t - mean_theta)) for t in thetas))
    mean_distance = float(np.mean([o.distance_m for o in observations]))

    count = len(observations)
    quality_count = min(count, 3) / 3.0
    quality_distance = max(
        0.0, min(1.0, 1.0 - mean_distance / max(max_tag_distance_m, 1e-6))
    )
    quality_spread = 1.0
    if count > 1 and outlier_distance_m > 0:
        quality_spread = max(0.0, min(1.0, 1.0 - spread_m / outlier_distance_m))
    quality = 0.4 * quality_count + 0.4 * quality_distance + 0.2 * quality_spread

    return TagPoseEstimate(
        pose=Pose2d(mean_x, mean_y, mean_theta),
        timestamp=timestamp,
        tag_ids=sorted(o.tag_id for o in observations),
        consistency_error_m=spread_m,
        consistency_error_rad=spread_rad,
        mean_distance_m=mean_distance,
        quality=quality,
    )


class AprilTagLocalizer:
    """Detects tags in camera frames and solves the robot's field pose.

    One ``pupil_apriltags.Detector`` is shared across cameras (it is
    stateless between calls); the per-camera intrinsics and mount
    transform are supplied per frame.
    """

    def __init__(self, cfg: AprilTagConfig, field_map: Optional[FieldMap]) -> None:
        self.cfg = cfg
        self.field_map = field_map
        self._detector: Any = None
        self.load_error: str = ""
        #: Counts of why individual tag solutions were thrown away; published
        #: for debugging because "no tag pose" is otherwise mysterious.
        self.rejections: Dict[str, int] = {}
        self._robot_from_camera: Dict[str, np.ndarray] = {}

    @property
    def available(self) -> bool:
        return self._detector is not None and self.field_map is not None

    def load(self) -> bool:
        """Create the native detector.  Returns False (never raises) on failure."""
        if self._detector is not None:
            return True
        if self.field_map is None:
            self.load_error = "no field layout loaded; AprilTag localization disabled"
            logger.error("apriltag: %s", self.load_error)
            return False
        if self.cfg.tag_size_m is None or self.cfg.tag_size_m <= 0:
            self.load_error = (
                "config.localization.apriltag.tag_size_m is unset; refusing to guess "
                "the 2026 tag size (see the game manual)"
            )
            logger.error("apriltag: %s", self.load_error)
            return False
        try:
            from pupil_apriltags import Detector
        except ImportError as exc:
            self.load_error = f"pupil-apriltags not installed: {exc}"
            logger.error("apriltag: %s", self.load_error)
            return False
        try:
            self._detector = Detector(
                families=self.cfg.tag_family,
                nthreads=self.cfg.nthreads,
                quad_decimate=self.cfg.quad_decimate,
                quad_sigma=self.cfg.quad_sigma,
                refine_edges=self.cfg.refine_edges,
                decode_sharpening=self.cfg.decode_sharpening,
            )
        except Exception as exc:  # noqa: BLE001 - native code can raise anything
            self.load_error = f"AprilTag detector construction failed: {exc}"
            logger.exception("apriltag: %s", self.load_error)
            return False
        logger.info(
            "apriltag: detector ready (%s, %d tags in layout, tag_size=%.4f m)",
            self.cfg.tag_family, len(self.field_map.tags), self.cfg.tag_size_m,
        )
        return True

    def _robot_from_camera_for(self, cam: CameraConfig) -> np.ndarray:
        transform = self._robot_from_camera.get(cam.name)
        if transform is None:
            transform = camera_to_robot_transform(cam)
            self._robot_from_camera[cam.name] = transform
        return transform

    def _reject(self, reason: str) -> None:
        self.rejections[reason] = self.rejections.get(reason, 0) + 1

    def detect_frame(
        self, gray_image: Any, cam: CameraConfig, timestamp: float
    ) -> List[TagObservation]:
        """Detect tags in one GRAYSCALE image and solve a pose per tag.

        Returns only observations that survive every acceptance filter.
        Rejections are counted (not silently dropped) so a camera that
        sees tags but produces no poses is diagnosable.
        """
        if self._detector is None or self.field_map is None:
            return []
        if not cam.has_calibrated_intrinsics:
            # An FOV-derived pinhole is nowhere near good enough for a 6-DOF
            # solve; accepting it would produce plausible, wrong poses.
            self._reject("uncalibrated_camera")
            return []

        fx, fy, cx, cy = cam.effective_intrinsics()
        assert self.cfg.tag_size_m is not None  # guaranteed by load()
        raw = self._detector.detect(
            gray_image,
            estimate_tag_pose=True,
            camera_params=(fx, fy, cx, cy),
            tag_size=self.cfg.tag_size_m,
        )

        robot_from_camera = self._robot_from_camera_for(cam)
        observations: List[TagObservation] = []
        for detection in raw:
            tag_id = int(detection.tag_id)
            tag_pose = self.field_map.get(tag_id)
            if tag_pose is None:
                self._reject("unknown_tag_id")
                continue
            if int(getattr(detection, "hamming", 0)) > self.cfg.max_hamming:
                self._reject("hamming")
                continue
            decision_margin = float(getattr(detection, "decision_margin", 0.0))
            if decision_margin < self.cfg.min_decision_margin:
                self._reject("decision_margin")
                continue
            if getattr(detection, "pose_R", None) is None:
                self._reject("no_pose")
                continue

            translation = np.asarray(detection.pose_t, dtype=float).reshape(3)
            distance = float(np.linalg.norm(translation))
            if distance > self.cfg.max_tag_distance_m:
                self._reject("too_far")
                continue

            pose_error = getattr(detection, "pose_err", None)
            if (
                self.cfg.max_pose_error is not None
                and pose_error is not None
                and float(pose_error) > self.cfg.max_pose_error
            ):
                self._reject("pose_error")
                continue

            camera_from_tag = make_transform(
                np.asarray(detection.pose_R, dtype=float).reshape(3, 3), translation
            )
            field_from_robot = robot_pose_from_tag(
                tag_pose.field_from_tag, camera_from_tag, robot_from_camera
            )

            implied_height = float(field_from_robot[2, 3])
            implied_tilt_deg = math.degrees(tilt_from_rotation(field_from_robot[:3, :3]))
            if abs(implied_height) > self.cfg.max_implied_height_m:
                self._reject("implied_height")
                continue
            if implied_tilt_deg > self.cfg.max_implied_tilt_deg:
                self._reject("implied_tilt")
                continue

            pose = pose2d_from_transform(field_from_robot)
            if not pose.is_finite():
                self._reject("non_finite")
                continue
            if not self.field_map.contains_point(
                pose.x, pose.y, self.cfg.field_bounds_margin_m
            ):
                self._reject("off_field")
                continue

            observations.append(
                TagObservation(
                    camera_id=cam.name,
                    timestamp=timestamp,
                    tag_id=tag_id,
                    robot_pose=pose,
                    distance_m=distance,
                    decision_margin=decision_margin,
                    hamming=int(getattr(detection, "hamming", 0)),
                    pose_error=None if pose_error is None else float(pose_error),
                    implied_height_m=implied_height,
                    implied_tilt_deg=implied_tilt_deg,
                )
            )
        return observations

    def estimate(
        self, observations: Sequence[TagObservation], timestamp: float
    ) -> Optional[TagPoseEstimate]:
        """Fuse this cycle's accepted tag observations into one estimate."""
        return aggregate_tag_poses(
            observations,
            timestamp,
            self.cfg.multi_tag_outlier_m,
            self.cfg.max_tag_distance_m,
        )


def to_grayscale(image: Any) -> Any:
    """BGR -> gray for the tag detector.  Requires OpenCV at call time."""
    import cv2

    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


__all__ = [
    "APRILTAG_TO_WPILIB_TAG",
    "AprilTagLocalizer",
    "aggregate_tag_poses",
    "circular_mean",
    "pose2d_from_transform",
    "robot_pose_from_tag",
    "to_grayscale",
]
