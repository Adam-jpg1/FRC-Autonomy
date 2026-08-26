"""Pure geometry: rotations, rigid transforms, ground ranging, frames.

Everything in this module is numpy-only.  No OpenCV, no hardware, no
global state -- so all of it is unit testable on a laptop, and it is.

CONVENTIONS (memorize these; every sign error in this project will come
from getting one of them wrong):

    Robot frame     +X forward, +Y left, +Z up. Origin on the floor at the
                    robot center. Yaw/theta is CCW-positive.
    Field frame     WPILib blue-origin. +X down field toward the red wall,
                    +Y to the left when standing at the blue wall, +Z up.
                    Robot heading theta is measured CCW from field +X.
    Camera frame    OpenCV: +X image right, +Y image down, +Z out the lens.
    Mount angles    pitch DOWN is positive, yaw CCW (left) is positive,
                    roll about the optical axis.

The camera-to-robot rotation is built as

    R_robot_from_camera = Rz(yaw) @ Ry(pitch) @ Rx(roll) @ CV_TO_FORWARD

where CV_TO_FORWARD re-labels the OpenCV axes onto a forward-looking
robot-style frame.  Rz/Ry/Rx are the standard right-handed rotations, so
Ry(+pitch) tips the +X axis downward -- which is why "down is positive".
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import numpy as np

from .config import CameraConfig, RangingConfig
from .types import RangeRejection, RangeResult


# ---------------------------------------------------------------------------
# Rotations and rigid transforms
# ---------------------------------------------------------------------------


def rot_x(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=float)


def rot_y(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=float)


def rot_z(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def make_transform(rotation: np.ndarray, translation: Sequence[float]) -> np.ndarray:
    """Assemble a 4x4 homogeneous transform from a 3x3 R and a 3-vector t."""
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = rotation
    transform[:3, 3] = np.asarray(translation, dtype=float)
    return transform


def invert_transform(transform: np.ndarray) -> np.ndarray:
    """Invert a RIGID 4x4 transform analytically (no general matrix inverse)."""
    rotation = transform[:3, :3]
    inverse = np.eye(4, dtype=float)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ transform[:3, 3]
    return inverse


def transform_point(transform: np.ndarray, point: Sequence[float]) -> np.ndarray:
    """Apply a 4x4 transform to a 3-vector."""
    homogeneous = np.array([point[0], point[1], point[2], 1.0], dtype=float)
    return (transform @ homogeneous)[:3]


def quaternion_to_rotation(w: float, x: float, y: float, z: float) -> np.ndarray:
    """Unit quaternion (WPILib JSON order W, X, Y, Z) to a 3x3 rotation."""
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm < 1e-12:
        raise ValueError("zero-norm quaternion")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def wrap_angle(angle: float) -> float:
    """Wrap an angle into (-pi, pi]."""
    wrapped = math.fmod(angle + math.pi, 2.0 * math.pi)
    if wrapped <= 0.0:
        wrapped += 2.0 * math.pi
    return wrapped - math.pi


def angle_difference(a: float, b: float) -> float:
    """Shortest signed difference a - b, wrapped into (-pi, pi]."""
    return wrap_angle(a - b)


def yaw_from_rotation(rotation: np.ndarray) -> float:
    """Z-axis yaw of a 3x3 rotation matrix."""
    return math.atan2(rotation[1, 0], rotation[0, 0])


def tilt_from_rotation(rotation: np.ndarray) -> float:
    """Angle in RADIANS between the rotated +Z axis and world +Z.

    Used to reject AprilTag solves that claim the robot is tipped over.
    """
    z_axis = rotation[:, 2]
    cos_tilt = float(np.clip(z_axis[2] / max(np.linalg.norm(z_axis), 1e-12), -1.0, 1.0))
    return math.acos(cos_tilt)


#: Re-labels OpenCV camera axes into a forward-looking robot-style frame:
#: camera +Z (out the lens) -> +X forward, camera +X (right) -> -Y,
#: camera +Y (down) -> -Z.
CV_TO_FORWARD = np.array(
    [
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=float,
)


def camera_to_robot_rotation(cam: CameraConfig) -> np.ndarray:
    """3x3 rotation mapping camera-frame vectors into the robot frame."""
    return (
        rot_z(math.radians(cam.yaw_deg))
        @ rot_y(math.radians(cam.pitch_deg))
        @ rot_x(math.radians(cam.roll_deg))
        @ CV_TO_FORWARD
    )


def camera_to_robot_transform(cam: CameraConfig) -> np.ndarray:
    """4x4 pose of the camera (OpenCV frame) expressed in the robot frame."""
    return make_transform(
        camera_to_robot_rotation(cam), (cam.mount_x_m, cam.mount_y_m, cam.mount_z_m)
    )


# ---------------------------------------------------------------------------
# Pixels -> rays
# ---------------------------------------------------------------------------


def undistort_pixel(
    px: float,
    py: float,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    dist: Sequence[float],
    iterations: int = 8,
) -> Tuple[float, float]:
    """Invert the OpenCV radial-tangential model (k1, k2, p1, p2, k3).

    Returns ideal normalized image coordinates.  Fixed-point iteration;
    exact and allocation-free when all coefficients are zero.
    """
    x_distorted = (px - cx) / fx
    y_distorted = (py - cy) / fy
    coefficients = (list(dist) + [0.0] * 5)[:5]
    k1, k2, p1, p2, k3 = coefficients
    if k1 == k2 == p1 == p2 == k3 == 0.0:
        return x_distorted, y_distorted
    x, y = x_distorted, y_distorted
    for _ in range(iterations):
        r2 = x * x + y * y
        radial = 1.0 + r2 * (k1 + r2 * (k2 + r2 * k3))
        if abs(radial) < 1e-9:
            break
        dx = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
        dy = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
        x = (x_distorted - dx) / radial
        y = (y_distorted - dy) / radial
    return x, y


def pixel_to_camera_ray(px: float, py: float, cam: CameraConfig) -> np.ndarray:
    """Unit ray through a pixel, in the camera (OpenCV) frame."""
    fx, fy, cx, cy = cam.effective_intrinsics()
    x, y = undistort_pixel(px, py, fx, fy, cx, cy, cam.dist_coeffs)
    ray = np.array([x, y, 1.0], dtype=float)
    return ray / float(np.linalg.norm(ray))


def pixel_to_robot_ray(px: float, py: float, cam: CameraConfig) -> np.ndarray:
    """Unit ray through a pixel, rotated into the robot frame."""
    return camera_to_robot_rotation(cam) @ pixel_to_camera_ray(px, py, cam)


# ---------------------------------------------------------------------------
# Ground-plane ranging
# ---------------------------------------------------------------------------


def ground_intersection(
    px: float,
    py: float,
    cam: CameraConfig,
    plane_z_m: float = 0.0,
    min_descent: float = 0.02,
) -> Optional[np.ndarray]:
    """Intersect the ray through a pixel with the horizontal plane z=plane_z_m.

    Returns the hit point ``[x, y, z]`` in the ROBOT frame, or ``None``
    when the ray does not descend fast enough to hit the plane in front
    of the camera.  ``min_descent`` is a guard on the near-horizon
    singularity: as the ray flattens out, the intersection distance grows
    without bound and a one-pixel error becomes a several-meter error.
    """
    ray = pixel_to_robot_ray(px, py, cam)
    origin = np.array([cam.mount_x_m, cam.mount_y_m, cam.mount_z_m], dtype=float)
    height_above_plane = origin[2] - plane_z_m
    if height_above_plane <= 0.0:
        # Camera is at or below the target plane: ground ranging is undefined.
        return None
    if ray[2] > -abs(min_descent):
        return None
    t = (plane_z_m - origin[2]) / ray[2]
    if t <= 0.0:
        return None
    return origin + t * ray


def pinhole_width_distance(
    pixel_width: float, cam: CameraConfig, true_width_m: float
) -> Optional[float]:
    """Slant range from apparent width: ``d = W * fx / w_px``.

    This is a CROSS-CHECK, not a primary measurement: it assumes the box
    width spans the object's true width, which is false for a partially
    occluded or clipped object.
    """
    if pixel_width <= 1.0 or true_width_m <= 0.0:
        return None
    fx, _fy, _cx, _cy = cam.effective_intrinsics()
    if fx <= 0.0:
        return None
    return fx * true_width_m / pixel_width


def apply_distance_polynomial(cam: CameraConfig, raw_distance_m: float) -> float:
    """Apply the per-camera empirical range correction.

    The polynomial is only trusted inside the range it was fitted over;
    beyond ``distance_poly_max_m`` the correction is held at its value at
    that limit rather than extrapolated.  A polynomial that produces a
    non-positive or non-finite result is discarded in favor of the raw
    geometric distance -- a broken calibration must degrade to "uncorrected",
    never to "behind the camera".
    """
    coefficients = np.asarray(cam.distance_poly, dtype=float)
    clamped_input = min(raw_distance_m, cam.distance_poly_max_m)
    corrected = float(np.polyval(coefficients, clamped_input))
    if raw_distance_m > cam.distance_poly_max_m:
        # Hold the correction OFFSET constant past the fitted range.
        corrected += raw_distance_m - clamped_input
    if not math.isfinite(corrected) or corrected <= 0.0:
        return raw_distance_m
    return corrected


def estimate_ground_object(
    bbox_xyxy: Sequence[float],
    cam: CameraConfig,
    ranging: RangingConfig,
    known_width_m: Optional[float] = None,
) -> RangeResult:
    """Range one detection assumed to be sitting on the floor.

    Primary estimate: the bottom-center pixel of the box is the object's
    ground contact point; the ray through it is intersected with the
    ground plane using the camera intrinsics and the measured mount pose.

    Secondary check: when the object's real width is known, the pinhole
    width distance is computed independently and compared.  Large
    disagreement degrades or rejects the measurement -- it never silently
    overrides the geometric answer.

    Returns a :class:`RangeResult` whose ``valid`` flag must be checked.
    Invalid results carry a rejection reason instead of a guessed number.
    """
    x1, y1, x2, y2 = (float(v) for v in bbox_xyxy)
    if x2 <= x1 or y2 <= y1:
        return RangeResult(False, RangeRejection.OUTSIDE_USABLE_IMAGE)

    fx, fy, _cx, _cy = cam.effective_intrinsics()
    if fx <= 0.0 or fy <= 0.0 or cam.mount_z_m <= ranging.ground_plane_z_m:
        return RangeResult(False, RangeRejection.BAD_INTRINSICS)

    # The ground contact point must be inside the usable band of the image.
    # A box whose bottom edge is clipped by the frame has no visible contact
    # point and would range far too near.
    bottom_fraction = y2 / float(cam.height)
    if not (cam.usable_bottom_min_frac <= bottom_fraction <= cam.usable_bottom_max_frac):
        return RangeResult(False, RangeRejection.OUTSIDE_USABLE_IMAGE)

    bottom_x = (x1 + x2) * 0.5
    hit = ground_intersection(
        bottom_x, y2, cam, ranging.ground_plane_z_m, ranging.min_ray_descent
    )
    if hit is None:
        return RangeResult(False, RangeRejection.ABOVE_HORIZON)

    camera_ground = np.array([cam.mount_x_m, cam.mount_y_m], dtype=float)
    offset = hit[:2] - camera_ground
    raw_distance = float(np.linalg.norm(offset))
    if raw_distance < 1e-6:
        return RangeResult(False, RangeRejection.DEGENERATE_RAY)

    corrected_distance = apply_distance_polynomial(cam, raw_distance)
    point = camera_ground + offset * (corrected_distance / raw_distance)
    x_robot, y_robot = float(point[0]), float(point[1])
    range_m = math.hypot(x_robot, y_robot)
    bearing = math.atan2(y_robot, x_robot)

    if range_m > ranging.max_range_m:
        return RangeResult(
            False, RangeRejection.TOO_FAR, raw_ground_distance_m=raw_distance
        )
    if range_m < ranging.min_range_m:
        return RangeResult(
            False, RangeRejection.TOO_NEAR, raw_ground_distance_m=raw_distance
        )

    result = RangeResult(
        valid=True,
        x_robot_m=x_robot,
        y_robot_m=y_robot,
        range_m=range_m,
        bearing_rad=bearing,
        raw_ground_distance_m=raw_distance,
    )

    # -- Independent width cross-check -------------------------------------
    if known_width_m:
        slant = pinhole_width_distance(x2 - x1, cam, known_width_m)
        if slant is not None:
            height = cam.mount_z_m - ranging.ground_plane_z_m
            ground_from_width = math.sqrt(max(slant * slant - height * height, 0.0))
            result.width_distance_m = ground_from_width
            disagreement = abs(ground_from_width - corrected_distance) / max(
                corrected_distance, 1e-6
            )
            result.width_disagreement = disagreement
            if disagreement > ranging.width_reject_ratio:
                return RangeResult(
                    False,
                    RangeRejection.WIDTH_DISAGREEMENT,
                    width_distance_m=ground_from_width,
                    width_disagreement=disagreement,
                    raw_ground_distance_m=raw_distance,
                )
            if disagreement > ranging.width_warn_ratio:
                result.quality = ranging.width_warn_quality
    return result


# ---------------------------------------------------------------------------
# Robot <-> field (2D)
# ---------------------------------------------------------------------------


def robot_to_field(
    x_robot: float,
    y_robot: float,
    robot_x: float,
    robot_y: float,
    robot_theta: float,
) -> Tuple[float, float]:
    """Rotate+translate a robot-relative point into field coordinates."""
    c, s = math.cos(robot_theta), math.sin(robot_theta)
    return (robot_x + c * x_robot - s * y_robot, robot_y + s * x_robot + c * y_robot)


def field_to_robot(
    x_field: float,
    y_field: float,
    robot_x: float,
    robot_y: float,
    robot_theta: float,
) -> Tuple[float, float]:
    """Inverse of :func:`robot_to_field`."""
    dx, dy = x_field - robot_x, y_field - robot_y
    c, s = math.cos(robot_theta), math.sin(robot_theta)
    return (c * dx + s * dy, -s * dx + c * dy)


def rotate_vector(vx: float, vy: float, angle: float) -> Tuple[float, float]:
    """Rotate a 2D vector CCW by ``angle``."""
    c, s = math.cos(angle), math.sin(angle)
    return (c * vx - s * vy, s * vx + c * vy)


def segment_point_distance(
    ax: float, ay: float, bx: float, by: float, px: float, py: float
) -> float:
    """Distance from point P to the SEGMENT AB (not the infinite line)."""
    abx, aby = bx - ax, by - ay
    length_squared = abx * abx + aby * aby
    if length_squared < 1e-12:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * abx + (py - ay) * aby) / length_squared
    t = max(0.0, min(1.0, t))
    return math.hypot(px - (ax + t * abx), py - (ay + t * aby))


def clamp(value: float, low: float, high: float) -> float:
    return low if value < low else (high if value > high else value)


__all__ = [
    "CV_TO_FORWARD",
    "angle_difference",
    "apply_distance_polynomial",
    "camera_to_robot_rotation",
    "camera_to_robot_transform",
    "clamp",
    "estimate_ground_object",
    "field_to_robot",
    "ground_intersection",
    "invert_transform",
    "make_transform",
    "pinhole_width_distance",
    "pixel_to_camera_ray",
    "pixel_to_robot_ray",
    "quaternion_to_rotation",
    "robot_to_field",
    "rot_x",
    "rot_y",
    "rot_z",
    "rotate_vector",
    "segment_point_distance",
    "tilt_from_rotation",
    "transform_point",
    "undistort_pixel",
    "wrap_angle",
    "yaw_from_rotation",
]
