"""Pure-math geometry: rays, ground-plane ranging, and frame transforms.

Everything here is numpy-only (no OpenCV, no hardware) so it can be unit
tested anywhere. Conventions match the package docstring:

  Robot frame:   +X forward, +Y left, +Z up. Yaw CCW-positive.
  Camera (CV):   +X right, +Y down, +Z out the lens (OpenCV standard).
  Mount pitch:   positive = camera tilted DOWN toward the floor.
"""

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from .config import CameraConfig


# ---------------------------------------------------------------------------
# Rotations and rigid transforms
# ---------------------------------------------------------------------------

def rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


def make_transform(rotation: np.ndarray, translation: Sequence[float]) -> np.ndarray:
    """Build a 4x4 homogeneous transform from a 3x3 R and a 3-vector t."""
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def invert_transform(transform: np.ndarray) -> np.ndarray:
    """Invert a rigid 4x4 transform without a general matrix inverse."""
    rotation = transform[:3, :3]
    inverse = np.eye(4)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ transform[:3, 3]
    return inverse


def quat_to_rot(w: float, x: float, y: float, z: float) -> np.ndarray:
    """Unit quaternion (WPILib W,X,Y,Z order) to a 3x3 rotation matrix."""
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n == 0:
        raise ValueError("zero-norm quaternion")
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def wrap_angle(a: float) -> float:
    """Wrap an angle to (-pi, pi]."""
    a = math.fmod(a + math.pi, 2.0 * math.pi)
    if a <= 0.0:
        a += 2.0 * math.pi
    return a - math.pi


def yaw_from_rotation(rotation: np.ndarray) -> float:
    """Extract Z-axis yaw from a 3x3 rotation matrix."""
    return math.atan2(rotation[1, 0], rotation[0, 0])


# Fixed basis change: camera CV axes (x right, y down, z forward) expressed
# in a "forward-aligned" frame (x forward, y left, z up).
CV_TO_FWD = np.array(
    [
        [0.0, 0.0, 1.0],   # cam +Z (out the lens)  -> +X forward
        [-1.0, 0.0, 0.0],  # cam +X (image right)   -> -Y (right)
        [0.0, -1.0, 0.0],  # cam +Y (image down)    -> -Z (down)
    ],
    dtype=float,
)


def camera_to_robot_rotation(cam: CameraConfig) -> np.ndarray:
    """3x3 rotation taking camera-CV coordinates to robot coordinates.

    Mount rotation order is yaw (about robot Z), then pitch (about the
    panned Y axis, positive = down), then roll about the lens axis.
    """
    yaw = math.radians(cam.mount_yaw_deg)
    pitch = math.radians(cam.mount_pitch_deg)
    roll = math.radians(cam.mount_roll_deg)
    return rot_z(yaw) @ rot_y(pitch) @ rot_x(roll) @ CV_TO_FWD


def camera_to_robot_transform(cam: CameraConfig) -> np.ndarray:
    """4x4 transform of the camera (CV frame) in the robot frame."""
    return make_transform(
        camera_to_robot_rotation(cam), (cam.mount_x, cam.mount_y, cam.mount_z)
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
    """Invert OpenCV radial-tangential distortion (k1,k2,p1,p2,k3).

    Returns ideal normalized image coordinates (x', y'). Iterative
    fixed-point inversion; exact for zero coefficients.
    """
    xd = (px - cx) / fx
    yd = (py - cy) / fy
    k1, k2, p1, p2, k3 = (list(dist) + [0.0] * 5)[:5]
    if k1 == k2 == p1 == p2 == k3 == 0.0:
        return xd, yd
    x, y = xd, yd
    for _ in range(iterations):
        r2 = x * x + y * y
        radial = 1.0 + r2 * (k1 + r2 * (k2 + r2 * k3))
        dx = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
        dy = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
        x = (xd - dx) / radial
        y = (yd - dy) / radial
    return x, y


def pixel_to_ray_cv(px: float, py: float, cam: CameraConfig) -> np.ndarray:
    """Unit ray in the camera-CV frame through pixel (px, py)."""
    fx, fy, cx, cy = cam.effective_intrinsics()
    x, y = undistort_pixel(px, py, fx, fy, cx, cy, cam.dist_coeffs)
    ray = np.array([x, y, 1.0])
    return ray / np.linalg.norm(ray)


# ---------------------------------------------------------------------------
# Ranging
# ---------------------------------------------------------------------------

@dataclass
class ObjectEstimate:
    """A ranged detection in the robot frame (+X fwd, +Y left, meters)."""

    x: float
    y: float
    ground_distance_m: float     # camera->object distance in the ground plane, after poly
    cross_check_ok: bool         # width-based range agreed with the angle range
    width_distance_m: Optional[float]  # slant range from box width, None if unusable


def ground_intersection(
    px: float, py: float, cam: CameraConfig, object_z: float = 0.0
) -> Optional[np.ndarray]:
    """Intersect the ray through a pixel with the horizontal plane z=object_z.

    Returns the hit point [x, y, z] in the ROBOT frame, or None when the
    ray points at/above the horizon (no ground hit ahead of the camera).
    """
    ray = camera_to_robot_rotation(cam) @ pixel_to_ray_cv(px, py, cam)
    origin = np.array([cam.mount_x, cam.mount_y, cam.mount_z])
    dz = ray[2]
    if dz >= -1e-6:  # not descending toward the floor
        return None
    t = (object_z - origin[2]) / dz
    if t <= 0:
        return None
    return origin + t * ray


def width_range(pixel_width: float, cam: CameraConfig, true_width_m: float) -> Optional[float]:
    """Slant range from apparent width of an object of known real width."""
    if pixel_width <= 1.0:
        return None
    fx, _, _, _ = cam.effective_intrinsics()
    return fx * true_width_m / pixel_width


def estimate_object_position(
    bbox_xyxy: Sequence[float],
    cam: CameraConfig,
    known_width_m: Optional[float],
    cross_check_tolerance: float = 0.35,
) -> Optional[ObjectEstimate]:
    """Range a detection: angle-to-ground primary, width cross-check, poly.

    The bottom-center of the box is assumed to touch the floor. The raw
    camera->object ground distance is corrected by the per-camera
    `distance_poly`, preserving bearing. Returns None if the geometry is
    unusable (above the horizon — e.g. an airborne or misdetected object).
    """
    x1, y1, x2, y2 = bbox_xyxy
    hit = ground_intersection((x1 + x2) / 2.0, y2, cam)
    if hit is None:
        return None

    cam_ground = np.array([cam.mount_x, cam.mount_y])
    offset = hit[:2] - cam_ground
    raw_dist = float(np.linalg.norm(offset))
    if raw_dist < 1e-6:
        return None

    width_dist = None
    cross_ok = True
    if known_width_m is not None:
        slant = width_range(x2 - x1, cam, known_width_m)
        if slant is not None:
            # Convert slant range to a ground-plane distance for comparison.
            width_dist = math.sqrt(max(slant * slant - cam.mount_z * cam.mount_z, 0.0))
            if raw_dist > 0.1 and abs(width_dist - raw_dist) / raw_dist > cross_check_tolerance:
                cross_ok = False

    corrected = float(np.polyval(np.asarray(cam.distance_poly, dtype=float), raw_dist))
    if corrected <= 0.0:
        corrected = raw_dist  # a broken poly must not produce a pose behind the lens
    point = cam_ground + offset * (corrected / raw_dist)
    return ObjectEstimate(
        x=float(point[0]),
        y=float(point[1]),
        ground_distance_m=corrected,
        cross_check_ok=cross_ok,
        width_distance_m=width_dist,
    )


# ---------------------------------------------------------------------------
# Robot <-> field (2D)
# ---------------------------------------------------------------------------

def robot_to_field(x: float, y: float, robot_x: float, robot_y: float, robot_theta: float) -> Tuple[float, float]:
    """Transform a robot-relative point into field coordinates."""
    c, s = math.cos(robot_theta), math.sin(robot_theta)
    return (robot_x + c * x - s * y, robot_y + s * x + c * y)


def field_to_robot(fx: float, fy: float, robot_x: float, robot_y: float, robot_theta: float) -> Tuple[float, float]:
    """Transform a field point into robot-relative coordinates."""
    dx, dy = fx - robot_x, fy - robot_y
    c, s = math.cos(robot_theta), math.sin(robot_theta)
    return (c * dx + s * dy, -s * dx + c * dy)


def segment_point_distance(
    ax: float, ay: float, bx: float, by: float, px: float, py: float
) -> float:
    """Distance from point P to segment AB. Used for path-blocking checks."""
    abx, aby = bx - ax, by - ay
    ab2 = abx * abx + aby * aby
    if ab2 < 1e-12:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * abx + (py - ay) * aby) / ab2))
    return math.hypot(px - (ax + t * abx), py - (ay + t * aby))
