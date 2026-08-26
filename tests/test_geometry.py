"""Geometry: ray-to-ground ranging, rejections, and frame transforms.

These tests use camera configurations with EXACT, hand-chosen mount
values so the expected answer can be computed on paper. They are not
measurements of the real robot -- they are checks that the math is the
math we think it is.
"""

import math

import numpy as np
import pytest

from frc_perception.config import CameraConfig, RangingConfig
from frc_perception.geometry import (
    apply_distance_polynomial,
    camera_to_robot_rotation,
    estimate_ground_object,
    field_to_robot,
    ground_intersection,
    invert_transform,
    camera_to_robot_transform,
    pinhole_width_distance,
    robot_to_field,
    rotate_vector,
    segment_point_distance,
    undistort_pixel,
    wrap_angle,
)
from frc_perception.types import RangeRejection


def make_camera(**overrides) -> CameraConfig:
    """A camera with clean, exactly-known intrinsics and mounting.

    fx = fy = 800 px, principal point at the image center, 1 m above the
    floor, pitched 30 degrees down, no yaw or roll.
    """
    defaults = dict(
        name="test",
        width=1280,
        height=800,
        fx=800.0,
        fy=800.0,
        cx=640.0,
        cy=400.0,
        mount_x_m=0.0,
        mount_y_m=0.0,
        mount_z_m=1.0,
        pitch_deg=30.0,
        yaw_deg=0.0,
        roll_deg=0.0,
    )
    defaults.update(overrides)
    return CameraConfig(**defaults)


RANGING = RangingConfig()


# ---------------------------------------------------------------------------
# Rotations and mounting
# ---------------------------------------------------------------------------


def test_camera_rotation_is_orthonormal_and_right_handed():
    rotation = camera_to_robot_rotation(make_camera(pitch_deg=17.0, yaw_deg=-40.0, roll_deg=5.0))
    assert np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)
    assert pytest.approx(1.0, abs=1e-12) == float(np.linalg.det(rotation))


def test_level_camera_optical_axis_points_forward():
    rotation = camera_to_robot_rotation(make_camera(pitch_deg=0.0))
    # Camera +Z (out of the lens) must land on robot +X (forward).
    assert np.allclose(rotation @ np.array([0.0, 0.0, 1.0]), [1.0, 0.0, 0.0], atol=1e-12)


def test_positive_pitch_tilts_the_camera_down():
    rotation = camera_to_robot_rotation(make_camera(pitch_deg=30.0))
    axis = rotation @ np.array([0.0, 0.0, 1.0])
    assert axis[2] < 0.0, "positive pitch must point the optical axis toward the floor"
    assert pytest.approx(-math.sin(math.radians(30.0)), abs=1e-12) == axis[2]


def test_positive_yaw_points_the_camera_left():
    rotation = camera_to_robot_rotation(make_camera(pitch_deg=0.0, yaw_deg=90.0))
    axis = rotation @ np.array([0.0, 0.0, 1.0])
    assert np.allclose(axis, [0.0, 1.0, 0.0], atol=1e-12), "+yaw is CCW (to the left)"


def test_camera_transform_inverts_cleanly():
    transform = camera_to_robot_transform(make_camera(mount_x_m=0.3, mount_y_m=-0.1))
    assert np.allclose(transform @ invert_transform(transform), np.eye(4), atol=1e-12)


# ---------------------------------------------------------------------------
# Ray to ground
# ---------------------------------------------------------------------------


def test_ground_intersection_matches_hand_computed_distance():
    """A camera 1 m up pitched 30 deg down: the image center hits the floor
    at 1 / tan(30 deg) = 1.732 m in front of it."""
    cam = make_camera(mount_z_m=1.0, pitch_deg=30.0)
    hit = ground_intersection(cam.cx, cam.cy, cam, 0.0, RANGING.min_ray_descent)
    assert hit is not None
    assert pytest.approx(1.0 / math.tan(math.radians(30.0)), abs=1e-9) == hit[0]
    assert pytest.approx(0.0, abs=1e-9) == hit[1]
    assert pytest.approx(0.0, abs=1e-12) == hit[2]


def test_ground_intersection_scales_with_camera_height():
    near = ground_intersection(640.0, 400.0, make_camera(mount_z_m=0.5), 0.0, 0.02)
    far = ground_intersection(640.0, 400.0, make_camera(mount_z_m=1.5), 0.0, 0.02)
    assert near is not None and far is not None
    assert pytest.approx(3.0, rel=1e-9) == far[0] / near[0]


def test_ray_above_the_horizon_is_rejected():
    """A pixel above the horizon line never touches the ground.

    With fy=800 and cy=400 the top image row sits atan(400/800) = 26.57 deg
    above the optical axis, so a 20 deg downward pitch leaves it 6.57 deg
    ABOVE horizontal.
    """
    cam = make_camera(pitch_deg=20.0)
    assert ground_intersection(640.0, 0.0, cam, 0.0, RANGING.min_ray_descent) is None
    # ... while a pixel below the center still hits the floor.
    assert ground_intersection(640.0, 600.0, cam, 0.0, RANGING.min_ray_descent) is not None


def test_near_horizon_ray_is_rejected_by_the_descent_guard():
    """A nearly-flat ray technically intersects but the range is meaningless."""
    cam = make_camera(pitch_deg=0.5)
    assert ground_intersection(640.0, 400.0, cam, 0.0, min_descent=0.05) is None
    # With the guard relaxed it does intersect -- the guard, not the geometry,
    # is what rejects it.
    assert ground_intersection(640.0, 400.0, cam, 0.0, min_descent=1e-9) is not None


def test_camera_below_the_plane_cannot_range():
    cam = make_camera(mount_z_m=0.0)
    assert ground_intersection(640.0, 500.0, cam, 0.0, RANGING.min_ray_descent) is None


def test_bearing_sign_follows_the_left_positive_convention():
    cam = make_camera()
    left_of_center = estimate_ground_object((100.0, 500.0, 140.0, 540.0), cam, RANGING)
    right_of_center = estimate_ground_object((1140.0, 500.0, 1180.0, 540.0), cam, RANGING)
    assert left_of_center.valid and right_of_center.valid
    assert left_of_center.bearing_rad > 0.0, "left of image center must be +Y (left)"
    assert right_of_center.bearing_rad < 0.0


# ---------------------------------------------------------------------------
# estimate_ground_object rejections
# ---------------------------------------------------------------------------


def test_object_above_the_horizon_returns_invalid_not_a_guess():
    cam = make_camera(pitch_deg=20.0)
    result = estimate_ground_object((600.0, 20.0, 680.0, 60.0), cam, RANGING)
    assert not result.valid
    assert result.rejection is RangeRejection.ABOVE_HORIZON
    assert result.range_m == 0.0


def test_too_far_is_rejected():
    cam = make_camera(pitch_deg=30.0)
    ranging = RangingConfig(max_range_m=1.0)
    result = estimate_ground_object((620.0, 400.0, 660.0, 410.0), cam, ranging)
    assert not result.valid and result.rejection is RangeRejection.TOO_FAR


def test_too_near_is_rejected():
    cam = make_camera(pitch_deg=30.0)
    ranging = RangingConfig(min_range_m=5.0)
    result = estimate_ground_object((620.0, 700.0, 660.0, 780.0), cam, ranging)
    assert not result.valid and result.rejection is RangeRejection.TOO_NEAR


def test_box_clipped_by_the_frame_edge_is_rejected():
    """No visible ground contact point means no usable range."""
    cam = make_camera()
    result = estimate_ground_object((600.0, 700.0, 680.0, 800.0), cam, RANGING)
    assert not result.valid
    assert result.rejection is RangeRejection.OUTSIDE_USABLE_IMAGE


def test_degenerate_box_is_rejected():
    cam = make_camera()
    result = estimate_ground_object((600.0, 500.0, 600.0, 500.0), cam, RANGING)
    assert not result.valid


def test_uncalibrated_camera_still_ranges_via_the_fov_fallback():
    cam = CameraConfig(name="raw", mount_z_m=1.0, pitch_deg=30.0, horizontal_fov_deg=70.0)
    result = estimate_ground_object((620.0, 500.0, 660.0, 520.0), cam, RANGING)
    assert result.valid and result.range_m > 0.0


# ---------------------------------------------------------------------------
# Width cross-check
# ---------------------------------------------------------------------------


def test_pinhole_width_distance_is_the_textbook_formula():
    cam = make_camera(fx=800.0)
    assert pytest.approx(4.0) == pinhole_width_distance(50.0, cam, 0.25)
    assert pinhole_width_distance(0.0, cam, 0.25) is None


def test_agreeing_width_estimate_keeps_full_quality():
    """Choose a box whose width implies the same distance the geometry does."""
    cam = make_camera(mount_z_m=1.0, pitch_deg=30.0)
    geometric = estimate_ground_object((630.0, 495.0, 650.0, 505.0), cam, RANGING)
    assert geometric.valid
    slant = math.hypot(geometric.range_m, cam.mount_z_m)
    true_width = slant * (650.0 - 630.0) / cam.fx  # exactly consistent
    checked = estimate_ground_object(
        (630.0, 495.0, 650.0, 505.0), cam, RANGING, known_width_m=true_width
    )
    assert checked.valid
    assert checked.quality == 1.0
    assert checked.width_disagreement < 1e-6


def test_wildly_disagreeing_width_estimate_rejects_the_measurement():
    cam = make_camera()
    result = estimate_ground_object(
        (630.0, 495.0, 650.0, 505.0), cam, RANGING, known_width_m=5.0
    )
    assert not result.valid
    assert result.rejection is RangeRejection.WIDTH_DISAGREEMENT
    assert result.width_disagreement is not None


def test_moderately_disagreeing_width_only_degrades_quality():
    cam = make_camera(mount_z_m=1.0, pitch_deg=30.0)
    geometric = estimate_ground_object((630.0, 495.0, 650.0, 505.0), cam, RANGING)
    slant = math.hypot(geometric.range_m, cam.mount_z_m)
    consistent_width = slant * 20.0 / cam.fx
    # 40% off: past the warn ratio (0.30) but inside the reject ratio (0.60).
    result = estimate_ground_object(
        (630.0, 495.0, 650.0, 505.0), cam, RANGING, known_width_m=consistent_width * 1.4
    )
    assert result.valid
    assert result.quality == RANGING.width_warn_quality


# ---------------------------------------------------------------------------
# Distance polynomial
# ---------------------------------------------------------------------------


def test_identity_polynomial_changes_nothing():
    cam = make_camera(distance_poly=(1.0, 0.0))
    assert pytest.approx(2.5) == apply_distance_polynomial(cam, 2.5)


def test_polynomial_is_held_constant_past_its_fitted_range():
    """Beyond the fitted range the OFFSET is held, not extrapolated."""
    cam = make_camera(distance_poly=(1.2, 0.0), distance_poly_max_m=3.0)
    assert pytest.approx(3.6) == apply_distance_polynomial(cam, 3.0)
    # 5 m is 2 m past the fit: 3.6 + 2.0, not 6.0.
    assert pytest.approx(5.6) == apply_distance_polynomial(cam, 5.0)


def test_broken_polynomial_falls_back_to_raw_distance():
    """A calibration that produces a negative range must degrade, not lie."""
    cam = make_camera(distance_poly=(-1.0, 0.0))
    assert pytest.approx(2.0) == apply_distance_polynomial(cam, 2.0)


def test_polynomial_shifts_range_but_not_bearing():
    plain = make_camera()
    scaled = make_camera(distance_poly=(1.5, 0.0))
    bbox = (700.0, 500.0, 740.0, 520.0)
    a = estimate_ground_object(bbox, plain, RANGING)
    b = estimate_ground_object(bbox, scaled, RANGING)
    assert a.valid and b.valid
    assert pytest.approx(a.bearing_rad, abs=1e-12) == b.bearing_rad
    assert b.range_m > a.range_m


# ---------------------------------------------------------------------------
# Robot <-> field
# ---------------------------------------------------------------------------


def test_robot_to_field_with_zero_pose_is_a_translation():
    assert pytest.approx((3.0, 4.0)) == robot_to_field(3.0, 4.0, 0.0, 0.0, 0.0)


def test_robot_to_field_rotates_by_the_heading():
    # Robot at the origin facing +Y (90 deg): its "forward" is field +Y.
    x, y = robot_to_field(2.0, 0.0, 0.0, 0.0, math.pi / 2)
    assert pytest.approx(0.0, abs=1e-12) == x
    assert pytest.approx(2.0, abs=1e-12) == y


def test_field_to_robot_is_the_exact_inverse():
    x, y = field_to_robot(5.0, 2.0, 1.0, 1.0, 0.7)
    back = robot_to_field(x, y, 1.0, 1.0, 0.7)
    assert pytest.approx((5.0, 2.0)) == back


@pytest.mark.parametrize(
    "point,pose",
    [
        ((1.0, 2.0), (0.0, 0.0, 0.0)),
        ((-3.5, 0.25), (8.0, 4.0, math.pi / 3)),
        ((0.0, 0.0), (16.0, 8.0, -2.9)),
        ((7.0, -7.0), (2.5, 1.5, math.pi)),
    ],
)
def test_transform_round_trip(point, pose):
    field = robot_to_field(point[0], point[1], *pose)
    back = field_to_robot(field[0], field[1], *pose)
    assert pytest.approx(point[0], abs=1e-9) == back[0]
    assert pytest.approx(point[1], abs=1e-9) == back[1]


def test_rotate_vector_matches_a_hand_rotation():
    x, y = rotate_vector(1.0, 0.0, math.pi / 2)
    assert pytest.approx(0.0, abs=1e-12) == x
    assert pytest.approx(1.0, abs=1e-12) == y


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "angle,expected",
    [(0.0, 0.0), (math.pi, math.pi), (-math.pi, math.pi), (3 * math.pi, math.pi), (2 * math.pi, 0.0)],
)
def test_wrap_angle(angle, expected):
    assert pytest.approx(expected, abs=1e-12) == wrap_angle(angle)


def test_segment_point_distance_clamps_to_the_endpoints():
    # Perpendicular from above the middle of the segment.
    assert pytest.approx(1.0) == segment_point_distance(0, 0, 10, 0, 5, 1)
    # Past the end: distance to the endpoint, not to the infinite line.
    assert pytest.approx(5.0) == segment_point_distance(0, 0, 10, 0, 15, 0)
    # Degenerate segment.
    assert pytest.approx(5.0) == segment_point_distance(0, 0, 0, 0, 3, 4)


def test_undistort_is_identity_without_coefficients():
    assert (0.5, -0.25) == undistort_pixel(
        800.0 * 0.5 + 640.0, 800.0 * -0.25 + 400.0, 800.0, 800.0, 640.0, 400.0, (0, 0, 0, 0, 0)
    )


def test_undistort_inverts_a_known_distortion():
    fx = fy = 800.0
    cx = cy = 400.0
    coefficients = (-0.25, 0.08, 0.001, -0.002, 0.0)
    ideal_x, ideal_y = 0.20, -0.15
    r2 = ideal_x**2 + ideal_y**2
    k1, k2, p1, p2, k3 = coefficients
    radial = 1.0 + r2 * (k1 + r2 * (k2 + r2 * k3))
    distorted_x = ideal_x * radial + 2 * p1 * ideal_x * ideal_y + p2 * (r2 + 2 * ideal_x**2)
    distorted_y = ideal_y * radial + p1 * (r2 + 2 * ideal_y**2) + 2 * p2 * ideal_x * ideal_y
    recovered = undistort_pixel(
        distorted_x * fx + cx, distorted_y * fy + cy, fx, fy, cx, cy, coefficients
    )
    assert pytest.approx(ideal_x, abs=1e-6) == recovered[0]
    assert pytest.approx(ideal_y, abs=1e-6) == recovered[1]
