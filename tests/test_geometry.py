"""Unit tests for the pure ranging/transform math (no hardware needed)."""

import math

import pytest

from frc_perception.config import CameraConfig
from frc_perception.geometry import (
    estimate_object_position,
    field_to_robot,
    ground_intersection,
    robot_to_field,
    segment_point_distance,
    undistort_pixel,
    width_range,
    wrap_angle,
)


def make_camera(**overrides) -> CameraConfig:
    """640x480 pinhole camera, fx=fy=600, centered principal point."""
    values = dict(
        name="test", device_index=0, width=640, height=480,
        fx=600.0, fy=600.0, cx=320.0, cy=240.0,
        mount_x=0.0, mount_y=0.0, mount_z=0.5,
        mount_pitch_deg=0.0, mount_yaw_deg=0.0, mount_roll_deg=0.0,
    )
    values.update(overrides)
    return CameraConfig(**values)


class TestGroundIntersection:
    def test_45_down_center_pixel_hits_at_height_distance(self):
        # Camera 0.5 m up, tilted 45 deg down: the optical axis hits the
        # floor exactly 0.5 m ahead.
        cam = make_camera(mount_pitch_deg=45.0)
        hit = ground_intersection(320.0, 240.0, cam)
        assert hit == pytest.approx([0.5, 0.0, 0.0], abs=1e-9)

    def test_yawed_camera_hits_to_the_left(self):
        cam = make_camera(mount_pitch_deg=45.0, mount_yaw_deg=90.0)
        hit = ground_intersection(320.0, 240.0, cam)
        assert hit == pytest.approx([0.0, 0.5, 0.0], abs=1e-9)

    def test_mount_offset_shifts_hit(self):
        cam = make_camera(mount_pitch_deg=45.0, mount_x=0.3, mount_y=-0.1)
        hit = ground_intersection(320.0, 240.0, cam)
        assert hit == pytest.approx([0.8, -0.1, 0.0], abs=1e-9)

    def test_above_horizon_returns_none(self):
        # Level camera, pixel above center -> ray goes up, never lands.
        cam = make_camera(mount_pitch_deg=0.0)
        assert ground_intersection(320.0, 100.0, cam) is None

    def test_pixel_below_center_of_level_camera_lands(self):
        cam = make_camera(mount_pitch_deg=0.0)
        # Pixel 240 px below center: ray slope 240/600 -> lands at
        # 0.5 * 600/240 = 1.25 m ahead.
        hit = ground_intersection(320.0, 480.0, cam)
        assert hit == pytest.approx([1.25, 0.0, 0.0], abs=1e-9)


class TestWidthRange:
    def test_known_geometry(self):
        cam = make_camera()
        # 0.15 m object spanning 60 px at fx=600 -> 1.5 m slant range.
        assert width_range(60.0, cam, 0.15) == pytest.approx(1.5)

    def test_degenerate_width_is_none(self):
        assert width_range(0.5, make_camera(), 0.15) is None


class TestEstimateObjectPosition:
    def test_matches_ground_intersection_with_identity_poly(self):
        cam = make_camera(mount_pitch_deg=45.0)
        est = estimate_object_position((300.0, 220.0, 340.0, 240.0), cam, None)
        hit = ground_intersection(320.0, 240.0, cam)
        assert est is not None
        assert (est.x, est.y) == pytest.approx((hit[0], hit[1]), abs=1e-9)

    def test_poly_rescales_along_bearing(self):
        cam = make_camera(mount_pitch_deg=45.0, distance_poly=(2.0, 0.0))
        est = estimate_object_position((300.0, 220.0, 340.0, 240.0), cam, None)
        assert (est.x, est.y) == pytest.approx((1.0, 0.0), abs=1e-9)
        assert est.ground_distance_m == pytest.approx(1.0)

    def test_cross_check_flags_disagreement(self):
        cam = make_camera(mount_pitch_deg=45.0)
        # Ground range says ~0.5 m but a 10 px wide "0.15 m" object implies
        # ~9 m slant range: the width cross-check must flag it.
        est = estimate_object_position((315.0, 220.0, 325.0, 240.0), cam, 0.15)
        assert est is not None and not est.cross_check_ok

    def test_cross_check_passes_when_consistent(self):
        cam = make_camera(mount_pitch_deg=45.0)
        # Ground distance 0.5 m, camera height 0.5 -> slant ~0.707 m.
        # A 0.15 m object at that slant spans 600*0.15/0.707 ~ 127 px.
        est = estimate_object_position((256.0, 200.0, 383.0, 240.0), cam, 0.15)
        assert est is not None and est.cross_check_ok

    def test_airborne_object_returns_none(self):
        cam = make_camera(mount_pitch_deg=0.0)
        assert estimate_object_position((300.0, 80.0, 340.0, 100.0), cam, None) is None


class TestFrames:
    def test_robot_to_field_known_case(self):
        # Robot at (5, 3) facing +Y (90 deg): 1 m forward is +Y in field.
        assert robot_to_field(1.0, 0.0, 5.0, 3.0, math.pi / 2) == pytest.approx((5.0, 4.0))

    def test_roundtrip(self):
        pose = (3.2, -1.1, 0.7)
        fx, fy = robot_to_field(1.5, -0.4, *pose)
        assert field_to_robot(fx, fy, *pose) == pytest.approx((1.5, -0.4))

    def test_wrap_angle(self):
        assert wrap_angle(3 * math.pi) == pytest.approx(math.pi)
        assert wrap_angle(-3 * math.pi) == pytest.approx(math.pi)
        assert wrap_angle(0.3) == pytest.approx(0.3)


class TestUndistort:
    def test_zero_coeffs_is_pinhole(self):
        x, y = undistort_pixel(400.0, 300.0, 600.0, 600.0, 320.0, 240.0, (0, 0, 0, 0, 0))
        assert (x, y) == pytest.approx((80.0 / 600.0, 60.0 / 600.0))

    def test_inverts_forward_distortion(self):
        fx, fy, cx, cy = 600.0, 600.0, 320.0, 240.0
        k = (-0.3, 0.1, 0.001, -0.002, 0.0)
        x_true, y_true = 0.25, -0.15
        # Apply the OpenCV forward model, then invert it.
        r2 = x_true**2 + y_true**2
        radial = 1 + k[0] * r2 + k[1] * r2**2 + k[4] * r2**3
        xd = x_true * radial + 2 * k[2] * x_true * y_true + k[3] * (r2 + 2 * x_true**2)
        yd = y_true * radial + k[2] * (r2 + 2 * y_true**2) + 2 * k[3] * x_true * y_true
        x, y = undistort_pixel(xd * fx + cx, yd * fy + cy, fx, fy, cx, cy, k)
        assert (x, y) == pytest.approx((x_true, y_true), abs=1e-6)


def test_segment_point_distance():
    assert segment_point_distance(0, 0, 4, 0, 2, 3) == pytest.approx(3.0)
    assert segment_point_distance(0, 0, 4, 0, -3, 4) == pytest.approx(5.0)  # past endpoint A
    assert segment_point_distance(1, 1, 1, 1, 4, 5) == pytest.approx(5.0)  # zero-length
