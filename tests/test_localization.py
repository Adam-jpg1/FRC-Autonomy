"""Tests for odometry staleness, tag pose math, and fusion behavior."""

import math

import numpy as np
import pytest

from frc_perception.config import CameraConfig, LocalizationConfig
from frc_perception.geometry import (
    camera_to_robot_transform,
    invert_transform,
    make_transform,
    quat_to_rot,
    rot_z,
)
from frc_perception.localization import (
    _ATAG_TO_WPILIB_TAG,
    OdometryTracker,
    Pose2D,
    PoseFuser,
    compose,
    invert,
    robot_pose_from_tag,
)


class TestOdometryStaleness:
    """Staleness must key off the RIO's PUBLISHED timestamp advancing."""

    def test_advancing_timestamp_is_fresh(self):
        odom = OdometryTracker(stale_s=0.5)
        odom.update(1.0, 2.0, 0.1, rio_timestamp=100.0, now=10.0)
        odom.update(1.1, 2.0, 0.1, rio_timestamp=100.02, now=10.02)
        assert odom.fresh(now=10.1)
        assert odom.pose.x == pytest.approx(1.1)

    def test_frozen_timestamp_goes_stale_despite_updates(self):
        # RIO loop wedged: NT keeps delivering the SAME timestamp value.
        odom = OdometryTracker(stale_s=0.5)
        odom.update(1.0, 2.0, 0.1, rio_timestamp=100.0, now=10.0)
        for i in range(1, 20):
            odom.update(1.0, 2.0, 0.1, rio_timestamp=100.0, now=10.0 + i * 0.1)
        assert not odom.fresh(now=11.9)

    def test_never_updated_is_stale(self):
        assert not OdometryTracker(stale_s=0.5).fresh(now=0.0)


class TestTransform2D:
    def test_compose_with_inverse_is_identity(self):
        t = Pose2D(2.0, -1.0, 0.8)
        ident = compose(t, invert(t))
        assert (ident.x, ident.y, ident.theta) == pytest.approx((0, 0, 0), abs=1e-12)

    def test_quat_identity_and_z90(self):
        assert quat_to_rot(1, 0, 0, 0) == pytest.approx(np.eye(3))
        r = quat_to_rot(math.cos(math.pi / 4), 0, 0, math.sin(math.pi / 4))
        assert r @ np.array([1.0, 0, 0]) == pytest.approx([0, 1, 0], abs=1e-12)


class TestRobotPoseFromTag:
    def test_roundtrip_recovers_robot_pose(self):
        """Synthesize a tag observation from a known robot pose and check
        the solver inverts it exactly (self-consistency of all frame
        conventions used in the pipeline)."""
        cam = CameraConfig(
            name="t", device_index=0, fx=600.0, fy=600.0, cx=320.0, cy=240.0,
            mount_x=0.2, mount_y=0.1, mount_z=0.4,
            mount_pitch_deg=10.0, mount_yaw_deg=15.0,
        )
        robot_cam = camera_to_robot_transform(cam)
        true_pose = Pose2D(3.0, 2.0, math.radians(30.0))
        field_robot = make_transform(rot_z(true_pose.theta), (true_pose.x, true_pose.y, 0.0))
        # Tag on the "red wall" facing -X, 1 m up.
        field_tag_wpilib = make_transform(rot_z(math.pi), (10.0, 3.0, 1.0))
        # What pupil-apriltags would report: the tag in ITS convention,
        # seen from the camera.
        field_tag_atag = field_tag_wpilib @ make_transform(_ATAG_TO_WPILIB_TAG, (0, 0, 0))
        cam_tag_atag = invert_transform(field_robot @ robot_cam) @ field_tag_atag

        pose, z, tilt = robot_pose_from_tag(
            field_tag_wpilib, cam_tag_atag[:3, :3], cam_tag_atag[:3, 3], robot_cam
        )
        assert (pose.x, pose.y) == pytest.approx((true_pose.x, true_pose.y), abs=1e-9)
        assert pose.theta == pytest.approx(true_pose.theta, abs=1e-9)
        assert z == pytest.approx(0.0, abs=1e-9)
        assert tilt == pytest.approx(0.0, abs=1e-6)


class TestPoseFuser:
    def make(self, **overrides) -> PoseFuser:
        cfg = LocalizationConfig(**overrides)
        return PoseFuser(cfg)

    def fresh_odom(self, x=1.0, y=1.0, theta=0.0, now=100.0) -> OdometryTracker:
        odom = OdometryTracker(stale_s=0.5)
        odom.update(x, y, theta, rio_timestamp=now, now=now)
        return odom

    def test_identity_correction_passes_odometry_through(self):
        fuser = self.make()
        odom = self.fresh_odom(2.0, 3.0, 0.5)
        est = fuser.estimate(odom, now=100.0)
        assert est.valid and est.source == "fused"
        assert (est.x, est.y, est.theta) == pytest.approx((2.0, 3.0, 0.5))

    def test_correction_converges_to_tag_pose(self):
        fuser = self.make(fusion_alpha=0.5)
        odom = self.fresh_odom(1.0, 1.0, 0.0)
        tag = Pose2D(2.0, 1.5, 0.1)  # odometry is off by a constant offset
        for i in range(40):
            fuser.update_from_tag(tag, odom, now=100.0 + i * 0.03)
        est = fuser.estimate(odom, now=101.5)
        assert (est.x, est.y) == pytest.approx((tag.x, tag.y), abs=1e-3)
        assert est.theta == pytest.approx(tag.theta, abs=1e-3)

    def test_stale_odom_no_tags_is_invalid(self):
        fuser = self.make()
        odom = self.fresh_odom(now=100.0)
        est = fuser.estimate(odom, now=200.0)  # long after odometry died
        assert not est.valid and est.source == "none"

    def test_tag_only_fallback_then_expiry(self):
        fuser = self.make(tag_only_timeout_s=1.0)
        dead_odom = OdometryTracker(stale_s=0.5)  # never updated
        fuser.update_from_tag(Pose2D(4.0, 4.0, 0.2), dead_odom, now=100.0)
        est = fuser.estimate(dead_odom, now=100.5)
        assert est.valid and est.source == "tag_only"
        assert (est.x, est.y) == pytest.approx((4.0, 4.0))
        assert not fuser.estimate(dead_odom, now=101.5).valid
