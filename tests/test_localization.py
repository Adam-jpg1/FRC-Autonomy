"""Localization: odometry freshness, tag fusion, disagreement, validity.

The AprilTag frame-convention tests here are the paper proof that the
transform chain in ``apriltag_locator.py`` is self-consistent. They do
NOT prove the mount angles on the real robot are right -- only
``scripts/verify_localization.py`` on a real field can do that.
"""

import math

import numpy as np
import pytest

from frc_perception.apriltag_locator import (
    aggregate_tag_poses,
    circular_mean,
    pose2d_from_transform,
    robot_pose_from_tag,
)
from frc_perception.config import CameraConfig, LocalizationConfig
from frc_perception.field_map import FieldLayoutError, FieldMap, parse_field_layout
from frc_perception.geometry import camera_to_robot_transform, make_transform, rot_z
from frc_perception.localization import (
    OdometryMonitor,
    PoseEstimator,
    SE2,
    pose_difference,
    se2_apply,
    se2_correction_between,
)
from frc_perception.types import (
    LocalizationSource,
    Pose2d,
    TagObservation,
    TagPoseEstimate,
)


def make_field_map(length=16.0, width=8.0) -> FieldMap:
    return FieldMap(tags={}, length_m=length, width_m=width, source_path="<test>")


def tag_estimate(x, y, theta, timestamp=0.0, quality=1.0, ids=(1,)) -> TagPoseEstimate:
    return TagPoseEstimate(
        pose=Pose2d(x, y, theta),
        timestamp=timestamp,
        tag_ids=list(ids),
        quality=quality,
        mean_distance_m=2.0,
    )


# ===========================================================================
# Odometry freshness -- the failure this project cares most about
# ===========================================================================


def test_advancing_rio_timestamp_is_fresh():
    monitor = OdometryMonitor(LocalizationConfig(odometry_stale_timeout_s=0.3))
    monitor.update(Pose2d(1, 2, 0), rio_timestamp=100.0, now=10.0)
    monitor.update(Pose2d(1, 2, 0), rio_timestamp=100.02, now=10.02)
    assert monitor.is_fresh(10.05)


def test_frozen_rio_timestamp_goes_stale():
    monitor = OdometryMonitor(LocalizationConfig(odometry_stale_timeout_s=0.3))
    monitor.update(Pose2d(1, 2, 0), 100.0, now=10.0)
    assert monitor.is_fresh(10.1)
    monitor.update(Pose2d(1, 2, 0), 100.0, now=10.2)
    assert monitor.is_fresh(10.2)
    monitor.update(Pose2d(1, 2, 0), 100.0, now=10.4)
    assert not monitor.is_fresh(10.4), "timestamp stopped advancing: must be stale"


def test_staleness_is_independent_of_how_often_we_read():
    """Hammering the table with reads must not manufacture freshness.

    This is the exact bug the timestamp rule exists to prevent: NT keeps
    serving the last published value forever after the publisher dies.
    """
    monitor = OdometryMonitor(LocalizationConfig(odometry_stale_timeout_s=0.3))
    monitor.update(Pose2d(1, 2, 0), 100.0, now=0.0)
    for index in range(1, 1001):
        monitor.update(Pose2d(1, 2, 0), 100.0, now=index * 0.001)
    assert not monitor.is_fresh(1.0)
    assert monitor.age_s(1.0) == pytest.approx(1.0)


def test_zero_valued_timestamp_still_counts_as_advancing():
    """0.0 is a legitimate timestamp, not "missing"."""
    monitor = OdometryMonitor(LocalizationConfig(odometry_stale_timeout_s=0.3))
    monitor.update(Pose2d(0, 0, 0), 0.0, now=5.0)
    monitor.update(Pose2d(0, 0, 0), 0.02, now=5.02)
    assert monitor.is_fresh(5.1)


def test_backward_rio_timestamp_counts_as_alive_and_is_recorded():
    monitor = OdometryMonitor(LocalizationConfig())
    monitor.update(Pose2d(1, 1, 0), 500.0, now=1.0)
    monitor.update(Pose2d(0, 0, 0), 0.5, now=1.1)
    assert monitor.backward_jumps == 1
    assert monitor.is_fresh(1.15)


def test_non_finite_odometry_is_ignored():
    monitor = OdometryMonitor(LocalizationConfig())
    monitor.update(Pose2d(1.0, 2.0, 0.0), 10.0, now=1.0)
    monitor.update(Pose2d(float("nan"), 2.0, 0.0), 11.0, now=1.05)
    assert monitor.reading.pose.x == 1.0


# ===========================================================================
# SE(2) correction algebra
# ===========================================================================


def test_correction_between_reproduces_the_target_exactly():
    target = Pose2d(3.0, -1.0, 1.1)
    source = Pose2d(-2.0, 4.0, -0.4)
    correction = se2_correction_between(target, source)
    result = se2_apply(correction, source)
    assert pytest.approx(target.x, abs=1e-12) == result.x
    assert pytest.approx(target.y, abs=1e-12) == result.y
    assert pytest.approx(target.theta, abs=1e-12) == result.theta


def test_identity_correction_is_a_no_op():
    pose = Pose2d(5.0, 6.0, 0.3)
    result = se2_apply(SE2.identity(), pose)
    assert pytest.approx(pose.x) == result.x
    assert pytest.approx(pose.y) == result.y
    assert pytest.approx(pose.theta) == result.theta


def test_pose_difference_wraps_the_heading():
    distance, heading = pose_difference(
        Pose2d(0, 0, math.pi - 0.05), Pose2d(3, 4, -math.pi + 0.05)
    )
    assert pytest.approx(5.0) == distance
    assert pytest.approx(0.1, abs=1e-9) == heading


# ===========================================================================
# Fusion behavior
# ===========================================================================


def fresh_estimator(**overrides) -> PoseEstimator:
    cfg = LocalizationConfig(**overrides)
    return PoseEstimator(cfg, make_field_map())


def test_first_tag_fix_is_adopted_outright():
    """Blending from an arbitrary starting correction would leave the robot
    lost for seconds; the first fix must snap."""
    estimator = fresh_estimator()
    estimator.update_odometry(Pose2d(0.0, 0.0, 0.0), 1.0, now=1.0)
    assert estimator.add_tag_estimate(tag_estimate(4.0, 2.0, 0.5), now=1.0)
    fused = estimator.update(1.0)
    assert fused.valid
    assert pytest.approx(4.0, abs=1e-9) == fused.pose.x
    assert pytest.approx(2.0, abs=1e-9) == fused.pose.y


def test_pose_tracks_odometry_between_tag_fixes():
    estimator = fresh_estimator()
    estimator.update_odometry(Pose2d(0.0, 0.0, 0.0), 1.0, now=1.0)
    estimator.add_tag_estimate(tag_estimate(4.0, 2.0, 0.0), now=1.0)
    # Odometry advances 1 m forward; the fused pose must move with it.
    estimator.update_odometry(Pose2d(1.0, 0.0, 0.0), 1.05, now=1.05)
    fused = estimator.update(1.05)
    assert pytest.approx(5.0, abs=1e-9) == fused.pose.x


def test_later_fixes_are_blended_not_snapped():
    estimator = fresh_estimator(
        correction_alpha_translation=0.2, max_correction_step_m=10.0
    )
    estimator.update_odometry(Pose2d(0.0, 0.0, 0.0), 1.0, now=1.0)
    estimator.add_tag_estimate(tag_estimate(0.0, 0.0, 0.0), now=1.0)
    estimator.update_odometry(Pose2d(0.0, 0.0, 0.0), 1.05, now=1.05)
    estimator.add_tag_estimate(tag_estimate(1.0, 0.0, 0.0), now=1.05)
    fused = estimator.update(1.05)
    assert 0.0 < fused.pose.x < 1.0, "a later fix must be blended, not snapped"
    assert pytest.approx(0.2, abs=1e-9) == fused.pose.x


def test_correction_step_is_clamped():
    estimator = fresh_estimator(
        correction_alpha_translation=1.0, max_correction_step_m=0.10
    )
    estimator.update_odometry(Pose2d(0.0, 0.0, 0.0), 1.0, now=1.0)
    estimator.add_tag_estimate(tag_estimate(0.0, 0.0, 0.0), now=1.0)
    estimator.update_odometry(Pose2d(0.0, 0.0, 0.0), 1.05, now=1.05)
    estimator.add_tag_estimate(tag_estimate(5.0, 0.0, 0.0), now=1.05)
    fused = estimator.update(1.05)
    assert fused.pose.x <= 0.10 + 1e-9


def test_wildly_disagreeing_tag_fix_is_refused():
    estimator = fresh_estimator(disagreement_reject_m=1.0)
    estimator.update_odometry(Pose2d(0.0, 0.0, 0.0), 1.0, now=1.0)
    estimator.add_tag_estimate(tag_estimate(0.0, 0.0, 0.0), now=1.0)
    estimator.update_odometry(Pose2d(0.0, 0.0, 0.0), 1.05, now=1.05)
    accepted = estimator.add_tag_estimate(tag_estimate(6.0, 0.0, 0.0), now=1.05)
    assert not accepted
    assert estimator.rejected_fixes == 1
    fused = estimator.update(1.05)
    assert pytest.approx(0.0, abs=1e-9) == fused.pose.x, "must not snap on disagreement"


def test_disagreement_is_reported_separately_for_translation_and_heading():
    estimator = fresh_estimator(
        disagreement_reject_m=100.0, disagreement_reject_rad=100.0
    )
    estimator.update_odometry(Pose2d(0.0, 0.0, 0.0), 1.0, now=1.0)
    estimator.add_tag_estimate(tag_estimate(0.0, 0.0, 0.0), now=1.0)
    estimator.update_odometry(Pose2d(0.0, 0.0, 0.0), 1.05, now=1.05)
    estimator.add_tag_estimate(tag_estimate(3.0, 4.0, 0.5), now=1.05)
    assert pytest.approx(5.0) == estimator.translation_disagreement_m
    assert pytest.approx(0.5) == estimator.heading_disagreement_rad


def test_sustained_disagreement_invalidates_rather_than_snapping():
    estimator = fresh_estimator(
        disagreement_reject_m=1.0, disagreement_invalidate_after_s=0.5
    )
    estimator.update_odometry(Pose2d(0.0, 0.0, 0.0), 1.0, now=1.0)
    estimator.add_tag_estimate(tag_estimate(0.0, 0.0, 0.0), now=1.0)
    for index in range(1, 20):
        moment = 1.0 + index * 0.1
        estimator.update_odometry(Pose2d(0.0, 0.0, 0.0), 1.0 + index * 0.1, now=moment)
        estimator.add_tag_estimate(tag_estimate(9.0, 0.0, 0.0), now=moment)
    result = estimator.update(1.0 + 19 * 0.1)
    assert not result.valid
    assert "disagreement" in result.invalid_reason


# ===========================================================================
# Pose validity -- "I have numbers" is not "the pose is valid"
# ===========================================================================


def test_pose_is_invalid_before_any_tag_fix():
    estimator = fresh_estimator(require_tag_initialization=True)
    estimator.update_odometry(Pose2d(3.0, 3.0, 0.0), 1.0, now=1.0)
    estimator.update_odometry(Pose2d(3.0, 3.0, 0.0), 1.02, now=1.02)
    result = estimator.update(1.02)
    assert not result.valid
    assert "AprilTag" in result.invalid_reason
    assert result.source is LocalizationSource.ODOMETRY_ONLY


def test_odometry_only_is_allowed_when_tag_initialization_is_not_required():
    estimator = PoseEstimator(
        LocalizationConfig(require_tag_initialization=False), make_field_map()
    )
    estimator.update_odometry(Pose2d(3.0, 3.0, 0.0), 1.0, now=1.0)
    estimator.update_odometry(Pose2d(3.0, 3.0, 0.0), 1.02, now=1.02)
    result = estimator.update(1.02)
    assert result.valid and result.source is LocalizationSource.ODOMETRY_ONLY


def test_stale_odometry_with_a_recent_tag_falls_back_to_tag_only():
    estimator = fresh_estimator(odometry_stale_timeout_s=0.2, tag_only_valid_s=1.0)
    estimator.update_odometry(Pose2d(0.0, 0.0, 0.0), 1.0, now=1.0)
    estimator.add_tag_estimate(tag_estimate(5.0, 3.0, 0.2), now=1.0)
    result = estimator.update(1.5)  # odometry frozen since 1.0
    assert result.valid
    assert result.source is LocalizationSource.TAG_ONLY
    assert not result.odometry_fresh
    assert pytest.approx(5.0) == result.pose.x


def test_stale_odometry_and_stale_tags_is_invalid():
    estimator = fresh_estimator(odometry_stale_timeout_s=0.2, tag_only_valid_s=0.5)
    estimator.update_odometry(Pose2d(0.0, 0.0, 0.0), 1.0, now=1.0)
    estimator.add_tag_estimate(tag_estimate(5.0, 3.0, 0.0), now=1.0)
    result = estimator.update(3.0)
    assert not result.valid
    assert "stale" in result.invalid_reason


def test_no_odometry_at_all_is_invalid():
    result = fresh_estimator().update(1.0)
    assert not result.valid
    assert result.source is LocalizationSource.NONE


def test_pose_outside_the_field_is_invalid():
    estimator = PoseEstimator(
        LocalizationConfig(require_tag_initialization=False, field_bounds_margin_m=0.5),
        make_field_map(length=16.0, width=8.0),
    )
    estimator.update_odometry(Pose2d(100.0, 3.0, 0.0), 1.0, now=1.0)
    estimator.update_odometry(Pose2d(100.0, 3.0, 0.0), 1.02, now=1.02)
    result = estimator.update(1.02)
    assert not result.valid and "outside the field" in result.invalid_reason


def test_tag_fix_outside_the_field_is_refused():
    estimator = fresh_estimator()
    estimator.update_odometry(Pose2d(1.0, 1.0, 0.0), 1.0, now=1.0)
    assert not estimator.add_tag_estimate(tag_estimate(500.0, 500.0, 0.0), now=1.0)
    assert estimator.rejected_fixes == 1


def test_non_finite_tag_fix_is_refused():
    estimator = fresh_estimator()
    estimator.update_odometry(Pose2d(1.0, 1.0, 0.0), 1.0, now=1.0)
    assert not estimator.add_tag_estimate(
        tag_estimate(float("nan"), 1.0, 0.0), now=1.0
    )


def test_tag_fix_is_adopted_when_odometry_is_dead():
    estimator = fresh_estimator(odometry_stale_timeout_s=0.1)
    assert estimator.add_tag_estimate(tag_estimate(7.0, 2.0, 1.0), now=5.0)
    result = estimator.update(5.0)
    assert result.valid and result.source is LocalizationSource.TAG_ONLY


# ===========================================================================
# AprilTag geometry -- the frame-convention proof
# ===========================================================================


def straight_ahead_camera() -> CameraConfig:
    return CameraConfig(
        name="tagcam",
        fx=800.0, fy=800.0, cx=640.0, cy=400.0,
        mount_x_m=0.0, mount_y_m=0.0, mount_z_m=0.5,
        pitch_deg=0.0, yaw_deg=0.0, roll_deg=0.0,
    )


def test_tag_solve_recovers_a_known_robot_pose():
    """Robot at (10, 4) facing +X, looking at a tag on the far wall.

    The tag is at (16, 4, 1) with its face toward the blue end (yaw 180),
    so it sits 6 m ahead and 0.5 m above the lens.
    """
    field_from_tag = make_transform(rot_z(math.pi), (16.0, 4.0, 1.0))
    robot_from_camera = camera_to_robot_transform(straight_ahead_camera())
    # OpenCV camera frame: x right, y down, z out of the lens.
    camera_from_tag = make_transform(np.eye(3), (0.0, -0.5, 6.0))
    pose = pose2d_from_transform(
        robot_pose_from_tag(field_from_tag, camera_from_tag, robot_from_camera)
    )
    assert pytest.approx(10.0, abs=1e-9) == pose.x
    assert pytest.approx(4.0, abs=1e-9) == pose.y
    assert pytest.approx(0.0, abs=1e-9) == pose.theta


def test_tag_solve_recovers_a_rotated_robot_pose():
    """Same tag and same robot position, but the robot is turned 45 deg CCW."""
    heading = math.radians(45.0)
    field_from_tag = make_transform(rot_z(math.pi), (16.0, 4.0, 1.0))
    robot_from_camera = camera_to_robot_transform(straight_ahead_camera())

    cos_h, sin_h = math.cos(heading), math.sin(heading)
    # Camera basis vectors expressed in field coordinates.
    z_cam = np.array([cos_h, sin_h, 0.0])   # optical axis
    x_cam = np.array([sin_h, -cos_h, 0.0])  # image right
    y_cam = np.array([0.0, 0.0, -1.0])      # image down

    def to_camera(vector):
        return np.array([vector @ x_cam, vector @ y_cam, vector @ z_cam])

    # pupil-apriltags TAG frame: x = printed right, y = printed down,
    # z = into the tag (away from the viewer).
    rotation = np.column_stack(
        [
            to_camera(np.array([0.0, -1.0, 0.0])),  # printed right  = field -Y
            to_camera(np.array([0.0, 0.0, -1.0])),  # printed down   = field -Z
            to_camera(np.array([1.0, 0.0, 0.0])),   # into the tag   = field +X
        ]
    )
    offset_robot = np.array([cos_h * 6.0, -sin_h * 6.0])  # tag in the robot frame
    camera_from_tag = make_transform(
        rotation, (-offset_robot[1], -0.5, offset_robot[0])
    )
    pose = pose2d_from_transform(
        robot_pose_from_tag(field_from_tag, camera_from_tag, robot_from_camera)
    )
    assert pytest.approx(10.0, abs=1e-9) == pose.x
    assert pytest.approx(4.0, abs=1e-9) == pose.y
    assert pytest.approx(heading, abs=1e-9) == pose.theta


# ===========================================================================
# Multi-tag aggregation
# ===========================================================================


def observation(tag_id, x, y, theta=0.0, distance=2.0) -> TagObservation:
    return TagObservation(
        camera_name="cam0", timestamp=1.0, tag_id=tag_id,
        robot_pose=Pose2d(x, y, theta), distance_m=distance,
        decision_margin=60.0, hamming=0,
    )


def test_aggregate_returns_none_without_observations():
    assert aggregate_tag_poses([], 1.0, 0.75, 5.0) is None


def test_aggregate_averages_consistent_tags_and_reports_the_spread():
    estimate = aggregate_tag_poses(
        [observation(1, 4.0, 2.0), observation(2, 4.1, 2.0), observation(3, 3.9, 2.0)],
        1.0, 0.75, 5.0,
    )
    assert estimate is not None
    assert pytest.approx(4.0, abs=1e-9) == estimate.pose.x
    assert estimate.tag_ids == [1, 2, 3]
    assert estimate.consistency_error_m == pytest.approx(0.1, abs=1e-9)


def test_aggregate_discards_a_flipped_outlier():
    estimate = aggregate_tag_poses(
        [observation(1, 4.0, 2.0), observation(2, 4.05, 2.0), observation(9, 12.0, 7.0)],
        1.0, 0.75, 5.0,
    )
    assert estimate is not None
    assert 9 not in estimate.tag_ids
    assert pytest.approx(4.025, abs=1e-9) == estimate.pose.x


def test_more_and_closer_tags_score_higher_quality():
    single_far = aggregate_tag_poses([observation(1, 4, 2, distance=4.8)], 1.0, 0.75, 5.0)
    several_near = aggregate_tag_poses(
        [observation(i, 4.0, 2.0, distance=1.0) for i in (1, 2, 3)], 1.0, 0.75, 5.0
    )
    assert several_near.quality > single_far.quality


def test_circular_mean_wraps_across_pi():
    assert pytest.approx(math.pi, abs=1e-9) == abs(
        circular_mean([math.pi - 0.01, -math.pi + 0.01])
    )


# ===========================================================================
# Field layout parsing
# ===========================================================================


def minimal_layout(tag_x=4.0):
    return {
        "tags": [
            {
                "ID": 1,
                "pose": {
                    "translation": {"x": tag_x, "y": 2.0, "z": 1.0},
                    "rotation": {"quaternion": {"W": 1.0, "X": 0.0, "Y": 0.0, "Z": 0.0}},
                },
            }
        ],
        "field": {"length": 16.0, "width": 8.0},
    }


def test_parse_valid_layout():
    field_map = parse_field_layout(minimal_layout())
    assert field_map.tag_ids == [1]
    assert field_map.length_m == 16.0
    assert np.allclose(field_map.get(1).translation, [4.0, 2.0, 1.0])


def test_layout_with_a_tag_off_the_field_is_rejected():
    with pytest.raises(FieldLayoutError):
        parse_field_layout(minimal_layout(tag_x=99.0))


def test_layout_without_tags_is_rejected():
    with pytest.raises(FieldLayoutError):
        parse_field_layout({"tags": [], "field": {"length": 16.0, "width": 8.0}})


def test_layout_without_dimensions_is_rejected():
    with pytest.raises(FieldLayoutError):
        parse_field_layout({"tags": minimal_layout()["tags"]})


def test_field_map_bounds_and_clamping():
    field_map = make_field_map(16.0, 8.0)
    assert field_map.contains_point(0.0, 0.0)
    assert not field_map.contains_point(-1.0, 4.0)
    assert field_map.contains_point(-0.4, 4.0, margin_m=0.5)
    assert field_map.clamp_point(-5.0, 100.0, margin_m=0.5) == (0.5, 7.5)
    assert field_map.center == (8.0, 4.0)


def test_a_rejected_tag_pose_is_never_published_as_the_pose():
    """Regression: with odometry dead, the TAG_ONLY branch must fall back to
    the last ACCEPTED fix, not to a fix that was refused for being off-field
    or wildly inconsistent."""
    estimator = fresh_estimator(odometry_stale_timeout_s=0.2, tag_only_valid_s=5.0)
    estimator.update_odometry(Pose2d(0.0, 0.0, 0.0), 1.0, now=1.0)
    estimator.add_tag_estimate(tag_estimate(4.0, 2.0, 0.0), now=1.0)   # accepted
    assert not estimator.add_tag_estimate(
        tag_estimate(900.0, 900.0, 0.0), now=1.1                       # off field
    )
    result = estimator.update(1.5)  # odometry now stale -> TAG_ONLY
    assert result.source is LocalizationSource.TAG_ONLY
    assert result.pose.x == pytest.approx(4.0)
