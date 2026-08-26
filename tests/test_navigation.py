"""Navigation: the pose-invalid safety rule, target selection, avoidance.

The first test class is the most important in this repository. If
``pose_valid`` is False the robot must not translate, ever, for any
reason -- a field-coordinate target computed from an unknown pose points
in an arbitrary direction.
"""

import math

import pytest

from frc_perception.config import CLASS_BUMPER_BLUE, CLASS_FUEL, NavigationConfig
from frc_perception.navigation import (
    FieldBounds,
    Navigator,
    cluster_fuel,
    filter_usable_tracks,
    path_clearance,
)
from frc_perception.types import (
    LocalizationSource,
    ObjectRole,
    Pose2d,
    PoseEstimate,
    Track,
    TrackFreshness,
)

BOUNDS = FieldBounds(length_m=16.0, width_m=8.0)


def valid_pose(x=8.0, y=4.0, theta=0.0, quality=1.0) -> PoseEstimate:
    return PoseEstimate(
        pose=Pose2d(x, y, theta),
        valid=True,
        source=LocalizationSource.ODOMETRY_TAG_CORRECTED,
        timestamp=1.0,
        odometry_fresh=True,
        tag_pose_available=True,
        quality=quality,
    )


def invalid_pose(reason="odometry stale and no recent AprilTag pose") -> PoseEstimate:
    return PoseEstimate(
        pose=Pose2d(8.0, 4.0, 0.0),
        valid=False,
        source=LocalizationSource.NONE,
        timestamp=1.0,
        invalid_reason=reason,
    )


def fuel(track_id, x, y, confidence=0.9, last_seen=1.0) -> Track:
    return Track(
        track_id=track_id,
        class_name=CLASS_FUEL,
        role=ObjectRole.GAME_PIECE,
        x=x,
        y=y,
        confidence=confidence,
        last_seen=last_seen,
        hit_count=5,
    )


def opponent(track_id, x, y, vx=0.0, vy=0.0, last_seen=1.0) -> Track:
    return Track(
        track_id=track_id,
        class_name=CLASS_BUMPER_BLUE,
        role=ObjectRole.OPPONENT,
        x=x,
        y=y,
        vx=vx,
        vy=vy,
        confidence=0.9,
        last_seen=last_seen,
        hit_count=5,
        velocity_valid=vx != 0.0 or vy != 0.0,
    )


# ===========================================================================
# THE SAFETY RULE
# ===========================================================================


def test_invalid_pose_and_no_fuel_never_translates():
    """The single most important test in this project.

    No pose means no field frame, which means every field-coordinate
    target -- including "drive to field center" -- points somewhere
    arbitrary. Translation must be exactly zero.
    """
    navigator = Navigator(
        NavigationConfig(search_point_field=(8.0, 4.0), search_rotation_radps=1.0), BOUNDS
    )
    motion = navigator.plan(invalid_pose(), [], [], now=1.0)
    assert motion.vx_mps == 0.0
    assert motion.vy_mps == 0.0
    assert motion.target is None
    assert motion.confidence == 0.0
    assert motion.omega_radps == pytest.approx(1.0)


def test_invalid_pose_never_translates_even_with_fuel_in_view():
    """Fuel tracks live in field coordinates too: they are just as useless."""
    navigator = Navigator(NavigationConfig(search_point_field=(8.0, 4.0)), BOUNDS)
    motion = navigator.plan(
        invalid_pose(), [fuel(1, 9.0, 4.0), fuel(2, 9.2, 4.1)], [], now=1.0
    )
    assert motion.vx_mps == 0.0 and motion.vy_mps == 0.0
    assert motion.target is None


def test_invalid_pose_can_be_configured_to_produce_fully_neutral_output():
    navigator = Navigator(
        NavigationConfig(allow_search_rotation_when_pose_invalid=False), BOUNDS
    )
    motion = navigator.plan(invalid_pose(), [], [], now=1.0)
    assert (motion.vx_mps, motion.vy_mps, motion.omega_radps) == (0.0, 0.0, 0.0)


def test_invalid_pose_command_is_not_marked_field_relative():
    """Nothing about the command came from field coordinates, and the
    controller adapter must not rotate it by an unknown heading."""
    navigator = Navigator(NavigationConfig(), BOUNDS)
    motion = navigator.plan(invalid_pose(), [], [], now=1.0)
    assert motion.field_relative is False


def test_invalid_pose_reason_is_carried_through_for_the_drive_team():
    navigator = Navigator(NavigationConfig(), BOUNDS)
    motion = navigator.plan(invalid_pose("odometry stale"), [], [], now=1.0)
    assert "odometry stale" in motion.reason


# ===========================================================================
# Search behavior with a VALID pose
# ===========================================================================


def test_valid_pose_with_no_fuel_and_no_search_point_spins_in_place():
    navigator = Navigator(
        NavigationConfig(search_point_field=None, search_rotation_radps=0.8), BOUNDS
    )
    motion = navigator.plan(valid_pose(), [], [], now=1.0)
    assert motion.vx_mps == 0.0 and motion.vy_mps == 0.0
    assert motion.omega_radps == pytest.approx(0.8)


def test_valid_pose_with_no_fuel_drives_to_the_configured_search_point():
    navigator = Navigator(
        NavigationConfig(search_point_field=(12.0, 4.0), max_speed_mps=2.0), BOUNDS
    )
    motion = navigator.plan(valid_pose(x=8.0, y=4.0), [], [], now=1.0)
    assert motion.vx_mps > 0.0
    assert motion.vy_mps == pytest.approx(0.0, abs=1e-9)
    assert motion.target == (12.0, 4.0)


def test_arriving_at_the_search_point_stops_translating_and_keeps_scanning():
    navigator = Navigator(
        NavigationConfig(search_point_field=(8.0, 4.0), search_rotation_radps=0.7), BOUNDS
    )
    motion = navigator.plan(valid_pose(x=8.0, y=4.0), [], [], now=1.0)
    assert motion.vx_mps == 0.0 and motion.vy_mps == 0.0
    assert motion.omega_radps == pytest.approx(0.7)


# ===========================================================================
# Fuel target selection
# ===========================================================================


def test_cluster_fuel_groups_by_proximity():
    clusters = cluster_fuel(
        [fuel(1, 0.0, 0.0), fuel(2, 0.5, 0.0), fuel(3, 5.0, 0.0)], radius_m=1.0
    )
    assert sorted(len(c) for c in clusters) == [1, 2]


def test_a_dense_cluster_beats_a_closer_isolated_piece():
    navigator = Navigator(NavigationConfig(), BOUNDS)
    tracks = [
        fuel(1, 9.0, 4.0),                     # isolated, 1 m away
        fuel(2, 12.0, 4.0),                    # cluster of three, 4 m away
        fuel(3, 12.3, 4.2),
        fuel(4, 12.1, 3.7),
    ]
    motion = navigator.plan(valid_pose(x=8.0, y=4.0), tracks, [], now=1.0)
    assert motion.target is not None
    assert motion.target[0] > 11.0, "the three-piece pocket should win"
    assert navigator.last_pockets[0].count == 3


def test_an_isolated_piece_wins_when_the_cluster_is_blocked():
    navigator = Navigator(NavigationConfig(), BOUNDS)
    tracks = [fuel(1, 9.0, 4.0), fuel(2, 12.0, 4.0), fuel(3, 12.3, 4.2), fuel(4, 12.1, 3.7)]
    blocker = opponent(10, 10.5, 4.0)  # sits on the path to the cluster
    motion = navigator.plan(valid_pose(x=8.0, y=4.0), tracks, [blocker], now=1.0)
    assert motion.target is not None
    assert motion.target[0] == pytest.approx(9.0, abs=0.3)


def test_fuel_beyond_the_maximum_target_distance_is_ignored():
    navigator = Navigator(
        NavigationConfig(max_target_distance_m=3.0, search_point_field=None), BOUNDS
    )
    motion = navigator.plan(valid_pose(x=1.0, y=4.0), [fuel(1, 14.0, 4.0)], [], now=1.0)
    assert motion.target is None
    assert motion.vx_mps == 0.0


def test_fuel_too_close_to_a_wall_is_not_targeted():
    navigator = Navigator(
        NavigationConfig(wall_margin_m=0.5, search_point_field=None), BOUNDS
    )
    motion = navigator.plan(valid_pose(x=2.0, y=4.0), [fuel(1, 0.1, 4.0)], [], now=1.0)
    assert motion.target is None


def test_the_command_points_at_the_target():
    navigator = Navigator(NavigationConfig(), BOUNDS)
    motion = navigator.plan(valid_pose(x=8.0, y=4.0), [fuel(1, 8.0, 6.0)], [], now=1.0)
    assert motion.vy_mps > 0.0, "target is to field +Y, so the command must be +Y"
    assert motion.vx_mps == pytest.approx(0.0, abs=1e-9)


def test_rotation_turns_the_robot_toward_the_target():
    navigator = Navigator(NavigationConfig(), BOUNDS)
    # Robot faces +X; the target is to its left (+Y), so omega must be CCW (+).
    motion = navigator.plan(valid_pose(x=8.0, y=4.0, theta=0.0), [fuel(1, 8.0, 6.0)], [], 1.0)
    assert motion.omega_radps > 0.0


def test_speed_is_capped_and_slows_on_approach():
    navigator = Navigator(NavigationConfig(max_speed_mps=1.5, approach_kp=5.0), BOUNDS)
    far = navigator.plan(valid_pose(x=2.0, y=4.0), [fuel(1, 7.0, 4.0)], [], 1.0)
    assert math.hypot(far.vx_mps, far.vy_mps) == pytest.approx(1.5)
    near = navigator.plan(valid_pose(x=6.6, y=4.0), [fuel(1, 7.0, 4.0)], [], 1.0)
    assert 0.0 < math.hypot(near.vx_mps, near.vy_mps) < 1.5


def test_arriving_at_the_target_stops_translation():
    navigator = Navigator(NavigationConfig(arrive_tolerance_m=0.3), BOUNDS)
    motion = navigator.plan(valid_pose(x=8.0, y=4.0), [fuel(1, 8.1, 4.0)], [], 1.0)
    assert motion.vx_mps == 0.0 and motion.vy_mps == 0.0
    assert "arrived" in motion.reason


def test_stale_fuel_is_ignored_by_the_caller_filter():
    """Expired tracks are dropped before planning; the model decides when a
    track is expired and navigation honors it."""
    def freshness(track, now):
        return TrackFreshness.EXPIRED if track.track_id == 2 else TrackFreshness.RECENT

    usable = filter_usable_tracks([fuel(1, 9.0, 4.0), fuel(2, 9.5, 4.0)], freshness, 1.0)
    assert [t.track_id for t in usable] == [1]


def test_aging_tracks_can_be_excluded():
    def freshness(track, now):
        return TrackFreshness.AGING

    assert filter_usable_tracks([fuel(1, 9.0, 4.0)], freshness, 1.0, allow_aging=False) == []


def test_low_confidence_fuel_scores_below_confident_fuel():
    navigator = Navigator(NavigationConfig(), BOUNDS)
    navigator.plan(
        valid_pose(x=8.0, y=4.0),
        [fuel(1, 10.0, 4.0, confidence=0.95), fuel(2, 10.0, 2.0, confidence=0.05)],
        [],
        1.0,
    )
    best = navigator.last_pockets[0]
    assert best.mean_confidence > 0.5


# ===========================================================================
# Opponent avoidance
# ===========================================================================


def test_path_clearance_measures_the_closest_opponent_to_the_path():
    assert path_clearance((0, 0), (10, 0), [(5, 2)]) == pytest.approx(2.0)
    assert path_clearance((0, 0), (10, 0), []) == float("inf")


def test_a_blocked_path_is_flagged_and_penalized():
    navigator = Navigator(NavigationConfig(), BOUNDS)
    navigator.plan(valid_pose(x=8.0, y=4.0), [fuel(1, 12.0, 4.0)], [opponent(9, 10.0, 4.0)], 1.0)
    pocket = navigator.last_pockets[0]
    assert pocket.blocked
    assert pocket.score < 0.0


def test_a_nearby_opponent_pushes_the_command_sideways():
    navigator = Navigator(NavigationConfig(avoid_gain=2.0), BOUNDS)
    clean = navigator.plan(valid_pose(x=8.0, y=4.0), [fuel(1, 11.0, 4.0)], [], 1.0)
    pushed = navigator.plan(
        valid_pose(x=8.0, y=4.0), [fuel(1, 11.0, 4.0)], [opponent(9, 8.4, 4.5)], 1.0
    )
    assert clean.vy_mps == pytest.approx(0.0, abs=1e-9)
    assert pushed.vy_mps < 0.0, "opponent to the left must push the command right"
    assert "avoidance" in pushed.reason


def test_opponents_are_avoided_where_they_will_be_not_where_they_were():
    navigator = Navigator(NavigationConfig(prediction_horizon_s=1.0), BOUNDS)
    # The opponent is clear of the path now but is driving straight into it.
    moving = opponent(9, 10.0, 6.0, vy=-2.0, last_seen=1.0)
    navigator.plan(valid_pose(x=8.0, y=4.0), [fuel(1, 12.0, 4.0)], [moving], now=1.0)
    assert navigator.last_pockets[0].blocked


def test_distant_opponents_do_not_perturb_the_command():
    navigator = Navigator(NavigationConfig(avoid_consider_radius_m=3.0), BOUNDS)
    motion = navigator.plan(
        valid_pose(x=8.0, y=4.0), [fuel(1, 9.0, 4.0)], [opponent(9, 8.0, 7.9)], 1.0
    )
    assert motion.vy_mps == pytest.approx(0.0, abs=1e-9)


def test_an_opponent_exactly_on_the_robot_does_not_divide_by_zero():
    navigator = Navigator(NavigationConfig(), BOUNDS)
    motion = navigator.plan(
        valid_pose(x=8.0, y=4.0), [fuel(1, 11.0, 4.0)], [opponent(9, 8.0, 4.0)], 1.0
    )
    assert math.isfinite(motion.vx_mps) and math.isfinite(motion.vy_mps)


# ===========================================================================
# Field bounds helper
# ===========================================================================


def test_field_bounds_contains_respects_the_margin():
    assert BOUNDS.contains(1.0, 1.0)
    assert not BOUNDS.contains(0.1, 4.0, margin_m=0.5)
    assert BOUNDS.center == (8.0, 4.0)


def test_navigation_works_without_field_bounds():
    """A missing field layout must not disable navigation entirely."""
    navigator = Navigator(NavigationConfig(), bounds=None)
    motion = navigator.plan(valid_pose(x=8.0, y=4.0), [fuel(1, 9.0, 4.0)], [], 1.0)
    assert motion.target is not None
