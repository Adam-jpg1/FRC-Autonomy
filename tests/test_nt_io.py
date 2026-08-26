"""The NetworkTables contract: required keys, types, and the simulator."""

import math

import pytest

from frc_perception.nt_io import (
    REQUIRED_PERCEPTION_KEYS,
    SimulatedNT,
    build_perception_payload,
    missing_required_keys,
)
from frc_perception.types import ControllerState


def payload(**overrides):
    defaults = dict(
        balls_field_x=[1.0, 2.0],
        balls_field_y=[3.0, 4.0],
        balls_field_vx=[0.0, 0.0],
        balls_field_vy=[0.0, 0.0],
        opponents_field_x=[5.0],
        opponents_field_y=[6.0],
        opponents_field_vx=[0.1],
        opponents_field_vy=[0.2],
        robot_x=8.0,
        robot_y=4.0,
        robot_theta=0.5,
        pose_valid=True,
        pose_invalid_reason="",
        frame_counter=42,
        camera_health=[(True, 29.5), (False, 0.0)],
        model_loaded=True,
        odometry_fresh=True,
        tag_pose_available=True,
        tag_count=2,
        localization_source="odometry_tag_corrected",
        localization_quality=0.8,
        translation_disagreement_m=0.05,
        heading_disagreement_deg=1.2,
        loop_fps=30.0,
        perception_latency_ms=45.0,
        inference_ms=18.0,
        apriltag_ms=6.0,
        nav_reason="approach fuel pocket",
        nav_vx=1.0,
        nav_vy=0.0,
        nav_omega=0.2,
        nav_target_x=10.0,
        nav_target_y=4.0,
        nav_has_target=True,
        virtual_controller_enabled=False,
    )
    defaults.update(overrides)
    return build_perception_payload(**defaults)


def test_every_required_contract_key_is_present():
    assert missing_required_keys(payload()) == []


def test_counts_match_the_arrays():
    result = payload()
    assert result["ball_count"] == 2
    assert result["opponent_count"] == 1
    assert len(result["balls_field_x"]) == len(result["balls_field_y"])


def test_per_camera_diagnostics_are_indexed_by_config_order():
    result = payload()
    assert result["camera_0_healthy"] is True
    assert result["camera_1_healthy"] is False
    assert result["camera_0_fps"] == pytest.approx(29.5)


def test_diagnostic_keys_required_by_the_contract_are_present():
    result = payload()
    for key in (
        "model_loaded", "odometry_fresh", "tag_pose_available",
        "localization_source", "loop_fps", "perception_latency_ms",
    ):
        assert key in result


def test_empty_world_publishes_empty_arrays_not_missing_keys():
    result = payload(
        balls_field_x=[], balls_field_y=[], balls_field_vx=[], balls_field_vy=[],
        opponents_field_x=[], opponents_field_y=[],
        opponents_field_vx=[], opponents_field_vy=[],
    )
    assert result["balls_field_x"] == []
    assert result["ball_count"] == 0
    assert missing_required_keys(result) == []


def test_missing_required_keys_detects_a_broken_contract():
    broken = payload()
    del broken["pose_valid"]
    assert missing_required_keys(broken) == ["pose_valid"]


def test_types_are_networktables_friendly():
    for key, value in payload().items():
        assert isinstance(value, (bool, int, float, str, list)), key
        if isinstance(value, list):
            assert all(isinstance(v, float) for v in value), key


def test_pose_valid_false_still_publishes_numbers_and_a_reason():
    """The RIO gates on pose_valid; the numbers are for debugging."""
    result = payload(pose_valid=False, pose_invalid_reason="odometry stale")
    assert result["pose_valid"] is False
    assert result["pose_invalid_reason"] == "odometry stale"
    assert math.isfinite(result["robot_x"])


# ---------------------------------------------------------------------------
# Simulated backend
# ---------------------------------------------------------------------------


def test_simulated_backend_advances_its_own_timestamp():
    backend = SimulatedNT(is_red=True)
    first = backend.read_odometry()
    second = backend.read_odometry()
    assert second[3] >= first[3]
    assert backend.read_is_red_alliance() is True


def test_simulated_backend_can_freeze_odometry_to_reproduce_a_dead_rio():
    backend = SimulatedNT()
    backend.read_odometry()
    backend.freeze_odometry = True
    first = backend.read_odometry()[3]
    second = backend.read_odometry()[3]
    assert first == second


def test_simulated_backend_records_what_was_published():
    backend = SimulatedNT()
    backend.publish_perception(payload())
    backend.publish_controller(ControllerState(enabled=False, heartbeat=7))
    assert backend.published_perception["frame_counter"] == 42
    assert backend.published_controller["heartbeat"] == 7
    assert backend.publish_count == 1


def test_unknown_alliance_is_reported_as_none_by_default():
    assert SimulatedNT().read_is_red_alliance() is None


def test_required_keys_tuple_matches_the_documented_contract():
    assert set(REQUIRED_PERCEPTION_KEYS) == {
        "balls_field_x", "balls_field_y",
        "opponents_field_x", "opponents_field_y",
        "robot_x", "robot_y", "robot_theta",
        "pose_valid", "frame_counter", "ball_count", "opponent_count",
    }
