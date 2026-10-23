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
        camera_health=[(True, 29.5, 14.8, 3), (False, 0.0, 0.0, 0)],
        model_loaded=True,
        detector_healthy=True,
        detector_classes_ok=True,
        detector_error="",
        detector_latency_ms=12.5,
        preprocess_ms=2.0,
        postprocess_ms=1.5,
        frames_skipped=7,
        stale_frames_dropped=1,
        stale_results_dropped=2,
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
        "camera_0_healthy", "camera_1_healthy", "camera_0_fps", "camera_1_fps",
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


def test_capture_and_inference_rates_are_published_separately():
    """With one GPU serving two cameras these genuinely differ, and
    conflating them hides whether capture or inference is the bottleneck."""
    result = payload()
    assert result["camera_0_fps"] == pytest.approx(29.5)
    assert result["camera_0_inference_fps"] == pytest.approx(14.8)
    assert result["camera_0_frames_skipped"] == 3


def test_detector_diagnostics_are_published():
    result = payload()
    assert result["detector_healthy"] is True
    assert result["detector_classes_ok"] is True
    assert result["detector_latency_ms"] == pytest.approx(12.5)
    assert result["stale_results_dropped"] == 2
    assert result["frames_skipped"] == 7


def test_an_unhealthy_detector_is_visible_on_the_dashboard():
    """A dead detector must be diagnosable from the driver station."""
    result = payload(
        detector_healthy=False,
        model_loaded=False,
        detector_error="failed to load models/best.pt",
        balls_field_x=[], balls_field_y=[],
        balls_field_vx=[], balls_field_vy=[],
    )
    assert result["detector_healthy"] is False
    assert result["model_loaded"] is False
    assert "best.pt" in result["detector_error"]
    assert result["ball_count"] == 0


# ---------------------------------------------------------------------------
# NetworkTables is OPTIONAL
# ---------------------------------------------------------------------------
#
# ntcore (pyntcore) has no cp310 wheel on JetPack 6.2, so the pipeline has
# to run without it. These tests pin the degraded mode so it cannot
# regress into a crash, and so the no-ops stay no-ops by intent rather
# than by accident.


def test_nt_availability_flag_matches_the_real_import_state():
    import importlib.util

    from frc_perception.nt_io import NT_AVAILABLE

    assert NT_AVAILABLE is (importlib.util.find_spec("ntcore") is not None)


def test_nt_status_describes_the_situation_either_way():
    from frc_perception.nt_io import NT_AVAILABLE, nt_status

    status = nt_status()
    assert status
    if NT_AVAILABLE:
        assert "available" in status
    else:
        assert "UNAVAILABLE" in status and "disabled" in status


def test_the_module_imports_even_without_ntcore():
    """The whole point: a missing optional dependency must not stop import."""
    import frc_perception.nt_io as module

    assert hasattr(module, "NetworkTablesIO")
    assert hasattr(module, "NT_AVAILABLE")


def unavailable_backend(monkeypatch):
    """A real NetworkTablesIO with ntcore forced unavailable."""
    from frc_perception.config import NetworkTablesConfig
    from frc_perception.nt_io import NetworkTablesIO

    import frc_perception.nt_io as module

    monkeypatch.setattr(module, "NT_AVAILABLE", False)
    monkeypatch.setattr(module, "NT_IMPORT_ERROR", "ModuleNotFoundError: no ntcore")
    return NetworkTablesIO(NetworkTablesConfig(team_number=1234))


def test_connect_does_not_raise_when_nt_is_unavailable(monkeypatch):
    backend = unavailable_backend(monkeypatch)
    backend.connect()  # must not raise
    assert backend.connected is False
    assert backend.available is False
    assert "UNAVAILABLE" in backend.load_error


def test_publishing_is_a_no_op_when_nt_is_unavailable(monkeypatch):
    """Nothing may be transmitted, and nothing may blow up."""
    from frc_perception.types import ControllerState

    backend = unavailable_backend(monkeypatch)
    backend.connect()
    backend.publish_perception(payload())
    backend.publish_controller(ControllerState())
    backend.flush()
    backend.close()  # also a no-op; must not raise


def test_reads_return_none_when_nt_is_unavailable(monkeypatch):
    """No odometry and no alliance -- handled upstream as UNKNOWN/invalid."""
    backend = unavailable_backend(monkeypatch)
    backend.connect()
    assert backend.read_odometry() is None
    assert backend.read_is_red_alliance() is None
    assert backend.poll_connection() is False


def test_the_unavailable_warning_is_logged_once_not_every_cycle(monkeypatch, caplog):
    import frc_perception.nt_io as module

    monkeypatch.setattr(module, "NT_AVAILABLE", False)
    monkeypatch.setattr(module, "NT_IMPORT_ERROR", "ModuleNotFoundError: no ntcore")
    monkeypatch.setattr(module, "_WARNED_UNAVAILABLE", False)

    with caplog.at_level("WARNING"):
        module.warn_nt_unavailable()
        module.warn_nt_unavailable()
        module.warn_nt_unavailable()

    emitted = [r for r in caplog.records if "NETWORKTABLES DISABLED" in r.message]
    assert len(emitted) == 1
    # The message must tell the reader the pipeline continues and how to fix it.
    assert "continues to run" in emitted[0].message
    assert "pyntcore" in emitted[0].message


def test_live_requires_both_the_import_and_a_started_client(monkeypatch):
    """A guard that only checked _instance would be fooled by a stale object."""
    import frc_perception.nt_io as module

    backend = unavailable_backend(monkeypatch)
    backend._instance = object()  # pretend a client object survived
    assert backend._live is False, "NT_AVAILABLE False must still mean not live"

    monkeypatch.setattr(module, "NT_AVAILABLE", True)
    assert backend._live is True
    backend._instance = None
    assert backend._live is False
