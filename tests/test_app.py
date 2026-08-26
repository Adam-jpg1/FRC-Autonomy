"""End-to-end loop behavior with no hardware at all.

The app is constructed with every camera disabled, no weights and no
field layout -- i.e. the worst case -- and must still run, publish a
complete contract, and command exactly nothing.
"""

import pytest

from frc_perception.app import PerceptionApp
from frc_perception.config import AppConfig, NetworkTablesConfig
from frc_perception.nt_io import SimulatedNT, missing_required_keys
from frc_perception.types import Alliance, ObjectRole, Pose2d, RobotObservation


def headless_config() -> AppConfig:
    cfg = AppConfig(networktables=NetworkTablesConfig(server_override="127.0.0.1"))
    cfg.cameras = []  # no capture threads, no OpenCV needed
    return cfg


def make_app(backend=None) -> PerceptionApp:
    return PerceptionApp(headless_config(), nt=backend or SimulatedNT())


def test_the_loop_runs_with_no_cameras_no_model_and_no_field_layout():
    app = make_app()
    app.step(now=1.0)
    app.step(now=1.033)
    assert app.stats.frame_counter == 2


def test_every_cycle_publishes_the_full_contract():
    backend = SimulatedNT()
    app = make_app(backend)
    app.step(now=1.0)
    assert missing_required_keys(backend.published_perception) == []
    assert backend.published_perception["model_loaded"] is False


def test_with_nothing_working_the_pose_is_invalid_and_nothing_moves():
    backend = SimulatedNT()
    app = make_app(backend)
    app.step(now=1.0)
    published = backend.published_perception
    assert published["pose_valid"] is False
    assert published["nav_vx_mps"] == 0.0
    assert published["nav_vy_mps"] == 0.0


def test_the_virtual_controller_stays_neutral_and_disabled():
    backend = SimulatedNT()
    app = make_app(backend)
    app.step(now=1.0)
    controller = backend.published_controller
    assert controller["enabled"] is False
    assert controller["left_x"] == 0.0 and controller["left_y"] == 0.0
    assert controller["right_x"] == 0.0


def test_the_heartbeat_advances_every_cycle():
    backend = SimulatedNT()
    app = make_app(backend)
    app.step(now=1.0)
    first = backend.published_controller["heartbeat"]
    app.step(now=1.033)
    assert backend.published_controller["heartbeat"] == first + 1


def test_alliance_is_read_from_the_simulated_fms():
    app = make_app(SimulatedNT(is_red=True))
    app.step(now=1.0)
    assert app.alliance is Alliance.RED


def test_unknown_alliance_is_not_guessed():
    app = make_app(SimulatedNT(is_red=None))
    app.step(now=1.0)
    assert app.alliance is Alliance.UNKNOWN


def test_an_invalid_pose_clears_the_world_model():
    """Field-space tracks recorded against an unknown pose are worse than
    no tracks: they look authoritative and are wrong."""
    app = make_app()
    observation = RobotObservation(
        camera_name="cam0", timestamp=1.0, class_name="fuel",
        role=ObjectRole.GAME_PIECE, confidence=0.9,
        x_robot_m=2.0, y_robot_m=0.0, range_m=2.0, bearing_rad=0.0,
    )
    invalid = app.estimator.update(1.0)
    assert not invalid.valid
    app._update_world([observation], invalid, now=1.0)
    assert app.world.all_tracks == []


def test_a_valid_pose_places_observations_in_field_coordinates():
    app = make_app()
    app.cfg.world_model.min_hits_to_confirm = 1
    app.cfg.localization.require_tag_initialization = False
    app.estimator.update_odometry(Pose2d(5.0, 3.0, 0.0), 1.0, now=1.0)
    app.estimator.update_odometry(Pose2d(5.0, 3.0, 0.0), 1.02, now=1.02)
    estimate = app.estimator.update(1.02)
    assert estimate.valid

    observation = RobotObservation(
        camera_name="cam0", timestamp=1.02, class_name="fuel",
        role=ObjectRole.GAME_PIECE, confidence=0.9,
        x_robot_m=2.0, y_robot_m=0.0, range_m=2.0, bearing_rad=0.0,
    )
    app._update_world([observation], estimate, now=1.02)
    tracks = app.world.game_pieces(1.02)
    assert len(tracks) == 1
    assert tracks[0].x == pytest.approx(7.0)  # 5.0 + 2.0 forward
    assert tracks[0].y == pytest.approx(3.0)


def test_a_failing_stage_is_contained_counted_and_recovers():
    app = make_app()
    boom = RuntimeError("stage exploded")

    def explode():
        raise boom

    assert app._stage("unit-test", explode, default="fallback") == "fallback"
    assert app._stage_failures["unit-test"] == 1
    assert app._stage("unit-test", lambda: "ok") == "ok"
    assert app._stage_failures["unit-test"] == 0


def test_stopping_leaves_the_gamepad_neutral():
    backend = SimulatedNT()
    app = make_app(backend)
    app.step(now=1.0)
    app.stop()
    assert backend.published_controller["enabled"] is False
    assert backend.published_controller["left_y"] == 0.0
