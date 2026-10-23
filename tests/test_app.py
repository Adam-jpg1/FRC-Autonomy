"""End-to-end loop behavior with no hardware at all.

The app is constructed with every camera disabled, no weights and no
field layout -- i.e. the worst case -- and must still run, publish a
complete contract, and command exactly nothing.
"""

import time

import pytest

from frc_perception.app import PerceptionApp
from frc_perception.config import AppConfig, CameraConfig, NetworkTablesConfig
from frc_perception.nt_io import SimulatedNT, missing_required_keys
from frc_perception.types import Frame
from frc_perception.types import (
    Alliance,
    Detection,
    DetectorHealth,
    ObjectRole,
    Pose2d,
    RobotObservation,
)


def headless_config() -> AppConfig:
    """Two cameras declared but DISABLED: no capture threads, no OpenCV.

    Declaring them keeps the ``camera_0_*`` / ``camera_1_*`` diagnostic keys
    in the published payload (indexed by config order) while nothing is
    actually opened.
    """
    cfg = AppConfig(networktables=NetworkTablesConfig(server_override="127.0.0.1"))
    cfg.cameras = [
        CameraConfig(name="cam0", enabled=False),
        CameraConfig(name="cam1", enabled=False),
    ]
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
        camera_id="cam0", timestamp=1.0, class_name="fuel",
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
        camera_id="cam0", timestamp=1.02, class_name="fuel",
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


# ---------------------------------------------------------------------------
# The detection stage is really wired up
# ---------------------------------------------------------------------------


def test_the_scheduler_is_actually_invoked_every_cycle():
    """Regression: per-stage exception containment can hide a broken call.

    A wrong signature at the scheduler call site would produce an empty
    detection list and a rate-limited warning -- indistinguishable from
    "nothing in view" unless something asserts the call happened.
    """
    app = make_app()
    calls = []
    original = app.scheduler.run_cycle

    def counting():
        calls.append(1)
        return original()

    app.scheduler.run_cycle = counting
    app.step(now=1.0)
    app.step(now=1.033)
    assert len(calls) == 2
    assert app._stage_failures.get("detect", 0) == 0


def test_no_pipeline_stage_reports_a_failure_on_a_clean_cycle():
    """Any stage failing here means a call site is broken, not that the
    hardware is absent -- absence is handled without raising."""
    app = make_app()
    app.step(now=1.0)
    app.step(now=1.033)
    assert {k: v for k, v in app._stage_failures.items() if v} == {}


def test_the_model_is_loaded_once_at_startup_not_per_frame():
    """Reloading weights per frame would dominate the frame budget."""
    app = make_app()
    loads = []
    original = app.detector.load

    def counting():
        loads.append(1)
        return original()

    app.detector.load = counting
    app.start()
    try:
        for index in range(5):
            app.step(now=1.0 + index * 0.033)
    finally:
        app.stop()
    assert len(loads) == 1


def test_an_unhealthy_detector_produces_no_observations_and_no_tracks():
    """A dead detector must not fabricate anything, and tracks must age."""
    backend = SimulatedNT()
    app = make_app(backend)
    assert not app.detector.healthy(), "no weights in the test environment"
    app.step(now=1.0)
    assert app.world.all_tracks == []
    published = backend.published_perception
    assert published["detector_healthy"] is False
    assert published["ball_count"] == 0
    assert published["opponent_count"] == 0


def test_detector_diagnostics_reach_networktables():
    backend = SimulatedNT()
    app = make_app(backend)
    app.step(now=1.0)
    published = backend.published_perception
    assert published["frames_skipped"] == 0
    assert published["stale_results_dropped"] == 0
    # Capture rate and inference rate are separate numbers, per camera.
    assert published["camera_0_healthy"] is False
    assert published["camera_0_fps"] == 0.0
    assert published["camera_0_inference_fps"] == 0.0
    assert published["camera_1_inference_fps"] == 0.0


def test_tracks_age_out_when_the_detector_dies_mid_match():
    """An inference failure must not freeze the last object locations.

    A world model that keeps reporting the last known ball position forever
    is worse than an empty one: navigation would confidently drive at
    something that is no longer there.
    """
    app = make_app()
    app.cfg.world_model.min_hits_to_confirm = 1
    app.cfg.localization.require_tag_initialization = False
    expire = app.cfg.world_model.expire_timeout_game_piece_s

    app.estimator.update_odometry(Pose2d(5.0, 3.0, 0.0), 1.0, now=1.0)
    app.estimator.update_odometry(Pose2d(5.0, 3.0, 0.0), 1.02, now=1.02)
    estimate = app.estimator.update(1.02)
    observation = RobotObservation(
        camera_id="cam0", timestamp=1.02, class_name="fuel",
        role=ObjectRole.GAME_PIECE, confidence=0.9,
        x_robot_m=2.0, y_robot_m=0.0, range_m=2.0, bearing_rad=0.0,
    )
    app._update_world([observation], estimate, now=1.02)
    assert len(app.world.game_pieces(1.02)) == 1

    moment = 1.02
    for _ in range(6):
        moment += expire / 3.0
        app.estimator.update_odometry(Pose2d(5.0, 3.0, 0.0), moment, now=moment)
        app._update_world([], app.estimator.update(moment), now=moment)

    assert app.world.all_tracks == [], "stale tracks must expire, not persist"


def test_navigation_stays_safe_while_the_detector_is_unavailable():
    """No detections plus no pose must still mean no translation."""
    backend = SimulatedNT()
    app = make_app(backend)
    app.step(now=1.0)
    published = backend.published_perception
    assert published["detector_healthy"] is False
    assert published["pose_valid"] is False
    assert published["nav_vx_mps"] == 0.0
    assert published["nav_vy_mps"] == 0.0
    assert backend.published_controller["enabled"] is False
    assert backend.published_controller["left_x"] == 0.0
    assert backend.published_controller["left_y"] == 0.0


def test_navigation_stays_safe_with_a_valid_pose_but_a_dead_detector():
    """With localization working and detection dead, the robot may search
    but must not chase objects it can no longer see."""
    backend = SimulatedNT()
    app = make_app(backend)
    app.cfg.localization.require_tag_initialization = False
    app.cfg.navigation.search_point_field = None  # spin-search, no field target

    for index in range(3):
        moment = 1.0 + index * 0.03
        app.estimator.update_odometry(Pose2d(5.0, 3.0, 0.0), moment, now=moment)
        app.step(now=moment)

    published = backend.published_perception
    assert published["pose_valid"] is True
    assert published["detector_healthy"] is False
    assert published["ball_count"] == 0
    assert published["nav_vx_mps"] == 0.0
    assert published["nav_vy_mps"] == 0.0
    assert published["nav_omega_radps"] != 0.0, "spin-search is the safe behavior"


# ---------------------------------------------------------------------------
# Full pipeline with real frames and a stand-in detector
# ---------------------------------------------------------------------------


def calibrated_camera(name: str) -> CameraConfig:
    """A camera with exact intrinsics and a mount that can see the ground."""
    return CameraConfig(
        name=name,
        width=640, height=480,
        fx=500.0, fy=500.0, cx=320.0, cy=240.0,
        mount_x_m=0.30, mount_y_m=0.0, mount_z_m=0.60,
        pitch_deg=20.0, yaw_deg=0.0, roll_deg=0.0,
        fps=30,
    )


class ScriptedDetector:
    """Stands in for YOLO on the GPU: one fuel box per frame."""

    name = "scripted"

    def __init__(self) -> None:
        self.frames_seen = []
        self.last_latency_ms = 0.0
        self.last_preprocess_ms = 0.0
        self.last_postprocess_ms = 0.0

    def load(self) -> bool:
        return True

    def healthy(self) -> bool:
        return True

    def health(self) -> DetectorHealth:
        return DetectorHealth(
            backend=self.name, loaded=True, healthy=True, classes_ok=True
        )

    @property
    def load_error(self) -> str:
        return ""

    def close(self) -> None:
        pass

    def predict(self, frame, camera_id="", capture_timestamp=0.0):
        self.frames_seen.append((frame.camera_id, frame.seq))
        return [
            Detection(
                class_name="fuel", confidence=0.93,
                x1=300.0, y1=330.0, x2=340.0, y2=370.0,
                camera_id=frame.camera_id,
                frame_sequence=frame.seq,
                capture_timestamp=frame.timestamp,
            )
        ]


def test_frames_flow_from_the_cameras_through_to_field_tracks(tmp_path):
    """The whole hardware-to-world-model path, with real Frame objects.

    Proves the layers are actually connected: a captured image reaches the
    detector, the resulting Detection is ranged with that camera's
    intrinsics and mount, transformed into field coordinates with the fused
    pose, and lands in the world model as a track.
    """
    cv2 = pytest.importorskip("cv2")
    numpy = pytest.importorskip("numpy")

    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    for index in range(4):
        image = numpy.full((480, 640, 3), 40, dtype=numpy.uint8)
        cv2.imwrite(str(frames_dir / f"f{index}.png"), image)

    cfg = headless_config()
    cfg.cameras = [calibrated_camera("cam0")]
    cfg.world_model.min_hits_to_confirm = 1
    cfg.localization.require_tag_initialization = False

    # Let the simulated roboRIO drive odometry: it advances its own
    # timestamp, which is what the freshness rule keys on.
    backend = SimulatedNT(start_pose=(5.0, 3.0, 0.0))
    app = PerceptionApp(cfg, nt=backend, replay_paths={"cam0": str(frames_dir)})
    app.detector = ScriptedDetector()
    app.scheduler.backend = app.detector

    app.cameras.start()
    try:
        deadline = time.monotonic() + 3.0
        tracks = []
        while time.monotonic() < deadline:
            moment = time.monotonic()
            app.step(now=moment)
            tracks = app.world.game_pieces(moment)
            if tracks:
                break
            time.sleep(0.02)
    finally:
        app.cameras.stop()

    assert app.detector.frames_seen, "the scheduler never reached the detector"
    assert tracks, "a detected object never became a field-space track"

    track = tracks[0]
    assert track.class_name == "fuel"
    assert track.role is ObjectRole.GAME_PIECE
    # The camera looks down field from a robot at (5, 3) facing +X, so the
    # object must land ahead of the robot, not behind it or on top of it.
    assert track.x > 5.0
    assert track.y == pytest.approx(3.0, abs=0.05)

    published = backend.published_perception
    assert published["pose_valid"] is True
    assert published["ball_count"] >= 1
    assert published["detector_healthy"] is True


def test_the_original_capture_timestamp_survives_the_whole_pipeline(tmp_path):
    """Field projection must use when the PICTURE was taken, not when
    inference finished -- on a moving robot those are different places."""
    cfg = headless_config()
    cfg.cameras = [calibrated_camera("cam0")]
    app = PerceptionApp(cfg, nt=SimulatedNT())
    app.detector = ScriptedDetector()

    capture_time = 1234.5
    frame = Frame(camera_id="cam0", seq=11, timestamp=capture_time, image=object())
    detections = app.detector.predict(frame)
    observations = app._range_detections(detections, cfg.cameras[0])

    assert observations, "the detection should have ranged successfully"
    assert observations[0].timestamp == pytest.approx(capture_time)
    assert observations[0].camera_id == "cam0"


# ---------------------------------------------------------------------------
# The main loop runs with NetworkTables absent
# ---------------------------------------------------------------------------


def test_the_loop_runs_with_the_real_nt_backend_and_no_ntcore(monkeypatch):
    """run.py's default path: the REAL NetworkTablesIO with ntcore missing.

    Pins the whole reason NT is optional -- the loop must iterate, the
    detector/pose/world-model stages must all execute, and nothing may
    raise. Uses the real backend, not SimulatedNT.
    """
    from frc_perception.config import NetworkTablesConfig
    import frc_perception.nt_io as nt_module
    from frc_perception.nt_io import NetworkTablesIO

    monkeypatch.setattr(nt_module, "NT_AVAILABLE", False)
    monkeypatch.setattr(nt_module, "NT_IMPORT_ERROR", "ModuleNotFoundError: no ntcore")

    cfg = headless_config()
    cfg.networktables = NetworkTablesConfig(team_number=1234)
    app = PerceptionApp(cfg, nt=NetworkTablesIO(cfg.networktables))

    app.start()
    try:
        for index in range(5):
            app.step(now=1.0 + index * 0.033)
    finally:
        app.stop()

    assert app.stats.frame_counter == 5
    # No stage may report a failure: NT being absent is a degraded mode, not
    # an error path.
    assert {k: v for k, v in app._stage_failures.items() if v} == {}


def test_without_networktables_the_pose_is_invalid_and_nothing_translates(monkeypatch):
    """No NT means no odometry, so the safe behavior must hold."""
    from frc_perception.config import NetworkTablesConfig
    import frc_perception.nt_io as nt_module
    from frc_perception.nt_io import NetworkTablesIO

    monkeypatch.setattr(nt_module, "NT_AVAILABLE", False)
    monkeypatch.setattr(nt_module, "NT_IMPORT_ERROR", "ModuleNotFoundError: no ntcore")

    cfg = headless_config()
    cfg.networktables = NetworkTablesConfig(team_number=1234)
    app = PerceptionApp(cfg, nt=NetworkTablesIO(cfg.networktables))

    app.start()
    try:
        app.step(now=1.0)
        estimate = app.estimator.last_estimate
        motion = app.navigator.plan(estimate, [], [], 1.0)
    finally:
        app.stop()

    assert estimate.valid is False
    assert motion.vx_mps == 0.0 and motion.vy_mps == 0.0
    assert app.world.all_tracks == []
