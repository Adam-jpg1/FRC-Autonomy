"""Configuration validation and the detector's class-name contract."""

import pytest

from frc_perception.config import (
    AppConfig,
    CameraConfig,
    ControllerConfig,
    DetectorConfig,
    EXPECTED_CLASS_NAMES,
    LocalizationConfig,
    NetworkTablesConfig,
    WorldModelConfig,
    config_summary,
    load_config,
    validate_config,
)
from frc_perception.detector import compare_class_names, resolve_device


def calibrated_camera(name="cam0", index=0) -> CameraConfig:
    return CameraConfig(
        name=name,
        device_index=index,
        device_path=f"/dev/v4l/by-id/usb-Arducam-{name}-video-index0",
        fx=800.0, fy=805.0, cx=640.0, cy=400.0,
        mount_x_m=0.3, mount_y_m=0.0, mount_z_m=0.6,
        pitch_deg=18.0, yaw_deg=0.0, roll_deg=0.0,
        distance_poly=(1.02, -0.05),
    )


def complete_config() -> AppConfig:
    cfg = AppConfig(
        networktables=NetworkTablesConfig(team_number=1234),
        cameras=[calibrated_camera(), calibrated_camera("cam1", 2)],
    )
    cfg.detector.fuel_diameter_m = 0.15
    cfg.detector.bumper_width_m = 0.8
    cfg.localization.apriltag.tag_size_m = 0.1651
    cfg.navigation.search_point_field = (8.0, 4.0)
    return cfg


def severities(cfg) -> set:
    return {issue.severity for issue in validate_config(cfg)}


# ---------------------------------------------------------------------------
# A fully filled-in configuration is clean
# ---------------------------------------------------------------------------


def test_a_complete_configuration_has_no_issues():
    assert validate_config(complete_config()) == []


def test_the_shipped_configuration_reports_its_must_inputs_rather_than_failing():
    """The default config must be honest about what is missing, and must not
    contain structural errors."""
    issues = validate_config(load_config())
    assert issues, "the shipped config has unfilled values and must say so"
    assert all(issue.severity != "FAIL" for issue in issues)
    assert any(issue.severity == "MUST INPUT" for issue in issues)


def test_config_summary_is_printable():
    assert "camera cam0" in config_summary(load_config())


# ---------------------------------------------------------------------------
# Structural failures
# ---------------------------------------------------------------------------


def test_duplicate_camera_names_fail():
    cfg = complete_config()
    cfg.cameras[1].name = "cam0"
    assert "FAIL" in severities(cfg)


def test_two_cameras_on_one_device_fail():
    cfg = complete_config()
    cfg.cameras[1].device_path = cfg.cameras[0].device_path
    assert "FAIL" in severities(cfg)


def test_no_enabled_cameras_fails():
    cfg = complete_config()
    for camera in cfg.cameras:
        camera.enabled = False
    assert "FAIL" in severities(cfg)


def test_renaming_a_yolo_class_fails():
    """Class names are a contract with the trained weights."""
    cfg = complete_config()
    cfg.detector = DetectorConfig(class_names=("ball", "bumper_blue", "bumper_red"))
    assert "FAIL" in severities(cfg)


def test_principal_point_outside_the_image_fails():
    cfg = complete_config()
    cfg.cameras[0].cx = 5000.0
    assert "FAIL" in severities(cfg)


def test_reject_threshold_below_the_warn_threshold_fails():
    cfg = complete_config()
    cfg.localization = LocalizationConfig(
        disagreement_warn_m=3.0, disagreement_reject_m=1.0
    )
    cfg.localization.apriltag.tag_size_m = 0.1651
    assert "FAIL" in severities(cfg)


def test_out_of_range_controller_limits_fail():
    cfg = complete_config()
    cfg.controller = ControllerConfig(output_limit=1.5)
    assert "FAIL" in severities(cfg)


def test_expire_timeout_below_the_fresh_timeout_fails():
    cfg = complete_config()
    cfg.world_model = WorldModelConfig(
        fresh_timeout_game_piece_s=3.0, expire_timeout_game_piece_s=1.0
    )
    assert "FAIL" in severities(cfg)


# ---------------------------------------------------------------------------
# MUST INPUT reporting
# ---------------------------------------------------------------------------


def test_missing_team_number_is_reported_as_must_input():
    cfg = complete_config()
    cfg.networktables.team_number = 0
    assert any(
        issue.severity == "MUST INPUT" and "team_number" in issue.where
        for issue in validate_config(cfg)
    )


def test_a_server_override_satisfies_the_team_number_requirement():
    cfg = complete_config()
    cfg.networktables.team_number = 0
    cfg.networktables.server_override = "127.0.0.1"
    assert validate_config(cfg) == []


def test_missing_tag_size_is_must_input_not_a_guess():
    cfg = complete_config()
    cfg.localization.apriltag.tag_size_m = None
    assert any(
        issue.severity == "MUST INPUT" and "tag_size_m" in issue.where
        for issue in validate_config(cfg)
    )


def test_an_uncalibrated_camera_is_must_input():
    cfg = complete_config()
    cfg.cameras[0].fx = None
    assert any(issue.severity == "MUST INPUT" for issue in validate_config(cfg))


def test_enabling_the_virtual_controller_raises_a_warning():
    cfg = complete_config()
    cfg.controller.enabled = True
    assert any(
        issue.severity == "WARNING" and issue.where == "controller.enabled"
        for issue in validate_config(cfg)
    )


# ---------------------------------------------------------------------------
# Intrinsics fallback
# ---------------------------------------------------------------------------


def test_uncalibrated_camera_falls_back_to_an_fov_pinhole():
    cam = CameraConfig(name="c", width=1280, height=800, horizontal_fov_deg=90.0)
    fx, fy, cx, cy = cam.effective_intrinsics()
    assert not cam.has_calibrated_intrinsics
    assert fx == pytest.approx(640.0)  # (w/2) / tan(45 deg)
    assert (cx, cy) == (640.0, 400.0)
    assert fx == fy


def test_calibrated_intrinsics_are_used_verbatim():
    cam = calibrated_camera()
    assert cam.effective_intrinsics() == (800.0, 805.0, 640.0, 400.0)


# ---------------------------------------------------------------------------
# Detector class-name contract
# ---------------------------------------------------------------------------


def test_matching_class_names_pass_regardless_of_index_order():
    ok, message = compare_class_names(
        {0: "bumper_red", 1: "fuel", 2: "bumper_blue"}, EXPECTED_CLASS_NAMES
    )
    assert ok and "match" in message


def test_a_missing_class_is_reported():
    ok, message = compare_class_names({0: "fuel", 1: "bumper_blue"}, EXPECTED_CLASS_NAMES)
    assert not ok and "missing" in message and "bumper_red" in message


def test_an_extra_class_is_reported():
    ok, message = compare_class_names(
        {0: "fuel", 1: "bumper_blue", 2: "bumper_red", 3: "robot"}, EXPECTED_CLASS_NAMES
    )
    assert not ok and "unexpected" in message


def test_explicit_device_requests_are_passed_through():
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("cuda:1") == "cuda:1"


def test_a_camera_without_a_persistent_path_is_must_input():
    """/dev/videoN numbers change across reboots; two identical OV9782
    modules will eventually swap places."""
    cfg = complete_config()
    cfg.cameras[0].device_path = ""
    assert any(
        issue.severity == "MUST INPUT" and "device_path" in issue.where
        for issue in validate_config(cfg)
    )


def test_persistent_path_overrides_the_index():
    cam = calibrated_camera("cam0", 3)
    assert cam.uses_persistent_path
    assert cam.source == cam.device_path
    cam.device_path = ""
    assert not cam.uses_persistent_path
    assert cam.source == 3


def test_disabling_require_cuda_warns():
    """Correct for laptop development, wrong for competition."""
    cfg = complete_config()
    cfg.detector.require_cuda = False
    assert any(
        issue.severity == "WARNING" and issue.where == "detector.require_cuda"
        for issue in validate_config(cfg)
    )


def test_a_result_age_below_the_frame_freshness_limit_warns():
    cfg = complete_config()
    cfg.max_detection_result_age_s = cfg.frame_freshness_s / 2
    assert any(
        issue.severity == "WARNING" and "max_detection_result_age_s" in issue.where
        for issue in validate_config(cfg)
    )


def test_a_zero_inference_budget_fails():
    cfg = complete_config()
    cfg.max_inferences_per_cycle = 0
    assert "FAIL" in severities(cfg)
