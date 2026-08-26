"""The detector hardware boundary: backends, class contract, safe failure.

No Hailo accelerator, no HailoRT, no Ultralytics and no torch are required
to run any of this. The Hailo backend's artifact preflight is a plain YAML
read on purpose, so a laptop can validate a deployed model, and the
"accelerator is missing" path is exactly what this environment exercises.
"""


import pytest
import yaml

from frc_perception.config import DetectorConfig, EXPECTED_CLASS_NAMES
from frc_perception.detector import (
    BACKENDS,
    DetectionError,
    DetectorBackend,
    HailoDetectorBackend,
    UltralyticsPyTorchBackend,
    build_detector_backend,
    compare_class_names,
    find_hef,
    names_from_metadata,
    read_hailo_metadata,
    resolve_torch_device,
)
from frc_perception.types import Detection, Frame


def write_export_dir(
    root,
    names=EXPECTED_CLASS_NAMES,
    arch="hailo8",
    include_hef=True,
    include_metadata=True,
    extra=None,
):
    """Build a fake Ultralytics Hailo export directory on disk."""
    directory = root / "best_hailo_model"
    directory.mkdir(exist_ok=True)
    if include_hef:
        (directory / "best.hef").write_bytes(b"\x00" * 32)
    if include_metadata:
        payload = {
            "names": {index: name for index, name in enumerate(names)},
            "hailo_arch": arch,
            "imgsz": [640, 640],
            "nms": True,
            "task": "detect",
        }
        payload.update(extra or {})
        (directory / "metadata.yaml").write_text(yaml.safe_dump(payload))
    return str(directory)


def hailo_config(model_dir, **overrides) -> DetectorConfig:
    defaults = dict(backend="hailo", hailo_model_path=model_dir, hailo_arch="hailo8")
    defaults.update(overrides)
    return DetectorConfig(**defaults)


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------


def test_both_backends_are_registered():
    assert set(BACKENDS) == {"hailo", "pytorch"}


def test_build_selects_the_configured_backend(tmp_path):
    hailo = build_detector_backend(hailo_config(str(tmp_path)))
    assert isinstance(hailo, HailoDetectorBackend)
    torch_backend = build_detector_backend(DetectorConfig(backend="pytorch"))
    assert isinstance(torch_backend, UltralyticsPyTorchBackend)


def test_an_unknown_backend_is_refused_loudly():
    with pytest.raises(ValueError, match="unknown detector backend"):
        build_detector_backend(DetectorConfig(backend="magic"))


def test_both_backends_implement_the_same_interface():
    for backend_class in BACKENDS.values():
        assert issubclass(backend_class, DetectorBackend)
        for method in ("load", "predict", "healthy", "close", "health"):
            assert callable(getattr(backend_class, method))


def test_model_path_follows_the_selected_backend(tmp_path):
    cfg = hailo_config("/models/hailo_dir", pytorch_model_path="/models/best.pt")
    assert cfg.model_path == "/models/hailo_dir"
    cfg.backend = "pytorch"
    assert cfg.model_path == "/models/best.pt"


# ---------------------------------------------------------------------------
# The class-name contract
# ---------------------------------------------------------------------------


def test_matching_names_pass_regardless_of_index_order():
    ok, message = compare_class_names(
        {0: "bumper_red", 1: "fuel", 2: "bumper_blue"}, EXPECTED_CLASS_NAMES
    )
    assert ok and "match" in message


def test_a_missing_class_is_reported():
    ok, message = compare_class_names({0: "fuel", 1: "bumper_blue"}, EXPECTED_CLASS_NAMES)
    assert not ok and "bumper_red" in message


def test_names_are_read_from_a_mapping_or_its_repr():
    assert names_from_metadata({"names": {0: "fuel"}}) == {0: "fuel"}
    assert names_from_metadata({"names": "{0: 'fuel', 1: 'bumper_blue'}"}) == {
        0: "fuel", 1: "bumper_blue"
    }
    assert names_from_metadata({"names": ["fuel", "bumper_blue"]}) == {
        0: "fuel", 1: "bumper_blue"
    }
    assert names_from_metadata({}) == {}
    assert names_from_metadata({"names": "not a dict"}) == {}


# ---------------------------------------------------------------------------
# Hailo artifact preflight -- runs with no hardware
# ---------------------------------------------------------------------------


def test_a_complete_export_directory_passes_preflight(tmp_path):
    backend = HailoDetectorBackend(hailo_config(write_export_dir(tmp_path)))
    ok, messages = backend.preflight()
    assert ok, messages
    assert any("classes match" in m for m in messages)
    assert backend.classes_ok


def test_a_missing_directory_fails_preflight(tmp_path):
    backend = HailoDetectorBackend(hailo_config(str(tmp_path / "nope")))
    ok, messages = backend.preflight()
    assert not ok and "not a directory" in messages[0]


def test_copying_only_the_hef_fails_preflight(tmp_path):
    """The single most likely deployment mistake.

    metadata.yaml carries the class names; without it there is nothing for
    class-name routing to route on, and the failure at runtime would look
    like a model that simply detects nothing.
    """
    directory = write_export_dir(tmp_path, include_metadata=False)
    ok, messages = HailoDetectorBackend(hailo_config(directory)).preflight()
    assert not ok
    assert "metadata.yaml" in messages[0]


def test_a_directory_without_a_hef_fails_preflight(tmp_path):
    directory = write_export_dir(tmp_path, include_hef=False)
    ok, messages = HailoDetectorBackend(hailo_config(directory)).preflight()
    assert not ok and any("no .hef" in m for m in messages)


def test_wrong_class_names_fail_preflight(tmp_path):
    directory = write_export_dir(tmp_path, names=("ball", "blue", "red"))
    backend = HailoDetectorBackend(hailo_config(directory))
    ok, messages = backend.preflight()
    assert not ok
    assert any("mismatch" in m for m in messages)
    assert not backend.classes_ok


def test_metadata_without_class_names_fails_preflight(tmp_path):
    directory = write_export_dir(tmp_path)
    (tmp_path / "best_hailo_model" / "metadata.yaml").write_text(
        yaml.safe_dump({"hailo_arch": "hailo8"})
    )
    ok, messages = HailoDetectorBackend(hailo_config(directory)).preflight()
    assert not ok and any("no class names" in m for m in messages)


def test_an_architecture_mismatch_is_warned_about(tmp_path):
    """A Hailo-8L binary does not run on a Hailo-8 and vice versa."""
    directory = write_export_dir(tmp_path, arch="hailo8l")
    backend = HailoDetectorBackend(hailo_config(directory, hailo_arch="hailo8"))
    ok, messages = backend.preflight()
    assert ok, "the class contract is still satisfied"
    assert any("hailo8l" in m and "WARNING" in m for m in messages)


def test_preflight_reports_when_nms_runs_on_the_cpu(tmp_path):
    directory = write_export_dir(tmp_path, extra={"nms": False})
    _ok, messages = HailoDetectorBackend(hailo_config(directory)).preflight()
    assert any("nms=false" in m for m in messages)


def test_read_metadata_handles_a_broken_file(tmp_path):
    directory = write_export_dir(tmp_path)
    (tmp_path / "best_hailo_model" / "metadata.yaml").write_text("{{{ not yaml")
    metadata, error = read_hailo_metadata(directory)
    assert metadata == {} and error


def test_find_hef_locates_the_binary(tmp_path):
    directory = write_export_dir(tmp_path)
    assert find_hef(directory).endswith("best.hef")
    assert find_hef(str(tmp_path / "missing")) is None


# ---------------------------------------------------------------------------
# Safe failure: no accelerator means no detections, never a CPU fallback
# ---------------------------------------------------------------------------


def test_hailo_load_fails_safely_when_the_artifact_is_missing(tmp_path):
    backend = HailoDetectorBackend(hailo_config(str(tmp_path / "absent")))
    assert backend.load() is False
    assert backend.healthy() is False
    assert backend.loaded is False
    assert backend.load_error


def test_an_unloaded_backend_returns_no_detections_rather_than_raising(tmp_path):
    """A dead detector must never fabricate boxes, and must never explode
    the main loop either."""
    backend = HailoDetectorBackend(hailo_config(str(tmp_path / "absent")))
    backend.load()
    frame = Frame(camera_id="cam0", seq=1, timestamp=0.0, image=object())
    assert backend.predict(frame) == []


def test_a_failed_load_never_substitutes_the_pytorch_backend(tmp_path):
    """There is no automatic fallback. On the robot, a Hailo failure means
    no detections -- not YOLO quietly eating the Pi's CPU."""
    cfg = hailo_config(str(tmp_path / "absent"))
    backend = build_detector_backend(cfg)
    backend.load()
    assert isinstance(backend, HailoDetectorBackend)
    assert backend.name == "hailo"
    assert not backend.healthy()


def test_repeated_failures_mark_the_backend_unhealthy(tmp_path):
    backend = HailoDetectorBackend(hailo_config(write_export_dir(tmp_path)))
    backend._health.healthy = True  # pretend the device came up
    for _ in range(backend.cfg.failures_before_unhealthy):
        backend._record_failure("device timeout")
    assert not backend.healthy()
    assert "device timeout" in backend.health().last_error


def test_a_success_resets_the_failure_counter(tmp_path):
    backend = HailoDetectorBackend(hailo_config(write_export_dir(tmp_path)))
    backend._record_failure("blip")
    backend._record_success(11.0)
    assert backend.health().consecutive_failures == 0
    assert backend.health().mean_latency_ms == pytest.approx(11.0)


def test_pytorch_backend_load_fails_safely_without_weights(tmp_path):
    backend = UltralyticsPyTorchBackend(
        DetectorConfig(backend="pytorch", pytorch_model_path=str(tmp_path / "nope.pt"))
    )
    assert backend.load() is False
    assert not backend.healthy()
    assert "not found" in backend.load_error


def test_health_snapshot_cannot_be_mutated_by_the_caller(tmp_path):
    backend = HailoDetectorBackend(hailo_config(write_export_dir(tmp_path)))
    snapshot = backend.health()
    snapshot.healthy = True
    assert backend.healthy() is False


def test_torch_device_selection_is_development_only():
    assert resolve_torch_device("cpu") == "cpu"
    assert resolve_torch_device("mps") == "mps"


# ---------------------------------------------------------------------------
# Both backends produce the same hardware-independent Detection
# ---------------------------------------------------------------------------


class FakeBox:
    def __init__(self, class_index, confidence, xyxy):
        self.cls = FakeScalar(class_index)
        self.conf = FakeScalar(confidence)
        self.xyxy = [FakeList(xyxy)]


class FakeScalar:
    def __init__(self, value):
        self._value = value

    def item(self):
        return self._value


class FakeList:
    def __init__(self, values):
        self._values = list(values)

    def tolist(self):
        return self._values


class FakeResult:
    def __init__(self, boxes, names):
        self.boxes = boxes
        self.names = names


NAMES = {0: "fuel", 1: "bumper_blue", 2: "bumper_red"}


@pytest.mark.parametrize("backend_class", [HailoDetectorBackend, UltralyticsPyTorchBackend])
def test_every_backend_emits_the_same_detection_shape(backend_class, tmp_path):
    """Whatever produced the boxes, downstream sees one type.

    This is the whole point of the boundary: swap the accelerator and
    geometry, the world model and navigation do not change.
    """
    backend = backend_class(hailo_config(write_export_dir(tmp_path)))
    results = [FakeResult([FakeBox(0, 0.91, [10.0, 20.0, 50.0, 80.0])], NAMES)]
    detections = backend._boxes_to_detections(results, NAMES, "front_left", 42, 123.5)

    assert len(detections) == 1
    detection = detections[0]
    assert isinstance(detection, Detection)
    assert detection.class_name == "fuel"
    assert detection.confidence == pytest.approx(0.91)
    assert (detection.x1, detection.y1, detection.x2, detection.y2) == (10.0, 20.0, 50.0, 80.0)
    assert detection.camera_id == "front_left"
    assert detection.frame_sequence == 42
    assert detection.capture_timestamp == pytest.approx(123.5)
    assert detection.bbox == (10.0, 20.0, 50.0, 80.0)
    assert detection.pixel_width == pytest.approx(40.0)
    assert detection.bottom_center == (30.0, 80.0)


def test_unknown_class_names_are_dropped_not_renumbered(tmp_path):
    backend = HailoDetectorBackend(hailo_config(write_export_dir(tmp_path)))
    names = {0: "fuel", 1: "traffic_cone"}
    results = [
        FakeResult(
            [FakeBox(0, 0.9, [1, 2, 3, 4]), FakeBox(1, 0.99, [5, 6, 7, 8])], names
        )
    ]
    detections = backend._boxes_to_detections(results, names, "cam0", 1, 0.0)
    assert [d.class_name for d in detections] == ["fuel"]


def test_per_class_confidence_gates_are_applied_once_for_all_backends(tmp_path):
    cfg = hailo_config(write_export_dir(tmp_path))
    cfg.per_class_conf = {"fuel": 0.80, "bumper_blue": 0.10, "bumper_red": 0.10}
    backend = HailoDetectorBackend(cfg)
    results = [
        FakeResult(
            [FakeBox(0, 0.50, [1, 2, 3, 4]), FakeBox(1, 0.50, [5, 6, 7, 8])], NAMES
        )
    ]
    detections = backend._boxes_to_detections(results, NAMES, "cam0", 1, 0.0)
    assert [d.class_name for d in detections] == ["bumper_blue"]


def test_unreadable_boxes_are_skipped_not_fatal(tmp_path):
    class ExplodingBox:
        @property
        def cls(self):
            raise RuntimeError("API drift")

    backend = HailoDetectorBackend(hailo_config(write_export_dir(tmp_path)))
    results = [FakeResult([ExplodingBox(), FakeBox(0, 0.9, [1, 2, 3, 4])], NAMES)]
    detections = backend._boxes_to_detections(results, NAMES, "cam0", 1, 0.0)
    assert len(detections) == 1


def test_detection_error_is_the_only_exception_type_from_predict(tmp_path):
    """The scheduler catches DetectionError specifically."""
    class ExplodingModel:
        def predict(self, *args, **kwargs):
            raise RuntimeError("accelerator fell off the bus")

    backend = HailoDetectorBackend(hailo_config(write_export_dir(tmp_path)))
    backend._model = ExplodingModel()
    frame = Frame(camera_id="cam0", seq=1, timestamp=0.0, image=object())
    with pytest.raises(DetectionError):
        backend.predict(frame)
    assert backend.health().consecutive_failures == 1
