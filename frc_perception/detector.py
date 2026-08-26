"""Detector backends: the hardware boundary of this project.

Everything above this line is hardware-specific (a Hailo-8 on a Raspberry
Pi AI HAT+, or PyTorch on a development laptop).  Everything below it sees
only :class:`~frc_perception.types.Detection` and has no idea which one
produced it.  Geometry contains no Hailo code, the world model imports no
Ultralytics, and navigation has never heard of a HEF file.  Swapping
accelerators should mean adding one class in this file and changing one
config value.

BACKENDS
--------
:class:`HailoDetectorBackend`
    Production. Loads an Ultralytics Hailo export DIRECTORY (``*.hef``
    plus ``metadata.yaml``) through the Ultralytics ``HailoBackend``,
    which runs it on the Hailo-8 via HailoRT.

:class:`UltralyticsPyTorchBackend`
    Development and testing ONLY: desktops, saved images, saved video,
    model verification. It is never selected automatically on the
    competition machine.

NO SILENT CPU FALLBACK
----------------------
If the Hailo accelerator fails to initialize on the robot, the detector
reports itself UNHEALTHY and produces no detections. It does not quietly
start running a YOLO model on the Pi's CPU: that would consume the four
cores the rest of the pipeline needs, deliver a few frames per second,
and hide a hardware fault behind a plausible-looking trickle of stale
boxes. Instead, existing world-model tracks age out normally, navigation
degrades safely, and ``model_loaded``/``detector_healthy`` on
NetworkTables tell the drive team exactly what happened.

CLASS ROUTING
-------------
Routing is by class NAME (``fuel``, ``bumper_blue``, ``bumper_red``),
never by index. A model whose metadata does not carry exactly those three
names fails detector preflight loudly rather than mislabeling everything.
For a Hailo export the names live in ``metadata.yaml`` beside the
``.hef`` -- which is why the whole export DIRECTORY is the deployable
artifact, not just the ``.hef``.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .config import DetectorConfig
from .types import Detection, DetectorHealth, Frame

logger = logging.getLogger(__name__)


class DetectionError(Exception):
    """Inference failed for one frame.  The scheduler catches and counts these."""


# ---------------------------------------------------------------------------
# Class-name contract
# ---------------------------------------------------------------------------


def compare_class_names(
    model_names: Dict[int, str], expected: Sequence[str]
) -> Tuple[bool, str]:
    """Compare a model's class names with the expected set.

    Order does not matter -- routing is by name -- but the SET must match
    exactly: a missing class means a blind spot, an extra class means
    these are not the weights this configuration describes.
    """
    actual = {str(name) for name in model_names.values()}
    wanted = set(expected)
    if actual == wanted:
        return True, f"classes match: {sorted(actual)}"
    missing = sorted(wanted - actual)
    extra = sorted(actual - wanted)
    parts = []
    if missing:
        parts.append(f"missing {missing}")
    if extra:
        parts.append(f"unexpected {extra}")
    return False, "class mismatch: " + ", ".join(parts)


def read_hailo_metadata(model_dir: str) -> Tuple[Dict[str, Any], str]:
    """Read ``metadata.yaml`` from an Ultralytics Hailo export directory.

    Returns ``(metadata, error)``.  Deliberately a plain YAML read: it
    needs neither HailoRT nor Ultralytics nor the accelerator, so
    ``run.py --check`` can validate the class contract of a deployed model
    on any machine, including a laptop.
    """
    if not os.path.isdir(model_dir):
        return {}, f"not a directory: {model_dir!r}"
    metadata_path = os.path.join(model_dir, "metadata.yaml")
    if not os.path.exists(metadata_path):
        return {}, (
            f"{metadata_path} is missing. The whole export directory is the "
            "deployable artifact -- copying only the .hef loses the class "
            "names, and class-name routing then has nothing to route on."
        )
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - PyYAML is a hard dependency
        return {}, f"PyYAML unavailable: {exc}"
    try:
        with open(metadata_path, "r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        # A truncated or corrupted copy must report itself, not raise out of
        # preflight -- this is a plausible outcome of a half-finished rsync.
        return {}, f"cannot read {metadata_path}: {exc}"
    if not isinstance(payload, dict):
        return {}, f"{metadata_path} is not a YAML mapping"
    return payload, ""


def names_from_metadata(metadata: Dict[str, Any]) -> Dict[int, str]:
    """Extract ``{index: name}`` from Ultralytics export metadata.

    Ultralytics writes ``names`` either as a real mapping or as the repr
    of one, depending on the format, so both are accepted.
    """
    names = metadata.get("names")
    if isinstance(names, str):
        import ast

        try:
            names = ast.literal_eval(names)
        except (ValueError, SyntaxError):
            return {}
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, (list, tuple)):
        return {index: str(value) for index, value in enumerate(names)}
    return {}


def find_hef(model_dir: str) -> Optional[str]:
    """Locate the ``.hef`` inside an export directory."""
    if not os.path.isdir(model_dir):
        return None
    for entry in sorted(os.listdir(model_dir)):
        if entry.endswith(".hef"):
            return os.path.join(model_dir, entry)
    return None


# ---------------------------------------------------------------------------
# The interface
# ---------------------------------------------------------------------------


class DetectorBackend:
    """What every inference backend must provide.

    Implementations must never raise out of :meth:`load` or :meth:`close`,
    and must raise only :class:`DetectionError` out of :meth:`predict`.
    """

    #: Short name published over NetworkTables for diagnostics.
    name: str = "base"

    def __init__(self, cfg: DetectorConfig) -> None:
        self.cfg = cfg
        self._health = DetectorHealth(backend=self.name)
        self._inference_count = 0
        self._latency_total_ms = 0.0
        self.last_latency_ms = 0.0
        self.last_preprocess_ms = 0.0
        self.last_postprocess_ms = 0.0

    # -- lifecycle ---------------------------------------------------------

    def load(self) -> bool:
        """Prepare the backend.  Returns success; never raises."""
        raise NotImplementedError

    def predict(
        self, frame: Frame, camera_id: str = "", capture_timestamp: float = 0.0
    ) -> List[Detection]:
        """Run inference on one frame.

        ``camera_id`` and ``capture_timestamp`` default to the frame's own
        values; they are accepted explicitly so a caller replaying stored
        images can supply them.
        """
        raise NotImplementedError

    def healthy(self) -> bool:
        return self._health.healthy

    def close(self) -> None:
        """Release the device.  Never raises."""
        return

    # -- shared bookkeeping ------------------------------------------------

    def health(self) -> DetectorHealth:
        self._health.inferences = self._inference_count
        self._health.mean_latency_ms = (
            self._latency_total_ms / self._inference_count
            if self._inference_count
            else 0.0
        )
        return DetectorHealth(**vars(self._health))

    @property
    def classes_ok(self) -> bool:
        return self._health.classes_ok

    @property
    def class_report(self) -> str:
        return self._health.class_report

    @property
    def load_error(self) -> str:
        return self._health.last_error

    @property
    def loaded(self) -> bool:
        return self._health.loaded

    def _record_success(self, latency_ms: float) -> None:
        self._inference_count += 1
        self._latency_total_ms += latency_ms
        self.last_latency_ms = latency_ms
        self._health.consecutive_failures = 0

    def _record_failure(self, message: str) -> None:
        self._health.consecutive_failures += 1
        self._health.last_error = message
        if self._health.consecutive_failures >= self.cfg.failures_before_unhealthy:
            if self._health.healthy:
                logger.error(
                    "detector %s: UNHEALTHY after %d consecutive failures (%s); "
                    "no new detections will be produced",
                    self.name, self._health.consecutive_failures, message,
                )
            self._health.healthy = False

    def _check_classes(self, names: Dict[int, str]) -> None:
        ok, report = compare_class_names(names, self.cfg.class_names)
        self._health.classes_ok = ok
        self._health.class_report = report
        if not ok:
            logger.error("detector %s: %s", self.name, report)

    def _boxes_to_detections(
        self,
        results: Any,
        fallback_names: Dict[int, str],
        camera_id: str,
        frame_sequence: int,
        capture_timestamp: float,
    ) -> List[Detection]:
        """Convert Ultralytics ``Results`` into our hardware-independent type.

        Shared by both backends: the Hailo backend goes through the same
        Ultralytics ``Results`` object, so there is exactly one place where
        boxes become :class:`Detection` and exactly one place where the
        per-class confidence gate is applied.
        """
        detections: List[Detection] = []
        allowed = set(self.cfg.class_names)
        for result in results or []:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            names = {
                int(k): str(v)
                for k, v in dict(getattr(result, "names", fallback_names)).items()
            }
            for box in boxes:
                try:
                    class_index = int(box.cls.item())
                    confidence = float(box.conf.item())
                    x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
                except Exception as exc:  # noqa: BLE001 - defensive against API drift
                    logger.debug("detector: unreadable box (%s)", exc)
                    continue
                class_name = names.get(class_index, "")
                if class_name not in allowed:
                    continue
                if confidence < self.cfg.per_class_conf.get(
                    class_name, self.cfg.conf_threshold
                ):
                    continue
                detections.append(
                    Detection(
                        class_name=class_name,
                        confidence=confidence,
                        x1=x1, y1=y1, x2=x2, y2=y2,
                        camera_id=camera_id,
                        frame_sequence=frame_sequence,
                        capture_timestamp=capture_timestamp,
                    )
                )
        return detections


# ---------------------------------------------------------------------------
# Hailo-8 (production)
# ---------------------------------------------------------------------------


class HailoDetectorBackend(DetectorBackend):
    """Hailo-8 inference through the Ultralytics Hailo integration.

    Ultralytics' ``AutoBackend`` recognizes a ``*_hailo_model`` directory
    and loads it with its ``HailoBackend``, which talks to the accelerator
    through HailoRT (``hailo_platform``).  Using that pathway rather than
    reimplementing HEF I/O and YOLO postprocessing by hand means the
    decode path is the same one the exporter was written against --
    including on-chip NMS, whose parameters were baked in at export time.

    ON-CHIP NMS, AND WHY THE EXPORT CONFIDENCE MATTERS
    --------------------------------------------------
    A detection export bakes ``nms_scores_th`` (the export-time ``conf``)
    into ``nms_config.json``. The accelerator will never return a box
    below that threshold, so the runtime ``conf_threshold`` can only
    filter FURTHER. Export at a low confidence and tune at runtime -- see
    ``training/export_hailo.py``, which defaults to doing exactly that.
    """

    name = "hailo"

    def __init__(self, cfg: DetectorConfig) -> None:
        super().__init__(cfg)
        self._model: Any = None
        self._names: Dict[int, str] = {}
        self._health.model_path = cfg.hailo_model_path

    # -- preflight (works with no accelerator and no Ultralytics) ----------

    def preflight(self) -> Tuple[bool, List[str]]:
        """Validate the model artifact on disk without touching hardware.

        Returns ``(ok, messages)``.  Used by :meth:`load` and by
        ``run.py --check``, so a laptop can verify a deployed model's class
        contract before it is ever copied to the Pi.
        """
        messages: List[str] = []
        model_dir = self.cfg.hailo_model_path

        metadata, error = read_hailo_metadata(model_dir)
        if error:
            return False, [error]

        hef = find_hef(model_dir)
        if hef is None:
            return False, [f"no .hef file inside {model_dir!r}"]
        messages.append(
            f"artifact: {os.path.basename(hef)} "
            f"({os.path.getsize(hef) / 1e6:.1f} MB) + metadata.yaml"
        )

        architecture = str(metadata.get("hailo_arch", "")) or "(not recorded)"
        messages.append(f"hailo_arch: {architecture}")
        expected_arch = self.cfg.hailo_arch
        if architecture not in ("(not recorded)", expected_arch):
            messages.append(
                f"WARNING: compiled for {architecture!r} but this machine is "
                f"configured as {expected_arch!r}; a Hailo-8L binary does not "
                "run on a Hailo-8 and vice versa"
            )

        names = names_from_metadata(metadata)
        if not names:
            return False, messages + [
                "metadata.yaml carries no class names; class-name routing "
                "cannot work with this artifact"
            ]
        ok, report = compare_class_names(names, self.cfg.class_names)
        self._health.classes_ok = ok
        self._health.class_report = report
        messages.append(report)
        if metadata.get("imgsz"):
            messages.append(f"imgsz: {metadata['imgsz']}")
        if not metadata.get("nms", False):
            messages.append(
                "NOTE: metadata says nms=false, so NMS runs on the Pi CPU"
            )
        return ok, messages

    # -- lifecycle ---------------------------------------------------------

    def load(self) -> bool:
        if self._model is not None:
            return True

        ok, messages = self.preflight()
        for message in messages:
            logger.info("detector hailo: %s", message)
        if not ok:
            self._health.last_error = "; ".join(messages) or "preflight failed"
            logger.error("detector hailo: preflight FAILED -- %s", self._health.last_error)
            return False

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            self._health.last_error = (
                f"ultralytics not installed: {exc}. Install it on the Pi "
                "(CPU torch is fine -- inference runs on the Hailo)."
            )
            logger.error("detector hailo: %s", self._health.last_error)
            return False

        try:
            # AutoBackend routes a *_hailo_model directory to its HailoBackend,
            # which imports hailo_platform (HailoRT). If the AI HAT+ is absent
            # or its driver is not loaded, this is where it fails -- and it is
            # allowed to fail: no fallback happens here.
            model = YOLO(self.cfg.hailo_model_path, task="detect")
            names = getattr(model, "names", None) or {}
            self._names = {int(k): str(v) for k, v in dict(names).items()}
            if self._names:
                self._check_classes(self._names)
            self._model = model
            self._health.loaded = True
            self._health.healthy = True
            self._health.device_report = f"Hailo ({self.cfg.hailo_arch}) via HailoRT"
            self._health.last_error = ""
            logger.info(
                "detector hailo: MODEL LOADED %s (%s)",
                self.cfg.hailo_model_path, self._health.class_report,
            )
            return True
        except Exception as exc:  # noqa: BLE001 - HailoRT can raise anything
            self._health.last_error = f"Hailo initialization failed: {exc}"
            self._health.loaded = False
            self._health.healthy = False
            logger.error(
                "detector hailo: %s -- the detector is UNHEALTHY. No CPU fallback "
                "will be started; tracks will age out and navigation will degrade "
                "safely. Check `hailortcli fw-control identify`.",
                self._health.last_error,
            )
            self._model = None
            return False

    def close(self) -> None:
        # HailoBackend releases its VDevice in __del__; dropping the reference
        # is the supported way to let go of the accelerator.
        self._model = None
        self._health.healthy = False
        self._health.loaded = False

    # -- inference ---------------------------------------------------------

    def predict(
        self, frame: Frame, camera_id: str = "", capture_timestamp: float = 0.0
    ) -> List[Detection]:
        if self._model is None:
            return []
        started = time.monotonic()
        try:
            results = self._model.predict(
                frame.image,
                conf=self.cfg.conf_threshold,
                iou=self.cfg.iou_threshold,
                imgsz=self.cfg.imgsz,
                max_det=self.cfg.max_detections,
                verbose=False,
            )
        except Exception as exc:  # noqa: BLE001 - inference must not kill the loop
            self._record_failure(str(exc))
            raise DetectionError(f"hailo inference failed: {exc}") from exc

        elapsed_ms = (time.monotonic() - started) * 1000.0
        self._record_success(elapsed_ms)
        self._read_speed(results)
        return self._boxes_to_detections(
            results,
            self._names,
            camera_id or frame.camera_id,
            frame.seq,
            capture_timestamp or frame.timestamp,
        )

    def _read_speed(self, results: Any) -> None:
        """Pull Ultralytics' own pre/post-process timings when present."""
        try:
            speed = getattr(results[0], "speed", None) or {}
            self.last_preprocess_ms = float(speed.get("preprocess", 0.0) or 0.0)
            self.last_postprocess_ms = float(speed.get("postprocess", 0.0) or 0.0)
        except Exception:  # noqa: BLE001 - diagnostics only
            self.last_preprocess_ms = self.last_postprocess_ms = 0.0


# ---------------------------------------------------------------------------
# PyTorch (development only)
# ---------------------------------------------------------------------------


def resolve_torch_device(requested: str) -> str:
    """Pick a torch device for the DEVELOPMENT backend only.

    This never runs on the competition machine: production inference is on
    the Hailo-8. On a workstation it will use a discrete GPU if one
    happens to be present, otherwise the CPU.
    """
    if requested != "auto":
        return requested
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda:0"
    except ImportError:
        return "cpu"
    return "cpu"


class UltralyticsPyTorchBackend(DetectorBackend):
    """PyTorch ``.pt`` inference -- DEVELOPMENT AND TESTING ONLY.

    Use it on a laptop against saved images and video, to verify a freshly
    trained model, and to develop the rest of the pipeline without the
    accelerator. It is selected only when the configuration explicitly
    asks for it, and ``run.py --check`` raises a WARNING if it is selected
    while ``platform`` says Raspberry Pi -- because running YOLO on the
    Pi's CPU during a match would starve capture, AprilTags and the main
    loop of the cores they need.
    """

    name = "pytorch"

    def __init__(self, cfg: DetectorConfig) -> None:
        super().__init__(cfg)
        self._model: Any = None
        self._names: Dict[int, str] = {}
        self.device: str = ""
        self._health.model_path = cfg.pytorch_model_path

    def load(self) -> bool:
        if self._model is not None:
            return True
        path = self.cfg.pytorch_model_path
        if not os.path.exists(path):
            self._health.last_error = (
                f"model file not found: {path!r} "
                "(train one with training/, then copy best.pt here)"
            )
            logger.error("detector pytorch: %s", self._health.last_error)
            return False
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            self._health.last_error = f"ultralytics not installed: {exc}"
            logger.error("detector pytorch: %s", self._health.last_error)
            return False
        try:
            self.device = resolve_torch_device(self.cfg.pytorch_device)
            model = YOLO(path)
            names = getattr(model, "names", None) or {}
            self._names = {int(k): str(v) for k, v in dict(names).items()}
            self._check_classes(self._names)
            self._model = model
            self._health.loaded = True
            self._health.healthy = True
            self._health.device_report = f"torch:{self.device}"
            self._health.last_error = ""
            logger.warning(
                "detector pytorch: MODEL LOADED %s on %s (%s) -- DEVELOPMENT "
                "BACKEND. Production inference belongs on the Hailo-8.",
                path, self.device, self._health.class_report,
            )
            return True
        except Exception as exc:  # noqa: BLE001 - a bad weights file can raise anything
            self._health.last_error = f"failed to load {path!r}: {exc}"
            logger.exception("detector pytorch: %s", self._health.last_error)
            self._model = None
            return False

    def close(self) -> None:
        self._model = None
        self._health.healthy = False
        self._health.loaded = False

    def predict(
        self, frame: Frame, camera_id: str = "", capture_timestamp: float = 0.0
    ) -> List[Detection]:
        if self._model is None:
            return []
        started = time.monotonic()
        try:
            results = self._model.predict(
                frame.image,
                conf=self.cfg.conf_threshold,
                iou=self.cfg.iou_threshold,
                imgsz=self.cfg.imgsz,
                device=self.device or None,
                max_det=self.cfg.max_detections,
                verbose=False,
            )
        except Exception as exc:  # noqa: BLE001
            self._record_failure(str(exc))
            raise DetectionError(f"pytorch inference failed: {exc}") from exc

        elapsed_ms = (time.monotonic() - started) * 1000.0
        self._record_success(elapsed_ms)
        return self._boxes_to_detections(
            results,
            self._names,
            camera_id or frame.camera_id,
            frame.seq,
            capture_timestamp or frame.timestamp,
        )


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

BACKENDS = {
    "hailo": HailoDetectorBackend,
    "pytorch": UltralyticsPyTorchBackend,
}


def build_detector_backend(cfg: DetectorConfig) -> DetectorBackend:
    """Construct the configured backend.  The ONLY place a backend is chosen.

    There is no automatic fallback: if ``backend`` is ``hailo`` and the
    accelerator is missing, the result is an unhealthy Hailo backend, not
    a silently substituted CPU one.
    """
    try:
        backend_class = BACKENDS[cfg.backend]
    except KeyError:
        raise ValueError(
            f"unknown detector backend {cfg.backend!r}; expected one of {sorted(BACKENDS)}"
        ) from None
    return backend_class(cfg)


__all__ = [
    "BACKENDS",
    "DetectionError",
    "DetectorBackend",
    "HailoDetectorBackend",
    "UltralyticsPyTorchBackend",
    "build_detector_backend",
    "compare_class_names",
    "find_hef",
    "names_from_metadata",
    "read_hailo_metadata",
    "resolve_torch_device",
]
