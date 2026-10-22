"""YOLO object detection (Ultralytics on CUDA), wrapped so failures stay local.

This module is the HARDWARE BOUNDARY.  Everything above it is specific to
the Jetson Orin Nano Super -- CUDA, FP16, a TensorRT engine if you build
one.  Everything below it sees only
:class:`~frc_perception.types.Detection` and has no idea what produced it:
``geometry.py`` contains no CUDA code, ``world_model.py`` imports no
Ultralytics, and ``navigation.py`` has never heard of a ``.engine`` file.
Swapping the inference hardware should mean changing this file and one
config value, nothing more.

Contract with the rest of the system:

* The model is loaded ONCE at startup and kept resident.  It is never
  reloaded per frame -- see :meth:`YoloDetector.load`.
* A load failure is NOT fatal.  The application keeps running with
  ``healthy() == False`` and publishes ``model_loaded = false`` and
  ``detector_healthy = false`` so the drive team can see why nothing is
  being detected.  A detector that cannot run produces NOTHING: never a
  fabricated box, never a frozen last result.  Existing world-model tracks
  then age out on their own and navigation degrades safely.
* Inference uses ``model.predict(...)``.  ``model.track(...)`` is
  deliberately never called: Ultralytics' tracker keeps per-call state, so
  interleaving two cameras through one model would cross-contaminate their
  IDs.  Tracking is our job and happens in field space, in
  ``world_model.py``, where two cameras seeing the same object is a
  feature rather than an ID collision.
* Class routing is by NAME.  The model's class indices are mapped through
  its own ``names`` dictionary, and a model whose names do not match the
  configured classes is reported loudly instead of silently mis-labeling
  everything.

MODEL FORMATS
-------------
``model_path`` may be a ``.pt`` (start here -- easiest to validate) or a
TensorRT ``.engine`` built on the Jetson itself.  Ultralytics' AutoBackend
picks the right loader from the extension, so switching is a config change
and nothing in this file changes.  A ``.engine`` is tied to the exact GPU,
JetPack and TensorRT version that built it.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Sequence, Tuple

from .config import DetectorConfig
from .types import Detection, DetectorHealth, Frame

logger = logging.getLogger(__name__)


class DetectionError(Exception):
    """Inference failed for one frame.  The scheduler catches and counts these."""


def resolve_device(requested: str) -> str:
    """Turn ``"auto"`` into a concrete torch device string.

    Returns ``"cpu"`` when CUDA is unavailable.  Whether that is acceptable
    is a policy decision made by the caller, not here:
    :attr:`DetectorConfig.require_cuda` decides whether a CPU device is a
    usable configuration or a refused one.  On a development laptop CPU is
    exactly what you want; on the competition Jetson it means the CUDA
    torch wheel is not installed, and running YOLO on the Orin's CPU would
    starve capture, AprilTags and the main loop of the cores they need.
    """
    if requested != "auto":
        return requested
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda:0"
        logger.warning(
            "torch reports no CUDA device. On the Jetson this means the NVIDIA "
            "CUDA torch wheel is not installed (see requirements.txt)."
        )
    except ImportError:
        logger.warning("torch not importable; letting Ultralytics choose the device")
        return "cpu"
    return "cpu"


def compare_class_names(
    model_names: Dict[int, str], expected: Sequence[str]
) -> Tuple[bool, str]:
    """Compare a model's class names with the expected set.

    Order does not matter -- routing is by name, so re-exporting with a
    different class order cannot silently turn opponents into game pieces
    -- but the SET must match exactly: a missing class means a blind spot,
    an extra class means these are not the weights this configuration
    describes.
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


class YoloDetector:
    """Ultralytics YOLO wrapped for predictable, contained behavior.

    One instance serves BOTH cameras.  The model is resident for the life
    of the process; :class:`~frc_perception.inference_scheduler.InferenceScheduler`
    decides which camera's newest frame gets the GPU on each cycle.
    """

    name = "ultralytics"

    def __init__(self, cfg: DetectorConfig) -> None:
        self.cfg = cfg
        self._model: Any = None
        self._names: Dict[int, str] = {}
        self.device: str = ""

        self._health = DetectorHealth(backend=self.name, model_path=cfg.model_path)
        self._inference_count = 0
        self._latency_total_ms = 0.0
        self.last_latency_ms = 0.0
        self.last_preprocess_ms = 0.0
        self.last_postprocess_ms = 0.0

    # -- introspection -----------------------------------------------------

    @property
    def loaded(self) -> bool:
        return self._health.loaded

    def healthy(self) -> bool:
        """True only while the detector can actually produce detections."""
        return self._health.healthy

    @property
    def load_error(self) -> str:
        return self._health.last_error

    @property
    def classes_ok(self) -> bool:
        return self._health.classes_ok

    @property
    def class_report(self) -> str:
        return self._health.class_report

    @property
    def mean_inference_ms(self) -> float:
        if self._inference_count == 0:
            return 0.0
        return self._latency_total_ms / self._inference_count

    def health(self) -> DetectorHealth:
        """Snapshot of detector state.  The caller cannot mutate ours."""
        self._health.inferences = self._inference_count
        self._health.mean_latency_ms = self.mean_inference_ms
        return DetectorHealth(**vars(self._health))

    # -- lifecycle ---------------------------------------------------------

    def load(self) -> bool:
        """Load the weights ONCE and keep them resident.  Never raises.

        Calling this again after a successful load is a no-op: re-creating
        the model per frame would dominate the frame budget and, for a
        TensorRT engine, re-run deserialization every time.
        """
        if self._model is not None:
            return True

        path = self.cfg.model_path
        if not os.path.exists(path):
            self._health.last_error = (
                f"model file not found: {path!r} "
                "(train one with training/, then copy best.pt here)"
            )
            logger.error("detector: %s", self._health.last_error)
            return False
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            self._health.last_error = f"ultralytics not installed: {exc}"
            logger.error("detector: %s", self._health.last_error)
            return False

        try:
            self.device = resolve_device(self.cfg.device)
            if self.cfg.require_cuda and not self.device.startswith("cuda"):
                # Refusing is the safe answer on the robot: a silent CPU
                # fallback hides a broken CUDA install behind a few frames
                # per second while starving the rest of the pipeline.
                self._health.last_error = (
                    f"require_cuda is set but the resolved device is {self.device!r}. "
                    "Install NVIDIA's CUDA torch wheel (see requirements.txt), or "
                    "set detector.require_cuda = False for CPU development."
                )
                logger.error("detector: %s", self._health.last_error)
                return False

            model = YOLO(path)
            names = getattr(model, "names", None) or {}
            self._names = {int(k): str(v) for k, v in dict(names).items()}
            ok, report = compare_class_names(self._names, self.cfg.class_names)
            self._health.classes_ok = ok
            self._health.class_report = report
            if not ok:
                # Not fatal: routing is by name, so unmatched classes are
                # simply never emitted. Loud, though -- this is almost
                # always the wrong weights file.
                logger.error("detector: %s (weights=%s)", report, path)

            self._model = model
            self._health.loaded = True
            self._health.healthy = True
            self._health.device_report = self.device
            self._health.last_error = ""
            logger.info(
                "detector: MODEL LOADED %s on %s (%s)", path, self.device, report
            )
            return True
        except Exception as exc:  # noqa: BLE001 - a bad weights file can raise anything
            self._health.last_error = f"failed to load {path!r}: {exc}"
            logger.exception("detector: %s", self._health.last_error)
            self._model = None
            self._health.loaded = False
            self._health.healthy = False
            return False

    def close(self) -> None:
        """Release the model.  Never raises."""
        self._model = None
        self._health.loaded = False
        self._health.healthy = False

    # -- inference ---------------------------------------------------------

    def predict(
        self, frame: Frame, camera_id: str = "", capture_timestamp: float = 0.0
    ) -> List[Detection]:
        """Run detection on one frame.

        ``camera_id`` and ``capture_timestamp`` default to the frame's own
        values; they are accepted explicitly so a caller replaying stored
        images can supply them.

        Returns an empty list when the model is not loaded.  Raises
        :class:`DetectionError` -- and only that -- when inference itself
        fails, so the scheduler can count it against this camera without
        losing the other one.
        """
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
                half=self.cfg.half and self.device.startswith("cuda"),
                max_det=self.cfg.max_detections,
                verbose=False,
            )
        except Exception as exc:  # noqa: BLE001 - inference must not kill the loop
            self.last_latency_ms = (time.monotonic() - started) * 1000.0
            self._record_failure(str(exc))
            raise DetectionError(f"inference failed: {exc}") from exc

        elapsed_ms = (time.monotonic() - started) * 1000.0
        self._record_success(elapsed_ms)
        self._read_speed(results)
        return self._convert(
            results,
            camera_id or frame.camera_id,
            frame.seq,
            capture_timestamp or frame.timestamp,
        )

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
                    "detector: UNHEALTHY after %d consecutive failures (%s); no new "
                    "detections will be produced and tracks will age out",
                    self._health.consecutive_failures, message,
                )
            self._health.healthy = False

    def _read_speed(self, results: Any) -> None:
        """Pull Ultralytics' own pre/post-process timings when present."""
        try:
            speed = getattr(results[0], "speed", None) or {}
            self.last_preprocess_ms = float(speed.get("preprocess", 0.0) or 0.0)
            self.last_postprocess_ms = float(speed.get("postprocess", 0.0) or 0.0)
        except Exception:  # noqa: BLE001 - diagnostics only
            self.last_preprocess_ms = self.last_postprocess_ms = 0.0

    def _convert(
        self,
        results: Any,
        camera_id: str,
        frame_sequence: int,
        capture_timestamp: float,
    ) -> List[Detection]:
        """Convert Ultralytics ``Results`` into our hardware-independent type.

        One place where boxes become :class:`Detection`, and one place
        where the per-class confidence gate is applied.
        """
        detections: List[Detection] = []
        allowed = set(self.cfg.class_names)
        for result in results or []:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            names = {
                int(k): str(v)
                for k, v in dict(getattr(result, "names", self._names)).items()
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


__all__ = [
    "DetectionError",
    "YoloDetector",
    "compare_class_names",
    "resolve_device",
]
