"""YOLO object detection (Ultralytics), wrapped so failures stay local.

Contract with the rest of the system:

* The model is loaded lazily and a load failure is NOT fatal.  The
  application keeps running with ``loaded == False`` and publishes
  ``model_loaded = false`` so the drive team can see why nothing is being
  detected.
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
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Sequence, Tuple

from .config import DetectorConfig
from .types import Detection, Frame

logger = logging.getLogger(__name__)


def resolve_device(requested: str) -> str:
    """Turn ``"auto"`` into a concrete torch device string.

    Falls back to CPU (slow but working) rather than failing when CUDA is
    missing -- on a laptop that is exactly what a developer wants, and on
    the Jetson the CPU fallback plus the logged warning is a much better
    symptom than a crash at a competition.
    """
    if requested != "auto":
        return requested
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda:0"
        logger.warning(
            "torch reports no CUDA device; running YOLO on CPU. On the Jetson this "
            "means the NVIDIA CUDA torch wheel is not installed (see requirements.txt)."
        )
    except ImportError:
        logger.warning("torch not importable; letting Ultralytics choose the device")
        return "cpu"
    return "cpu"


def compare_class_names(
    model_names: Dict[int, str], expected: Sequence[str]
) -> Tuple[bool, str]:
    """Compare a model's class names with the expected set.

    Order does not matter (routing is by name) but the SET must match
    exactly: a missing class means blind spots, an extra class means the
    weights are not the ones this configuration describes.
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
    """Ultralytics YOLO wrapped for predictable, contained behavior."""

    def __init__(self, cfg: DetectorConfig) -> None:
        self.cfg = cfg
        self._model: Any = None
        self._names: Dict[int, str] = {}
        self.load_error: str = ""
        self.classes_ok: bool = False
        self.class_report: str = "not loaded"
        self.device: str = ""
        self.last_inference_ms: float = 0.0
        self._inference_count: int = 0
        self._inference_total_ms: float = 0.0

    # -- lifecycle ---------------------------------------------------------

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def mean_inference_ms(self) -> float:
        if self._inference_count == 0:
            return 0.0
        return self._inference_total_ms / self._inference_count

    def load(self) -> bool:
        """Load the weights.  Returns True on success; never raises."""
        if self._model is not None:
            return True
        path = self.cfg.model_path
        if not os.path.exists(path):
            self.load_error = (
                f"model file not found: {path!r} "
                "(train one with training/, then copy best.pt here)"
            )
            logger.error("detector: %s", self.load_error)
            return False
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            self.load_error = f"ultralytics not installed: {exc}"
            logger.error("detector: %s", self.load_error)
            return False

        try:
            self.device = resolve_device(self.cfg.device)
            model = YOLO(path)
            names = getattr(model, "names", None) or {}
            self._names = {int(k): str(v) for k, v in dict(names).items()}
            self.classes_ok, self.class_report = compare_class_names(
                self._names, self.cfg.class_names
            )
            if not self.classes_ok:
                # Not fatal: routing is by name, so unmatched classes are simply
                # never emitted. Loud, though -- this is almost always the wrong
                # weights file.
                logger.error("detector: %s (weights=%s)", self.class_report, path)
            self._model = model
            logger.info(
                "detector: MODEL LOADED %s on %s (%s)", path, self.device, self.class_report
            )
            return True
        except Exception as exc:  # noqa: BLE001 - a bad weights file can raise anything
            self.load_error = f"failed to load {path!r}: {exc}"
            logger.exception("detector: %s", self.load_error)
            self._model = None
            return False

    # -- inference ---------------------------------------------------------

    def detect(self, frame: Frame) -> List[Detection]:
        """Run detection on one frame.

        Returns an empty list -- never raises -- when the model is not
        loaded or inference fails; the caller decides how loudly to
        complain.  ``predict`` is called with ``stream=False`` and no
        persistent state so two cameras cannot interfere.
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
            self.last_inference_ms = (time.monotonic() - started) * 1000.0
            raise DetectionError(str(exc)) from exc

        elapsed_ms = (time.monotonic() - started) * 1000.0
        self.last_inference_ms = elapsed_ms
        self._inference_count += 1
        self._inference_total_ms += elapsed_ms
        return self._convert(results, frame)

    def _convert(self, results: Any, frame: Frame) -> List[Detection]:
        detections: List[Detection] = []
        allowed = set(self.cfg.class_names)
        for result in results or []:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            names = {int(k): str(v) for k, v in dict(getattr(result, "names", self._names)).items()}
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
                        camera_name=frame.camera_name,
                        frame_seq=frame.seq,
                        timestamp=frame.timestamp,
                        class_name=class_name,
                        confidence=confidence,
                        bbox=(x1, y1, x2, y2),
                    )
                )
        return detections


class DetectionError(Exception):
    """Inference failed for one frame.  The loop catches and counts these."""


__all__ = [
    "DetectionError",
    "YoloDetector",
    "compare_class_names",
    "resolve_device",
]
