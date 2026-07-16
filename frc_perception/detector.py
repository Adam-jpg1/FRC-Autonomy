"""YOLO object detection (Ultralytics), stateless per frame.

Uses `.predict` — deliberately NOT `.track`. Tracker state is per-model
and would be corrupted by interleaving frames from two cameras; all
cross-frame association happens later in world_model, in field space.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from .config import CLASS_ROLES, DetectorConfig

log = logging.getLogger(__name__)


@dataclass
class Detection:
    """One detection in one image."""

    class_name: str
    role: str                 # "game_piece" or "robot", from CLASS_ROLES
    confidence: float
    bbox_xyxy: Tuple[float, float, float, float]


class Detector:
    """Wraps one Ultralytics model shared by all cameras."""

    def __init__(self, cfg: DetectorConfig):
        self.cfg = cfg
        from ultralytics import YOLO  # local import: heavy, GPU-touching

        self.model = YOLO(cfg.model_path, task="detect")
        self.device = self._resolve_device(cfg.device)

        # Map the model's class indices to configured names, by NAME.
        model_names: Dict[int, str] = dict(self.model.names)
        self.wanted_ids = {
            idx: name for idx, name in model_names.items() if name in cfg.class_names
        }
        missing = set(cfg.class_names) - set(model_names.values())
        if missing:
            log.error(
                "model %s lacks required classes %s (has %s) — those roles "
                "will never be detected",
                cfg.model_path, sorted(missing), sorted(model_names.values()),
            )
        log.info("detector ready on %s, classes %s", self.device, self.wanted_ids)

    @staticmethod
    def _resolve_device(requested: str) -> str:
        if requested != "auto":
            return requested
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda:0"
        except ImportError:
            pass
        log.warning("CUDA unavailable — running YOLO on CPU (will be slow)")
        return "cpu"

    def detect_batch(self, frames: Sequence) -> List[List[Detection]]:
        """Run stateless prediction on a batch of BGR images.

        Returns one detection list per input frame, order preserved.
        Batching both cameras into one call keeps the GPU efficient.
        """
        if not frames:
            return []
        results = self.model.predict(
            list(frames),
            conf=self.cfg.conf_threshold,
            iou=self.cfg.iou_threshold,
            imgsz=self.cfg.imgsz,
            device=self.device,
            half=self.cfg.half and self.device.startswith("cuda"),
            verbose=False,
        )
        out: List[List[Detection]] = []
        for result in results:
            dets: List[Detection] = []
            boxes = result.boxes
            if boxes is not None:
                for cls_id, conf, xyxy in zip(
                    boxes.cls.tolist(), boxes.conf.tolist(), boxes.xyxy.tolist()
                ):
                    name = self.wanted_ids.get(int(cls_id))
                    if name is None:
                        continue
                    dets.append(
                        Detection(
                            class_name=name,
                            role=CLASS_ROLES[name],
                            confidence=float(conf),
                            bbox_xyxy=tuple(float(v) for v in xyxy),
                        )
                    )
            out.append(dets)
        return out
