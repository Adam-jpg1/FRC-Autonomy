"""Official WPILib AprilTag field layout: load, index, and query.

The layout JSON is the one WPILib ships (blue-origin field coordinates).
Tag orientation convention in that file: the tag's local +X axis points
OUT of the tag face (verified against past layouts, where blue-wall tags
facing downfield have yaw 0 and red-wall tags have yaw 180).
"""

import json
import logging
from typing import Dict, Optional

import numpy as np

from .geometry import make_transform, quat_to_rot

log = logging.getLogger(__name__)


class FieldMap:
    """Tag poses (4x4, field frame) and field dimensions from the layout."""

    def __init__(self, layout_path: str,
                 fallback_length_m: float, fallback_width_m: float):
        self.tag_poses: Dict[int, np.ndarray] = {}
        self.length_m = fallback_length_m
        self.width_m = fallback_width_m
        self.loaded = False
        try:
            with open(layout_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            log.error(
                "Field layout %s unusable (%s). AprilTag localization DISABLED; "
                "using fallback field size %.2fx%.2f m. Run "
                "scripts/download_field_layout.py.",
                layout_path, e, fallback_length_m, fallback_width_m,
            )
            return

        for tag in data.get("tags", []):
            t = tag["pose"]["translation"]
            q = tag["pose"]["rotation"]["quaternion"]
            rotation = quat_to_rot(q["W"], q["X"], q["Y"], q["Z"])
            self.tag_poses[int(tag["ID"])] = make_transform(
                rotation, (t["x"], t["y"], t["z"])
            )
        field = data.get("field", {})
        self.length_m = float(field.get("length", fallback_length_m))
        self.width_m = float(field.get("width", fallback_width_m))
        self.loaded = True
        log.info(
            "Field layout loaded: %d tags, field %.3f x %.3f m",
            len(self.tag_poses), self.length_m, self.width_m,
        )

    def tag_pose(self, tag_id: int) -> Optional[np.ndarray]:
        return self.tag_poses.get(tag_id)

    @property
    def center(self) -> tuple:
        return (self.length_m / 2.0, self.width_m / 2.0)

    def contains(self, x: float, y: float, margin: float = 0.0) -> bool:
        """True if (x, y) is inside the field, shrunk by `margin` on all sides."""
        return (margin <= x <= self.length_m - margin
                and margin <= y <= self.width_m - margin)

    def clamp(self, x: float, y: float, margin: float) -> tuple:
        """Clamp a point into the field shrunk by `margin`."""
        return (
            min(max(x, margin), self.length_m - margin),
            min(max(y, margin), self.width_m - margin),
        )
