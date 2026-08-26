"""The official WPILib AprilTag field layout.

This module ONLY reads the official JSON that WPILib publishes for the
season.  Tag poses are never hard-coded here and the file is never
synthesized: a wrong tag pose puts the robot meters away from where it
thinks it is, with no symptom other than bad behavior on the field.

Get the file with ``python scripts/download_field_layout.py``.

WPILib layout JSON schema (2023 onward)::

    {
      "tags": [
        {"ID": 1,
         "pose": {"translation": {"x":.., "y":.., "z":..},
                  "rotation": {"quaternion": {"W":.., "X":.., "Y":.., "Z":..}}}},
        ...
      ],
      "field": {"length": .., "width": ..}
    }

Poses are in the WPILib blue-origin field frame, in meters.  The tag's
own frame has +X pointing OUT of the tag face toward a viewer, +Z up, and
+Y completing the right-handed set (which is the viewer's right).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .geometry import make_transform, quaternion_to_rotation


class FieldLayoutError(Exception):
    """The layout file is missing, malformed, or internally inconsistent."""


@dataclass(frozen=True)
class TagPose:
    """One AprilTag's pose in the field frame."""

    tag_id: int
    #: 4x4 transform taking TAG-frame points into FIELD-frame points.
    field_from_tag: np.ndarray

    @property
    def translation(self) -> np.ndarray:
        return self.field_from_tag[:3, 3]


@dataclass
class FieldMap:
    """Loaded field layout: tag poses plus the field's outer dimensions."""

    tags: Dict[int, TagPose]
    length_m: float
    width_m: float
    source_path: str = ""

    def has(self, tag_id: int) -> bool:
        return tag_id in self.tags

    def get(self, tag_id: int) -> Optional[TagPose]:
        return self.tags.get(tag_id)

    @property
    def tag_ids(self) -> List[int]:
        return sorted(self.tags)

    def contains_point(self, x: float, y: float, margin_m: float = 0.0) -> bool:
        """Is (x, y) inside the field, allowing ``margin_m`` of slop?"""
        return (
            -margin_m <= x <= self.length_m + margin_m
            and -margin_m <= y <= self.width_m + margin_m
        )

    def clamp_point(self, x: float, y: float, margin_m: float = 0.0) -> tuple:
        """Clamp a point to the field interior, inset by ``margin_m``."""
        low_x, high_x = margin_m, max(margin_m, self.length_m - margin_m)
        low_y, high_y = margin_m, max(margin_m, self.width_m - margin_m)
        return (min(max(x, low_x), high_x), min(max(y, low_y), high_y))

    @property
    def center(self) -> tuple:
        return (self.length_m / 2.0, self.width_m / 2.0)


def parse_field_layout(payload: dict, source_path: str = "") -> FieldMap:
    """Validate and convert a parsed layout JSON into a :class:`FieldMap`.

    Raises :class:`FieldLayoutError` with a specific message rather than
    returning something half-usable.
    """
    if not isinstance(payload, dict):
        raise FieldLayoutError("layout root is not a JSON object")

    raw_tags = payload.get("tags")
    if not isinstance(raw_tags, list) or not raw_tags:
        raise FieldLayoutError("layout has no non-empty 'tags' array")

    field_block = payload.get("field")
    if not isinstance(field_block, dict):
        raise FieldLayoutError("layout has no 'field' object with length/width")
    try:
        length = float(field_block["length"])
        width = float(field_block["width"])
    except (KeyError, TypeError, ValueError) as exc:
        raise FieldLayoutError(f"bad field dimensions: {exc}") from exc
    if not (math.isfinite(length) and math.isfinite(width)) or length <= 0 or width <= 0:
        raise FieldLayoutError(f"implausible field dimensions {length}x{width}")

    tags: Dict[int, TagPose] = {}
    for index, entry in enumerate(raw_tags):
        try:
            tag_id = int(entry["ID"])
            pose = entry["pose"]
            translation = pose["translation"]
            quaternion = pose["rotation"]["quaternion"]
            position = [
                float(translation["x"]),
                float(translation["y"]),
                float(translation["z"]),
            ]
            rotation = quaternion_to_rotation(
                float(quaternion["W"]),
                float(quaternion["X"]),
                float(quaternion["Y"]),
                float(quaternion["Z"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise FieldLayoutError(f"tag entry #{index} is malformed: {exc}") from exc

        if tag_id in tags:
            raise FieldLayoutError(f"duplicate tag ID {tag_id} in layout")
        if not all(math.isfinite(v) for v in position):
            raise FieldLayoutError(f"tag {tag_id} has a non-finite translation")
        if not (-1.0 <= position[0] <= length + 1.0 and -1.0 <= position[1] <= width + 1.0):
            raise FieldLayoutError(
                f"tag {tag_id} at ({position[0]:.2f},{position[1]:.2f}) lies outside "
                f"the declared {length:.2f}x{width:.2f} field"
            )
        tags[tag_id] = TagPose(tag_id, make_transform(rotation, position))

    return FieldMap(tags=tags, length_m=length, width_m=width, source_path=source_path)


def load_field_map(path: str) -> FieldMap:
    """Load and validate the layout JSON at ``path``."""
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError as exc:
        raise FieldLayoutError(
            f"field layout not found at {path!r}; run scripts/download_field_layout.py"
        ) from exc
    except json.JSONDecodeError as exc:
        raise FieldLayoutError(f"field layout at {path!r} is not valid JSON: {exc}") from exc
    return parse_field_layout(payload, source_path=path)


def try_load_field_map(path: str):
    """Load the layout, returning ``(map_or_None, error_message)``.

    Used by the application, which must keep running (without AprilTag
    localization) when the layout is missing.
    """
    try:
        return load_field_map(path), ""
    except FieldLayoutError as exc:
        return None, str(exc)


__all__ = [
    "FieldLayoutError",
    "FieldMap",
    "TagPose",
    "load_field_map",
    "parse_field_layout",
    "try_load_field_map",
]
