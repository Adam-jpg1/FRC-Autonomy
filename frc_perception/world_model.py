"""Field-coordinate world model: cross-camera dedupe, tracking, velocity.

Detections arrive already projected into FIELD coordinates. Tracks are
matched by greedy nearest-neighbor within a per-role gate, so the same
ball seen by both cameras collapses into one track. Velocity is an EMA
of finite differences. Tracks age out when unseen.

If the robot pose is invalid, `update` must NOT be called with new
detections (there is no valid projection); call `prune` instead so stale
tracks still die.
"""

import math
from dataclasses import dataclass, field
from itertools import count
from typing import List, Optional, Tuple

from .config import WorldModelConfig
from .field_map import FieldMap

_track_ids = count(1)


@dataclass
class Track:
    class_name: str            # "fuel", "bumper_blue", "bumper_red"
    role: str                  # "game_piece" or "robot"
    x: float                   # smoothed position
    y: float
    vx: float = 0.0
    vy: float = 0.0
    last_seen: float = 0.0
    hits: int = 1
    # Last RAW measurement — velocity is differenced from these, not from
    # the smoothed position (which would bias velocity high by 1/alpha).
    meas_x: float = 0.0
    meas_y: float = 0.0
    track_id: int = field(default_factory=lambda: next(_track_ids))

    def __post_init__(self):
        self.meas_x = self.x
        self.meas_y = self.y

    def distance_to(self, x: float, y: float) -> float:
        return math.hypot(self.x - x, self.y - y)


@dataclass
class FieldDetection:
    """A detection already projected to field coordinates."""

    class_name: str
    role: str
    x: float
    y: float
    confidence: float


class WorldModel:
    def __init__(self, cfg: WorldModelConfig, field_map: FieldMap):
        self.cfg = cfg
        self.field_map = field_map
        self.tracks: List[Track] = []

    # -- update ---------------------------------------------------------------

    def update(self, detections: List[FieldDetection], now: float) -> None:
        """Associate this cycle's detections (all cameras) with tracks.

        Call once per perception cycle with a VALID robot pose only.
        """
        detections = self._merge_same_cycle(
            [
                d for d in detections
                if self.field_map.contains(d.x, d.y, margin=-self.cfg.field_margin_m)
            ]
        )
        # Greedy nearest-neighbor: consider all (track, det) pairs of the
        # same class within gate, closest first, each side used once.
        pairs = []
        for ti, track in enumerate(self.tracks):
            for di, det in enumerate(detections):
                if det.class_name != track.class_name:
                    continue
                dist = track.distance_to(det.x, det.y)
                if dist <= self._gate(det.role):
                    pairs.append((dist, ti, di))
        pairs.sort(key=lambda p: p[0])
        used_tracks, used_dets = set(), set()
        for _, ti, di in pairs:
            if ti in used_tracks or di in used_dets:
                continue
            used_tracks.add(ti)
            used_dets.add(di)
            self._update_track(self.tracks[ti], detections[di], now)
        for di, det in enumerate(detections):
            if di not in used_dets:
                self.tracks.append(
                    Track(det.class_name, det.role, det.x, det.y, last_seen=now)
                )
        self.prune(now)

    def _update_track(self, track: Track, det: FieldDetection, now: float) -> None:
        dt = now - track.last_seen
        if dt >= self.cfg.min_dt_s:
            vx = (det.x - track.meas_x) / dt
            vy = (det.y - track.meas_y) / dt
            speed = math.hypot(vx, vy)
            if speed <= self.cfg.max_speed_game_piece_mps or track.role == "robot":
                a = self.cfg.velocity_alpha
                track.vx += a * (vx - track.vx)
                track.vy += a * (vy - track.vy)
        a = self.cfg.position_alpha
        track.x += a * (det.x - track.x)
        track.y += a * (det.y - track.y)
        track.meas_x = det.x
        track.meas_y = det.y
        track.last_seen = now
        track.hits += 1

    def _merge_same_cycle(self, detections: List[FieldDetection]) -> List[FieldDetection]:
        """Fuse duplicate sightings of one object within a single cycle.

        Both cameras can see the same ball at once; without this each
        camera would spawn its own track. Same-class detections within
        the association gate are averaged, weighted by confidence.
        """
        merged: List[FieldDetection] = []
        weights: List[float] = []
        for det in sorted(detections, key=lambda d: -d.confidence):
            for i, existing in enumerate(merged):
                if (
                    existing.class_name == det.class_name
                    and math.hypot(existing.x - det.x, existing.y - det.y)
                    <= self._gate(det.role)
                ):
                    w = weights[i] + det.confidence
                    merged[i] = FieldDetection(
                        existing.class_name,
                        existing.role,
                        (existing.x * weights[i] + det.x * det.confidence) / w,
                        (existing.y * weights[i] + det.y * det.confidence) / w,
                        max(existing.confidence, det.confidence),
                    )
                    weights[i] = w
                    break
            else:
                merged.append(det)
                weights.append(det.confidence)
        return merged

    def prune(self, now: float) -> None:
        """Drop tracks not seen recently. Safe to call with pose invalid."""
        self.tracks = [
            t for t in self.tracks
            if (now - t.last_seen) <= self._timeout(t.role)
        ]

    # -- queries --------------------------------------------------------------

    def game_pieces(self) -> List[Track]:
        """Confirmed fuel tracks."""
        return [
            t for t in self.tracks
            if t.role == "game_piece" and t.hits >= self.cfg.min_hits_to_confirm
        ]

    def opponents(self, is_red_alliance: Optional[bool]) -> List[Track]:
        """Confirmed OPPONENT robot tracks. Unknown alliance -> report none
        (never guess: reporting teammates as obstacles is worse)."""
        if is_red_alliance is None:
            return []
        opponent_class = "bumper_blue" if is_red_alliance else "bumper_red"
        return [
            t for t in self.tracks
            if t.class_name == opponent_class
            and t.hits >= self.cfg.min_hits_to_confirm
        ]

    # -- internals -------------------------------------------------------------

    def _gate(self, role: str) -> float:
        return (
            self.cfg.gate_robot_m if role == "robot"
            else self.cfg.gate_game_piece_m
        )

    def _timeout(self, role: str) -> float:
        return (
            self.cfg.track_timeout_robot_s if role == "robot"
            else self.cfg.track_timeout_game_piece_s
        )


def positions(tracks: List[Track]) -> Tuple[List[float], List[float]]:
    """Split tracks into parallel x/y lists for NT double-array publishing."""
    return [t.x for t in tracks], [t.y for t in tracks]
