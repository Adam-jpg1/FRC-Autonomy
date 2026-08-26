"""Navigation assistance: what the robot SHOULD do, given the world model.

This module is isolated from perception on purpose.  It never touches a
camera, a model, or NetworkTables; it is a pure function of

    robot pose + pose validity + fuel tracks + opponent tracks + field bounds

and it returns a :class:`DesiredMotion` RECOMMENDATION.  Nothing here
moves the robot -- the virtual controller adapter turns this into stick
values, and the roboRIO decides whether to honor them.

THE SAFETY RULE
---------------
If ``pose_valid`` is False the robot must NEVER translate.  Every
field-coordinate target -- a fuel pocket, the search point, field center
-- is meaningless when we do not know where we are; driving toward one
would send the robot in an arbitrary direction at full assist speed.  The
only motion permitted is an in-place search rotation, which needs no
field coordinates at all, and even that is configurable off.  This is
checked FIRST, before any target selection runs, and
``tests/test_navigation.py`` asserts it explicitly.

SCORING
-------
Fuel is grouped into "pockets" by single-link clustering, and pockets are
scored on density, distance, heading cost, track freshness and the risk
that an opponent sits on the straight-line path.  Every weight lives in
``NavigationConfig``; there are no scoring constants in this file.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from .config import NavigationConfig
from .field_map import FieldMap
from .geometry import clamp, segment_point_distance, wrap_angle
from .types import DesiredMotion, PoseEstimate, Track, TrackFreshness

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FieldBounds:
    """The only thing navigation needs to know about the field."""

    length_m: float
    width_m: float

    @staticmethod
    def from_field_map(field_map: Optional[FieldMap]) -> Optional["FieldBounds"]:
        if field_map is None:
            return None
        return FieldBounds(field_map.length_m, field_map.width_m)

    def contains(self, x: float, y: float, margin_m: float = 0.0) -> bool:
        return (
            margin_m <= x <= self.length_m - margin_m
            and margin_m <= y <= self.width_m - margin_m
        )

    @property
    def center(self) -> Tuple[float, float]:
        return (self.length_m / 2.0, self.width_m / 2.0)


@dataclass
class FuelPocket:
    """A cluster of fuel tracks considered as one target."""

    x: float
    y: float
    count: int
    mean_confidence: float
    distance_m: float
    heading_error_rad: float
    path_clearance_m: float
    blocked: bool
    score: float
    track_ids: List[int]


def cluster_fuel(tracks: Sequence[Track], radius_m: float) -> List[List[Track]]:
    """Single-link clustering of fuel tracks within ``radius_m``."""
    remaining = list(tracks)
    clusters: List[List[Track]] = []
    while remaining:
        cluster = [remaining.pop(0)]
        changed = True
        while changed:
            changed = False
            for candidate in list(remaining):
                if any(
                    math.hypot(candidate.x - member.x, candidate.y - member.y) <= radius_m
                    for member in cluster
                ):
                    cluster.append(candidate)
                    remaining.remove(candidate)
                    changed = True
        clusters.append(cluster)
    return clusters


def path_clearance(
    start: Tuple[float, float],
    end: Tuple[float, float],
    opponents: Sequence[Tuple[float, float]],
) -> float:
    """Smallest distance from the straight path to any opponent position."""
    if not opponents:
        return float("inf")
    return min(
        segment_point_distance(start[0], start[1], end[0], end[1], ox, oy)
        for ox, oy in opponents
    )


class Navigator:
    """Turns the world model into one motion recommendation per cycle."""

    def __init__(self, cfg: NavigationConfig, bounds: Optional[FieldBounds] = None) -> None:
        self.cfg = cfg
        self.bounds = bounds
        #: Pockets scored on the last call, for the debugging tools.
        self.last_pockets: List[FuelPocket] = []

    # -- public API --------------------------------------------------------

    def plan(
        self,
        pose_estimate: PoseEstimate,
        fuel_tracks: Sequence[Track],
        opponent_tracks: Sequence[Track],
        now: float,
    ) -> DesiredMotion:
        """Recommend a motion.  Safe by construction when the pose is bad."""
        self.last_pockets = []

        # ---- RULE 1: no valid pose => no field-relative translation. ----
        # This must stay the first branch in this function.
        if not pose_estimate.valid:
            omega = (
                self.cfg.search_rotation_radps
                if self.cfg.allow_search_rotation_when_pose_invalid
                else 0.0
            )
            return DesiredMotion(
                vx_mps=0.0,
                vy_mps=0.0,
                omega_radps=omega,
                reason=f"pose invalid ({pose_estimate.invalid_reason or 'unknown'}): "
                "translation suppressed",
                target=None,
                confidence=0.0,
                # No field coordinates were used, so this command is not
                # field-relative even when the drive normally is.
                field_relative=False,
            )

        pose = pose_estimate.pose
        obstacles = self._predicted_obstacles(opponent_tracks, now)
        pockets = self._score_pockets(pose, fuel_tracks, obstacles, now)
        self.last_pockets = pockets

        if pockets:
            best = pockets[0]
            return self._approach(pose, (best.x, best.y), obstacles, best, pose_estimate)

        return self._search(pose, obstacles, pose_estimate)

    # -- internals ---------------------------------------------------------

    def _predicted_obstacles(
        self, opponent_tracks: Sequence[Track], now: float
    ) -> List[Tuple[float, float]]:
        """Where opponents are expected to be, one horizon INTO THE FUTURE.

        Two separate extrapolations, and the distinction matters: first the
        track is brought up to the present (it was last seen some
        milliseconds ago), then it is projected forward by
        ``prediction_horizon_s``.  Avoiding where an opponent WAS is how you
        drive into where it is going.
        """
        horizon = self.cfg.prediction_horizon_s
        obstacles: List[Tuple[float, float]] = []
        for track in opponent_tracks:
            x, y = track.predict(now, horizon)
            if track.velocity_valid:
                x += track.vx * horizon
                y += track.vy * horizon
            obstacles.append((x, y))
        return obstacles

    @property
    def _safe_distance_m(self) -> float:
        return (
            self.cfg.opponent_radius_m
            + self.cfg.own_robot_radius_m
            + self.cfg.safety_margin_m
        )

    def _score_pockets(
        self,
        pose,
        fuel_tracks: Sequence[Track],
        obstacles: Sequence[Tuple[float, float]],
        now: float,
    ) -> List[FuelPocket]:
        pockets: List[FuelPocket] = []
        safe_distance = self._safe_distance_m

        for cluster in cluster_fuel(fuel_tracks, self.cfg.cluster_radius_m):
            weights = [max(track.confidence, 1e-6) for track in cluster]
            total = sum(weights)
            x = sum(t.x * w for t, w in zip(cluster, weights)) / total
            y = sum(t.y * w for t, w in zip(cluster, weights)) / total
            if self.bounds is not None and not self.bounds.contains(
                x, y, self.cfg.wall_margin_m
            ):
                # Never aim at a point we would have to hit a wall to reach.
                continue

            distance = math.hypot(x - pose.x, y - pose.y)
            if distance > self.cfg.max_target_distance_m:
                continue
            heading_error = abs(
                wrap_angle(math.atan2(y - pose.y, x - pose.x) - pose.theta)
            )
            clearance = path_clearance((pose.x, pose.y), (x, y), obstacles)
            blocked = clearance < safe_distance
            risk = (
                0.0
                if not math.isfinite(clearance) or clearance >= safe_distance
                else (safe_distance - clearance) / max(safe_distance, 1e-6)
            )
            mean_confidence = sum(t.confidence for t in cluster) / len(cluster)

            score = (
                self.cfg.weight_density * (len(cluster) - 1)
                + self.cfg.weight_freshness * mean_confidence
                - self.cfg.weight_distance * distance
                - self.cfg.weight_heading * heading_error
                - self.cfg.weight_obstacle_risk * risk
                - (self.cfg.blocked_path_penalty if blocked else 0.0)
            )
            pockets.append(
                FuelPocket(
                    x=x,
                    y=y,
                    count=len(cluster),
                    mean_confidence=mean_confidence,
                    distance_m=distance,
                    heading_error_rad=heading_error,
                    path_clearance_m=clearance,
                    blocked=blocked,
                    score=score,
                    track_ids=[t.track_id for t in cluster],
                )
            )

        pockets.sort(key=lambda pocket: pocket.score, reverse=True)
        return pockets

    def _avoidance_push(
        self, x: float, y: float, obstacles: Sequence[Tuple[float, float]]
    ) -> Tuple[float, float]:
        """Unit-ish vector pushing away from nearby opponents.

        A plain repulsion field: understandable, tunable with two numbers,
        and adequate for keeping a driver-assist command out of an
        opponent.  It is NOT a planner and will not route around a wall of
        robots -- that is the human driver's job, and pretending otherwise
        would be the more dangerous choice.
        """
        safe_distance = self._safe_distance_m
        push_x = push_y = 0.0
        for ox, oy in obstacles:
            dx, dy = x - ox, y - oy
            distance = math.hypot(dx, dy)
            if distance >= self.cfg.avoid_consider_radius_m:
                continue
            if distance < 1e-6:
                # Sitting exactly on an obstacle: push backward rather than
                # dividing by zero.
                push_x -= self.cfg.avoid_gain
                continue
            overlap = max(0.0, safe_distance - distance)
            if overlap <= 0.0:
                continue
            magnitude = self.cfg.avoid_gain * (overlap / max(safe_distance, 1e-6))
            push_x += magnitude * dx / distance
            push_y += magnitude * dy / distance
        return (push_x, push_y)

    def _approach(
        self,
        pose,
        target: Tuple[float, float],
        obstacles: Sequence[Tuple[float, float]],
        pocket: FuelPocket,
        pose_estimate: PoseEstimate,
    ) -> DesiredMotion:
        dx, dy = target[0] - pose.x, target[1] - pose.y
        distance = math.hypot(dx, dy)

        heading_error = wrap_angle(math.atan2(dy, dx) - pose.theta)
        omega = clamp(
            self.cfg.heading_kp * heading_error,
            -self.cfg.max_omega_radps,
            self.cfg.max_omega_radps,
        )

        if distance <= self.cfg.arrive_tolerance_m:
            return DesiredMotion(
                vx_mps=0.0,
                vy_mps=0.0,
                omega_radps=omega,
                reason=f"arrived at fuel pocket ({pocket.count} piece(s))",
                target=target,
                confidence=pose_estimate.quality,
            )

        speed = min(self.cfg.approach_kp * distance, self.cfg.max_speed_mps)
        if distance < self.cfg.slow_radius_m:
            speed *= max(distance / max(self.cfg.slow_radius_m, 1e-6), 0.0)

        direction_x, direction_y = dx / distance, dy / distance
        push_x, push_y = self._avoidance_push(pose.x, pose.y, obstacles)
        direction_x += push_x
        direction_y += push_y
        norm = math.hypot(direction_x, direction_y)
        if norm < 1e-6:
            return DesiredMotion(
                vx_mps=0.0,
                vy_mps=0.0,
                omega_radps=omega,
                reason="avoidance cancelled approach; holding position",
                target=target,
                confidence=pose_estimate.quality * 0.5,
            )
        direction_x /= norm
        direction_y /= norm

        reason = f"approach fuel pocket ({pocket.count} piece(s), {distance:.1f} m)"
        if push_x or push_y:
            reason += " with opponent avoidance"
        return DesiredMotion(
            vx_mps=speed * direction_x,
            vy_mps=speed * direction_y,
            omega_radps=omega,
            reason=reason,
            target=target,
            confidence=pose_estimate.quality * min(1.0, pocket.mean_confidence + 0.2),
        )

    def _search(
        self,
        pose,
        obstacles: Sequence[Tuple[float, float]],
        pose_estimate: PoseEstimate,
    ) -> DesiredMotion:
        """No usable fuel, but the pose IS valid: search."""
        omega = self.cfg.search_rotation_radps if self.cfg.rotate_while_searching else 0.0

        target = self.cfg.search_point_field
        if target is None:
            # No configured staging point: spin in place to acquire targets.
            # This is the conservative behavior and needs no MUST INPUT value.
            return DesiredMotion(
                vx_mps=0.0,
                vy_mps=0.0,
                omega_radps=self.cfg.search_rotation_radps,
                reason="no fuel known; rotating to search (no search point configured)",
                target=None,
                confidence=0.0,
            )

        dx, dy = target[0] - pose.x, target[1] - pose.y
        distance = math.hypot(dx, dy)
        if distance <= self.cfg.arrive_tolerance_m:
            return DesiredMotion(
                vx_mps=0.0,
                vy_mps=0.0,
                omega_radps=self.cfg.search_rotation_radps,
                reason="at search point; rotating to search",
                target=target,
                confidence=pose_estimate.quality * 0.5,
            )

        speed = min(self.cfg.approach_kp * distance, self.cfg.max_speed_mps)
        direction_x, direction_y = dx / distance, dy / distance
        push_x, push_y = self._avoidance_push(pose.x, pose.y, obstacles)
        direction_x += push_x
        direction_y += push_y
        norm = math.hypot(direction_x, direction_y)
        if norm < 1e-6:
            return DesiredMotion(
                vx_mps=0.0, vy_mps=0.0, omega_radps=omega,
                reason="avoidance cancelled search move; holding position",
                target=target, confidence=0.0,
            )
        return DesiredMotion(
            vx_mps=speed * direction_x / norm,
            vy_mps=speed * direction_y / norm,
            omega_radps=omega,
            reason="no fuel known; moving to search point",
            target=target,
            confidence=pose_estimate.quality * 0.5,
        )


def filter_usable_tracks(
    tracks: Sequence[Track],
    freshness_of,
    now: float,
    allow_aging: bool = True,
) -> List[Track]:
    """Drop expired tracks (and optionally aging ones) before planning.

    ``freshness_of`` is ``WorldModel.freshness``; passing it in keeps this
    module free of any dependency on the world model's internals.
    """
    usable = []
    for track in tracks:
        freshness = freshness_of(track, now)
        if freshness is TrackFreshness.EXPIRED:
            continue
        if freshness is TrackFreshness.AGING and not allow_aging:
            continue
        usable.append(track)
    return usable


__all__ = [
    "FieldBounds",
    "FuelPocket",
    "Navigator",
    "cluster_fuel",
    "filter_usable_tracks",
    "path_clearance",
]
