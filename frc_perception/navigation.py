"""Navigation policy: chase the densest reachable fuel pocket, else search.

Outputs FIELD-FRAME velocity commands in normalized stick units [-1, 1];
the virtual controller converts them for the robot's drive bindings.

Degradation ladder (safety first):
  pose valid + balls    -> ACQUIRE the best pocket
  pose valid + no balls -> DRIVE to field center, then SPIN_SEARCH
  pose INVALID          -> SPIN_SEARCH IN PLACE ONLY. Zero translation,
                           always — driving to a "center" computed from a
                           bogus pose would send the robot into a wall.
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .config import NavConfig
from .field_map import FieldMap
from .geometry import segment_point_distance, wrap_angle
from .localization import PoseEstimate
from .world_model import Track

# Navigation states (strings so they log/publish readably).
STATE_IDLE = "IDLE"
STATE_ACQUIRE = "ACQUIRE"
STATE_DRIVE_TO_CENTER = "DRIVE_TO_CENTER"
STATE_SPIN_SEARCH = "SPIN_SEARCH"
STATE_SPIN_SEARCH_BLIND = "SPIN_SEARCH_BLIND"  # pose invalid


@dataclass
class NavCommand:
    """Field-frame normalized command. vx toward +X, vy toward +Y."""

    vx: float = 0.0
    vy: float = 0.0
    omega: float = 0.0        # CCW positive
    state: str = STATE_IDLE
    target: Optional[Tuple[float, float]] = None


class Navigator:
    def __init__(self, cfg: NavConfig, field_map: FieldMap):
        self.cfg = cfg
        self.field_map = field_map

    def update(
        self,
        pose: PoseEstimate,
        balls: List[Track],
        opponents: List[Track],
    ) -> NavCommand:
        if not pose.valid:
            # DEGRADED: no trustworthy position. Rotating in place is the
            # only safe motion — it can re-acquire tags without traveling.
            return NavCommand(
                omega=self.cfg.search_spin_omega, state=STATE_SPIN_SEARCH_BLIND
            )

        if balls:
            target = self._pick_pocket(pose, balls, opponents)
            return self._drive_to(pose, target, STATE_ACQUIRE)

        center = self.field_map.clamp(*self.field_map.center, self.cfg.wall_margin_m)
        dist = math.hypot(center[0] - pose.x, center[1] - pose.y)
        if dist <= self.cfg.arrive_tolerance_m:
            return NavCommand(
                omega=self.cfg.search_spin_omega, state=STATE_SPIN_SEARCH,
                target=center,
            )
        return self._drive_to(pose, center, STATE_DRIVE_TO_CENTER)

    # -- target selection -------------------------------------------------------

    def _pick_pocket(
        self,
        pose: PoseEstimate,
        balls: List[Track],
        opponents: List[Track],
    ) -> Tuple[float, float]:
        """Score each ball's neighborhood; return the best cluster's
        nearest member (that's the one the intake reaches first)."""
        best_score = -math.inf
        best_target = (balls[0].x, balls[0].y)
        for anchor in balls:
            cluster = [
                b for b in balls
                if anchor.distance_to(b.x, b.y) <= self.cfg.pocket_radius_m
            ]
            cx = sum(b.x for b in cluster) / len(cluster)
            cy = sum(b.y for b in cluster) / len(cluster)
            nearest = min(
                cluster, key=lambda b: math.hypot(b.x - pose.x, b.y - pose.y)
            )
            dist = math.hypot(nearest.x - pose.x, nearest.y - pose.y)
            score = float(len(cluster)) - self.cfg.distance_weight * dist
            if self._path_blocked(pose, (cx, cy), opponents):
                score -= self.cfg.blocked_penalty
            if score > best_score:
                best_score = score
                best_target = (nearest.x, nearest.y)
        return self.field_map.clamp(*best_target, self.cfg.wall_margin_m)

    def _path_blocked(
        self,
        pose: PoseEstimate,
        target: Tuple[float, float],
        opponents: List[Track],
    ) -> bool:
        return any(
            segment_point_distance(pose.x, pose.y, target[0], target[1], o.x, o.y)
            < self.cfg.avoid_radius_m
            for o in opponents
        )

    # -- motion -------------------------------------------------------------------

    def _drive_to(
        self, pose: PoseEstimate, target: Tuple[float, float], state: str
    ) -> NavCommand:
        """P-control toward a field point, facing the direction of travel
        (fuel intake is assumed forward-facing)."""
        ex, ey = target[0] - pose.x, target[1] - pose.y
        dist = math.hypot(ex, ey)
        if dist < 1e-6:
            return NavCommand(state=state, target=target)
        speed = min(
            self.cfg.max_speed,
            self.cfg.approach_kp * dist,
            self.cfg.max_speed * dist / max(self.cfg.slow_radius_m, 1e-6),
        )
        heading_error = wrap_angle(math.atan2(ey, ex) - pose.theta)
        omega = max(
            -self.cfg.max_omega,
            min(self.cfg.max_omega, self.cfg.heading_kp * heading_error),
        )
        # Don't translate hard while pointed the wrong way — turn first.
        if abs(heading_error) > math.pi / 2:
            speed *= 0.25
        return NavCommand(
            vx=speed * ex / dist,
            vy=speed * ey / dist,
            omega=omega,
            state=state,
            target=target,
        )
