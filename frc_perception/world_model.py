"""Persistent field-space world model: de-duplication, association, tracking.

This is deliberately OUR tracker rather than Ultralytics' -- see
``detector.py`` for why.  Tracking in field coordinates instead of image
coordinates buys three things that matter on a robot:

* the same object seen by two cameras is ONE object, not two IDs;
* an object stays in the same place while the robot spins, so a target
  does not evaporate when it leaves the frame;
* velocities come out in m/s in field coordinates, which is what
  navigation actually needs.

Pipeline per update, all pure functions of the inputs plus stored state:

    field observations (both cameras)
        -> cross-camera de-duplication (merge)
        -> association to existing tracks (gated nearest neighbor)
        -> position smoothing + velocity fit
        -> aging / expiry

VELOCITY
--------
Velocity is fitted by least squares over the RAW merged measurements in a
short window.  It is NOT computed by differencing the smoothed position:
the smoothing lags the truth by a fixed fraction, and differencing a
lagged signal against a less-lagged one biases speed low and can invert
its sign during direction changes.  ``tests/test_world_model.py`` pins
this behavior down with a constant-velocity object.
"""

from __future__ import annotations

import itertools
import logging
import math
from collections import deque
from typing import Deque, Dict, Iterable, List, Optional, Sequence, Tuple

from .alliance import is_obstacle
from .config import WorldModelConfig
from .field_map import FieldMap
from .types import FieldObservation, ObjectRole, Track, TrackFreshness

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cross-camera de-duplication
# ---------------------------------------------------------------------------


def merge_observations(
    observations: Sequence[FieldObservation],
    merge_distance_game_piece_m: float,
    merge_distance_robot_m: float,
) -> List[FieldObservation]:
    """Collapse observations of the same physical object into one.

    Two observations merge when they share a class name and lie within the
    class's merge radius.  Merging is single-link (transitive) so three
    overlapping sightings become one object rather than two.

    The merged position is a confidence-weighted mean, which lets a
    confident close-range sighting dominate a marginal distant one, and
    the merged confidence is the MAXIMUM rather than the sum: seeing the
    same ball twice does not make it more certainly a ball, it just gives
    a better position.
    """
    remaining = list(observations)
    merged: List[FieldObservation] = []

    while remaining:
        seed = remaining.pop(0)
        group = [seed]
        radius = (
            merge_distance_game_piece_m
            if seed.role is ObjectRole.GAME_PIECE
            else merge_distance_robot_m
        )
        changed = True
        while changed:
            changed = False
            for candidate in list(remaining):
                if candidate.class_name != seed.class_name:
                    continue
                if any(
                    math.hypot(candidate.x_field_m - member.x_field_m,
                               candidate.y_field_m - member.y_field_m) <= radius
                    for member in group
                ):
                    group.append(candidate)
                    remaining.remove(candidate)
                    changed = True
        merged.append(_combine(group))
    return merged


def _combine(group: List[FieldObservation]) -> FieldObservation:
    if len(group) == 1:
        return group[0]
    weights = [max(o.confidence * o.quality, 1e-6) for o in group]
    total = sum(weights)
    x = sum(o.x_field_m * w for o, w in zip(group, weights)) / total
    y = sum(o.y_field_m * w for o, w in zip(group, weights)) / total
    best = max(group, key=lambda o: o.confidence * o.quality)
    return FieldObservation(
        camera_id="+".join(sorted({o.camera_id for o in group})),
        timestamp=max(o.timestamp for o in group),
        class_name=best.class_name,
        role=best.role,
        confidence=max(o.confidence for o in group),
        x_field_m=x,
        y_field_m=y,
        range_m=min(o.range_m for o in group),
        quality=max(o.quality for o in group),
        merged_count=sum(o.merged_count for o in group),
    )


# ---------------------------------------------------------------------------
# Velocity estimation
# ---------------------------------------------------------------------------


def fit_velocity(
    samples: Sequence[Tuple[float, float, float]],
    min_samples: int,
    min_span_s: float,
    max_speed_mps: float,
) -> Tuple[float, float, bool]:
    """Least-squares velocity from raw ``(t, x, y)`` samples.

    Returns ``(vx, vy, valid)``.  Invalid when there are too few samples,
    when they span too little time (where noise divided by a tiny dt
    explodes), or when the fitted speed is physically impossible -- an
    absurd velocity is worse than no velocity, because navigation would
    extrapolate on it.
    """
    if len(samples) < max(min_samples, 2):
        return (0.0, 0.0, False)
    times = [s[0] for s in samples]
    span = times[-1] - times[0]
    if span < min_span_s or span <= 0.0:
        return (0.0, 0.0, False)

    mean_t = sum(times) / len(times)
    denominator = sum((t - mean_t) ** 2 for t in times)
    if denominator <= 1e-12:
        return (0.0, 0.0, False)

    mean_x = sum(s[1] for s in samples) / len(samples)
    mean_y = sum(s[2] for s in samples) / len(samples)
    vx = sum((t - mean_t) * (s[1] - mean_x) for t, s in zip(times, samples)) / denominator
    vy = sum((t - mean_t) * (s[2] - mean_y) for t, s in zip(times, samples)) / denominator

    speed = math.hypot(vx, vy)
    if not math.isfinite(speed) or speed > max_speed_mps:
        return (0.0, 0.0, False)
    return (vx, vy, True)


# ---------------------------------------------------------------------------
# The world model
# ---------------------------------------------------------------------------


class _TrackState:
    """A track plus the private history used to fit its velocity."""

    __slots__ = ("track", "samples", "created")

    def __init__(self, track: Track, created: float) -> None:
        self.track = track
        self.created = created
        self.samples: Deque[Tuple[float, float, float]] = deque()


class WorldModel:
    """Field-space tracks for game pieces and robots."""

    def __init__(self, cfg: WorldModelConfig, field_map: Optional[FieldMap] = None) -> None:
        self.cfg = cfg
        self.field_map = field_map
        self._states: Dict[int, _TrackState] = {}
        self._next_id = itertools.count(1)

    # -- queries -----------------------------------------------------------

    @property
    def all_tracks(self) -> List[Track]:
        """Every live track, including unconfirmed ones (diagnostics only)."""
        return [state.track for state in self._states.values()]

    def tracks(self, role: Optional[ObjectRole] = None, now: Optional[float] = None) -> List[Track]:
        """Confirmed, sufficiently confident tracks, optionally by role."""
        results = []
        for state in self._states.values():
            track = state.track
            if role is not None and track.role is not role:
                continue
            if track.hit_count < self.cfg.min_hits_to_confirm:
                continue
            confidence = (
                track.confidence
                if now is None
                else self._decayed_confidence(track, now)
            )
            if confidence < self.cfg.min_report_confidence:
                continue
            results.append(track)
        return results

    def game_pieces(self, now: Optional[float] = None) -> List[Track]:
        return self.tracks(ObjectRole.GAME_PIECE, now)

    def opponents(self, now: Optional[float] = None) -> List[Track]:
        """Robots positively identified as opposing."""
        return self.tracks(ObjectRole.OPPONENT, now)

    def obstacle_robots(self, now: Optional[float] = None) -> List[Track]:
        """Every robot that must be avoided.

        This is what navigation and the ``opponents_field_*`` NT keys use.
        With a known alliance it equals :meth:`opponents`; with the alliance
        unknown it also includes unclassifiable robots, because refusing to
        drive through a robot we cannot identify is always safe and the
        opposite is not.  Friendly robots are never included.
        """
        treat_unknown = self.cfg.treat_unknown_robots_as_obstacles
        return [
            track
            for track in self.tracks(None, now)
            if track.role is not ObjectRole.GAME_PIECE
            and is_obstacle(track.role, treat_unknown)
        ]

    def freshness(self, track: Track, now: float) -> TrackFreshness:
        """Classify how much this track can still be trusted."""
        staleness = track.staleness(now)
        fresh_limit, expire_limit = self._timeouts(track.role)
        if staleness <= fresh_limit:
            return TrackFreshness.RECENT
        if staleness <= expire_limit:
            return TrackFreshness.AGING
        return TrackFreshness.EXPIRED

    # -- update ------------------------------------------------------------

    def update(self, observations: Sequence[FieldObservation], now: float) -> None:
        """Fold one cycle's observations in and age everything else out.

        Call this even with an EMPTY observation list -- that is how tracks
        age and expire when the cameras stop seeing things.
        """
        merged = merge_observations(
            self._filter_to_field(observations),
            self.cfg.merge_distance_game_piece_m,
            self.cfg.merge_distance_robot_m,
        )
        assignments, unmatched = self._associate(merged, now)

        for track_id, observation in assignments.items():
            self._update_track(self._states[track_id], observation, now)
        for observation in unmatched:
            self._create_track(observation, now)

        matched_ids = set(assignments)
        for state in self._states.values():
            if state.track.track_id not in matched_ids:
                state.track.misses += 1
            state.track.age = now - state.created
            state.track.confidence = self._decayed_confidence(state.track, now)

        self._expire(now)

    def _filter_to_field(
        self, observations: Iterable[FieldObservation]
    ) -> List[FieldObservation]:
        """Drop observations that land outside the field.

        A projection well off the field means the pose, the mount angles or
        the range was wrong; keeping it would create a phantom obstacle.
        """
        if self.field_map is None:
            return list(observations)
        kept = []
        for observation in observations:
            if self.field_map.contains_point(
                observation.x_field_m, observation.y_field_m, self.cfg.field_margin_m
            ):
                kept.append(observation)
        return kept

    def _timeouts(self, role: ObjectRole) -> Tuple[float, float]:
        if role is ObjectRole.GAME_PIECE:
            return (self.cfg.fresh_timeout_game_piece_s, self.cfg.expire_timeout_game_piece_s)
        return (self.cfg.fresh_timeout_robot_s, self.cfg.expire_timeout_robot_s)

    def _gate(self, role: ObjectRole) -> float:
        return (
            self.cfg.gate_game_piece_m
            if role is ObjectRole.GAME_PIECE
            else self.cfg.gate_robot_m
        )

    def _max_speed(self, role: ObjectRole) -> float:
        return (
            self.cfg.max_speed_game_piece_mps
            if role is ObjectRole.GAME_PIECE
            else self.cfg.max_speed_robot_mps
        )

    def _associate(
        self, observations: Sequence[FieldObservation], now: float
    ) -> Tuple[Dict[int, FieldObservation], List[FieldObservation]]:
        """Greedy nearest-neighbor association with a per-class gate.

        Greedy (closest pair first) rather than optimal assignment: with
        the handful of objects an FRC camera sees, the two agree almost
        always, and greedy is trivial to reason about when it does not.
        Tracks are compared at their PREDICTED position so a moving robot
        associates correctly.
        """
        pairs: List[Tuple[float, int, int]] = []
        for observation_index, observation in enumerate(observations):
            for track_id, state in self._states.items():
                track = state.track
                if track.class_name != observation.class_name:
                    continue
                predicted_x, predicted_y = track.predict(
                    now, self.cfg.max_prediction_horizon_s
                )
                distance = math.hypot(
                    observation.x_field_m - predicted_x, observation.y_field_m - predicted_y
                )
                if distance <= self._gate(track.role):
                    pairs.append((distance, track_id, observation_index))

        pairs.sort()
        assignments: Dict[int, FieldObservation] = {}
        used_observations: set = set()
        for _distance, track_id, observation_index in pairs:
            if track_id in assignments or observation_index in used_observations:
                continue
            assignments[track_id] = observations[observation_index]
            used_observations.add(observation_index)

        unmatched = [
            observation
            for index, observation in enumerate(observations)
            if index not in used_observations
        ]
        return assignments, unmatched

    def _create_track(self, observation: FieldObservation, now: float) -> Track:
        track = Track(
            track_id=next(self._next_id),
            class_name=observation.class_name,
            role=observation.role,
            x=observation.x_field_m,
            y=observation.y_field_m,
            confidence=observation.confidence * observation.quality,
            last_seen=observation.timestamp,
            age=0.0,
            hit_count=1,
        )
        state = _TrackState(track, created=now)
        state.samples.append((observation.timestamp, observation.x_field_m, observation.y_field_m))
        self._states[track.track_id] = state
        return track

    def _update_track(self, state: _TrackState, observation: FieldObservation, now: float) -> None:
        track = state.track
        alpha = self.cfg.position_alpha
        track.x += alpha * (observation.x_field_m - track.x)
        track.y += alpha * (observation.y_field_m - track.y)
        track.last_seen = observation.timestamp
        track.hit_count += 1
        track.misses = 0
        track.confidence = max(track.confidence, observation.confidence * observation.quality)
        # A robot that turns out to be an opponent (alliance data arrived)
        # must update its role rather than keep an UNKNOWN classification.
        track.role = observation.role

        state.samples.append((observation.timestamp, observation.x_field_m, observation.y_field_m))
        cutoff = observation.timestamp - self.cfg.velocity_window_s
        while state.samples and state.samples[0][0] < cutoff:
            state.samples.popleft()

        vx, vy, valid = fit_velocity(
            list(state.samples),
            self.cfg.velocity_min_samples,
            self.cfg.velocity_min_span_s,
            self._max_speed(track.role),
        )
        track.vx, track.vy, track.velocity_valid = vx, vy, valid

    def _decayed_confidence(self, track: Track, now: float) -> float:
        _fresh_limit, expire_limit = self._timeouts(track.role)
        staleness = max(0.0, track.staleness(now))
        if expire_limit <= 0.0:
            return 0.0
        return max(0.0, track.confidence * (1.0 - staleness / expire_limit))

    def _expire(self, now: float) -> None:
        expired = [
            track_id
            for track_id, state in self._states.items()
            if self.freshness(state.track, now) is TrackFreshness.EXPIRED
        ]
        for track_id in expired:
            del self._states[track_id]

    def clear(self) -> None:
        """Drop every track.

        Called when the pose becomes invalid: field-space tracks recorded
        against a bad pose are worse than no tracks at all.
        """
        if self._states:
            logger.info("world model: cleared %d tracks", len(self._states))
        self._states.clear()


__all__ = ["WorldModel", "fit_velocity", "merge_observations"]
