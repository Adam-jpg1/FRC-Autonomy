"""Pose estimation: roboRIO odometry corrected by AprilTag fixes.

WHAT THIS IS
------------
The roboRIO's odometry is the continuous backbone: it is smooth, it never
drops out mid-motion, and it is available at a much higher rate than
vision.  AprilTags supply the absolute truth that odometry lacks, but only
intermittently and with an error that grows quickly with tag distance.

The two are combined by a **complementary correction**, not a Kalman
filter and not an average.  The estimator holds one SE(2) correction
transform ``C`` and publishes

    fused = C  o  odometry

Each accepted tag fix produces the correction that would make the fused
pose match the tag pose exactly; ``C`` is moved a configurable fraction of
the way there and the resulting pose step is clamped.  Between fixes, the
fused pose moves exactly as odometry moves, so the robot never stutters.

This is honestly weaker than a proper filter: it has no covariance, it
cannot weight a fix by its uncertainty except through a scalar quality
term, and it does not model odometry drift rate.  It is chosen because
its failure modes are obvious and a student can reason about them at 2 am
in a pit.  :class:`PoseEstimator` is small and its interface is narrow, so
replacing it with a real filter later is a contained change.

WHAT IT REFUSES TO DO
---------------------
* It never averages two poses that disagree wildly.  Beyond
  ``disagreement_reject_*`` the tag fix is not blended at all, and if the
  disagreement persists the pose is declared INVALID rather than snapped.
* It never treats a repeated NetworkTables read as evidence of freshness.
  Odometry is fresh only while the RIO's own timestamp keeps ADVANCING.
* It never reports ``valid`` merely because it has numbers.  See
  :meth:`PoseEstimator.update` for the full list of invalidating
  conditions.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional, Tuple

from .config import LocalizationConfig
from .field_map import FieldMap
from .geometry import wrap_angle
from .logging_setup import StateLogger
from .types import (
    LocalizationSource,
    OdometryReading,
    Pose2d,
    PoseEstimate,
    TagPoseEstimate,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SE(2) helpers (pure functions -- unit tested)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SE2:
    """A 2D rigid transform used as the odometry correction."""

    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0

    @staticmethod
    def identity() -> "SE2":
        return SE2()


def se2_apply(transform: SE2, pose: Pose2d) -> Pose2d:
    """Apply ``transform`` to ``pose`` (transform o pose)."""
    c, s = math.cos(transform.theta), math.sin(transform.theta)
    return Pose2d(
        x=transform.x + c * pose.x - s * pose.y,
        y=transform.y + s * pose.x + c * pose.y,
        theta=wrap_angle(transform.theta + pose.theta),
    )


def se2_correction_between(target: Pose2d, source: Pose2d) -> SE2:
    """The transform ``C`` with ``C o source == target``."""
    theta = wrap_angle(target.theta - source.theta)
    c, s = math.cos(theta), math.sin(theta)
    return SE2(
        x=target.x - (c * source.x - s * source.y),
        y=target.y - (s * source.x + c * source.y),
        theta=theta,
    )


def se2_blend(current: SE2, measured: SE2, alpha_translation: float, alpha_heading: float) -> SE2:
    """Move ``current`` a fraction of the way toward ``measured``."""
    return SE2(
        x=current.x + alpha_translation * (measured.x - current.x),
        y=current.y + alpha_translation * (measured.y - current.y),
        theta=wrap_angle(
            current.theta + alpha_heading * wrap_angle(measured.theta - current.theta)
        ),
    )


def pose_difference(a: Pose2d, b: Pose2d) -> Tuple[float, float]:
    """(translation distance, |heading difference|) between two poses."""
    return (math.hypot(a.x - b.x, a.y - b.y), abs(wrap_angle(a.theta - b.theta)))


# ---------------------------------------------------------------------------
# Odometry freshness
# ---------------------------------------------------------------------------


class OdometryMonitor:
    """Tracks whether the roboRIO's odometry stream is actually alive.

    Freshness is decided ONLY by the RIO-published timestamp advancing.
    Reading the same value ten thousand times does not make it fresh, and
    that is the exact failure this class exists to catch: when robot code
    crashes or the NT publisher stops, the last values sit in the table
    looking perfectly plausible forever.
    """

    def __init__(self, cfg: LocalizationConfig) -> None:
        self.cfg = cfg
        self._reading: Optional[OdometryReading] = None
        self._last_rio_timestamp: Optional[float] = None
        self._last_change_time: Optional[float] = None
        self._backward_jumps: int = 0

    @property
    def reading(self) -> Optional[OdometryReading]:
        return self._reading

    @property
    def backward_jumps(self) -> int:
        return self._backward_jumps

    def update(
        self, pose: Pose2d, rio_timestamp: float, now: float
    ) -> Optional[OdometryReading]:
        """Record one odometry sample.  Returns the stored reading.

        A timestamp that goes BACKWARD means the RIO restarted its clock.
        That counts as a change (the stream is alive) but is counted and
        reported, because the pose that comes with it is usually a reset
        pose and any accumulated correction is now meaningless.
        """
        if not pose.is_finite() or not math.isfinite(rio_timestamp):
            return self._reading

        changed = (
            self._last_rio_timestamp is None or rio_timestamp != self._last_rio_timestamp
        )
        if (
            self._last_rio_timestamp is not None
            and rio_timestamp < self._last_rio_timestamp
        ):
            self._backward_jumps += 1
            logger.warning(
                "odometry: RIO timestamp went backward (%.3f -> %.3f); "
                "robot code probably restarted",
                self._last_rio_timestamp, rio_timestamp,
            )
        if changed:
            self._last_rio_timestamp = rio_timestamp
            self._last_change_time = now

        self._reading = OdometryReading(
            pose=pose,
            rio_timestamp=rio_timestamp,
            local_time_of_change=self._last_change_time if self._last_change_time else now,
        )
        return self._reading

    def is_fresh(self, now: float) -> bool:
        if self._reading is None or self._last_change_time is None:
            return False
        return (now - self._last_change_time) <= self.cfg.odometry_stale_timeout_s

    def age_s(self, now: float) -> float:
        if self._last_change_time is None:
            return float("inf")
        return now - self._last_change_time


# ---------------------------------------------------------------------------
# The estimator
# ---------------------------------------------------------------------------


class PoseEstimator:
    """Odometry backbone + AprilTag correction + the pose-validity gate."""

    def __init__(self, cfg: LocalizationConfig, field_map: Optional[FieldMap] = None) -> None:
        self.cfg = cfg
        self.field_map = field_map
        self.odometry = OdometryMonitor(cfg)
        self._states = StateLogger(logger)

        self._correction = SE2.identity()
        self._initialized = False

        #: The most recent estimate OFFERED, accepted or not (diagnostics).
        self._last_tag_estimate: Optional[TagPoseEstimate] = None
        #: The most recent estimate ACCEPTED. This is the only one that may
        #: ever be published as a pose -- a rejected fix is rejected.
        self._last_accepted_tag_estimate: Optional[TagPoseEstimate] = None
        self._last_accepted_tag_time: Optional[float] = None
        self._disagreement_since: Optional[float] = None

        self.translation_disagreement_m: float = 0.0
        self.heading_disagreement_rad: float = 0.0
        self.accepted_fixes: int = 0
        self.rejected_fixes: int = 0
        self._last_estimate = PoseEstimate(
            pose=Pose2d(), valid=False, source=LocalizationSource.NONE, timestamp=0.0,
            invalid_reason="not initialized",
        )

    # -- introspection -----------------------------------------------------

    @property
    def correction(self) -> SE2:
        return self._correction

    @property
    def initialized(self) -> bool:
        return self._initialized

    @property
    def last_estimate(self) -> PoseEstimate:
        return self._last_estimate

    def fused_from(self, odom_pose: Pose2d) -> Pose2d:
        return se2_apply(self._correction, odom_pose)

    # -- inputs ------------------------------------------------------------

    def update_odometry(self, pose: Pose2d, rio_timestamp: float, now: float) -> None:
        self.odometry.update(pose, rio_timestamp, now)

    def add_tag_estimate(self, estimate: Optional[TagPoseEstimate], now: float) -> bool:
        """Offer an AprilTag pose to the estimator.

        Returns True if it was accepted into the correction.  Rejection is
        never silent: the reason is logged as a state transition and the
        counters are published over NetworkTables.
        """
        if estimate is None:
            return False
        self._last_tag_estimate = estimate

        if not estimate.pose.is_finite():
            self.rejected_fixes += 1
            self._states.log_value("tag_reject", "non_finite", "tag pose rejected", logging.WARNING)
            return False

        if self.field_map is not None and not self.field_map.contains_point(
            estimate.pose.x, estimate.pose.y, self.cfg.field_bounds_margin_m
        ):
            self.rejected_fixes += 1
            self._states.log_value("tag_reject", "off_field", "tag pose rejected", logging.WARNING)
            return False

        odom = self.odometry.reading
        if odom is None or not self.odometry.is_fresh(now):
            # No usable odometry: adopt the tag pose outright. There is
            # nothing to disagree with, and a tag fix is far better than
            # nothing. The correction is set so that if odometry comes
            # back, the fused pose is continuous from here.
            self._correction = se2_correction_between(
                estimate.pose, odom.pose if odom else Pose2d()
            )
            self._accept(estimate, now, "adopted (no fresh odometry)")
            return True

        if not self._initialized:
            # First fix. The RIO's odometry origin is arbitrary until vision
            # anchors it, so a large offset here is EXPECTED and must not be
            # treated as a disagreement -- and blending toward it from an
            # identity correction would leave the robot lost for seconds.
            self._correction = se2_correction_between(estimate.pose, odom.pose)
            self._accept(estimate, now, "initial fix adopted")
            return True

        current_fused = self.fused_from(odom.pose)
        translation_error, heading_error = pose_difference(estimate.pose, current_fused)
        self.translation_disagreement_m = translation_error
        self.heading_disagreement_rad = heading_error

        over_reject = (
            translation_error > self.cfg.disagreement_reject_m
            or heading_error > self.cfg.disagreement_reject_rad
        )
        over_warn = (
            translation_error > self.cfg.disagreement_warn_m
            or heading_error > self.cfg.disagreement_warn_rad
        )

        if over_reject:
            if self._disagreement_since is None:
                self._disagreement_since = now
            self.rejected_fixes += 1
            self._states.log(
                "disagreement_reject", False,
                "localization disagreement resolved",
                f"LARGE LOCALIZATION DISAGREEMENT: tag vs fused "
                f"{translation_error:.2f} m / {math.degrees(heading_error):.1f} deg "
                f"(tags={estimate.tag_ids}); tag fix NOT applied",
            )
            return False

        self._disagreement_since = None
        self._states.log(
            "disagreement_reject", True,
            "localization disagreement resolved", "large localization disagreement",
        )
        if over_warn:
            self._states.log(
                "disagreement_warn", False,
                "localization disagreement back within limits",
                f"localization disagreement {translation_error:.2f} m / "
                f"{math.degrees(heading_error):.1f} deg",
            )
        else:
            self._states.log(
                "disagreement_warn", True,
                "localization disagreement back within limits",
                "localization disagreement",
            )

        alpha_translation = self.cfg.correction_alpha_translation
        alpha_heading = self.cfg.correction_alpha_heading
        if self.cfg.scale_alpha_by_quality:
            quality = max(0.0, min(1.0, estimate.quality))
            alpha_translation *= quality
            alpha_heading *= quality

        measured = se2_correction_between(estimate.pose, odom.pose)
        candidate = se2_blend(self._correction, measured, alpha_translation, alpha_heading)
        candidate = self._clamp_step(candidate, odom.pose)
        self._correction = candidate
        self._accept(estimate, now, "")
        return True

    def _clamp_step(self, candidate: SE2, odom_pose: Pose2d) -> SE2:
        """Limit how far one correction may move the published pose."""
        before = self.fused_from(odom_pose)
        after = se2_apply(candidate, odom_pose)
        moved_m, moved_rad = pose_difference(after, before)

        scale = 1.0
        if moved_m > self.cfg.max_correction_step_m > 0:
            scale = min(scale, self.cfg.max_correction_step_m / moved_m)
        if moved_rad > self.cfg.max_correction_step_rad > 0:
            scale = min(scale, self.cfg.max_correction_step_rad / moved_rad)
        if scale >= 1.0:
            return candidate
        return se2_blend(self._correction, candidate, scale, scale)

    def _accept(self, estimate: TagPoseEstimate, now: float, note: str) -> None:
        self.accepted_fixes += 1
        self._last_accepted_tag_time = now
        self._last_accepted_tag_estimate = estimate
        if not self._initialized:
            self._initialized = True
            logger.info(
                "localization: INITIALIZED from tags %s at (%.2f, %.2f, %.1f deg) %s",
                estimate.tag_ids, estimate.pose.x, estimate.pose.y,
                math.degrees(estimate.pose.theta), note,
            )
        elif note:
            logger.info("localization: %s (tags=%s)", note, estimate.tag_ids)
        self._states.prime("tag_reject", "")

    # -- output ------------------------------------------------------------

    def update(self, now: float) -> PoseEstimate:
        """Produce the pose the rest of the robot may act on.

        The pose is INVALID when any of these hold:

        * no tag fix has ever been accepted and
          ``require_tag_initialization`` is set (the field origin would
          otherwise be whatever the RIO happened to assume);
        * odometry is stale AND no tag pose is recent enough;
        * the resulting numbers are not finite;
        * the pose lies outside the field by more than the margin;
        * tag and odometry have disagreed beyond the reject threshold for
          longer than ``disagreement_invalidate_after_s``.
        """
        odom = self.odometry.reading
        odometry_fresh = self.odometry.is_fresh(now)
        tag_recent = (
            self._last_accepted_tag_time is not None
            and (now - self._last_accepted_tag_time) <= self.cfg.tag_stale_timeout_s
        )
        tag_usable_alone = (
            self._last_accepted_tag_time is not None
            and (now - self._last_accepted_tag_time) <= self.cfg.tag_only_valid_s
        )

        pose = Pose2d()
        source = LocalizationSource.NONE
        reason = ""

        if odom is not None and odometry_fresh:
            pose = self.fused_from(odom.pose)
            source = (
                LocalizationSource.ODOMETRY_TAG_CORRECTED
                if self._initialized
                else LocalizationSource.ODOMETRY_ONLY
            )
        elif tag_usable_alone and self._last_accepted_tag_estimate is not None:
            pose = self._last_accepted_tag_estimate.pose
            source = LocalizationSource.TAG_ONLY
        else:
            reason = (
                "odometry stale and no recent AprilTag pose"
                if odom is not None
                else "no odometry received"
            )

        valid = source is not LocalizationSource.NONE
        if valid and self.cfg.require_tag_initialization and not self._initialized:
            valid, reason = False, "awaiting first AprilTag fix (estimator not initialized)"
        if valid and not pose.is_finite():
            valid, reason = False, "pose is not finite"
        if (
            valid
            and self.field_map is not None
            and not self.field_map.contains_point(pose.x, pose.y, self.cfg.field_bounds_margin_m)
        ):
            valid, reason = False, (
                f"pose ({pose.x:.2f}, {pose.y:.2f}) is outside the field"
            )
        if (
            valid
            and self._disagreement_since is not None
            and (now - self._disagreement_since) > self.cfg.disagreement_invalidate_after_s
        ):
            valid, reason = False, (
                "unresolved AprilTag/odometry disagreement for "
                f"{now - self._disagreement_since:.1f}s"
            )

        quality = 0.0
        if valid:
            quality = 0.5 if source is LocalizationSource.ODOMETRY_ONLY else 1.0
            if self._last_accepted_tag_estimate is not None and tag_recent:
                quality = min(1.0, 0.4 + 0.6 * self._last_accepted_tag_estimate.quality)
            elif source is LocalizationSource.ODOMETRY_TAG_CORRECTED:
                quality = 0.6

        self._states.log(
            "odometry_fresh", odometry_fresh,
            "odometry recovered", "ODOMETRY STALE (RIO timestamp stopped advancing)",
        )
        self._states.log(
            "pose_valid", valid,
            "pose valid: field-relative navigation enabled",
            f"POSE INVALID: {reason} -- field-relative navigation disabled",
        )
        self._states.log_value("localization_source", source.value, "localization source")

        estimate = PoseEstimate(
            pose=pose,
            valid=valid,
            source=source,
            timestamp=now,
            odometry_fresh=odometry_fresh,
            tag_pose_available=tag_recent,
            translation_disagreement_m=self.translation_disagreement_m,
            heading_disagreement_rad=self.heading_disagreement_rad,
            tag_count=(
                self._last_accepted_tag_estimate.tag_count
                if self._last_accepted_tag_estimate is not None and tag_recent
                else 0
            ),
            quality=quality,
            invalid_reason="" if valid else reason,
        )
        self._last_estimate = estimate
        return estimate


__all__ = [
    "OdometryMonitor",
    "PoseEstimator",
    "SE2",
    "pose_difference",
    "se2_apply",
    "se2_blend",
    "se2_correction_between",
]
