"""Feeding two cameras to one accelerator without building a backlog.

THE PROBLEM
-----------
Two OV9782 cameras each produce frames on their own thread.  There is one
Hailo-8.  If inference is slower than capture -- and it will sometimes be,
under thermal throttling or a heavy frame -- the naive design queues
frames and the queue grows without bound.  Every frame in that queue is
older than the last, so the robot ends up steering from a picture of where
things were a second ago.  On a machine that is driving, that is worse
than not detecting at all.

THE POLICY
----------
Latest frame only, freshness-aware round robin:

    camera 0 thread ── latest frame ──┐
                                      ├── scheduler ──► Hailo-8
    camera 1 thread ── latest frame ──┘

* There is **no queue**.  Each camera holds exactly one frame -- its
  newest -- and the scheduler reads it at the moment it is ready to run
  inference.  If camera 0 produced frames 100, 101, 102 and 103 while the
  accelerator was busy, the scheduler takes 103 and counts three skipped.
  Skipping is the correct behavior, not a fault, but the count is
  published so the rate is visible.
* **Round robin** across cameras, with the starting position advancing
  every cycle, so neither camera can starve the other even if one is
  producing frames much faster.
* A frame already inferred on (same camera, same sequence number) is not
  inferred on again.
* A frame older than the freshness limit is refused before it reaches the
  accelerator -- a frozen camera must never feed YOLO.
* A RESULT that finished too late to be useful is dropped rather than
  injected into the field-space world model, because a stale object
  position is indistinguishable from a real one once it is a track.

Everything here is pure scheduling logic over the ``FrameSource`` and
``DetectorBackend`` interfaces, so it is fully unit tested with fakes and
needs neither cameras nor an accelerator.
"""

from __future__ import annotations

import logging
import time
from typing import Callable, Dict, List, Optional

from .camera import CameraManager
from .config import AppConfig
from .detector import DetectionError, DetectorBackend
from .logging_setup import RateLimiter
from .types import Frame, InferenceResult

logger = logging.getLogger(__name__)


class CameraInferenceStats:
    """Per-camera counters for the diagnostics published to NetworkTables."""

    __slots__ = (
        "camera_id",
        "inferences",
        "skipped_frames",
        "stale_frames",
        "stale_results",
        "failures",
        "last_sequence",
        "_times",
        "last_latency_ms",
    )

    def __init__(self, camera_id: str) -> None:
        self.camera_id = camera_id
        self.inferences = 0
        self.skipped_frames = 0
        self.stale_frames = 0
        self.stale_results = 0
        self.failures = 0
        self.last_sequence: Optional[int] = None
        self.last_latency_ms = 0.0
        self._times: List[float] = []

    def record_inference(self, now: float, latency_ms: float) -> None:
        self.inferences += 1
        self.last_latency_ms = latency_ms
        self._times.append(now)
        cutoff = now - 2.0
        while self._times and self._times[0] < cutoff:
            self._times.pop(0)

    def fps(self, now: float) -> float:
        """Inference frames per second for this camera, over a 2 s window."""
        cutoff = now - 2.0
        while self._times and self._times[0] < cutoff:
            self._times.pop(0)
        if len(self._times) < 2:
            return 0.0
        span = self._times[-1] - self._times[0]
        return (len(self._times) - 1) / span if span > 0 else 0.0


class InferenceScheduler:
    """Runs one inference cycle across the cameras, newest frames first."""

    def __init__(
        self,
        cameras: CameraManager,
        backend: DetectorBackend,
        cfg: AppConfig,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.cameras = cameras
        self.backend = backend
        self.cfg = cfg
        # ONE clock, read at every point that needs a time. Selecting frames
        # against an injected timestamp while measuring staleness against the
        # wall clock would make the two decisions disagree -- and would make
        # this class untestable, which is how such a bug would survive.
        self._clock = clock
        self.stats: Dict[str, CameraInferenceStats] = {
            name: CameraInferenceStats(name) for name in cameras.names
        }
        self._round_robin_start = 0
        self._rate_limiter = RateLimiter(cfg.logging.rate_limit_s)

    # -- diagnostics -------------------------------------------------------

    def stats_for(self, camera_id: str) -> CameraInferenceStats:
        stats = self.stats.get(camera_id)
        if stats is None:
            stats = CameraInferenceStats(camera_id)
            self.stats[camera_id] = stats
        return stats

    @property
    def total_skipped(self) -> int:
        return sum(s.skipped_frames for s in self.stats.values())

    @property
    def total_stale_frames(self) -> int:
        return sum(s.stale_frames for s in self.stats.values())

    @property
    def total_stale_results(self) -> int:
        return sum(s.stale_results for s in self.stats.values())

    # -- selection ---------------------------------------------------------

    def _ordered_camera_names(self) -> List[str]:
        """Round-robin ordering, advancing one position per cycle."""
        names = self.cameras.names
        if not names:
            return []
        offset = self._round_robin_start % len(names)
        self._round_robin_start = (self._round_robin_start + 1) % len(names)
        return names[offset:] + names[:offset]

    def select_frames(self, now: Optional[float] = None) -> List[Frame]:
        """Pick this cycle's frames: newest per camera, fresh, not yet seen.

        Returns at most one frame per camera and at most
        ``max_inferences_per_cycle`` frames overall, in round-robin order.
        """
        moment = self._clock() if now is None else now
        selected: List[Frame] = []
        limit = max(1, self.cfg.max_inferences_per_cycle)

        for camera_id in self._ordered_camera_names():
            if len(selected) >= limit:
                break
            source = next(
                (s for s in self.cameras.sources if s.name == camera_id), None
            )
            if source is None:
                continue

            latest = source.read_latest()
            if latest is None:
                continue

            stats = self.stats_for(camera_id)
            if latest.age(moment) > self.cfg.frame_freshness_s:
                # Frozen or disconnected camera: refuse it outright rather
                # than spending the accelerator on a picture of the past.
                stats.stale_frames += 1
                self._rate_limiter.log(
                    logger, logging.WARNING, f"stale_frame:{camera_id}",
                    "camera %s: frame %.0f ms old, refusing to infer on it",
                    camera_id, latest.age(moment) * 1000.0, now=moment,
                )
                continue

            if stats.last_sequence is not None:
                if latest.seq <= stats.last_sequence:
                    continue  # nothing new since the last cycle
                # Everything between the last processed frame and this one was
                # skipped in favor of the newest. That is the policy working.
                stats.skipped_frames += latest.seq - stats.last_sequence - 1
            selected.append(latest)

        return selected

    # -- execution ---------------------------------------------------------

    def run_cycle(self) -> List[InferenceResult]:
        """Select frames, run inference, and drop results that arrived too late.

        Returns only results that are still worth acting on.  Never raises:
        a backend failure is recorded and reported, and the rest of the
        cycle continues.  Time comes from ``self._clock`` throughout.
        """
        moment = self._clock()
        if not self.backend.healthy():
            return []

        results: List[InferenceResult] = []
        for frame in self.select_frames(moment):
            stats = self.stats_for(frame.camera_id)
            started = self._clock()
            try:
                detections = self.backend.predict(
                    frame, frame.camera_id, frame.timestamp
                )
                error = ""
            except DetectionError as exc:
                detections, error = [], str(exc)
            except Exception as exc:  # noqa: BLE001 - a backend must not kill the loop
                detections, error = [], f"unexpected backend error: {exc}"
            finished = self._clock()

            # The sequence is marked processed either way: retrying a frame
            # that just failed would spend the accelerator on the same
            # failure and stall the other camera.
            skipped = (
                0
                if stats.last_sequence is None
                else max(0, frame.seq - stats.last_sequence - 1)
            )
            stats.last_sequence = frame.seq

            result = InferenceResult(
                camera_id=frame.camera_id,
                frame_sequence=frame.seq,
                capture_timestamp=frame.timestamp,
                inference_start=started,
                inference_end=finished,
                detections=detections,
                skipped_frames=skipped,
                error=error,
            )

            if error:
                stats.failures += 1
                self._rate_limiter.log(
                    logger, logging.WARNING, f"infer:{frame.camera_id}",
                    "inference failed on %s: %s", frame.camera_id, error, now=moment,
                )
                continue

            stats.record_inference(finished, result.inference_latency_s * 1000.0)

            age = result.frame_age(self._clock())
            if age > self.cfg.max_detection_result_age_s:
                # Correct answer, wrong moment. Injecting it would put objects
                # into the world model where they USED to be.
                stats.stale_results += 1
                self._rate_limiter.log(
                    logger, logging.WARNING, f"stale_result:{frame.camera_id}",
                    "dropping %s result: %.0f ms end-to-end (limit %.0f ms)",
                    frame.camera_id, age * 1000.0,
                    self.cfg.max_detection_result_age_s * 1000.0, now=moment,
                )
                continue

            results.append(result)

        return results


__all__ = ["CameraInferenceStats", "InferenceScheduler"]
