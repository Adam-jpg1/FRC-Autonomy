"""Scheduling two cameras onto one accelerator.

The rules under test, in order of how much damage breaking them does:

1. No unbounded queue. Each camera holds exactly one frame -- its newest.
2. A newer frame replaces an older waiting one; the skip is counted.
3. A stale frame never reaches the accelerator.
4. A result that finished too late never reaches the world model.
5. Round robin: neither camera can starve the other.
6. One camera failing does not block the other.

All of it runs against fakes: no cameras, no Hailo, no Ultralytics.
"""

import pytest

from frc_perception.camera import CameraManager, FrameSource
from frc_perception.config import AppConfig
from frc_perception.detector import DetectionError, DetectorBackend
from frc_perception.inference_scheduler import InferenceScheduler
from frc_perception.types import CameraHealth, Detection, Frame


class FakeSource(FrameSource):
    """A camera whose latest frame the test sets directly."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._frame = None
        self.reads = 0

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def produce(self, seq: int, timestamp: float) -> Frame:
        self._frame = Frame(camera_id=self.name, seq=seq, timestamp=timestamp, image=object())
        return self._frame

    def read_latest(self):
        self.reads += 1
        return self._frame

    def health(self) -> CameraHealth:
        return CameraHealth(name=self.name, opened=True, healthy=self._frame is not None)


class FakeBackend(DetectorBackend):
    """Records every frame it was asked to infer on."""

    name = "fake"

    def __init__(self, cfg, latency_s: float = 0.0, fail_for=()) -> None:
        super().__init__(cfg)
        self.seen = []
        self.latency_s = latency_s
        self.fail_for = set(fail_for)
        self._health.healthy = True
        self._health.loaded = True

    def load(self) -> bool:
        return True

    def healthy(self) -> bool:
        return self._health.healthy

    def predict(self, frame, camera_id="", capture_timestamp=0.0):
        self.seen.append((frame.camera_id, frame.seq))
        if frame.camera_id in self.fail_for:
            raise DetectionError(f"{frame.camera_id} exploded")
        if self.latency_s:
            # Simulated cost, without actually sleeping the test suite.
            frame.timestamp -= self.latency_s
        return [
            Detection(
                class_name="fuel", confidence=0.9,
                x1=0.0, y1=0.0, x2=10.0, y2=10.0,
                camera_id=frame.camera_id,
                frame_sequence=frame.seq,
                capture_timestamp=frame.timestamp,
            )
        ]


class FakeClock:
    """A controllable monotonic clock.

    The scheduler reads one clock for frame selection, inference timing and
    staleness, so driving that clock is enough to reproduce any timing the
    real robot could see -- including inference that takes longer than the
    result is allowed to be old.
    """

    def __init__(self, start: float = 100.0, step: float = 0.0) -> None:
        self.now = start
        self.step = step

    def __call__(self) -> float:
        value = self.now
        self.now += self.step
        return value

    def set(self, value: float) -> None:
        self.now = value

    def advance(self, seconds: float) -> None:
        self.now += seconds


def build(cameras=("cam0", "cam1"), clock=None, **cfg_overrides):
    cfg = AppConfig(**cfg_overrides)
    sources = [FakeSource(name) for name in cameras]
    manager = CameraManager(sources, cfg.frame_freshness_s)
    backend = FakeBackend(cfg.detector)
    clock = clock or FakeClock()
    scheduler = InferenceScheduler(manager, backend, cfg, clock=clock)
    scheduler.clock = clock
    return scheduler, sources, backend, cfg


def _run_at(scheduler, moment: float):
    """Run one cycle with the clock parked at ``moment``."""
    scheduler.clock.set(moment)
    return scheduler.run_cycle()


# ---------------------------------------------------------------------------
# Latest-frame policy
# ---------------------------------------------------------------------------


def test_a_single_fresh_frame_is_processed():
    scheduler, sources, backend, _cfg = build()
    sources[0].produce(seq=1, timestamp=100.0)
    results = _run_at(scheduler, 100.01)
    assert len(results) == 1
    assert results[0].camera_id == "cam0"
    assert results[0].frame_sequence == 1
    assert backend.seen == [("cam0", 1)]


def test_a_newer_frame_replaces_an_older_waiting_one():
    """Frames 100..103 arrive while the accelerator is busy: process 103."""
    scheduler, sources, backend, _cfg = build(cameras=("cam0",))
    sources[0].produce(seq=100, timestamp=100.0)
    _run_at(scheduler, 100.01)
    for seq in (101, 102, 103):
        sources[0].produce(seq=seq, timestamp=100.0 + seq * 0.001)
    results = _run_at(scheduler, 100.2)
    assert [r.frame_sequence for r in results] == [103]
    assert backend.seen == [("cam0", 100), ("cam0", 103)]


def test_skipped_frames_are_counted_not_hidden():
    scheduler, sources, _backend, _cfg = build(cameras=("cam0",))
    sources[0].produce(seq=10, timestamp=100.0)
    _run_at(scheduler, 100.01)
    sources[0].produce(seq=14, timestamp=100.05)
    results = _run_at(scheduler, 100.06)
    assert results[0].skipped_frames == 3
    assert scheduler.stats_for("cam0").skipped_frames == 3
    assert scheduler.total_skipped == 3


def test_the_same_frame_is_never_inferred_on_twice():
    scheduler, sources, backend, _cfg = build(cameras=("cam0",))
    sources[0].produce(seq=5, timestamp=100.0)
    assert len(_run_at(scheduler, 100.01)) == 1
    assert _run_at(scheduler, 100.02) == []
    assert backend.seen == [("cam0", 5)]


def test_the_scheduler_holds_no_queue_at_all():
    """There is nowhere for a backlog to accumulate.

    A hundred frames produced between cycles yield exactly one inference,
    and the scheduler stores no per-frame state beyond the last sequence.
    """
    scheduler, sources, backend, _cfg = build(cameras=("cam0",))
    for seq in range(1, 101):
        sources[0].produce(seq=seq, timestamp=100.0 + seq * 0.0001)
    results = _run_at(scheduler, 100.02)
    assert len(results) == 1
    assert results[0].frame_sequence == 100
    assert len(backend.seen) == 1


def test_no_frame_yet_is_not_an_error():
    scheduler, _sources, _backend, _cfg = build()
    assert _run_at(scheduler, 1.0) == []


# ---------------------------------------------------------------------------
# Freshness
# ---------------------------------------------------------------------------


def test_a_stale_frame_never_reaches_the_accelerator():
    scheduler, sources, backend, cfg = build(cameras=("cam0",))
    sources[0].produce(seq=1, timestamp=100.0)
    results = _run_at(scheduler, 100.0 + cfg.frame_freshness_s + 0.01)
    assert results == []
    assert backend.seen == []
    assert scheduler.stats_for("cam0").stale_frames == 1
    assert scheduler.total_stale_frames == 1


def test_a_frozen_camera_does_not_consume_the_accelerator_forever():
    """A camera stuck on one frame must not keep spending inference."""
    scheduler, sources, backend, _cfg = build(cameras=("cam0", "cam1"))
    sources[0].produce(seq=1, timestamp=100.0)   # cam0 freezes here
    sources[1].produce(seq=1, timestamp=100.0)
    _run_at(scheduler, 100.01)
    for cycle in range(2, 12):
        sources[1].produce(seq=cycle, timestamp=100.0 + cycle * 0.03)
        _run_at(scheduler, 100.0 + cycle * 0.03)
    cam0_inferences = [c for c, _s in backend.seen if c == "cam0"]
    assert len(cam0_inferences) == 1, "the frozen frame was inferred on once, then refused"
    assert scheduler.stats_for("cam1").inferences >= 5


def test_a_stale_result_is_dropped_rather_than_entering_the_world_model():
    """Correct answer, wrong moment.

    The frame was fresh when selected but by the time inference finished
    it described the past. Injecting it would place objects where they
    used to be.
    """
    clock = FakeClock(start=100.0, step=0.05)  # every clock read advances 50 ms
    scheduler, sources, _backend, _cfg = build(
        cameras=("cam0",), clock=clock, max_detection_result_age_s=0.10
    )
    sources[0].produce(seq=1, timestamp=100.0)
    # The frame is fresh when selected, but inference takes long enough that
    # the finished result describes the past.
    results = scheduler.run_cycle()
    assert results == []
    assert scheduler.stats_for("cam0").stale_results == 1
    assert scheduler.total_stale_results == 1


def test_a_fresh_result_is_accepted():
    scheduler, sources, _backend, _cfg = build(cameras=("cam0",))
    sources[0].produce(seq=1, timestamp=100.0)
    results = _run_at(scheduler, 100.0)
    assert len(results) == 1
    assert results[0].ok
    assert results[0].detections


# ---------------------------------------------------------------------------
# Fairness
# ---------------------------------------------------------------------------


def test_both_cameras_are_served_in_one_cycle():
    scheduler, sources, backend, _cfg = build()
    sources[0].produce(seq=1, timestamp=100.0)
    sources[1].produce(seq=1, timestamp=100.0)
    results = _run_at(scheduler, 100.01)
    assert {r.camera_id for r in results} == {"cam0", "cam1"}
    assert len(backend.seen) == 2


def test_round_robin_alternates_who_goes_first():
    scheduler, sources, backend, _cfg = build(max_inferences_per_cycle=1)
    order = []
    for cycle in range(6):
        for index, source in enumerate(sources):
            source.produce(seq=cycle + 1, timestamp=100.0 + cycle * 0.03)
        results = _run_at(scheduler, 100.0 + cycle * 0.03)
        order.extend(r.camera_id for r in results)
    assert order.count("cam0") == 3
    assert order.count("cam1") == 3
    assert order[0] != order[1], "the starting camera must advance each cycle"


def test_a_fast_camera_cannot_starve_a_slow_one():
    """cam0 produces on every cycle, cam1 on every third. cam1 still runs."""
    scheduler, sources, _backend, _cfg = build(max_inferences_per_cycle=1)
    served = []
    for cycle in range(12):
        sources[0].produce(seq=cycle + 1, timestamp=100.0 + cycle * 0.03)
        if cycle % 3 == 0:
            sources[1].produce(seq=cycle + 1, timestamp=100.0 + cycle * 0.03)
        served.extend(r.camera_id for r in _run_at(scheduler, 100.0 + cycle * 0.03))
    assert served.count("cam1") >= 3
    assert served.count("cam0") >= 3


def test_the_per_cycle_inference_budget_is_respected():
    scheduler, sources, backend, _cfg = build(max_inferences_per_cycle=1)
    sources[0].produce(seq=1, timestamp=100.0)
    sources[1].produce(seq=1, timestamp=100.0)
    assert len(_run_at(scheduler, 100.01)) == 1
    assert len(backend.seen) == 1


# ---------------------------------------------------------------------------
# Failure containment
# ---------------------------------------------------------------------------


def test_one_camera_failing_inference_does_not_block_the_other():
    scheduler, sources, backend, _cfg = build()
    backend.fail_for = {"cam0"}
    sources[0].produce(seq=1, timestamp=100.0)
    sources[1].produce(seq=1, timestamp=100.0)
    results = _run_at(scheduler, 100.01)
    assert [r.camera_id for r in results] == ["cam1"]
    assert scheduler.stats_for("cam0").failures == 1
    assert scheduler.stats_for("cam1").inferences == 1


def test_a_failing_frame_is_not_retried_forever():
    """Retrying the same failing frame would stall the other camera."""
    scheduler, sources, backend, _cfg = build(cameras=("cam0",))
    backend.fail_for = {"cam0"}
    sources[0].produce(seq=1, timestamp=100.0)
    _run_at(scheduler, 100.01)
    _run_at(scheduler, 100.02)
    assert backend.seen == [("cam0", 1)]


def test_an_unhealthy_detector_produces_nothing_at_all():
    """No detections, no exception, no fabricated boxes."""
    scheduler, sources, backend, _cfg = build()
    backend._health.healthy = False
    sources[0].produce(seq=1, timestamp=100.0)
    assert _run_at(scheduler, 100.01) == []
    assert backend.seen == []


def test_an_unexpected_backend_exception_is_contained():
    class RudeBackend(FakeBackend):
        def predict(self, frame, camera_id="", capture_timestamp=0.0):
            raise MemoryError("not even a DetectionError")

    scheduler, sources, _backend, cfg = build(cameras=("cam0",))
    scheduler.backend = RudeBackend(cfg.detector)
    sources[0].produce(seq=1, timestamp=100.0)
    assert _run_at(scheduler, 100.01) == []
    assert scheduler.stats_for("cam0").failures == 1


# ---------------------------------------------------------------------------
# Timing chain
# ---------------------------------------------------------------------------


def test_the_result_carries_the_whole_timing_chain():
    scheduler, sources, _backend, _cfg = build(cameras=("cam0",))
    sources[0].produce(seq=7, timestamp=100.0)
    result = _run_at(scheduler, 100.0)[0]
    assert result.camera_id == "cam0"
    assert result.frame_sequence == 7
    assert result.capture_timestamp == pytest.approx(100.0)
    assert result.inference_end >= result.inference_start
    assert result.inference_latency_s >= 0.0
    assert result.frame_age(result.inference_end) >= 0.0
    assert result.result_age(result.inference_end + 0.5) == pytest.approx(0.5)


def test_detections_inherit_the_capture_timestamp_not_the_inference_time():
    """Downstream ages objects from when the picture was taken."""
    scheduler, sources, _backend, _cfg = build(cameras=("cam0",))
    sources[0].produce(seq=1, timestamp=100.0)
    result = _run_at(scheduler, 100.0)[0]
    assert result.detections[0].capture_timestamp == pytest.approx(100.0)
    assert result.detections[0].frame_sequence == 1


def test_inference_fps_is_measured_per_camera():
    scheduler, sources, _backend, _cfg = build(cameras=("cam0",))
    stats = scheduler.stats_for("cam0")
    for index in range(11):
        stats.record_inference(100.0 + index * 0.05, latency_ms=8.0)
    assert stats.fps(100.5) == pytest.approx(20.0, rel=0.05)
    assert stats.last_latency_ms == pytest.approx(8.0)


def test_inference_fps_decays_when_inference_stops():
    scheduler, _sources, _backend, _cfg = build(cameras=("cam0",))
    stats = scheduler.stats_for("cam0")
    for index in range(5):
        stats.record_inference(100.0 + index * 0.05, latency_ms=8.0)
    assert stats.fps(105.0) == 0.0
