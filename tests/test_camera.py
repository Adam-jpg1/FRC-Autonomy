"""Camera plumbing: freshness rejection, health reporting, manager behavior.

No hardware is touched. A fake frame source stands in for a real camera so
the rule that matters -- a stale frame is never handed to the pipeline --
can be tested deterministically.
"""

import pytest

from frc_perception.camera import CameraManager, FrameSource, _FpsCounter
from frc_perception.config import CameraConfig
from frc_perception.types import CameraHealth, Frame


class FakeSource(FrameSource):
    """A frame source whose frame age is set by the test."""

    def __init__(self, name: str, healthy: bool = True) -> None:
        self.name = name
        self.started = False
        self.stopped = False
        self._frame = None
        self._healthy = healthy

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def set_frame(self, timestamp: float, seq: int = 1) -> None:
        self._frame = Frame(self.name, seq, timestamp, image=object())

    def read_latest(self):
        return self._frame

    def health(self) -> CameraHealth:
        return CameraHealth(name=self.name, opened=True, healthy=self._healthy, fps=30.0)


class ExplodingSource(FakeSource):
    def start(self) -> None:
        raise RuntimeError("device on fire")

    def stop(self) -> None:
        raise RuntimeError("still on fire")


# ---------------------------------------------------------------------------
# Freshness
# ---------------------------------------------------------------------------


def test_a_fresh_frame_is_returned():
    source = FakeSource("cam0")
    source.set_frame(timestamp=10.0)
    assert source.read_fresh(max_age_s=0.25, now=10.1) is not None


def test_a_stale_frame_is_refused():
    """A frozen camera must produce NOTHING rather than an old image.

    Handing a stale frame to YOLO yields confident, current-looking,
    completely wrong detections -- the worst possible failure mode.
    """
    source = FakeSource("cam0")
    source.set_frame(timestamp=10.0)
    assert source.read_fresh(max_age_s=0.25, now=10.5) is None


def test_no_frame_yet_is_not_an_error():
    assert FakeSource("cam0").read_fresh(max_age_s=0.25, now=1.0) is None


def test_frame_age_is_computed_from_the_capture_timestamp():
    frame = Frame("cam0", 1, 100.0, image=None)
    assert frame.age(100.5) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


def test_manager_returns_only_the_fresh_cameras():
    fresh, stale = FakeSource("cam0"), FakeSource("cam1")
    fresh.set_frame(timestamp=10.0)
    stale.set_frame(timestamp=9.0)
    manager = CameraManager([fresh, stale], freshness_s=0.25)
    frames = manager.fresh_frames(now=10.1)
    assert set(frames) == {"cam0"}


def test_one_dead_camera_does_not_hide_the_other():
    """The system must keep running on one healthy camera."""
    good, dead = FakeSource("cam0", healthy=True), FakeSource("cam1", healthy=False)
    good.set_frame(timestamp=10.0)
    manager = CameraManager([good, dead], freshness_s=0.25)
    assert manager.any_healthy
    assert list(manager.fresh_frames(now=10.05)) == ["cam0"]


def test_manager_reports_no_healthy_cameras_when_all_are_down():
    manager = CameraManager([FakeSource("cam0", healthy=False)], freshness_s=0.25)
    assert not manager.any_healthy


def test_a_camera_that_fails_to_start_does_not_stop_the_others():
    good = FakeSource("cam0")
    manager = CameraManager([ExplodingSource("cam1"), good], freshness_s=0.25)
    manager.start()
    manager.stop()
    assert good.started and good.stopped


def test_manager_exposes_health_for_every_camera():
    manager = CameraManager([FakeSource("cam0"), FakeSource("cam1")], freshness_s=0.25)
    health = manager.health()
    assert set(health) == {"cam0", "cam1"}
    assert health["cam0"].fps == 30.0


# ---------------------------------------------------------------------------
# FPS counter
# ---------------------------------------------------------------------------


def test_fps_counter_measures_a_steady_rate():
    counter = _FpsCounter(window_s=2.0)
    for index in range(21):
        counter.tick(100.0 + index * 0.05)  # 20 Hz
    assert counter.fps(101.0) == pytest.approx(20.0, rel=0.05)


def test_fps_counter_needs_two_samples():
    counter = _FpsCounter()
    assert counter.fps(1.0) == 0.0
    counter.tick(1.0)
    assert counter.fps(1.0) == 0.0


def test_fps_counter_decays_when_frames_stop():
    counter = _FpsCounter(window_s=1.0)
    for index in range(10):
        counter.tick(100.0 + index * 0.1)
    assert counter.fps(101.0) == pytest.approx(10.0, rel=0.05), "half the window still holds ticks"
    assert counter.fps(102.0) == 0.0


# ---------------------------------------------------------------------------
# Configuration plumbing
# ---------------------------------------------------------------------------


def test_device_path_overrides_the_index_when_set():
    assert CameraConfig(name="c", device_index=3).source == 3
    assert CameraConfig(name="c", device_index=3, device_path="/dev/video9").source == "/dev/video9"
