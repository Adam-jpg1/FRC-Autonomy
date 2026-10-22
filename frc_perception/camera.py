"""Threaded camera capture with health tracking, reconnect, and freshness.

Each camera owns one thread that does nothing but pull frames as fast as
the device delivers them and publish the LATEST one.  Consumers never see
a queue: on a Jetson running a 30 Hz perception loop, an old frame is
worthless, and a growing queue converts a transient hiccup into permanent
latency.

Two failure modes are handled explicitly because both are silent:

* **Disconnect** -- ``read()`` starts returning False.  The thread closes
  the device, backs off, and retries.  ``healthy`` goes False immediately.
* **Freeze** -- the device keeps returning the SAME frame, or stops
  producing new ones while still reporting success.  The sequence number
  and timestamp stop advancing, and :meth:`FrameSource.read_fresh`
  refuses to hand out a frame older than the freshness threshold.  This is
  the case that must never reach YOLO: inferring on a frozen image
  produces confident, current-looking, completely wrong detections.

Nothing here imports OpenCV at module import time, so the rest of the
package (and the unit tests) work on a machine without ``cv2``.

PLATFORM NOTE
-------------
The cameras are USB/UVC devices (Arducam OV9782) captured through OpenCV's
V4L2 backend on the Jetson Orin Nano Super.  Prefer a persistent
``/dev/v4l/by-id/...`` path over a ``/dev/videoN`` index: the numbers are
assigned in enumeration order, and two identical OV9782 modules will
eventually swap between boots.  :func:`list_video_devices` prints what is
available on this machine.
"""

from __future__ import annotations

import glob
import logging
import os
import threading
import time
from typing import Dict, List, Optional

from .config import CameraConfig
from .types import CameraHealth, Frame

logger = logging.getLogger(__name__)


class FrameSource:
    """Interface every frame producer implements.

    Implemented by :class:`ThreadedCamera` (real USB camera) and
    :class:`ReplaySource` (hardware-free development).
    """

    name: str

    def start(self) -> None:  # pragma: no cover - trivial
        raise NotImplementedError

    def stop(self) -> None:  # pragma: no cover - trivial
        raise NotImplementedError

    def read_latest(self) -> Optional[Frame]:
        """Most recent frame, regardless of age.  ``None`` if none yet."""
        raise NotImplementedError

    def health(self) -> CameraHealth:
        raise NotImplementedError

    def read_fresh(self, max_age_s: float, now: Optional[float] = None) -> Optional[Frame]:
        """Latest frame, but only if it is younger than ``max_age_s``.

        This is the method the perception loop must use.  Returning
        ``None`` for a stale frame is the whole point: a frozen camera
        produces no detections rather than wrong ones.
        """
        frame = self.read_latest()
        if frame is None:
            return None
        moment = time.monotonic() if now is None else now
        if frame.age(moment) > max_age_s:
            return None
        return frame


class _FpsCounter:
    """Rolling frames-per-second over a fixed time window."""

    def __init__(self, window_s: float = 2.0) -> None:
        self._window_s = window_s
        self._times: List[float] = []

    def tick(self, now: float) -> None:
        self._times.append(now)
        cutoff = now - self._window_s
        while self._times and self._times[0] < cutoff:
            self._times.pop(0)

    def fps(self, now: float) -> float:
        cutoff = now - self._window_s
        while self._times and self._times[0] < cutoff:
            self._times.pop(0)
        if len(self._times) < 2:
            return 0.0
        span = self._times[-1] - self._times[0]
        if span <= 0.0:
            return 0.0
        return (len(self._times) - 1) / span


class ThreadedCamera(FrameSource):
    """One USB camera, captured on its own thread.

    The thread is a daemon so a wedged V4L2 call can never prevent process
    exit; :meth:`stop` still joins with a timeout for a clean shutdown in
    the normal case.
    """

    #: Reconnect backoff schedule, seconds. The last value repeats.
    RECONNECT_BACKOFF_S = (0.5, 1.0, 2.0, 4.0)
    #: Consecutive read failures before the device is closed and reopened.
    FAILURES_BEFORE_REOPEN = 5

    def __init__(self, cfg: CameraConfig, freshness_s: float) -> None:
        self.cfg = cfg
        self.name = cfg.name
        self._freshness_s = freshness_s

        self._lock = threading.Lock()
        self._frame: Optional[Frame] = None
        self._seq = 0

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._capture = None  # cv2.VideoCapture, created on the thread

        self._fps = _FpsCounter()
        self._health = CameraHealth(name=cfg.name)

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name=f"camera-{self.name}", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=2.0)
            if thread.is_alive():
                logger.warning("camera %s: capture thread did not exit cleanly", self.name)
        self._release()

    # -- consumer side -----------------------------------------------------

    def read_latest(self) -> Optional[Frame]:
        with self._lock:
            return self._frame

    def health(self) -> CameraHealth:
        now = time.monotonic()
        with self._lock:
            frame = self._frame
            health = self._health
            health.fps = self._fps.fps(now)
            health.last_frame_age_s = float("inf") if frame is None else frame.age(now)
            health.healthy = bool(
                health.opened
                and frame is not None
                and health.last_frame_age_s <= self._freshness_s
            )
            # Copy so callers cannot mutate our state.
            return CameraHealth(**vars(health))

    # -- capture thread ----------------------------------------------------

    def _open(self) -> bool:
        """Open the device and apply capture settings.  Never raises."""
        try:
            import cv2  # imported here so the package works without OpenCV
        except ImportError as exc:  # pragma: no cover - environment dependent
            self._note_error(f"OpenCV unavailable: {exc}")
            return False

        try:
            source = self.cfg.source
            # CAP_V4L2 is the right backend on the Jetson for USB UVC cameras;
            # the default backend sometimes ignores FOURCC/FPS requests.
            capture = cv2.VideoCapture(source, cv2.CAP_V4L2)
            if not capture.isOpened():
                capture.release()
                self._note_error(f"cannot open device {source!r}")
                return False

            capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.cfg.fourcc))
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.height)
            capture.set(cv2.CAP_PROP_FPS, self.cfg.fps)
            # A driver-side buffer defeats the point of latest-frame capture.
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if self.cfg.manual_exposure is not None:
                # 1 == manual on V4L2 UVC; 3 == aperture priority/auto.
                capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
                capture.set(cv2.CAP_PROP_EXPOSURE, self.cfg.manual_exposure)
            if self.cfg.manual_gain is not None:
                capture.set(cv2.CAP_PROP_GAIN, self.cfg.manual_gain)

            actual_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if (actual_w, actual_h) != (self.cfg.width, self.cfg.height):
                logger.warning(
                    "camera %s: requested %dx%d but driver gave %dx%d; intrinsics "
                    "are only valid at the calibrated resolution",
                    self.name, self.cfg.width, self.cfg.height, actual_w, actual_h,
                )

            self._capture = capture
            with self._lock:
                self._health.opened = True
                self._health.consecutive_failures = 0
                self._health.last_error = ""
            logger.info("camera %s: opened %r at %dx%d", self.name, source, actual_w, actual_h)
            return True
        except Exception as exc:  # noqa: BLE001 - a capture backend can raise anything
            self._note_error(f"open failed: {exc}")
            return False

    def _release(self) -> None:
        capture, self._capture = self._capture, None
        if capture is not None:
            try:
                capture.release()
            except Exception:  # noqa: BLE001 - release must never propagate
                pass
        with self._lock:
            self._health.opened = False

    def _note_error(self, message: str) -> None:
        with self._lock:
            self._health.last_error = message
            self._health.opened = False

    def _run(self) -> None:
        attempt = 0
        was_healthy = False
        while not self._stop_event.is_set():
            if self._capture is None:
                if not self._open():
                    delay = self.RECONNECT_BACKOFF_S[
                        min(attempt, len(self.RECONNECT_BACKOFF_S) - 1)
                    ]
                    attempt += 1
                    if attempt == 1 or attempt % 10 == 0:
                        logger.error(
                            "camera %s: not available (%s); retrying every %.1fs",
                            self.name, self._health.last_error, delay,
                        )
                    self._stop_event.wait(delay)
                    continue
                if attempt > 0:
                    with self._lock:
                        self._health.reconnect_count += 1
                    logger.info("camera %s: RECOVERED after %d attempts", self.name, attempt)
                attempt = 0

            try:
                ok, image = self._capture.read()
            except Exception as exc:  # noqa: BLE001 - backend can raise on unplug
                ok, image = False, None
                self._note_error(f"read raised: {exc}")

            now = time.monotonic()
            if not ok or image is None:
                with self._lock:
                    self._health.consecutive_failures += 1
                    self._health.frames_dropped += 1
                    failures = self._health.consecutive_failures
                if failures == 1 and was_healthy:
                    logger.warning("camera %s: DISCONNECTED or read failure", self.name)
                    was_healthy = False
                if failures >= self.FAILURES_BEFORE_REOPEN:
                    self._release()
                    self._stop_event.wait(0.1)
                continue

            self._seq += 1
            frame = Frame(
                camera_id=self.name, seq=self._seq, timestamp=now, image=image
            )
            with self._lock:
                self._frame = frame
                self._health.frames_captured += 1
                self._health.consecutive_failures = 0
                self._fps.tick(now)
            if not was_healthy:
                logger.info("camera %s: streaming", self.name)
                was_healthy = True
        self._release()


class ReplaySource(FrameSource):
    """Hardware-free frame source: a video file or a directory of images.

    Frames are paced at ``cfg.fps`` so the rest of the pipeline sees
    realistic timing and its own freshness logic actually gets exercised.
    """

    IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")

    def __init__(
        self,
        cfg: CameraConfig,
        path: str,
        freshness_s: float,
        loop: bool = True,
    ) -> None:
        self.cfg = cfg
        self.name = cfg.name
        self.path = path
        self.loop = loop
        self._freshness_s = freshness_s

        self._lock = threading.Lock()
        self._frame: Optional[Frame] = None
        self._seq = 0
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._fps = _FpsCounter()
        self._health = CameraHealth(name=cfg.name)
        self._exhausted = False

    @property
    def exhausted(self) -> bool:
        """True once a non-looping source has played out."""
        return self._exhausted

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name=f"replay-{self.name}", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=2.0)

    def read_latest(self) -> Optional[Frame]:
        with self._lock:
            return self._frame

    def health(self) -> CameraHealth:
        now = time.monotonic()
        with self._lock:
            frame = self._frame
            self._health.fps = self._fps.fps(now)
            self._health.last_frame_age_s = (
                float("inf") if frame is None else frame.age(now)
            )
            self._health.healthy = bool(
                self._health.opened
                and frame is not None
                and self._health.last_frame_age_s <= self._freshness_s
            )
            return CameraHealth(**vars(self._health))

    def _list_images(self) -> List[str]:
        if os.path.isdir(self.path):
            found: List[str] = []
            for extension in self.IMAGE_EXTENSIONS:
                found.extend(glob.glob(os.path.join(self.path, f"*{extension}")))
                found.extend(glob.glob(os.path.join(self.path, f"*{extension.upper()}")))
            return sorted(set(found))
        if os.path.isfile(self.path) and self.path.lower().endswith(self.IMAGE_EXTENSIONS):
            return [self.path]
        return []

    def _run(self) -> None:
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - environment dependent
            with self._lock:
                self._health.last_error = f"OpenCV unavailable: {exc}"
            logger.error("replay %s: %s", self.name, exc)
            return

        images = self._list_images()
        video = None
        if not images:
            if not os.path.isfile(self.path):
                with self._lock:
                    self._health.last_error = f"no such replay path: {self.path}"
                logger.error("replay %s: no such path %s", self.name, self.path)
                return
            video = cv2.VideoCapture(self.path)
            if not video.isOpened():
                with self._lock:
                    self._health.last_error = f"cannot open video {self.path}"
                logger.error("replay %s: cannot open video %s", self.name, self.path)
                return

        with self._lock:
            self._health.opened = True
        period = 1.0 / max(self.cfg.fps, 1)
        index = 0
        logger.info(
            "replay %s: %s (%s)",
            self.name, self.path, f"{len(images)} images" if images else "video",
        )

        while not self._stop_event.is_set():
            started = time.monotonic()
            if images:
                image = cv2.imread(images[index % len(images)])
                index += 1
                if not self.loop and index >= len(images):
                    self._exhausted = True
            else:
                ok, image = video.read()
                if not ok:
                    if self.loop:
                        video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        self._stop_event.wait(period)
                        continue
                    self._exhausted = True
                    break

            if image is None:
                self._stop_event.wait(period)
                continue

            now = time.monotonic()
            self._seq += 1
            with self._lock:
                self._frame = Frame(self.name, self._seq, now, image)
                self._health.frames_captured += 1
                self._fps.tick(now)
            if self._exhausted:
                break
            self._stop_event.wait(max(0.0, period - (time.monotonic() - started)))

        if video is not None:
            video.release()
        with self._lock:
            self._health.opened = False


class CameraManager:
    """Owns every frame source and answers "which cameras still work?".

    One dead camera must never take the system down, so start/stop are
    best-effort per camera and the manager keeps serving the survivors.
    """

    def __init__(self, sources: List[FrameSource], freshness_s: float) -> None:
        self._sources = list(sources)
        self._freshness_s = freshness_s

    @property
    def sources(self) -> List[FrameSource]:
        return list(self._sources)

    @property
    def names(self) -> List[str]:
        return [source.name for source in self._sources]

    def start(self) -> None:
        for source in self._sources:
            try:
                source.start()
            except Exception as exc:  # noqa: BLE001 - one camera must not kill startup
                logger.error("camera %s: start failed: %s", source.name, exc)

    def stop(self) -> None:
        for source in self._sources:
            try:
                source.stop()
            except Exception as exc:  # noqa: BLE001 - shutdown is best effort
                logger.error("camera %s: stop failed: %s", source.name, exc)

    def fresh_frames(self, now: Optional[float] = None) -> Dict[str, Frame]:
        """Every camera's latest frame that is young enough to use."""
        moment = time.monotonic() if now is None else now
        frames: Dict[str, Frame] = {}
        for source in self._sources:
            frame = source.read_fresh(self._freshness_s, moment)
            if frame is not None:
                frames[source.name] = frame
        return frames

    def health(self) -> Dict[str, CameraHealth]:
        return {source.name: source.health() for source in self._sources}

    @property
    def any_healthy(self) -> bool:
        return any(source.health().healthy for source in self._sources)


def build_camera_manager(
    cameras: List[CameraConfig],
    freshness_s: float,
    replay_paths: Optional[Dict[str, str]] = None,
    loop_replay: bool = True,
) -> CameraManager:
    """Create a manager for the enabled cameras.

    ``replay_paths`` maps a camera name to a video/image path; those
    cameras are served from disk instead of hardware, which is how the
    hardware-free development mode works.
    """
    replay_paths = replay_paths or {}
    sources: List[FrameSource] = []
    for cfg in cameras:
        if not cfg.enabled:
            continue
        replay = replay_paths.get(cfg.name)
        if replay:
            sources.append(ReplaySource(cfg, replay, freshness_s, loop=loop_replay))
        else:
            sources.append(ThreadedCamera(cfg, freshness_s))
    return CameraManager(sources, freshness_s)


__all__ = [
    "CameraManager",
    "FrameSource",
    "ReplaySource",
    "ThreadedCamera",
    "build_camera_manager",
]


def list_video_devices() -> List[Dict[str, str]]:
    """Enumerate USB video devices and their persistent paths.

    Returns one entry per ``/dev/v4l/by-id`` (or ``by-path``) symlink, with
    the ``/dev/videoN`` node it currently resolves to.  Used by
    ``run.py --check`` and ``scripts/camera_preview.py --list`` so a
    student can fill in ``device_path`` without guessing.

    Returns an empty list on a machine with no V4L2 devices (a laptop, or
    this project's own CI) rather than raising.
    """
    devices: List[Dict[str, str]] = []
    for directory, kind in (("/dev/v4l/by-id", "by-id"), ("/dev/v4l/by-path", "by-path")):
        if not os.path.isdir(directory):
            continue
        for entry in sorted(os.listdir(directory)):
            link = os.path.join(directory, entry)
            try:
                target = os.path.realpath(link)
            except OSError:
                continue
            devices.append({"kind": kind, "path": link, "node": target})
    return devices
