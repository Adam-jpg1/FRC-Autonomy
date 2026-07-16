"""Threaded USB camera capture with freshness enforcement.

One `CameraWorker` thread per camera. Each grab is stamped with a
monotonic timestamp; consumers call `get_frame(max_age)` and get None
instead of a stale frame. A frozen or unplugged camera therefore feeds
the pipeline NOTHING rather than an old image, and the worker keeps
trying to reopen the device in the background.
"""

import logging
import threading
import time
from typing import Optional, Tuple

from .config import CameraConfig

log = logging.getLogger(__name__)

_REOPEN_BACKOFF_S = 2.0


class CameraWorker:
    """Owns one cv2.VideoCapture on a background thread."""

    def __init__(self, cfg: CameraConfig):
        self.cfg = cfg
        self._lock = threading.Lock()
        self._frame = None           # latest BGR image (numpy array)
        self._frame_time = 0.0       # time.monotonic() at capture
        self._frame_seq = 0
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._cap = None
        self.consecutive_failures = 0

    # -- lifecycle ----------------------------------------------------------

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(
            target=self._loop, name=f"cam-{self.cfg.name}", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        self._release()

    # -- consumer API ---------------------------------------------------------

    def get_frame(self, max_age_s: float) -> Optional[Tuple["object", float, int]]:
        """Return (frame, capture_time, seq) if fresh enough, else None.

        The returned array is the worker's latest buffer; treat it as
        read-only (YOLO and the tag detector never write to it).
        """
        now = time.monotonic()
        with self._lock:
            if self._frame is None or (now - self._frame_time) > max_age_s:
                return None
            return self._frame, self._frame_time, self._frame_seq

    @property
    def age_s(self) -> float:
        """Seconds since the last good frame (inf if none yet)."""
        with self._lock:
            if self._frame is None:
                return float("inf")
            return time.monotonic() - self._frame_time

    # -- capture thread -------------------------------------------------------

    def _open(self) -> bool:
        import cv2  # local import: keeps this module importable without OpenCV

        self._release()
        cap = cv2.VideoCapture(self.cfg.device_index, cv2.CAP_V4L2)
        if not cap.isOpened():
            return False
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.cfg.fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.height)
        cap.set(cv2.CAP_PROP_FPS, self.cfg.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # always read the newest frame
        got_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        got_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        if (got_w, got_h) != (self.cfg.width, self.cfg.height):
            log.warning(
                "camera %s: asked %dx%d, driver gave %dx%d — intrinsics assume "
                "the configured size, fix the mismatch",
                self.cfg.name, self.cfg.width, self.cfg.height,
                int(got_w), int(got_h),
            )
        self._cap = cap
        log.info("camera %s opened on /dev/video%d", self.cfg.name, self.cfg.device_index)
        return True

    def _release(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

    def _loop(self) -> None:
        while self._running:
            if self._cap is None:
                if not self._open():
                    log.warning(
                        "camera %s: cannot open /dev/video%d, retrying in %.0fs",
                        self.cfg.name, self.cfg.device_index, _REOPEN_BACKOFF_S,
                    )
                    time.sleep(_REOPEN_BACKOFF_S)
                    continue
            ok, frame = self._cap.read()
            if not ok or frame is None:
                self.consecutive_failures += 1
                if self.consecutive_failures >= 5:
                    log.error("camera %s: read failing, reopening device", self.cfg.name)
                    self._release()
                    self.consecutive_failures = 0
                    time.sleep(_REOPEN_BACKOFF_S)
                continue
            self.consecutive_failures = 0
            now = time.monotonic()
            with self._lock:
                self._frame = frame
                self._frame_time = now
                self._frame_seq += 1
        self._release()
