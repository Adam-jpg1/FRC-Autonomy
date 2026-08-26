"""Logging: console + rotating file, with transition and rate-limit helpers.

A 30 Hz loop will happily write 30 identical lines per second and bury the
one line that mattered.  Two helpers exist to stop that:

:class:`StateLogger`
    Logs only when a named boolean/enum state CHANGES ("camera cam0
    disconnected" ... "camera cam0 recovered").  These transitions are the
    events a student debugging on the field actually needs.

:class:`RateLimiter`
    Lets a repeated error through once per configured interval and reports
    how many were suppressed, so a persistent fault is visible without
    flooding the file.
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import time
from typing import Any, Dict, Optional

from .config import LoggingConfig

_CONSOLE_FORMAT = "%(asctime)s %(levelname)-7s %(name)-22s %(message)s"
_FILE_FORMAT = "%(asctime)s %(levelname)-7s %(name)-22s [%(threadName)s] %(message)s"


def setup_logging(cfg: LoggingConfig, verbose: bool = False) -> logging.Logger:
    """Configure the root logger.  Safe to call more than once.

    A failure to open the log file (read-only filesystem, full disk) is
    reported on the console and does not prevent the program from running:
    losing the log is bad, refusing to run the robot is worse.
    """
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()

    level = logging.DEBUG if verbose else getattr(logging, cfg.level.upper(), logging.INFO)
    console_level = (
        logging.DEBUG
        if verbose
        else getattr(logging, cfg.console_level.upper(), logging.INFO)
    )
    root.setLevel(min(level, console_level))

    console = logging.StreamHandler()
    console.setLevel(console_level)
    console.setFormatter(logging.Formatter(_CONSOLE_FORMAT, datefmt="%H:%M:%S"))
    root.addHandler(console)

    try:
        os.makedirs(cfg.directory, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            os.path.join(cfg.directory, cfg.filename),
            maxBytes=cfg.max_bytes,
            backupCount=cfg.backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(_FILE_FORMAT))
        root.addHandler(file_handler)
    except OSError as exc:
        root.error("file logging disabled: %s", exc)

    return root


class StateLogger:
    """Emits a log line only when a named state changes.

    ``log("odometry", fresh, "odometry recovered", "ODOMETRY STALE")``
    logs nothing while the state is steady and one line at each edge.
    """

    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger
        self._states: Dict[str, Any] = {}

    def changed(self, key: str, value: Any) -> bool:
        previous = self._states.get(key, _UNSET)
        if previous is _UNSET or previous != value:
            self._states[key] = value
            return previous is not _UNSET
        return False

    def log(
        self,
        key: str,
        value: Any,
        on_true: str,
        on_false: str,
        level_true: int = logging.INFO,
        level_false: int = logging.WARNING,
    ) -> None:
        """Log a transition of a boolean state."""
        if self.changed(key, bool(value)):
            if value:
                self._logger.log(level_true, "%s", on_true)
            else:
                self._logger.log(level_false, "%s", on_false)

    def log_value(self, key: str, value: Any, message: str, level: int = logging.INFO) -> None:
        """Log when a non-boolean state (e.g. an enum) changes value."""
        if self.changed(key, value):
            self._logger.log(level, "%s -> %s", message, value)

    def prime(self, key: str, value: Any) -> None:
        """Record a state without logging, so the first real change logs."""
        self._states[key] = value


class _Unset:
    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<unset>"


_UNSET = _Unset()


class RateLimiter:
    """Allows one message per key per interval, counting the suppressed ones."""

    def __init__(self, interval_s: float) -> None:
        self.interval_s = interval_s
        self._last: Dict[str, float] = {}
        self._suppressed: Dict[str, int] = {}

    def allow(self, key: str, now: Optional[float] = None) -> bool:
        moment = time.monotonic() if now is None else now
        last = self._last.get(key)
        if last is None or moment - last >= self.interval_s:
            self._last[key] = moment
            return True
        self._suppressed[key] = self._suppressed.get(key, 0) + 1
        return False

    def take_suppressed(self, key: str) -> int:
        return self._suppressed.pop(key, 0)

    def log(
        self,
        logger: logging.Logger,
        level: int,
        key: str,
        message: str,
        *args: Any,
        now: Optional[float] = None,
    ) -> None:
        """Log ``message`` at most once per interval for ``key``."""
        if self.allow(key, now):
            suppressed = self.take_suppressed(key)
            suffix = f" (+{suppressed} suppressed)" if suppressed else ""
            logger.log(level, message + suffix, *args)


__all__ = ["RateLimiter", "StateLogger", "setup_logging"]
