"""Structured logging: rotating file + console, one call to set up."""

import logging
import logging.handlers
import os

_FORMAT = "%(asctime)s.%(msecs)03d %(levelname)-7s %(name)s: %(message)s"
_DATEFMT = "%H:%M:%S"


def setup_logging(log_dir: str, level: str = "INFO") -> None:
    """Configure the root logger. Safe to call once at process start."""
    os.makedirs(log_dir, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(_FORMAT, _DATEFMT))
    root.addHandler(console)

    file_handler = logging.handlers.RotatingFileHandler(
        os.path.join(log_dir, "perception.log"),
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
    )
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    )
    root.addHandler(file_handler)

    # Ultralytics is chatty at INFO even with verbose=False on predict.
    logging.getLogger("ultralytics").setLevel(logging.WARNING)
