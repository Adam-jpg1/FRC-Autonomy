"""Convenience alias for :mod:`frc_perception.config`.

The real configuration -- every tunable value and every ``MUST INPUT``
marker -- lives in ``frc_perception/config.py``.  It sits inside the
package so that nothing has to put the repository root on ``sys.path`` to
import it.  This module re-exports the public names so that ``import
config`` also works from the project root::

    python -c "import config; print(config.config_summary(config.load_config()))"

The names are listed explicitly rather than star-imported so that editors
and static checkers can resolve them.

EDIT ``frc_perception/config.py``.  There is nothing to change here.
"""

from frc_perception.config import (
    AppConfig,
    AprilTagConfig,
    CameraConfig,
    CLASS_BUMPER_BLUE,
    CLASS_BUMPER_RED,
    CLASS_FUEL,
    CLASS_TO_KIND,
    ConfigIssue,
    ControllerConfig,
    DetectorConfig,
    EXPECTED_CLASS_NAMES,
    LocalizationConfig,
    LoggingConfig,
    NavigationConfig,
    NetworkTablesConfig,
    RangingConfig,
    WorldModelConfig,
    config_summary,
    load_config,
    validate_config,
)

__all__ = [
    "AppConfig",
    "AprilTagConfig",
    "CameraConfig",
    "CLASS_BUMPER_BLUE",
    "CLASS_BUMPER_RED",
    "CLASS_FUEL",
    "CLASS_TO_KIND",
    "ConfigIssue",
    "ControllerConfig",
    "DetectorConfig",
    "EXPECTED_CLASS_NAMES",
    "LocalizationConfig",
    "LoggingConfig",
    "NavigationConfig",
    "NetworkTablesConfig",
    "RangingConfig",
    "WorldModelConfig",
    "config_summary",
    "load_config",
    "validate_config",
]
