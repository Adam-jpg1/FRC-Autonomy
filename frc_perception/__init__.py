"""FRC 2026 Jetson perception, localization, world model and navigation.

Layer map (each layer depends only on the ones above it)::

    camera.py             frame capture, health, freshness
    detector.py           YOLO object detection (predict only)
    geometry.py           pixel -> robot-frame range and bearing
    apriltag_locator.py   AprilTag detection and per-tag pose solving
    field_map.py          the official WPILib tag layout
    localization.py       odometry + tag fusion, pose validity gate
    world_model.py        field-space de-duplication and tracking
    navigation.py         motion recommendation
    virtual_controller.py motion -> gamepad axes
    nt_io.py              NetworkTables contract
    app.py                the loop that wires it together

Conventions, units and frames are documented in ``types.py``; every
tunable value lives in ``config.py``.
"""

__all__ = ["__version__"]

__version__ = "1.0.0"
