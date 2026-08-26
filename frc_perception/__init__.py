"""FRC 2026 perception, localization, world model and navigation.

Target platform: Raspberry Pi 5 (8 GB) + Raspberry Pi AI HAT+ (Hailo-8),
two Arducam OV9782 USB cameras, roboRIO over NetworkTables 4.

Layer map (each layer depends only on the ones above it)::

    camera.py             USB/V4L2 frame capture, health, freshness
    detector.py           THE HARDWARE BOUNDARY: Hailo-8 and PyTorch
                          detector backends behind one interface
    inference_scheduler.py  two cameras -> one accelerator, newest frame
                          only, no queue
    geometry.py           pixel -> robot-frame range and bearing
    apriltag_locator.py   AprilTag detection and per-tag pose solving
    field_map.py          the official WPILib tag layout
    localization.py       odometry + tag fusion, pose validity gate
    world_model.py        field-space de-duplication and tracking
    navigation.py         motion recommendation
    virtual_controller.py motion -> gamepad axes
    nt_io.py              NetworkTables contract
    app.py                the loop that wires it together

Only ``camera.py``, ``detector.py`` and ``inference_scheduler.py`` know
what hardware this is.  From ``types.Detection`` downward the stack is
hardware-independent: swapping the accelerator should not change a line of
geometry, localization, world modeling or navigation.

Conventions, units and frames are documented in ``types.py``; every
tunable value lives in ``config.py``.
"""

__all__ = ["__version__"]

__version__ = "1.0.0"
