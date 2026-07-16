"""FRC 2026 Jetson perception + navigation system.

Runs on a Jetson Orin Nano Super with 2x USB global-shutter cameras and
talks to the roboRIO over NetworkTables.

Coordinate conventions used EVERYWHERE in this package:
  Robot frame:  +X forward, +Y left, +Z up.  Yaw CCW-positive, radians.
  Field frame:  WPILib blue-origin. +X toward the red alliance wall,
                +Y left when standing at the blue wall looking downfield.
  Camera (CV):  OpenCV convention. +X right, +Y down, +Z out the lens.
All distances are meters, all published angles are radians.
"""

__version__ = "1.0.0"
