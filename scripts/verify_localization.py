#!/usr/bin/env python3
"""Live localization verifier: prove the signs and transforms on a real field.

    python scripts/verify_localization.py
    python scripts/verify_localization.py --camera cam0 --window
    python scripts/verify_localization.py --sim-nt        # plumbing check, no RIO

WHAT THIS IS FOR
----------------
Every convention in this project -- +X down field, +Y left, CCW-positive
heading, camera pitch positive down, the AprilTag-to-WPILib tag frame
conversion -- is internally consistent and unit tested.  None of that
proves the numbers you measured for the camera mount are right, or that
your roboRIO publishes odometry in the frame this code assumes.

Only pushing the real robot around a real field proves that.  Run this,
watch the table, and confirm each of the following BEFORE trusting
field-relative navigation:

  1. Push the robot DOWN FIELD (toward the red wall): odometry X and tag X
     both INCREASE, together.
  2. Push the robot to its LEFT while facing down field: both Y INCREASE.
  3. Rotate the robot COUNTER-CLOCKWISE seen from above: both headings
     INCREASE.
  4. Park at a known spot with a tape measure: tag X/Y match reality to a
     few centimeters. If X and Y are swapped or mirrored, the camera yaw
     or the mount sign is wrong -- fix the CONFIG, never the math.
  5. Point a camera at a tag from the side: the tag pose must stay put as
     you move the robot around it.
  6. Watch the disagreement columns while driving a lap. Growing
     disagreement means odometry drift (expected) or a bad transform
     (not).

Columns:
  ODOM      raw roboRIO odometry (blue-origin field frame)
  TAG       robot pose implied by the AprilTags seen this instant
  FUSED     what the rest of the system will act on
  VALID     the pose-validity gate; navigation translates only when true
  d_trans   translation disagreement, tag vs fused, meters
  d_head    heading disagreement, tag vs fused, degrees
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from frc_perception.apriltag_locator import AprilTagLocalizer, to_grayscale  # noqa: E402
from frc_perception.camera import build_camera_manager  # noqa: E402
from frc_perception.config import load_config  # noqa: E402
from frc_perception.field_map import try_load_field_map  # noqa: E402
from frc_perception.localization import PoseEstimator  # noqa: E402
from frc_perception.logging_setup import setup_logging  # noqa: E402
from frc_perception.nt_io import NetworkTablesIO, SimulatedNT  # noqa: E402
from frc_perception.types import Pose2d  # noqa: E402

HEADER = (
    f"{'ODOM x':>8} {'y':>7} {'deg':>7} | "
    f"{'TAG x':>8} {'y':>7} {'deg':>7} | "
    f"{'FUSED x':>8} {'y':>7} {'deg':>7} | "
    f"{'VALID':>6} {'src':>22} {'d_trans':>8} {'d_head':>7} {'tags':>5}  ids"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--camera", action="append", help="limit to these cameras")
    parser.add_argument("--sim-nt", action="store_true", help="simulated NetworkTables")
    parser.add_argument("--window", action="store_true", help="show the tag detections")
    parser.add_argument("--rate", type=float, default=5.0, help="print rate, Hz")
    args = parser.parse_args()

    cfg = load_config()
    setup_logging(cfg.logging)

    field_map, error = try_load_field_map(cfg.localization.apriltag.field_layout_path)
    if field_map is None:
        print(f"FAIL: {error}")
        print("Run scripts/download_field_layout.py first -- there is nothing to")
        print("verify without the official tag layout.")
        return 1

    uncalibrated = [c.name for c in cfg.enabled_cameras if not c.has_calibrated_intrinsics]
    if uncalibrated:
        print(
            f"FAIL: {uncalibrated} have no calibrated intrinsics. AprilTag pose\n"
            "solving needs them; run scripts/calibrate_intrinsics.py first."
        )
        return 1
    if cfg.localization.apriltag.tag_size_m is None:
        print("FAIL: config.localization.apriltag.tag_size_m is unset (MUST INPUT).")
        return 1

    cameras = cfg.enabled_cameras
    if args.camera:
        wanted = set(args.camera)
        cameras = [c for c in cameras if c.name in wanted]
        if not cameras:
            print(f"FAIL: no enabled camera matches {sorted(wanted)}")
            return 1

    localizer = AprilTagLocalizer(cfg.localization.apriltag, field_map)
    if not localizer.load():
        print(f"FAIL: {localizer.load_error}")
        return 1

    estimator = PoseEstimator(cfg.localization, field_map)
    nt = SimulatedNT() if args.sim_nt else NetworkTablesIO(cfg.networktables)
    nt.connect()

    manager = build_camera_manager(cameras, cfg.frame_freshness_s)
    manager.start()

    by_name = {c.name: c for c in cameras}
    period = 1.0 / max(args.rate, 0.5)
    last_print = 0.0
    row = 0
    print("\n" + HEADER)
    print("-" * len(HEADER))

    try:
        while True:
            now = time.monotonic()
            reading = nt.read_odometry()
            if reading is not None:
                x, y, theta, rio_timestamp = reading
                estimator.update_odometry(Pose2d(x, y, theta), rio_timestamp, now)

            frames = manager.fresh_frames(now)
            observations = []
            for name, frame in frames.items():
                camera_cfg = by_name.get(name)
                if camera_cfg is None or not camera_cfg.use_for_apriltags:
                    continue
                try:
                    gray = to_grayscale(frame.image)
                    observations.extend(
                        localizer.detect_frame(gray, camera_cfg, frame.timestamp)
                    )
                except Exception as exc:  # noqa: BLE001 - keep the tool alive
                    print(f"  ({name} detect failed: {exc})")

            tag_estimate = localizer.estimate(observations, now)
            if tag_estimate is not None:
                estimator.add_tag_estimate(tag_estimate, now)
            fused = estimator.update(now)

            if now - last_print >= period:
                last_print = now
                odom = estimator.odometry.reading
                if odom is None:
                    odom_text = f"{'--':>8} {'--':>7} {'--':>7}"
                else:
                    odom_text = (
                        f"{odom.pose.x:8.3f} {odom.pose.y:7.3f} "
                        f"{math.degrees(odom.pose.theta):7.2f}"
                    )
                if tag_estimate is None:
                    tag_text = f"{'--':>8} {'--':>7} {'--':>7}"
                    ids = ""
                else:
                    tag_text = (
                        f"{tag_estimate.pose.x:8.3f} {tag_estimate.pose.y:7.3f} "
                        f"{math.degrees(tag_estimate.pose.theta):7.2f}"
                    )
                    ids = ",".join(str(i) for i in tag_estimate.tag_ids)

                row += 1
                if row % 25 == 1:
                    print("\n" + HEADER)
                    print("-" * len(HEADER))
                print(
                    f"{odom_text} | {tag_text} | "
                    f"{fused.pose.x:8.3f} {fused.pose.y:7.3f} "
                    f"{math.degrees(fused.pose.theta):7.2f} | "
                    f"{'YES' if fused.valid else 'NO':>6} {fused.source.value:>22} "
                    f"{fused.translation_disagreement_m:8.3f} "
                    f"{math.degrees(fused.heading_disagreement_rad):7.2f} "
                    f"{len(observations):5d}  {ids}"
                )
                if not fused.valid and fused.invalid_reason:
                    print(f"        reason: {fused.invalid_reason}")
                if localizer.rejections:
                    print(f"        tag rejections so far: {localizer.rejections}")

            if args.window and frames:
                import cv2

                for name, frame in frames.items():
                    display = frame.image.copy()
                    for observation in observations:
                        if observation.camera_id != name:
                            continue
                        cv2.putText(
                            display,
                            f"tag {observation.tag_id} @ {observation.distance_m:.2f} m",
                            (10, 30 + 26 * observation.tag_id % 300),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA,
                        )
                    cv2.imshow(f"verify {name}", display)
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                    break

            time.sleep(0.01)
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        manager.stop()
        nt.close()
        if args.window:
            try:
                import cv2

                cv2.destroyAllWindows()
            except ImportError:
                pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
