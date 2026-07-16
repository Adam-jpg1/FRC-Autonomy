#!/usr/bin/env python3
"""Live localization verifier — validate sign conventions ON THE FIELD.

Prints odometry, per-camera AprilTag pose, fused pose, and disagreement
side by side at ~5 Hz. Use it to confirm, with the robot at a known
spot (e.g. field center, facing the red wall):

  * robot_x grows as you push the robot toward the RED alliance wall
  * robot_y grows as you push it LEFT (viewed from behind the blue wall)
  * theta is 0 facing +X and grows COUNTER-clockwise
  * tag pose and odometry roughly agree (if not, the RIO is probably
    publishing alliance-flipped odometry instead of raw blue-origin)

    python3 scripts/verify_localization.py           # tags + NT odometry
    python3 scripts/verify_localization.py --no-nt   # tags only, no RIO needed
"""

import argparse
import math
import sys
import time

import cv2

sys.path.insert(0, ".")  # run from the repo root

from frc_perception.camera import CameraWorker
from frc_perception.config import default_config
from frc_perception.field_map import FieldMap
from frc_perception.localization import OdometryTracker, PoseFuser, TagLocalizer
from frc_perception.logging_setup import setup_logging


def fmt(pose) -> str:
    if pose is None:
        return "        ---        "
    return f"({pose.x:6.2f},{pose.y:6.2f},{math.degrees(pose.theta):7.1f}°)"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-nt", action="store_true",
                        help="skip NetworkTables (tag-only verification)")
    args = parser.parse_args()

    cfg = default_config()
    setup_logging(cfg.log_dir, "WARNING")  # keep the console table readable
    field_map = FieldMap(
        cfg.localization.field_layout_path,
        cfg.localization.fallback_field_length_m,
        cfg.localization.fallback_field_width_m,
    )
    if not field_map.loaded:
        print("field layout missing — run scripts/download_field_layout.py first",
              file=sys.stderr)
        return 1

    nt = None
    if not args.no_nt:
        from frc_perception.nt_io import NTInterface
        nt = NTInterface(cfg)

    localizer = TagLocalizer(cfg.localization, field_map, cfg.cameras)
    odom = OdometryTracker(cfg.localization.odom_stale_s)
    fuser = PoseFuser(cfg.localization)
    cameras = [CameraWorker(c) for c in cfg.cameras]
    for cam in cameras:
        cam.start()

    print(f"{'odometry':^21} | {'tag solve':^21} | {'fused':^21} | tags | disagree")
    try:
        while True:
            now = time.monotonic()
            if nt is not None:
                sample = nt.get_odometry()
                if sample is not None:
                    odom.update(*sample, now=now)

            fixes = []
            for cam in cameras:
                got = cam.get_frame(cfg.frame_freshness_s)
                if got is None:
                    continue
                gray = cv2.cvtColor(got[0], cv2.COLOR_BGR2GRAY)
                fixes.extend(localizer.process(gray, cam.cfg))
            tag_pose = localizer.combine(fixes)
            if tag_pose is not None:
                fuser.update_from_tag(tag_pose, odom, now)
            fused = fuser.estimate(odom, now)

            disagree = ""
            if tag_pose is not None and fused.valid:
                disagree = f"{math.hypot(tag_pose.x - fused.x, tag_pose.y - fused.y):.2f} m"
            tag_ids = ",".join(str(f.tag_id) for f in fixes) or "-"
            fused_pose = fused if fused.valid else None
            print(
                f"{fmt(odom.pose if odom.fresh(now) else None)} | {fmt(tag_pose)} | "
                f"{fmt(fused_pose)} | {tag_ids:>4} | {disagree}"
            )
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        for cam in cameras:
            cam.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
