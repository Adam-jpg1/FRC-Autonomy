#!/usr/bin/env python3
"""Distance calibration: fit the per-camera range-correction polynomial.

Prereqs: camera intrinsics + mount values already entered in config, the
YOLO model present, and one fuel piece on the floor.

    python3 scripts/calibrate_distance.py --camera front

Procedure (terminal-only, no display needed):
  1. Place the fuel at a tape-measured distance from the CAMERA LENS,
     projected on the floor (horizontal distance, not slant).
  2. Enter that distance in meters at the prompt.
  3. The script averages the RAW (uncorrected) vision estimate over ~30
     frames and stores the pair.
  4. Repeat over your working envelope (e.g. 0.5 - 5 m, ~6-10 points),
     then enter 'fit'.

It prints `distance_poly=(...)` to paste into that camera's CameraConfig.
The poly maps raw ground distance -> true ground distance (numpy.polyval
order, highest power first).
"""

import argparse
import sys
import time

import numpy as np

sys.path.insert(0, ".")  # run from the repo root

from frc_perception.camera import CameraWorker
from frc_perception.config import default_config
from frc_perception.detector import Detector
from frc_perception.geometry import estimate_object_position


def measure_raw_distance(camera: CameraWorker, detector: Detector,
                         cam_cfg, samples: int = 30, timeout_s: float = 15.0):
    """Average raw (identity-poly) camera->fuel ground distance."""
    values = []
    deadline = time.monotonic() + timeout_s
    while len(values) < samples and time.monotonic() < deadline:
        got = camera.get_frame(max_age_s=0.25)
        if got is None:
            time.sleep(0.02)
            continue
        dets = detector.detect_batch([got[0]])[0]
        fuel = [d for d in dets if d.class_name == "fuel"]
        if not fuel:
            time.sleep(0.02)
            continue
        best = max(fuel, key=lambda d: d.confidence)
        est = estimate_object_position(best.bbox_xyxy, cam_cfg, known_width_m=None)
        if est is not None:
            values.append(est.ground_distance_m)
        time.sleep(0.02)
    return (float(np.mean(values)), float(np.std(values)), len(values)) if values else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", required=True, help="camera name from config")
    parser.add_argument("--degree", type=int, default=2, help="polynomial degree")
    args = parser.parse_args()

    cfg = default_config()
    cam_cfg = next((c for c in cfg.cameras if c.name == args.camera), None)
    if cam_cfg is None:
        print(f"no camera named '{args.camera}' in config", file=sys.stderr)
        return 1
    if cam_cfg.distance_poly != (1.0, 0.0):
        print("NOTE: overriding existing distance_poly with identity for this run.")
    cam_cfg = type(cam_cfg)(**{**cam_cfg.__dict__, "distance_poly": (1.0, 0.0)})

    detector = Detector(cfg.detector)
    camera = CameraWorker(cam_cfg)
    camera.start()
    print("warming up camera ...")
    time.sleep(2.0)

    pairs = []  # (raw_estimate, true_distance)
    try:
        while True:
            entry = input(
                f"\n[{len(pairs)} points] true distance in meters "
                "(or 'fit' / 'quit'): "
            ).strip().lower()
            if entry == "quit":
                return 0
            if entry == "fit":
                break
            try:
                true_dist = float(entry)
            except ValueError:
                print("enter a number, 'fit', or 'quit'")
                continue
            print("measuring ...")
            result = measure_raw_distance(camera, detector, cam_cfg)
            if result is None:
                print("no fuel detected — check placement/lighting and retry")
                continue
            mean, std, n = result
            print(f"raw estimate {mean:.3f} m (std {std:.3f}, n={n}) vs true {true_dist:.3f} m")
            pairs.append((mean, true_dist))
    finally:
        camera.stop()

    if len(pairs) <= args.degree:
        print(f"need more than {args.degree} points for a degree-{args.degree} fit",
              file=sys.stderr)
        return 1
    raw = np.array([p[0] for p in pairs])
    true = np.array([p[1] for p in pairs])
    coeffs = np.polyfit(raw, true, args.degree)
    residuals = np.polyval(coeffs, raw) - true
    print(f"\nfit residuals: max {np.max(np.abs(residuals)):.3f} m, "
          f"rms {np.sqrt(np.mean(residuals ** 2)):.3f} m")
    print(f"Paste into CameraConfig '{args.camera}':\n")
    print("    distance_poly=(" + ", ".join(f"{c:.6f}" for c in coeffs) + "),")
    return 0


if __name__ == "__main__":
    sys.exit(main())
