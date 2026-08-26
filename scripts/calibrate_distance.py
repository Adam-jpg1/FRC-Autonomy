#!/usr/bin/env python3
"""Fit the empirical range-correction polynomial for one camera.

WHY
---
The ray-to-ground range is only as good as the mount measurements behind
it. A 1 cm error in camera height or half a degree of pitch turns into a
range error that grows with distance. Rather than pretend the geometry is
perfect, measure the residual and fit it out.

The fit maps the RAW geometric ground distance to the TRUE measured
distance::

    true = a0 + a1*raw + a2*raw^2 + ...

and is stored per camera in ``CameraConfig.distance_poly`` (numpy.polyval
order: highest power first).

COLLECTING DATA
---------------
Interactive, with the real camera and a tape measure::

    python scripts/calibrate_distance.py --camera cam0 --collect

    Put the object on the floor at a measured distance from the ROBOT
    CENTER, click it in the window (click the point where it touches the
    floor), type the true distance, repeat. Ten to fifteen points spread
    over the range you actually care about is plenty.

Fitting data you already have::

    python scripts/calibrate_distance.py --camera cam0 \
        --data data/distance/cam0.csv --order 2

The CSV is ``raw_m,true_m`` with an optional header.

READING THE RESULT
------------------
The script prints the residual error before and after the fit, and the
worst single point. If the residual barely improves, the errors are not
systematic -- suspect the mount measurements or the detector's box bottom
rather than reaching for a higher polynomial order. A high-order fit over
a short range will look wonderful and behave terribly outside it, which
is why the runtime clamps the correction past ``distance_poly_max_m``.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from frc_perception.config import CameraConfig, load_config  # noqa: E402
from frc_perception.geometry import ground_intersection  # noqa: E402


def load_samples(path: str) -> List[Tuple[float, float]]:
    samples: List[Tuple[float, float]] = []
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.reader(handle):
            if not row or row[0].strip().startswith("#"):
                continue
            try:
                raw, true = float(row[0]), float(row[1])
            except (ValueError, IndexError):
                continue  # header or junk line
            samples.append((raw, true))
    return samples


def save_samples(path: str, samples: List[Tuple[float, float]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["raw_m", "true_m"])
        writer.writerows(samples)


def fit(samples: List[Tuple[float, float]], order: int):
    """Least-squares polynomial fit with an honest residual report."""
    if len(samples) < order + 1:
        raise SystemExit(
            f"need at least {order + 1} points for order {order}; have {len(samples)}"
        )
    raw = np.array([s[0] for s in samples], dtype=float)
    true = np.array([s[1] for s in samples], dtype=float)

    coefficients = np.polyfit(raw, true, order)
    corrected = np.polyval(coefficients, raw)

    before = true - raw
    after = true - corrected
    return coefficients, before, after, raw, true


def report(camera_name: str, coefficients, before, after, raw, true, order: int) -> None:
    def rms(values):
        return float(np.sqrt(np.mean(np.square(values))))

    print("\n" + "=" * 72)
    print(f"DISTANCE CALIBRATION for {camera_name}  ({len(raw)} points, order {order})")
    print("=" * 72)
    print(f"  range covered      : {raw.min():.2f} m to {raw.max():.2f} m")
    print(f"  RMS error BEFORE   : {rms(before):.4f} m  (max {np.abs(before).max():.4f} m)")
    print(f"  RMS error AFTER    : {rms(after):.4f} m  (max {np.abs(after).max():.4f} m)")
    print(f"  mean bias BEFORE   : {float(np.mean(before)):+.4f} m")
    print(f"  mean bias AFTER    : {float(np.mean(after)):+.4f} m")

    print("\n  per-point residuals (m):")
    print("    raw      true     corrected   before    after")
    corrected = true - after
    for index in range(len(raw)):
        print(
            f"    {raw[index]:6.3f}  {true[index]:7.3f}  {corrected[index]:9.3f}  "
            f"{before[index]:+8.3f} {after[index]:+8.3f}"
        )

    improvement = rms(before) - rms(after)
    if rms(before) > 1e-9 and improvement / rms(before) < 0.2:
        print(
            "\n  WARNING: the fit barely improves anything. The error is probably not\n"
            "  systematic. Re-measure the camera height and pitch before trusting this,\n"
            "  and check that the detector's box bottom really sits on the floor."
        )
    if order >= 2 and (raw.max() - raw.min()) < 2.0:
        print(
            "\n  WARNING: a high-order fit over a short range extrapolates badly.\n"
            "  Collect points across the full distance you care about, and keep\n"
            "  distance_poly_max_m at the far end of what you measured."
        )

    print("\nPaste into frc_perception/config.py for this camera:\n")
    print("    distance_poly=({}),".format(
        ", ".join(f"{c:.6f}" for c in coefficients)
    ))
    print(f"    distance_poly_max_m={float(raw.max()):.2f},")
    print("=" * 72)


def collect(camera_cfg: CameraConfig, output_path: str) -> List[Tuple[float, float]]:
    """Interactively pair clicked ground points with tape-measure truth."""
    import cv2

    from frc_perception.config import RangingConfig

    ranging = RangingConfig()
    capture = cv2.VideoCapture(camera_cfg.source, cv2.CAP_V4L2)
    if not capture.isOpened():
        raise SystemExit(f"cannot open {camera_cfg.source}")
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*camera_cfg.fourcc))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, camera_cfg.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_cfg.height)

    samples: List[Tuple[float, float]] = []
    clicked = {}

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked["point"] = (x, y)

    cv2.namedWindow("distance calibration")
    cv2.setMouseCallback("distance calibration", on_mouse)
    print(
        "\nClick the point where the object TOUCHES THE FLOOR, then type the true\n"
        "distance from the ROBOT CENTER in meters. Empty input cancels. q quits.\n"
    )
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                print("read failed")
                break
            display = frame.copy()
            text = f"samples: {len(samples)}   click the ground contact point"
            cv2.putText(display, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(display, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow("distance calibration", display)

            if "point" in clicked:
                x, y = clicked.pop("point")
                hit = ground_intersection(
                    x, y, camera_cfg, ranging.ground_plane_z_m, ranging.min_ray_descent
                )
                if hit is None:
                    print("  that pixel does not intersect the ground (above the horizon)")
                    continue
                raw = float(np.hypot(hit[0], hit[1]))
                print(f"  raw geometric distance from robot center: {raw:.3f} m")
                answer = input("  true measured distance (m), blank to cancel: ").strip()
                if not answer:
                    continue
                try:
                    true = float(answer)
                except ValueError:
                    print("  not a number")
                    continue
                samples.append((raw, true))
                save_samples(output_path, samples)
                print(f"  recorded ({raw:.3f}, {true:.3f}); {len(samples)} total")

            if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()
    return samples


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--camera", default="cam0", help="camera name from config")
    parser.add_argument("--collect", action="store_true", help="collect samples live")
    parser.add_argument("--data", default="", help="CSV of raw_m,true_m")
    parser.add_argument("--order", type=int, default=2, help="polynomial order (1-3)")
    args = parser.parse_args()

    if not 1 <= args.order <= 3:
        raise SystemExit("--order must be 1, 2 or 3; higher orders extrapolate wildly")

    cfg = load_config()
    try:
        camera_cfg = cfg.camera(args.camera)
    except KeyError:
        raise SystemExit(
            f"no camera named {args.camera!r}; have {[c.name for c in cfg.cameras]}"
        )
    if camera_cfg.mount_z_m <= 0:
        raise SystemExit(
            f"camera {args.camera} has mount_z_m = {camera_cfg.mount_z_m}. "
            "Measure and set the mount pose in config.py FIRST -- this script "
            "corrects the residual of a measured geometry, it cannot replace it."
        )

    path = args.data or os.path.join("data", "distance", f"{args.camera}.csv")
    if args.collect:
        samples = collect(camera_cfg, path)
    else:
        if not os.path.exists(path):
            raise SystemExit(f"no sample file at {path}; run with --collect first")
        samples = load_samples(path)

    if not samples:
        print("no samples collected; nothing to fit")
        return 1

    coefficients, before, after, raw, true = fit(samples, args.order)
    report(args.camera, coefficients, before, after, raw, true, args.order)
    print(f"\nsamples: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
