#!/usr/bin/env python3
"""Camera intrinsic calibration from a printed chessboard.

WHY THIS MATTERS
----------------
Ground ranging and AprilTag pose solving both assume a correct lens model.
An uncalibrated camera can be off by 10-20% in range at the far end of the
field and several degrees in tag pose, and neither error looks like an
error -- the numbers just come out confidently wrong.

WHAT YOU NEED
-------------
1. A chessboard printed FLAT and mounted on something rigid (foam board,
   clipboard). A wavy printout ruins the calibration silently.
2. Its INNER corner count and square size. A "9x6" board has 9x6 inner
   corners and 10x7 squares. Measure one square with calipers.
3. 20-30 captures with the board at many angles, distances and image
   positions -- especially near the image CORNERS, which is where the
   distortion coefficients get their information.

USAGE
-----
Capture and calibrate live::

    python scripts/calibrate_intrinsics.py --camera cam0 \
        --board 9x6 --square-size 0.025

    SPACE  capture the current view (only accepted if corners are found)
    c      run the calibration on what has been captured
    d      delete the last capture
    q      quit

Or calibrate from images you already saved::

    python scripts/calibrate_intrinsics.py --images data/calib/cam0/*.png \
        --board 9x6 --square-size 0.025 --camera cam0

Results are written to ``data/calibration/<camera>.json`` and printed as a
config snippet to paste into ``frc_perception/config.py``.

READING THE RESULT
------------------
The RMS reprojection error should be well under 1.0 px for a global
shutter camera with a decent board. Above ~1.5 px, something is wrong:
a bent board, motion blur, a wrong square size, or too few views.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from frc_perception.config import CameraConfig, load_config  # noqa: E402

CORNER_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def parse_board(text: str):
    try:
        columns, rows = (int(part) for part in text.lower().split("x"))
    except ValueError:
        raise SystemExit(f"--board expects COLSxROWS of INNER corners, got {text!r}")
    if columns < 3 or rows < 3:
        raise SystemExit("a usable board has at least 3x3 inner corners")
    return columns, rows


def object_points(columns: int, rows: int, square_size: float) -> np.ndarray:
    """One board's 3D corner coordinates, Z=0, in meters."""
    grid = np.zeros((columns * rows, 3), np.float32)
    grid[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2)
    return grid * square_size


def find_corners(gray: np.ndarray, columns: int, rows: int):
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, (columns, rows), flags)
    if not found:
        return None
    return cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), CORNER_CRITERIA)


def calibrate(image_points, columns, rows, square_size, image_size):
    """Run the calibration and report per-view reprojection error."""
    objects = [object_points(columns, rows, square_size) for _ in image_points]
    rms, camera_matrix, distortion, rvecs, tvecs = cv2.calibrateCamera(
        objects, image_points, image_size, None, None
    )
    per_view = []
    for index, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
        projected, _ = cv2.projectPoints(
            objects[index], rvec, tvec, camera_matrix, distortion
        )
        error = cv2.norm(image_points[index], projected, cv2.NORM_L2) / len(projected)
        per_view.append(float(error))
    return rms, camera_matrix, distortion, per_view


def report(name, rms, camera_matrix, distortion, per_view, image_size, count):
    fx = float(camera_matrix[0, 0])
    fy = float(camera_matrix[1, 1])
    cx = float(camera_matrix[0, 2])
    cy = float(camera_matrix[1, 2])
    coefficients = [float(v) for v in np.asarray(distortion).ravel()[:5]]
    while len(coefficients) < 5:
        coefficients.append(0.0)

    print("\n" + "=" * 72)
    print(f"CALIBRATION RESULT for {name}   ({count} views, {image_size[0]}x{image_size[1]})")
    print("=" * 72)
    print(f"  RMS reprojection error : {rms:.4f} px")
    print(f"  worst single view      : {max(per_view):.4f} px")
    print(f"  fx, fy                 : {fx:.3f}, {fy:.3f}")
    print(f"  cx, cy                 : {cx:.3f}, {cy:.3f}")
    print(f"  distortion (k1 k2 p1 p2 k3): {['%.6f' % c for c in coefficients]}")
    if rms > 1.5:
        print("\n  WARNING: RMS above 1.5 px. Do NOT ship this. Common causes:")
        print("    - the board is not flat, or the printed square size is wrong")
        print("    - too few views, or all views from the same angle/distance")
        print("    - motion blur (hold still; the OV9782 is global shutter but the")
        print("      board still has to be stationary at capture)")
    elif rms > 0.8:
        print("\n  NOTE: RMS is usable but not great. More corner-of-frame views help.")

    horizontal_fov = 2.0 * np.degrees(np.arctan(image_size[0] / (2.0 * fx)))
    print(f"\n  implied horizontal FOV : {horizontal_fov:.1f} deg")

    print("\nPaste into frc_perception/config.py for this camera:\n")
    print(f"    fx={fx:.3f},")
    print(f"    fy={fy:.3f},")
    print(f"    cx={cx:.3f},")
    print(f"    cy={cy:.3f},")
    print("    dist_coeffs=({}),".format(", ".join(f"{c:.6f}" for c in coefficients)))
    print(f"    calibration_rms_px={rms:.4f},")
    print(f"    width={image_size[0]}, height={image_size[1]},")
    print(
        "\nIMPORTANT: intrinsics are only valid at the resolution they were "
        "calibrated at. If you change the capture resolution, recalibrate."
    )
    return {
        "camera": name,
        "fx": fx, "fy": fy, "cx": cx, "cy": cy,
        "dist_coeffs": coefficients,
        "rms_px": float(rms),
        "worst_view_px": float(max(per_view)),
        "image_width": image_size[0],
        "image_height": image_size[1],
        "views": count,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def save(result, camera_name):
    directory = os.path.join("data", "calibration")
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"{camera_name}.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(f"\nsaved {path}")


def from_images(paths, columns, rows, square_size, name):
    image_points = []
    image_size = None
    for path in paths:
        image = cv2.imread(path)
        if image is None:
            print(f"  skip (unreadable): {path}")
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = (gray.shape[1], gray.shape[0])
        elif (gray.shape[1], gray.shape[0]) != image_size:
            print(f"  skip (different size): {path}")
            continue
        corners = find_corners(gray, columns, rows)
        if corners is None:
            print(f"  skip (no corners): {path}")
            continue
        image_points.append(corners)
        print(f"  ok: {path}")
    if len(image_points) < 5:
        raise SystemExit(f"only {len(image_points)} usable views; need at least 5 (20+ preferred)")
    rms, matrix, distortion, per_view = calibrate(
        image_points, columns, rows, square_size, image_size
    )
    result = report(name, rms, matrix, distortion, per_view, image_size, len(image_points))
    save(result, name)


def live(camera_cfg: CameraConfig, columns, rows, square_size, save_dir):
    capture = cv2.VideoCapture(camera_cfg.source, cv2.CAP_V4L2)
    if not capture.isOpened():
        raise SystemExit(f"cannot open {camera_cfg.source}")
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*camera_cfg.fourcc))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, camera_cfg.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_cfg.height)

    image_points = []
    image_size = None
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    print(
        "\nSPACE = capture (needs visible corners), c = calibrate, "
        "d = drop last, q = quit\n"
    )
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                print("read failed")
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            image_size = (gray.shape[1], gray.shape[0])
            corners = find_corners(gray, columns, rows)

            display = frame.copy()
            if corners is not None:
                cv2.drawChessboardCorners(display, (columns, rows), corners, True)
            status = f"views: {len(image_points)}  corners: {'YES' if corners is not None else 'no'}"
            cv2.putText(display, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(display, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 0) if corners is not None else (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imshow("intrinsic calibration", display)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" ") and corners is not None:
                image_points.append(corners)
                print(f"captured view {len(image_points)}")
                if save_dir:
                    path = os.path.join(save_dir, f"view_{len(image_points):03d}.png")
                    cv2.imwrite(path, frame)
            if key == ord("d") and image_points:
                image_points.pop()
                print(f"dropped; {len(image_points)} remain")
            if key == ord("c"):
                if len(image_points) < 5:
                    print(f"only {len(image_points)} views; capture at least 5 (20+ preferred)")
                    continue
                rms, matrix, distortion, per_view = calibrate(
                    image_points, columns, rows, square_size, image_size
                )
                result = report(
                    camera_cfg.name, rms, matrix, distortion, per_view,
                    image_size, len(image_points),
                )
                save(result, camera_cfg.name)
    finally:
        capture.release()
        cv2.destroyAllWindows()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--camera", default="cam0", help="camera name from config")
    parser.add_argument("--images", nargs="+", help="calibrate from saved images instead")
    parser.add_argument("--board", default="9x6", help="INNER corners, e.g. 9x6")
    parser.add_argument(
        "--square-size", type=float, required=True,
        help="chessboard square edge length in METERS (measure it)",
    )
    parser.add_argument("--save-dir", default="", help="also save captured views here")
    args = parser.parse_args()

    columns, rows = parse_board(args.board)
    if args.square_size <= 0 or args.square_size > 0.5:
        raise SystemExit("--square-size is in meters; 0.025 means 25 mm")

    if args.images:
        paths = []
        for pattern in args.images:
            paths.extend(sorted(glob.glob(pattern)) or [pattern])
        from_images(paths, columns, rows, args.square_size, args.camera)
        return 0

    cfg = load_config()
    try:
        camera_cfg = cfg.camera(args.camera)
    except KeyError:
        raise SystemExit(
            f"no camera named {args.camera!r} in config; have "
            f"{[c.name for c in cfg.cameras]}"
        )
    live(camera_cfg, columns, rows, args.square_size, args.save_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
