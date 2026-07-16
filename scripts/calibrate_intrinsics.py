#!/usr/bin/env python3
"""Camera intrinsics calibration with an OpenCV chessboard.

Print a chessboard (e.g. 9x6 INNER corners, known square size), then:

    python3 scripts/calibrate_intrinsics.py --device 0 --width 1280 --height 800 \
        --corners 9x6 --square-size-m 0.024

Controls (needs a display; use `ssh -X` or a monitor on the Jetson):
    SPACE  capture the current view (aim for 15-25 captures, varied
           angles/distances, corners reaching the image edges)
    c      calibrate with captures so far and print the config snippet
    q      quit

Paste the printed fx/fy/cx/cy/dist_coeffs into the matching CameraConfig
in frc_perception/config.py. Calibrate at the SAME resolution you run at.
"""

import argparse
import sys

import cv2
import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--corners", default="9x6", help="inner corners, COLSxROWS")
    parser.add_argument("--square-size-m", type=float, required=True)
    args = parser.parse_args()

    cols, rows = (int(v) for v in args.corners.lower().split("x"))
    pattern = (cols, rows)
    # Object points for one board view: a flat grid in board coordinates.
    template = np.zeros((cols * rows, 3), np.float32)
    template[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * args.square_size_m

    cap = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        print(f"cannot open /dev/video{args.device}", file=sys.stderr)
        return 1

    object_points, image_points = [], []
    image_size = None
    print(__doc__)
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        image_size = gray.shape[::-1]
        found, corners = cv2.findChessboardCorners(gray, pattern, None)
        display = frame.copy()
        if found:
            cv2.drawChessboardCorners(display, pattern, corners, found)
        cv2.putText(
            display, f"captures: {len(object_points)}  (SPACE=capture c=calibrate q=quit)",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
        )
        cv2.imshow("calibrate", display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" ") and found:
            refined = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
            )
            object_points.append(template)
            image_points.append(refined)
            print(f"captured ({len(object_points)})")
        elif key == ord("c"):
            break
        elif key == ord("q"):
            return 0

    if len(object_points) < 8:
        print("need at least 8 captures for a trustworthy calibration", file=sys.stderr)
        return 1

    rms, camera_matrix, dist, _, _ = cv2.calibrateCamera(
        object_points, image_points, image_size, None, None
    )
    d = dist.ravel()
    print(f"\nRMS reprojection error: {rms:.3f} px (want < 0.5)")
    print("Paste into the matching CameraConfig in frc_perception/config.py:\n")
    print(f"    fx={camera_matrix[0, 0]:.2f},")
    print(f"    fy={camera_matrix[1, 1]:.2f},")
    print(f"    cx={camera_matrix[0, 2]:.2f},")
    print(f"    cy={camera_matrix[1, 2]:.2f},")
    print(f"    dist_coeffs=({d[0]:.6f}, {d[1]:.6f}, {d[2]:.6f}, {d[3]:.6f}, {d[4]:.6f}),")
    return 0


if __name__ == "__main__":
    sys.exit(main())
