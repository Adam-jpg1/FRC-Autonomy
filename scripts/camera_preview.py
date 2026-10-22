#!/usr/bin/env python3
"""Live camera preview with FPS, resolution and frame age. No YOLO involved.

    python scripts/camera_preview.py --list                   # find device paths
    python scripts/camera_preview.py --camera cam0            # by config name
    python scripts/camera_preview.py --camera /dev/v4l/by-id/usb-Arducam-...
    python scripts/camera_preview.py --camera 0 --no-window   # headless stats

This is bring-up stage 3/4: prove each camera streams, at the resolution
and rate you asked for, before anything else touches it.

Keys in the window:  q / ESC quit,  s save a snapshot to data/snapshots/
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402

from frc_perception.camera import ThreadedCamera, list_video_devices  # noqa: E402
from frc_perception.config import CameraConfig, load_config  # noqa: E402
from frc_perception.logging_setup import setup_logging  # noqa: E402


def print_devices() -> int:
    """List the persistent camera paths available on this machine."""
    devices = list_video_devices()
    if not devices:
        print(
            "No /dev/v4l/by-id or /dev/v4l/by-path entries.\n"
            "On the Jetson that means no USB camera is attached (or v4l-utils\n"
            "is not installed: sudo apt install v4l-utils)."
        )
        return 1
    print("Persistent camera paths (use these for CameraConfig.device_path):\n")
    for entry in devices:
        print(f"  [{entry['kind']:<7}] {entry['path']}")
        print(f"            -> {entry['node']}")
    print(
        "\nEach USB camera usually claims TWO /dev/video nodes; the second is a\n"
        "metadata node with no image formats. Confirm which is which with:\n"
        "  v4l2-ctl -d <path> --list-formats-ext\n"
        "\nIf both cameras are identical modules with the same USB serial, by-id\n"
        "will not tell them apart -- use by-path (keyed on the physical port)\n"
        "and always plug each camera into the same port."
    )
    return 0


def resolve_camera(cfg, requested: str) -> CameraConfig:
    """Accept a configured camera name, a device path, or a raw index."""
    for camera in cfg.cameras:
        if camera.name == requested:
            return camera
    if requested.startswith("/dev/"):
        return CameraConfig(name=os.path.basename(requested), device_path=requested)
    try:
        index = int(requested)
    except ValueError:
        names = [c.name for c in cfg.cameras]
        raise SystemExit(
            f"no camera named {requested!r}; configured: {names}. "
            "Run with --list to see the device paths on this machine."
        )
    return CameraConfig(name=f"video{index}", device_index=index)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--camera", help="config name, /dev path, or video index")
    parser.add_argument("--list", action="store_true", help="list persistent device paths and exit")
    parser.add_argument("--width", type=int, help="override capture width")
    parser.add_argument("--height", type=int, help="override capture height")
    parser.add_argument("--fps", type=int, help="override requested frame rate")
    parser.add_argument("--no-window", action="store_true", help="print stats only")
    parser.add_argument("--seconds", type=float, default=0.0, help="exit after N seconds")
    args = parser.parse_args()

    if args.list:
        return print_devices()
    if not args.camera:
        raise SystemExit("give --camera NAME|/dev/path|INDEX, or --list")

    cfg = load_config()
    setup_logging(cfg.logging)
    camera_cfg = resolve_camera(cfg, args.camera)
    if args.width:
        camera_cfg.width = args.width
    if args.height:
        camera_cfg.height = args.height
    if args.fps:
        camera_cfg.fps = args.fps

    print(f"opening {camera_cfg.name} (device {camera_cfg.source}) ...")
    camera = ThreadedCamera(camera_cfg, cfg.frame_freshness_s)
    camera.start()

    snapshot_dir = os.path.join("data", "snapshots")
    started = time.monotonic()
    last_print = 0.0
    exit_code = 0
    try:
        while True:
            now = time.monotonic()
            if args.seconds and (now - started) > args.seconds:
                break

            frame = camera.read_latest()
            health = camera.health()
            age_ms = health.last_frame_age_s * 1000.0

            if now - last_print >= 1.0:
                last_print = now
                shape = "no frame yet" if frame is None else (
                    f"{frame.image.shape[1]}x{frame.image.shape[0]}"
                )
                print(
                    f"{camera_cfg.name}: {shape} fps={health.fps:5.1f} "
                    f"age={age_ms:7.1f}ms frames={health.frames_captured} "
                    f"dropped={health.frames_dropped} reconnects={health.reconnect_count} "
                    f"{'HEALTHY' if health.healthy else 'UNHEALTHY ' + health.last_error}"
                )

            if args.no_window:
                time.sleep(0.05)
                continue

            if frame is None:
                time.sleep(0.05)
                continue

            display = frame.image.copy()
            lines = [
                f"{camera_cfg.name}  {display.shape[1]}x{display.shape[0]}",
                f"fps {health.fps:.1f}   age {age_ms:.0f} ms   seq {frame.seq}",
                "STALE FRAME" if age_ms / 1000.0 > cfg.frame_freshness_s else "fresh",
            ]
            for index, text in enumerate(lines):
                cv2.putText(
                    display, text, (10, 30 + index * 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 0), 4, cv2.LINE_AA,
                )
                cv2.putText(
                    display, text, (10, 30 + index * 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2, cv2.LINE_AA,
                )
            cv2.imshow("camera preview", display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                os.makedirs(snapshot_dir, exist_ok=True)
                path = os.path.join(
                    snapshot_dir, f"{camera_cfg.name}_{int(time.time())}.png"
                )
                cv2.imwrite(path, frame.image)
                print(f"saved {path}")
    except KeyboardInterrupt:
        pass
    finally:
        camera.stop()
        if not args.no_window:
            cv2.destroyAllWindows()

    final = camera.health()
    if final.frames_captured == 0:
        print("FAIL: no frames were ever captured -- check the device index and cabling")
        exit_code = 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
