#!/usr/bin/env python3
"""Benchmark the deployed detector on images or video.  Measure, never assume.

    python scripts/benchmark_detector.py --images data/samples/
    python scripts/benchmark_detector.py --video data/clips/practice.mp4 --frames 300
    python scripts/benchmark_detector.py --backend pytorch --images data/samples/

Reports frames processed, mean/median/p95/p99 latency, effective FPS, and
how many results would have been discarded as stale at the configured
``max_detection_result_age_s``.

NO EXPECTED NUMBER IS PRINTED ANYWHERE.  The only frame rate that means
anything is the one this produces on the Raspberry Pi 5 with the AI HAT+
attached, at the resolution and image size you actually run.  Run it warm,
run it with the case closed, and run it again after ten minutes -- the Pi
5 throttles, and a number measured in the first thirty seconds is not the
number you get in the third match of the day.
"""

from __future__ import annotations

import argparse
import glob
import os
import statistics
import sys
import time
from typing import List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from frc_perception.config import load_config  # noqa: E402
from frc_perception.detector import build_detector_backend  # noqa: E402
from frc_perception.types import Frame  # noqa: E402

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def gather_images(target: str) -> List[str]:
    if os.path.isdir(target):
        paths: List[str] = []
        for extension in IMAGE_EXTENSIONS:
            paths.extend(glob.glob(os.path.join(target, f"*{extension}")))
            paths.extend(glob.glob(os.path.join(target, f"*{extension.upper()}")))
        return sorted(set(paths))
    return sorted(glob.glob(target)) or [target]


def percentile(values: List[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return ordered[index]


def report(
    latencies: List[float],
    detections_total: int,
    stale: int,
    wall_time: float,
    warmup: int,
    limit_ms: float,
) -> None:
    if not latencies:
        print("no frames were processed")
        return
    warm = latencies[warmup:] or latencies

    print("\n" + "=" * 72)
    print("DETECTOR BENCHMARK -- measured on this machine, just now")
    print("=" * 72)
    print(f"  frames processed     : {len(latencies)}  ({warmup} warm-up excluded below)")
    print(f"  detections total     : {detections_total}")
    print(f"  wall time            : {wall_time:.2f} s")
    print()
    print(f"  mean latency         : {statistics.fmean(warm):8.2f} ms")
    print(f"  median latency       : {statistics.median(warm):8.2f} ms")
    print(f"  p95 latency          : {percentile(warm, 0.95):8.2f} ms")
    print(f"  p99 latency          : {percentile(warm, 0.99):8.2f} ms")
    print(f"  min / max            : {min(warm):8.2f} / {max(warm):.2f} ms")
    print()
    effective = 1000.0 / statistics.fmean(warm) if statistics.fmean(warm) > 0 else 0.0
    print(f"  effective FPS        : {effective:8.2f}  (one camera, back to back)")
    print(f"  two cameras alternating: {effective / 2:6.2f} FPS each")
    print(f"  stale results        : {stale} (older than {limit_ms:.0f} ms end to end)")
    print()
    print(
        "  Interpretation: the main loop runs at loop_hz. If effective FPS is\n"
        "  below 2 x loop_hz, the scheduler will skip frames -- which is correct\n"
        "  behavior, not a bug, but tells you the accelerator is the bottleneck.\n"
        "  Lower detector.imgsz or max_inferences_per_cycle if the per-camera\n"
        "  rate is too low for how fast the robot drives."
    )
    print("=" * 72)


def main() -> int:
    cfg = load_config()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--images", default="", help="image file, glob, or directory")
    parser.add_argument("--video", default="", help="video file")
    parser.add_argument("--model", default="", help="override the model path")
    parser.add_argument("--backend", choices=("hailo", "pytorch"), default="",
                        help="override the detector backend")
    parser.add_argument("--frames", type=int, default=200, help="frames to process")
    parser.add_argument("--warmup", type=int, default=5,
                        help="initial frames excluded from the statistics")
    parser.add_argument("--imgsz", type=int, default=0, help="override inference size")
    args = parser.parse_args()

    if not args.images and not args.video:
        parser.error("give --images or --video")

    if args.backend:
        cfg.detector.backend = args.backend
    if args.model:
        if cfg.detector.backend == "hailo":
            cfg.detector.hailo_model_path = args.model
        else:
            cfg.detector.pytorch_model_path = args.model
    if args.imgsz:
        cfg.detector.imgsz = args.imgsz

    backend = build_detector_backend(cfg.detector)
    print(f"backend : {cfg.detector.backend}")
    print(f"model   : {cfg.detector.model_path}")
    print(f"imgsz   : {cfg.detector.imgsz}")
    if not backend.load():
        print(f"\nFAIL: detector did not load: {backend.load_error}")
        return 1
    print(f"device  : {backend.health().device_report}")
    print(f"classes : {backend.class_report}\n")

    import cv2

    frames_source: List = []
    capture = None
    if args.images:
        paths = gather_images(args.images)
        if not paths:
            print(f"FAIL: no images matched {args.images!r}")
            return 1
        frames_source = [cv2.imread(p) for p in paths]
        frames_source = [f for f in frames_source if f is not None]
        if not frames_source:
            print("FAIL: no readable images")
            return 1
        print(f"source  : {len(frames_source)} image(s), cycled to {args.frames} frames")
    else:
        capture = cv2.VideoCapture(args.video)
        if not capture.isOpened():
            print(f"FAIL: cannot open {args.video}")
            return 1
        print(f"source  : {args.video}")

    latencies: List[float] = []
    detections_total = 0
    stale = 0
    limit = cfg.max_detection_result_age_s
    started_wall = time.monotonic()

    try:
        for index in range(args.frames):
            if capture is not None:
                ok, image = capture.read()
                if not ok:
                    break
            else:
                image = frames_source[index % len(frames_source)]

            capture_time = time.monotonic()
            frame = Frame(camera_id="bench", seq=index, timestamp=capture_time, image=image)
            started = time.monotonic()
            try:
                detections = backend.predict(frame)
            except Exception as exc:  # noqa: BLE001
                print(f"inference failed on frame {index}: {exc}")
                break
            finished = time.monotonic()

            latencies.append((finished - started) * 1000.0)
            detections_total += len(detections)
            if finished - capture_time > limit:
                stale += 1

            if index and index % 50 == 0:
                print(f"  ... {index} frames, last {latencies[-1]:.1f} ms")
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        if capture is not None:
            capture.release()
        backend.close()

    report(
        latencies, detections_total, stale,
        time.monotonic() - started_wall, args.warmup, limit * 1000.0,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
