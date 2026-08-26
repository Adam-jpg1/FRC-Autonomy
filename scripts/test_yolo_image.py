#!/usr/bin/env python3
"""Run a trained .pt model on an image (or folder) and show what it found.

This is a TRAINING-SIDE tool: it loads PyTorch weights, not the deployed
Hailo model. Use it to check a fresh ``best.pt`` before compiling it.
For the deployed accelerator, use instead:

    python scripts/test_hailo.py --model models/best_hailo_model --image x.jpg
    python scripts/benchmark_detector.py --images data/samples/

    python scripts/test_yolo_image.py --model models/best.pt --image sample.jpg
    python scripts/test_yolo_image.py --model models/best.pt --image data/samples/ \
        --save-dir data/predictions --no-window

Prints every box with its class name and confidence, and verifies that the
weights contain EXACTLY the classes this system routes on
(fuel, bumper_blue, bumper_red).  A mismatch is reported loudly: routing
is by name, so an unexpected class name is silently never detected at
runtime, which looks exactly like a model that cannot see.

With ``--range`` it also reports the robot-relative range and bearing each
box would produce for a configured camera, which is the quickest way to
sanity check mount measurements against a photo of a known layout.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402

from frc_perception.config import EXPECTED_CLASS_NAMES, load_config  # noqa: E402
from frc_perception.detector import compare_class_names, resolve_torch_device  # noqa: E402
from frc_perception.geometry import estimate_ground_object  # noqa: E402

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")

CLASS_COLORS = {
    "fuel": (0, 200, 255),
    "bumper_blue": (255, 100, 0),
    "bumper_red": (0, 0, 255),
}


def gather_images(target: str):
    if os.path.isdir(target):
        paths = []
        for extension in IMAGE_EXTENSIONS:
            paths.extend(glob.glob(os.path.join(target, f"*{extension}")))
            paths.extend(glob.glob(os.path.join(target, f"*{extension.upper()}")))
        return sorted(set(paths))
    matches = sorted(glob.glob(target))
    return matches or [target]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="", help=".pt weights (default: config)")
    parser.add_argument("--image", required=True, help="image file, glob, or directory")
    parser.add_argument("--conf", type=float, default=None, help="override confidence threshold")
    parser.add_argument("--save-dir", default="", help="write annotated images here")
    parser.add_argument("--no-window", action="store_true", help="do not open a window")
    parser.add_argument(
        "--range", dest="range_camera", default="",
        help="also report robot-relative range/bearing using this camera's config",
    )
    args = parser.parse_args()

    cfg = load_config()
    model_path = args.model or cfg.detector.pytorch_model_path
    if not os.path.exists(model_path):
        print(f"FAIL: no weights at {model_path}")
        return 1

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        print(f"FAIL: ultralytics is not installed ({exc})")
        return 1

    model = YOLO(model_path)
    names = {int(k): str(v) for k, v in dict(getattr(model, "names", {})).items()}
    ok, message = compare_class_names(names, EXPECTED_CLASS_NAMES)
    print(f"model   : {model_path}")
    print(f"device  : {resolve_torch_device(cfg.detector.pytorch_device)}")
    print(f"classes : {names}")
    print(f"contract: {'OK' if ok else 'MISMATCH'} -- {message}")
    if not ok:
        print(
            "\n  Routing is by class NAME. Any name not in "
            f"{list(EXPECTED_CLASS_NAMES)} is ignored at runtime, so this model\n"
            "  will appear blind for those objects. Retrain with the correct\n"
            "  names in training/data.yaml.\n"
        )

    range_camera = None
    if args.range_camera:
        try:
            range_camera = cfg.camera(args.range_camera)
        except KeyError:
            print(f"FAIL: no camera named {args.range_camera!r} in config")
            return 1

    paths = gather_images(args.image)
    if not paths:
        print(f"FAIL: no images matched {args.image!r}")
        return 1
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    confidence = args.conf if args.conf is not None else cfg.detector.conf_threshold
    total_boxes = 0
    for path in paths:
        image = cv2.imread(path)
        if image is None:
            print(f"\n{path}: unreadable")
            continue
        results = model.predict(
            image,
            conf=confidence,
            iou=cfg.detector.iou_threshold,
            imgsz=cfg.detector.imgsz,
            verbose=False,
        )
        print(f"\n{path}  ({image.shape[1]}x{image.shape[0]})")
        count = 0
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                class_index = int(box.cls.item())
                class_name = names.get(class_index, f"<index {class_index}>")
                score = float(box.conf.item())
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
                count += 1
                line = (
                    f"  {class_name:<12} conf={score:.3f}  "
                    f"box=({x1:.0f},{y1:.0f})-({x2:.0f},{y2:.0f})  "
                    f"{x2 - x1:.0f}x{y2 - y1:.0f} px"
                )
                if range_camera is not None:
                    known_width = (
                        cfg.detector.fuel_diameter_m
                        if class_name == "fuel"
                        else cfg.detector.bumper_width_m
                    )
                    estimate = estimate_ground_object(
                        (x1, y1, x2, y2), range_camera, cfg.ranging, known_width
                    )
                    if estimate.valid:
                        line += (
                            f"  ->  range={estimate.range_m:.2f} m "
                            f"bearing={estimate.bearing_rad * 57.2958:+.1f} deg "
                            f"(x={estimate.x_robot_m:.2f}, y={estimate.y_robot_m:.2f})"
                        )
                    else:
                        line += f"  ->  NO RANGE ({estimate.rejection.value})"
                print(line)

                color = CLASS_COLORS.get(class_name, (200, 200, 200))
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(
                    image, f"{class_name} {score:.2f}", (int(x1), max(int(y1) - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA,
                )
        if count == 0:
            print("  (no detections above the confidence threshold)")
        total_boxes += count

        if args.save_dir:
            out = os.path.join(args.save_dir, os.path.basename(path))
            cv2.imwrite(out, image)
            print(f"  saved {out}")
        if not args.no_window:
            cv2.imshow("prediction", image)
            if (cv2.waitKey(0) & 0xFF) in (ord("q"), 27):
                break

    if not args.no_window:
        cv2.destroyAllWindows()
    print(f"\n{len(paths)} image(s), {total_boxes} detection(s) total")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
