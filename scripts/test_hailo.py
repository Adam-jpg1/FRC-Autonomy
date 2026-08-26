#!/usr/bin/env python3
"""Verify the Hailo-8 accelerator end to end, on the Raspberry Pi.

    python scripts/test_hailo.py --model models/best_hailo_model --image sample.jpg
    python scripts/test_hailo.py --artifact-only     # no hardware needed

Seven stages, each reported separately so a failure tells you WHICH part
is wrong:

    1. HailoRT runtime importable          (hailo_platform)
    2. Hailo-8 device present and awake    (hailortcli fw-control identify)
    3. model artifact on disk              (.hef + metadata.yaml)
    4. model loads onto the accelerator    (Ultralytics AutoBackend)
    5. class metadata matches the contract (fuel/bumper_blue/bumper_red)
    6. inference on a real image
    7. conversion into our Detection type  (the hardware boundary)

Stages 1, 2, 4 and 6 need the physical accelerator. Stages 3, 5 and 7 do
not, so ``--artifact-only`` validates a model on a laptop before it is ever
copied to the robot.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from frc_perception.config import EXPECTED_CLASS_NAMES, load_config  # noqa: E402
from frc_perception.detector import (  # noqa: E402
    HailoDetectorBackend,
    compare_class_names,
    find_hef,
    names_from_metadata,
    read_hailo_metadata,
)
from frc_perception.types import Frame  # noqa: E402

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"


class Stages:
    def __init__(self) -> None:
        self.results = []

    def add(self, status: str, name: str, detail: str = "") -> None:
        self.results.append((status, name, detail))
        marker = {PASS: "  OK ", FAIL: " FAIL", SKIP: " SKIP"}[status]
        print(f"[{marker}] {name}")
        for line in detail.splitlines():
            if line.strip():
                print(f"         {line}")

    @property
    def failed(self) -> int:
        return sum(1 for status, _n, _d in self.results if status == FAIL)

    @property
    def skipped(self) -> int:
        return sum(1 for status, _n, _d in self.results if status == SKIP)


def stage_runtime(stages: Stages) -> bool:
    try:
        import hailo_platform

        version = getattr(hailo_platform, "__version__", "unknown version")
        stages.add(PASS, "1. HailoRT runtime", f"hailo_platform ({version})")
        return True
    except ImportError as exc:
        stages.add(
            FAIL, "1. HailoRT runtime",
            f"{exc}\nOn Raspberry Pi OS:  sudo apt install hailo-all  &&  sudo reboot",
        )
        return False


def stage_device(stages: Stages) -> bool:
    executable = shutil.which("hailortcli")
    if not executable:
        stages.add(
            FAIL, "2. Hailo device",
            "hailortcli not on PATH. Install hailo-all and reboot.",
        )
        return False
    try:
        completed = subprocess.run(
            [executable, "fw-control", "identify"],
            capture_output=True, text=True, timeout=30, check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        stages.add(FAIL, "2. Hailo device", f"could not run hailortcli: {exc}")
        return False
    if completed.returncode != 0:
        stages.add(
            FAIL, "2. Hailo device",
            (completed.stderr or completed.stdout).strip()[:400]
            + "\nCheck that the AI HAT+ is seated and that the PCIe driver loaded "
            "(dmesg | grep -i hailo).",
        )
        return False
    stages.add(PASS, "2. Hailo device", completed.stdout.strip())
    return True


def stage_artifact(stages: Stages, model_dir: str) -> dict:
    metadata, error = read_hailo_metadata(model_dir)
    if error:
        stages.add(FAIL, "3. Model artifact", error)
        return {}
    hef = find_hef(model_dir)
    if hef is None:
        stages.add(FAIL, "3. Model artifact", f"no .hef inside {model_dir}")
        return {}
    entries = "\n".join(
        f"{name:<26} {os.path.getsize(os.path.join(model_dir, name)) / 1e6:8.2f} MB"
        for name in sorted(os.listdir(model_dir))
        if os.path.isfile(os.path.join(model_dir, name))
    )
    stages.add(
        PASS, "3. Model artifact",
        f"{model_dir}\n{entries}\nhailo_arch: {metadata.get('hailo_arch', '(not recorded)')}"
        f"\nimgsz: {metadata.get('imgsz', '(not recorded)')}"
        f"\non-chip NMS: {metadata.get('nms', '(not recorded)')}",
    )
    return metadata


def stage_classes(stages: Stages, metadata: dict) -> bool:
    names = names_from_metadata(metadata)
    if not names:
        stages.add(
            FAIL, "5. Class metadata",
            "metadata.yaml carries no class names. Did you copy only the .hef? "
            "The whole export directory is the artifact.",
        )
        return False
    ok, report = compare_class_names(names, EXPECTED_CLASS_NAMES)
    stages.add(
        PASS if ok else FAIL, "5. Class metadata",
        f"{names}\n{report}"
        + ("" if ok else "\nRouting is by NAME; unmatched classes are dropped at runtime."),
    )
    return ok


def main() -> int:
    cfg = load_config()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=cfg.detector.hailo_model_path,
                        help="Hailo export DIRECTORY (not just the .hef)")
    parser.add_argument("--image", default="", help="image to run inference on")
    parser.add_argument("--artifact-only", action="store_true",
                        help="check the model files and class metadata only")
    parser.add_argument("--runs", type=int, default=10,
                        help="inference repetitions for a rough latency figure")
    args = parser.parse_args()

    print("=" * 72)
    print("Hailo-8 verification")
    print(f"model: {args.model}")
    print("=" * 72)

    stages = Stages()

    have_runtime = have_device = False
    if args.artifact_only:
        stages.add(SKIP, "1. HailoRT runtime", "--artifact-only")
        stages.add(SKIP, "2. Hailo device", "--artifact-only")
    else:
        have_runtime = stage_runtime(stages)
        have_device = stage_device(stages)

    metadata = stage_artifact(stages, args.model)
    if not metadata:
        summarize(stages)
        return 1

    cfg.detector.hailo_model_path = args.model
    backend = HailoDetectorBackend(cfg.detector)

    if args.artifact_only or not (have_runtime and have_device):
        stages.add(SKIP, "4. Model load", "no accelerator available")
        stage_classes(stages, metadata)
        stages.add(SKIP, "6. Inference", "no accelerator available")
        stages.add(SKIP, "7. Detection conversion", "no accelerator available")
        summarize(stages)
        return 1 if stages.failed else 0

    if backend.load():
        health = backend.health()
        stages.add(PASS, "4. Model load", f"{health.device_report}; {health.class_report}")
    else:
        stages.add(FAIL, "4. Model load", backend.load_error)
        stage_classes(stages, metadata)
        summarize(stages)
        return 1

    stage_classes(stages, metadata)

    if not args.image:
        stages.add(SKIP, "6. Inference", "no --image supplied")
        stages.add(SKIP, "7. Detection conversion", "no --image supplied")
        summarize(stages)
        return 1 if stages.failed else 0

    if not os.path.exists(args.image):
        stages.add(FAIL, "6. Inference", f"no such image: {args.image}")
        summarize(stages)
        return 1

    import cv2

    image = cv2.imread(args.image)
    if image is None:
        stages.add(FAIL, "6. Inference", f"could not decode {args.image}")
        summarize(stages)
        return 1

    latencies = []
    detections = []
    try:
        for index in range(max(1, args.runs)):
            frame = Frame(camera_id="test", seq=index, timestamp=time.monotonic(), image=image)
            started = time.monotonic()
            detections = backend.predict(frame)
            latencies.append((time.monotonic() - started) * 1000.0)
    except Exception as exc:  # noqa: BLE001
        stages.add(FAIL, "6. Inference", str(exc))
        summarize(stages)
        return 1

    latencies.sort()
    warm = latencies[1:] or latencies  # the first call includes lazy setup
    stages.add(
        PASS, "6. Inference",
        f"{len(latencies)} run(s) on {args.image} ({image.shape[1]}x{image.shape[0]})\n"
        f"first: {latencies[0]:.1f} ms   median: {warm[len(warm) // 2]:.1f} ms   "
        f"max: {warm[-1]:.1f} ms\n"
        "(measured on THIS machine right now -- no performance figure is assumed "
        "anywhere in this project)",
    )

    lines = [f"{len(detections)} detection(s), converted to frc_perception.types.Detection:"]
    for detection in detections[:20]:
        lines.append(
            f"  {detection.class_name:<12} conf={detection.confidence:.3f} "
            f"box=({detection.x1:.0f},{detection.y1:.0f})-({detection.x2:.0f},{detection.y2:.0f}) "
            f"camera_id={detection.camera_id} seq={detection.frame_sequence}"
        )
    unexpected = {d.class_name for d in detections} - set(EXPECTED_CLASS_NAMES)
    status = FAIL if unexpected else PASS
    if unexpected:
        lines.append(f"  UNEXPECTED class names leaked through: {sorted(unexpected)}")
    stages.add(status, "7. Detection conversion", "\n".join(lines))

    backend.close()
    summarize(stages)
    return 1 if stages.failed else 0


def summarize(stages: Stages) -> None:
    print("\n" + "=" * 72)
    if stages.failed:
        print(f"RESULT: {stages.failed} stage(s) FAILED, {stages.skipped} skipped")
    elif stages.skipped:
        print(
            f"RESULT: no failures, but {stages.skipped} stage(s) were SKIPPED. "
            "This is NOT a verified accelerator -- run it again on the Pi with "
            "the AI HAT+ attached and an --image."
        )
    else:
        print("RESULT: all stages passed on this machine.")
    print("=" * 72)


if __name__ == "__main__":
    sys.exit(main())
