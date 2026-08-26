#!/usr/bin/env python3
"""Validate a YOLO dataset and/or a trained model, before or after training.

    # dataset structure, class ids, label sanity, empty/duplicate checks
    python training/validate_yolo.py --dataset ./dataset

    # trained weights: class names, metrics on the val split
    python training/validate_yolo.py --model models/best.pt --data training/data.yaml

    # both
    python training/validate_yolo.py --dataset ./dataset --model models/best.pt \
        --data training/data.yaml

Runs standalone (no Ultralytics needed) for the dataset checks, which are
the ones that catch the mistakes that actually happen: a label file with
class index 3 when only 0-2 exist, boxes outside [0,1], images without
labels, and labels without images.
"""

from __future__ import annotations

import argparse
import os
from collections import Counter
from typing import Dict, List, Tuple

EXPECTED_CLASSES = ("fuel", "bumper_blue", "bumper_red")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
SPLITS = ("train", "val", "test")


class Findings:
    def __init__(self) -> None:
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.notes: List[str] = []

    def error(self, message: str) -> None:
        self.errors.append(message)

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    def note(self, message: str) -> None:
        self.notes.append(message)

    def report(self) -> int:
        for message in self.notes:
            print(f"  .  {message}")
        for message in self.warnings:
            print(f"  !  WARNING: {message}")
        for message in self.errors:
            print(f"  X  ERROR:   {message}")
        print(
            f"\n  {len(self.errors)} error(s), {len(self.warnings)} warning(s)"
        )
        return 1 if self.errors else 0


def list_images(directory: str) -> List[str]:
    if not os.path.isdir(directory):
        return []
    return sorted(
        os.path.join(directory, name)
        for name in os.listdir(directory)
        if name.lower().endswith(IMAGE_EXTENSIONS)
    )


def label_path_for(image_path: str) -> str:
    parts = image_path.split(os.sep)
    try:
        index = len(parts) - 1 - parts[::-1].index("images")
    except ValueError:
        return os.path.splitext(image_path)[0] + ".txt"
    parts[index] = "labels"
    return os.path.splitext(os.sep.join(parts))[0] + ".txt"


def parse_label_file(path: str, findings: Findings) -> List[Tuple[int, float, float, float, float]]:
    boxes = []
    with open(path, "r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                findings.error(
                    f"{path}:{number}: expected 5 fields "
                    f"(class cx cy w h), got {len(parts)}"
                )
                continue
            try:
                class_index = int(parts[0])
                cx, cy, width, height = (float(v) for v in parts[1:])
            except ValueError:
                findings.error(f"{path}:{number}: non-numeric field")
                continue
            if not 0 <= class_index < len(EXPECTED_CLASSES):
                findings.error(
                    f"{path}:{number}: class index {class_index} outside "
                    f"0..{len(EXPECTED_CLASSES) - 1}"
                )
            for name, value in (("cx", cx), ("cy", cy), ("w", width), ("h", height)):
                if not 0.0 <= value <= 1.0:
                    findings.error(
                        f"{path}:{number}: {name}={value} is not normalized to [0,1]"
                    )
            if width <= 0 or height <= 0:
                findings.error(f"{path}:{number}: zero-area box")
            elif width < 0.004 or height < 0.004:
                findings.warn(
                    f"{path}:{number}: very small box ({width:.4f}x{height:.4f}); "
                    "at 1280x800 that is only a few pixels"
                )
            boxes.append((class_index, cx, cy, width, height))
    return boxes


def check_dataset(root: str) -> int:
    print(f"\n=== DATASET: {root} ===")
    findings = Findings()

    if not os.path.isdir(root):
        print(f"  X  ERROR:   no such directory: {root}")
        return 1

    for split in SPLITS:
        for kind in ("images", "labels"):
            directory = os.path.join(root, kind, split)
            if not os.path.isdir(directory):
                message = f"missing {kind}/{split}/"
                if split == "test":
                    findings.warn(message + " (optional, but recommended)")
                else:
                    findings.error(message)

    totals: Dict[str, Counter] = {}
    grand_total = Counter()
    for split in SPLITS:
        image_directory = os.path.join(root, "images", split)
        images = list_images(image_directory)
        if not images:
            if split != "test":
                findings.error(f"{split}: no images found in {image_directory}")
            continue

        counts = Counter()
        unlabeled = 0
        background = 0
        stems = Counter(os.path.splitext(os.path.basename(p))[0] for p in images)
        duplicates = [stem for stem, count in stems.items() if count > 1]
        if duplicates:
            findings.warn(
                f"{split}: {len(duplicates)} image stem(s) appear more than once "
                f"(different extensions), e.g. {duplicates[:3]}"
            )

        for image_path in images:
            label_path = label_path_for(image_path)
            if not os.path.exists(label_path):
                unlabeled += 1
                continue
            boxes = parse_label_file(label_path, findings)
            if not boxes:
                background += 1
            for class_index, *_rest in boxes:
                if 0 <= class_index < len(EXPECTED_CLASSES):
                    counts[EXPECTED_CLASSES[class_index]] += 1

        totals[split] = counts
        grand_total.update(counts)

        findings.note(
            f"{split}: {len(images)} images, {sum(counts.values())} boxes, "
            f"{background} background (empty-label) images, {unlabeled} UNLABELED"
        )
        if unlabeled:
            findings.error(
                f"{split}: {unlabeled} image(s) have no label file at all. An image "
                "with no .txt is not a negative example to YOLO -- it is skipped. "
                "Create an EMPTY .txt for true background images."
            )
        if background and background / max(len(images), 1) > 0.5:
            findings.warn(
                f"{split}: more than half the images are backgrounds; that will "
                "suppress recall"
            )

        # Orphan labels
        label_directory = os.path.join(root, "labels", split)
        if os.path.isdir(label_directory):
            image_stems = {os.path.splitext(os.path.basename(p))[0] for p in images}
            orphans = [
                name for name in os.listdir(label_directory)
                if name.endswith(".txt")
                and os.path.splitext(name)[0] not in image_stems
            ]
            if orphans:
                findings.warn(
                    f"{split}: {len(orphans)} label file(s) with no matching image, "
                    f"e.g. {orphans[:3]}"
                )

    print("\n  class balance:")
    header = "    {:<14}".format("split") + "".join(f"{c:>14}" for c in EXPECTED_CLASSES)
    print(header)
    for split in SPLITS:
        counts = totals.get(split)
        if counts is None:
            continue
        row = "    {:<14}".format(split)
        row += "".join(f"{counts.get(name, 0):>14}" for name in EXPECTED_CLASSES)
        print(row)
    row = "    {:<14}".format("TOTAL")
    row += "".join(f"{grand_total.get(name, 0):>14}" for name in EXPECTED_CLASSES)
    print(row)

    for name in EXPECTED_CLASSES:
        if grand_total.get(name, 0) == 0:
            findings.error(
                f"class {name!r} has ZERO labeled instances; the model cannot learn it"
            )
    if grand_total:
        most = max(grand_total.values())
        fewest = min(grand_total.get(name, 0) for name in EXPECTED_CLASSES)
        if fewest and most / fewest > 10:
            findings.warn(
                f"class imbalance is {most / fewest:.0f}:1; the rare class will be "
                "under-detected. Label more of it or weight the sampling."
            )

    print()
    return findings.report()


def check_model(model_path: str, data_yaml: str, run_val: bool) -> int:
    print(f"\n=== MODEL: {model_path} ===")
    if not os.path.exists(model_path):
        print(f"  X  ERROR:   no weights at {model_path}")
        return 1
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        print(f"  !  WARNING: ultralytics not installed ({exc}); skipping model checks")
        return 0

    model = YOLO(model_path)
    names = {int(k): str(v) for k, v in dict(getattr(model, "names", {})).items()}
    print(f"  classes: {names}")

    actual = {name for name in names.values()}
    expected = set(EXPECTED_CLASSES)
    status = 0
    if actual != expected:
        print(f"  X  ERROR:   class mismatch. expected {sorted(expected)}, got {sorted(actual)}")
        print("              The runtime routes by NAME; unmatched classes are ignored.")
        status = 1
    else:
        print("  .  class names match the runtime contract")

    if run_val:
        if not os.path.exists(data_yaml):
            print(f"  !  WARNING: no data.yaml at {data_yaml}; skipping metrics")
            return status
        print("\n  running validation ...")
        metrics = model.val(data=data_yaml, verbose=False)
        box = getattr(metrics, "box", None)
        if box is not None:
            print(f"    mAP50-95 : {float(box.map):.4f}")
            print(f"    mAP50    : {float(box.map50):.4f}")
            print(f"    precision: {float(box.mp):.4f}")
            print(f"    recall   : {float(box.mr):.4f}")
            per_class = getattr(box, "maps", None)
            if per_class is not None:
                print("    per-class mAP50-95:")
                for index, value in enumerate(per_class):
                    print(f"      {names.get(index, index):<14} {float(value):.4f}")
        save_dir = getattr(metrics, "save_dir", None)
        if save_dir:
            print(f"\n  plots (confusion matrix, PR curves): {save_dir}")
    return status


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default="", help="dataset root to validate")
    parser.add_argument("--model", default="", help="weights to validate")
    parser.add_argument("--data", default="training/data.yaml", help="data.yaml for --model")
    parser.add_argument("--no-val", action="store_true", help="skip running metrics")
    args = parser.parse_args()

    if not args.dataset and not args.model:
        parser.error("give --dataset, --model, or both")

    status = 0
    if args.dataset:
        status |= check_dataset(args.dataset)
    if args.model:
        status |= check_model(args.model, args.data, run_val=not args.no_val)

    print("\n" + ("FAILED" if status else "OK"))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
