#!/usr/bin/env python3
"""End-to-end YOLO training for the FRC 2026 model -- built for Google Colab.

HOW TO USE THIS IN COLAB
------------------------
Every step below is a function, and every function is safe to run on its
own in a cell.  Paste this file into Colab (or ``!git clone`` the repo and
``%run training/train_yolo_colab.py``) and then call the steps in order,
or call :func:`run_all` to do the whole thing.

    #  Cell 1
    !pip install -q ultralytics
    #  Cell 2  (optional, if the dataset lives in Drive)
    from google.colab import drive; drive.mount('/content/drive')
    #  Cell 3
    %run training/train_yolo_colab.py
    #  Cell 4
    run_all(dataset_zip='/content/drive/MyDrive/frc2026_dataset.zip')

RUNTIME
-------
Set Runtime -> Change runtime type -> GPU before training. On CPU this
takes hours instead of minutes; the script says which one you are on.

THE OUTPUT
----------
``best.pt`` is what the Jetson loads. It must contain exactly the classes
``fuel``, ``bumper_blue``, ``bumper_red`` -- :func:`verify_model_classes`
checks that before you waste a trip to the robot.
"""

from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
import sys
import zipfile
from typing import Dict, List, Optional, Tuple

CLASS_NAMES: Tuple[str, ...] = ("fuel", "bumper_blue", "bumper_red")
SPLITS = ("train", "val", "test")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

DEFAULT_DATASET_ROOT = "/content/dataset"
DEFAULT_PROJECT = "/content/runs"
DEFAULT_RUN_NAME = "frc2026"


# ---------------------------------------------------------------------------
# 1. Install
# ---------------------------------------------------------------------------


def install_ultralytics(version: str = "") -> None:
    """Install Ultralytics. Colab already has torch with CUDA, so leave it alone."""
    package = f"ultralytics=={version}" if version else "ultralytics"
    print(f"installing {package} ...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", package], check=True)
    import ultralytics

    print(f"ultralytics {ultralytics.__version__}")
    report_runtime()


def report_runtime() -> bool:
    """Print the device and return True if a GPU is available."""
    try:
        import torch
    except ImportError:
        print("torch not importable")
        return False
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}  (torch {torch.__version__})")
        return True
    print(
        f"NO GPU -- training on CPU (torch {torch.__version__}).\n"
        "In Colab: Runtime -> Change runtime type -> Hardware accelerator -> GPU."
    )
    return False


# ---------------------------------------------------------------------------
# 2. Drive (optional)
# ---------------------------------------------------------------------------


def mount_drive(mount_point: str = "/content/drive") -> bool:
    """Mount Google Drive. Returns False (harmlessly) outside Colab."""
    try:
        from google.colab import drive
    except ImportError:
        print("not running in Colab; skipping Drive mount")
        return False
    drive.mount(mount_point)
    print(f"Drive mounted at {mount_point}")
    return True


# ---------------------------------------------------------------------------
# 3. Get the dataset in place
# ---------------------------------------------------------------------------


def extract_dataset(zip_path: str, destination: str = DEFAULT_DATASET_ROOT) -> str:
    """Unzip a dataset archive and find the real root inside it.

    Handles the common case where the zip contains a single top-level
    folder, so ``destination/whatever/images/train`` becomes the root
    rather than leaving you with a broken path.
    """
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"no dataset archive at {zip_path}")
    os.makedirs(destination, exist_ok=True)
    print(f"extracting {zip_path} -> {destination} ...")
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(destination)

    root = find_dataset_root(destination)
    print(f"dataset root: {root}")
    return root


def find_dataset_root(start: str) -> str:
    """Locate the directory that actually contains ``images/train``."""
    if os.path.isdir(os.path.join(start, "images", "train")):
        return start
    for entry in sorted(os.listdir(start)):
        candidate = os.path.join(start, entry)
        if os.path.isdir(candidate) and os.path.isdir(
            os.path.join(candidate, "images", "train")
        ):
            return candidate
    raise FileNotFoundError(
        f"could not find images/train under {start}. Expected layout:\n"
        "  dataset/\n"
        "  |-- images/{train,val,test}/\n"
        "  `-- labels/{train,val,test}/"
    )


def split_flat_dataset(
    images_dir: str,
    labels_dir: str,
    destination: str = DEFAULT_DATASET_ROOT,
    ratios: Tuple[float, float, float] = (0.7, 0.2, 0.1),
    seed: int = 0,
) -> str:
    """Turn a flat images+labels folder into the train/val/test layout.

    Use this only if your labeling tool exported everything in one pile.
    The split is deterministic given ``seed`` so a re-run does not shuffle
    images between train and val -- doing that silently invalidates every
    metric you have ever recorded.
    """
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"ratios must sum to 1.0, got {ratios}")
    images = sorted(
        name for name in os.listdir(images_dir)
        if name.lower().endswith(IMAGE_EXTENSIONS)
    )
    if not images:
        raise FileNotFoundError(f"no images in {images_dir}")

    random.Random(seed).shuffle(images)
    train_end = int(len(images) * ratios[0])
    val_end = train_end + int(len(images) * ratios[1])
    assignment = {
        "train": images[:train_end],
        "val": images[train_end:val_end],
        "test": images[val_end:],
    }

    for split in SPLITS:
        os.makedirs(os.path.join(destination, "images", split), exist_ok=True)
        os.makedirs(os.path.join(destination, "labels", split), exist_ok=True)

    for split, names in assignment.items():
        for name in names:
            shutil.copy2(
                os.path.join(images_dir, name),
                os.path.join(destination, "images", split, name),
            )
            label = os.path.splitext(name)[0] + ".txt"
            source = os.path.join(labels_dir, label)
            target = os.path.join(destination, "labels", split, label)
            if os.path.exists(source):
                shutil.copy2(source, target)
            else:
                # An image with no .txt is SKIPPED by YOLO. An empty .txt is a
                # background example. Make the intent explicit.
                open(target, "w", encoding="utf-8").close()
        print(f"  {split}: {len(names)} images")
    return destination


# ---------------------------------------------------------------------------
# 4. Validate the dataset before spending GPU time on it
# ---------------------------------------------------------------------------


def label_path_for(image_path: str) -> str:
    parts = image_path.split(os.sep)
    try:
        index = len(parts) - 1 - parts[::-1].index("images")
    except ValueError:
        return os.path.splitext(image_path)[0] + ".txt"
    parts[index] = "labels"
    return os.path.splitext(os.sep.join(parts))[0] + ".txt"


def validate_dataset(root: str, strict: bool = True) -> Dict[str, Dict[str, int]]:
    """Check structure, label syntax, class indices and balance.

    Raises on anything that would produce a silently useless model.
    """
    print(f"\nvalidating {root}")
    problems: List[str] = []
    summary: Dict[str, Dict[str, int]] = {}

    for split in SPLITS:
        image_directory = os.path.join(root, "images", split)
        if not os.path.isdir(image_directory):
            message = f"missing {image_directory}"
            (problems.append if split != "test" else print)(message)
            continue

        images = sorted(
            os.path.join(image_directory, name)
            for name in os.listdir(image_directory)
            if name.lower().endswith(IMAGE_EXTENSIONS)
        )
        counts = {name: 0 for name in CLASS_NAMES}
        unlabeled = 0
        backgrounds = 0

        for image_path in images:
            label_path = label_path_for(image_path)
            if not os.path.exists(label_path):
                unlabeled += 1
                continue
            with open(label_path, "r", encoding="utf-8") as handle:
                lines = [line.strip() for line in handle if line.strip()]
            if not lines:
                backgrounds += 1
            for number, line in enumerate(lines, start=1):
                parts = line.split()
                if len(parts) != 5:
                    problems.append(f"{label_path}:{number}: expected 5 fields")
                    continue
                try:
                    class_index = int(parts[0])
                    values = [float(v) for v in parts[1:]]
                except ValueError:
                    problems.append(f"{label_path}:{number}: non-numeric")
                    continue
                if not 0 <= class_index < len(CLASS_NAMES):
                    problems.append(
                        f"{label_path}:{number}: class {class_index} outside "
                        f"0..{len(CLASS_NAMES) - 1}"
                    )
                    continue
                if any(not 0.0 <= v <= 1.0 for v in values):
                    problems.append(f"{label_path}:{number}: coordinates not normalized")
                counts[CLASS_NAMES[class_index]] += 1

        summary[split] = counts
        print(
            f"  {split:<6} {len(images):>5} images  "
            + "  ".join(f"{name}={counts[name]}" for name in CLASS_NAMES)
            + f"  backgrounds={backgrounds} unlabeled={unlabeled}"
        )
        if unlabeled:
            problems.append(
                f"{split}: {unlabeled} image(s) have no label file. YOLO SKIPS those "
                "images entirely -- give true background images an EMPTY .txt."
            )

    for name in CLASS_NAMES:
        total = sum(summary.get(split, {}).get(name, 0) for split in SPLITS)
        if total == 0:
            problems.append(f"class {name!r} has no labeled instances anywhere")

    if problems:
        print("\nPROBLEMS:")
        for problem in problems[:40]:
            print(f"  - {problem}")
        if len(problems) > 40:
            print(f"  ... and {len(problems) - 40} more")
        if strict:
            raise ValueError(f"{len(problems)} dataset problem(s); fix before training")
    else:
        print("  dataset looks structurally sound")
    return summary


def write_data_yaml(root: str, path: str = "/content/data.yaml") -> str:
    """Write the data.yaml Ultralytics will read, with the exact class names."""
    lines = [
        "# generated by training/train_yolo_colab.py",
        "# Class names are a contract with frc_perception/config.py. Do not rename.",
        f"path: {os.path.abspath(root)}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "names:",
    ]
    lines.extend(f"  {index}: {name}" for index, name in enumerate(CLASS_NAMES))
    content = "\n".join(lines) + "\n"
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(content)
    print(f"\nwrote {path}:\n{content}")
    return path


# ---------------------------------------------------------------------------
# 5. Look at the data with your own eyes
# ---------------------------------------------------------------------------


def visualize_samples(root: str, split: str = "train", count: int = 9, seed: int = 0):
    """Draw labeled boxes on a grid of random images.

    Do not skip this. Half of all "the model does not work" problems are
    visible here in ten seconds: boxes off by a frame, a class swapped,
    normalized coordinates that are actually pixels.
    """
    import cv2
    import matplotlib.pyplot as plt

    image_directory = os.path.join(root, "images", split)
    images = sorted(
        os.path.join(image_directory, name)
        for name in os.listdir(image_directory)
        if name.lower().endswith(IMAGE_EXTENSIONS)
    )
    if not images:
        print(f"no images in {image_directory}")
        return
    chosen = random.Random(seed).sample(images, min(count, len(images)))

    colors = {0: (255, 200, 0), 1: (0, 100, 255), 2: (255, 0, 0)}  # RGB
    columns = 3
    rows = (len(chosen) + columns - 1) // columns
    _figure, axes = plt.subplots(rows, columns, figsize=(15, 5 * rows))
    axes = axes.ravel() if hasattr(axes, "ravel") else [axes]

    for axis, image_path in zip(axes, chosen):
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        label_path = label_path_for(image_path)
        if os.path.exists(label_path):
            with open(label_path, "r", encoding="utf-8") as handle:
                for line in handle:
                    parts = line.split()
                    if len(parts) != 5:
                        continue
                    class_index = int(parts[0])
                    cx, cy, box_width, box_height = (float(v) for v in parts[1:])
                    x1 = int((cx - box_width / 2) * width)
                    y1 = int((cy - box_height / 2) * height)
                    x2 = int((cx + box_width / 2) * width)
                    y2 = int((cy + box_height / 2) * height)
                    color = colors.get(class_index, (0, 255, 0))
                    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        image,
                        CLASS_NAMES[class_index] if class_index < len(CLASS_NAMES) else "?",
                        (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2,
                    )
        axis.imshow(image)
        axis.set_title(os.path.basename(image_path), fontsize=8)
        axis.axis("off")
    for axis in axes[len(chosen):]:
        axis.axis("off")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 6. Train
# ---------------------------------------------------------------------------


def train(
    data_yaml: str,
    model: str = "yolov8n.pt",
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    patience: int = 25,
    project: str = DEFAULT_PROJECT,
    name: str = DEFAULT_RUN_NAME,
    **overrides,
):
    """Train and return the Ultralytics results object.

    ``yolov8n`` is the right starting point for an Orin Nano: it is the
    one that comfortably runs two camera streams in real time. Move up to
    ``yolov8s`` only after you have measured FPS on the actual Jetson and
    found headroom.

    The augmentation defaults below are chosen for FRC:
    * ``fliplr`` is ON -- the field is symmetric and a mirrored ball is
      still a ball.
    * ``flipud`` is OFF -- nothing on an FRC field is upside down, and
      training on it wastes capacity.
    * ``degrees`` is small -- cameras are bolted to a robot, not handheld.
    * HSV jitter is generous -- venue lighting varies enormously, and this
      is the augmentation that actually earns its keep between events.
    """
    from ultralytics import YOLO

    if not report_runtime():
        print("WARNING: training on CPU will take hours. Continuing anyway.")

    settings = dict(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=patience,
        project=project,
        name=name,
        exist_ok=True,
        seed=0,
        deterministic=True,
        # augmentation
        fliplr=0.5,
        flipud=0.0,
        degrees=5.0,
        translate=0.1,
        scale=0.4,
        shear=2.0,
        perspective=0.0,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        mosaic=1.0,
        close_mosaic=10,
        # bookkeeping
        plots=True,
        val=True,
    )
    settings.update(overrides)

    print("\ntraining with:")
    for key in sorted(settings):
        print(f"  {key}: {settings[key]}")

    yolo = YOLO(model)
    results = yolo.train(**settings)
    print(f"\ntraining finished. run directory: {os.path.join(project, name)}")
    return results


# ---------------------------------------------------------------------------
# 7-11. Validate, inspect, predict
# ---------------------------------------------------------------------------


def find_best_weights(project: str = DEFAULT_PROJECT, name: str = DEFAULT_RUN_NAME) -> str:
    """Locate best.pt, searching the run directory and then the whole project."""
    direct = os.path.join(project, name, "weights", "best.pt")
    if os.path.exists(direct):
        return direct
    candidates = []
    for dirpath, _dirnames, filenames in os.walk(project):
        if "best.pt" in filenames:
            path = os.path.join(dirpath, "best.pt")
            candidates.append((os.path.getmtime(path), path))
    if not candidates:
        raise FileNotFoundError(f"no best.pt anywhere under {project}")
    candidates.sort()
    newest = candidates[-1][1]
    print(f"using most recent weights: {newest}")
    return newest


def validate_model(weights: str, data_yaml: str, split: str = "val") -> Dict[str, float]:
    """Run validation and print precision, recall and mAP, per class."""
    from ultralytics import YOLO

    model = YOLO(weights)
    metrics = model.val(data=data_yaml, split=split, plots=True, verbose=False)
    box = metrics.box

    print(f"\n=== metrics on the {split} split ===")
    print(f"  mAP50-95 : {float(box.map):.4f}")
    print(f"  mAP50    : {float(box.map50):.4f}")
    print(f"  precision: {float(box.mp):.4f}")
    print(f"  recall   : {float(box.mr):.4f}")
    names = {int(k): str(v) for k, v in dict(model.names).items()}
    print("\n  per class (mAP50-95):")
    for index, value in enumerate(box.maps):
        print(f"    {names.get(index, index):<14} {float(value):.4f}")

    save_dir = getattr(metrics, "save_dir", None)
    if save_dir:
        print(f"\n  confusion matrix and PR curves: {save_dir}")
        show_training_plots(str(save_dir))

    return {
        "map50_95": float(box.map),
        "map50": float(box.map50),
        "precision": float(box.mp),
        "recall": float(box.mr),
    }


def show_training_plots(run_dir: str) -> None:
    """Display the confusion matrix and curves Ultralytics wrote."""
    try:
        import matplotlib.image as mpimg
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable; open the PNGs in the run directory instead")
        return
    wanted = [
        "confusion_matrix_normalized.png",
        "confusion_matrix.png",
        "results.png",
        "PR_curve.png",
        "F1_curve.png",
    ]
    for name in wanted:
        path = os.path.join(run_dir, name)
        if not os.path.exists(path):
            continue
        plt.figure(figsize=(11, 8))
        plt.imshow(mpimg.imread(path))
        plt.title(name)
        plt.axis("off")
        plt.show()


def verify_model_classes(weights: str) -> bool:
    """The check that saves a wasted trip to the robot.

    The Jetson routes detections by class NAME. Weights whose names do not
    match are not "mostly fine" -- the mismatched classes are silently
    dropped at runtime.
    """
    from ultralytics import YOLO

    names = {int(k): str(v) for k, v in dict(YOLO(weights).names).items()}
    actual = {name for name in names.values()}
    expected = set(CLASS_NAMES)
    print(f"\nmodel classes: {names}")
    if actual == expected:
        print("OK: classes match the runtime contract exactly")
        return True
    print("MISMATCH -- this model is NOT compatible with the Jetson code:")
    if expected - actual:
        print(f"  missing   : {sorted(expected - actual)}")
    if actual - expected:
        print(f"  unexpected: {sorted(actual - expected)}")
    print("Fix training/data.yaml (and the labels) and retrain.")
    return False


def predict_samples(weights: str, root: str, split: str = "test", count: int = 6, conf: float = 0.35):
    """Run the trained model on held-out images and show the predictions."""
    import matplotlib.pyplot as plt
    from ultralytics import YOLO

    image_directory = os.path.join(root, "images", split)
    if not os.path.isdir(image_directory):
        print(f"no {split} split at {image_directory}; falling back to val")
        image_directory = os.path.join(root, "images", "val")
    images = sorted(
        os.path.join(image_directory, name)
        for name in os.listdir(image_directory)
        if name.lower().endswith(IMAGE_EXTENSIONS)
    )
    if not images:
        print("no images to predict on")
        return
    chosen = random.Random(0).sample(images, min(count, len(images)))

    model = YOLO(weights)
    columns = 3
    rows = (len(chosen) + columns - 1) // columns
    _figure, axes = plt.subplots(rows, columns, figsize=(15, 5 * rows))
    axes = axes.ravel() if hasattr(axes, "ravel") else [axes]

    for axis, image_path in zip(axes, chosen):
        result = model.predict(image_path, conf=conf, verbose=False)[0]
        axis.imshow(result.plot()[:, :, ::-1])  # BGR -> RGB
        axis.set_title(f"{os.path.basename(image_path)}  ({len(result.boxes)} boxes)", fontsize=8)
        axis.axis("off")
    for axis in axes[len(chosen):]:
        axis.axis("off")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 12-13. Export
# ---------------------------------------------------------------------------


def export_best(
    weights: str,
    destination: str = "/content/best.pt",
    drive_dir: str = "",
    download: bool = True,
) -> str:
    """Copy best.pt somewhere you will still have it after the runtime dies.

    Colab runtimes are deleted without warning. Copy to Drive, download, or
    both -- losing a trained model to a disconnected runtime the night
    before an event is a rite of passage worth skipping.
    """
    shutil.copyfile(weights, destination)
    print(f"copied {weights} -> {destination}")

    metadata = {"classes": list(CLASS_NAMES), "source": weights}
    metadata_path = os.path.splitext(destination)[0] + "_classes.json"
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    if drive_dir:
        os.makedirs(drive_dir, exist_ok=True)
        drive_path = os.path.join(drive_dir, "best.pt")
        shutil.copyfile(weights, drive_path)
        print(f"copied to Drive: {drive_path}")

    if download:
        try:
            from google.colab import files

            files.download(destination)
        except ImportError:
            print("not in Colab; skipping browser download")

    print(
        "\nOn the Jetson:\n"
        "  1. copy best.pt into  models/best.pt\n"
        "  2. python scripts/test_yolo_image.py --model models/best.pt --image <a photo>\n"
        "  3. python run.py --check      (confirms the class names match)"
    )
    return destination


# ---------------------------------------------------------------------------
# Everything, in order
# ---------------------------------------------------------------------------


def run_all(
    dataset_zip: str = "",
    dataset_root: str = DEFAULT_DATASET_ROOT,
    epochs: int = 100,
    model: str = "yolov8n.pt",
    imgsz: int = 640,
    batch: int = 16,
    drive_dir: str = "",
    visualize: bool = True,
) -> Optional[str]:
    """Install, validate, train, validate, export."""
    install_ultralytics()

    root = extract_dataset(dataset_zip, dataset_root) if dataset_zip else find_dataset_root(dataset_root)
    validate_dataset(root, strict=True)
    data_yaml = write_data_yaml(root)

    if visualize:
        try:
            visualize_samples(root, "train")
        except Exception as exc:  # noqa: BLE001 - plotting must not stop training
            print(f"(sample visualization skipped: {exc})")

    train(data_yaml, model=model, epochs=epochs, imgsz=imgsz, batch=batch)

    weights = find_best_weights()
    validate_model(weights, data_yaml)
    if not verify_model_classes(weights):
        print("\nSTOP: fix the class names before deploying this model.")
        return None
    try:
        predict_samples(weights, root)
    except Exception as exc:  # noqa: BLE001
        print(f"(sample predictions skipped: {exc})")
    return export_best(weights, drive_dir=drive_dir)


if __name__ == "__main__":
    print(__doc__)
    print(
        "This file is a library of Colab steps. Import it or %run it, then call\n"
        "run_all(dataset_zip=...) -- or the individual steps in order."
    )
