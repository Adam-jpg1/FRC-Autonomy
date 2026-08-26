#!/usr/bin/env python3
"""Compile a trained ``best.pt`` into a Hailo-8 model directory.

WHERE THIS RUNS
---------------
**Not on the Raspberry Pi.**  Compilation needs the Hailo Dataflow
Compiler, which is x86_64 Linux only -- Ultralytics enforces this itself::

    assert LINUX and not ARM64, "Hailo export is only supported on Linux x86_64."
    (ultralytics/engine/exporter.py, verified against 8.4.129)

The deployment flow is therefore::

    YOLO dataset
         |
    Google Colab / training machine  ->  best.pt
         |
    validation (training/validate_yolo.py)
         |
    Linux x86_64 + Hailo Dataflow Compiler  ->  THIS SCRIPT
         |
    best_hailo_model/  (best.hef + metadata.yaml + nms_config.json)
         |
    copy the WHOLE DIRECTORY to the Pi
         |
    HailoRT inference on the AI HAT+

WHAT YOU NEED ON THE COMPILE MACHINE
------------------------------------
* x86_64 Linux (a Colab CPU runtime works; so does a desktop or WSL2).
* ``pip install ultralytics``
* The **Hailo AI Software Suite / Dataflow Compiler**, which provides
  ``hailo_sdk_client``. It is a registered download from the Hailo
  Developer Zone -- it is not on PyPI and cannot be pip-installed
  unattended, so this script checks for it and explains rather than
  failing with an ImportError.
* TensorFlow (the compiler's optimizer uses it for calibration).

INT8 QUANTIZATION IS MANDATORY
------------------------------
The Hailo-8 runs INT8. Ultralytics rejects ``quantize=32`` for this format
and forces ``quantize=8``.  Quantization changes accuracy -- sometimes a
little, occasionally a lot for small objects, which for us means distant
FUEL. That is why:

* calibration uses **our own FRC images**, via ``--data training/data.yaml``,
  not a generic COCO calibration set; and
* the script prints the exact command to re-validate the compiled model and
  compare it against the ``.pt`` baseline. Do not skip that comparison, and
  do not report the ``.pt`` mAP as if it were the deployed model's.

ON-CHIP NMS AND THE EXPORT CONFIDENCE
-------------------------------------
A detection export bakes ``nms_scores_th`` (the export-time ``--conf``)
into ``nms_config.json``, and the accelerator will never return a box below
it. The runtime threshold in ``frc_perception/config.py`` can then only
filter FURTHER. So export LOW (default 0.05 here) and tune at runtime;
exporting at 0.4 would permanently cap how sensitive the robot can be.

USAGE
-----
    python training/export_hailo.py \\
        --model models/best.pt \\
        --data training/data.yaml \\
        --arch hailo8 \\
        --imgsz 640

    python training/export_hailo.py --check      # environment only, no compile
"""

from __future__ import annotations

import argparse
import importlib
import os
import platform
import shutil
import sys

# The Raspberry Pi AI HAT+ 26 TOPS carries a Hailo-8; the 13 TOPS board
# carries a Hailo-8L. A binary compiled for one will not run on the other.
ARCHITECTURES = ("hailo8", "hailo8l", "hailo10h", "hailo15h", "hailo15l")

EXPECTED_CLASSES = ("fuel", "bumper_blue", "bumper_red")

DFC_INSTRUCTIONS = """
The Hailo Dataflow Compiler (`hailo_sdk_client`) is not importable.

It is a registered download from the Hailo Developer Zone, not a PyPI
package, so it cannot be installed unattended:

  1. Create an account at https://hailo.ai/developer-zone/
  2. Download the "Hailo AI Software Suite" / "Dataflow Compiler" wheel for
     your Python version (x86_64 Linux only).
  3. pip install hailo_dataflow_compiler-<version>-py3-none-linux_x86_64.whl
  4. pip install tensorflow          # the optimizer needs it
  5. Re-run this script.

The Raspberry Pi needs NONE of this. The Pi only needs HailoRT
(`sudo apt install hailo-all`) to RUN an already-compiled model.
"""


def check_environment(verbose: bool = True) -> bool:
    """Report whether this machine can compile a HEF.  Returns True if it can."""
    ok = True
    machine = platform.machine()
    system = platform.system()

    def line(status: str, message: str) -> None:
        if verbose:
            print(f"  [{status}] {message}")

    if system == "Linux" and machine in ("x86_64", "AMD64"):
        line("OK", f"{system} {machine}")
    else:
        ok = False
        line(
            "FAIL",
            f"{system} {machine}: the Hailo Dataflow Compiler is Linux x86_64 only. "
            "This is why the Raspberry Pi cannot compile its own model.",
        )

    try:
        import ultralytics

        line("OK", f"ultralytics {ultralytics.__version__}")
    except ImportError:
        ok = False
        line("FAIL", "ultralytics not installed (pip install ultralytics)")

    try:
        importlib.import_module("hailo_sdk_client")
        line("OK", "hailo_sdk_client (Dataflow Compiler) importable")
    except ImportError:
        ok = False
        line("FAIL", "hailo_sdk_client not importable")
        if verbose:
            print(DFC_INSTRUCTIONS)

    try:
        importlib.import_module("tensorflow")
        line("OK", "tensorflow importable")
    except ImportError:
        ok = False
        line("FAIL", "tensorflow not installed (the DFC optimizer needs it)")

    return ok


def verify_source_model(path: str) -> bool:
    """Confirm the .pt carries exactly the classes the runtime routes on."""
    if not os.path.exists(path):
        print(f"FAIL: no weights at {path}")
        return False
    try:
        from ultralytics import YOLO
    except ImportError:
        print("WARNING: ultralytics missing; cannot verify class names")
        return False
    names = {int(k): str(v) for k, v in dict(YOLO(path).names).items()}
    actual = set(names.values())
    expected = set(EXPECTED_CLASSES)
    print(f"source model classes: {names}")
    if actual == expected:
        print("OK: class names match the runtime contract")
        return True
    print(f"FAIL: expected {sorted(expected)}, got {sorted(actual)}")
    print("      The runtime routes by NAME. Fix training/data.yaml and retrain.")
    return False


def export(args: argparse.Namespace) -> str:
    """Run the Ultralytics Hailo export and return the output directory."""
    from ultralytics import YOLO

    model = YOLO(args.model)

    # Verified against ultralytics 8.4.129: format="hailo" accepts
    # name/quantize/data/fraction/simplify/conf/iou (plus the universal
    # imgsz/batch/device). `name` selects the accelerator architecture and
    # defaults to "hailo8l" -- we must set it explicitly for a Hailo-8.
    # INT8 is forced by the exporter; `end2end` must NOT be passed.
    print(
        f"\nexporting {args.model} -> Hailo {args.arch}\n"
        f"  imgsz={args.imgsz}  calibration data={args.data}\n"
        f"  conf={args.conf} (baked into on-chip NMS)  iou={args.iou}\n"
    )
    output = model.export(
        format="hailo",
        name=args.arch,
        data=args.data,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        fraction=args.fraction,
    )
    return str(output)


def describe_output(directory: str) -> None:
    """List the artifact and say plainly that all of it must be deployed."""
    print("\n" + "=" * 72)
    print(f"EXPORT DIRECTORY: {directory}")
    print("=" * 72)
    if not os.path.isdir(directory):
        print("  (not a directory -- check the export output above)")
        return
    for entry in sorted(os.listdir(directory)):
        path = os.path.join(directory, entry)
        size = os.path.getsize(path) / 1e6 if os.path.isfile(path) else 0.0
        print(f"  {entry:<28} {size:8.2f} MB")
    print(
        "\nDEPLOY THE WHOLE DIRECTORY, not just the .hef.\n"
        "  metadata.yaml carries the class names. Ultralytics reads them from\n"
        "  that sidecar; without it the loaded model has no names and this\n"
        "  project's class-name routing has nothing to route on."
    )


def print_next_steps(directory: str, args: argparse.Namespace) -> None:
    name = os.path.basename(directory.rstrip("/"))
    print(
        f"""
NEXT STEPS

1. Compare accuracy. INT8 quantization is not free, and the honest number
   is the one measured on the COMPILED model:

       # baseline, the .pt you trained
       yolo val model={args.model} data={args.data} imgsz={args.imgsz}

       # deployed model (needs HailoRT + a Hailo device to run)
       yolo val model={directory} data={args.data} imgsz={args.imgsz}

   Record both mAP50 numbers in your build log. If the compiled model
   loses meaningful recall on `fuel`, that is small-object quantization
   loss: try more calibration images before changing anything else.

2. Copy the whole directory to the Raspberry Pi:

       rsync -av {directory}/ pi@<robot-pi>:~/FRC/models/{name}/

3. On the Pi, verify it end to end:

       python scripts/test_hailo.py --model models/{name} --image sample.jpg
       python scripts/benchmark_detector.py --images data/samples/
       python run.py --check

4. Point the config at it if the name differs:

       frc_perception/config.py -> detector.hailo_model_path
"""
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="models/best.pt", help="trained .pt weights")
    parser.add_argument(
        "--data", default="training/data.yaml",
        help="dataset yaml used for INT8 calibration -- use OUR images, not a "
             "generic calibration set",
    )
    parser.add_argument("--arch", default="hailo8", choices=ARCHITECTURES,
                        help="accelerator to compile for (AI HAT+ 26 TOPS = hailo8)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="input size; must match detector.imgsz at runtime")
    parser.add_argument("--conf", type=float, default=0.05,
                        help="baked into on-chip NMS; keep LOW and filter at runtime")
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--fraction", type=float, default=1.0,
                        help="fraction of the calibration set to use")
    parser.add_argument("--out", default="",
                        help="copy the export directory here when finished")
    parser.add_argument("--check", action="store_true",
                        help="check the compile environment and exit")
    args = parser.parse_args()

    print("Hailo export environment:")
    environment_ok = check_environment()
    if args.check:
        print("\n" + ("READY to compile." if environment_ok else "NOT ready to compile."))
        return 0 if environment_ok else 1
    if not environment_ok:
        print("\nRefusing to continue: this machine cannot compile a HEF (see above).")
        return 1

    if not verify_source_model(args.model):
        print("\nRefusing to compile a model with the wrong class names.")
        return 1
    if not os.path.exists(args.data):
        print(f"FAIL: calibration dataset yaml not found: {args.data}")
        print("      INT8 calibration must use representative FRC images.")
        return 1

    try:
        directory = export(args)
    except Exception as exc:  # noqa: BLE001 - the DFC surfaces many failure modes
        print(f"\nEXPORT FAILED: {exc}")
        return 1

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        if os.path.isdir(args.out):
            shutil.rmtree(args.out)
        shutil.copytree(directory, args.out)
        print(f"\ncopied export -> {args.out}")
        directory = args.out

    describe_output(directory)
    print_next_steps(directory, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
