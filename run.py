#!/usr/bin/env python3
"""Entry point: run the perception system, or check it out first.

    python run.py --check          # verify everything possible without driving
    python run.py                  # run against real cameras and a roboRIO
    python run.py --sim-nt         # run with simulated NetworkTables
    python run.py --replay cam0=data/clips/practice.mp4 --sim-nt

``--check`` is the command to run before every bring-up session: it
reports PASS / WARNING / FAIL / MUST INPUT for everything that can be
verified without moving the robot, and its exit code is non-zero if
anything is FAIL.
"""

from __future__ import annotations

import argparse
import importlib
import os
import platform
import shutil
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

# Make the package importable when run from a checkout without installing.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from frc_perception.camera import list_video_devices  # noqa: E402
from frc_perception.config import (  # noqa: E402
    AppConfig,
    config_summary,
    load_config,
    validate_config,
)
from frc_perception.detector import (  # noqa: E402
    HailoDetectorBackend,
    compare_class_names,
)

PASS = "PASS"
WARN = "WARNING"
FAIL = "FAIL"
MUST = "MUST INPUT"

_ORDER = {PASS: 0, WARN: 1, MUST: 2, FAIL: 3}


class CheckReport:
    """Collects check results and prints them grouped by severity."""

    def __init__(self) -> None:
        self.results: List[Tuple[str, str, str]] = []

    def add(self, status: str, name: str, detail: str = "") -> None:
        self.results.append((status, name, detail))

    def count(self, status: str) -> int:
        return sum(1 for s, _n, _d in self.results if s == status)

    def print(self) -> None:
        width = max((len(name) for _s, name, _d in self.results), default=10)
        for status in (FAIL, MUST, WARN, PASS):
            rows = [r for r in self.results if r[0] == status]
            if not rows:
                continue
            print(f"\n--- {status} ({len(rows)}) ---")
            for _status, name, detail in rows:
                print(f"  {name.ljust(width)}  {detail}")

    @property
    def exit_code(self) -> int:
        return 1 if self.count(FAIL) else 0


#: (module, why we need it, fatal if missing)
_REQUIRED_IMPORTS = (
    ("numpy", "all geometry and fusion math", True),
    ("cv2", "USB camera capture, calibration, image handling", True),
    ("yaml", "reading the Hailo export metadata (class names)", True),
    ("ultralytics", "loading the model and running inference", False),
    ("hailo_platform", "HailoRT: talking to the Hailo-8 on the AI HAT+", False),
    ("pupil_apriltags", "AprilTag detection for absolute localization", False),
    ("ntcore", "NetworkTables 4 client (pip package: pyntcore)", False),
)


def check_python(report: CheckReport) -> None:
    version = sys.version_info
    text = f"{version.major}.{version.minor}.{version.micro}"
    if version < (3, 8):
        report.add(FAIL, "python.version", f"{text}; this project needs 3.8+")
    else:
        report.add(PASS, "python.version", text)


def check_imports(report: CheckReport) -> Dict[str, bool]:
    available: Dict[str, bool] = {}
    for module, why, fatal in _REQUIRED_IMPORTS:
        try:
            imported = importlib.import_module(module)
            available[module] = True
            version = getattr(imported, "__version__", "?")
            report.add(PASS, f"import.{module}", f"{version} ({why})")
        except Exception as exc:  # noqa: BLE001 - a broken install can raise anything
            available[module] = False
            report.add(
                FAIL if fatal else WARN,
                f"import.{module}",
                f"unavailable: {exc} -- needed for {why}",
            )
    return available


def read_model_name() -> str:
    """Raspberry Pi models publish their name in the device tree."""
    for path in ("/proc/device-tree/model", "/sys/firmware/devicetree/base/model"):
        try:
            with open(path, "rb") as handle:
                return handle.read().decode("utf-8", "replace").strip("\x00\n ")
        except OSError:
            continue
    return ""


def check_platform(cfg: AppConfig, report: CheckReport) -> bool:
    """Identify the machine.  Returns True if it looks like the target Pi."""
    machine = platform.machine()
    bits = platform.architecture()[0]
    model = read_model_name()

    report.add(PASS, "platform.configured", cfg.platform)
    report.add(PASS, "platform.arch", f"{machine} {bits} ({platform.system()})")

    if model:
        report.add(PASS, "platform.model", model)
    is_pi = "raspberry pi" in model.lower()
    is_pi5 = is_pi and " 5" in model

    if cfg.platform == "raspberry_pi_5":
        if not is_pi:
            report.add(
                WARN, "platform.model",
                f"config targets raspberry_pi_5 but this machine reports "
                f"{model or 'no device-tree model'}. Fine on a development "
                "computer; the Hailo checks below will be skipped or warn.",
            )
        elif not is_pi5:
            report.add(WARN, "platform.model", f"Raspberry Pi detected, but not a Pi 5: {model}")
        if is_pi and machine not in ("aarch64", "arm64"):
            report.add(
                FAIL, "platform.arch",
                f"Raspberry Pi running a {machine} userland. HailoRT and the "
                "Ultralytics wheels need 64-bit Raspberry Pi OS (aarch64).",
            )
        if bits != "64bit":
            report.add(FAIL, "platform.arch", f"{bits} Python; 64-bit is required")
    return is_pi


def check_hailo(cfg: AppConfig, report: CheckReport, have_hailort: bool, on_pi: bool) -> None:
    """Runtime + device + artifact checks for the Hailo-8 accelerator."""
    if cfg.detector.backend != "hailo":
        report.add(
            WARN, "hailo.backend",
            f"detector backend is {cfg.detector.backend!r}, so the Hailo is not in use. "
            "That is correct for development and WRONG for competition.",
        )
    # -- runtime --------------------------------------------------------
    if have_hailort:
        report.add(PASS, "hailo.runtime", "hailo_platform (HailoRT) importable")
    else:
        report.add(
            FAIL if on_pi else WARN, "hailo.runtime",
            "hailo_platform not importable. On Raspberry Pi OS: "
            "`sudo apt install hailo-all` then reboot. "
            "(Expected to be absent on a development computer.)",
        )

    # -- device ----------------------------------------------------------
    identify = shutil.which("hailortcli")
    if identify:
        try:
            completed = subprocess.run(
                [identify, "fw-control", "identify"],
                capture_output=True, text=True, timeout=20, check=False,
            )
            if completed.returncode == 0:
                summary = " | ".join(
                    line.strip()
                    for line in completed.stdout.splitlines()
                    if any(k in line for k in ("Device Architecture", "Serial", "Part Number"))
                )
                report.add(PASS, "hailo.device", summary or "accelerator responded")
            else:
                report.add(
                    FAIL if on_pi else WARN, "hailo.device",
                    f"`hailortcli fw-control identify` failed: "
                    f"{(completed.stderr or completed.stdout).strip()[:160]}",
                )
        except (OSError, subprocess.SubprocessError) as exc:
            report.add(WARN, "hailo.device", f"could not run hailortcli: {exc}")
    else:
        report.add(
            FAIL if on_pi else WARN, "hailo.device",
            "hailortcli not on PATH; cannot confirm the accelerator. "
            "Install hailo-all and reboot.",
        )

    # -- artifact (works with no hardware at all) -------------------------
    backend = HailoDetectorBackend(cfg.detector)
    ok, messages = backend.preflight()
    detail = "; ".join(messages) if messages else "no detail"
    if ok:
        report.add(PASS, "hailo.model", detail)
    elif not os.path.exists(cfg.detector.hailo_model_path):
        report.add(
            MUST, "hailo.model",
            f"{cfg.detector.hailo_model_path} not found. Compile it on an x86_64 "
            "Linux machine with training/export_hailo.py and copy the whole "
            "directory here -- the Pi cannot compile a HEF.",
        )
    else:
        report.add(FAIL, "hailo.model", detail)


def check_config(cfg: AppConfig, report: CheckReport) -> None:
    issues = validate_config(cfg)
    if not issues:
        report.add(PASS, "config", "valid, no unfilled MUST INPUT values")
        return
    for issue in issues:
        status = {"FAIL": FAIL, "WARNING": WARN, "MUST INPUT": MUST}[issue.severity]
        report.add(status, f"config.{issue.where}", issue.message)


def check_pytorch_model(cfg: AppConfig, report: CheckReport, have_ultralytics: bool) -> None:
    """Check the .pt artifact -- the development backend and the export source."""
    path = cfg.detector.pytorch_model_path
    required = cfg.detector.backend == "pytorch"
    if not os.path.exists(path):
        report.add(
            MUST if required else WARN,
            "model.pytorch",
            f"{path} not found. It is the TRAINING output and the source for the "
            "Hailo export; it is not needed at runtime when backend='hailo'.",
        )
        return
    size_mb = os.path.getsize(path) / 1e6
    report.add(PASS, "model.pytorch", f"{path} ({size_mb:.1f} MB)")
    if not have_ultralytics:
        report.add(WARN, "model.classes", "cannot inspect classes: ultralytics missing")
        return
    try:
        from ultralytics import YOLO

        model = YOLO(path)
        names = {int(k): str(v) for k, v in dict(getattr(model, "names", {})).items()}
        ok, message = compare_class_names(names, cfg.detector.class_names)
        report.add(PASS if ok else FAIL, "model.classes", message)
    except Exception as exc:  # noqa: BLE001
        report.add(WARN, "model.classes", f"could not load weights: {exc}")


def check_field_layout(cfg: AppConfig, report: CheckReport) -> None:
    from frc_perception.field_map import try_load_field_map

    path = cfg.localization.apriltag.field_layout_path
    field_map, error = try_load_field_map(path)
    if field_map is None:
        report.add(
            MUST,
            "apriltag.layout",
            f"{error} -- AprilTag localization stays disabled until this exists",
        )
        return
    report.add(
        PASS,
        "apriltag.layout",
        f"{path}: {len(field_map.tags)} tags, field "
        f"{field_map.length_m:.3f} x {field_map.width_m:.3f} m",
    )


def check_calibration(cfg: AppConfig, report: CheckReport) -> None:
    for cam in cfg.cameras:
        if cam.has_calibrated_intrinsics:
            fx, fy, cx, cy = cam.effective_intrinsics()
            rms = (
                f", rms={cam.calibration_rms_px:.3f} px"
                if cam.calibration_rms_px is not None
                else ""
            )
            report.add(
                PASS,
                f"calibration.{cam.name}",
                f"fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}{rms}",
            )
        else:
            report.add(
                MUST,
                f"calibration.{cam.name}",
                "not calibrated -- run scripts/calibrate_intrinsics.py "
                f"--camera {cam.name}",
            )


def check_camera_paths(cfg: AppConfig, report: CheckReport) -> None:
    """Report the persistent device paths available on this machine."""
    devices = list_video_devices()
    by_id = [d for d in devices if d["kind"] == "by-id"]
    if devices:
        report.add(
            PASS, "camera.devices",
            f"{len(by_id)} by-id path(s): "
            + ", ".join(f"{os.path.basename(d['path'])} -> {d['node']}" for d in by_id[:4]),
        )
    else:
        report.add(
            WARN, "camera.devices",
            "no /dev/v4l/by-id entries (normal on a development computer; on the "
            "Pi it means no USB camera is attached)",
        )

    for cam in cfg.cameras:
        if not cam.enabled:
            continue
        if cam.uses_persistent_path:
            exists = os.path.exists(cam.device_path)
            report.add(
                PASS if exists else FAIL,
                f"camera.{cam.name}.path",
                f"{cam.device_path}" + ("" if exists else " does not exist"),
            )
        else:
            report.add(
                MUST, f"camera.{cam.name}.path",
                f"using /dev/video{cam.device_index} with no persistent path; "
                "set device_path from `ls -l /dev/v4l/by-id/`",
            )


def check_cameras(cfg: AppConfig, report: CheckReport, have_cv2: bool) -> None:
    if not have_cv2:
        report.add(WARN, "camera.probe", "skipped: OpenCV unavailable")
        return
    import cv2

    for cam in cfg.cameras:
        if not cam.enabled:
            report.add(WARN, f"camera.{cam.name}", "disabled in config")
            continue
        source = cam.source
        if isinstance(source, str) and not os.path.exists(source):
            report.add(FAIL, f"camera.{cam.name}", f"{source} does not exist")
            continue
        if isinstance(source, int) and not os.path.exists(f"/dev/video{source}"):
            report.add(
                FAIL,
                f"camera.{cam.name}",
                f"/dev/video{source} does not exist (check `ls /dev/video*`)",
            )
            continue
        capture = None
        try:
            capture = cv2.VideoCapture(source)
            if not capture.isOpened():
                report.add(FAIL, f"camera.{cam.name}", f"{source} will not open")
                continue
            ok, frame = capture.read()
            if not ok or frame is None:
                report.add(FAIL, f"camera.{cam.name}", f"{source} opened but read failed")
                continue
            report.add(
                PASS,
                f"camera.{cam.name}",
                f"{source} delivered a {frame.shape[1]}x{frame.shape[0]} frame",
            )
        except Exception as exc:  # noqa: BLE001
            report.add(FAIL, f"camera.{cam.name}", f"probe raised: {exc}")
        finally:
            if capture is not None:
                capture.release()


def check_networktables(cfg: AppConfig, report: CheckReport) -> None:
    nt = cfg.networktables
    if nt.server_override:
        report.add(PASS, "networktables.server", f"explicit server {nt.server_override}")
    elif nt.team_number > 0:
        report.add(PASS, "networktables.server", f"team {nt.team_number} discovery")
    else:
        report.add(MUST, "networktables.server", "set team_number or server_override")
    report.add(PASS, "networktables.identity", nt.client_identity)


def check_controller_safety(cfg: AppConfig, report: CheckReport) -> None:
    if cfg.controller.enabled:
        report.add(
            WARN,
            "controller.enabled",
            "VIRTUAL CONTROLLER IS ENABLED. The Pi will publish live stick "
            "commands; confirm the roboRIO deadman gate before powering the drivetrain.",
        )
    else:
        report.add(PASS, "controller.enabled", "disabled (safe default)")


def scan_must_input(root: str) -> List[Tuple[str, int, str]]:
    """Find every ``MUST INPUT`` marker in the project source."""
    hits: List[Tuple[str, int, str]] = []
    skip_dirs = {".git", "__pycache__", ".pytest_cache", "logs", "models", "data"}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]
        for filename in filenames:
            if not filename.endswith((".py", ".md", ".yaml", ".yml", ".txt")):
                continue
            path = os.path.join(dirpath, filename)
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    for number, line in enumerate(handle, start=1):
                        if "MUST INPUT" in line:
                            hits.append(
                                (os.path.relpath(path, root), number, line.strip())
                            )
            except OSError:
                continue
    return hits


def run_check(cfg: AppConfig, root: str, probe_cameras: bool = True) -> int:
    print("=" * 78)
    print("FRC 2026 perception -- preflight check (Raspberry Pi 5 + Hailo-8)")
    print("=" * 78)
    print(config_summary(cfg))

    report = CheckReport()
    check_python(report)
    on_pi = check_platform(cfg, report)
    available = check_imports(report)
    check_config(cfg, report)
    check_hailo(cfg, report, available.get("hailo_platform", False), on_pi)
    check_pytorch_model(cfg, report, available.get("ultralytics", False))
    check_field_layout(cfg, report)
    check_calibration(cfg, report)
    check_camera_paths(cfg, report)
    if probe_cameras:
        check_cameras(cfg, report, available.get("cv2", False))
    else:
        report.add(WARN, "camera.probe", "skipped (--no-camera-probe)")
    check_networktables(cfg, report)
    check_controller_safety(cfg, report)

    report.print()

    markers = scan_must_input(root)
    print(f"\n--- MUST INPUT markers in source ({len(markers)}) ---")
    for path, number, line in markers:
        print(f"  {path}:{number}: {line[:110]}")

    print("\n" + "=" * 78)
    print(
        f"PASS={report.count(PASS)}  WARNING={report.count(WARN)}  "
        f"MUST INPUT={report.count(MUST)}  FAIL={report.count(FAIL)}"
    )
    if report.exit_code:
        print("RESULT: FAIL -- fix the items above before running on the robot.")
    elif report.count(MUST):
        print(
            "RESULT: NOT READY FOR THE FIELD -- no hard failures, but MUST INPUT "
            "values are still unfilled. The system will run and degrade safely."
        )
    else:
        print("RESULT: all checks passed.")
    print("=" * 78)
    return report.exit_code


def parse_replay(values: Optional[List[str]]) -> Dict[str, str]:
    """``--replay cam0=path`` pairs into a dict."""
    mapping: Dict[str, str] = {}
    for item in values or []:
        if "=" not in item:
            raise SystemExit(f"--replay expects NAME=PATH, got {item!r}")
        name, path = item.split("=", 1)
        mapping[name.strip()] = path.strip()
    return mapping


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check", action="store_true", help="run preflight checks and exit"
    )
    parser.add_argument(
        "--no-camera-probe",
        action="store_true",
        help="with --check, do not try to open the cameras",
    )
    parser.add_argument(
        "--sim-nt",
        action="store_true",
        help="use a simulated NetworkTables backend instead of a roboRIO",
    )
    parser.add_argument(
        "--sim-red",
        action="store_true",
        help="with --sim-nt, report the RED alliance (default: unknown)",
    )
    parser.add_argument(
        "--replay",
        action="append",
        metavar="NAME=PATH",
        help="serve a camera from a video file or image directory instead of hardware",
    )
    parser.add_argument(
        "--backend",
        choices=("hailo", "pytorch"),
        help="override the detector backend for this run (pytorch is DEVELOPMENT only)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="debug logging")
    args = parser.parse_args(argv)

    root = os.path.dirname(os.path.abspath(__file__))
    cfg = load_config()
    if args.backend:
        cfg.detector.backend = args.backend

    if args.check:
        return run_check(cfg, root, probe_cameras=not args.no_camera_probe)

    from frc_perception.app import PerceptionApp
    from frc_perception.logging_setup import setup_logging
    from frc_perception.nt_io import SimulatedNT

    setup_logging(cfg.logging, verbose=args.verbose)

    issues = [i for i in validate_config(cfg) if i.severity == "FAIL"]
    if issues:
        for issue in issues:
            print(f"CONFIG FAIL {issue.where}: {issue.message}", file=sys.stderr)
        print("refusing to start with an invalid configuration", file=sys.stderr)
        return 1

    backend = None
    if args.sim_nt:
        backend = SimulatedNT(is_red=True if args.sim_red else None)

    app = PerceptionApp(cfg, nt=backend, replay_paths=parse_replay(args.replay))
    app.install_signal_handlers()
    return app.run()


if __name__ == "__main__":
    raise SystemExit(main())
