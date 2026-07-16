#!/usr/bin/env python3
"""Entry point.

    python3 run.py            # run the full perception + navigation stack
    python3 run.py --check    # validate config/files/imports, no hardware

--check is safe anywhere (laptop, CI, Jetson without cameras): it loads
the config, verifies referenced files exist, and reports which optional
runtime dependencies import cleanly.
"""

import argparse
import os
import sys


def check(cfg) -> int:
    """Preflight: report problems without touching hardware."""
    problems = 0

    def report(ok: bool, label: str, detail: str = "") -> None:
        nonlocal problems
        print(f"  [{'ok' if ok else 'FAIL'}] {label}" + (f" — {detail}" if detail else ""))
        if not ok:
            problems += 1

    print("config:")
    report(cfg.team_number > 0 or bool(cfg.nt_server_override),
           "NT server", "set AppConfig.team_number" if cfg.team_number == 0 else "")
    report(len(cfg.cameras) > 0, "cameras configured")
    for cam in cfg.cameras:
        report(cam.has_calibrated_intrinsics, f"camera '{cam.name}' intrinsics",
               "uncalibrated — FOV fallback; AprilTag pose will be poor")
    report(os.path.isfile(cfg.detector.model_path), "YOLO model file",
           cfg.detector.model_path)
    report(os.path.isfile(cfg.localization.field_layout_path), "field layout JSON",
           f"{cfg.localization.field_layout_path} (run scripts/download_field_layout.py)")
    print(f"  [info] virtual controller enabled: {cfg.controller.enabled}")

    print("imports:")
    for module in ("cv2", "numpy", "ultralytics", "pupil_apriltags", "ntcore"):
        try:
            __import__(module)
            report(True, module)
        except ImportError as e:
            report(False, module, str(e))

    print(f"\n{problems} problem(s) found." if problems else "\nAll checks passed.")
    return 1 if problems else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="validate config and environment, then exit")
    args = parser.parse_args()

    from frc_perception.config import default_config
    cfg = default_config()

    if args.check:
        return check(cfg)

    from frc_perception.logging_setup import setup_logging
    setup_logging(cfg.log_dir, cfg.log_level)

    from frc_perception.app import PerceptionApp
    PerceptionApp(cfg).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
