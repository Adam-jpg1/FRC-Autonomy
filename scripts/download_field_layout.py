#!/usr/bin/env python3
"""Obtain the official WPILib AprilTag field layout JSON -- or explain how to.

    python scripts/download_field_layout.py                 # try to fetch
    python scripts/download_field_layout.py --validate-only # check what you placed
    python scripts/download_field_layout.py --from-file ~/Downloads/layout.json

WHY THIS SCRIPT IS CAUTIOUS
---------------------------
The exact filename and URL of the FRC 2026 layout are published by WPILib
for the season and cannot be known in advance with certainty.  This script
therefore TRIES a list of candidate locations, and if none of them work it
tells you exactly what to download by hand and where to put it, rather
than writing a fabricated file.  A wrong tag layout is worse than no tag
layout: the robot localizes confidently to the wrong place.

Whatever route the file arrives by, it is validated the same way: parsed,
checked for duplicate IDs, checked that every tag lies inside the declared
field, and summarized so you can eyeball the numbers against the official
field drawings.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from frc_perception.config import load_config  # noqa: E402
from frc_perception.field_map import FieldLayoutError, parse_field_layout  # noqa: E402

# Candidate locations, most likely first. These follow the naming pattern
# WPILib has used in recent seasons; NONE of them is guaranteed for 2026.
CANDIDATE_URLS = (
    "https://raw.githubusercontent.com/wpilibsuite/allwpilib/main/apriltag/src/main/native/resources/edu/wpi/first/apriltag/2026-reefscape-welded.json",
    "https://raw.githubusercontent.com/wpilibsuite/allwpilib/main/apriltag/src/main/native/resources/edu/wpi/first/apriltag/2026-field.json",
    "https://raw.githubusercontent.com/wpilibsuite/allwpilib/main/apriltag/src/main/native/resources/edu/wpi/first/apriltag/2026-official.json",
)

# Where the file lives if WPILib (robotpy-apriltag / the WPILib install) is
# already present on this machine.
LOCAL_HINTS = (
    "/usr/local/share/wpilib/apriltag",
    os.path.expanduser("~/wpilib/2026/apriltag"),
)

MANUAL_INSTRUCTIONS = """
COULD NOT FETCH THE LAYOUT AUTOMATICALLY.

Do this instead -- it takes a minute and removes all doubt:

  1. Get the official file. Any ONE of these works:
       a. WPILib source tree:
            https://github.com/wpilibsuite/allwpilib/tree/main/apriltag/src/main/native/resources/edu/wpi/first/apriltag
          Pick the 2026 season file (the welded/official field variant that
          matches the field you are playing on).
       b. Your local WPILib installation, which ships the same JSON:
            ~/wpilib/2026/apriltag/   (or the WPILib install directory)
       c. A robotpy install:
            python -c "import robotpy_apriltag, os; \\
                       print(os.path.dirname(robotpy_apriltag.__file__))"
          and look for the season JSON alongside it.

  2. Put it here:
       {destination}

  3. Validate it:
       python scripts/download_field_layout.py --validate-only

DO NOT hand-write this file and do not reuse a previous season's file.
The tag IDs and poses change, and a stale layout makes the robot localize
confidently to the wrong place.
"""


def validate(path: str) -> int:
    if not os.path.exists(path):
        print(f"FAIL: nothing at {path}")
        return 1
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as exc:
        print(f"FAIL: {path} is not valid JSON: {exc}")
        return 1
    try:
        field_map = parse_field_layout(payload, source_path=path)
    except FieldLayoutError as exc:
        print(f"FAIL: {path} is not a usable WPILib layout: {exc}")
        return 1

    print(f"OK: {path}")
    print(f"  tags        : {len(field_map.tags)}  ids {field_map.tag_ids}")
    print(f"  field size  : {field_map.length_m:.4f} m x {field_map.width_m:.4f} m")
    print(f"  field center: ({field_map.center[0]:.3f}, {field_map.center[1]:.3f})")
    print("\n  tag positions (field frame, meters):")
    for tag_id in field_map.tag_ids:
        translation = field_map.get(tag_id).translation
        print(
            f"    {tag_id:>3}: x={translation[0]:7.3f}  y={translation[1]:7.3f}  "
            f"z={translation[2]:6.3f}"
        )
    print(
        "\n  Check these numbers against the official field drawings before you\n"
        "  trust AprilTag localization. Also confirm the tag SIZE in\n"
        "  frc_perception/config.py (localization.apriltag.tag_size_m) -- it is\n"
        "  NOT in this file."
    )
    return 0


def try_local(destination: str) -> bool:
    """Copy the layout out of a local WPILib/robotpy installation, if present."""
    candidates = list(LOCAL_HINTS)
    try:
        import robotpy_apriltag

        candidates.append(os.path.dirname(robotpy_apriltag.__file__))
    except ImportError:
        pass

    for directory in candidates:
        if not os.path.isdir(directory):
            continue
        for name in sorted(os.listdir(directory)):
            if not name.endswith(".json") or "2026" not in name:
                continue
            source = os.path.join(directory, name)
            try:
                with open(source, "r", encoding="utf-8") as handle:
                    parse_field_layout(json.load(handle))
            except (OSError, ValueError, FieldLayoutError):
                continue
            shutil.copyfile(source, destination)
            print(f"copied local layout {source} -> {destination}")
            return True
    return False


def try_download(destination: str, timeout: float) -> bool:
    for url in CANDIDATE_URLS:
        print(f"trying {url}")
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                data = response.read()
        except (urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
            print(f"  no: {exc}")
            continue
        try:
            parse_field_layout(json.loads(data.decode("utf-8")))
        except (ValueError, FieldLayoutError) as exc:
            print(f"  fetched, but it is not a usable layout: {exc}")
            continue
        os.makedirs(os.path.dirname(destination) or ".", exist_ok=True)
        with open(destination, "wb") as handle:
            handle.write(data)
        print(f"  downloaded -> {destination}")
        return True
    return False


def main() -> int:
    cfg = load_config()
    default_destination = cfg.localization.apriltag.field_layout_path

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=default_destination, help="where to write it")
    parser.add_argument("--validate-only", action="store_true", help="just check --out")
    parser.add_argument("--from-file", default="", help="install a file you downloaded")
    parser.add_argument("--timeout", type=float, default=10.0)
    args = parser.parse_args()

    if args.validate_only:
        return validate(args.out)

    if args.from_file:
        if not os.path.exists(args.from_file):
            print(f"FAIL: no such file {args.from_file}")
            return 1
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        shutil.copyfile(args.from_file, args.out)
        print(f"installed {args.from_file} -> {args.out}")
        return validate(args.out)

    if os.path.exists(args.out):
        print(f"{args.out} already exists; validating it instead of overwriting.")
        print("Delete it first if you want to re-fetch.\n")
        return validate(args.out)

    if try_local(args.out) or try_download(args.out, args.timeout):
        return validate(args.out)

    print(MANUAL_INSTRUCTIONS.format(destination=os.path.abspath(args.out)))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
