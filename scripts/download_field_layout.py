#!/usr/bin/env python3
"""Download the official WPILib 2026 AprilTag field layout JSON.

WPILib ships layouts in the allwpilib repo. This script lists that
directory via the GitHub API, finds files starting with "2026", and
saves your choice to data/field_layout.json (the path the config uses).

    python3 scripts/download_field_layout.py            # list + download first 2026 layout
    python3 scripts/download_field_layout.py --name 2026-rebuilt.json
    python3 scripts/download_field_layout.py --url https://.../layout.json

Some seasons ship multiple variants (e.g. "welded" vs "andymark" in
2025). If more than one 2026 file exists you will be shown the list —
pick the one matching your event's field with --name.
"""

import argparse
import json
import os
import sys
import urllib.request

LAYOUT_DIR_API = (
    "https://api.github.com/repos/wpilibsuite/allwpilib/contents/"
    "apriltag/src/main/native/resources/edu/wpi/first/apriltag?ref=main"
)
DEFAULT_OUT = os.path.join(os.path.dirname(__file__), "..", "data", "field_layout.json")


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "frc-perception-setup"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", help="exact layout filename to download")
    parser.add_argument("--url", help="direct URL to a layout JSON (skips discovery)")
    parser.add_argument("--out", default=DEFAULT_OUT, help="output path")
    args = parser.parse_args()

    if args.url:
        chosen_name, url = args.url.rsplit("/", 1)[-1], args.url
    else:
        print("Listing WPILib layout directory ...")
        entries = json.loads(fetch(LAYOUT_DIR_API))
        layouts = sorted(
            e["name"] for e in entries
            if e["name"].startswith("2026") and e["name"].endswith(".json")
        )
        if not layouts:
            print(
                "No 2026 layout found upstream yet. Pass --url with a direct "
                "link, or check https://github.com/wpilibsuite/allwpilib/tree/"
                "main/apriltag/src/main/native/resources/edu/wpi/first/apriltag",
                file=sys.stderr,
            )
            return 1
        print("Available 2026 layouts:", ", ".join(layouts))
        if args.name:
            if args.name not in layouts:
                print(f"--name {args.name} not in the list above", file=sys.stderr)
                return 1
            chosen_name = args.name
        else:
            chosen_name = layouts[0]
            if len(layouts) > 1:
                print(
                    f"Multiple variants exist — defaulting to '{chosen_name}'. "
                    "Re-run with --name if your event uses a different field type."
                )
        url = next(e["download_url"] for e in entries if e["name"] == chosen_name)

    data = fetch(url)
    parsed = json.loads(data)  # validate before writing
    tags = len(parsed.get("tags", []))
    field = parsed.get("field", {})
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "wb") as f:
        f.write(data)
    print(
        f"Saved {chosen_name} -> {args.out} "
        f"({tags} tags, field {field.get('length')}x{field.get('width')} m)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
