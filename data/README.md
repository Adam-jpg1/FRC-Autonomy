# `data/`

Runtime data files. Most of these are generated or downloaded, and are
**not committed to git**.

| path | what it is | how to get it |
|---|---|---|
| `field_layout.json` | official WPILib AprilTag field layout | `python scripts/download_field_layout.py` |
| `calibration/<camera>.json` | camera intrinsics | `python scripts/calibrate_intrinsics.py --camera <name> --square-size <m>` |
| `distance/<camera>.csv` | raw/true distance pairs | `python scripts/calibrate_distance.py --camera <name> --collect` |
| `snapshots/` | frames saved from `camera_preview.py` | press `s` in the preview window |
| `samples/` | still images for hardware-free testing | put your own here |
| `clips/` | recorded video for hardware-free testing | put your own here |

## The field layout

**Never hand-write this file and never reuse a previous season's file.**
Tag IDs and poses change between seasons; a stale layout makes the robot
localize confidently to the wrong place, with no symptom other than bad
behavior on the field.

`scripts/download_field_layout.py` tries several known WPILib locations
and, if none work, prints exact instructions for placing the file by
hand — then validates whatever you placed:

```bash
python scripts/download_field_layout.py
python scripts/download_field_layout.py --from-file ~/Downloads/2026-layout.json
python scripts/download_field_layout.py --validate-only
```

Validation checks for duplicate IDs, malformed poses, and tags that fall
outside the declared field, and prints every tag position so you can
compare against the official field drawings.

The tag **size** is not in this file — set
`localization.apriltag.tag_size_m` in `frc_perception/config.py` from the
official 2026 game manual.

## Calibration files

`calibrate_intrinsics.py` and `calibrate_distance.py` write JSON/CSV here
as a record, and print a config snippet to paste into
`frc_perception/config.py`. The runtime reads the config, not these files —
that keeps the values that affect robot behavior visible in one place and
under version control.

## Hardware-free testing

```bash
python run.py --sim-nt --replay cam0=data/clips/practice.mp4
python run.py --sim-nt --replay cam0=data/samples/
```
