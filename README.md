# FRC 2026 Jetson Perception + Navigation

Vision coprocessor stack for a Jetson Orin Nano Super (8 GB) with two
Arducam OV9782 USB global-shutter cameras, talking to a roboRIO over
NetworkTables 4.

What it does, per ~30 Hz cycle:

1. Captures both cameras on background threads; **stale frames are
   dropped** — a frozen/unplugged camera contributes nothing.
2. Detects `fuel`, `bumper_blue`, `bumper_red` with YOLO
   (stateless `.predict`, one batched call for both cameras).
3. Ranges each detection (bottom-of-box ground intersection, checked
   against known-diameter width, corrected by a per-camera polynomial).
4. Localizes with AprilTags (official WPILib layout) **fused** with
   RIO odometry read over NT; odometry staleness is judged by whether
   the RIO's published timestamp actually advances.
5. Tracks objects in **field coordinates** (dedupe across cameras,
   velocity estimate, aging).
6. Navigates to the densest reachable fuel pocket; searches from field
   center when nothing is visible; **spins in place only when the pose
   is invalid** (never drives blind).
7. Publishes a virtual gamepad over NT so your existing teleop bindings
   drive the robot. **Ships disabled** (`ControllerConfig.enabled=False`).
8. Publishes the perception contract (below) and logs to
   `logs/perception.log` (rotating) + console.

## Repository layout

```
frc_perception/         the package (config.py is the ONLY tuning surface)
scripts/                setup & calibration tools
tests/                  unit tests for the pure math (run anywhere)
run.py                  entry point (--check = hardware-free preflight)
data/                   field layout JSON lands here
models/                 put your trained YOLO weights here
```

## Install (Jetson)

```bash
sudo apt install python3-pip cmake
# 1. NVIDIA CUDA torch/torchvision wheels FIRST — see requirements.txt header.
# 2. Then:
pip3 install -r requirements.txt
python3 scripts/download_field_layout.py
python3 run.py --check          # tells you exactly what is still missing
```

`run.py --check` is safe with no cameras, no RIO, and no GPU.

## Bring-up order (the MUST INPUT checklist)

Search the code for `MUST INPUT` — every entry is in
`frc_perception/config.py` except the button policy. In order:

1. **`team_number`** — your FRC team number (NT server discovery).
2. **`device_index` per camera** — `ls /dev/video*`; USB cameras usually
   claim two nodes each, use the even one. Pin with udev rules if order
   changes across boots.
3. **`horizontal_fov_deg`** — which OV9782 lens variant you bought
   (70°/100°/120° versions exist). Only a fallback until step 4.
4. **Intrinsics (`fx fy cx cy dist_coeffs`)** —
   `python3 scripts/calibrate_intrinsics.py --device 0 --width 1280 --height 800 --corners 9x6 --square-size-m <yours>`
5. **Mount pose per camera** (`mount_x/y/z`, `pitch/yaw/roll`) — tape
   measure + angle gauge, robot frame (+X fwd, +Y left, +Z up, pitch
   positive = down).
6. **`model_path`** — your trained YOLO weights with classes exactly
   `fuel`, `bumper_blue`, `bumper_red`. Export to TensorRT on the Jetson
   for competition framerates.
7. **`fuel_diameter_m`** — from the official 2026 game manual.
8. **`tag_size_m`** — verify against the 2026 manual (default is the
   6.5 in / 0.1651 m used in 2024-25).
9. **`distance_poly` per camera** —
   `python3 scripts/calibrate_distance.py --camera front`
10. **`field_oriented_drive` + `invert_*` flags** — match your teleop
    bindings; verify with the robot ON BLOCKS.
11. **`mechanism_policy()`** in `virtual_controller.py` — your
    intake/shoot button logic (ships empty: no mechanisms actuate).
12. **Fallback field dims** — only if you run without the layout JSON.

Then validate signs on a real field:
`python3 scripts/verify_localization.py` (add `--no-nt` before the RIO
side exists) — push the robot toward the red wall and confirm `x` grows,
push it left and confirm `y` grows, rotate CCW and confirm theta grows.

## NetworkTables contract

Field frame: WPILib **blue origin**, meters/radians, +X toward the red
alliance wall, +Y left. The RIO must publish **RAW blue-origin**
odometry (never alliance-flipped); the Jetson logs a warning if tag
poses and odometry disagree by more than `disagreement_warn_m`.

**Jetson publishes — table `Perception`:**

| key | type | meaning |
|---|---|---|
| `balls_field_x`, `balls_field_y` | double[] | confirmed fuel positions |
| `opponents_field_x`, `opponents_field_y` | double[] | opponent robots only |
| `robot_x`, `robot_y`, `robot_theta` | double | fused pose |
| `pose_valid` | boolean | if false, IGNORE the pose fields |
| `frame_counter` | int | increments every cycle (perception heartbeat) |
| `ball_count`, `opponent_count` | int | array lengths |

**Jetson publishes — table `VirtualController`:**
`enabled`, `heartbeat` (increments every cycle even while disabled — the
robot code should require it to advance), `pov`, axes `axis_left_x`,
`axis_left_y`, `axis_right_x`, `axis_right_y`, `axis_left_trigger`,
`axis_right_trigger`, buttons `button_a/b/x/y`,
`button_left_bumper/right_bumper`, `button_back/start`,
`button_left_stick/right_stick`.

**Jetson subscribes:**
`FMSInfo/IsRedAlliance` (boolean), `Odometry/robot_x`, `robot_y`,
`robot_theta` (radians, CCW+), `timestamp` (any RIO clock in seconds —
it just has to advance while odometry is alive).

## Running

```bash
python3 run.py            # full stack
python3 run.py --check    # preflight, no hardware
python3 -m pytest tests/  # math unit tests, run anywhere
```

To start on boot, wrap `python3 run.py` in a systemd unit with
`Restart=always` and `WorkingDirectory=` set to this repo.

## Safety model

- The virtual controller defaults **OFF**; when off it still publishes
  `enabled=False`, zeroed axes, and an advancing heartbeat.
- Robot code should gate everything on: `enabled == true` AND heartbeat
  advancing AND a driver-held deadman. The Jetson cannot guarantee its
  own liveness — the RIO side must.
- Invalid pose ⇒ spin-search in place only; no translation commands.
- Axes are deadbanded, clamped, and slew-rate limited.

## Honest caveats

- **Frame conventions were verified in software round-trips, not on a
  field.** The apriltag-library ↔ WPILib tag-frame correction and the
  stick sign mapping are exactly what `verify_localization.py` and
  on-blocks testing exist to confirm. Budget an hour with a real tag
  before trusting either.
- Tag solves and odometry are sampled ~one frame apart with no clock
  sync; at FRC speeds the resulting error is small, and the fusion EMA
  absorbs it, but fast-spinning tag fixes will be slightly laggy.
- Single-tag PnP has a known pose ambiguity at long range/shallow
  angles; the height/tilt/distance filters reject most of it, not all.
- Range assumes the object touches the floor; an airborne fuel ranges
  long (the width cross-check flags but does not fix this).
- Ultralytics `.predict` on Orin Nano with plain PyTorch FP16 will do
  roughly 15–30 FPS at 640 px depending on model size; export to
  TensorRT if you need the full 30 Hz loop with two cameras.
- 2026 specifics I could not verify from here are marked `MUST INPUT`
  (fuel diameter, tag size, layout filename — the download script
  discovers whatever WPILib actually published for 2026).
