# FRC 2026 — Raspberry Pi 5 + Hailo-8 perception, localization, navigation assist

A perception stack for a **Raspberry Pi 5 (8 GB)** with a **Raspberry Pi
AI HAT+ (Hailo-8, 26 TOPS)** and two Arducam OV9782 USB global-shutter
cameras, talking to a roboRIO over NetworkTables 4.

> **Migrated from a Jetson Orin Nano.** Only the two hardware-facing
> layers changed — frame capture and YOLO inference. Everything from
> `Detection` downward (geometry, localization, world model, navigation,
> the virtual controller, the NetworkTables contract) is the same code the
> Jetson build used, and the roboRIO-facing interface is byte-for-byte
> identical. See [Migration notes](#migration-notes-jetson--raspberry-pi-5).

It turns camera frames and roboRIO telemetry into a field-coordinate model
of what is around the robot, and recommends motion. **It never drives the
robot.** The roboRIO is the safety gate, always.

> **Status.** Everything here is written, syntax-checked, and covered by a
> hardware-free test suite that passes. Nothing in it has been run against
> real cameras, a real roboRIO, a real field, or a trained model, because
> none of those were present when it was written. See
> [What is verified and what is not](#what-is-verified-and-what-is-not).

---

## Contents

1. [Architecture](#1-architecture)
2. [Data flow](#2-data-flow)
3. [Coordinate conventions](#3-coordinate-conventions)
4. [Setup — development machine](#4-setup--development-machine)
5. [Setup — Raspberry Pi 5 + AI HAT+](#5-setup--raspberry-pi-5--ai-hat)
6. [Training, exporting and installing the model](#6-training-exporting-and-installing-the-model)
7. [NetworkTables contract](#7-networktables-contract)
8. [Camera calibration](#8-camera-calibration)
9. [Distance calibration](#9-distance-calibration)
10. [AprilTag setup](#10-apriltag-setup)
11. [Validating localization](#11-validating-localization)
12. [Hardware-free testing](#12-hardware-free-testing)
13. [Unit tests](#13-unit-tests)
14. [First robot bring-up](#14-first-robot-bring-up)
15. [Safety](#15-safety)
16. [Troubleshooting](#16-troubleshooting)
17. [Power and cooling](#17-power-and-cooling)
18. [MUST INPUT checklist](#18-must-input-checklist)
19. [Migration notes](#migration-notes-jetson--raspberry-pi-5)
20. [What is verified and what is not](#what-is-verified-and-what-is-not)

---

## 1. Architecture

The system is a stack of layers. Each depends only on the ones above it,
and each has one job. A failure in a lower layer never corrupts a higher
one.

| module | responsibility | hardware-specific? |
|---|---|---|
| `frc_perception/camera.py` | threaded USB/V4L2 capture, health, reconnect, **frame freshness** | yes |
| `frc_perception/detector.py` | **the hardware boundary**: Hailo-8 and PyTorch backends behind one interface | yes |
| `frc_perception/inference_scheduler.py` | two cameras → one accelerator, newest frame only, no queue | yes |
| `frc_perception/geometry.py` | pixels → robot-relative range and bearing; frame transforms | no |
| `frc_perception/apriltag_locator.py` | AprilTag detection and per-tag pose solving (Pi CPU) | no |
| `frc_perception/field_map.py` | the official WPILib tag layout | no |
| `frc_perception/localization.py` | odometry + tag fusion, **the pose-validity gate** | no |
| `frc_perception/alliance.py` | which bumper color is an opponent | no |
| `frc_perception/world_model.py` | cross-camera de-duplication, field-space tracking | no |
| `frc_perception/navigation.py` | motion recommendation (no cameras, no NT) | no |
| `frc_perception/virtual_controller.py` | motion → gamepad axes | no |
| `frc_perception/nt_io.py` | the NetworkTables contract | no |
| `frc_perception/app.py` | the loop that wires it together | — |
| `frc_perception/config.py` | **every** tunable value in the system | — |

### The hardware boundary

```
              HARDWARE-SPECIFIC                    HARDWARE-INDEPENDENT
   ┌─────────────────────────────────┐   ┌────────────────────────────────┐
   │ OV9782 USB cameras              │   │ Detection[]                    │
   │   ↓  camera.py (V4L2 threads)   │   │   ↓  geometry.py               │
   │   ↓  inference_scheduler.py     │   │   ↓  localization.py           │
   │   ↓  detector.py                │   │   ↓  world_model.py            │
   │        HailoDetectorBackend     │──►│   ↓  navigation.py             │
   │        UltralyticsPyTorchBackend│   │   ↓  virtual_controller.py     │
   └─────────────────────────────────┘   │   ↓  nt_io.py                  │
                                          │   ↓  roboRIO safety gate       │
                                          └────────────────────────────────┘
```

`geometry.py` contains no Hailo code. `world_model.py` imports no
Ultralytics. `navigation.py` has never heard of a HEF file. Swapping
accelerators means adding one class in `detector.py` and changing one
config value — everything below `Detection[]` stays as it is.

### Deviations from the requested file layout

Three, each for a specific reason:

- **`frc_perception/apriltag_locator.py`** was split out of
  `localization.py`. Everything needing the native `pupil_apriltags`
  extension lives there, so the fusion logic and the validity rules stay
  in a module that imports nothing but numpy — and is therefore fully unit
  testable on any machine.
- **`frc_perception/alliance.py`** holds the alliance→role routing.
  Deciding which bumper color is an opponent is a safety decision that
  deserves its own file and its own test file (`tests/test_alliance_logic.py`).
- **`config.py` at the repository root** is a thin re-export of
  `frc_perception/config.py`, which is the real file with every value and
  every `MUST INPUT` marker. The package owns the module so nothing has to
  put the repo root on `sys.path`; the root alias exists so `import config`
  works from the project root as the docs assume. **Edit
  `frc_perception/config.py`.**

- **`frc_perception/inference_scheduler.py`** exists because there are two
  cameras and one accelerator. It is scheduling policy, not detection, and
  keeping it out of `detector.py` lets both be tested independently.

Tests for `camera.py`, `nt_io.py`, `config.py`, the detector backends, the
scheduler and the app loop were added beyond the requested set.

### Design principles, in priority order

1. **Safety.** No valid pose ⇒ no translation. Ever.
2. **Correctness.** Reject impossible geometry instead of forcing an answer.
3. **Determinism.** No hidden state, no wall-clock surprises, seeded tests.
4. **Graceful degradation.** One dead camera, a missing model, a missing
   field layout, a disconnected roboRIO — each degrades one capability and
   leaves the rest running.
5. **Observability.** Every rejection is counted; every state change is
   logged once, not every frame.
6. **Maintainability.** A student should be able to read a layer in one
   sitting.
7. **Performance.** Measured, not assumed.

---

## 2. Data flow

```
        ┌──────────────┐   ┌──────────────┐
        │  camera 0    │   │  camera 1    │      independent threads on the
        │  OV9782 USB  │   │  OV9782 USB  │      Pi 5 CPU, latest frame only
        └──────┬───────┘   └──────┬───────┘
               │ Frame(camera_id, seq, capture_timestamp, image)
               │  ── frames older than frame_freshness_s are DISCARDED ──
               ▼                  ▼
     ┌───────────────────────────────────────┐
     │  inference_scheduler.py               │  freshness-aware round robin,
     │  NO QUEUE: newest frame per camera    │  skipped frames counted
     └──────────────┬────────────────────────┘
                    │ one frame at a time
                    ▼
     ┌───────────────────────────────────────┐
     │  detector.py -> HailoDetectorBackend  │  fuel / bumper_blue / bumper_red
     │  Hailo-8 on the AI HAT+ (HailoRT)     │  routing by class NAME
     └──────────────┬────────────────────────┘
                    │ InferenceResult
                    │  ── results older than max_detection_result_age_s
                    │     are DISCARDED, not injected into the world model ──
                    ▼
     ┌───────────────────────────────────────┐
     │  Detection[]  ==  THE HARDWARE        │
     │  BOUNDARY. Nothing below knows what   │
     │  produced these.                      │
     └──────────────┬────────────────────────┘
                    ▼
     ┌───────────────────────────────────────┐   alliance.py
     │  class name ─► role                   │◄── FMSInfo/IsRedAlliance
     │  friendly bumpers dropped here        │
     └──────────────┬────────────────────────┘
                    ▼
     ┌───────────────────────────────────────┐
     │  geometry.py   ray → ground plane     │  bottom-center pixel + intrinsics
     │  + pinhole-width cross-check          │  + mount pose  ⇒  x, y, range,
     │  + per-camera distance polynomial     │     bearing (robot frame)
     │  INVALID geometry ⇒ rejected, not     │
     │  guessed                              │
     └──────────────┬────────────────────────┘
                    │ RobotObservation
                    │
   ┌────────────────┼─────────────────────────────────────────┐
   │                │                                          │
   │   ┌────────────▼───────────┐      ┌─────────────────────┐ │
   │   │  apriltag_locator.py   │      │  Odometry/* from    │ │
   │   │  tags ⇒ robot pose     │      │  the roboRIO        │ │
   │   └────────────┬───────────┘      └──────────┬──────────┘ │
   │                │ TagPoseEstimate             │ OdometryReading
   │                ▼                             ▼            │
   │   ┌────────────────────────────────────────────────────┐  │
   │   │  localization.py                                   │  │
   │   │  odometry = continuous backbone                    │  │
   │   │  tags     = absolute correction (complementary)    │  │
   │   │  ⇒ PoseEstimate { pose, VALID, source, quality }   │  │
   │   └────────────────────┬───────────────────────────────┘  │
   └────────────────────────┼──────────────────────────────────┘
                            │
              pose VALID? ──┴── no ──► world model CLEARED,
                            │          navigation translation SUPPRESSED
                           yes
                            ▼
     ┌───────────────────────────────────────┐
     │  robot frame ──► FIELD frame          │
     └──────────────┬────────────────────────┘
                    ▼
     ┌───────────────────────────────────────┐
     │  world_model.py                       │  cross-camera de-duplication,
     │  persistent field-space tracks        │  gated association,
     │  position, velocity, age, confidence  │  least-squares velocity
     └──────────────┬────────────────────────┘
                    ▼
     ┌───────────────────────────────────────┐
     │  navigation.py                        │  fuel pocket scoring,
     │  ⇒ DesiredMotion (vx, vy, omega)      │  opponent avoidance
     └──────────────┬────────────────────────┘
                    ▼
     ┌───────────────────────────────────────┐
     │  virtual_controller.py                │  deadband, clamp, slew,
     │  ⇒ ControllerState (axes + buttons)   │  heartbeat  [DISABLED by default]
     └──────────────┬────────────────────────┘
                    ▼
     ┌───────────────────────────────────────┐
     │  nt_io.py    Perception/               │
     │              VirtualController/        │
     └──────────────┬────────────────────────┘
                    ▼
         ╔═══════════════════════════════════╗
         ║  roboRIO  —  THE SAFETY GATE      ║
         ║  deadman + mode + heartbeat check ║
         ╚═══════════════════════════════════╝
```

---

## 3. Coordinate conventions

Every sign error you will hit comes from one of these. They are documented
in `frc_perception/types.py` and enforced by `tests/test_geometry.py`.

### Robot frame
```
        +X  forward
         ▲
         │
+Y ◄─────┼          +Z is up, out of the page
  left   │
```
Origin on the floor at the robot center. `theta` is **CCW-positive**.

### Field frame — WPILib blue-origin
```
   +Y  ▲                                            ┌──────────┐
       │                                            │   RED    │
  ┌────┴─────┐                                      │   wall   │
  │  BLUE    │  ────────────  +X  ────────────►     └──────────┘
  │  wall    │
  └──────────┘  origin (0,0) at the blue wall, right-hand corner
```
The roboRIO publishes **raw blue-origin coordinates regardless of
alliance**, and this code never mirrors them. If your robot code mirrors
by alliance, remove that before connecting this.

### Camera frame — OpenCV
`+X` image right, `+Y` image **down**, `+Z` out of the lens.

### Mount angles
| angle | positive direction |
|---|---|
| `pitch_deg` | **down**, toward the floor |
| `yaw_deg` | **CCW**, to the robot's left |
| `roll_deg` | about the optical axis |

Composed as `R_robot←camera = Rz(yaw) · Ry(pitch) · Rx(roll) · CV_TO_FORWARD`.

### AprilTag frames

`pupil_apriltags` reports the tag pose in its own "optical" tag frame
(`+X` printed-right, `+Y` printed-down, `+Z` **into** the tag). WPILib's
layout uses `+X` **out of** the tag face, `+Z` up.
`apriltag_locator.APRILTAG_TO_WPILIB_TAG` converts between them, and the
full chain is:

```
field←robot = field←tag_WPI · (camera←tag_WPI)⁻¹ · (robot←camera)⁻¹
camera←tag_WPI = camera←tag_AprilTag · APRILTAG_TO_WPILIB_TAG⁻¹
```

This is the single most likely place for a sign error in the project. It
is proven self-consistent by `tests/test_localization.py`, and proven
*correct for your robot* only by `scripts/verify_localization.py` on a
real field.

### Units

Meters, seconds, radians — everywhere, except config fields ending in
`_deg`, which humans type in degrees and the code converts.

---

## 4. Setup — development machine

Everything except the accelerator itself runs on a laptop. The unit suite
needs no Hailo hardware, no HailoRT, no Ultralytics and no torch.

```bash
git clone <this-repo>
cd <this-repo>

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install numpy PyYAML pytest              # enough for the whole unit suite
pip install opencv-python                    # for the calibration/preview tooling
pip install ultralytics pupil-apriltags pyntcore   # optional, for the full stack

pytest                            # must pass
python run.py --check --no-camera-probe

# on a laptop, select the development detector backend explicitly
python run.py --backend pytorch --sim-nt --replay cam0=data/samples/
```

In PyCharm: open the repo root as the project root, point the interpreter
at `.venv`, and mark nothing as a sources root — `conftest.py` handles the
import path for tests and each script inserts the repo root itself.

---

## 5. Setup — Raspberry Pi 5 + AI HAT+

### Hardware

| item | note |
|---|---|
| Raspberry Pi 5, 8 GB | 64-bit Raspberry Pi OS (Bookworm or newer) |
| Raspberry Pi AI HAT+, 26 TOPS | carries a **Hailo-8**. The 13 TOPS board carries a Hailo-8L — a model compiled for one will not run on the other |
| 2× Arducam OV9782 USB | UVC cameras. **Not** CSI, so Picamera2/libcamera is not used |
| Active cooling | required — see [Power and cooling](#17-power-and-cooling) |

### Three separate dependency sets

Keep these straight; mixing them is the most common setup failure.

| set | where | what |
|---|---|---|
| **APT / system** | the Pi | `hailo-all` — PCIe driver, HailoRT, `hailo_platform` bindings |
| **Python runtime** | the Pi and dev machines | `requirements.txt` |
| **Model compilation** | an x86_64 Linux machine, **never the Pi** | Hailo Dataflow Compiler + TensorFlow |

### 1. System packages and the accelerator

```bash
sudo apt update
sudo apt full-upgrade          # AI HAT+ support wants a current firmware/kernel
sudo apt install dkms
sudo apt install hailo-all
sudo reboot
```

`hailo-all` pulls in the PCIe driver, HailoRT, the `hailo_platform` Python
bindings and `hailortcli`. After the reboot, confirm the accelerator is
alive:

```bash
hailortcli fw-control identify
```

That must report a device and its architecture (`HAILO8`). If it does not:

```bash
dmesg | grep -i hailo          # did the PCIe driver bind?
lspci | grep -i hailo          # is the board seen at all?
```

Re-seat the HAT and its ribbon before assuming anything is broken. On some
boards PCIe Gen 3 must be enabled in `/boot/firmware/config.txt`
(`dtparam=pciex1_gen=3`) — check the current Raspberry Pi AI HAT+
documentation rather than trusting a value copied from a forum post.

### 2. Python environment

Create the venv with `--system-site-packages` so the apt-installed
`hailo_platform` bindings stay importable:

```bash
sudo apt install python3-opencv cmake build-essential v4l-utils
python3 -m venv --system-site-packages ~/frcvenv
source ~/frcvenv/bin/activate
pip install -r requirements.txt
python -c "import hailo_platform, cv2, ultralytics; print('ok')"
```

`ultralytics` will install a **CPU** build of torch on the Pi. That is
correct and expected: production inference runs on the Hailo-8, and torch
is only present as an Ultralytics dependency. There is no GPU build to
look for and none is needed.

### 3. Stable camera device paths

USB cameras re-enumerate across boots, and each OV9782 typically claims two
`/dev/videoN` nodes (the second is a metadata node). Find the persistent
paths:

```bash
python scripts/camera_preview.py --list
ls -l /dev/v4l/by-id/
v4l2-ctl -d /dev/v4l/by-id/<...> --list-formats-ext
```

Put those paths in `cameras[].device_path` in
`frc_perception/config.py`. If both cameras are identical modules with the
same USB serial, `by-id` cannot tell them apart — use `/dev/v4l/by-path/`,
which is keyed on the physical USB port, and always plug each camera into
the same port. `run.py --check` reports a MUST INPUT until this is done.

### 4. Performance

Start by measuring. Nothing in this repository claims a frame rate.

```bash
python scripts/benchmark_detector.py --images data/samples/
python scripts/benchmark_detector.py --video data/clips/practice.mp4 --frames 300
```

Then watch `loop_fps`, `camera_N_fps`, `camera_N_inference_fps`,
`detector_latency_ms` and `frames_skipped` on the `Perception` table while
the robot runs. Measure again after ten minutes with the case closed: the
Pi 5 throttles, and a number taken in the first thirty seconds is not the
number you get in the third match of the day.

Knobs, in the order worth trying:

1. Active cooling that actually works (check `vcgencmd measure_temp`).
2. `detector.imgsz` — must match the size the HEF was compiled at, so
   changing it means re-exporting.
3. `max_inferences_per_cycle` — drop to 1 to halve accelerator load; the
   round-robin still alternates fairly, each camera just runs at half rate.
4. `apriltag.detect_every_n_frames` and `apriltag.quad_decimate` — AprilTag
   detection runs on the CPU and is often the real bottleneck on a Pi, not
   the accelerator.

A high `frames_skipped` count is not a bug. It means capture is outrunning
inference and the scheduler is correctly discarding old frames. It is a
signal about how fast you can drive before the world model lags, not an
error to suppress.

## 6. Training, exporting and installing the model

Training and deployment are **separate stages**. You do not retrain to
re-export.

```
   TRAINING                          DEPLOYMENT
   ────────                          ──────────
   dataset                           best.pt
      ↓  Colab / training machine       ↓  x86_64 Linux + Hailo DFC
   best.pt                           best_hailo_model/
      ↓  validate (.pt)                 ├── best.hef
   mAP, confusion matrix             ├── metadata.yaml   ← class names live here
                                     └── nms_config.json
                                        ↓  copy the WHOLE directory
                                     Raspberry Pi + HailoRT
```

### 1. Train (unchanged)

See [`training/README.md`](training/README.md). Produces `best.pt` with
exactly the classes `fuel`, `bumper_blue`, `bumper_red`.

```bash
python training/validate_yolo.py --model models/best.pt --data training/data.yaml
```

### 2. Compile for the Hailo-8 — **not on the Pi**

Hailo compilation requires the Hailo Dataflow Compiler, which is x86_64
Linux only. Ultralytics enforces this itself:

```python
assert LINUX and not ARM64, "Hailo export is only supported on Linux x86_64."
# ultralytics/engine/exporter.py
```

On an x86_64 Linux machine (a Colab CPU runtime works) with the Dataflow
Compiler installed:

```bash
python training/export_hailo.py --check            # environment only
python training/export_hailo.py \
    --model models/best.pt \
    --data training/data.yaml \
    --arch hailo8 \
    --imgsz 640
```

* `--arch hailo8` — the AI HAT+ 26 TOPS carries a Hailo-8. The 13 TOPS
  board is `hailo8l`, and the binaries are not interchangeable.
* `--data` — INT8 calibration uses **our FRC images**, not a generic
  calibration set. The Hailo-8 runs INT8; Ultralytics forces `quantize=8`
  for this format and refuses FP32.
* `--conf` defaults to a low 0.05 because for detection exports the
  confidence is **baked into on-chip NMS** (`nms_config.json`). The
  accelerator never returns a box below it, so the runtime threshold can
  only filter further. Export low, tune at runtime.

### 3. Compare accuracy honestly

Quantization is not free, and small objects — distant `fuel` — lose the
most. Record both numbers:

```bash
yolo val model=models/best.pt data=training/data.yaml imgsz=640           # baseline
yolo val model=models/best_hailo_model data=training/data.yaml imgsz=640  # deployed
```

The second needs HailoRT and a Hailo device. Do not report the `.pt` mAP as
if it were the deployed model's. If recall on `fuel` drops meaningfully,
try more calibration images before changing anything else.

### 4. Install on the Pi

```bash
rsync -av best_hailo_model/ pi@<robot-pi>:~/FRC/models/best_hailo_model/
```

**Copy the whole directory.** `metadata.yaml` carries the class names;
Ultralytics reads them from that sidecar, and this project's class-name
routing has nothing to route on without it. Copying only the `.hef`
produces a model that loads and detects nothing — the single most likely
deployment mistake, and the one `run.py --check` and
`scripts/test_hailo.py` both look for first.

```bash
python scripts/test_hailo.py --model models/best_hailo_model --image sample.jpg
python run.py --check
```

### Development backend

For laptop work with no accelerator:

```bash
python run.py --backend pytorch --sim-nt --replay cam0=data/clips/practice.mp4
```

This loads `models/best.pt` through PyTorch. It is **development only**:
`run.py --check` raises a WARNING if it is selected while `platform` is
`raspberry_pi_5`, because running YOLO on the Pi's CPU during a match
starves capture, AprilTags and the main loop of the cores they need.

## 7. NetworkTables contract

### Subscribed — the roboRIO publishes these

| key | type | meaning |
|---|---|---|
| `FMSInfo/IsRedAlliance` | boolean | true ⇒ we are RED |
| `Odometry/robot_x` | double | meters, blue-origin field frame |
| `Odometry/robot_y` | double | meters, blue-origin field frame |
| `Odometry/robot_theta` | double | radians, CCW from field +X |
| `Odometry/timestamp` | double | the RIO's own clock |

> **`Odometry/timestamp` must keep advancing.** Freshness is decided by
> that value changing, never by when we read it. NetworkTables serves the
> last published value forever after a publisher dies, so a frozen
> timestamp is exactly how this code detects that robot code has stopped.
> Publish it every loop from `Timer.getFPGATimestamp()` or equivalent.

### Published — table `Perception`

Required contract keys:

| key | type |
|---|---|
| `balls_field_x`, `balls_field_y` | double[] |
| `opponents_field_x`, `opponents_field_y` | double[] |
| `robot_x`, `robot_y`, `robot_theta` | double |
| `pose_valid` | boolean |
| `frame_counter` | int |
| `ball_count`, `opponent_count` | int |

Diagnostics (also always published):

`camera_0_healthy`, `camera_1_healthy`, `camera_0_fps`, `camera_1_fps`,
`camera_0_inference_fps`, `camera_1_inference_fps`,
`camera_0_frames_skipped`, `camera_1_frames_skipped`,
`model_loaded`, `detector_healthy`, `detector_backend`,
`detector_classes_ok`, `detector_error`, `detector_latency_ms`,
`preprocess_ms`, `postprocess_ms`, `frames_skipped`,
`stale_frames_dropped`, `stale_results_dropped`,
`odometry_fresh`, `tag_pose_available`, `tag_count`,
`localization_source`, `localization_quality`, `pose_invalid_reason`,
`translation_disagreement_m`, `heading_disagreement_deg`, `loop_fps`,
`perception_latency_ms`, `inference_ms`, `apriltag_ms`,
`balls_field_vx`, `balls_field_vy`, `opponents_field_vx`,
`opponents_field_vy`, `nav_reason`, `nav_vx_mps`, `nav_vy_mps`,
`nav_omega_radps`, `nav_target_x`, `nav_target_y`, `nav_has_target`,
`virtual_controller_enabled`.

`camera_N_*` is indexed by position in `config.cameras`. Capture rate
(`camera_N_fps`) and inference rate (`camera_N_inference_fps`) are separate
numbers: with one accelerator serving two cameras they are expected to
differ, and conflating them hides whether capture or inference is the
bottleneck.

**The roboRIO-facing contract is unchanged from the Jetson build.** The
required keys, their types, the subscriptions and the coordinate
conventions are identical; robot code does not know or care which
coprocessor is on the other end.

`opponents_field_*` contains every robot that must be **avoided**. With a
known alliance that is the opposing robots. With the alliance unknown or
stale it also includes robots that cannot be classified — refusing to
drive through a robot we cannot identify is always safe. Robots positively
identified as **friendly are never in this list**.

### Published — table `VirtualController`

`enabled`, `heartbeat`, `left_x`, `left_y`, `right_x`, `right_y`,
`left_trigger`, `right_trigger`, `a`, `b`, `x`, `y`, `left_bumper`,
`right_bumper`, `back`, `start`, `left_stick`, `right_stick`, `pov`.

Every key is published every cycle, so the roboRIO never has to infer a
missing value. See [Safety](#15-safety).

---

## 8. Camera calibration

Uncalibrated cameras fall back to an ideal pinhole derived from
`horizontal_fov_deg`. That is fine for bring-up and **not** fine for
AprilTag pose solving — the tag detector is refused uncalibrated frames
outright, because a plausible-but-wrong 6-DOF solve is worse than none.

```bash
python scripts/calibrate_intrinsics.py --camera cam0 \
    --board 9x6 --square-size 0.025
```

- `--board` is **inner** corners (a 9×6 board has 10×7 squares).
- `--square-size` is in **meters** — measure a printed square with calipers.
- Print the board flat and mount it rigidly. A wavy printout silently
  ruins the result.
- Capture 20–30 views at many angles and distances, especially near the
  image **corners**, which is where the distortion coefficients get their
  information.
- `SPACE` captures, `c` calibrates, `d` drops the last, `q` quits.

Aim for RMS reprojection error well under 1.0 px. Above ~1.5 px, do not
ship it — the script tells you the usual causes.

Paste the printed snippet into `frc_perception/config.py`.

> Intrinsics are only valid at the resolution they were calibrated at. If
> you change the capture resolution, recalibrate.

Then **measure the mount pose** — height above the floor, forward/left
offset from robot center, and the pitch/yaw/roll of the optical axis.
Camera height is the dominant term in ground-range error; measure it to
the lens, not to the case.

---

## 9. Distance calibration

The geometric range is only as good as the mount measurements behind it.
Rather than assume they are perfect, measure the residual and fit it out:

```bash
python scripts/calibrate_distance.py --camera cam0 --collect
# click the point where the object touches the floor, type the true
# distance from ROBOT CENTER, repeat 10-15 times across the useful range

python scripts/calibrate_distance.py --camera cam0 --order 2
```

The script reports RMS error before and after the fit, the mean bias, and
every per-point residual. If the fit barely improves anything, the error
is not systematic — re-measure the mount pose rather than reaching for a
higher polynomial order.

Set `distance_poly_max_m` to the far end of what you actually measured.
Beyond that the runtime holds the correction constant instead of
extrapolating.

---

## 10. AprilTag setup

```bash
python scripts/download_field_layout.py
# if that cannot reach a verified source it prints exact manual
# instructions, then:
python scripts/download_field_layout.py --from-file ~/Downloads/2026-layout.json
python scripts/download_field_layout.py --validate-only
```

Validation checks for duplicate IDs, malformed poses and tags outside the
declared field, and prints every tag position so you can compare against
the official field drawings.

Then set the tag size — it is **not** in the layout file:

```python
# frc_perception/config.py
localization.apriltag.tag_size_m = <edge length in meters>  # MUST INPUT
```

> Never hand-write the layout and never reuse a previous season's file.
> Tag IDs and poses change, and a stale layout makes the robot localize
> confidently to the wrong place.

### How localization actually works

Odometry is the continuous backbone. Tags supply absolute correction. The
estimator holds one SE(2) correction transform `C` and publishes
`fused = C ∘ odometry`. Each accepted tag fix nudges `C` toward the
correction that would make the fused pose match the tag pose, by a
configurable fraction, with the resulting pose step clamped.

**This is a complementary correction, not a Kalman filter.** It has no
covariance and does not model drift rate. It is used because its failure
modes are obvious and a student can reason about it in a pit at 2 am.
`PoseEstimator` is small and narrow, so replacing it with a real filter
later is a contained change.

What it refuses to do:

- Beyond `disagreement_reject_m` / `_rad`, a tag fix is **not blended at
  all**. Averaging two poses that disagree wildly produces a pose that is
  wrong in a new and creative way.
- If that disagreement persists for `disagreement_invalidate_after_s`, the
  pose is declared **INVALID** rather than snapped. The robot stops
  translating; it does not teleport.
- The first fix, and only the first, is adopted outright — the RIO's
  odometry origin is arbitrary until vision anchors it.

`pose_valid` is False when any of these hold:

- no tag fix has ever been accepted (`require_tag_initialization`, default on);
- odometry is stale **and** no tag pose is recent enough;
- the numbers are not finite;
- the pose is outside the field by more than the margin;
- unresolved tag/odometry disagreement.

---

## 11. Validating localization

This is the step teams skip and then lose a match to.

```bash
python scripts/verify_localization.py
python scripts/verify_localization.py --camera cam0 --window
```

Live table of odometry pose, tag pose, fused pose, validity, translation
and heading disagreement, tag count and tag IDs. Push the robot around a
real field and confirm all six:

1. Push **down field** (toward red): odometry X and tag X both **increase**.
2. Push to the robot's **left** while facing down field: both Y **increase**.
3. Rotate **counter-clockwise** seen from above: both headings **increase**.
4. Park at a measured spot: tag X/Y match a tape measure within a few cm.
5. Circle a single tag: the reported robot pose stays consistent.
6. Drive a lap: disagreement stays small. Growing disagreement means
   odometry drift (expected) or a bad transform (not).

If X and Y come out swapped or mirrored, the camera **yaw** or a mount
sign is wrong. **Fix the config, never the math** — the math is tested.

---

## 12. Hardware-free testing

Most development needs no robot.

```bash
# Simulated NetworkTables, no cameras, no accelerator, no model: the loop
# still runs, publishes the full contract, and commands nothing.
python run.py --sim-nt

# Development detector backend (PyTorch .pt) on a laptop
python run.py --backend pytorch --sim-nt --replay cam0=data/clips/practice.mp4

# Validate a deployed Hailo model's class contract without any hardware
python scripts/test_hailo.py --model models/best_hailo_model --artifact-only

# Replay a recorded clip through the real pipeline
python run.py --sim-nt --replay cam0=data/clips/practice.mp4

# Or a directory of stills
python run.py --sim-nt --replay cam0=data/samples/ --replay cam1=data/samples2/

# Pretend we are on the red alliance
python run.py --sim-nt --sim-red

# Model checks on saved images, no cameras needed
python scripts/test_yolo_image.py --model models/best.pt --image data/samples/
python scripts/test_yolo_image.py --model models/best.pt --image shot.jpg --range cam0
```

`SimulatedNT` advances its own odometry timestamp, so the freshness logic
is genuinely exercised. Set `freeze_odometry = True` on it to reproduce a
dead roboRIO publisher and watch the pose correctly go invalid.

`--range cam0` reports what range and bearing each detection would produce
for that camera's configured mount — the quickest way to sanity-check
mount numbers against a photo of a known layout.

---

## 13. Unit tests

```bash
pytest                      # everything
pytest -v tests/test_navigation.py
pytest -k "pose_invalid"    # the safety rule
```

The suite covers geometry (ray-to-ground, every rejection path, transform
round trips), localization (freshness, fusion, disagreement, validity, the
AprilTag frame chain), the world model (de-duplication, association,
velocity, aging), alliance routing, navigation (**the pose-invalid rule**),
the virtual controller (neutral default, clamp, deadband, slew,
heartbeat), the camera freshness rule, the NT contract, config validation,
and the app loop end to end.

Migration-specific coverage:

```bash
pytest tests/test_detector_backends.py    # the hardware boundary
pytest tests/test_inference_scheduler.py  # two cameras, one accelerator
```

* both backends emit an identical `Detection`;
* a Hailo export directory missing `metadata.yaml` fails preflight;
* a missing accelerator produces an unhealthy detector, **no detections,
  and no PyTorch substitution**;
* a newer frame replaces an older waiting one, and the skip is counted;
* the scheduler holds no queue — 100 frames between cycles yield exactly
  one inference;
* stale frames never reach the accelerator; stale results never reach the
  world model;
* round robin keeps a fast camera from starving a slow one;
* one camera's inference failure does not block the other;
* a dead detector lets tracks age out, and navigation stays safe.

It requires only `numpy`, `PyYAML` and `pytest` — **no HailoRT, no
accelerator, no Ultralytics, no torch, no OpenCV.** Hailo hardware is
mocked; the "accelerator absent" path is exactly what a development
machine exercises.

---

## 14. First robot bring-up

Do these in order. Do not skip ahead to autonomous driving.

| stage | command | what you are proving |
|---|---|---|
| **1** | `pytest` | the math and the logic are intact |
| **2** | `hailortcli fw-control identify` | the AI HAT+ is seated, powered and enumerated |
| **3** | `python run.py --check` | platform, HailoRT, model artifact, config, calibration, MUST INPUTs |
| **4** | `python scripts/camera_preview.py --list` | the persistent device paths, for `device_path` |
| **5** | `python scripts/camera_preview.py --camera cam0` | camera 0 streams at the expected resolution and rate |
| **6** | `python scripts/camera_preview.py --camera cam1` | camera 1 does too, independently |
| **7** | `python scripts/test_hailo.py --model models/best_hailo_model --image shot.jpg` | the accelerator loads the model, has the right classes, and infers |
| **8** | `python scripts/benchmark_detector.py --images data/samples/` | **measured** latency and FPS on this Pi |
| **9** | `python run.py --sim-nt --replay cam0=clip.mp4` | the whole pipeline on recorded video |
| **10** | `python run.py --sim-nt` with both cameras live | both cameras + the accelerator; watch `camera_N_inference_fps` and `frames_skipped` |
| **11** | `python run.py` (controller still disabled) | NetworkTables connects; watch the `Perception` table |
| **12** | watch `odometry_fresh` | roboRIO odometry arrives and its timestamp advances |
| **13** | `python scripts/verify_localization.py` | tags are detected and produce poses |
| **14** | `verify_localization.py`, pushing the robot | **X, Y and heading signs are right** (section 11) |
| **15** | watch `balls_field_*` / `opponents_field_*` | detections land where the objects really are |
| **16** | **robot on blocks**, `controller.enabled = True` | joystick polarity, with the wheels off the ground |
| **17** | controlled drivetrain test, human on the deadman | assisted motion, small speed caps first |

Stage 16 is where `invert_left_x` / `invert_left_y` / `invert_right_x` and
`field_oriented_drive` get confirmed. **On blocks. Every time.**

---

## 15. Safety

### The virtual controller is off by default

`controller.enabled = False`. While it is off, the adapter publishes a
fully neutral gamepad — every axis 0.0, every trigger 0.0, every button
False, POV −1, `enabled = false`.

### The roboRIO is the safety gate, not the coprocessor

Robot code must accept virtual controls only when **all** of these hold:

1. `VirtualController/enabled == true`;
2. `VirtualController/heartbeat` is **advancing** (a frozen heartbeat means
   the Pi is dead or wedged — this code advances it every cycle, enabled or
   not, precisely so you can check);
3. a **human is holding a deadman** button on a real controller;
4. the robot mode permits automation;
5. and your own sanity limits on the resulting speeds.

Never make the Pi the last line of defense. It is a Linux box on a robot:
it can hang, thermal throttle, brown out on a weak supply, or lose a USB
camera mid-match.

### Mechanisms ship neutral

`mechanism_policy()` returns no intents and no button is ever pressed. The
mapping from intents to buttons is robot-specific and is not guessed here —
a wrong guess fires a shooter in the pit. If you implement the policy but
forget the bindings, the code logs an error rather than dropping the intent
silently.

### The pose-invalid rule

If `pose_valid` is False, navigation returns **zero translation**, always,
no matter what is in the world model. The only permitted motion is an
in-place search rotation, which needs no field coordinates — and that is
configurable off. This is the first branch in `Navigator.plan()` and it has
its own tests.

### A dead accelerator fails safe, and loudly

If the Hailo-8 does not initialize or stops responding:

- the detector reports **unhealthy** and produces **no detections** — never
  fabricated ones;
- **no CPU fallback is started.** Running YOLO on the Pi's CPU would starve
  capture, AprilTags and the main loop, deliver a few frames per second,
  and hide a hardware fault behind a plausible trickle of stale boxes;
- existing world-model tracks **age out** on their own — the last known ball
  position is never frozen in place;
- `detector_healthy`, `model_loaded` and `detector_error` on the
  `Perception` table say exactly what happened;
- localization, AprilTags and navigation keep running, so the robot still
  knows where it is and still refuses to translate when it does not.

### Stale data never masquerades as fresh

- Frames older than `frame_freshness_s` are discarded before the
  accelerator sees them.
- Odometry is fresh only while the RIO's timestamp advances.
- Inference results older than `max_detection_result_age_s` are dropped
  rather than injected into the world model — a correct answer about where
  objects *were* is a wrong answer about where they are.
- The scheduler holds no queue, so a slow cycle can never accumulate a
  backlog of old frames.
- Tracks age out; navigation is told whether a track is recent, aging, or
  expired.

---

## 16. Troubleshooting

| symptom | likely cause | what to do |
|---|---|---|
| `camera N UNHEALTHY` in the log | wrong device index, cable, or USB bandwidth | `ls /dev/video*`, `v4l2-ctl --list-devices`, try each camera alone; two OV9782 at 1280×800×60 on one USB bus may exceed bandwidth — lower FPS or split buses |
| `camera_N_fps` far below configured | driver ignored the FOURCC/resolution request | check the "driver gave" warning in the log; OV9782 needs `MJPG` for full rate |
| Detections but no ranges | mount pose unmeasured (`mount_z_m = 0`) or box bottoms clipped | `run.py --check`; use `test_yolo_image.py --range cam0` to see the rejection reason |
| Ranges consistently long or short | mount height or pitch off | re-measure, then `calibrate_distance.py` |
| `model_loaded = false` / `detector_healthy = false` | no accelerator, no model directory, or a class mismatch | `run.py --check`, then `scripts/test_hailo.py` — it reports which of the seven stages failed |
| `hailortcli` not found | `hailo-all` not installed | `sudo apt install hailo-all && sudo reboot` |
| `hailortcli fw-control identify` fails | HAT not seated, or the PCIe driver did not bind | `dmesg \| grep -i hailo`, `lspci \| grep -i hailo`, re-seat the board |
| Model loads but detects nothing | only the `.hef` was copied, so there are no class names | copy the whole export directory; `test_hailo.py` stage 5 checks this |
| `class mismatch` at preflight | the export was built from the wrong weights | re-export from a `best.pt` whose names are exactly the three expected |
| Model refuses to load on the Pi | compiled for the wrong architecture (`hailo8l` vs `hailo8`) | check `hailo_arch` in `metadata.yaml`; re-export with `--arch hailo8` |
| Detection quality worse than the `.pt` | INT8 quantization loss, worst on small/distant objects | compare `yolo val` on both; use more representative calibration images |
| High `frames_skipped` | capture is outrunning inference | not a bug — it is the latest-frame policy. Lower `max_inferences_per_cycle` or `imgsz`, or improve cooling |
| High `stale_results_dropped` | inference is slower than `max_detection_result_age_s` | benchmark, cool the Pi, reduce `imgsz` (requires re-export) |
| Everything slows down after a few minutes | Pi 5 thermal throttling | `vcgencmd measure_temp`, `vcgencmd get_throttled`; fix cooling |
| Random reboots / USB dropouts | undersized or unstable supply | see [Power and cooling](#17-power-and-cooling) |
| `pose_valid` never becomes true | no accepted tag fix yet | check `tag_count`, `apriltag.layout` in `--check`, and camera calibration — the tag detector refuses uncalibrated cameras |
| `ODOMETRY STALE` but the robot is running | `Odometry/timestamp` not being republished | publish it every robot loop |
| `LARGE LOCALIZATION DISAGREEMENT` | camera mount sign wrong, or a mirrored pose from robot code | `scripts/verify_localization.py`, section 11 |
| Robot pose mirrored in Y | robot code mirroring odometry by alliance | publish raw blue-origin coordinates; this code never mirrors |
| Phantom obstacles | pose was briefly invalid, or ranges are wrong | the world model is cleared on invalid pose; check `pose_invalid_reason` |
| Teammates treated as obstacles | alliance unknown/stale | check `FMSInfo/IsRedAlliance`; with it unknown, every robot is an obstacle by design |
| NT never connects | wrong team number, or mDNS flaky | set `networktables.server_override = "10.TE.AM.2"` |
| Virtual controller does nothing | it is disabled by default | that is the safe default; see section 15 before changing it |
| Cameras swapped after a reboot | using `/dev/videoN` instead of a persistent path | set `device_path` from `camera_preview.py --list` |
| Robot drives the wrong way on blocks | joystick polarity | flip `invert_left_x/y`, `invert_right_x` — never patch the math |

Logs: console plus `logs/perception.log` (rotating, 5 MB × 5). State
transitions are logged once per change, not once per frame; repeated
errors are rate-limited with a suppressed count.

---

## 17. Power and cooling

The Pi 5 with an AI HAT+ draws meaningfully more than a Pi 4, and both the
SoC and the Hailo-8 throttle when hot. Neither failure is loud: the robot
just gets slower, usually in the third match of the day.

### Power

* The Pi 5 expects a **5 V supply with real headroom**. The official
  desktop supply is 5 V / 5 A (27 W); the AI HAT+ and two USB cameras all
  draw from the same rail. Undervoltage shows up as USB dropouts and
  spontaneous reboots long before it shows up as a clear error.
* An FRC robot's 12 V bus is **not** a clean supply. It sags hard when the
  drivetrain draws current, and a regulator that is adequate at rest can
  brown the Pi out during a push match — exactly when you want perception
  working.
* Check for undervoltage on the Pi:

  ```bash
  vcgencmd get_throttled     # 0x0 is what you want
  dmesg | grep -i voltage
  ```

```
# MUST INPUT: the actual 12 V -> 5 V regulator, its continuous and peak
# current rating, its fusing, and how it is wired into the robot's power
# distribution. This depends on your electrical system and is not invented
# here. Size it for the Pi 5 + AI HAT+ + two USB cameras with margin for
# brownout during drivetrain current spikes, and verify with
# `vcgencmd get_throttled` under real driving load, not on the bench.
```

### Cooling

* **Active cooling is required**, not optional — the official Pi 5 Active
  Cooler or an equivalent heatsink-and-fan. The AI HAT+ mounts over the Pi
  and restricts airflow, so a passive heatsink that was fine on a bare
  board may not be.
* Watch it under real load:

  ```bash
  vcgencmd measure_temp
  vcgencmd get_throttled
  ```

* Benchmark **warm**: run `scripts/benchmark_detector.py` for several
  minutes with the electronics enclosure closed. A number taken in the
  first thirty seconds is not the number you get in a match.
* Mount the Pi where it gets air and is not sitting against a motor
  controller or a battery.

---

## 18. MUST INPUT checklist

Values that cannot be known without the physical robot, a calibration run,
a trained model, or the official 2026 game manual. `python run.py --check`
lists every one still unfilled, and scans the source for the markers.

| # | value | where | how to get it |
|---|---|---|---|
| 1 | Team number | `networktables.team_number` | you know this one |
| 2 | Camera persistent device paths | `cameras[].device_path` | `python scripts/camera_preview.py --list` |
| 3 | Camera intrinsics (fx, fy, cx, cy, distortion) | `cameras[]` | `scripts/calibrate_intrinsics.py` |
| 4 | Lens horizontal FOV | `cameras[].horizontal_fov_deg` | your Arducam variant's spec (fallback only) |
| 5 | Camera mount positions | `cameras[].mount_x_m/_y_m/_z_m` | tape measure, on the robot |
| 6 | Camera mount rotations | `cameras[].pitch_deg/yaw_deg/roll_deg` | measure on the robot; down/CCW positive |
| 7 | Distance polynomials | `cameras[].distance_poly` | `scripts/calibrate_distance.py` |
| 8 | Exposure / gain | `cameras[].manual_exposure/manual_gain` | `scripts/camera_preview.py` under real field lighting |
| 9 | Compiled Hailo model | `detector.hailo_model_path` | `training/export_hailo.py` on x86_64 Linux, then copy the whole directory |
| 10 | Trained weights | `detector.pytorch_model_path` | `training/`, then copy `best.pt` |
| 11 | FUEL diameter | `detector.fuel_diameter_m` | official FRC 2026 game manual |
| 12 | Bumper face width | `detector.bumper_width_m` | measure a real robot |
| 13 | AprilTag size | `localization.apriltag.tag_size_m` | official FRC 2026 game manual |
| 14 | Field layout JSON | `data/field_layout.json` | `scripts/download_field_layout.py` |
| 15 | Fallback field dimensions | `localization.fallback_field_*` | 2026 manual (only if running without the layout) |
| 16 | Assisted speed caps | `navigation.max_speed_mps`, `max_omega_radps` | your drivetrain, conservatively |
| 17 | Robot radii | `navigation.opponent_radius_m`, `own_robot_radius_m` | half the diagonal of a bumpered robot |
| 18 | Field search point | `navigation.search_point_field` | 2026 field coordinates (None ⇒ spin-search, which is safe) |
| 19 | Field-oriented drive? | `controller.field_oriented_drive` | match your teleop bindings |
| 20 | Joystick polarity | `controller.invert_left_x/_y`, `invert_right_x` | **robot on blocks** |
| 21 | Full-scale stick speeds | `controller.full_scale_speed_mps`, `full_scale_omega_radps` | what a fully deflected stick means to your robot |
| 22 | Mechanism policy and bindings | `virtual_controller.mechanism_policy` / `_apply_mechanism` | your robot's mechanisms |
| 23 | **Robot 12 V → 5 V power solution** | README §17 | your electrical system — size it and verify under load |

---

## Migration notes: Jetson → Raspberry Pi 5

The project was originally built for an NVIDIA Jetson Orin Nano Super.
Migrating to the Pi 5 + Hailo-8 changed only the hardware-facing layers.

### What changed

| Jetson | Raspberry Pi 5 + Hailo-8 |
|---|---|
| CUDA PyTorch / TensorRT inference | Hailo-8 via HailoRT, through Ultralytics' `HailoBackend` |
| `.pt` / `.engine` single-file model | `best_hailo_model/` directory (`.hef` + `metadata.yaml` + `nms_config.json`) |
| TensorRT engine built on-device | HEF compiled on **x86_64 Linux only**; the Pi never compiles |
| `YoloDetector` (one concrete class) | `DetectorBackend` interface + `HailoDetectorBackend` / `UltralyticsPyTorchBackend` |
| detection ran on every fresh frame inline | `InferenceScheduler`: latest-frame, round-robin, no queue |
| JetPack / CUDA / `torch.cuda.is_available()` setup | `sudo apt install hailo-all`, `hailortcli fw-control identify` |
| FP16 on GPU | INT8, calibrated on our own FRC images |
| `/dev/videoN` indexes | persistent `/dev/v4l/by-id/` paths |

### What did **not** change

Geometry, ranging, AprilTag detection and pose solving, the field map,
localization and the pose-validity gate, alliance routing, the world model,
navigation and the pose-invalid safety rule, the virtual controller, and
the entire NetworkTables contract. Those modules are untouched except for
one identifier rename (`camera_name` → `camera_id`) applied consistently
across the boundary types.

The roboRIO does not know the coprocessor changed.

---

## What is verified and what is not

### Verified in the environment this was written in

- Every file byte-compiles (`python -m compileall`).
- The hardware-free test suite passes.
- Pure math: ray-to-ground ranging and its rejection paths, robot↔field
  transforms and round trips, the AprilTag→WPILib frame chain (proven
  self-consistent against hand-computed poses), SE(2) correction algebra,
  least-squares velocity, polynomial correction and its clamping.
- Logic: odometry freshness by timestamp advancement (including that
  repeated reads do not manufacture freshness), pose-validity rules,
  disagreement handling, alliance routing, cross-camera de-duplication,
  track association and aging, fuel pocket scoring, opponent avoidance,
  **the pose-invalid no-translation rule**, controller neutral/clamp/
  deadband/slew/heartbeat behavior.
- Contracts: config validation, the `Perception` key set, the model
  class-name check.
- `python run.py --check` runs and reports correctly on a machine with no
  accelerator (warnings, not import crashes).
- The detector hardware boundary: both backends emit an identical
  `Detection`; the Hailo artifact preflight (directory, `.hef`,
  `metadata.yaml`, class names, architecture) using a fabricated export
  directory; safe failure with no accelerator and **no CPU fallback**.
- The inference scheduler against a fake clock and fake cameras:
  latest-frame selection, skip counting, no queue, stale-frame refusal,
  stale-result rejection, round-robin fairness, per-camera failure
  isolation, and the full timing chain.
- The Ultralytics Hailo export and inference API was read directly from
  the installed package source (8.4.129) rather than recalled: format name,
  the `name`/architecture argument and its valid values, forced INT8, the
  x86_64-only assertion, the output directory layout, and the fact that
  `metadata.yaml` is where class names come from.

### Written but NOT verified — requires the Pi and the AI HAT+

- **Anything involving the Hailo-8 itself.** No Hailo hardware, no
  HailoRT, and no Hailo Dataflow Compiler were present. Specifically
  unverified: `hailo-all` installation, `hailortcli fw-control identify`,
  loading a real `.hef`, on-chip NMS output, and inference latency.
  `scripts/test_hailo.py` has never had a stage 1, 2, 4 or 6 pass.
- **The Hailo export itself.** `training/export_hailo.py` was written
  against the Ultralytics exporter source but has never compiled a HEF —
  the Dataflow Compiler is a registered download that cannot be installed
  unattended.
- **Quantization accuracy.** INT8 loss versus the `.pt` baseline is
  unmeasured. Measure it before trusting the model on distant `fuel`.
- Real OV9782 behavior on a Pi 5: USB bandwidth with two cameras, MJPG
  negotiation, exposure/gain units, reconnect on a real unplug.
- Actual persistent device paths and their stability across boots.
- **Any frame rate whatsoever.** No FPS number is claimed anywhere in this
  repository because none has been measured. Run
  `scripts/benchmark_detector.py` on the real Pi, warm.
- Thermal behavior and the power solution under drivetrain load.
- A NetworkTables connection to a real roboRIO, and end-to-end latency.
- Real AprilTag detection, and whether the camera mount numbers you enter
  are correct. The frame chain is proven self-consistent; only
  `scripts/verify_localization.py` on a real field proves it is right for
  your robot.
- Camera mount transforms, and therefore real-world range accuracy.
- Joystick polarity and `field_oriented_drive` — stage 14, on blocks.
- Real FMS alliance data.
- YOLO detection quality, which depends entirely on your dataset.

**This system has never been run on a robot, and has never touched a Hailo
accelerator.** Nothing here is "production verified". Work the bring-up
order in section 14 and verify each stage yourself.
