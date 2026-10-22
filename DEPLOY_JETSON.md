# Deploying and testing the decision logic on a Jetson Orin Nano Super 8 GB

Branch: **`frc-2026-jetson-orin-v2`**

This guide gets the branch onto your Jetson and lets you personally test
the autonomy decision-making with your own inputs — **before** any camera,
model or roboRIO exists.

Everything here is offline. The virtual controller is force-disabled, the
NetworkTables backend is in-memory only, and no motor is ever commanded.

---

## Legend: what you can run when

| tag | meaning |
|---|---|
| **[NOW]** | works on a stock JetPack install, no extra packages, no model, no camera |
| **[NEEDS pytest]** | one `pip install pytest` away |
| **[NEEDS TORCH]** | requires NVIDIA's CUDA PyTorch wheel installed first |
| **[NEEDS MODEL]** | requires trained weights at `models/best.pt` |

The whole decision harness is **[NOW]**. That is the point of it.

---

## 1. Get the branch onto the Jetson

From the Jetson desktop, open **Terminal**.

```bash
sudo apt update
sudo apt install -y git
```

```bash
mkdir -p ~/frc && cd ~/frc
git clone https://github.com/Adam-jpg1/FRC.git
cd FRC
git fetch origin frc-2026-jetson-orin-v2
git checkout frc-2026-jetson-orin-v2
```

Confirm you are on the right commit:

```bash
git branch --show-current     # -> frc-2026-jetson-orin-v2
git log --oneline -1          # -> 4f78e53 Add offline decision-logic test harness (or later)
```

If the repo is private, GitHub will ask for credentials. Use a personal
access token as the password, or set up SSH and clone
`git@github.com:Adam-jpg1/FRC.git` instead.

> The preserved baseline is `a277004` on branch
> `claude/frc-2026-jetson-perception-5dz5p0`. Nothing in this guide
> touches it.

---

## 2. Environment setup that cannot break JetPack

### The rule

JetPack ships CUDA, TensorRT, cuDNN, a CUDA-enabled OpenCV and NumPy, all
built against each other. **The way people break a Jetson is `pip
install`ing over those.** Specifically: never pip-install `numpy`,
`opencv-python` or `torch` from PyPI on this machine. The PyPI aarch64
torch wheels are CPU-only, and a newer pip NumPy will break the ABI that
NVIDIA's torch was compiled against.

### First, confirm what JetPack already gave you — **[NOW]**

```bash
python3 --version
python3 -c "import numpy;  print('numpy ', numpy.__version__)"
python3 -c "import cv2;    print('opencv', cv2.__version__)"
dpkg -l 'nvidia-jetpack*' 2>/dev/null | grep -E '^ii' || cat /etc/nv_tegra_release
```

Confirm OpenCV really is the CUDA build (so you know not to replace it):

```bash
python3 -c "import cv2; print('CUDA devices seen by OpenCV:', cv2.cuda.getCudaEnabledDeviceCount())" 2>/dev/null \
  || python3 -c "import cv2; print([l for l in cv2.getBuildInformation().splitlines() if 'CUDA' in l][:3])"
```

### Then: nothing else is required for the decision tests — **[NOW]**

The decision harness imports **only NumPy** from outside the repo. I
verified this by blocking `pytest`, `cv2`, `torch`, `ultralytics`,
`ntcore`, `pupil_apriltags` and `yaml` from the import system and running
all 26 scenarios: 26 passed.

So you can go straight to section 3 with **zero installs**.

### Optional: a venv for the pytest suite — **[NEEDS pytest]**

Use `--system-site-packages` so JetPack's NumPy, OpenCV and (later)
NVIDIA's torch stay visible. Without that flag the venv is isolated and
you lose them.

```bash
cd ~/frc/FRC
python3 -m venv --system-site-packages .venv
source .venv/bin/activate

pip install pytest            # ONLY this. Not numpy, not opencv, not torch.
```

Verify you did not shadow the system packages:

```bash
python3 -c "import numpy, cv2; print(numpy.__file__); print(cv2.__file__)"
```

Both paths should still point at `/usr/lib/python3/dist-packages/...` (or
wherever JetPack put them), **not** inside `.venv`. If either moved into
`.venv`, you shadowed it:

```bash
pip uninstall -y numpy opencv-python opencv-python-headless
```

> **Do not run `pip install -r requirements.txt` on the Jetson** until
> torch is settled (section 8). `requirements.txt` is for the full stack
> and lists `ultralytics`, which drags in a CPU torch. README §5 covers
> the full-stack order.

---

## 3. Run the offline decision tests

All **[NOW]**.

```bash
cd ~/frc/FRC

python3 test_decisions.py --list                  # what the 26 scenarios cover
python3 test_decisions.py --scenario all --quiet   # one line each + summary
python3 test_decisions.py --scenario all           # full input/decision trace
```

Expected tail of `--scenario all --quiet`:

```
============================================================================
SUMMARY: 26 passed, 0 failed, 0 skipped (GPU-dependent)
============================================================================
```

Exit code is `0` only if every assertion passed:

```bash
python3 test_decisions.py --scenario all --quiet; echo "exit=$?"
```

One scenario at a time, with the intermediate navigation scoring:

```bash
python3 test_decisions.py --scenario valid_fuel_ahead --verbose
python3 test_decisions.py --scenario pose_invalid_never_translates --verbose
python3 test_decisions.py --help-expect            # all 35 assertion keys
```

### The pytest suite — **[NEEDS pytest]**

```bash
pytest                                      # 326 passed
pytest tests/test_decision_scenarios.py     # the 26 scenarios + harness integrity
pytest -k "pose_invalid"                    # the safety rule
```

Without OpenCV you get **325 passed, 1 skipped** — one integration test
needs `cv2` to write sample frames and skips cleanly. On JetPack OpenCV is
present, so expect 326.

### Interactive mode — **[NOW]**

```bash
python3 test_decisions.py --interactive
```

---

## 4. Interactive input format

Coordinates are **robot-relative: `+x` forward, `+y` left, in metres.**

These are the only commands that exist:

```
fuel  <x> <y> [conf]      stage a fuel at robot-relative (x, y)
blue  <x> <y> [conf]      stage a blue bumper
red   <x> <y> [conf]      stage a red bumper
list                      show what is staged
clear                     remove everything staged
run                       execute one cycle and print the decision
pose                      change pose mode / robot pose
alliance                  change alliance (red / blue / unknown)
detector                  toggle detector health
save <file.json>          write the current scenario to JSON
help                      this text
quit                      exit
```

`conf` defaults to `0.90` if you leave it off.

### What interactive mode can and cannot stage

It builds **one cycle, one frame on `cam0`, captured now**. So:

| input | interactive? |
|---|---|
| fuel / bumper at a position | **yes** |
| no detection | **yes** (`clear`, then `run`) |
| invalid robot pose | **yes** (`pose` → `invalid`) |
| unhealthy detector | **yes** (`detector`) |
| different alliance | **yes** (`alliance`) |
| **stale frame** | **no** — needs `capture_offset_s`, which is JSON-only |
| **out-of-order frame** | **no** — needs a second cycle with a lower `seq` |
| **multi-cycle track aging** | **no** — needs several cycles |

For those three, use the JSON file in section 6 or the built-in scenarios
(`stale_frame_refused`, `out_of_order_frame_ignored`,
`detector_failure_no_fabrication`).

### Example A — fuel detected ahead

```
decision> fuel 3.0 0.0 0.92
decision> run
```

### Example B — no detection

```
decision> clear
decision> run
```

### Example C — stale frame

**Not available interactively.** Use:

```bash
python3 test_decisions.py --json examples/decision_tests.json --scenario c_stale_frame
python3 test_decisions.py --scenario stale_frame_refused --verbose
```

### Example D — invalid robot pose

```
decision> pose
  pose_mode (odometry_only/tag_initialized/invalid) [odometry_only]: invalid
  robot field x [5.0]:
  robot field y [3.0]:
  robot heading deg [0.0]:
decision> fuel 3.0 0.0 0.99
decision> run
```

Blank input accepts the bracketed default.

### Example E — unhealthy detector

```
decision> clear
decision> fuel 3.0 0.0 0.95
decision> detector
  detector_healthy = False
decision> run
```

`detector` is a toggle — run it again to go back to healthy.

### Freezing an interactive session into a test

```
decision> save my_test.json
```

Then open `my_test.json`, replace its `expect` block with what you believe
should happen, and run it:

```bash
python3 test_decisions.py --json my_test.json
```

> Interactive `run` asserts only the safety invariants, deliberately. It
> is for exploring. A scenario with real expectations is what proves
> something.

---

## 5. What to expect, and what PASS / FAIL means

### Reading the trace

| label | meaning |
|---|---|
| `INPUT` | a value **you** fed in |
| `CODE` | a value **read back out of the real implementation** |
| `NOTE` | harness commentary — **proves nothing** |
| `PASS` / `FAIL` | an assertion result |

Only `CODE` lines are evidence about the system.

A scenario is **PASS** only if every listed check passes, including the
automatic `SAFETY` check that runs whether you asked for it or not. Any
`FAIL` line fails the scenario and makes the process exit non-zero.

### Example A — fuel ahead (real output)

```
  INPUT | frame cam0 seq=1
  INPUT | detector_healthy    = True
  CODE  | scheduler inferred  = [('cam0', 1)]
  CODE  | skipped=0 stale_frames=0 stale_results=0
  CODE  | ranged observations (ROBOT frame):
  CODE  |   cam0 fuel  role=game_piece conf=0.92 x=+3.000 y=+0.000 range=3.000 bearing=+0.0deg
  CODE  | pose VALID   (odometry_only) x=5.000 y=3.000 theta=+0.00deg
  CODE  | fuel tracks (FIELD frame) = 1
  CODE  |   id=1 x=8.000 y=3.000 v=(+0.00,+0.00) conf=0.92 hits=1
  CODE  | DECISION vx=+2.0000 vy=+0.0000 omega=+0.0000  target=(8.0, 3.0)
  CODE  | DECISION reason     = approach fuel pocket (1 piece(s), 3.0 m)
  CODE  | controller enabled=False axes=(0.0, 0.0, 0.0) heartbeat=1
```

**Good** means: the scheduler picked the frame; ranging put the object at
`x=+3.0, y=0.0` in the **robot** frame; the pose was valid so it became
field `(8, 3)` (robot at `(5,3)` + 3 m forward); navigation chose it and
commanded `+X`. Controller stayed disabled with zero axes.

**Suspicious** would be: `pose INVALID` with a non-zero `vx`/`vy` (a
safety violation — the automatic check would FAIL), a field position that
is not robot-position + offset, or `controller enabled=True`.

### Example B — no detection (real output)

```
  CODE  | DECISION vx=+0.0000 vy=+0.0000 omega=+1.0000  target=None
  CODE  | DECISION reason     = no fuel known; rotating to search (no search point configured)
```

**PASS** means `translation_zero` and `omega_nonzero`: it spins to look
around but does not drive anywhere. If you see non-zero `vx`/`vy` here,
something is driving toward a target it does not have.

### Example C — stale frame (real output)

```
  INPUT | frame cam0 seq=1  captured 600 ms ago
  CODE  | scheduler inferred  = nothing
  CODE  | skipped=0 stale_frames=1 stale_results=0
  CODE  | DECISION vx=+0.0000 vy=+0.0000 omega=+1.0000  target=None

  PASS  inferred_frames        want [], got []
  PASS  stale_frames_dropped   want 1, got 1
  PASS  fuel_tracks            want 0, got 0
  PASS  translation_zero       vx=0.0000 vy=0.0000
```

The key line is **`scheduler inferred = nothing`** with
`stale_frames=1`. The frame was refused *before* inference. That is the
whole point: a 600 ms-old image would produce confident, current-looking,
wrong detections. A **FAIL** here — `inferred = [('cam0', 1)]` — would
mean stale imagery is reaching the detector.

### Example D — invalid robot pose (real output)

```
  CODE  | ranged observations (ROBOT frame):
  CODE  |   cam0 fuel  role=game_piece conf=0.99 x=+3.000 y=+0.000 range=3.000 bearing=+0.0deg
  CODE  | pose INVALID (odometry_only) x=5.000 y=3.000 theta=+0.00deg
  CODE  | pose_invalid_reason = awaiting first AprilTag fix (estimator not initialized)
  CODE  | DECISION vx=+0.0000 vy=+0.0000 omega=+1.0000  target=None
  CODE  | DECISION reason     = pose invalid (awaiting first AprilTag fix (estimator not initialized)): translation suppressed

  PASS  all_tracks             want 0, got 0
  PASS  translation_zero       vx=0.0000 vy=0.0000
```

This is **the most important behavior in the project**. Note what happens:
the fuel *is* detected and *is* ranged in the robot frame — but because
the pose is invalid, no field track is created (`all_tracks = 0`) and
translation is exactly zero. Only the search rotation remains.

Any non-zero `vx` or `vy` on a line where `pose INVALID` appears is a
**hard safety failure**, and the automatic `SAFETY` check will FAIL with
`no_translation_without_pose`.

### Example E — unhealthy detector (real output)

```
  INPUT | detector_healthy    = False
  CODE  | scheduler inferred  = nothing
  CODE  | DECISION vx=+0.0000 vy=+0.0000 omega=+1.0000  target=None

  PASS  detector_healthy       want False, got False
  PASS  observations           want 0, got 0
  PASS  all_tracks             want 0, got 0
  PASS  translation_zero       vx=0.0000 vy=0.0000
```

An unhealthy detector produces **nothing** — the scheduler does not even
spend a frame on it. No fabricated boxes, no frozen last result.

To watch tracks actually *age out* after the detector dies mid-match (a
three-cycle scenario interactive mode cannot express):

```bash
python3 test_decisions.py --scenario detector_failure_no_fabrication --verbose
```

### Two things the harness refuses to let slide

1. **A scenario with no `expect` block FAILS.** A test that asserts
   nothing is not a test.
2. **A detection the camera physically cannot see FAILS**, with a
   `NOTE | WARNING:` naming it. `cam0`'s horizontal half-FOV is
   `atan(320/500) ≈ 32.6°`, so e.g. `fuel 1.5 -1.6` is out of view. If you
   see that warning, move the object, do not ignore it.

---

## 6. Testing the production pipeline with synthetic inputs

The scenarios call `PerceptionApp.step()` — the same method the robot's
main loop calls. Only two things are replaced:

```
   REPLACED (hardware)                REAL (production decision code)
   ScriptedFrameSource   ──────────▶  InferenceScheduler
   ScriptedDetector      ──────────▶  YoloDetector._convert()  (confidence gate)
   (ScriptedNT: odometry timing)      alliance routing
                                      ray-to-ground ranging
                                      PoseEstimator + validity gate
                                      robot→field transform
                                      WorldModel (dedup, tracks, aging)
                                      Navigator (scoring, avoidance)
                                      VirtualController (stick mapping)
                                      nt_io payload
```

All three `Scripted*` classes **subclass** their production counterparts,
so a scenario cannot test a more permissive path than the robot runs.

### Run the five worked examples — **[NOW]**

Already in the repo, already passing:

```bash
python3 test_decisions.py --json examples/decision_tests.json
python3 test_decisions.py --json examples/decision_tests.json --quiet
python3 test_decisions.py --json examples/decision_tests.json --scenario c_stale_frame
```

```
  PASS  a_fuel_detected_ahead
  PASS  b_no_detection
  PASS  c_stale_frame
  PASS  d_invalid_robot_pose
  PASS  e_unhealthy_detector
SUMMARY: 5 passed, 0 failed, 0 skipped (GPU-dependent)
```

Copy that file and edit it — it is the fastest way to start.

### The exact JSON fields (no others exist)

Scenario level:

| field | default | notes |
|---|---|---|
| `name` | **required** | |
| `description` | `""` | printed as a `NOTE` |
| `pose_mode` | `"odometry_only"` | `odometry_only` / `tag_initialized` / `invalid` |
| `robot_pose` | `[5.0, 3.0, 0.0]` | field x, y, **theta in radians** |
| `is_red_alliance` | `true` | `true` / `false` / `null` (no FMS data) |
| `cameras` | `["cam0","cam1"]` | only `cam0` and `cam1` exist |
| `cycles` | `[]` | list of cycles |
| `expect` | `{}` | assertions — empty FAILS |
| `config` | `{}` | dotted overrides |
| `requires_gpu` | `false` | |

Cycle level:

| field | default | notes |
|---|---|---|
| `advance_s` | `0.033` | clock advance **before** this cycle |
| `note` | `""` | |
| `frames` | `[]` | |
| `detector_healthy` | `null` | `true`/`false` to flip it; `null` leaves it |
| `freeze_odometry` | `null` | `true` stops the RIO timestamp advancing |
| `inject_tag_pose` | `null` | `[x, y, theta_deg]` — **degrees here** |

Frame level:

| field | default | notes |
|---|---|---|
| `camera_id` | `"cam0"` | |
| `seq` | `1` | a value ≤ the last one is ignored (out-of-order) |
| `capture_offset_s` | `0.0` | **negative = captured that long ago** |
| `detections` | `[]` | |

Detection level:

| field | default | notes |
|---|---|---|
| `class_name` | `"fuel"` | `fuel` / `bumper_blue` / `bumper_red` |
| `confidence` | `0.90` | gate is 0.40 fuel, 0.50 bumpers |
| `x_robot_m` | `null` | forward, metres |
| `y_robot_m` | `null` | left, metres |
| `bbox` | `null` | `[x1,y1,x2,y2]` pixels, instead of a position |
| `pixel_width` | `40.0` | |
| `pixel_height` | `40.0` | |

> `robot_pose` theta is **radians**; `inject_tag_pose` theta is
> **degrees**. That is what the code does — do not mix them up.

Assertion keys: `python3 test_decisions.py --help-expect` (35 of them).
A misspelled key **FAILS** rather than being silently ignored.

### Useful recipes

Age tracks out — large `advance_s` with an empty later cycle:

```json
"cycles": [
  {"advance_s": 0.033, "frames": [{"seq": 1, "detections": [
      {"class_name": "fuel", "x_robot_m": 3.0, "y_robot_m": 0.0}]}]},
  {"advance_s": 2.5, "note": "nothing seen for 2.5 s", "frames": [{"seq": 2}]}
],
"expect": {"fuel_tracks": 0, "all_tracks": 0}
```

Kill odometry mid-run:

```json
{"advance_s": 1.0, "freeze_odometry": true, "frames": [{"seq": 2}]}
```

Slow the robot down for a test:

```json
"config": {"navigation.max_speed_mps": 0.5}
```

An unknown config field is **rejected**, not ignored.

### Also available without hardware — **[NOW]**

```bash
python3 run.py --sim-nt            # the real app loop, simulated NT, no cameras
```

It will log that the model and field layout are missing and that the
detector is unhealthy, then run the loop at ~30 Hz with the pose invalid
and translation suppressed. That exercises the *loop*; the harness
exercises the *decisions*.

---

## 7. Offline and safety guarantees

Verified, not asserted:

| guarantee | enforced by |
|---|---|
| NetworkTables is in-memory only; `NetworkTablesIO` is never imported, so no socket exists | `assert_offline()` |
| `controller.enabled` forced `False`, **re-asserted after** config overrides | `DecisionHarness._build_config()` |
| Every cycle: controller disabled, axes exactly `0.0`, no buttons | `safety_checks()` |
| Every cycle: no translation while `pose_valid` is False | `safety_checks()` |
| Cameras must be synthetic sources | `assert_offline()` |

`safety_checks()` runs on **every** scenario regardless of what it
declares, so you cannot write a scenario that skips it. A scenario that
tries `{"controller.enabled": true}` is proven not to work
(`tests/test_decision_scenarios.py`).

Every `CODE` line in every example above shows
`controller enabled=False axes=(0.0, 0.0, 0.0)`. Nothing in this guide can
move a robot: the stack only ever *recommends* motion, and the roboRIO is
the gate (README §15).

**Do not** pass `run.py` a real team number or run it without `--sim-nt`
while a robot is on the same network, until you are deliberately at
README §14 stage 11.

---

## 8. CUDA and PyTorch — separately, and only when you want the GPU check

None of sections 3–6 need any of this.

### Step 1 — what JetPack gives you

```bash
dpkg -l 'nvidia-jetpack*' 2>/dev/null | grep -E '^ii' || cat /etc/nv_tegra_release
nvcc --version 2>/dev/null || ls /usr/local/cuda*/bin/nvcc
python3 -c "import tensorrt; print('tensorrt', tensorrt.__version__)" 2>/dev/null || echo "tensorrt: check with dpkg -l | grep tensorrt"
```

Note your **JetPack version** — the torch wheel must match it.

### Step 2 — install NVIDIA's CUDA PyTorch

**[NEEDS TORCH]** — this is the step that breaks Jetsons if done wrong.

PyPI's aarch64 `torch` is **CPU-only**. You need NVIDIA's JetPack-matched
wheel, from NVIDIA's Jetson PyTorch resources (the "PyTorch for Jetson"
page / NVIDIA's JetPack pip index) for *your* JetPack version.

I am deliberately not pasting a wheel URL or version number here: they are
specific to your JetPack release and go stale. Get it from NVIDIA, matched
to the version you printed in step 1.

```bash
# inside the venv, with --system-site-packages
source ~/frc/FRC/.venv/bin/activate
pip install <the NVIDIA torch wheel for your JetPack>
```

### Step 3 — verify. This MUST print True

```bash
python3 -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

If it prints `False`, you have the CPU wheel. Fix that before going on —
the project defaults to `detector.require_cuda = True` and the detector
will refuse to start rather than quietly run YOLO on the CPU.

### Step 4 — Ultralytics without losing torch

```bash
pip install ultralytics --no-deps
python3 -c "import torch; print('cuda still ok:', torch.cuda.is_available())"
```

`--no-deps` matters: a plain install will happily replace your CUDA torch
with the CPU one. **Re-verify after every pip install.**

### Step 5 — preflight

```bash
python3 run.py --check --no-camera-probe
```

> **Expected before torch is installed:** this reports
> `FAIL cuda.torch` and exits `1`, because the config targets the Jetson
> and `require_cuda` is `True`. That is correct and does **not** block
> anything in sections 3–6.

### Step 6 — the GPU check — **[NEEDS TORCH] [NEEDS MODEL]**

```bash
python3 test_decisions.py --gpu-check
```

This is the **only** GPU-dependent test in the harness, separate on
purpose. `pytest` asserts no built-in scenario requires a GPU.

It loads the production `YoloDetector` with the configured weights,
confirms CUDA, checks the class contract (`fuel`, `bumper_blue`,
`bumper_red`), runs ten inferences on a synthetic frame and reports
**measured** latency.

Without torch it fails honestly:

```
  INPUT | model_path   = models/best.pt
  INPUT | device       = auto
  INPUT | require_cuda = True
  FAIL  torch.import                           No module named 'torch'
```

Without a model it reports `FAIL model.file`. It never skips quietly, and
it claims no frame rate it did not measure. For a real benchmark use
`scripts/benchmark_detector.py` (README §5), warm, in the closed
electronics bay.

---

## 9. The known observation-merging issue

Documented in **`TESTING_DECISIONS.md` §7** — **STATUS: OPEN, production
behavior deliberately unchanged.**

Short version: `world_model.merge_observations()` is documented as
*cross-camera* de-duplication but merges same-class observations from the
**same camera** within the merge radius. Three fuels from one camera at
0.32 m spacing become **two** tracks
(`merge_distance_game_piece_m = 0.40`). The same applies to adjacent
opponent robots at `merge_distance_robot_m = 0.90`.

See it yourself — **[NOW]**:

```bash
python3 test_decisions.py --scenario adjacent_objects_merge_within_radius --verbose
```

That scenario **pins the current behavior** so any future change is
visible and deliberate. It is not asserting the behavior is correct.

Nothing in this guide changes it. `TESTING_DECISIONS.md` §7 carries the
one-line fix and the one existing test that would need updating, for when
you decide.

---

## Quick reference

```bash
# --- [NOW] no installs, no hardware -------------------------------------
python3 test_decisions.py --list
python3 test_decisions.py --scenario all --quiet
python3 test_decisions.py --scenario pose_invalid_never_translates --verbose
python3 test_decisions.py --json examples/decision_tests.json
python3 test_decisions.py --interactive
python3 test_decisions.py --help-expect
python3 run.py --sim-nt

# --- [NEEDS pytest] ------------------------------------------------------
pytest                                      # 326 passed (325 + 1 skip without cv2)

# --- [NEEDS TORCH] [NEEDS MODEL] ----------------------------------------
python3 -c "import torch; print(torch.cuda.is_available())"   # must be True
python3 run.py --check --no-camera-probe
python3 test_decisions.py --gpu-check
```

Related docs: `TESTING_DECISIONS.md` (harness reference), README §5
(full Jetson stack), README §14 (robot bring-up order), README §15
(safety).
