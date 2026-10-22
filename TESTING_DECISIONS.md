# Testing the decision logic — offline, on the Jetson

`test_decisions.py` drives the **real** decision pipeline with synthetic
inputs so you can test autonomy behavior before the cameras or the roboRIO
are connected.

```bash
python test_decisions.py --scenario all      # 26 scenarios, PASS/FAIL, exit code
python test_decisions.py --list              # what each scenario covers
python test_decisions.py --interactive       # type values, see decisions
```

Needs only `numpy` + `PyYAML`. No camera, no GPU, no model file, no
roboRIO, no network.

---

## 1. Where the boundary is

Only two things are replaced. Everything that makes a *decision* is the
production code the robot runs.

```
        REPLACED (hardware)                  REAL (production code)
  ┌──────────────────────────────┐   ┌────────────────────────────────────┐
  │ ScriptedFrameSource          │   │ InferenceScheduler                 │
  │   implements FrameSource     │──►│   latest-frame, round robin, no    │
  │   you set seq + timestamp    │   │   queue, freshness + staleness     │
  ├──────────────────────────────┤   ├────────────────────────────────────┤
  │ ScriptedDetector             │   │ YoloDetector._convert()            │
  │   SUBCLASSES YoloDetector;   │──►│   per-class confidence gate,       │
  │   only the model call is     │   │   class-name allow-list,           │
  │   replaced                   │   │   Detection construction           │
  ├──────────────────────────────┤   ├────────────────────────────────────┤
  │ ScriptedNT                   │   │ alliance.role_for_class()          │
  │   SUBCLASSES SimulatedNT;    │──►│ geometry.estimate_ground_object()  │
  │   only odometry timing is    │   │ PoseEstimator + validity gate      │
  │   made deterministic         │   │ geometry.robot_to_field()          │
  └──────────────────────────────┘   │ WorldModel (dedup, tracks, aging)  │
                                     │ Navigator (scoring, avoidance)     │
                                     │ VirtualController (stick mapping)  │
                                     │ nt_io.build_perception_payload()   │
                                     └────────────────────────────────────┘
```

The scenarios call `PerceptionApp.step()` — the same method the robot's
main loop calls. There is no second, simplified decision model anywhere.

The three `Scripted*` classes subclass their production counterparts
rather than reimplementing them, so a scenario cannot accidentally test a
more permissive detector or a different conversion path than the robot
runs. (An earlier draft did reimplement the detector, and it silently
skipped the confidence gate — which is why it now subclasses.)

---

## 2. Safety

Nothing here can move a robot.

| guard | where |
|---|---|
| NetworkTables is always in-memory `SimulatedNT`; `NetworkTablesIO` is never imported | `assert_offline()` |
| `controller.enabled` forced `False`, **re-asserted after** config overrides | `DecisionHarness._build_config()` |
| Every cycle asserted: controller disabled, all axes exactly `0.0`, no button pressed | `safety_checks()` |
| Every cycle asserted: no translation while `pose_valid` is False | `safety_checks()` |
| Cameras must be synthetic sources | `assert_offline()` |

`safety_checks()` runs on **every** scenario whether it declares those
expectations or not, so you cannot write a scenario that skips them. The
guards are themselves tested in `tests/test_decision_scenarios.py` — a
scenario that tries to enable the controller via a config override is
proven not to.

`SimulatedNT.publish_perception()` stores a dict. It has no socket, no
publishers and no `ntcore` instance; a test asserts those attributes do
not exist. **Simulated output cannot reach a real robot.**

---

## 3. Running scenarios

```bash
python test_decisions.py --scenario all                 # full trace
python test_decisions.py --scenario all --quiet          # one line each
python test_decisions.py --scenario valid_fuel_ahead -v  # + intermediate scoring
python test_decisions.py --help-expect                   # every assertion key
```

Exit code is `0` only if every assertion passed, so this works in CI. The
same scenarios also run under plain `pytest` via
`tests/test_decision_scenarios.py`.

### Reading the output

Four labels, and the distinction matters:

| label | meaning |
|---|---|
| `INPUT` | a value **the harness fed in** |
| `CODE` | a value **read back out of the real implementation** |
| `NOTE` | the harness **explaining** something — proves nothing |
| `PASS`/`FAIL` | an assertion result |

Only `CODE` lines are evidence about the system. Example:

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

Read that as: one frame in, the scheduler chose it, ranging put the object
3 m ahead in the **robot** frame, the pose was valid so it transformed to
**field** (8, 3), and navigation decided to drive +X at 2 m/s. The
controller stayed disabled and neutral.

`-v` adds the pocket scoring table — the intermediate reasoning behind the
target choice:

```
  CODE  |   pocket (9.17,3.47) count=3 dist=4.19 blocked=False clearance=None score=+1.021
```

---

## 4. The scenarios

| scenario | asserts |
|---|---|
| `projection_round_trip` | the harness's placement math agrees with production ranging |
| `valid_fuel_ahead` | detect → range → field track → target → +X command |
| `no_detections` | zero translation, spin-search only |
| `low_confidence_rejected` | 0.20 conf dropped by the production gate |
| `low_confidence_boundary_accepted` | 0.45 conf accepted — it is a threshold, not a blanket |
| `stale_frame_refused` | 600 ms-old frame never reaches inference |
| `stale_result_dropped_after_inference` | inference ran, result too old → discarded |
| `out_of_order_frame_ignored` | a late, lower sequence number is ignored |
| `skipped_frames_counted` | newest frame processed, 2 skips counted, no queue |
| `detector_failure_no_fabrication` | no detections, tracks age out, safe navigation |
| `two_cameras_same_object_deduped` | both cameras, one field track |
| `two_cameras_different_objects_stay_separate` | distinct objects stay distinct |
| `object_positions_map_correctly` | four quadrants transform correctly |
| `rotated_robot_transforms_correctly` | 90° robot → object at field (5, 6) not (8, 3) |
| **`pose_invalid_never_translates`** | **the critical rule: vx = vy = 0 exactly** |
| `pose_invalid_fully_neutral_when_configured` | rotation suppressible too |
| `odometry_goes_stale_invalidates_pose` | frozen RIO timestamp → invalid → stop |
| `tag_fix_makes_pose_valid` | real `add_tag_estimate()` enables navigation |
| `red_alliance_blue_bumper_is_opponent` | friend/foe routing |
| `red_alliance_red_bumper_is_friendly` | teammate never an obstacle |
| `unknown_alliance_treats_robots_as_obstacles` | conservative with no FMS data |
| `dense_cluster_beats_isolated_piece` | density outweighs distance |
| `adjacent_objects_merge_within_radius` | **documents a finding — see §7** |
| `opponent_blocking_path_is_penalized` | blocked pocket flagged and penalized |
| `opponent_near_robot_pushes_command_sideways` | repulsion fires inside the safe radius |
| `contract_and_heartbeat_hold` | full NT contract, advancing heartbeat |

---

## 5. Interactive mode

```bash
python test_decisions.py --interactive
```

```
decision> fuel 3.0 0.0 0.95      # a fuel 3 m forward, 0 m left, 0.95 confidence
decision> blue 1.2 0.4           # a blue bumper 1.2 m forward, 0.4 m left
decision> list
decision> run                    # execute one cycle, print every decision
decision> pose                   # change pose mode / robot pose
decision> alliance               # red / blue / unknown
decision> detector               # toggle detector health
decision> save my_test.json      # freeze it as a reusable scenario
decision> quit
```

Coordinates are **robot-relative: +x forward, +y left, in metres.**

`save` writes a JSON scenario. Add an `expect` block to it and it becomes
a real test.

> Interactive runs assert only the safety invariants — they are for
> exploring, not for proving. Promote anything interesting to a scenario.

---

## 6. Adding your own tests

### From JSON (no Python)

```bash
python test_decisions.py --write-example-json my_tests.json
# edit my_tests.json
python test_decisions.py --json my_tests.json
```

```json
{
  "scenarios": [
    {
      "name": "my_first_test",
      "description": "A fuel 2.5 m ahead and 1 m left should be chased.",
      "pose_mode": "odometry_only",
      "robot_pose": [5.0, 3.0, 0.0],
      "is_red_alliance": true,
      "cameras": ["cam0"],
      "config": {"navigation.search_point_field": null},
      "cycles": [
        {
          "advance_s": 0.033,
          "note": "one fuel in view",
          "frames": [
            {
              "camera_id": "cam0", "seq": 1, "capture_offset_s": 0.0,
              "detections": [
                {"class_name": "fuel", "confidence": 0.9,
                 "x_robot_m": 2.5, "y_robot_m": 1.0}
              ]
            }
          ]
        }
      ],
      "expect": {
        "pose_valid": true,
        "fuel_tracks": 1,
        "fuel_track_near": [7.5, 4.0, 0.1],
        "translation_nonzero": true,
        "controller_enabled": false
      }
    }
  ]
}
```

Field notes:

- **`pose_mode`** — `odometry_only` (valid from odometry alone),
  `tag_initialized` (valid only after an injected tag fix), `invalid`
  (never becomes valid).
- **`capture_offset_s`** — negative means the frame was grabbed that long
  ago. `-0.6` stages a stale frame.
- **`x_robot_m` / `y_robot_m`** — robot-relative metres, projected to a
  pixel box through that camera's real intrinsics and mount. Use `bbox`
  instead to test geometry rejections directly.
- **`advance_s`** — clock advance before the cycle. Use large values to age
  tracks out.
- **`detector_healthy` / `freeze_odometry` / `inject_tag_pose`** — per-cycle
  fault and event injection.
- **`config`** — dotted overrides, e.g. `"navigation.max_speed_mps": 1.0`.
  An unknown field is rejected, not ignored.

`python test_decisions.py --help-expect` lists all 30 assertion keys.

### In Python

Append a `ScenarioSpec` to `builtin_scenarios()` in `test_decisions.py`.
It is picked up automatically by `--scenario all`, `--list` **and**
`pytest`.

### Two things the harness will refuse to let you get away with

1. **A scenario with no `expect` block fails.** A test that asserts
   nothing is not a test.
2. **A detection the camera physically cannot see fails the scenario**,
   with a `NOTE | WARNING:` line naming it. Silently dropping an
   out-of-view object is how a test ends up asserting the wrong thing —
   this caught three of my own mistakes while building this. Set
   `"allow_unstaged_detections": true` only when the placement is
   deliberate.

### Synthetic camera geometry

`cam0` looks straight ahead; `cam1` is yawed 30° left so the two overlap
and cross-camera de-duplication gets exercised. Both are 640×480,
fx = fy = 500, 0.6 m up, pitched 20° down — exact round numbers so
expected positions can be checked on paper. Horizontal half-FOV is
`atan(320/500) ≈ 32.6°`, which is why an object at `y = ±1.6 m` at 1.5 m
range is out of view.

**These are not your robot's numbers.** They are test fixtures. Real
bring-up fills in `frc_perception/config.py` from
`scripts/calibrate_intrinsics.py` and a tape measure.

---

## 7. Findings from building this

### Production finding: same-camera detections are merged

`world_model.merge_observations()` is documented as *cross-camera*
de-duplication, but it merges any two same-class observations within the
merge radius **regardless of which camera produced them**.

Observed: three fuels from one camera at robot-relative `(4.0, 0.8)`,
`(4.1, 0.5)`, `(4.3, 1.1)` produce **two** tracks. The first pair is
0.32 m apart, inside `merge_distance_game_piece_m = 0.40`.

Why it matters: two boxes from one camera are, by construction, two
different objects — YOLO's own NMS already removed duplicate boxes of the
same object. Adjacent game pieces are exactly the dense cluster the
navigation scoring is built to prefer, so the dedup stage under-counts the
thing the scoring rewards. The same applies to robots at
`merge_distance_robot_m = 0.90`, where collapsing two adjacent opponents
into one obstacle is a safety under-count.

**The production code is unchanged.** It is a genuine design trade-off
(merging same-camera detections does clean up a split detection of one
partially occluded object), and silently changing the semantics of the
system while building a harness to measure it would be the wrong order of
operations. `adjacent_objects_merge_within_radius` pins the current
behavior so a future change is a visible, deliberate decision.

The fix, if you want it: require `candidate.camera_id` not already present
in the group before merging, in `merge_observations()`. That keeps the
stated cross-camera intent and stops conflating distinct adjacent objects.
It would require updating `test_merging_is_transitive_across_a_chain` in
`tests/test_world_model.py`, which currently asserts the same-camera
behavior.

### Confirmed-correct behavior that looked wrong at first

- **A blocked pocket is still selected when it is the only one.** The
  penalty (score −6.76 with an opponent 0.35 m off the path) ranks options;
  it does not forbid the only option. Correct.
- **Avoidance repulsion only fires within the safe radius** (1.45 m from
  the *robot*, not from the path). An opponent on the path but 1.64 m away
  flags the pocket blocked without yet pushing the command sideways. That
  is the documented "repulsion field, not a planner" design: penalize now,
  steer when close.

---

## 8. What this does and does not prove

### Verified in software, offline

Alliance routing; the production confidence gate; ray-to-ground ranging and
its rejections; robot→field transforms including a rotated robot;
pose-validity gating (odometry-only, tag-initialized, stale-odometry,
never-initialized); frame freshness; post-inference result staleness;
out-of-order and skipped frames with no queue; cross-camera
de-duplication; track aging after detector death; pocket scoring and
density preference; blocked-path penalties; avoidance repulsion;
**zero translation whenever the pose is invalid**; controller neutrality;
the full NetworkTables payload contract; the heartbeat.

### Still requires real hardware

| needs | why the harness cannot cover it |
|---|---|
| **Camera intrinsics and mount pose** | the fixtures are invented round numbers. Every range and field position depends on *your* measured values — `scripts/calibrate_intrinsics.py` and a tape measure |
| **CUDA / TensorRT inference** | `--gpu-check` is the only GPU test here, and it has never passed: no torch or model in the build environment |
| **Any frame rate** | no number is claimed anywhere. Use `scripts/benchmark_detector.py` on the Jetson, warm |
| **USB camera behavior** | two OV9782 on one bus, MJPG negotiation, exposure, unplug/reconnect |
| **Real YOLO accuracy** | scripted detections are perfect by construction; a real model's false positives and misses are not modelled |
| **AprilTag detection and its frame conventions** | tag poses are injected directly; `scripts/verify_localization.py` on a real field is the only proof of the sign conventions |
| **roboRIO link** | NT timing, odometry publication rate, FMS alliance data |
| **Joystick polarity** | stage 16, robot on blocks |

The harness proves the **logic** is right given correct inputs. It cannot
prove the inputs are right. Work the bring-up order in README §14.

---

## 9. The one GPU-dependent test

```bash
python test_decisions.py --gpu-check
```

Separate on purpose, and the only thing in this file that needs the
Jetson. It loads the production `YoloDetector` with the configured
weights, confirms CUDA, checks the class contract, runs ten inferences on
a synthetic frame, and reports **measured** latency. It fails loudly when
torch, CUDA or the model file is missing rather than skipping quietly.

Everything else is deliberately GPU-free; `pytest` asserts that no
built-in scenario sets `requires_gpu`.
