# `models/`

Model artifacts. **Not committed to git** (see `.gitignore`) — they are
large binaries that change often.

| path | what it is | needed at runtime? |
|---|---|---|
| `best_hailo_model/` | the deployed Hailo-8 artifact **directory** | **yes**, on the Pi |
| `best_hailo_model/best.hef` | the compiled network | yes |
| `best_hailo_model/metadata.yaml` | class names, imgsz, `hailo_arch` | **yes — this is where class names come from** |
| `best_hailo_model/nms_config.json` | on-chip NMS parameters baked at export | yes |
| `best.pt` | trained PyTorch weights | no — training output and export source |

## The directory is the artifact

```bash
rsync -av best_hailo_model/ pi@<robot-pi>:~/FRC/models/best_hailo_model/
```

Copying **only the `.hef`** is the most likely deployment mistake. Ultralytics
reads class names from the `metadata.yaml` sidecar; without it the loaded
model has no names, and this project routes detections by name — so the
model loads cleanly and then detects nothing. Both `run.py --check` and
`scripts/test_hailo.py` check for this before anything else.

The artifact must contain exactly these class names:

```
fuel   bumper_blue   bumper_red
```

## Verifying an installed model

```bash
# no accelerator needed: files, class names, architecture
python scripts/test_hailo.py --model models/best_hailo_model --artifact-only

# on the Pi, with the AI HAT+ attached
python scripts/test_hailo.py --model models/best_hailo_model --image sample.jpg
python scripts/benchmark_detector.py --images data/samples/
python run.py --check
```

## Architecture matters

The Raspberry Pi AI HAT+ **26 TOPS** carries a **Hailo-8**. The 13 TOPS
board carries a **Hailo-8L**. A binary compiled for one does not run on the
other. `metadata.yaml` records `hailo_arch`, and preflight warns when it
disagrees with `detector.hailo_arch` in the config.

## Where models come from

Compilation happens on an **x86_64 Linux** machine — never on the Pi (the
Hailo Dataflow Compiler is x86_64-only, and Ultralytics asserts it):

```bash
python training/export_hailo.py --check          # is this machine able to compile?
python training/export_hailo.py --model models/best.pt --arch hailo8
```

See `training/README.md` for the full training → export → deploy flow.
