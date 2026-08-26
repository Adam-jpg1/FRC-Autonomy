# Training the FRC 2026 detection model

The runtime routes every detection by **class name**. These three
names are a hard contract with `frc_perception/config.py`:

```
fuel
bumper_blue
bumper_red
```

Rename one and the runtime silently ignores that class forever. Do not
rename them, do not add classes the runtime does not use, and keep the
index order in `data.yaml` stable across retrains so old label files stay
valid.

| class | meaning |
|---|---|
| `fuel` | the 2026 game piece |
| `bumper_blue` | the blue bumper region of a robot |
| `bumper_red` | the red bumper region of a robot |

Alliance logic happens at runtime, not in the dataset: label the bumper
color you actually see and let `frc_perception/alliance.py` decide which
color is an opponent.

---

## 1. Dataset layout

Ultralytics requires exactly this structure. `labels/` mirrors `images/`
file-for-file, with `.txt` instead of the image extension.

```
dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

Label format, one line per object, coordinates **normalized to 0..1**
relative to image width/height:

```
<class_index> <x_center> <y_center> <width> <height>
```

```
0 0.4531 0.7120 0.0562 0.0900     # fuel
1 0.2010 0.6640 0.1800 0.0700     # bumper_blue
```

A rough starting point: **70% train / 20% val / 10% test**, split by
*scene* rather than by frame. Frames pulled from the same video clip are
nearly identical; putting some in train and some in val makes your
validation score a lie.

Aim for at least a few hundred instances per class, from more than one
venue and more than one lighting condition. Two thousand near-identical
frames from one practice field are worth less than three hundred varied
ones.

---

## 2. Labeling rules

Consistency matters more than perfection. An inconsistent dataset teaches
the model that the boundary is arbitrary, and it will produce boxes whose
bottom edge wanders — which directly corrupts range, because ranging uses
the **bottom edge** as the object's ground contact point.

### `fuel`

- Box the **visible extent** of the game piece, tight, no padding.
- The bottom edge is the most important edge in the dataset. It is what
  the range estimate is computed from. If the piece rests on the carpet,
  the bottom edge is where it meets the carpet.
- Do not "complete" a piece that is partly hidden — box what you can see.

### `bumper_blue` / `bumper_red`

- Box the **bumper region**, not the whole robot. The bumper is the part
  whose color identifies the alliance and whose bottom edge is a known
  height above the floor.
- One box per visible bumper face. A robot showing a corner presents two
  faces; a single box around both is acceptable if they are contiguous in
  the image, but be consistent about which you do.
- Never label a bumper you cannot determine the color of. Skip it, or make
  that image a background image.

### Hard cases — decide once, then be consistent

| situation | rule |
|---|---|
| **Partial visibility** | Label if roughly a third or more is visible and the class is unambiguous. Box only the visible part. |
| **Occlusion** | Box the visible portion. Do not guess the hidden extent. If a ball is split into two visible pieces by a bumper, one box around the visible whole. |
| **Cut off by the image edge** | Label it, boxed to the frame edge. The runtime rejects detections whose bottom edge touches the frame (`usable_bottom_max_frac`), so they cost nothing and their presence teaches the detector not to ignore edge objects. |
| **Multiple robots** | One box per robot bumper, even when robots overlap. Do not merge two robots into one box. |
| **Distant / tiny objects** | Label down to roughly 10–12 px. Below that, skip — sub-10 px boxes mostly teach noise, and the runtime's `max_range_m` throws them out anyway. |
| **Motion blur** | Label if the class is still recognizable to you. If you cannot tell, neither can the model. The OV9782 is global shutter, so heavy blur means the exposure is too long — fix that instead. |
| **Ambiguous color** | Skip the object entirely. A `bumper_red` labeled on a purple-looking bumper under stage lighting teaches the model to confuse alliances, which is the single worst failure this system can have. |
| **Reflections / screens** | Do not label a ball on a video screen or a reflection in the glass. Include such images as background images. |
| **Field elements that resemble fuel** | Do not label. Include them as background images so the model learns the difference. |

### Background images

Images with **no** objects of interest are valuable — they suppress false
positives. To use one, create an **empty `.txt` file** for it.

> An image with **no label file at all** is *skipped entirely* by YOLO. It
> is not a negative example. `validate_yolo.py` flags this because it is
> the most common silent dataset bug.

Aim for roughly 5–15% background images: empty carpet, the pit, crowds,
other field elements, red and blue objects that are not bumpers.

---

## 3. Train in Google Colab

Set **Runtime → Change runtime type → GPU** first.

```python
# Cell 1
!pip install -q ultralytics

# Cell 2 (optional — if the dataset lives in Drive)
from google.colab import drive
drive.mount('/content/drive')

# Cell 3
!git clone https://github.com/<your-team>/<this-repo>.git
%run <this-repo>/training/train_yolo_colab.py

# Cell 4 — the whole pipeline
run_all(dataset_zip='/content/drive/MyDrive/frc2026_dataset.zip',
        epochs=100,
        drive_dir='/content/drive/MyDrive/frc2026_models')
```

`run_all` performs, in order: install → extract → **validate the dataset**
→ write `data.yaml` → visualize sample annotations → train → validate →
confusion matrix and PR curves → verify class names → sample predictions
→ export `best.pt`.

Run the steps individually if you want to look at each one:

```python
root      = extract_dataset('/content/dataset.zip')
validate_dataset(root)          # raises on structural problems
visualize_samples(root)         # LOOK AT THIS. It catches half of all bugs.
data_yaml = write_data_yaml(root)
train(data_yaml, epochs=100, model='yolov8n.pt')
weights   = find_best_weights()
validate_model(weights, data_yaml)
verify_model_classes(weights)   # must print OK
predict_samples(weights, root)
export_best(weights, drive_dir='/content/drive/MyDrive/frc2026_models')
```

**Colab runtimes are deleted without warning.** Copy `best.pt` to Drive
*and* download it.

### Which model size

Start with `yolov8n`. Move up only after you have **measured** actual FPS
on the actual Raspberry Pi 5 + Hailo-8 with
`scripts/benchmark_detector.py` and found headroom. No frame rate is
assumed anywhere in this project.

Ultralytics' Hailo export supports YOLOv8, YOLO11 and YOLO26 detection
models. If you train something else, it will not compile.

### Reading the results

- **mAP50** is the honest headline number for this task.
- Per-class recall matters more than overall mAP: a model that finds every
  ball but half the bumpers is dangerous, because bumpers are obstacles.
- The **confusion matrix** is where `bumper_blue` ↔ `bumper_red` confusion
  shows up. Any meaningful confusion between those two classes must be
  fixed before the model goes on a robot — it maps directly onto driving
  into a teammate or through an opponent.

---

## 4. Deploy to the Hailo-8

Training and deployment are **separate stages**. You do not retrain in
order to re-export.

```
TRAINING                    DEPLOYMENT
--------                    ----------
dataset -> best.pt          best.pt -> best_hailo_model/
(Colab, GPU)                (x86_64 Linux + Hailo Dataflow Compiler)
```

### Compilation does NOT happen on the Raspberry Pi

The Hailo Dataflow Compiler is x86_64 Linux only, and Ultralytics enforces
it:

```python
assert LINUX and not ARM64, "Hailo export is only supported on Linux x86_64."
```

The Pi only needs HailoRT (`sudo apt install hailo-all`) to *run* an
already-compiled model.

### Exporting

On an x86_64 Linux machine with the Dataflow Compiler installed (a Colab
CPU runtime works):

```bash
python training/export_hailo.py --check           # can this machine compile?
python training/export_hailo.py \
    --model models/best.pt \
    --data training/data.yaml \
    --arch hailo8 \
    --imgsz 640
```

* `--arch hailo8` — the AI HAT+ 26 TOPS is a Hailo-8; the 13 TOPS board is
  `hailo8l`, and the binaries are not interchangeable.
* `--data` — INT8 calibration uses **our own FRC images**. The Hailo-8 runs
  INT8; Ultralytics forces `quantize=8` for this format and refuses FP32.
  A generic calibration set gives a model calibrated for the wrong scene.
* `--conf` defaults low (0.05) because for detection exports the confidence
  is **baked into on-chip NMS**. The accelerator never returns a box below
  it, so the runtime threshold can only filter further.

The output is a **directory**:

```
best_hailo_model/
├── best.hef
├── metadata.yaml       <- class names live here
└── nms_config.json
```

Deploy all of it. Copying only the `.hef` produces a model with no class
names, and this project routes by name — it would load and detect nothing.

### Quantization changes accuracy — measure it

```bash
yolo val model=models/best.pt data=training/data.yaml imgsz=640            # baseline
yolo val model=models/best_hailo_model data=training/data.yaml imgsz=640   # deployed
```

Record both `mAP50` numbers in your build log. INT8 loss falls hardest on
small objects — for us that is distant `fuel`. If recall drops
meaningfully, use more (and more representative) calibration images before
changing anything else. Never quote the `.pt` number as if it were the
deployed model's.

---

## 5. Validate before deploying

Locally, without Colab:

```bash
python training/validate_yolo.py --dataset ./dataset
python training/validate_yolo.py --model models/best.pt --data training/data.yaml
```

Then on the Raspberry Pi:

```bash
# the deployed artifact
python scripts/test_hailo.py --model models/best_hailo_model --image sample.jpg
python scripts/benchmark_detector.py --images data/samples/
python run.py --check          # confirms the class names match the runtime
```

On a development machine, the `.pt` can be checked directly:

```bash
python scripts/test_yolo_image.py --model models/best.pt --image data/samples/
```

`run.py --check` reports a class mismatch as FAIL for both the `.pt` and
the compiled Hailo artifact. Fix that before anything else — every
downstream layer routes by class name.
