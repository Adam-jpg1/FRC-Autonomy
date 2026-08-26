# `models/`

Trained YOLO weights live here. **They are not committed to git** (see
`.gitignore`) — they are large binaries that change often.

| file | what it is |
|---|---|
| `best.pt` | the weights the runtime loads (`config.detector.model_path`) |
| `best.engine` | optional TensorRT engine, built **on the Jetson** |
| `best.onnx` | optional intermediate export |

## Installing a model

```bash
# copy the file trained in Colab
cp ~/Downloads/best.pt models/best.pt

# verify it has exactly the three expected classes
python scripts/test_yolo_image.py --model models/best.pt --image <a real photo>
python run.py --check
```

The weights **must** contain exactly these class names:

```
fuel   bumper_blue   bumper_red
```

Routing is by name. A model with different names is not partially usable —
the mismatched classes are silently dropped at runtime.

## TensorRT (optional, later)

Do this only after `.pt` inference is working and you have measured the
frame rate. The engine is tied to the exact GPU, JetPack, and TensorRT
version of the machine that built it, so it **must** be built on the
Jetson itself and rebuilt after any JetPack upgrade.

```bash
yolo export model=models/best.pt format=engine half=True device=0
# then point config.detector.model_path at models/best.engine
```

Measure before and after with the `loop_fps` / `inference_ms` values the
system publishes to NetworkTables. Do not assume a speedup — measure it.
