# Lettuce Health Detection (YOLOv8)

Real-time detection of lettuce health (`healthy` vs `not healthy`) using a YOLOv8 model trained on the annotated dataset in this project.

## Project structure

```
lettuce yolo/
  .venv/                                              Python 3.12 virtual environment (CUDA torch + Ultralytics)
  dataset/                                            Merged YOLO dataset (images/labels, train/val split, data.yaml)
  runs/detect/runs/lettuce_health_v1/weights/best.pt  Trained model weights
  scripts/
    prepare_dataset.py                                Rebuilds dataset/ from the raw annotated source folders
    realtime_detect.py                                Real-time detection from webcam / video / image
  README.md
```

## Requirements

Already installed in `.venv` — no setup needed unless the environment is recreated:

- Python 3.12
- `torch` + `torchvision` (CUDA 12.1 build — uses your NVIDIA GPU automatically, falls back to CPU if unavailable)
- `ultralytics` (YOLOv8)
- `opencv-python`

To recreate the environment from scratch:

```powershell
cd "D:\Lettuce\lettuce yolo"
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
.\.venv\Scripts\python.exe -m pip install ultralytics
```

## Running real-time detection

No activation needed — just call the venv's Python directly:

```powershell
cd "D:\Lettuce\lettuce yolo"
.\.venv\Scripts\python.exe scripts\realtime_detect.py
```

This opens a live window from your default webcam (index `0`) with bounding boxes drawn per detection:

- **Green** = healthy
- **Red** = not healthy

An FPS counter is shown top-left. Press **`q`** to close the window.

### Options

| Flag        | Default                                              | Description                                             |
| ----------- | ---------------------------------------------------- | ------------------------------------------------------- |
| `--source`  | `0`                                                  | Webcam index, path to a video file, or path to an image |
| `--weights` | `runs/detect/runs/lettuce_health_v1/weights/best.pt` | Path to trained `.pt` weights                           |
| `--conf`    | `0.4`                                                | Minimum confidence to display a detection               |
| `--imgsz`   | `640`                                                | Inference resolution                                    |
| `--device`  | auto                                                 | `cpu`, `0` (first GPU), etc.                            |
| `--save`    | off                                                  | Save the annotated output to `runs/realtime/output.mp4` |

### Examples

```powershell
# Use a second camera
.\.venv\Scripts\python.exe scripts\realtime_detect.py --source 0

# Run on a recorded video and save the annotated result
.\.venv\Scripts\python.exe scripts\realtime_detect.py --source "C:\path\to\video.mp4" --save

# Lower the threshold if detections are being missed
.\.venv\Scripts\python.exe scripts\realtime_detect.py --conf 0.25

# Force CPU inference
.\.venv\Scripts\python.exe scripts\realtime_detect.py --device cpu
```

## Troubleshooting

- **"Could not open video source"** — try a different `--source` index (`0`, `1`, `2`...), and make sure no other app (Zoom, Teams, camera app) is holding the webcam.
- **No detections showing** — lower `--conf` (e.g. `0.25`), make sure the lettuce fills a reasonable portion of the frame, and keep lighting/framing similar to the training images (top-down view, plain background).
- **Slow/choppy FPS** — confirm the GPU is being used:
  ```powershell
  .\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"
  ```
  Should print `True`. If it prints `False`, the CUDA build of torch may need reinstalling.

## Retraining / updating the model

If the dataset or classes change (e.g. adding early/severe leaf-spot severity classes later):

```powershell
.\.venv\Scripts\python.exe scripts\prepare_dataset.py
```

then retrain and point `realtime_detect.py --weights` at the new `best.pt`. The detection script works unchanged with any number of classes.
