# VSDAS — Vision-Based Smart Driver Assistance System

A real-time, monocular Advanced Driver Assistance System (ADAS) that watches a single
camera or dashcam feed and continuously recommends a safe driving speed. VSDAS detects
vehicles, pedestrians, animals, traffic lights, stop signs, potholes, and speed breakers,
estimates their distance and relative speed, computes time-to-collision (TTC), and issues
a live on-screen HUD plus optional spoken alerts.

The pipeline runs entirely on a single RGB stream — no LiDAR, radar, or stereo rig
required — making it suitable for low-cost dashcams and academic demonstration.

```
Camera / Video → YOLOv8 → DeepSORT → Road-Hazard Detector
        → Traffic-Light Classifier → Speed Estimator
        → Speed Advisor → Annotated Frame + Voice Alert
```

---

## Overview

VSDAS fuses deep-learning object detection with classical computer-vision heuristics and a
rule-based decision engine:

- **Detection** — YOLOv8 (COCO) finds vehicles, people, animals, and traffic-control objects.
- **Tracking** — DeepSORT assigns stable IDs across frames for per-object speed estimation.
- **Road hazards** — a classical CV detector flags potholes (Laplacian energy + darkness) and
  speed breakers (horizontal Sobel bands) that COCO models cannot see.
- **Distance** — a pin-hole camera model (`D = f·H / h_px`) converts box height to metres.
- **Speed & TTC** — per-track pixel displacement is scaled to real-world speed; TTC derives from
  closing speed and distance.
- **Decision** — a `SpeedAdvisor` aggregates all active hazards by priority and proximity, then
  emits a single recommended speed and action label with temporal smoothing.
- **Output** — an annotated HUD (boxes, distances, TTC arcs, decision panel) and throttled
  gTTS voice alerts.

---

## Features

- 🚗 Real-time multi-class object detection (YOLOv8)
- 🎯 Persistent multi-object tracking (DeepSORT)
- 📏 Monocular distance estimation (pin-hole model, per-class real heights)
- ⏱️ Relative speed and time-to-collision (TTC) estimation
- 🕳️ Classical pothole and speed-breaker detection with temporal smoothing
- 🚦 Traffic-light state classification (red / yellow / green) via HSV zoning
- 🧠 Priority- and proximity-aware speed advisory with smoothing
- 🖥️ Annotated HUD: hazard legend, decision panel, speed-limit sign, TTC arcs
- 🔊 Throttled spoken alerts (gTTS, inline audio in Colab/Jupyter)
- 📊 Telemetry logging and matplotlib analysis plots

---

## Folder Structure

```
VSDAS/
├── main.py                 # VSDASRealTime pipeline + CLI entry point
├── requirements.txt        # runtime dependencies
├── requirements-lock.txt   # fully pinned environment snapshot
├── README.md
├── assets/
│   ├── models/             # YOLOv8 weights (yolov8n.pt)
│   └── videos/             # input videos
├── docs/                   # documentation assets
├── outputs/                # generated annotated videos
├── notebooks/
│   └── VSVE_DL (2).ipynb   # original research notebook (source of all modules)
└── src/
    ├── __init__.py
    ├── config.py           # global settings + COCO class maps
    ├── utils.py            # distance estimation, IoU, distance bands, helpers
    ├── detector.py         # TrafficLightClassifier (HSV state classifier)
    ├── hazard_detector.py  # RoadHazardDetector (potholes / speed breakers)
    ├── speed_estimator.py  # VehicleSpeedEstimator (speed + TTC)
    ├── advisor.py          # Hazard dataclass + SpeedAdvisor decision engine
    ├── alert_system.py     # VoiceAlert (gTTS, throttled)
    └── visualizer.py       # draw_hazard_boxes + draw_hud renderer
```

---

## Requirements

- Python 3.10+
- A CUDA-capable GPU is recommended for real-time FPS (CPU works but slower)
- YOLOv8 weights (`yolov8n.pt`) — downloaded automatically on first run by `ultralytics`

Core dependencies (see `requirements.txt`):

| Package | Purpose |
|---|---|
| `ultralytics` | YOLOv8 detection |
| `torch`, `torchvision` | deep-learning backend |
| `deep-sort-realtime` | multi-object tracking |
| `opencv-python` | image / video processing |
| `numpy`, `scipy`, `pandas` | numerics and analysis |
| `matplotlib`, `seaborn` | telemetry plots |
| `gTTS` | text-to-speech alerts |
| `ipython` | inline display in Colab / Jupyter |
| `pillow`, `PyYAML`, `tqdm`, `yt-dlp`, `requests` | I/O and utilities |

For an exact, reproducible environment, use `requirements-lock.txt`.

---

## Installation

```bash
# 1. Clone
git clone <your-repo-url> VSDAS
cd VSDAS

# 2. Create and activate a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
# or, for the exact pinned environment:
pip install -r requirements-lock.txt
```

---

## Usage

Configure the run in `src/config.py`:

```python
INPUT_SOURCE    = 0          # 0 = webcam | "assets/videos/<file>.mp4" | YouTube URL
OUTPUT_VIDEO    = "outputs/vsdas_live.mp4"
EGO_SPEED_KPH   = 40.0       # your current speed (or feed from OBD-II)
SPEED_LIMIT_KPH = 60.0       # road speed limit
ENABLE_VOICE    = True       # spoken alerts via gTTS
YOLO_MODEL      = "assets/models/yolov8n.pt"   # yolov8s.pt for better GPU accuracy
CONF_THRESH     = 0.40
```

Run the pipeline:

```bash
python main.py
```

This processes `INPUT_SOURCE` and writes the annotated result to `OUTPUT_VIDEO`.

Programmatic use:

```python
from main import VSDASRealTime

vsdas = VSDASRealTime(speed_limit=60.0, ego_speed=40.0, enable_voice=True)
vsdas.run_video("dashcam.mp4", output_path="out.mp4", display_in_colab=False)
vsdas.plot_telemetry()   # save telemetry analysis figure
```

In Colab / Jupyter, `display_in_colab=True` renders the output video inline, and
`run_live()` captures the webcam for a live preview.

---

## Results

VSDAS produces an annotated video stream with:

- Per-object bounding boxes coloured by hazard class, with label, distance, and speed.
- TTC arcs on closing objects (red under 3 s).
- A decision panel showing the current action (`STOP`, `BRAKE HARD`, `SLOW DOWN`,
  `REDUCE SPD`, `MAINTAIN`, `ACCELERATE`, `CRUISE`), recommended speed, and the binding reason.
- A speed-limit sign and live FPS / frame counter.

Telemetry is logged per frame and `plot_telemetry()` saves `vsdas_telemetry.png` with:
decision distribution, recommended-speed timeline, processing FPS, and active hazards per frame.

> Real-time performance depends on hardware and YOLO model size. `yolov8n` targets real-time
> on GPU; `yolov8s` trades speed for accuracy.

---

## Future Work

- Fine-tune YOLOv8 on a dedicated pothole / speed-breaker dataset (e.g. IDD) to replace the
  classical road-hazard heuristics. Training scaffolding is documented in `src/hazard_detector.py`.
- Integrate OBD-II for true ego-speed instead of a configured constant.
- Add camera calibration (checkerboard) to replace the assumed focal length and improve
  distance accuracy.
- Lane detection and lane-departure warnings.
- Cross-platform native audio playback (the current voice path is optimised for Colab/Jupyter).
- Quantization / TensorRT export for embedded deployment.

---

## Acknowledgements

Built on [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics),
[deep-sort-realtime](https://github.com/levan92/deep_sort_realtime), OpenCV, and gTTS.
