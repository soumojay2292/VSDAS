"""
config.py — VSDAS global configuration.
Source: Part B, CELL 2 (Imports & global config).
All other modules import from here; no logic lives here.
"""

import warnings
warnings.filterwarnings("ignore")

# ── User-facing runtime settings ─────────────────────────────────────────────
INPUT_SOURCE     = 0               # 0 = webcam | "assets/videos/<file>.mp4" | YouTube URL
OUTPUT_VIDEO     = "outputs/vsdas_live.mp4"
EGO_SPEED_KPH    = 40.0           # your current speed (replace with OBD-II if available)
SPEED_LIMIT_KPH  = 60.0           # road speed limit
ENABLE_VOICE     = True            # spoken alerts via gTTS
MAX_FRAMES       = None            # None = run until video ends / Ctrl-C
DISPLAY_WIDTH    = 1280
DISPLAY_HEIGHT   = 720

# ── Detection model ───────────────────────────────────────────────────────────
YOLO_MODEL       = "assets/models/yolov8n.pt"   # yolov8s.pt for better accuracy on GPU
CONF_THRESH      = 0.40

# Decision-engine gate: classical road-hazard detections (pothole / speed
# breaker) below this confidence are ignored by SpeedAdvisor so weak
# false positives cannot trigger braking. Does NOT affect detection itself.
HAZARD_MIN_CONF  = 0.50

# ── COCO class id → label maps ────────────────────────────────────────────────
VEHICLES       = {2:"car", 3:"motorcycle", 5:"bus", 7:"truck", 1:"bicycle", 6:"train"}
ANIMALS        = {14:"bird", 15:"cat", 16:"dog", 17:"horse", 18:"sheep",
                  19:"cow", 20:"elephant", 21:"bear", 22:"zebra", 23:"giraffe"}
PEDESTRIANS    = {0:"person"}
TRAFFIC_CTRL   = {9:"traffic light", 11:"stop sign", 12:"parking meter"}
ALL_COCO_IDS   = {**VEHICLES, **ANIMALS, **PEDESTRIANS, **TRAFFIC_CTRL}