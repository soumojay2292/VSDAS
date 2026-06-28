"""
YOLOv8-COCO doesn't include potholes or speed breakers.

Production approach: fine-tune YOLOv8 on the IDD Pothole Dataset
  https://idd.insaan.iiit.ac.in/  (free academic download)
  Training code is provided in CELL 5b below.

For the demo we use a classical CV heuristic that works well enough
to demonstrate the concept to faculty:
  • Convert ROI to grayscale → Laplacian → threshold → contour filter
  • Speed breakers: detect horizontal bright band near horizon
"""

from typing import Dict, List

import cv2
import numpy as np


class RoadHazardDetector:
    """
    Lightweight classical CV detector for road-surface hazards.
    Replace with fine-tuned YOLOv8 for production.
    """

    # Cap the working resolution for the gradient stages. Wider frames
    # (e.g. 1920×1080) are downscaled to this width before Laplacian/Sobel,
    # then detections are scaled back to original coordinates. Bounds the
    # float buffers that caused ArrayMemoryError on full-res frames.
    PROC_WIDTH = 640

    def __init__(self, frame_w: int, frame_h: int):
        self.fw = frame_w
        self.fh = frame_h
        self.vanish_y = int(frame_h * 0.42)
        # Kalman-style smoothing for detections
        self.prev_potholes: List = []
        self.prev_breakers: List = []
        self.frame_idx = 0

    def _downscale(self, frame: np.ndarray):
        """Return (proc_frame, scale). `scale` maps proc-pixels → original
        pixels (>= 1). Aspect ratio preserved, so the same factor applies to
        x and y. No copy when the frame is already within PROC_WIDTH."""
        h, w = frame.shape[:2]
        if w <= self.PROC_WIDTH:
            return frame, 1.0
        s = self.PROC_WIDTH / float(w)
        proc = cv2.resize(frame, (self.PROC_WIDTH, int(round(h * s))),
                          interpolation=cv2.INTER_AREA)
        return proc, w / float(self.PROC_WIDTH)

    def _road_roi(self, frame: np.ndarray) -> np.ndarray:
        """Return the road surface trapezoid ROI, sized to the given frame."""
        h, w = frame.shape[:2]
        vy = int(h * 0.42)
        mask = np.zeros((h, w), dtype=np.uint8)
        pts  = np.array([
            [int(w*0.10), h],
            [int(w*0.90), h],
            [int(w*0.65), vy + 30],
            [int(w*0.35), vy + 30],
        ], dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)
        return mask

    def detect_potholes(self, frame: np.ndarray) -> List[Dict]:
        """
        Potholes: dark irregular depressions on road surface.
        Signal: high Laplacian energy + darker than surroundings.
        """
        # Run the gradient stage at a bounded resolution; map boxes back after.
        oh, ow = frame.shape[:2]
        proc, scale = self._downscale(frame)
        fh, fw = proc.shape[:2]
        a2 = scale * scale                       # area scales with the square

        gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
        mask = self._road_roi(proc)

        # Focus on lower 60% of road
        gray_road = cv2.bitwise_and(gray, gray, mask=mask)

        # Laplacian edge energy
        # CV_32F (not CV_64F) halves the buffer; convertScaleAbs does
        # |x| + clip[0,255] + uint8 cast in one C pass — identical result,
        # no float64 numpy temporaries (np.abs/np.clip/np.uint8 chain).
        lap = cv2.Laplacian(gray_road, cv2.CV_32F, ksize=5)
        lap_abs = cv2.convertScaleAbs(lap)

        # Threshold high-energy regions
        _, thr = cv2.threshold(lap_abs, 30, 255, cv2.THRESH_BINARY)
        thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)))

        contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        for c in contours:
            area = cv2.contourArea(c) * a2   # normalise to full-res area for filtering
            if area < 400 or area > 25000:   # filter by plausible pothole size
                continue
            x,y,w,h = cv2.boundingRect(c)
            # Must be in lower half of image
            if y + h < fh * 0.50:
                continue
            # Aspect ratio: potholes roughly circular/square
            if w > 0 and not (0.3 < h/w < 3.0):
                continue
            # Darkness check: pothole is darker than road avg
            roi_gray = gray[y:y+h, x:x+w]
            roi_mean = float(np.mean(roi_gray))
            road_mean = float(np.mean(gray[fh//2:, fw//4:3*fw//4]))
            if roi_mean > road_mean * 0.90:   # must be darker
                continue
            # Map proc-space box → original resolution, clamped to frame
            X1 = min(max(x * scale, 0.0), ow)
            Y1 = min(max(y * scale, 0.0), oh)
            X2 = min(max((x + w) * scale, 0.0), ow)
            Y2 = min(max((y + h) * scale, 0.0), oh)
            detections.append({
                "label": "pothole",
                "bbox": (float(X1), float(Y1), float(X2), float(Y2)),
                "conf": min(0.85, area/8000),
                "center": (int((x + w//2) * scale), int((y + h//2) * scale)),
            })

        # Temporal smoothing: merge with previous frame
        merged = self._temporal_smooth(detections, self.prev_potholes, iou_thr=0.3)
        self.prev_potholes = merged
        return merged[:4]   # max 4 pothole boxes per frame

    def detect_speed_breakers(self, frame: np.ndarray) -> List[Dict]:
        """
        Speed breakers: horizontal bright/dark band across road.
        Signal: horizontal Sobel + horizontal line Hough.
        """
        # Run the gradient stage at a bounded resolution; map boxes back after.
        oh, ow = frame.shape[:2]
        proc, scale = self._downscale(frame)
        fh, fw = proc.shape[:2]

        gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
        # Look in bottom 55% of frame
        roi_y0 = int(fh * 0.45)
        roi    = gray[roi_y0:, :]

        # CV_32F + convertScaleAbs: one C pass, no float64 numpy temporaries.
        sobel_y = cv2.Sobel(roi, cv2.CV_32F, 0, 1, ksize=5)
        sobel_y = cv2.convertScaleAbs(sobel_y)
        _, thr  = cv2.threshold(sobel_y, 40, 255, cv2.THRESH_BINARY)

        # Horizontal line detection (kernel length scaled to proc resolution)
        klen   = max(1, int(round(60 / scale)))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (klen,1))
        hlines = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(hlines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        for c in contours:
            x,y,w,h = cv2.boundingRect(c)
            if w < fw * 0.20:        # must span at least 20% of width
                continue
            if h > 30 / scale:       # thin horizontal bands (proc-space threshold)
                continue
            real_y = y + roi_y0
            # Map proc-space box → original resolution, clamped to frame
            X1 = min(max(x * scale, 0.0), ow)
            Y1 = min(max(real_y * scale, 0.0), oh)
            X2 = min(max((x + w) * scale, 0.0), ow)
            Y2 = min(max((real_y + max(h, 8)) * scale, 0.0), oh)
            detections.append({
                "label": "speed_breaker",
                "bbox": (float(X1), float(Y1), float(X2), float(Y2)),
                "conf": min(0.80, w/(fw*0.5)),
                "center": (int((x + w//2) * scale), int((real_y + 4) * scale)),
            })

        merged = self._temporal_smooth(detections, self.prev_breakers, iou_thr=0.4)
        self.prev_breakers = merged
        self.frame_idx += 1
        return merged[:2]

    @staticmethod
    def _iou(b1, b2) -> float:
        x1 = max(b1[0],b2[0]); y1 = max(b1[1],b2[1])
        x2 = min(b1[2],b2[2]); y2 = min(b1[3],b2[3])
        inter = max(0,x2-x1)*max(0,y2-y1)
        a1 = (b1[2]-b1[0])*(b1[3]-b1[1])
        a2 = (b2[2]-b2[0])*(b2[3]-b2[1])
        return inter / max(a1+a2-inter, 1e-6)

    def _temporal_smooth(self, cur, prev, iou_thr=0.3) -> List:
        """Keep previous detections that overlap current ones (stability)."""
        merged = list(cur)
        for p in prev:
            if not any(self._iou(p["bbox"], c["bbox"]) > iou_thr for c in cur):
                p = dict(p);  p["conf"] *= 0.7   # decay confidence
                if p["conf"] > 0.25:
                    merged.append(p)
        return merged


# ════════════════════════════════════════════════════════════════════════════
# CELL 5b  ▸  Fine-tuning YOLOv8 on pothole dataset  (run once, not in demo)
# ════════════════════════════════════════════════════════════════════════════
FINETUNE_CODE = '''
# ── Download IDD Pothole dataset ──────────────────────────────────────────
# Register at https://idd.insaan.iiit.ac.in/ and download the pothole subset.
# Place images in /content/pothole_dataset/images/{train,val}
# Place YOLO-format labels in    /content/pothole_dataset/labels/{train,val}

# ── dataset.yaml ─────────────────────────────────────────────────────────
import yaml
ds_cfg = {
    "path": "/content/pothole_dataset",
    "train": "images/train",
    "val":   "images/val",
    "nc": 2,
    "names": ["pothole", "speed_breaker"]
}
with open("pothole_dataset.yaml","w") as f:
    yaml.dump(ds_cfg, f)

# ── Fine-tune ─────────────────────────────────────────────────────────────
from ultralytics import YOLO
model = YOLO("yolov8s.pt")          # start from COCO pretrained
results = model.train(
    data    = "pothole_dataset.yaml",
    epochs  = 50,
    imgsz   = 640,
    batch   = 16,
    name    = "vsdas_pothole",
    patience= 10,
    device  = 0,   # GPU
)
# Best weights saved to: runs/detect/vsdas_pothole/weights/best.pt
# Set POTHOLE_MODEL = "runs/detect/vsdas_pothole/weights/best.pt" in CELL 2
'''
# This cell is documentation — not executed in the demo run.

