import argparse
import base64
import os
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

try:                                  # Colab/Jupyter display helpers (optional)
    from IPython.display import HTML, display, clear_output
except Exception:                     # not in a notebook
    HTML = display = clear_output = None

from src.config import (
    YOLO_MODEL, CONF_THRESH, SPEED_LIMIT_KPH, EGO_SPEED_KPH, ENABLE_VOICE,
    OUTPUT_VIDEO, INPUT_SOURCE, MAX_FRAMES, DISPLAY_WIDTH, DISPLAY_HEIGHT,
    ALL_COCO_IDS, VEHICLES, PEDESTRIANS,
)
from src.advisor import Hazard, SpeedAdvisor
from src.alert_system import VoiceAlert
from src.detector import TrafficLightClassifier
from src.hazard_detector import RoadHazardDetector
from src.speed_estimator import VehicleSpeedEstimator
from src.utils import estimate_distance, _iou, _col_to_css
from src.visualizer import draw_hazard_boxes, draw_hud


class VSDASRealTime:
    """
    Full real-time ADAS pipeline:
      Camera/Video → YOLOv8 → DeepSORT → RoadHazardDetector
              → TrafficLightClassifier → VehicleSpeedEstimator
              → SpeedAdvisor → Annotated Frame + Voice Alert
    """

    def __init__(self,
                 model_path:   str   = YOLO_MODEL,
                 speed_limit:  float = SPEED_LIMIT_KPH,
                 ego_speed:    float = EGO_SPEED_KPH,
                 enable_voice: bool  = ENABLE_VOICE):

        # Fall back to bare name so ultralytics auto-downloads if the
        # bundled weight is absent (e.g. fresh clone without the binary).
        if not os.path.exists(model_path):
            model_path = os.path.basename(model_path)
        print(f"[VSDAS] Loading YOLOv8: {model_path}")
        self.detector   = YOLO(model_path)
        # DeepSort() builds the embedder and runs a warmup forward in its
        # constructor; that forward is not under no_grad in the vendor code,
        # so it allocates autograd buffers → CPU OOM at init. Build under
        # no_grad to suppress the graph (inference only — tracking unaffected).
        with torch.no_grad():
            self.tracker = DeepSort(max_age=25, n_init=2,
                                    embedder="mobilenet", half=True)
        self.road_haz   = None   # initialised on first frame
        self.tl_clf     = TrafficLightClassifier()
        self.spd_est    = VehicleSpeedEstimator()
        self.advisor    = SpeedAdvisor(speed_limit)
        self.voice      = VoiceAlert() if enable_voice else None

        self.ego_speed  = ego_speed
        self.speed_limit= speed_limit
        self.frame_idx  = 0
        self.fps_smooth = 30.0
        self._t_prev    = time.perf_counter()

        # Telemetry log for paper analysis
        self.telemetry: List[Dict] = []
        print("[VSDAS] System ready.")

    # ── Process one frame ─────────────────────────────────────────────────
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        t0 = time.perf_counter()
        fh, fw = frame.shape[:2]

        # Lazy init road hazard detector (needs frame dims)
        if self.road_haz is None:
            self.road_haz = RoadHazardDetector(fw, fh)

        # ── 1. YOLOv8 detection ───────────────────────────────────────────
        results = self.detector(
            frame,
            conf    = CONF_THRESH,
            iou     = 0.45,
            classes = list(ALL_COCO_IDS.keys()),
            verbose = False,
        )

        raw_dets = []   # for DeepSORT: [(ltwh, conf, class_name), ...]
        all_boxes: List[Dict] = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                if cls_id not in ALL_COCO_IDS:
                    continue
                x1,y1,x2,y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                lbl  = ALL_COCO_IDS[cls_id]
                raw_dets.append(([x1,y1,x2-x1,y2-y1], conf, lbl))
                all_boxes.append({"bbox":(x1,y1,x2,y2),"label":lbl,"conf":conf})

        # ── 2. Classical road hazard detection ───────────────────────────
        potholes  = self.road_haz.detect_potholes(frame)
        breakers  = self.road_haz.detect_speed_breakers(frame)
        road_dets = potholes + breakers

        # ── 3. DeepSORT tracking (vehicles+pedestrians only) ──────────────
        track_dets = [d for d in raw_dets
                      if d[2] in {**VEHICLES,**PEDESTRIANS}.values()]
        # no_grad: DeepSORT's MobileNetV2 embedder runs forward() without it,
        # so autograd buffers accumulate per frame → CPU OOM. Suppress here.
        with torch.no_grad():
            tracks = self.tracker.update_tracks(track_dets, frame=frame)
        confirmed = {t.track_id: t for t in tracks if t.is_confirmed()}

        # ── 4. Build Hazard list ──────────────────────────────────────────
        hazards: List[Hazard] = []
        vanish_y = int(fh * 0.42)

        # From YOLO detections
        for det in all_boxes:
            lbl  = det["label"]
            bbox = det["bbox"]
            dist = estimate_distance(lbl, bbox, fh, vanish_y)

            # Traffic light state
            tl_state = None
            if lbl == "traffic light":
                tl_state = self.tl_clf.classify(frame, bbox)

            # Find matching track
            track_id = None
            spd_kph  = None
            ttc_val  = None
            for tid, tr in confirmed.items():
                tb = tr.to_ltrb()
                iou_val = _iou(bbox, tb)
                if iou_val > 0.35:
                    track_id = tid
                    spd_kph  = self.spd_est.update(tid, tb, dist, self.frame_idx)
                    ttc_val  = self.spd_est.ttc(self.ego_speed, spd_kph, dist)
                    break

            hazards.append(Hazard(
                label=lbl, bbox=bbox, dist_m=dist, conf=det["conf"],
                track_id=track_id, tl_state=tl_state,
                speed_kph=spd_kph, ttc=ttc_val
            ))

        # From road hazard detector
        for rd in road_dets:
            bbox = rd["bbox"]
            dist = estimate_distance(rd["label"], bbox, fh, vanish_y)
            hazards.append(Hazard(
                label=rd["label"], bbox=bbox,
                dist_m=dist, conf=rd["conf"]
            ))

        # Sort by distance (closest first)
        hazards.sort(key=lambda h: h.dist_m)

        # ── 5. Speed recommendation ───────────────────────────────────────
        decision = self.advisor.recommend(hazards, self.ego_speed)

        # ── 6. Voice alert ────────────────────────────────────────────────
        if self.voice:
            self.voice.alert_for_decision(
                decision["action"],
                decision["reason"],
                decision["rec_speed"]
            )

        # ── 7. Annotate frame ─────────────────────────────────────────────
        # In-place annotation: cap.read() yields a fresh owned frame each
        # iteration and the caller does not reuse the original, so no copy.
        out = frame
        urgent_id = decision["urgent_hazard"].track_id \
                    if decision["urgent_hazard"] else None
        draw_hazard_boxes(out, hazards, urgent_id)
        draw_hud(out, decision["action"], decision["rec_speed"],
                 self.ego_speed, decision["reason"],
                 hazards, self.fps_smooth, self.frame_idx,
                 self.speed_limit)

        # ── 8. FPS ────────────────────────────────────────────────────────
        dt = time.perf_counter() - t0
        self.fps_smooth = 0.9*self.fps_smooth + 0.1*(1/max(dt,1e-5))
        self.frame_idx += 1

        # ── 9. Log ────────────────────────────────────────────────────────
        self.telemetry.append({
            "frame":      self.frame_idx,
            "action":     decision["action"],
            "rec_speed":  decision["rec_speed"],
            "ego_speed":  self.ego_speed,
            "n_hazards":  len(hazards),
            "hazards":    [h.label for h in hazards],
            "fps":        self.fps_smooth,
        })

        return out, decision

    # ── Run on video file ─────────────────────────────────────────────────
    def run_video(self,
                  source,
                  output_path: str = OUTPUT_VIDEO,
                  max_frames:  Optional[int] = None,
                  display_in_colab: bool = True):

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open: {source}")

        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)  or DISPLAY_WIDTH)
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or DISPLAY_HEIGHT)
        self.spd_est.fps = orig_fps

        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            orig_fps, (W, H)
        )
        if not writer.isOpened():
            cap.release()
            raise RuntimeError(
                f"Cannot open VideoWriter for {output_path!r} — "
                f"'mp4v' codec unavailable in this OpenCV build."
            )
        print(f"[VSDAS] Processing: {source}  ({W}×{H} @ {orig_fps:.0f}fps)")
        print(f"[VSDAS] Output   : {output_path}")

        fi = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if max_frames and fi >= max_frames:
                    break

                annotated, decision = self.process_frame(frame)
                writer.write(annotated)
                fi += 1

                if fi % 30 == 0:
                    print(f"  Frame {fi:5d}  action={decision['action']:12s}  "
                          f"rec={decision['rec_speed']:4.0f}km/h  "
                          f"fps={self.fps_smooth:.1f}", end="\r")
        finally:
            cap.release()
            writer.release()
        print(f"\n[VSDAS] ✅  Done. {fi} frames → {output_path}")

        if display_in_colab:
            self._show_video_colab(output_path)

        return output_path

    # ── Live webcam / dashcam feed in Colab ───────────────────────────────
    def run_live(self, source=0, max_seconds: float = 30.0):
        """
        Capture from webcam and show frame-by-frame in Colab.
        Uses javascript_capture trick for Colab webcam access.
        """
        from IPython.display import Javascript
        # Colab webcam capture via JS
        js = Javascript("""
        async function captureStream() {
            const video = document.createElement('video');
            const stream = await navigator.mediaDevices.getUserMedia({video: true});
            video.srcObject = stream;
            await video.play();
            window._vsdas_stream = stream;
            window._vsdas_video  = video;
            console.log('Webcam started');
        }
        captureStream();
        """)
        display(js)
        print("[VSDAS] Webcam started. Capturing for", max_seconds, "seconds...")

        # Fallback: use OpenCV capture and display frames as base64 HTML
        cap = cv2.VideoCapture(source)
        t_start = time.time()
        fi = 0

        while time.time() - t_start < max_seconds:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
            annotated, decision = self.process_frame(frame)

            # Display in Colab
            if fi % 3 == 0:   # update every 3 frames to reduce flicker
                clear_output(wait=True)
                _, enc = cv2.imencode(".jpg", annotated,
                                      [cv2.IMWRITE_JPEG_QUALITY, 80])
                b64 = base64.b64encode(enc.tobytes()).decode()
                html = f"""
                <div style="font-family:monospace;background:#07080d;padding:10px;border-radius:8px">
                  <div style="color:#4af0c4;margin-bottom:6px;font-size:13px">
                    ▶ VSDAS LIVE  |  Action: <b style="color:{_col_to_css(decision['color'])}">{decision['action']}</b>
                    &nbsp;|&nbsp; Rec Speed: <b>{decision['rec_speed']:.0f} km/h</b>
                    &nbsp;|&nbsp; {decision['reason']}
                  </div>
                  <img src="data:image/jpeg;base64,{b64}" style="border-radius:6px;max-width:100%">
                </div>
                """
                display(HTML(html))
            fi += 1

        cap.release()
        print("[VSDAS] Live session ended.")

    @staticmethod
    def _show_video_colab(path: str, width: int = 900):
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        display(HTML(f"""
        <div style="background:#07080d;padding:14px;border-radius:10px;
                    border:1px solid #1a2030;display:inline-block">
          <div style="color:#4af0c4;font-family:monospace;font-size:12px;
                      letter-spacing:.1em;margin-bottom:8px">
            ▶ VSDAS — Real-Time Driver Assistance Output
          </div>
          <video width="{width}" controls autoplay loop muted
                 style="border-radius:6px;display:block">
            <source src="data:video/mp4;base64,{b64}" type="video/mp4">
          </video>
          <div style="color:#3a4050;font-family:monospace;font-size:10px;margin-top:6px">
            YOLOv8 · DeepSORT · Optical Flow · TTC · Speed Advisory
          </div>
        </div>
        """))

    # ── Paper metrics ─────────────────────────────────────────────────────
    def plot_telemetry(self):
        if not self.telemetry:
            print("No telemetry yet."); return
        df_actions = {}
        for entry in self.telemetry:
            a = entry["action"]
            df_actions[a] = df_actions.get(a, 0) + 1

        speeds = [e["rec_speed"] for e in self.telemetry]
        fpss   = [e["fps"]       for e in self.telemetry]
        n_haz  = [e["n_hazards"] for e in self.telemetry]

        fig, axes = plt.subplots(2, 2, figsize=(14,9))
        fig.suptitle("VSDAS Telemetry Analysis", fontsize=14, fontweight="bold")

        # Action distribution
        axes[0,0].bar(df_actions.keys(), df_actions.values(),
                      color=["#e53935","#ff7043","#ffa726","#66bb6a","#42a5f5","#26a69a"])
        axes[0,0].set_title("Decision Distribution")
        axes[0,0].set_ylabel("Frames")

        # Recommended speed over time
        axes[0,1].plot(speeds, color="#42a5f5", linewidth=0.8)
        axes[0,1].axhline(SPEED_LIMIT_KPH, color="red", linestyle="--",
                          linewidth=1, label=f"Limit {SPEED_LIMIT_KPH}km/h")
        axes[0,1].set_title("Recommended Speed Timeline")
        axes[0,1].set_ylabel("km/h"); axes[0,1].legend(fontsize=8)

        # FPS
        axes[1,0].plot(fpss, color="#66bb6a", linewidth=0.8)
        axes[1,0].axhline(25, color="orange", linestyle=":", label="Real-time (25fps)")
        axes[1,0].set_title("Processing FPS")
        axes[1,0].set_ylabel("FPS"); axes[1,0].legend(fontsize=8)

        # Hazard count per frame
        axes[1,1].fill_between(range(len(n_haz)), n_haz,
                                color="#ffa726", alpha=0.6)
        axes[1,1].set_title("Active Hazards per Frame")
        axes[1,1].set_ylabel("Count")

        for ax in axes.flat:
            ax.set_facecolor("#f4f4f4")
        plt.tight_layout()
        plt.savefig("vsdas_telemetry.png", dpi=150)
        plt.show()
        print("✅  Saved: vsdas_telemetry.png")


# ── Entry point ────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="VSDAS real-time driver assistance")
    parser.add_argument("--input",  default=INPUT_SOURCE,
                        help="0=webcam | video path | YouTube URL")
    parser.add_argument("--output", default=OUTPUT_VIDEO, help="annotated output video path")
    args = parser.parse_args()

    source = args.input
    if isinstance(source, str) and source.isdigit():   # "0" → webcam index 0
        source = int(source)

    vsdas = VSDASRealTime(
        model_path   = YOLO_MODEL,
        speed_limit  = SPEED_LIMIT_KPH,
        ego_speed    = EGO_SPEED_KPH,
        enable_voice = ENABLE_VOICE,
    )
    vsdas.run_video(
        source            = source,
        output_path       = args.output,
        max_frames        = MAX_FRAMES,
        display_in_colab  = False,
    )


if __name__ == "__main__":
    main()
