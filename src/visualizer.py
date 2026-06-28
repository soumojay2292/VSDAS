from typing import List, Optional

import cv2
import numpy as np

from src.advisor import Hazard

ACTION_COLORS = {
    "STOP":       (0,   0,   255),
    "BRAKE HARD": (0,   0,   255),
    "SLOW DOWN":  (0,   80,  255),
    "REDUCE SPD": (0,  165,  255),
    "MAINTAIN":   (60, 200,   60),
    "ACCELERATE": (200,200,    0),
    "CRUISE":     (60, 200,   60),
}
HAZARD_BOX_COLORS = {
    "person":        (0,  60, 255),
    "dog":           (0, 140, 255),
    "cat":           (0, 140, 255),
    "horse":         (0, 100, 255),
    "cow":           (0, 100, 255),
    "elephant":      (0,  50, 200),
    "pothole":       (0, 200, 255),
    "speed_breaker": (0, 220, 220),
    "traffic light": (0, 255, 200),
    "stop sign":     (0,   0, 220),
    "car":           (200,200, 60),
    "truck":         (180,180, 40),
    "bus":           (160,160, 30),
    "motorcycle":    (220,220, 80),
    "bicycle":       (200,240,100),
}
DEFAULT_BOX_COLOR = (180,180,180)

TL_COLORS = {"red":(0,0,220),"yellow":(0,200,200),"green":(0,200,0),"unknown":(120,120,120)}


def draw_hazard_boxes(frame: np.ndarray, hazards: List[Hazard],
                      urgent_id: Optional[int]):
    for h in hazards:
        x1,y1,x2,y2 = [int(v) for v in h.bbox]
        is_urgent = (h.track_id is not None and h.track_id == urgent_id) \
                    or (urgent_id is None and h == hazards[0] if hazards else False)
        col   = HAZARD_BOX_COLORS.get(h.label, DEFAULT_BOX_COLOR)
        thick = 3 if is_urgent else 1

        # Main box
        cv2.rectangle(frame, (x1,y1),(x2,y2), col, thick)

        # Label bar
        bar_h = 22
        cv2.rectangle(frame, (x1,y1-bar_h),(x2,y1), col, -1)

        label_str = h.label.upper()
        if h.tl_state and h.tl_state != "unknown":
            label_str = f"TL:{h.tl_state.upper()}"
        label_str += f"  {h.dist_m:.0f}m"
        if h.speed_kph is not None:
            label_str += f"  {h.speed_kph:.0f}km/h"

        cv2.putText(frame, label_str,
                    (x1+3, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                    (10,10,10), 1, cv2.LINE_AA)

        # Urgency glow for closest hazard — ROI blend (no full-frame copy).
        # Copy only the stroke bounding region: rect is at box ±4, thickness 5
        # extends 3px outside the path; +4 margin covers that and the
        # exclusive slice end so all stroke pixels are inside the ROI.
        if is_urgent:
            fh_, fw_ = frame.shape[:2]
            gx0 = max(0, x1-4-4); gy0 = max(0, y1-4-4)
            gx1 = min(fw_, x2+4+4); gy1 = min(fh_, y2+4+4)
            if gx1 > gx0 and gy1 > gy0:
                roi = frame[gy0:gy1, gx0:gx1]
                ov  = roi.copy()
                cv2.rectangle(ov, (x1-4-gx0, y1-4-gy0),
                              (x2+4-gx0, y2+4-gy0), col, 5)
                cv2.addWeighted(ov, 0.35, roi, 0.65, 0, roi)

        # TTC arc
        if h.ttc is not None and h.ttc < 10:
            r   = 14
            cx_ = x2 + 16
            cy_ = y1 + 16
            if cx_ < frame.shape[1] - 5 and cy_ > 5:
                frac = max(0, min(1, 1 - h.ttc/10))
                arc_col = (0,0,220) if h.ttc < 3 else (0,120,255)
                cv2.circle(frame,(cx_,cy_),r,(40,40,40),-1)
                cv2.ellipse(frame,(cx_,cy_),(r,r),0,-90,
                            int(-90+frac*360), arc_col, 2)
                cv2.putText(frame, f"{h.ttc:.1f}s",
                            (cx_-10, cy_+4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (220,220,220), 1)


def _blend_rect(frame: np.ndarray, x0: int, y0: int, x1: int, y1: int,
                color, alpha: float):
    """In-place semi-transparent filled rectangle. Equivalent to:
        ov = frame.copy(); cv2.rectangle(ov,(x0,y0),(x1,y1),color,-1)
        cv2.addWeighted(ov, alpha, frame, 1-alpha, 0, frame)
    but blends only the rectangle ROI — no full-frame copy. Output identical
    (OpenCV rect pt2 is inclusive, so the slice end is +1)."""
    h, w = frame.shape[:2]
    x0 = max(0, x0); y0 = max(0, y0)
    x1 = min(w, x1 + 1); y1 = min(h, y1 + 1)
    if x1 <= x0 or y1 <= y0:
        return
    roi = frame[y0:y1, x0:x1]
    overlay = np.empty_like(roi)
    overlay[:] = color
    cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0, roi)


def draw_hud(frame: np.ndarray,
             action: str, rec_speed: float,
             ego_speed: float, reason: str,
             hazards: List[Hazard],
             fps: float, frame_idx: int,
             speed_limit: float):

    h, w = frame.shape[:2]
    act_col = ACTION_COLORS.get(action, (200,200,200))

    # ── Top bar ──────────────────────────────────────────────────────────────
    _blend_rect(frame, 0, 0, w, 52, (8,10,14), 0.80)
    cv2.putText(frame,"VSDAS  Real-Time Driver Assistance",
                (14,34), cv2.FONT_HERSHEY_DUPLEX, 0.78, (180,230,160), 2, cv2.LINE_AA)
    cv2.putText(frame, f"FPS:{fps:.0f}  F:{frame_idx:05d}",
                (w-180,32), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (80,90,100), 1, cv2.LINE_AA)

    # ── Action panel (bottom-left) ────────────────────────────────────────────
    panel_h = 100
    _blend_rect(frame, 0, h-panel_h, 420, h, (8,10,14), 0.82)
    # Left accent bar colour
    cv2.rectangle(frame,(0,h-panel_h),(5,h), act_col, -1)

    cv2.putText(frame,"SYSTEM DECISION",
                (14,h-panel_h+18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100,110,120), 1, cv2.LINE_AA)
    cv2.putText(frame, action,
                (14,h-panel_h+68),
                cv2.FONT_HERSHEY_DUPLEX, 1.6, act_col, 3, cv2.LINE_AA)
    # Reason
    cv2.putText(frame, reason[:48],
                (14,h-14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (130,140,150), 1, cv2.LINE_AA)

    # ── Speed panel (bottom-right) ────────────────────────────────────────────
    sp_x = w - 240
    _blend_rect(frame, sp_x, h-panel_h, w, h, (8,10,14), 0.82)

    # Speed limit circle
    cv2.circle(frame,(sp_x+34,h-60),26,(10,10,10),-1)
    cv2.circle(frame,(sp_x+34,h-60),26,(220,220,220),3)
    cv2.putText(frame,f"{int(speed_limit)}",
                (sp_x+20,h-54),
                cv2.FONT_HERSHEY_DUPLEX,0.65,(220,220,220),2,cv2.LINE_AA)
    cv2.putText(frame,"LIMIT",(sp_x+14,h-33),
                cv2.FONT_HERSHEY_SIMPLEX,0.3,(150,150,150),1,cv2.LINE_AA)

    # Recommended speed
    cv2.putText(frame,"REC SPEED",
                (sp_x+70, h-panel_h+18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38,(100,110,120),1,cv2.LINE_AA)
    speed_col = (0,0,220) if rec_speed < 15 else (0,120,255) if rec_speed < 35 else (60,200,60)
    cv2.putText(frame, f"{rec_speed:.0f}",
                (sp_x+70, h-panel_h+72),
                cv2.FONT_HERSHEY_DUPLEX, 1.8, speed_col, 3, cv2.LINE_AA)
    cv2.putText(frame,"km/h",
                (sp_x+160,h-panel_h+72),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55,(120,130,140),1,cv2.LINE_AA)

    # Ego speed small
    cv2.putText(frame,f"Your speed: {ego_speed:.0f} km/h",
                (sp_x+70,h-14),
                cv2.FONT_HERSHEY_SIMPLEX,0.38,(100,110,120),1,cv2.LINE_AA)

    # ── Hazard legend (mid-right) ─────────────────────────────────────────────
    if hazards:
        leg_y0 = 60
        leg_x  = w - 220
        _blend_rect(frame, leg_x-6, leg_y0, w, leg_y0+len(hazards[:8])*22+12,
                    (8,10,14), 0.75)
        cv2.putText(frame,"DETECTED HAZARDS",
                    (leg_x,leg_y0+12),
                    cv2.FONT_HERSHEY_SIMPLEX,0.36,(80,90,100),1,cv2.LINE_AA)
        for i,hz in enumerate(hazards[:8]):
            col = HAZARD_BOX_COLORS.get(hz.label, DEFAULT_BOX_COLOR)
            cy_ = leg_y0 + 26 + i*22
            cv2.circle(frame,(leg_x+6,cy_-3),5,col,-1)
            lab = hz.label
            if hz.tl_state: lab += f"({hz.tl_state})"
            cv2.putText(frame,f"{lab}  {hz.dist_m:.0f}m  {hz.conf:.0%}",
                        (leg_x+16,cy_),
                        cv2.FONT_HERSHEY_SIMPLEX,0.36,(180,190,200),1,cv2.LINE_AA)
