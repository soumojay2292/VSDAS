"""
advisor.py — Hazard dataclass and SpeedAdvisor decision engine.
Sources: Part B, CELL 3 (PRIORITY, SPEED_RULES) · CELL 7 (Hazard, SpeedAdvisor).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.config import SPEED_LIMIT_KPH, HAZARD_MIN_CONF
from src.utils import dist_band

# Road-hazard labels gated by confidence before influencing the decision.
_GATED_LABELS = ("pothole", "speed_breaker")

# ── Hazard priority (lower = more urgent) ─────────────────────────────────────
PRIORITY: Dict[str, int] = {
    "person":        1,
    "dog":           1, "cat":    1, "cow":     1,
    "horse":         1, "elephant": 1, "bear":  1,
    "motorcycle":    2,
    "bicycle":       2,
    "stop sign":     2,
    "traffic light": 3,
    "pothole":       2,
    "speed_breaker": 3,
    "debris":        2,
    "car":           4,
    "truck":         4,
    "bus":           4,
}

# ── Speed reduction rules: (hazard_category, dist_band) → max kph ─────────────
SPEED_RULES: Dict[Tuple[str, str], int] = {
    # large animals
    ("animal_large",   "critical"):  10,
    ("animal_large",   "near"):      20,
    ("animal_large",   "medium"):    30,
    # small animals
    ("animal_small",   "critical"):  15,
    ("animal_small",   "near"):      25,
    # pedestrians
    ("pedestrian",     "critical"):   5,
    ("pedestrian",     "near"):      15,
    ("pedestrian",     "medium"):    25,
    ("pedestrian",     "far"):       40,
    # road surface hazards
    ("pothole",        "critical"):  10,
    ("pothole",        "near"):      20,
    ("speed_breaker",  "critical"):   5,
    ("speed_breaker",  "near"):      10,
    ("speed_breaker",  "medium"):    20,
    # traffic control
    ("stop_sign",      "near"):       0,
    ("red_light",      "near"):       0,
    ("yellow_light",   "near"):      15,
    # vehicles ahead
    ("vehicle_slow",   "critical"):   5,
    ("vehicle_slow",   "near"):      20,
    ("vehicle_slow",   "medium"):    40,
    ("vehicle_fast",   "critical"):  10,
    ("vehicle_fast",   "near"):      30,
}

# ── Action → display colour (BGR) ────────────────────────────────────────────
ACTION_COLORS: Dict[str, Tuple[int, int, int]] = {
    "STOP":       (0,   0, 255),
    "BRAKE HARD": (0,   0, 255),
    "SLOW DOWN":  (0,  80, 255),
    "REDUCE SPD": (0, 165, 255),
    "MAINTAIN":   (60, 200,  60),
    "ACCELERATE": (200, 200,  0),
    "CRUISE":     (60, 200,  60),
}


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class Hazard:
    """Single detected hazard, ready for SpeedAdvisor consumption."""
    label:     str
    bbox:      Tuple[float, float, float, float]   # (x1, y1, x2, y2)
    dist_m:    float
    conf:      float
    track_id:  Optional[int]   = None
    tl_state:  Optional[str]   = None   # "red" | "yellow" | "green" — traffic lights only
    speed_kph: Optional[float] = None   # relative speed of vehicle ahead
    ttc:       Optional[float] = None   # time-to-collision in seconds


# ── Decision engine ───────────────────────────────────────────────────────────

class SpeedAdvisor:
    """
    Aggregates all active Hazard objects for a frame and returns a single
    recommended speed + action label.

    Temporal smoothing (deque of recent recommendations, 20th-percentile
    filter) prevents speed from jumping upward too quickly.
    """

    def __init__(self, speed_limit: float = SPEED_LIMIT_KPH) -> None:
        self.speed_limit = speed_limit
        self._history: deque = deque(maxlen=8)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _classify(self, h: Hazard) -> Tuple[str, str]:
        """Map a Hazard to (category, dist_band) for SPEED_RULES lookup."""
        lab  = h.label.lower()
        band = dist_band(h.dist_m)

        if lab == "person":
            return ("pedestrian", band)
        if lab in ("dog", "cat"):
            return ("animal_small", band)
        if lab in ("horse", "cow", "elephant", "bear", "zebra", "giraffe", "sheep"):
            return ("animal_large", band)
        if lab == "pothole":
            return ("pothole", band)
        if lab == "speed_breaker":
            return ("speed_breaker", band)
        if lab == "stop sign":
            return ("stop_sign", band)
        if lab == "traffic light":
            state = (h.tl_state or "unknown").lower()
            if state == "red":    return ("red_light",    band)
            if state == "yellow": return ("yellow_light", band)
            return ("green_light", band)
        if lab in ("car", "truck", "bus", "motorcycle", "bicycle"):
            cat = "vehicle_slow" if (h.speed_kph or 0.0) < 30 else "vehicle_fast"
            return (cat, band)
        return ("generic_obstacle", band)

    @staticmethod
    def _action_and_color(
        smoothed_speed: float,
        ego_speed:      float,
    ) -> Tuple[str, Tuple[int, int, int]]:
        if   smoothed_speed == 0:                 return "STOP",       ACTION_COLORS["STOP"]
        elif smoothed_speed < 15:                 return "BRAKE HARD", ACTION_COLORS["BRAKE HARD"]
        elif smoothed_speed < 30:                 return "SLOW DOWN",  ACTION_COLORS["SLOW DOWN"]
        elif smoothed_speed < ego_speed - 5:      return "REDUCE SPD", ACTION_COLORS["REDUCE SPD"]
        elif smoothed_speed > ego_speed + 8:      return "ACCELERATE", ACTION_COLORS["ACCELERATE"]
        else:                                     return "MAINTAIN",   ACTION_COLORS["MAINTAIN"]

    # ── Public API ────────────────────────────────────────────────────────────

    def recommend(
        self,
        hazards:   List[Hazard],
        ego_speed: float,
    ) -> Dict:
        """
        Parameters
        ----------
        hazards   : all Hazard objects detected in the current frame
        ego_speed : current vehicle speed in kph

        Returns
        -------
        dict with keys: action, rec_speed, reason, urgent_hazard, color
        """
        if not hazards:
            rec = min(self.speed_limit, ego_speed + 5.0)
            self._history.append(rec)
            return {
                "action":        "CRUISE",
                "rec_speed":     rec,
                "reason":        "Road clear",
                "urgent_hazard": None,
                "color":         ACTION_COLORS["CRUISE"],
            }

        min_speed: float   = self.speed_limit
        urgent:    Optional[Hazard] = None
        reasons:   List[str] = []

        # Sort by proximity then priority; nearest + most urgent binds first
        for h in sorted(hazards, key=lambda x: (x.dist_m, PRIORITY.get(x.label, 5))):
            # Confidence gate: weak classical road-hazard detections are
            # ignored so false positives cannot drive the recommendation.
            if h.label in _GATED_LABELS and (h.conf or 0.0) < HAZARD_MIN_CONF:
                continue
            cat, band = self._classify(h)
            rule_speed = SPEED_RULES.get((cat, band))
            if rule_speed is not None and rule_speed < min_speed:
                min_speed = rule_speed
                urgent    = h
                reasons.insert(0, f"{h.label} @ {h.dist_m:.0f} m ({band})")
            else:
                reasons.append(f"{h.label} @ {h.dist_m:.0f} m")

        # Conservative smoothing: 20th-percentile of recent history
        self._history.append(min_speed)
        smoothed = float(np.percentile(list(self._history), 20))

        action, color = self._action_and_color(smoothed, ego_speed)

        return {
            "action":        action,
            "rec_speed":     smoothed,
            "reason":        reasons[0] if reasons else "obstacle",
            "urgent_hazard": urgent,
            "color":         color,
        }