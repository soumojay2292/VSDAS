"""
utils.py — Pure helper functions shared across VSDAS modules.
Sources: Part B, CELL 3 (dist_band) · CELL 4 (estimate_distance) · CELL 11 (_iou, _col_to_css).
No side effects, no I/O, no model dependencies.
"""

from typing import Optional, Tuple

# ── Per-class real-world heights (metres) — used by estimate_distance ─────────
REAL_HEIGHTS: dict = {
    "person":        1.75,
    "car":           1.50,
    "truck":         3.50,
    "bus":           3.20,
    "motorcycle":    1.20,
    "bicycle":       1.10,
    "horse":         1.60,
    "cow":           1.40,
    "elephant":      3.00,
    "dog":           0.50,
    "cat":           0.30,
    "bear":          1.20,
    "sheep":         0.80,
    "traffic light": 0.50,
    "stop sign":     1.80,
    "pothole":       0.10,
    "speed_breaker": 0.15,
}
DEFAULT_HEIGHT:    float = 1.0
FOCAL_LENGTH_PX:   float = 800.0   # typical dashcam — calibrate with checkerboard


# ── Distance band classifier ──────────────────────────────────────────────────

def dist_band(d: float) -> str:
    """Map a distance in metres to a named urgency band."""
    if   d < 8:   return "critical"
    elif d < 20:  return "near"
    elif d < 50:  return "medium"
    else:         return "far"


# ── Monocular distance estimator ──────────────────────────────────────────────

def estimate_distance(
    label:     str,
    bbox:      Tuple[float, float, float, float],
    frame_h:   int,
    vanish_y:  Optional[int] = None,
) -> float:
    """
    Pin-hole camera model: D = (f × H_real) / h_px.

    Ground-plane refinement applied for road hazards (pothole,
    speed_breaker, debris) when vanish_y is supplied.

    Returns distance in metres, rounded to 1 decimal place.
    Returns 999.0 when the bounding box is too small to be reliable.
    """
    x1, y1, x2, y2 = bbox
    h_px = y2 - y1
    if h_px < 2:
        return 999.0

    h_real = REAL_HEIGHTS.get(label, DEFAULT_HEIGHT)
    dist   = (FOCAL_LENGTH_PX * h_real) / h_px

    if label in ("pothole", "speed_breaker", "debris") and vanish_y is not None:
        ground_y = y2
        t    = max(0.01, (ground_y - vanish_y) / (frame_h - vanish_y))
        dist = min(dist, 80.0 / t)

    return round(float(dist), 1)


# ── Bounding-box IoU ──────────────────────────────────────────────────────────

def _iou(
    b1: Tuple[float, float, float, float],
    b2: Tuple[float, float, float, float],
) -> float:
    """Intersection-over-Union for two (x1,y1,x2,y2) boxes."""
    x1 = max(b1[0], b2[0]);  y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]);  y2 = min(b1[3], b2[3])
    i  = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a  = (b1[2] - b1[0]) * (b1[3] - b1[1])
    b  = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return i / max(a + b - i, 1e-6)


# ── Colour conversion ─────────────────────────────────────────────────────────

def _col_to_css(bgr: Tuple[int, int, int]) -> str:
    """Convert an OpenCV BGR tuple to a CSS rgb() string."""
    b, g, r = bgr
    return f"rgb({r},{g},{b})"