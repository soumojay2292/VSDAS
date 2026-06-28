import math
from collections import defaultdict, deque
from typing import Dict, Tuple

import numpy as np


class VehicleSpeedEstimator:
    # Drop per-track state this many frames after a track was last seen.
    # > DeepSORT max_age (25) so live tracks are never pruned → outputs unchanged.
    _STALE_AFTER = 60

    def __init__(self, fps: float = 30.0):
        self.fps  = fps
        self.prev: Dict[int, Tuple] = {}   # track_id → (cx, cy, dist, frame_idx)
        self.speed_buf: Dict[int, deque] = defaultdict(lambda: deque(maxlen=8))

    def _prune(self, frame_idx: int) -> None:
        """Evict state for track_ids not seen for _STALE_AFTER frames,
        bounding memory on long runs. Active tracks update every frame so
        they are never stale; speed/TTC results are unaffected."""
        stale = [tid for tid, (_, _, _, pfi) in self.prev.items()
                 if frame_idx - pfi > self._STALE_AFTER]
        for tid in stale:
            self.prev.pop(tid, None)
            self.speed_buf.pop(tid, None)

    def update(self, track_id: int, bbox: Tuple,
               dist_m: float, frame_idx: int) -> float:
        x1,y1,x2,y2 = bbox
        cx = (x1+x2)/2; cy = (y1+y2)/2
        h_px = y2 - y1

        if track_id in self.prev:
            pcx,pcy,pdist,pfi = self.prev[track_id]
            dt = (frame_idx - pfi) / self.fps
            if dt > 0 and h_px > 1:
                px_disp = math.hypot(cx-pcx, cy-pcy)
                # Scale by depth proxy
                label_h = max(h_px, 1)
                m_disp  = (px_disp / label_h) * 1.5
                raw_spd = m_disp / dt * 3.6
                self.speed_buf[track_id].append(raw_spd)

        self.prev[track_id] = (cx, cy, dist_m, frame_idx)
        buf = list(self.speed_buf[track_id])
        self._prune(frame_idx)
        return float(np.median(buf)) if buf else 0.0

    def ttc(self, ego_speed: float, veh_speed: float, dist_m: float) -> float:
        rel = max(0.0, ego_speed - veh_speed) / 3.6
        return dist_m / rel if rel > 0.1 else float('inf')
