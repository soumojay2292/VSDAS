from typing import Tuple

import cv2
import numpy as np


class TrafficLightClassifier:
    """
    Given the cropped bounding box of a detected 'traffic light',
    classify its state (red / yellow / green) by colour channel analysis.
    """

    @staticmethod
    def classify(frame: np.ndarray, bbox: Tuple) -> str:
        x1,y1,x2,y2 = [int(v) for v in bbox]
        x1,y1 = max(0,x1), max(0,y1)
        x2,y2 = min(frame.shape[1],x2), min(frame.shape[0],y2)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return "unknown"
        hsv   = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        h, w  = hsv.shape[:2]
        # Divide into thirds: top=red, mid=yellow, bot=green
        t3    = h // 3
        zones = {"red": hsv[:t3,:,:], "yellow": hsv[t3:2*t3,:,:], "green": hsv[2*t3:,:,:]}
        scores = {}
        masks_def = {
            "red":    [(0,120,100),(10,255,255)],
            "yellow": [(15,100,100),(35,255,255)],
            "green":  [(40,60,60),(90,255,255)],
        }
        for colour, roi in zones.items():
            lo = np.array(masks_def[colour][0])
            hi = np.array(masks_def[colour][1])
            mask = cv2.inRange(roi, lo, hi)
            scores[colour] = float(np.sum(mask)) / max(roi.size, 1)

        best = max(scores, key=scores.get)
        return best if scores[best] > 0.02 else "unknown"