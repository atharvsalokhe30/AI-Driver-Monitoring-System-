"""
Seatbelt Detector
──────────────────
Heuristic approach: looks for a diagonal bright edge in the
driver's left shoulder-to-right-hip region of the frame.

Works best under normal indoor / car lighting.
Returns seatbelt_on = True when a diagonal belt-like structure is found.
"""

import cv2
import numpy as np


class SeatbeltDetector:
    """
    Region of Interest (ROI): left-to-centre, upper-mid frame
    (roughly where the shoulder belt crosses on the driver's torso).

    Strategy:
      1. Extract ROI (top 30–75 % of height, left 20–70 % of width)
      2. Convert to grayscale → edge detection (Canny)
      3. Hough line detection → filter lines with 30–60 ° angle
      4. If enough diagonal lines found → seatbelt detected
    """

    def __init__(self):
        self._counter_on  = 0
        self._counter_off = 0
        self.seatbelt_on  = True   # assume on at start (conservative)
        self._HYSTERESIS  = 12

    def update(self, frame) -> dict:
        h, w = frame.shape[:2]

        # ── ROI ──────────────────────────────────────────────────────────
        y1 = int(h * 0.30)
        y2 = int(h * 0.78)
        x1 = int(w * 0.10)
        x2 = int(w * 0.55)
        roi = frame[y1:y2, x1:x2]

        # ── Pre-process ───────────────────────────────────────────────────
        gray  = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur  = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)

        # ── Hough lines ───────────────────────────────────────────────────
        lines = cv2.HoughLinesP(
            edges,
            rho=1, theta=np.pi / 180,
            threshold=40,
            minLineLength=40,
            maxLineGap=20,
        )

        diagonal_count = 0
        if lines is not None:
            for ln in lines:
                coords = ln[0] if hasattr(ln[0], '__len__') else ln
                x_a, y_a, x_b, y_b = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
                dx = x_b - x_a
                dy = y_b - y_a
                if abs(dx) < 1e-3:
                    continue
                angle = abs(np.degrees(np.arctan2(dy, dx)))
                # Diagonal belt typically 25–65 degrees
                if 25 < angle < 65:
                    diagonal_count += 1

        belt_visible = diagonal_count >= 3

        if belt_visible:
            self._counter_on  += 1
            self._counter_off  = 0
        else:
            self._counter_off += 1
            self._counter_on   = 0

        if self._counter_on  >= self._HYSTERESIS:
            self.seatbelt_on = True
        if self._counter_off >= self._HYSTERESIS:
            self.seatbelt_on = False

        return {
            "seatbelt_on":     self.seatbelt_on,
            "diagonal_lines":  diagonal_count,
        }

    def reset(self):
        self._counter_on  = 0
        self._counter_off = 0
        self.seatbelt_on  = True
