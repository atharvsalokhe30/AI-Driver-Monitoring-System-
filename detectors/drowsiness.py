"""
Drowsiness & Yawning Detector
──────────────────────────────
Uses MediaPipe Tasks FaceLandmarker landmarks to compute:
  • Eye Aspect Ratio (EAR)  → drowsiness / eye closure
  • Mouth Aspect Ratio (MAR) → yawning
"""

import numpy as np

from config import (
    EAR_THRESHOLD, EAR_CONSEC_FRAMES,
    MAR_THRESHOLD, MAR_CONSEC_FRAMES,
)

# MediaPipe Face Mesh landmark indices (478-point model)
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]
MOUTH     = [61,  291,  39, 181,   0,  17, 269, 405]


def _euclidean(p1, p2) -> float:
    return np.linalg.norm(np.array(p1) - np.array(p2))


def _ear(landmarks, indices, w, h) -> float:
    pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in indices]
    A = _euclidean(pts[1], pts[5])
    B = _euclidean(pts[2], pts[4])
    C = _euclidean(pts[0], pts[3])
    return (A + B) / (2.0 * C + 1e-6)


def _mar(landmarks, indices, w, h) -> float:
    pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in indices]
    A = _euclidean(pts[2], pts[6])
    B = _euclidean(pts[3], pts[7])
    C = _euclidean(pts[0], pts[1])
    return (A + B) / (2.0 * C + 1e-6)


class DrowsinessDetector:
    def __init__(self):
        self._ear_ctr = 0
        self._mar_ctr = 0
        self.is_drowsy  = False
        self.is_yawning = False
        self.ear = 0.0
        self.mar = 0.0

    def update(self, landmarks, frame_w: int, frame_h: int) -> dict:
        """
        landmarks: list of NormalizedLandmark objects from FaceLandmarker result.
        """
        self.ear = (
            _ear(landmarks, LEFT_EYE,  frame_w, frame_h) +
            _ear(landmarks, RIGHT_EYE, frame_w, frame_h)
        ) / 2.0
        self.mar = _mar(landmarks, MOUTH, frame_w, frame_h)

        if self.ear < EAR_THRESHOLD:
            self._ear_ctr += 1
        else:
            self._ear_ctr = max(0, self._ear_ctr - 1)
        self.is_drowsy = self._ear_ctr >= EAR_CONSEC_FRAMES

        if self.mar > MAR_THRESHOLD:
            self._mar_ctr += 1
        else:
            self._mar_ctr = max(0, self._mar_ctr - 1)
        self.is_yawning = self._mar_ctr >= MAR_CONSEC_FRAMES

        return {
            "is_drowsy":  self.is_drowsy,
            "is_yawning": self.is_yawning,
            "ear":        round(self.ear, 3),
            "mar":        round(self.mar, 3),
            "ear_frames": self._ear_ctr,
            "mar_frames": self._mar_ctr,
        }

    def reset(self):
        self._ear_ctr = 0
        self._mar_ctr = 0
        self.is_drowsy  = False
        self.is_yawning = False
