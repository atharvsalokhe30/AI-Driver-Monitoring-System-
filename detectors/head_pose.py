"""
Head Pose Estimator
────────────────────
Uses MediaPipe Tasks FaceLandmarker landmarks + cv2.solvePnP to
estimate pitch / yaw / roll and classify driver gaze direction.
"""

import cv2
import numpy as np

from config import (
    HEAD_PITCH_DOWN_THRESH, HEAD_PITCH_UP_THRESH,
    HEAD_YAW_LEFT_THRESH,   HEAD_YAW_RIGHT_THRESH,
    HEAD_CONSEC_FRAMES,
)

_MODEL_POINTS = np.array([
    (0.0,    0.0,    0.0  ),
    (0.0,  -330.0, -65.0 ),
    (-225.0, 170.0,-135.0),
    ( 225.0, 170.0,-135.0),
    (-150.0,-150.0,-125.0),
    ( 150.0,-150.0,-125.0),
], dtype=np.float64)

_LM_INDICES = [1, 152, 263, 33, 287, 57]


class HeadPoseEstimator:
    def __init__(self):
        self._off_ctr = 0
        self.direction = "Forward"
        self.pitch = 0.0
        self.yaw   = 0.0
        self.roll  = 0.0
        self.is_distracted = False

    def update(self, landmarks, frame_w: int, frame_h: int) -> dict:
        """
        landmarks: list of NormalizedLandmark objects from FaceLandmarker result.
        """
        image_pts = np.array([
            (landmarks[i].x * frame_w, landmarks[i].y * frame_h)
            for i in _LM_INDICES
        ], dtype=np.float64)

        fl = frame_w
        cam_mat = np.array([
            [fl, 0,  frame_w / 2],
            [0,  fl, frame_h / 2],
            [0,  0,  1          ],
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        success, rvec, _ = cv2.solvePnP(
            _MODEL_POINTS, image_pts, cam_mat, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            return self._result()

        rmat, _ = cv2.Rodrigues(rvec)
        sy = np.sqrt(rmat[0, 0]**2 + rmat[1, 0]**2)
        if sy >= 1e-6:
            self.pitch = np.degrees(np.arctan2(-rmat[2, 1], rmat[2, 2]))
            self.yaw   = np.degrees(np.arctan2( rmat[2, 0], sy))
            self.roll  = np.degrees(np.arctan2(-rmat[1, 0], rmat[0, 0]))
        else:
            self.pitch = np.degrees(np.arctan2(-rmat[1, 2], rmat[1, 1]))
            self.yaw   = np.degrees(np.arctan2( rmat[2, 0], sy))
            self.roll  = 0.0

        self.direction = self._classify()
        if self.direction != "Forward":
            self._off_ctr += 1
        else:
            self._off_ctr = max(0, self._off_ctr - 1)
        self.is_distracted = self._off_ctr >= HEAD_CONSEC_FRAMES
        return self._result()

    def _classify(self) -> str:
        if self.pitch > HEAD_PITCH_DOWN_THRESH:   return "Looking Down"
        if self.pitch < HEAD_PITCH_UP_THRESH:     return "Looking Up"
        if self.yaw   < HEAD_YAW_LEFT_THRESH:     return "Looking Left"
        if self.yaw   > HEAD_YAW_RIGHT_THRESH:    return "Looking Right"
        return "Forward"

    def _result(self) -> dict:
        return {
            "direction":     self.direction,
            "pitch":         round(self.pitch, 1),
            "yaw":           round(self.yaw,   1),
            "roll":          round(self.roll,  1),
            "is_distracted": self.is_distracted,
            "off_frames":    self._off_ctr,
        }

    def reset(self):
        self._off_ctr = 0
        self.direction = "Forward"
        self.is_distracted = False
