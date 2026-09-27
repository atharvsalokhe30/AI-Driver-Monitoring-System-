"""
Phone Detector
───────────────
Uses YOLOv8n (COCO) to detect a mobile phone in the frame.
COCO class 67 = 'cell phone'.
"""

import numpy as np
from ultralytics import YOLO

from config import YOLO_MODEL, YOLO_CONF_THRESHOLD, PHONE_CLASS_ID, PHONE_CONSEC_FRAMES


class PhoneDetector:
    def __init__(self, model: YOLO):
        self._model = model
        self._counter = 0
        self.phone_detected = False
        self.boxes: list = []

    def update(self, frame) -> dict:
        results = self._model(frame, verbose=False, conf=YOLO_CONF_THRESHOLD, classes=[PHONE_CLASS_ID])
        self.boxes = []

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                if cls == PHONE_CLASS_ID:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    self.boxes.append((x1, y1, x2, y2, conf))

        if self.boxes:
            self._counter += 1
        else:
            self._counter = max(0, self._counter - 1)

        self.phone_detected = self._counter >= PHONE_CONSEC_FRAMES

        return {
            "phone_detected": self.phone_detected,
            "boxes":          self.boxes,
            "frames":         self._counter,
        }

    def reset(self):
        self._counter = 0
        self.phone_detected = False
        self.boxes = []
