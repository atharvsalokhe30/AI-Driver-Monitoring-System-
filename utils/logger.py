"""
Event Logger
─────────────
Appends timestamped safety events to a CSV file for post-trip review.
"""

import csv
import os
from datetime import datetime

from config import LOG_FILE_PATH

_HEADER = ["timestamp", "event_type", "value", "confidence"]


class EventLogger:
    def __init__(self):
        os.makedirs(os.path.dirname(LOG_FILE_PATH), exist_ok=True)
        # Write header if file is new
        if not os.path.exists(LOG_FILE_PATH):
            with open(LOG_FILE_PATH, "w", newline="") as f:
                csv.writer(f).writerow(_HEADER)
        self._prev_state: dict = {}

    def log(self, state: dict):
        """Log transitions (new alerts only, not repeated ones every frame)."""
        events = []
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        def _check(key, label, value="", conf=""):
            if state.get(key) and not self._prev_state.get(key):
                events.append([ts, label, value, conf])

        _check("is_drowsy",    "DROWSY",      f"EAR={state.get('ear','')}")
        _check("is_yawning",   "YAWNING",     f"MAR={state.get('mar','')}")
        _check("is_distracted","DISTRACTED",  state.get("direction",""))
        _check("phone_detected","PHONE_USAGE","")
        if not state.get("seatbelt_on", True) and self._prev_state.get("seatbelt_on", True):
            events.append([ts, "NO_SEATBELT", "", ""])
        if state.get("emotion_alert") and not self._prev_state.get("emotion_alert"):
            events.append([ts, "EMOTION_ALERT", state.get("emotion",""), str(state.get("emotion_conf",""))])

        if events:
            with open(LOG_FILE_PATH, "a", newline="") as f:
                csv.writer(f).writerows(events)

        self._prev_state = dict(state)
