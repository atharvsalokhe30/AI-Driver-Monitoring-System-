"""
Alert Manager
──────────────
Handles audio (pygame) and visual alert state for the system.
Prevents repeated beeps with a configurable cooldown timer.
"""

import time
import os
import pygame

from config import ALERT_COOLDOWN_SEC, ALERT_SOUND_PATH


class AlertManager:
    def __init__(self):
        pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=512)
        self._last_alert_time: dict[str, float] = {}
        self._sound = None
        self._load_sound()

        # Active visual banners: list of (label, color_bgr)
        self.active_alerts: list[tuple[str, tuple]] = []

    def _load_sound(self):
        if os.path.exists(ALERT_SOUND_PATH):
            try:
                self._sound = pygame.mixer.Sound(ALERT_SOUND_PATH)
            except Exception:
                self._sound = None

    def _beep(self, alert_key: str):
        """Play the alert sound, respecting the cooldown per alert type."""
        now = time.time()
        last = self._last_alert_time.get(alert_key, 0)
        if now - last >= ALERT_COOLDOWN_SEC:
            self._last_alert_time[alert_key] = now
            if self._sound:
                self._sound.play()
            else:
                # Fallback: system beep
                try:
                    import winsound
                    winsound.Beep(1000, 300)
                except Exception:
                    pass

    def process(self, state: dict):
        """
        Evaluate detection state and trigger alerts as needed.
        state keys: is_drowsy, is_yawning, is_distracted,
                    phone_detected, seatbelt_on, emotion_alert
        """
        self.active_alerts = []

        if state.get("is_drowsy"):
            self.active_alerts.append(("⚠ DROWSY!", (0, 40, 220)))
            self._beep("drowsy")

        if state.get("is_yawning"):
            self.active_alerts.append(("YAWNING", (0, 140, 255)))
            self._beep("yawn")

        if state.get("is_distracted"):
            self.active_alerts.append(("DISTRACTED", (0, 165, 255)))
            self._beep("distracted")

        if state.get("phone_detected"):
            self.active_alerts.append(("📱 PHONE!", (0, 60, 255)))
            self._beep("phone")

        if not state.get("seatbelt_on", True):
            self.active_alerts.append(("NO SEATBELT!", (30, 30, 220)))
            self._beep("seatbelt")

        if state.get("emotion_alert"):
            emotion = state.get("emotion", "").upper()
            self.active_alerts.append((f"EMOTION: {emotion}", (200, 80, 0)))

    def shutdown(self):
        pygame.mixer.quit()
