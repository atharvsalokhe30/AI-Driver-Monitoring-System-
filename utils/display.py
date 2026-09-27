"""
HUD Display Renderer
─────────────────────
Draws all overlays onto the camera frame:
  • Top-left status panel (semi-transparent)
  • Per-feature metric badges
  • Alert banners (full-width, bottom)
  • YOLO bounding boxes
"""

import cv2
import numpy as np

from config import (
    HUD_ALPHA, HUD_FONT_SCALE, HUD_THICKNESS,
    COLOR_OK, COLOR_WARN, COLOR_DANGER, COLOR_INFO, COLOR_TEXT, COLOR_BG,
)

FONT = cv2.FONT_HERSHEY_SIMPLEX


def _put_text(frame, text, pos, color=COLOR_TEXT, scale=None, thickness=None):
    sc = scale    or HUD_FONT_SCALE
    th = thickness or HUD_THICKNESS
    cv2.putText(frame, text, pos, FONT, sc, (0, 0, 0), th + 2, cv2.LINE_AA)
    cv2.putText(frame, text, pos, FONT, sc, color,     th,     cv2.LINE_AA)


def draw_hud(frame, state: dict, active_alerts: list):
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # ── Side panel background ─────────────────────────────────────────────
    panel_w = 300
    cv2.rectangle(overlay, (0, 0), (panel_w, h), COLOR_BG, -1)
    cv2.addWeighted(overlay, HUD_ALPHA, frame, 1 - HUD_ALPHA, 0, frame)

    # ── Title ─────────────────────────────────────────────────────────────
    _put_text(frame, "DRIVER MONITOR", (10, 30), COLOR_INFO, scale=0.65, thickness=2)
    cv2.line(frame, (10, 38), (panel_w - 10, 38), COLOR_INFO, 1)

    y = 65
    line_h = 30

    def badge(label: str, value: str, alert: bool):
        nonlocal y
        color = COLOR_DANGER if alert else COLOR_OK
        _put_text(frame, f"{label}:", (12, y), COLOR_TEXT, scale=0.52)
        _put_text(frame, value,       (155, y), color,     scale=0.52)
        y += line_h

    # ── Feature rows ──────────────────────────────────────────────────────
    ear  = state.get("ear", 0.0)
    mar  = state.get("mar", 0.0)
    badge("Eyes (EAR)",
          f"{ear:.2f} {'CLOSED' if state.get('is_drowsy') else 'OPEN'}",
          state.get("is_drowsy", False))

    badge("Yawn (MAR)",
          f"{mar:.2f} {'YAWN' if state.get('is_yawning') else 'OK'}",
          state.get("is_yawning", False))

    badge("Head Pose",
          state.get("direction", "Forward"),
          state.get("is_distracted", False))

    badge("Pitch/Yaw",
          f"{state.get('pitch',0):.0f} / {state.get('yaw',0):.0f}",
          False)

    badge("Phone",
          "DETECTED" if state.get("phone_detected") else "None",
          state.get("phone_detected", False))

    badge("Seatbelt",
          "OK" if state.get("seatbelt_on", True) else "MISSING",
          not state.get("seatbelt_on", True))

    emoji   = state.get("emoji", "😐")
    emotion = state.get("emotion", "neutral").capitalize()
    econf   = state.get("emotion_conf", 0.0)
    badge("Emotion",
          f"{emotion} ({econf:.0%})",
          state.get("emotion_alert", False))

    # ── FPS ───────────────────────────────────────────────────────────────
    fps = state.get("fps", 0)
    _put_text(frame, f"FPS: {fps:.0f}", (12, h - 15), COLOR_TEXT, scale=0.45)

    # ── YOLO phone boxes ──────────────────────────────────────────────────
    for (x1, y1, x2, y2, conf) in state.get("phone_boxes", []):
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_DANGER, 2)
        _put_text(frame, f"Phone {conf:.0%}", (x1, y1 - 8), COLOR_DANGER, scale=0.5)

    # ── Alert banners (bottom) ────────────────────────────────────────────
    banner_h = 50
    bx = panel_w + 10
    for i, (label, color) in enumerate(active_alerts):
        by = h - banner_h - i * (banner_h + 4)
        ov2 = frame.copy()
        cv2.rectangle(ov2, (bx, by), (w - 10, by + banner_h), color, -1)
        cv2.addWeighted(ov2, 0.75, frame, 0.25, 0, frame)
        text_y = by + 33
        _put_text(frame, label, (bx + 15, text_y), COLOR_TEXT, scale=0.85, thickness=2)

    return frame
