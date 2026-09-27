"""
AI Driver Monitoring System — Main Entry Point
================================================
Run:  python main.py
Quit: press Q in the video window
"""

import sys
import time
import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark
from ultralytics import YOLO

import config
from detectors.drowsiness         import DrowsinessDetector
from detectors.head_pose          import HeadPoseEstimator
from detectors.phone_detector     import PhoneDetector
from detectors.seatbelt_detector  import SeatbeltDetector
from detectors.emotion_detector   import EmotionDetector
from utils.alert   import AlertManager
from utils.logger  import EventLogger
from utils.display import draw_hud

MODEL_PATH = os.path.join("models", "face_landmarker.task")


def main():
    print("=" * 55)
    print("  AI Driver Monitoring System  |  Press Q to quit")
    print("=" * 55)

    # ── Camera ────────────────────────────────────────────────
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera. Check CAMERA_INDEX in config.py")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    # ── MediaPipe FaceLandmarker (Tasks API) ──────────────────
    print("[INFO] Loading MediaPipe FaceLandmarker …")
    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    fl_options   = FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    face_landmarker = FaceLandmarker.create_from_options(fl_options)

    # ── YOLO ─────────────────────────────────────────────────
    print("[INFO] Loading YOLOv8n …")
    yolo_model = YOLO(config.YOLO_MODEL)

    # ── Detectors ─────────────────────────────────────────────
    drowsiness_det = DrowsinessDetector()
    head_pose_det  = HeadPoseEstimator()
    phone_det      = PhoneDetector(yolo_model)
    seatbelt_det   = SeatbeltDetector()
    emotion_det    = EmotionDetector()

    # ── Utilities ─────────────────────────────────────────────
    alert_mgr  = AlertManager()
    event_log  = EventLogger()

    # ── FPS tracking ──────────────────────────────────────────
    fps_counter = 0
    fps_timer   = time.time()
    fps_display = 0.0

    print("[INFO] All modules loaded. Starting webcam …")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]

        # ── FPS ───────────────────────────────────────────────
        fps_counter += 1
        elapsed = time.time() - fps_timer
        if elapsed >= 1.0:
            fps_display = fps_counter / elapsed
            fps_counter = 0
            fps_timer   = time.time()

        state: dict = {"fps": fps_display}

        # ── MediaPipe inference (Tasks API) ───────────────────
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        fl_result = face_landmarker.detect(mp_image)

        face_found = bool(fl_result.face_landmarks)
        if face_found:
            landmarks = fl_result.face_landmarks[0]  # list of NormalizedLandmark
            d_res = drowsiness_det.update(landmarks, w, h)
            h_res = head_pose_det.update(landmarks,  w, h)
            state.update(d_res)
            state.update(h_res)
        else:
            drowsiness_det.reset()
            head_pose_det.reset()
            state.update({
                "is_drowsy": False, "is_yawning": False,
                "ear": 0.0, "mar": 0.0,
                "is_distracted": False, "direction": "No Face",
                "pitch": 0.0, "yaw": 0.0, "roll": 0.0,
            })

        # ── YOLO phone detection ───────────────────────────────
        p_res = phone_det.update(frame)
        state.update(p_res)
        state["phone_boxes"] = p_res.get("boxes", [])

        # ── Seatbelt ──────────────────────────────────────────
        if config.SEATBELT_CHECK_ENABLED:
            sb_res = seatbelt_det.update(frame)
            state.update(sb_res)
        else:
            state["seatbelt_on"] = True

        # ── Emotion ───────────────────────────────────────────
        e_res = emotion_det.update(frame)
        state["emotion"]       = e_res["emotion"]
        state["emotion_conf"]  = e_res["confidence"]
        state["emoji"]         = e_res["emoji"]
        state["emotion_alert"] = e_res["should_alert"]

        # ── Alerts & Log ──────────────────────────────────────
        alert_mgr.process(state)
        event_log.log(state)

        # ── Draw HUD ──────────────────────────────────────────
        frame = draw_hud(frame, state, alert_mgr.active_alerts)

        if not face_found:
            cv2.putText(frame, "No face detected",
                        (w // 2 - 120, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0, 200, 255), 2, cv2.LINE_AA)

        cv2.imshow("AI Driver Monitoring System", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q")):
            break

    # ── Cleanup ───────────────────────────────────────────────
    cap.release()
    face_landmarker.close()
    alert_mgr.shutdown()
    cv2.destroyAllWindows()
    print("[INFO] Session ended. Log saved to:", config.LOG_FILE_PATH)


if __name__ == "__main__":
    main()
