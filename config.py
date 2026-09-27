# ─────────────────────────────────────────────
#  AI Driver Monitoring System — Configuration
# ─────────────────────────────────────────────

# ── Camera ────────────────────────────────────
CAMERA_INDEX = 0          # 0 = default laptop webcam
FRAME_WIDTH  = 1280
FRAME_HEIGHT = 720

# ── Drowsiness (Eye Aspect Ratio) ─────────────
EAR_THRESHOLD          = 0.25   # below this → eye considered closed
EAR_CONSEC_FRAMES      = 20     # frames eye must stay closed → DROWSY alert

# ── Yawning (Mouth Aspect Ratio) ──────────────
MAR_THRESHOLD          = 0.65   # above this → mouth considered open (yawn)
MAR_CONSEC_FRAMES      = 15     # frames mouth must stay open → YAWN alert

# ── Head Pose ─────────────────────────────────
HEAD_PITCH_DOWN_THRESH  = 20    # degrees downward  → "Looking Down"
HEAD_PITCH_UP_THRESH    = -15   # degrees upward    → "Looking Up"
HEAD_YAW_LEFT_THRESH    = -25   # degrees left yaw  → "Looking Left"
HEAD_YAW_RIGHT_THRESH   =  25   # degrees right yaw → "Looking Right"
HEAD_CONSEC_FRAMES      = 20    # frames off-road to trigger alert

# ── YOLO (Phone & Seatbelt) ───────────────────
YOLO_MODEL              = "yolov8n.pt"
YOLO_CONF_THRESHOLD     = 0.45
PHONE_CLASS_ID          = 67    # COCO class: cell phone
PHONE_CONSEC_FRAMES     = 10    # frames phone visible → alert

# ── Seatbelt ──────────────────────────────────
SEATBELT_CHECK_ENABLED  = True

# ── Emotion Recognition ───────────────────────
EMOTION_FRAME_SKIP      = 5     # run FER every N frames
EMOTION_CONF_THRESHOLD  = 0.40  # minimum confidence to display

# ── Alert System ──────────────────────────────
ALERT_COOLDOWN_SEC      = 3.0   # seconds between repeated audio beeps
ALERT_SOUND_PATH        = "assets/alert.wav"

# ── Logging ───────────────────────────────────
LOG_FILE_PATH           = "logs/events.csv"

# ── Display / HUD ─────────────────────────────
HUD_FONT_SCALE          = 0.6
HUD_THICKNESS           = 2
HUD_ALPHA               = 0.55  # transparency of HUD overlay

# Colour palette  (BGR)
COLOR_OK      = (0,   220,  80)
COLOR_WARN    = (0,   165, 255)
COLOR_DANGER  = (0,    40, 220)
COLOR_INFO    = (220, 220,   0)
COLOR_TEXT    = (255, 255, 255)
COLOR_BG      = (20,   20,  20)
