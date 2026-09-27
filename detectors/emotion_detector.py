"""
Emotion Detector
─────────────────
Uses the FER library (lightweight CNN) to detect facial emotions.
Runs every N frames defined by EMOTION_FRAME_SKIP for CPU efficiency.
"""

from fer.fer import FER

from config import EMOTION_FRAME_SKIP, EMOTION_CONF_THRESHOLD


_EMOTION_EMOJI = {
    "angry":    "😠",
    "disgust":  "🤢",
    "fear":     "😨",
    "happy":    "😊",
    "sad":      "😢",
    "surprise": "😲",
    "neutral":  "😐",
}

_ALERT_EMOTIONS = {"angry", "fear", "sad"}   # emotions worth flagging


class EmotionDetector:
    def __init__(self):
        self._detector  = FER(mtcnn=False)
        self._frame_idx = 0
        self.emotion    = "neutral"
        self.scores: dict = {}
        self.confidence = 0.0
        self.should_alert = False
        self.emoji = "😐"

    def update(self, frame) -> dict:
        self._frame_idx += 1

        if self._frame_idx % EMOTION_FRAME_SKIP != 0:
            return self._result()

        try:
            result = self._detector.detect_emotions(frame)
            if result:
                top = result[0]  # first face
                self.scores = top.get("emotions", {})
                if self.scores:
                    self.emotion    = max(self.scores, key=self.scores.get)
                    self.confidence = self.scores[self.emotion]
                    if self.confidence < EMOTION_CONF_THRESHOLD:
                        self.emotion    = "neutral"
                        self.confidence = 0.0
                    self.emoji        = _EMOTION_EMOJI.get(self.emotion, "😐")
                    self.should_alert = self.emotion in _ALERT_EMOTIONS
        except Exception:
            pass  # silently continue on FER errors

        return self._result()

    def _result(self) -> dict:
        return {
            "emotion":      self.emotion,
            "confidence":   round(self.confidence, 2),
            "scores":       self.scores,
            "emoji":        self.emoji,
            "should_alert": self.should_alert,
        }

    def reset(self):
        self.emotion      = "neutral"
        self.scores       = {}
        self.confidence   = 0.0
        self.should_alert = False
        self.emoji        = "😐"
