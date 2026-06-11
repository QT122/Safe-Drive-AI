"""Pipeline orchestrator that combines all detection models."""
import json
import os
import tempfile
import time
from collections import deque
from datetime import datetime
from pathlib import Path

from app import config
from app.models.yolo_detector import YOLODetector
from app.models.face_recognizer import FaceRecognizer
from app.models.fatigue_detector import FatigueDetector
from app.models.lstm_classifier import LSTMInference


class DetectionPipeline:
    """Orchestrates the full detection pipeline: YOLO → Face → Fatigue → LSTM."""

    def __init__(self):
        # Suppress TF warnings
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

        self.yolo = YOLODetector()
        self.face_recognizer = FaceRecognizer()
        self.fatigue = FatigueDetector()
        self.lstm = LSTMInference()

        # Rolling feature buffer for LSTM
        self.feature_buffer = deque(maxlen=config.FEATURE_BUFFER_MAXLEN)

        # Rolling windows for fatigue behavior counting (5-second windows)
        self.blink_timestamps: deque[float] = deque()
        self.yawn_timestamps: deque[float] = deque()

        # Frame counter for LSTM trigger
        self._frame_count = 0

    def process_frame(self, image_path: str | Path) -> dict:
        """Process a single frame through all models.

        Returns a result dict suitable for the frontend JSON response.
        """
        image_path = Path(image_path)
        start_time = time.time()
        violations: list[tuple[str, str]] = []

        # ---- Step 1: YOLO detection ----
        yolo_scores = self.yolo.detect(image_path)

        # ---- Step 2: Face recognition ----
        import cv2
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is not None:
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            person_name, _ = self.face_recognizer.recognize(image_rgb)
        else:
            person_name = "Unknown"

        # ---- Step 3: Fatigue detection ----
        fatigue_result = self.fatigue.analyze(str(image_path), person_name)

        avg_ear = fatigue_result.get("avg_ear", 0.0)
        avg_mar = fatigue_result.get("mar", 0.0)
        reasons = fatigue_result.get("reasons", [])
        is_drowsy = fatigue_result.get("is_drowsy", False)

        # ---- Step 4: Build feature vector ----
        features = [
            yolo_scores["objects"],
            yolo_scores["seatbelt_on"],
            yolo_scores["seatbelt_off"],
            yolo_scores["extra_hand"],
            yolo_scores["hands_off"],
            yolo_scores["hands_on"],
            0.0,  # wheel placeholder
            avg_ear,
            avg_mar,
            float(is_drowsy),
        ]
        self.feature_buffer.append([round(float(x), 4) for x in features])

        # ---- Step 5: Track fatigue events ----
        now = time.time()
        self._prune_window(self.blink_timestamps, config.BLINK_WINDOW_SECONDS)
        self._prune_window(self.yawn_timestamps, config.YAWN_WINDOW_SECONDS)

        if "闭眼" in reasons:
            self.blink_timestamps.append(now)
            violations.append(("Action Detection", "Detected eye closure"))

        if "打哈欠" in reasons:
            self.yawn_timestamps.append(now)
            violations.append(("Action Detection", "Detected yawning"))

        # ---- Step 6: Behavior violation checks ----
        f = features
        if f[0] > config.OBJECT_CONFIDENCE:
            violations.append(("Action Detection", "Detected an object"))
        if f[1] < 0.5 and f[2] > config.SEATBELT_OFF_CONFIDENCE:
            violations.append(("Action Detection", "Seatbelt not fastened"))
        if f[3] > config.EXTRA_HAND_CONFIDENCE:
            violations.append(("Action Detection", "Extra hand detected"))
        if f[4] > 0.5 and f[5] < 0.5:
            violations.append(("Action Detection", "Hand left the steering wheel"))

        # ---- Step 7: LSTM classification (every N frames) ----
        self._frame_count += 1
        lstm_result = None
        if (
            self._frame_count >= config.LSTM_SEQUENCE_LENGTH
            and len(self.feature_buffer) >= config.LSTM_SEQUENCE_LENGTH
        ):
            recent_features = list(self.feature_buffer)[-config.LSTM_SEQUENCE_LENGTH:]
            lstm_result = self.lstm.predict(recent_features)
            self._frame_count = 0

        # ---- Step 8: Build result ----
        blink_count = len(self.blink_timestamps)
        yawn_count = len(self.yawn_timestamps)

        result = {
            "predicted_class": lstm_result["predicted_class"] if lstm_result else 4,
            "class_name": lstm_result["class_name"] if lstm_result else "normal",
            "class_probabilities": (
                lstm_result["class_probabilities"] if lstm_result else [0, 0, 0, 0, 1]
            ),
            "objects": int(f[0] > config.OBJECT_CONFIDENCE),
            "with_seatbelt": int(f[1] > config.SEATBELT_ON_CONFIDENCE),
            "without_seatbelt": int(f[2] > config.SEATBELT_OFF_CONFIDENCE),
            "extra_hand": int(f[3] > config.EXTRA_HAND_CONFIDENCE),
            "hands_off": int(f[4] > config.HANDS_OFF_CONFIDENCE),
            "hands_on": int(f[5] > config.HANDS_ON_CONFIDENCE),
            "wheel": features[6],
            "name": person_name,
            "avg_ear": round(float(avg_ear), 4),
            "avg_mar": round(float(avg_mar), 4),
            "blink_count": blink_count,
            "yawn_count": yawn_count,
            "violations": violations,
            "processing_time_ms": round((time.time() - start_time) * 1000, 1),
        }

        # ---- Step 9: Persist results ----
        self._save_result(result)
        self._log_violations(violations, Path(image_path).name)

        return result

    def _save_result(self, result: dict) -> None:
        """Atomically write result.json."""
        result_path = config.RESULT_PATH
        result_path.parent.mkdir(parents=True, exist_ok=True)

        # Atomic write via temp file
        with tempfile.NamedTemporaryFile(
            "w", delete=False, encoding="utf-8", dir=str(result_path.parent)
        ) as tmpf:
            json.dump(result, tmpf, ensure_ascii=False, indent=2)
            temp_name = tmpf.name
        os.replace(temp_name, str(result_path))

    def _log_violations(
        self, violations: list[tuple[str, str]], image_name: str
    ) -> None:
        """Append violations to the log file."""
        if not violations:
            return

        log_path = config.VIOLATION_LOG_PATH
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(str(log_path), "a", encoding="utf-8") as f:
            for category, message in violations:
                line = f"[{timestamp}] {category}: {message}\n"
                f.write(line)

    @staticmethod
    def _prune_window(window: deque, max_age_seconds: float) -> None:
        """Remove entries older than max_age_seconds from a deque of timestamps."""
        now = time.time()
        while window and (now - window[0]) > max_age_seconds:
            window.popleft()
