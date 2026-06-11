"""Application configuration. All paths are relative to the project root."""
import os
from pathlib import Path

# Project root
ROOT_DIR = Path(__file__).resolve().parent.parent

# Model weights
MODEL_DIR = ROOT_DIR / "models"
OBJECT_YOLO_PATH = MODEL_DIR / "object.pt"       # YOLO model for object detection
BODY_YOLO_PATH = MODEL_DIR / "body.pt"           # YOLO model for body/posture detection
LSTM_MODEL_PATH = MODEL_DIR / "best_lstm_v3.pth"  # LSTM classifier weights

# Data files
DATA_DIR = ROOT_DIR / "data"
FEATURES_PATH = DATA_DIR / "features.npy"         # Face embeddings database
EAR_MAR_PATH = DATA_DIR / "avg_ear_mar.csv"       # Per-person EAR/MAR thresholds

# Runtime
UPLOAD_DIR = ROOT_DIR / "uploads"
RESULT_PATH = ROOT_DIR / "result.json"
VIOLATION_LOG_PATH = ROOT_DIR / "violation_log.txt"

# Server
FLASK_HOST = os.environ.get("HOST", "127.0.0.1")
FLASK_PORT = int(os.environ.get("PORT", 5000))
FLASK_DEBUG = os.environ.get("DEBUG", "false").lower() == "true"

# Detection thresholds
FACE_SIMILARITY_THRESHOLD = 0.6
LSTM_SEQUENCE_LENGTH = 5       # frames before running LSTM classification
FEATURE_BUFFER_MAXLEN = 100    # max frames in rolling buffer

# Behavior thresholds
OBJECT_CONFIDENCE = 0.3
SEATBELT_ON_CONFIDENCE = 0.3
SEATBELT_OFF_CONFIDENCE = 0.5
EXTRA_HAND_CONFIDENCE = 0.1
HANDS_OFF_CONFIDENCE = 0.03
HANDS_ON_CONFIDENCE = 0.1

# Fatigue thresholds (5-second window)
BLINK_WINDOW_SECONDS = 5
YAWN_WINDOW_SECONDS = 5
FATIGUE_BLINK_THRESHOLD = 2
FATIGUE_YAWN_THRESHOLD = 2

# Class names for LSTM output
CLASS_NAMES = ["no_belt", "disturb", "phone", "eat/drink", "normal"]

# MediaPipe face mesh indices
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH = [[61, 291], [39, 181], [0, 17], [269, 405]]

# Ensure directories exist
for d in [MODEL_DIR, DATA_DIR, UPLOAD_DIR]:
    d.mkdir(parents=True, exist_ok=True)
