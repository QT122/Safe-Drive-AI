# Safe-Drive-AI

Real-time driver monitoring system using computer vision and deep learning. Detects unsafe driving behaviors — seatbelt violations, handheld objects, hands-off-wheel, fatigue (eye closure / yawning), and distracted driving (phone use, eating/drinking) — then alerts the driver with voice warnings.

## Features

- **Object & Posture Detection** — YOLO-based dual-model pipeline detects seatbelt status, handheld objects, extra hands, and hands-off-wheel.
- **Face Recognition** — FaceNet + MTCNN identifies individual drivers from a pre-enrolled face database.
- **Fatigue Monitoring** — MediaPipe FaceMesh computes Eye Aspect Ratio (EAR) and Mouth Aspect Ratio (MAR) to catch drowsiness and yawning in real time.
- **Behavior Classification** — LSTM sequence model classifies driving behavior across 5 categories: `no_belt`, `disturb`, `phone`, `eat/drink`, `normal`.
- **Voice Alerts** — Browser TTS warns the driver on detected violations (English).
- **Driver Score** — Live 0-100 safety score based on recent violations, viewable at `/score_page`.

## Architecture

```
webcam (browser) ──POST /upload──▶ Flask API ──▶ DetectionPipeline
                                                  ├── YOLODetector      (object.pt + body.pt)
                                                  ├── FaceRecognizer    (FaceNet + MTCNN)
                                                  ├── FatigueDetector   (MediaPipe FaceMesh)
                                                  └── LSTMInference     (best_lstm_v3.pth)
                                                       │
                                                  result.json / violation_log.txt
```

## Prerequisites

- Python 3.10+
- CUDA-capable GPU (optional; falls back to CPU)
- Webcam for live monitoring

## Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/Safe-Drive-AI.git
cd Safe-Drive-AI

# Create and activate a virtual environment (recommended)
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Model Weights & Data Files

The following files are **not** included in the repository (gitignored). You must provide your own:

| File | Purpose |
|---|---|
| `models/object.pt` | YOLO model for object detection (seatbelt_on, seatbelt_off, objects) |
| `models/body.pt` | YOLO model for body/posture detection (extra_hand, hands_off, hands_on) |
| `models/best_lstm_v3.pth` | Trained LSTM classifier weights |
| `data/features.npy` | Face embeddings database (dict: `{name: 512-d embedding}`) |
| `data/avg_ear_mar.csv` | Per-person EAR/MAR baseline thresholds |

Place these files in the respective directories before running the application. Without them, the pipeline will return default/safe values.

## Usage

```bash
python app.py
```

Then open **http://127.0.0.1:5000** in your browser. Grant camera permission when prompted. The dashboard captures frames every 500 ms and displays real-time driver status.

- Main dashboard: **http://127.0.0.1:5000/**
- Driver score panel: **http://127.0.0.1:5000/score_page**

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Main monitoring dashboard |
| `GET` | `/score_page` | Driver safety score panel |
| `POST` | `/upload` | Upload a video frame (multipart `image` field), runs the full pipeline |
| `GET` | `/result.json` | Latest pipeline result as JSON |
| `GET` | `/score_data` | Driver score (0-100) with recent violations |

## Configuration

All tunable parameters live in `app/config.py`:

- Detection confidence thresholds (`OBJECT_CONFIDENCE`, `SEATBELT_OFF_CONFIDENCE`, etc.)
- Face similarity threshold (`FACE_SIMILARITY_THRESHOLD`)
- LSTM sequence length (`LSTM_SEQUENCE_LENGTH`)
- Fatigue detection windows (`BLINK_WINDOW_SECONDS`, `YAWN_WINDOW_SECONDS`)
- Server host/port (`FLASK_HOST`, `FLASK_PORT`)

Environment variables: `HOST`, `PORT`, `DEBUG=true`.

## Project Structure

```
Safe-Drive-AI-main/
├── app.py                      # Flask entry point
├── requirements.txt            # Python dependencies
├── .gitignore
├── app/
│   ├── config.py               # Centralized configuration
│   ├── pipeline.py             # Detection pipeline orchestrator
│   ├── watcher.py              # File-system watcher (watchdog)
│   └── models/
│       ├── yolo_detector.py    # YOLO object & posture detection
│       ├── face_recognizer.py  # FaceNet + MTCNN face recognition
│       ├── fatigue_detector.py # MediaPipe EAR/MAR fatigue detection
│       └── lstm_classifier.py  # LSTM behavior classifier
├── static/
│   ├── index.html              # Main dashboard (webcam + status)
│   └── score_page.html         # Driver score panel
├── models/                     # (gitignored) YOLO / LSTM weight files
├── data/                       # (gitignored) features.npy, avg_ear_mar.csv
└── uploads/                    # (gitignored) uploaded frames at runtime
```

## License

This project is provided for educational and research purposes. Use at your own risk in real driving scenarios.
