"""Flask application entry point for Safe-Drive-AI.

Starts the web server, serves the dashboard, and handles image upload /
pipeline inference requests from the frontend.
"""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from app import config
from app.pipeline import DetectionPipeline

app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)

pipeline = DetectionPipeline()

# Ensure upload directory exists
config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Page routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    """Serve the main driver monitoring dashboard."""
    return send_from_directory("static", "index.html")


@app.route("/score_page")
def score_page():
    """Serve the driver score panel."""
    return send_from_directory("static", "score_page.html")


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------

@app.route("/upload", methods=["POST"])
def upload():
    """Receive a video frame, save it, run the detection pipeline.

    Returns the pipeline result directly so the frontend can update
    immediately without a second request.
    """
    if "image" not in request.files:
        return jsonify({"error": "No image file in request"}), 400

    file = request.files["image"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"frame_{timestamp}.jpg"
    filepath = config.UPLOAD_DIR / filename
    file.save(str(filepath))

    try:
        result = pipeline.process_frame(filepath)
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/result.json")
def get_result():
    """Return the latest pipeline result as JSON."""
    result_path = config.RESULT_PATH
    if not result_path.exists():
        return jsonify({"error": "No result yet"}), 404

    with open(str(result_path), "r", encoding="utf-8") as fh:
        return jsonify(json.load(fh))


@app.route("/score_data")
def score_data():
    """Compute and return a driver safety score (0-100).

    Score starts at 100 and deductions are applied for violations recorded
    in the last 5 minutes.  Also returns the most recent violation entries
    for display.
    """
    log_path = config.VIOLATION_LOG_PATH
    all_lines: list[str] = []
    if log_path.exists():
        with open(str(log_path), "r", encoding="utf-8") as fh:
            all_lines = [line.strip() for line in fh if line.strip()]

    # Only penalise violations from the last 5 minutes
    cutoff = datetime.now() - timedelta(minutes=5)
    recent: list[str] = []
    score = 100

    for line in all_lines:
        try:
            ts_str = line[1:20]  # "[YYYY-MM-DD HH:MM:SS]"
            ts = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
            if ts >= cutoff:
                recent.append(line)
                score -= 5
        except (ValueError, IndexError):
            recent.append(line)
            score -= 5

    score = max(0, min(100, score))

    return jsonify({"score": score, "violations": recent[-10:]})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(
        host=config.FLASK_HOST,
        port=config.FLASK_PORT,
        debug=config.FLASK_DEBUG,
    )
