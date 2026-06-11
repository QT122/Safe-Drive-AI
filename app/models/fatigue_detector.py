"""Fatigue detection via EAR (Eye Aspect Ratio) and MAR (Mouth Aspect Ratio)."""
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp

from app import config


class FatigueDetector:
    """Detects drowsiness from eye closure and yawning using MediaPipe face mesh."""

    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        )

    def _get_ear(self, landmarks, eye_indices, w, h) -> float:
        """Compute Eye Aspect Ratio for one eye."""
        coords = [
            (int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_indices
        ]
        A = np.linalg.norm(np.array(coords[1]) - np.array(coords[5]))
        B = np.linalg.norm(np.array(coords[2]) - np.array(coords[4]))
        C = np.linalg.norm(np.array(coords[0]) - np.array(coords[3]))
        return (A + B) / (2.0 * C) if C > 0 else 0.0

    def _get_mar(self, landmarks, w, h) -> float:
        """Compute Mouth Aspect Ratio."""
        coords = [
            (int(landmarks[i].x * w), int(landmarks[i].y * h))
            for pair in config.MOUTH
            for i in pair
        ]
        N1 = np.linalg.norm(np.array(coords[2]) - np.array(coords[3]))
        N2 = np.linalg.norm(np.array(coords[4]) - np.array(coords[5]))
        N3 = np.linalg.norm(np.array(coords[6]) - np.array(coords[7]))
        D = np.linalg.norm(np.array(coords[0]) - np.array(coords[1]))
        return (N1 + N2 + N3) / (3.0 * D) if D > 0 else 0.0

    def analyze(self, image_path: str, person_name: str) -> dict:
        """Analyze a single image for fatigue indicators.

        Returns dict with avg_ear, mar, ear_thresh, mar_thresh, is_drowsy, reasons.
        """
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            return {"error": f"Cannot read image: {image_path}"}

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h, w, _ = image_rgb.shape

        # Load per-person thresholds
        ear_mar_path = config.EAR_MAR_PATH
        if not ear_mar_path.exists():
            return {"error": "EAR/MAR threshold file not found"}

        ear_mar = pd.read_csv(str(ear_mar_path), index_col="person_id")
        if person_name == "Unknown" or person_name not in ear_mar.index:
            # Use default thresholds
            ear_thresh = 0.22
            mar_thresh = 0.70
        else:
            ear_thresh = float(ear_mar.loc[person_name, "avg_EAR"])
            mar_thresh = float(ear_mar.loc[person_name, "avg_MAR"])

        results = self.face_mesh.process(image_rgb)
        if not results.multi_face_landmarks:
            return {"error": "No face mesh detected"}

        landmarks = results.multi_face_landmarks[0].landmark
        ear_l = self._get_ear(landmarks, config.LEFT_EYE, w, h)
        ear_r = self._get_ear(landmarks, config.RIGHT_EYE, w, h)
        mar = self._get_mar(landmarks, w, h)
        avg_ear = (ear_l + ear_r) / 2.0

        reasons = []
        if avg_ear < ear_thresh:
            reasons.append("闭眼")
        if mar > mar_thresh:
            reasons.append("打哈欠")

        return {
            "avg_ear": round(float(avg_ear), 4),
            "mar": round(float(mar), 4),
            "ear_thresh": round(float(ear_thresh), 4),
            "mar_thresh": round(float(mar_thresh), 4),
            "is_drowsy": bool(reasons),
            "reasons": reasons,
        }

    def visualize(self, image_path: str, result: dict, person_name: str) -> np.ndarray | None:
        """Draw fatigue detection overlay on the image. Returns annotated BGR image."""
        if "error" in result:
            return None

        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            return None

        h, w, _ = image_bgr.shape
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image_rgb)

        if not results.multi_face_landmarks:
            return image_bgr

        landmarks = results.multi_face_landmarks[0].landmark

        # Draw info
        cv2.putText(
            image_bgr, f"User: {person_name}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2,
        )
        cv2.putText(
            image_bgr,
            f"EAR: {result['avg_ear']:.2f}  MAR: {result['mar']:.2f}",
            (10, 65),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2,
        )

        # Draw landmarks
        all_indices = (
            config.LEFT_EYE
            + config.RIGHT_EYE
            + [i for pair in config.MOUTH for i in pair]
        )
        for idx in all_indices:
            cx, cy = int(landmarks[idx].x * w), int(landmarks[idx].y * h)
            cv2.circle(image_bgr, (cx, cy), 2, (0, 255, 0), -1)

        return image_bgr
