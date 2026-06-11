"""YOLO-based object and body/posture detection."""
from pathlib import Path
from PIL import Image
from ultralytics import YOLO

from app import config


class YOLODetector:
    """Handles YOLO-based object detection and body/posture detection."""

    def __init__(
        self,
        object_model_path: Path | None = None,
        body_model_path: Path | None = None,
    ):
        object_path = object_model_path or config.OBJECT_YOLO_PATH
        body_path = body_model_path or config.BODY_YOLO_PATH

        self.object_model = YOLO(str(object_path)) if object_path.exists() else None
        self.body_model = YOLO(str(body_path)) if body_path.exists() else None

    def _crop_center_square(self, img: Image.Image) -> Image.Image:
        """Center-crop to a square and resize to 640x640."""
        width, height = img.size
        left = (width - height) // 2
        right = left + height
        img_cropped = img.crop((left, 0, right, height))
        return img_cropped.resize((640, 640))

    def detect(self, img_path: str | Path) -> dict:
        """Run both YOLO models and return aggregated feature scores.

        Returns a dict with scores for:
          - object classes: objects, seatbelt_on, seatbelt_off
          - body classes: extra_hand, hands_off, hands_on
        """
        img = Image.open(img_path)
        img = self._crop_center_square(img)

        scores = {
            "objects": 0.0,
            "seatbelt_on": 0.0,
            "seatbelt_off": 0.0,
            "extra_hand": 0.0,
            "hands_off": 0.0,
            "hands_on": 0.0,
        }

        # Object model detection (classes: 0=seatbelt_on, 1=seatbelt_off, 2=objects)
        if self.object_model:
            results_obj = self.object_model(img, verbose=False)
            if results_obj and results_obj[0].boxes is not None:
                for cls, conf in zip(
                    results_obj[0].boxes.cls.cpu().tolist(),
                    results_obj[0].boxes.conf.cpu().tolist(),
                ):
                    cls_id = int(cls)
                    if cls_id == 0:
                        scores["seatbelt_on"] = max(scores["seatbelt_on"], conf)
                    elif cls_id == 1:
                        scores["seatbelt_off"] = max(scores["seatbelt_off"], conf)
                    elif cls_id == 2:
                        scores["objects"] = max(scores["objects"], conf)

        # Body/posture model detection (classes: 0=extra_hand, 1=hands_off, 2=hands_on)
        if self.body_model:
            results_body = self.body_model(img, verbose=False)
            if results_body and results_body[0].boxes is not None:
                for cls, conf in zip(
                    results_body[0].boxes.cls.cpu().tolist(),
                    results_body[0].boxes.conf.cpu().tolist(),
                ):
                    cls_id = int(cls)
                    if cls_id == 0:
                        scores["extra_hand"] = max(scores["extra_hand"], conf)
                    elif cls_id == 1:
                        scores["hands_off"] = max(scores["hands_off"], conf)
                    elif cls_id == 2:
                        scores["hands_on"] = max(scores["hands_on"], conf)

        return scores
