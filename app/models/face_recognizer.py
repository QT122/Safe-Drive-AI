"""Face recognition using FaceNet + MTCNN."""
import numpy as np
import torch
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.metrics.pairwise import cosine_similarity

from app import config


class FaceRecognizer:
    """Face detection, embedding extraction, and identity matching."""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mtcnn = MTCNN(
            image_size=160, margin=0, keep_all=False, device=self.device
        )
        self.facenet = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)

        self._face_db: dict | None = None

    @property
    def face_db(self) -> dict:
        """Lazy-load face embeddings database."""
        if self._face_db is None:
            features_path = config.FEATURES_PATH
            if features_path.exists():
                self._face_db = np.load(
                    str(features_path), allow_pickle=True
                ).item()
            else:
                self._face_db = {}
        return self._face_db

    def recognize(self, image_rgb: np.ndarray) -> tuple[str, float | None]:
        """Recognize a face in an RGB image array.

        Returns (name, similarity_score) or ("Unknown", None).
        """
        pil_img = Image.fromarray(image_rgb)
        face = self.mtcnn(pil_img)
        if face is None:
            return "Unknown", None

        with torch.no_grad():
            emb = self.facenet(face.unsqueeze(0).to(self.device)).cpu().numpy()

        best_score = -1.0
        best_name = "Unknown"
        for name, db_feat in self.face_db.items():
            sim = cosine_similarity(emb, db_feat.reshape(1, -1))[0][0]
            if sim > best_score:
                best_score = sim
                best_name = name

        if best_score >= config.FACE_SIMILARITY_THRESHOLD:
            return best_name, float(best_score)
        return "Unknown", None
