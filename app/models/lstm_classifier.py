"""LSTM sequence classifier for driving behavior classification."""
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from app import config


class LSTMClassifier(nn.Module):
    """LSTM model for sequence-based driver behavior classification."""

    def __init__(self, input_size=10, hidden_size=32, output_size=5):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size, batch_first=True
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        final_step = lstm_out[:, -1, :]
        return self.fc(final_step)


class LSTMInference:
    """Wrapper for LSTM model inference."""

    def __init__(self, model_path: Path | None = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = LSTMClassifier(
            input_size=10, hidden_size=32, output_size=len(config.CLASS_NAMES)
        ).to(self.device)

        path = model_path or config.LSTM_MODEL_PATH
        if path.exists():
            self.model.load_state_dict(
                torch.load(str(path), map_location=self.device, weights_only=True)
            )
        self.model.eval()
        self._loaded = path.exists()

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def predict(self, features_sequence: list) -> dict:
        """Run inference on a sequence of feature vectors.

        Args:
            features_sequence: list of 10-element feature lists, length = sequence_length.

        Returns:
            dict with predicted_class, class_name, class_probabilities.
        """
        if not self._loaded:
            return {
                "predicted_class": 4,
                "class_name": "normal",
                "class_probabilities": [0.0, 0.0, 0.0, 0.0, 1.0],
            }

        tensor = (
            torch.tensor(features_sequence, dtype=torch.float32)
            .unsqueeze(0)
            .to(self.device)
        )
        with torch.no_grad():
            output = self.model(tensor)
            probs = F.softmax(output, dim=1).cpu().numpy()[0]

        predicted_class = int(np.argmax(probs))
        return {
            "predicted_class": predicted_class,
            "class_name": config.CLASS_NAMES[predicted_class],
            "class_probabilities": [round(float(p), 4) for p in probs],
        }
