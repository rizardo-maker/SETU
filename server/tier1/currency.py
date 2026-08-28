"""
Currency denomination classifier — Tier 1 "reflex" path.

Deliberately NOT a vision-language model. See the architecture notes
in the project document: a calibrated small classifier is the only
thing we're willing to let tell a blind user how much money they're
holding, because it's the only thing here whose confidence we can
measure and threshold. The Ollama VLM in tier2/vlm.py handles
everything open-ended instead.

This module is honest about not having a trained model yet:
`CurrencyClassifier.ready` is False until you drop a real
currency_classifier.onnx + currency_labels.json into models/ (see
training/train_currency_classifier.py). main.py checks `.ready` and
tells the user the truth rather than guessing with an untrained graph.
"""
from __future__ import annotations
import json
import logging
from typing import Optional

import cv2
import numpy as np

from server import config

log = logging.getLogger("setu.currency")

# ImageNet normalisation — matches the MobileNetV3/EfficientNet-B0
# pretrained weights the training script starts from. Change both
# here and in training/ together if you switch base architectures.
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(bgr_frame: np.ndarray, size: int = config.CURRENCY_INPUT_SIZE) -> np.ndarray:
    """BGR uint8 HWC -> normalised float32 NCHW, batch size 1."""
    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    arr = resized.astype(np.float32) / 255.0
    arr = (arr - _MEAN) / _STD
    chw = np.transpose(arr, (2, 0, 1))
    return np.expand_dims(chw, axis=0).astype(np.float32)


class ModelNotReady(RuntimeError):
    """Raised when predict() is called but no trained model is loaded.
    Callers must catch this and tell the user honestly — never fall
    back to a fabricated answer."""


class CurrencyClassifier:
    def __init__(self):
        self.ready = False
        self.labels: list[str] = []
        self._session = None
        self._input_name: Optional[str] = None
        self._load()

    def _load(self) -> None:
        if not config.CURRENCY_MODEL_PATH.exists():
            log.warning(
                "No currency model at %s — Tier 1 currency mode will report "
                "itself unavailable until you train one.",
                config.CURRENCY_MODEL_PATH,
            )
            return
        if not config.CURRENCY_LABELS_PATH.exists():
            log.warning("Currency model found but labels file missing at %s", config.CURRENCY_LABELS_PATH)
            return
        try:
            import onnxruntime as ort  # lazy: keeps the server bootable without onnxruntime installed
        except ImportError:
            log.warning("onnxruntime not installed — pip install -r requirements-full.txt to enable currency mode")
            return

        providers = ["CPUExecutionProvider"]
        # CoreMLExecutionProvider is available on some onnxruntime builds for
        # Apple Silicon and is worth adding once you've confirmed it's present:
        # if "CoreMLExecutionProvider" in ort.get_available_providers():
        #     providers.insert(0, "CoreMLExecutionProvider")
        self._session = ort.InferenceSession(str(config.CURRENCY_MODEL_PATH), providers=providers)
        self._input_name = self._session.get_inputs()[0].name
        self.labels = json.loads(config.CURRENCY_LABELS_PATH.read_text())
        self.ready = True
        log.info("Currency classifier loaded: %d classes, providers=%s", len(self.labels), providers)

    def predict_logits(self, bgr_frame: np.ndarray) -> np.ndarray:
        if not self.ready:
            raise ModelNotReady(
                "Currency model isn't trained yet. Run training/train_currency_classifier.py "
                "and place the exported .onnx + labels.json in models/."
            )
        x = preprocess(bgr_frame)
        outputs = self._session.run(None, {self._input_name: x})
        return outputs[0][0]  # (num_classes,) logits for the single image in the batch


# Module-level singleton — one model, shared across connections. Loading
# is the expensive part; inference is cheap and stateless per call.
classifier = CurrencyClassifier()
