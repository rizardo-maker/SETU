"""
Currency denomination detector — Tier 1 "reflex" path.

Uses a YOLO detector trained on the Indian currency dataset (see
`training/currency_yolo/` and `datasets/currency/`). Ships with
`models/currency_best.pt` already trained on 10 classes:
  10_new, 10_old, 20, 50_new, 50_old, 100_new, 100_old, 200, 500, 2000

Why YOLO rather than a single-image classifier: a user may hold more
than one note, and the per-detection confidence YOLO gives us is
what we actually need to decide "abstain or speak". The old
placeholder ONNX single-image classifier is retained as a fallback
below only for teams who prefer that path — it self-reports
`.ready = False` until they train and drop in weights.

Design principle carried over from the ONNX version: `ready` is
False when nothing works, and we tell the user honestly rather than
fabricating an answer with an untrained graph.
"""
from __future__ import annotations
import json
import logging
import re
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from server import config

log = logging.getLogger("setu.currency")


# ---- Denomination parsing --------------------------------------------------

def extract_denomination(label: str) -> int:
    """`10_new` -> 10, `100_old` -> 100, `2000` -> 2000, unknown -> 0."""
    match = re.search(r"\d+", label)
    return int(match.group(0)) if match else 0


# ---- Multi-frame arbitration -----------------------------------------------
# YOLO gives us per-detection confidence directly, so we don't need temperature
# scaling like the ONNX classifier did. We just want to require N consecutive
# frames to agree on the same *multiset* of denominations before speaking, to
# avoid speaking a jittery mid-motion frame.

@dataclass
class YOLODecision:
    answered: bool
    speak: str
    denominations: list[int] = field(default_factory=list)   # e.g. [500, 100]
    total_value: int = 0
    confidence: float = 0.0   # min over all detections in the winning frame
    detection_count: int = 0


class YOLOFrameArbiter:
    """Requires `frames_required` consecutive frames to agree on the same
    (sorted) tuple of denominations before speaking. Any change resets."""

    def __init__(self, frames_required: int = config.CURRENCY_FRAMES_REQUIRED):
        self.frames_required = frames_required
        self._buffer: deque[tuple[int, ...]] = deque(maxlen=frames_required)
        self._last_frame: Optional[dict] = None

    def reset(self) -> None:
        self._buffer.clear()
        self._last_frame = None

    def submit(self, detections: list[dict]) -> YOLODecision:
        """
        Feed one frame's YOLO detections (each: {label, denomination, confidence}).
        Returns YOLODecision — check .answered before treating output as final.
        """
        if not detections:
            self._buffer.clear()
            self._last_frame = None
            return YOLODecision(answered=False, speak="No currency detected. Hold a note in view.")

        denoms = tuple(sorted((d["denomination"] for d in detections), reverse=True))
        self._buffer.append(denoms)
        self._last_frame = {"denoms": denoms, "detections": detections}

        # Wait until the buffer's full AND every frame in the buffer agrees.
        if len(self._buffer) < self.frames_required:
            return YOLODecision(
                answered=False,
                speak="Hold steady, still checking.",
                denominations=list(denoms),
                total_value=sum(denoms),
                detection_count=len(detections),
            )

        if not all(b == denoms for b in self._buffer):
            return YOLODecision(
                answered=False,
                speak="Hold steady, still checking.",
                denominations=list(denoms),
                total_value=sum(denoms),
                detection_count=len(detections),
            )

        min_conf = min(float(d["confidence"]) for d in detections)
        if min_conf < config.CURRENCY_CONF_FLOOR:
            return YOLODecision(
                answered=False,
                speak=config.PHRASES["abstain_low_conf"],
                denominations=list(denoms),
                total_value=sum(denoms),
                confidence=min_conf,
                detection_count=len(detections),
            )

        total = sum(denoms)
        speak = _speak_for(list(denoms), total)
        return YOLODecision(
            answered=True,
            speak=speak,
            denominations=list(denoms),
            total_value=total,
            confidence=min_conf,
            detection_count=len(detections),
        )


def _speak_for(denominations: list[int], total: int) -> str:
    """Compact spoken phrasing for one or many notes."""
    if not denominations:
        return "No currency detected."
    if len(denominations) == 1:
        return f"{denominations[0]} rupees."
    # Group identical denominations for readability: [500,500,100] -> "two 500s and a 100"
    counts: dict[int, int] = {}
    for d in denominations:
        counts[d] = counts.get(d, 0) + 1
    parts: list[str] = []
    for denom in sorted(counts.keys(), reverse=True):
        n = counts[denom]
        parts.append(f"{n} times {denom}" if n > 1 else f"{denom}")
    joined = ", ".join(parts)
    return f"{joined}. Total {total} rupees."


# ---- Detector --------------------------------------------------------------

class CurrencyDetector:
    """YOLO-based multi-note detector. Falls back to `.ready = False` if
    ultralytics is not installed or the trained weights aren't in models/."""

    def __init__(self):
        self.ready = False
        self._model = None
        self.labels: list[str] = []      # kept for main.py compatibility
        self._load()

    def _load(self) -> None:
        if not config.CURRENCY_YOLO_MODEL_PATH.exists():
            log.warning(
                "No currency YOLO model at %s — Tier 1 currency mode will report "
                "itself unavailable until you drop in trained weights.",
                config.CURRENCY_YOLO_MODEL_PATH,
            )
            return
        try:
            from ultralytics import YOLO  # lazy
        except ImportError:
            log.warning("ultralytics not installed — pip install -r requirements-full.txt to enable currency mode")
            return
        try:
            self._model = YOLO(str(config.CURRENCY_YOLO_MODEL_PATH))
        except Exception as e:
            log.warning("Failed to load currency model %s: %s", config.CURRENCY_YOLO_MODEL_PATH, e)
            return
        self.labels = list(self._model.names.values())
        self.ready = True
        log.info("Currency detector loaded (YOLO): %d classes", len(self.labels))

    def detect(
        self,
        bgr_frame: np.ndarray,
        conf_threshold: float = config.CURRENCY_CONF_FLOOR,
        iou_threshold: float = 0.45,
    ) -> list[dict]:
        """Returns list of {label, denomination, confidence, bbox}."""
        if not self.ready:
            raise RuntimeError(
                "Currency model isn't loaded. Drop weights at "
                f"{config.CURRENCY_YOLO_MODEL_PATH} (see training/currency_yolo/README.md)."
            )
        results = self._model.predict(
            source=bgr_frame,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False,
        )
        detections: list[dict] = []
        if not results:
            return detections
        res = results[0]
        if res.boxes is None or len(res.boxes) == 0:
            return detections
        xyxy = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        cls_ids = res.boxes.cls.cpu().numpy().astype(int)
        for box, score, cid in zip(xyxy, confs, cls_ids):
            label = res.names.get(cid, str(cid))
            detections.append({
                "label": label,
                "denomination": extract_denomination(label),
                "confidence": round(float(score), 4),
                "bbox": [round(float(c), 1) for c in box],
            })
        return detections


# Module-level singleton — one model, shared across connections.
classifier = CurrencyDetector()
