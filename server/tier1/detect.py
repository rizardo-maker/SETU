"""
Obstacle / object detection — Tier 1.

Uses Ultralytics YOLO if installed. NOTE — licensing: ultralytics is
AGPL-3.0. That's a deliberate, accepted choice for this project (see
the project document's licensing section) — if you ever need a
permissive licence, swap for an Apache-2.0 detector such as
MMDetection or a torchvision (BSD) detection model; this module's
interface (`detect(frame) -> list[Detection]`) is what you'd keep.

Lazy-imported so the base server runs without it installed.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass

import numpy as np

from server import config

log = logging.getLogger("setu.detect")

# A small set of classes worth announcing at conversational range for a
# blind user walking indoors/outdoors. COCO class names — trim or extend
# once you see what fires too often in real testing.
ANNOUNCE_CLASSES = {
    "person", "chair", "couch", "dining table", "bed", "door",
    "stairs", "car", "bicycle", "motorcycle", "dog", "bench",
}

CLOSE_RANGE_AREA_FRACTION = 0.18  # box area / frame area above which we call it "close"


@dataclass
class Detection:
    label: str
    confidence: float
    close: bool


class ObstacleDetector:
    def __init__(self):
        self._model = None
        self.ready = False
        self._load()

    def _load(self) -> None:
        try:
            from ultralytics import YOLO  # lazy
            # 'n' (nano) — smallest/fastest, appropriate for a CPU-only laptop.
            self._model = YOLO("yolo11n.pt")
            self.ready = True
            log.info("Obstacle detector loaded (YOLO11n, AGPL-3.0 — see README)")
        except ImportError:
            log.warning("ultralytics not installed — obstacle mode unavailable. "
                        "pip install -r requirements-full.txt to enable it.")
        except Exception as e:
            log.warning("YOLO failed to load: %s", e)

    def detect(self, bgr_frame: np.ndarray, conf: float = 0.4) -> list[Detection]:
        if not self.ready:
            raise RuntimeError("Obstacle detector not available.")
        h, w = bgr_frame.shape[:2]
        frame_area = h * w
        results = self._model.predict(bgr_frame, conf=conf, verbose=False)
        out: list[Detection] = []
        for r in results:
            for box in r.boxes:
                label = r.names[int(box.cls[0])]
                if label not in ANNOUNCE_CLASSES:
                    continue
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                area_fraction = ((x2 - x1) * (y2 - y1)) / frame_area
                out.append(Detection(label=label, confidence=confidence,
                                      close=area_fraction > CLOSE_RANGE_AREA_FRACTION))
        return out

    @staticmethod
    def speak(detections: list[Detection]) -> str:
        if not detections:
            return "Nothing notable ahead."
        close = [d for d in detections if d.close]
        if close:
            names = ", ".join(sorted({d.label for d in close}))
            return f"Close by: {names}."
        names = ", ".join(sorted({d.label for d in detections}))
        return f"I can see: {names}."


detector = ObstacleDetector()
