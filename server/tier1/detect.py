"""
Obstacle / object detection + collision alerting — Tier 1.

Two closely related jobs, one shared YOLO model:

  1. `detect(frame)` — Object mode: announces things in view at conversational range: "close by: chair, table."
  2. `scan_for_collision(frame)` — Collision mode: continuous stream. Only fires when a *hazard-class*
     object fills enough of the frame to be considered near/very-near, and skips small clutter
     (bottles, cups, keyboards etc.) that shouldn't spook the user or trigger false alarms while walking.

Uses Ultralytics YOLO (AGPL-3.0). Weights ship at models/yolo11n.pt.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from server import config

log = logging.getLogger("setu.detect")

# COCO classes we announce for general awareness
ANNOUNCE_CLASSES = {
    "person", "chair", "couch", "dining table", "bed", "door",
    "stairs", "car", "bicycle", "motorcycle", "dog", "bench",
}

# Classes that trigger a *collision alert* if they fill enough of the frame.
HAZARD_CLASSES = {
    # Structural / furniture — the "wall/bench/table" family
    "bench", "chair", "couch", "bed", "dining table", "potted plant",
    "toilet", "refrigerator", "oven", "sink", "tv", "door", "stairs",
    # Vehicles — hazards on the walking path
    "car", "truck", "bus", "motorcycle", "bicycle", "train",
    # People and larger animals
    "person", "dog", "horse", "cow", "sheep", "cat", "bear",
    # Street Blockers
    "traffic light", "stop sign", "fire hydrant", "parking meter",
}

CLOSE_RANGE_AREA_FRACTION = 0.18


@dataclass
class Detection:
    label: str
    confidence: float
    close: bool


@dataclass
class CollisionThreat:
    label: str
    confidence: float
    area_fraction: float
    severity: str            # "warn" (close) or "urgent" (very close / imminent)
    bbox: list[float]        # [x1, y1, x2, y2]


class ObstacleDetector:
    def __init__(self):
        self._model = None
        self.ready = False
        self._load()

    def _load(self) -> None:
        try:
            from ultralytics import YOLO
        except ImportError:
            log.warning("ultralytics not installed — obstacle/collision modes unavailable.")
            return
        try:
            weights = str(config.YOLO_GENERAL_MODEL_PATH) if config.YOLO_GENERAL_MODEL_PATH.exists() else "yolo11n.pt"
            self._model = YOLO(weights)
            self.ready = True
            log.info("Obstacle/collision detector loaded (%s, AGPL-3.0)", weights)
        except Exception as e:
            log.warning("YOLO failed to load: %s", e)

    def frame_is_dominated_by(self, bgr_frame: np.ndarray, labels: set[str], area_fraction: float = 0.25, conf: float = 0.4) -> bool:
        if not self.ready:
            return False
        h, w = bgr_frame.shape[:2]
        frame_area = float(h * w)
        results = self._model.predict(bgr_frame, conf=conf, verbose=False)
        for r in results:
            for box in r.boxes:
                label = r.names[int(box.cls[0])]
                if label not in labels:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                if ((x2 - x1) * (y2 - y1)) / frame_area >= area_fraction:
                    return True
        return False

    def detect(self, bgr_frame: np.ndarray, conf: float = 0.35) -> list[Detection]:
        """Object detection mode: detect objects in view."""
        if not self.ready:
            raise RuntimeError("Object detector not available.")
        h, w = bgr_frame.shape[:2]
        frame_area = h * w
        results = self._model.predict(bgr_frame, conf=conf, verbose=False)
        out: list[Detection] = []
        for r in results:
            for box in r.boxes:
                label = r.names[int(box.cls[0])]
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                area_fraction = ((x2 - x1) * (y2 - y1)) / frame_area
                out.append(Detection(label=label, confidence=confidence,
                                     close=area_fraction > CLOSE_RANGE_AREA_FRACTION))
        return out

    def scan_for_collision(self, bgr_frame: np.ndarray) -> list[CollisionThreat]:
        """
        Collision mode: return a list of HAZARD-class objects big enough
        in the frame to matter, sorted by severity (urgent first).

        Empty list = "path is clear".
        """
        if not self.ready:
            raise RuntimeError("Collision detector not available.")

        # Safety check: intentional currency inspection should not raise collision alarms
        try:
            from server.tier1 import currency
            if currency.classifier.ready:
                c_dets = currency.classifier.detect(bgr_frame, conf_threshold=0.30)
                if c_dets:
                    return []
        except Exception:
            pass

        h, w = bgr_frame.shape[:2]
        frame_area = float(h * w)
        results = self._model.predict(bgr_frame, conf=config.COLLISION_CONF_FLOOR, verbose=False)
        threats: list[CollisionThreat] = []

        for r in results:
            for box in r.boxes:
                label = r.names[int(box.cls[0])]
                if label not in HAZARD_CLASSES:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                area = max(0.0, (x2 - x1) * (y2 - y1))
                area_fraction = area / frame_area if frame_area > 0 else 0.0
                if area_fraction < config.COLLISION_CLOSE_AREA_FRACTION:
                    continue   # detected but not close enough to alert on
                severity = "urgent" if area_fraction >= config.COLLISION_URGENT_AREA_FRACTION else "warn"
                threats.append(CollisionThreat(
                    label=label,
                    confidence=float(box.conf[0]),
                    area_fraction=round(area_fraction, 3),
                    severity=severity,
                    bbox=[float(x1), float(y1), float(x2), float(y2)],
                ))

        threats.sort(key=lambda t: (t.severity == "warn", -t.area_fraction))  # urgent first, then biggest
        return threats

    @staticmethod
    def speak(detections: list[Detection]) -> str:
        if not detections:
            return "No clear objects detected ahead."
        close = [d for d in detections if d.close]
        if close:
            names = ", ".join(sorted({d.label for d in close}))
            return f"Close by: {names}."
        names = ", ".join(sorted({d.label for d in detections}))
        return f"I can see: {names}."

    @staticmethod
    def speak_collision(threats: list[CollisionThreat]) -> tuple[str, Optional[str]]:
        """
        Returns (spoken_alert, severity_level).
        severity_level is None when the path is clear; otherwise "warn" or "urgent".
        """
        if not threats:
            return (config.PHRASES["collision_clear"], None)

        by_label: dict[str, list[CollisionThreat]] = {}
        for t in threats:
            by_label.setdefault(t.label, []).append(t)

        urgent = [lbl for lbl, ts in by_label.items() if any(t.severity == "urgent" for t in ts)]
        warn = [lbl for lbl, ts in by_label.items() if lbl not in urgent]

        if urgent:
            names = ", ".join(sorted(urgent))
            return (f"Stop. {names} right in front of you.", "urgent")

        names = ", ".join(sorted(warn))
        return (f"Careful, {names} close ahead.", "warn")


detector = ObstacleDetector()
