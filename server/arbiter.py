"""
Calibration and abstention logic for Tier 1 classifiers.

Kept separate from quality_gate.py deliberately: the gate decides
"is this frame worth looking at", the arbiter decides "given what the
model saw, are we confident enough to speak an answer". Two different
failure modes, two different modules, both feeding the same principle —
abstain rather than guess.

    "We would rather be useless ten percent of the time than
     wrong two percent of the time."

Fit CURRENCY_TEMPERATURE properly once you have a held-out validation
set (see training/train_currency_classifier.py) — the value in
config.py is a placeholder, not a measurement.
"""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

from server import config


@dataclass
class Decision:
    answered: bool
    label_idx: Optional[int]
    confidence: float
    margin: float
    speak: str


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def calibrated_probs(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Temperature scaling: divide logits before softmax. temperature > 1
    softens an overconfident model; temperature < 1 sharpens an
    underconfident one. Fit on a held-out set — see training/README."""
    return _softmax(np.asarray(logits, dtype=np.float64) / temperature)


class MultiFrameArbiter:
    """
    One instance per active classification attempt (e.g. per WebSocket
    connection, reset when the user switches modes). Buffers logits
    across consecutive accepted frames and only commits to an answer
    once enough frames agree.
    """

    def __init__(
        self,
        labels: list[str],
        temperature: float = config.CURRENCY_TEMPERATURE,
        conf_floor: float = config.CURRENCY_CONF_FLOOR,
        margin_floor: float = config.CURRENCY_MARGIN_FLOOR,
        frames_required: int = config.CURRENCY_FRAMES_REQUIRED,
    ):
        self.labels = labels
        self.temperature = temperature
        self.conf_floor = conf_floor
        self.margin_floor = margin_floor
        self.frames_required = frames_required
        self._buffer: deque[np.ndarray] = deque(maxlen=frames_required)

    def reset(self) -> None:
        self._buffer.clear()

    def submit(self, logits: np.ndarray) -> Decision:
        """Feed one frame's raw logits. Returns a Decision — check
        `.answered` before treating `.label_idx` as meaningful; while
        the buffer is still filling, this returns answered=False with
        an in-progress message rather than a premature guess."""
        probs = calibrated_probs(logits, self.temperature)
        self._buffer.append(probs)

        if len(self._buffer) < self.frames_required:
            return Decision(
                answered=False,
                label_idx=None,
                confidence=0.0,
                margin=0.0,
                speak="Hold steady, still checking.",
            )

        avg = np.mean(np.stack(self._buffer), axis=0)
        order = avg.argsort()[::-1]
        top, second = int(order[0]), int(order[1])
        conf, margin = float(avg[top]), float(avg[top] - avg[second])

        if conf < self.conf_floor or margin < self.margin_floor:
            return Decision(
                answered=False,
                label_idx=None,
                confidence=conf,
                margin=margin,
                speak=config.PHRASES["abstain_low_conf"],
            )

        label = self.labels[top] if top < len(self.labels) else str(top)
        return Decision(
            answered=True,
            label_idx=top,
            confidence=conf,
            margin=margin,
            speak=f"{label} rupees",
        )
