"""
Cheap, model-free frame quality checks that run before any inference.

This module does double duty, which is the point of it:
1. It rejects frames not worth classifying (saves compute, avoids
   confidently-wrong answers on garbage input).
2. Its outputs feed the client's audio "sonar" guidance loop, since a
   blind user cannot see a viewfinder to aim the camera themselves.

Everything here is plain OpenCV/NumPy — no model, no GPU, <5ms on a
640px frame on a laptop CPU.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from server import config


@dataclass
class GateResult:
    sharpness: float           # variance of Laplacian; higher = sharper
    mean_luminance: float      # 0-255
    clipped_fraction: float    # fraction of pixels near white/black (glare/underexposure)
    motion: Optional[float]    # mean abs diff vs previous frame, None on first frame
    framing_score: float       # 0..1 combined score for the audio sonar
    accept: bool                # True if good enough to run Tier 1 inference on
    torch_suggested: bool
    hint: Optional[str]         # one of config.PHRASES values, or None if nothing to say


def _sharpness(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _exposure(gray: np.ndarray) -> tuple[float, float]:
    mean_lum = float(gray.mean())
    clipped = np.count_nonzero((gray < 8) | (gray > 247)) / gray.size
    return mean_lum, float(clipped)


def _motion(gray: np.ndarray, prev_gray: Optional[np.ndarray]) -> Optional[float]:
    if prev_gray is None or prev_gray.shape != gray.shape:
        return None
    return float(np.mean(cv2.absdiff(gray, prev_gray)))


def assess(bgr_frame: np.ndarray, prev_gray: Optional[np.ndarray]) -> tuple[GateResult, np.ndarray]:
    """
    Returns (GateResult, gray_frame). Caller keeps gray_frame and passes it
    back in as prev_gray on the next call — that's the only state this
    function needs, and keeping it caller-owned makes the function trivially
    testable.
    """
    gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
    sharpness = _sharpness(gray)
    mean_lum, clipped = _exposure(gray)
    motion = _motion(gray, prev_gray)

    hint: Optional[str] = None
    torch_suggested = False

    too_dark = mean_lum < config.GATE_MIN_LUMINANCE
    too_blurred = sharpness < config.GATE_SHARPNESS_FLOOR
    too_glary = clipped > config.GATE_MAX_CLIPPED_FRACTION
    too_much_motion = motion is not None and motion > config.GATE_MAX_MOTION

    if too_dark:
        hint = config.PHRASES["abstain_dark"]
        torch_suggested = True
    elif too_glary:
        hint = config.PHRASES["abstain_low_conf"]
    elif too_much_motion:
        hint = config.PHRASES["abstain_motion"]
    elif too_blurred:
        hint = config.PHRASES["abstain_blurred"]

    accept = not (too_dark or too_blurred or too_glary or too_much_motion)

    # Framing score: normalise sharpness and luminance into 0..1 and combine.
    # This is a heuristic, not a learned metric — good enough to drive a
    # continuous tone, not precise enough to gate a safety decision alone
    # (the discrete `accept` flag above does that).
    sharp_norm = min(sharpness / (config.GATE_SHARPNESS_FLOOR * 3), 1.0)
    lum_norm = 1.0 - abs(mean_lum - 130.0) / 130.0
    lum_norm = max(0.0, min(lum_norm, 1.0))
    framing_score = 0.6 * sharp_norm + 0.4 * lum_norm
    if not accept:
        framing_score = min(framing_score, 0.5)  # never claim "ready" while rejecting

    return (
        GateResult(
            sharpness=sharpness,
            mean_luminance=mean_lum,
            clipped_fraction=clipped,
            motion=motion,
            framing_score=round(framing_score, 3),
            accept=accept,
            torch_suggested=torch_suggested,
            hint=hint,
        ),
        gray,
    )
