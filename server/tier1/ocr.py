"""
Text / signboard reading — Tier 1.

Tries PaddleOCR first (better on skewed, real-world text per the
project doc's comparison), falls back to pytesseract if that's not
installed, and reports itself unavailable rather than crashing if
neither is. Both are lazy-imported so the base server boots without
either installed.
"""
from __future__ import annotations
import logging
from typing import Optional

import numpy as np

log = logging.getLogger("setu.ocr")


class OCREngine:
    def __init__(self):
        self.backend: Optional[str] = None
        self._paddle = None
        self._load()

    def _load(self) -> None:
        try:
            from paddleocr import PaddleOCR  # lazy
            self._paddle = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
            self.backend = "paddleocr"
            log.info("OCR backend: PaddleOCR")
            return
        except ImportError:
            pass
        except Exception as e:  # PaddleOCR's own init can throw on missing model downloads
            log.warning("PaddleOCR present but failed to initialise: %s", e)

        try:
            import pytesseract  # lazy
            import shutil
            if shutil.which("tesseract") is None:
                raise RuntimeError("tesseract binary not found on PATH")
            self.backend = "tesseract"
            log.info("OCR backend: pytesseract (fallback)")
            return
        except Exception as e:
            log.warning("No OCR backend available: %s", e)
        self.backend = None

    @property
    def ready(self) -> bool:
        return self.backend is not None

    def read(self, bgr_frame: np.ndarray) -> str:
        if self.backend == "paddleocr":
            result = self._paddle.ocr(bgr_frame, cls=True)
            lines = []
            for page in result or []:
                for _box, (text, _conf) in page:
                    lines.append(text)
            return " ".join(lines).strip()

        if self.backend == "tesseract":
            import pytesseract
            import cv2
            gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
            return pytesseract.image_to_string(gray).strip()

        raise RuntimeError(
            "No OCR backend installed. pip install -r requirements-full.txt "
            "(PaddleOCR), or install tesseract + pytesseract as a lighter fallback."
        )


engine = OCREngine()
