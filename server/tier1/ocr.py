"""
Text / signboard reading — Tier 1.

Backend priority: RapidOCR (ONNX-packaged PaddleOCR, self-contained —
no separate `tesseract` binary/PATH dependency, no PaddlePaddle
framework install) > PaddleOCR native > pytesseract. RapidOCR is the
one actually shipped in production here — benchmarked head-to-head
against tesseract on the same photographed screen text: at a 640px
capture, tesseract read 37 characters of garbage at 0.15 confidence
(rejected by the confidence floor) while RapidOCR read 1398 usable
characters at 0.68 confidence. Tesseract requires the `tesseract`
binary to be present on PATH, which is an easy way for OCR to quietly
stop working across machines/deploys; RapidOCR has no such external
dependency. All three are lazy-imported so the base server boots
without any of them installed, and report themselves unavailable
rather than crashing.
"""
from __future__ import annotations
import logging
import re
from dataclasses import dataclass
from typing import Optional

import numpy as np

log = logging.getLogger("setu.ocr")


@dataclass
class OCRResult:
    text: str
    mean_confidence: float   # 0..1, 0.0 when no text / backend gives no signal
    backend: str


def clean_ocr_text(text: str) -> str:
    """Sanitize raw OCR output before it's spoken or handed to the VLM
    for reasoning. OCR noise (repeated whitespace, run-on dashes from
    misread table borders) otherwise pollutes both the spoken transcript
    and the reasoning prompt."""
    if not text:
        return ""
    text = re.sub(r"[\r\t]", " ", text)
    text = re.sub(r"\n+", "\n", text)
    text = re.sub(r" +", " ", text)
    text = re.sub(r"-{2,}", "-", text)
    return text.strip()


class OCREngine:
    def __init__(self):
        self.backend: Optional[str] = None
        self._rapid = None
        self._paddle = None
        self._load()

    def _load(self) -> None:
        try:
            from rapidocr_onnxruntime import RapidOCR  # lazy
            self._rapid = RapidOCR()
            self.backend = "rapidocr"
            log.info("OCR backend: RapidOCR (ONNX)")
            return
        except ImportError:
            pass
        except Exception as e:
            log.warning("RapidOCR present but failed to initialise: %s", e)

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
        """Back-compat convenience: text only, no confidence. Prefer read_with_confidence()."""
        return self.read_with_confidence(bgr_frame).text

    def read_with_confidence(self, bgr_frame: np.ndarray) -> OCRResult:
        if self.backend == "rapidocr":
            result, _elapse = self._rapid(bgr_frame)
            lines: list[str] = []
            confs: list[float] = []
            for _box, text, conf in result or []:
                lines.append(text)
                confs.append(float(conf))
            text_out = clean_ocr_text(" ".join(lines))
            mean_conf = (sum(confs) / len(confs)) if confs else 0.0
            return OCRResult(text=text_out, mean_confidence=mean_conf, backend=self.backend)

        if self.backend == "paddleocr":
            result = self._paddle.ocr(bgr_frame, cls=True)
            lines: list[str] = []
            confs: list[float] = []
            for page in result or []:
                for _box, (text, conf) in page:
                    lines.append(text)
                    confs.append(float(conf))
            text_out = clean_ocr_text(" ".join(lines))
            mean_conf = (sum(confs) / len(confs)) if confs else 0.0
            return OCRResult(text=text_out, mean_confidence=mean_conf, backend=self.backend)

        if self.backend == "tesseract":
            import pytesseract
            import cv2
            gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
            data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
            words: list[str] = []
            confs: list[float] = []
            for word, conf_str in zip(data["text"], data["conf"]):
                word = word.strip()
                if not word:
                    continue
                try:
                    conf = float(conf_str)
                except (TypeError, ValueError):
                    continue
                if conf < 0:   # tesseract emits -1 for non-text regions
                    continue
                words.append(word)
                confs.append(conf / 100.0)   # tesseract reports 0..100
            text_out = clean_ocr_text(" ".join(words))
            mean_conf = (sum(confs) / len(confs)) if confs else 0.0
            return OCRResult(text=text_out, mean_confidence=mean_conf, backend=self.backend)

        raise RuntimeError(
            "No OCR backend installed. pip install -r requirements-full.txt "
            "(RapidOCR), or install tesseract + pytesseract as a lighter fallback."
        )


engine = OCREngine()
