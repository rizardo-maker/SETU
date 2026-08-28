"""
Text / signboard / document reading — 100% Offline Local Engine (No LLM).
Based on the high-accuracy SETU OCR-to-Speech pipeline:
  - RapidOCR (PaddleOCR ONNX Runtime engine - ultra-fast & high accuracy)
  - Spatial Line & Paragraph Reconstruction (top-to-bottom, natural reading order)
  - Adaptive Contrast Enhancement (CLAHE) for faint, small, or low-light text
  - Tesseract OCR fallback
"""
from __future__ import annotations
import logging
import re
from dataclasses import dataclass
from typing import Optional, List

import cv2
import numpy as np

log = logging.getLogger("setu.ocr")


@dataclass
class OCRResult:
    text: str
    mean_confidence: float   # 0..1, 0.0 when no text
    backend: str
    chunks: List[str]


def clean_ocr_text(text: str) -> str:
    """Sanitizes OCR text for natural audio pronunciation while preserving all content."""
    if not text:
        return ""
    # Normalize tabs and weird spacing
    text = re.sub(r'[\r\t]', ' ', text)
    text = re.sub(r' +', ' ', text)
    # Fix repeated hyphenation or OCR artifacts
    text = re.sub(r'-{3,}', '---', text)
    # Remove isolated non-ASCII noise glyphs but keep all letters, numbers, punctuation
    lines = []
    for line in text.split('\n'):
        cleaned_line = line.strip()
        if cleaned_line:
            lines.append(cleaned_line)
    return "\n".join(lines).strip()


def chunk_text(text: str, max_chunk_len: int = 250) -> List[str]:
    """Splits full document text into user-friendly audio chunks."""
    if not text:
        return []
    paragraphs = text.split("\n\n")
    chunks = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(para) <= max_chunk_len:
            chunks.append(para)
        else:
            sentences = re.split(r'(?<=[.!?])\s+', para)
            current_chunk = ""
            for sent in sentences:
                if len(current_chunk) + len(sent) + 1 <= max_chunk_len:
                    current_chunk += (" " if current_chunk else "") + sent
                else:
                    if current_chunk:
                        chunks.append(current_chunk)
                    current_chunk = sent
            if current_chunk:
                chunks.append(current_chunk)
    return chunks if chunks else [text]


class OCREngine:
    def __init__(self):
        self.backend: Optional[str] = None
        self._engine = None
        self._load()

    def _load(self) -> None:
        # 1. Try RapidOCR (PaddleOCR ONNX engine)
        try:
            from rapidocr_onnxruntime import RapidOCR
            self._engine = RapidOCR()
            self.backend = "rapidocr"
            log.info("OCR backend: RapidOCR (PaddleOCR ONNX Runtime)")
            return
        except ImportError:
            pass
        except Exception as e:
            log.warning("RapidOCR present but failed to initialise: %s", e)

        # 2. Try native PaddleOCR
        try:
            from paddleocr import PaddleOCR
            self._engine = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
            self.backend = "paddleocr"
            log.info("OCR backend: PaddleOCR Native")
            return
        except ImportError:
            pass
        except Exception as e:
            log.warning("PaddleOCR present but failed to initialise: %s", e)

        # 3. Try Tesseract OCR
        try:
            import pytesseract
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
        return self.read_with_confidence(bgr_frame).text

    def read_with_confidence(self, bgr_frame: np.ndarray) -> OCRResult:
        if not self.ready:
            raise RuntimeError("No OCR backend installed.")

        # 1. RapidOCR Execution with Complete Extraction & Spatial Ordering
        if self.backend == "rapidocr":
            # Pass 1: Standard frame
            result, _ = self._engine(bgr_frame)

            # Pass 2: If very few results, try adaptive contrast enhancement
            if not result or len(result) < 2:
                try:
                    gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
                    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
                    enhanced = clahe.apply(gray)
                    enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
                    res2, _ = self._engine(enhanced_bgr)
                    if res2 and (not result or len(res2) > len(result)):
                        result = res2
                except Exception:
                    pass

            if not result:
                return OCRResult(text="", mean_confidence=0.0, backend=self.backend, chunks=[])

            # Extract all detected bounding boxes and sort in natural reading order
            items = []
            confs = []
            for item in result:
                if len(item) >= 2:
                    box = item[0]
                    text_str = str(item[1]).strip()
                    conf = float(item[2]) if len(item) > 2 else 0.90
                    if text_str:
                        # Compute center Y and top-left X for sorting
                        y_center = (box[0][1] + box[2][1]) / 2.0 if len(box) >= 3 else box[0][1]
                        x_left = box[0][0]
                        items.append((y_center, x_left, text_str, conf))
                        confs.append(conf)

            # Sort top-to-bottom (bucketed by vertical row ~25px) then left-to-right
            items.sort(key=lambda it: (round(it[0] / 24.0) * 24.0, it[1]))
            ordered_lines = [it[2] for it in items]
            full_text = clean_ocr_text("\n".join(ordered_lines))
            mean_conf = (sum(confs) / len(confs)) if confs else 0.0

            return OCRResult(
                text=full_text,
                mean_confidence=round(mean_conf, 4),
                backend=self.backend,
                chunks=chunk_text(full_text),
            )

        # 2. Native PaddleOCR Execution
        if self.backend == "paddleocr":
            result = self._engine.ocr(bgr_frame, cls=True)
            lines: list[str] = []
            confs: list[float] = []
            for page in result or []:
                for _box, (text, conf) in page:
                    text_str = str(text).strip()
                    if text_str:
                        lines.append(text_str)
                        confs.append(float(conf))
            full_text = clean_ocr_text("\n".join(lines))
            mean_conf = (sum(confs) / len(confs)) if confs else 0.0
            return OCRResult(
                text=full_text,
                mean_confidence=round(mean_conf, 4),
                backend=self.backend,
                chunks=chunk_text(full_text),
            )

        # 3. Tesseract OCR Execution
        if self.backend == "tesseract":
            import pytesseract
            gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
            # PSM 6: Assume a single uniform block of text for complete extraction
            custom_config = r'--oem 3 --psm 6'
            extracted = pytesseract.image_to_string(gray, config=custom_config)
            full_text = clean_ocr_text(extracted)
            return OCRResult(
                text=full_text,
                mean_confidence=0.85 if full_text else 0.0,
                backend=self.backend,
                chunks=chunk_text(full_text),
            )

        raise RuntimeError("OCR Engine failure")

    def find_target(self, bgr_frame: np.ndarray, target_query: str) -> Optional[dict]:
        """
        Searches the frame for a specific navigation sign / room target (e.g. 'C-214', 'Exit', 'Room 101').
        Returns dict with match text, direction ('on your left', 'straight ahead', 'on your right'), and confidence.
        """
        if not self.ready or not target_query:
            return None

        target_norm = re.sub(r'[^a-zA-Z0-9]', '', target_query.lower())
        if not target_norm:
            return None

        h, w = bgr_frame.shape[:2]

        if self.backend == "rapidocr":
            result, _ = self._engine(bgr_frame)
            if not result:
                try:
                    gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
                    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
                    enhanced = cv2.cvtColor(clahe.apply(gray), cv2.COLOR_GRAY2BGR)
                    result, _ = self._engine(enhanced)
                except Exception:
                    pass

            if not result:
                return None

            for item in result:
                if len(item) >= 2:
                    box = item[0]
                    text_str = str(item[1]).strip()
                    conf = float(item[2]) if len(item) > 2 else 0.90
                    clean_text = re.sub(r'[^a-zA-Z0-9]', '', text_str.lower())
                    if target_norm in clean_text or clean_text in target_norm:
                        x_coords = [p[0] for p in box] if box else [0]
                        x_center = (sum(x_coords) / len(x_coords)) / w if w > 0 else 0.5
                        if x_center < 0.35:
                            direction = "on your left"
                        elif x_center > 0.65:
                            direction = "on your right"
                        else:
                            direction = "straight ahead"
                        return {
                            "found": True,
                            "target": target_query,
                            "matched_text": text_str,
                            "direction": direction,
                            "confidence": round(conf, 3),
                        }

        return None


engine = OCREngine()
