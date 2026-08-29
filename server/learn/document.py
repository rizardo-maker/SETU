"""
Document ingestion, text extraction, and structure parsing for SETU Learn.
Supports:
  - PDF: Fast digital text extraction via PyMuPDF (fitz) with automatic OCR fallback for scanned pages.
  - Images (PNG, JPG, JPEG): OCR via SETU RapidOCR engine.
  - Plain Text (TXT).
"""
from __future__ import annotations
import io
import logging
import os
import re
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional

import cv2
import numpy as np
import pymupdf

from server.tier1 import ocr

log = logging.getLogger("setu.learn.document")


@dataclass
class DocumentSection:
    section_id: str
    heading: str
    level: int  # 1 for main heading, 2 for subheading
    page_start: int
    page_end: int
    paragraphs: List[str]
    full_text: str


@dataclass
class ParsedDocument:
    document_id: str
    filename: str
    title: str
    page_count: int
    sections: List[DocumentSection]
    full_text: str
    created_at: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "filename": self.filename,
            "title": self.title,
            "page_count": self.page_count,
            "sections": [asdict(s) for s in self.sections],
            "full_text": self.full_text,
            "created_at": self.created_at,
        }


def _clean_paragraph(text: str) -> str:
    """Normalizes whitespace and broken hyphens in extracted text."""
    if not text:
        return ""
    text = re.sub(r'[\r\t]', ' ', text)
    text = re.sub(r' +', ' ', text)
    # Fix hyphenation across line breaks: e.g. "com-\nputer" -> "computer"
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
    # Convert single newlines into spaces within a paragraph
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    return " ".join(lines).strip()


def _is_heading(line: str) -> tuple[bool, int]:
    """
    Heuristic detection of headings & subheadings.
    Returns (is_heading, heading_level).
    """
    clean = line.strip()
    if not clean or len(clean) > 90:
        return False, 0

    # Pattern 1: Numbered headings like "1. Introduction", "2.1 Paging", "Chapter 3: Memory"
    if re.match(r'^(?:Chapter\s+\d+|Unit\s+\d+|\d+\.\d*(?:\.\d+)?)\s+[\w\s]{2,}', clean, re.IGNORECASE):
        dots = clean.split()[0].count('.')
        level = 2 if dots >= 2 else (1 if dots <= 1 else 1)
        return True, level

    # Pattern 2: Short All-Caps Header (e.g. "ABSTRACT", "OPERATING SYSTEMS", "VIRTUAL MEMORY")
    if clean.isupper() and 3 <= len(clean) <= 50 and not clean.endswith('.'):
        return True, 1

    # Pattern 3: Short standalone title line (e.g. "Key Takeaways", "Summary", "Overview")
    if clean.endswith(':') and len(clean) <= 45:
        return True, 2

    return False, 0


def _parse_sections_from_pages(page_texts: List[tuple[int, str]], default_title: str) -> tuple[str, List[DocumentSection]]:
    """
    Constructs hierarchical sections and paragraphs from page-by-page extracted text.
    """
    all_paragraphs: List[tuple[int, str, bool, int]] = []  # (page_num, text, is_header, level)

    doc_title = default_title

    for page_num, raw_page in page_texts:
        if not raw_page.strip():
            continue
        # Split page into raw blocks by double newlines
        blocks = raw_page.split("\n\n")
        for block in blocks:
            lines = [l.strip() for l in block.split("\n") if l.strip()]
            if not lines:
                continue

            # Check if block is a heading
            if len(lines) == 1:
                is_hdr, lvl = _is_heading(lines[0])
                if is_hdr:
                    all_paragraphs.append((page_num, lines[0], True, lvl))
                    if doc_title == default_title and lvl == 1 and len(lines[0]) > 3:
                        doc_title = lines[0]
                    continue

            # Multi-line block: check if first line is a heading
            is_first_hdr, lvl = _is_heading(lines[0])
            if is_first_hdr and len(lines) > 1:
                all_paragraphs.append((page_num, lines[0], True, lvl))
                body = _clean_paragraph("\n".join(lines[1:]))
                if body:
                    all_paragraphs.append((page_num, body, False, 0))
            else:
                body = _clean_paragraph(block)
                if body:
                    all_paragraphs.append((page_num, body, False, 0))

    # Assemble into DocumentSection objects
    sections: List[DocumentSection] = []
    current_heading = "Introduction / Overview"
    current_level = 1
    current_paras: List[str] = []
    current_start_page = page_texts[0][0] if page_texts else 1
    current_end_page = current_start_page

    sec_counter = 1

    for page_num, text, is_hdr, lvl in all_paragraphs:
        if is_hdr:
            if current_paras:
                sections.append(DocumentSection(
                    section_id=f"sec_{sec_counter}",
                    heading=current_heading,
                    level=current_level,
                    page_start=current_start_page,
                    page_end=current_end_page,
                    paragraphs=current_paras,
                    full_text="\n\n".join(current_paras),
                ))
                sec_counter += 1
                current_paras = []
            current_heading = text
            current_level = lvl
            current_start_page = page_num
            current_end_page = page_num
        else:
            current_paras.append(text)
            current_end_page = page_num

    if current_paras or not sections:
        sections.append(DocumentSection(
            section_id=f"sec_{sec_counter}",
            heading=current_heading,
            level=current_level,
            page_start=current_start_page,
            page_end=current_end_page,
            paragraphs=current_paras if current_paras else ["No text content found in section."],
            full_text="\n\n".join(current_paras) if current_paras else "",
        ))

    return doc_title, sections


class DocumentExtractor:
    """Extracts accessible structured text from files with PyMuPDF and RapidOCR fallback."""

    def extract_from_bytes(self, file_bytes: bytes, filename: str) -> ParsedDocument:
        doc_id = str(uuid.uuid4())[:8]
        ext = Path(filename).suffix.lower()
        import time
        t0 = time.time()

        page_texts: List[tuple[int, str]] = []
        default_title = Path(filename).stem.replace("_", " ").replace("-", " ").title()

        # 1. Plain Text Ingestion
        if ext in (".txt", ".md", ".text"):
            try:
                raw = file_bytes.decode("utf-8")
            except UnicodeDecodeError:
                raw = file_bytes.decode("latin-1", errors="ignore")
            page_texts.append((1, raw))
            doc_title, sections = _parse_sections_from_pages(page_texts, default_title)
            full_text = "\n\n".join(s.full_text for s in sections)
            return ParsedDocument(
                document_id=doc_id,
                filename=filename,
                title=doc_title,
                page_count=1,
                sections=sections,
                full_text=full_text,
                created_at=t0,
            )

        # 2. Image Ingestion (OCR)
        if ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
            nparr = np.frombuffer(file_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Could not decode image file.")

            if ocr.engine.ready:
                res = ocr.engine.read_with_confidence(img)
                ocr_text = res.text
            else:
                ocr_text = "OCR engine not available."

            page_texts.append((1, ocr_text))
            doc_title, sections = _parse_sections_from_pages(page_texts, default_title)
            full_text = "\n\n".join(s.full_text for s in sections)
            return ParsedDocument(
                document_id=doc_id,
                filename=filename,
                title=doc_title,
                page_count=1,
                sections=sections,
                full_text=full_text,
                created_at=t0,
            )

        # 3. PDF Ingestion (PyMuPDF with automatic OCR for scanned pages)
        if ext == ".pdf":
            try:
                pdf_doc = pymupdf.open(stream=file_bytes, filetype="pdf")
            except Exception as e:
                raise ValueError(f"Could not open PDF file: {e}")

            page_count = len(pdf_doc)
            log.info("📄 Processing PDF '%s' with %d pages...", filename, page_count)

            for page_idx in range(page_count):
                page_num = page_idx + 1
                page = pdf_doc[page_idx]
                text = page.get_text("text").strip()

                # If the page has little or no digital text (< 35 chars), OCR the rendered page
                if len(text) < 35 and ocr.engine.ready:
                    log.info("🔍 Page %d has no digital text (%d chars); running OCR...", page_num, len(text))
                    try:
                        pix = page.get_pixmap(dpi=150)
                        img_bytes = pix.tobytes("png")
                        nparr = np.frombuffer(img_bytes, np.uint8)
                        bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        if bgr is not None:
                            res = ocr.engine.read_with_confidence(bgr)
                            if res.text:
                                text = res.text
                    except Exception as ocr_err:
                        log.warning("OCR failed on PDF page %d: %s", page_num, ocr_err)

                page_texts.append((page_num, text))

            pdf_doc.close()

            doc_title, sections = _parse_sections_from_pages(page_texts, default_title)
            full_text = "\n\n".join(s.full_text for s in sections)
            return ParsedDocument(
                document_id=doc_id,
                filename=filename,
                title=doc_title,
                page_count=page_count,
                sections=sections,
                full_text=full_text,
                created_at=t0,
            )

        raise ValueError(f"Unsupported file format: {ext}. Supported: PDF, Images (PNG/JPG), TXT.")


extractor = DocumentExtractor()
