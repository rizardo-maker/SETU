import fitz  # PyMuPDF
import cv2
import numpy as np
from PIL import Image
import io
import re
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional
from config import DPI_FOR_PDF, OCR_LANG, USE_ANGLE_CLS

class DocumentProcessor:
    """
    SETU Document & Image Processing Pipeline
    -----------------------------------------
    Determines input type (Image vs PDF) and applies the optimal, fastest local extraction:
      - PDF with text layer  -> PyMuPDF vector text extraction (instant, 100% accurate)
      - Scanned PDF          -> Render pages to images -> RapidOCR / PaddleOCR engine
      - Image file           -> RapidOCR / PaddleOCR engine
    """

    def __init__(self, preferred_ocr: str = "auto"):
        self.preferred_ocr = preferred_ocr
        self.ocr_engine = None
        self._init_ocr_engine()

    def _init_ocr_engine(self):
        """Lazy load local OCR engine (RapidOCR / PaddleOCR)."""
        try:
            # 1. Try RapidOCR (Production ONNX wrapper of PaddleOCR, fast & self-contained)
            from rapidocr_onnxruntime import RapidOCR
            self.ocr_engine = RapidOCR()
            self.engine_name = "RapidOCR (PaddleOCR ONNX)"
            print("✓ Initialized RapidOCR (PaddleOCR ONNX Engine)")
        except ImportError:
            try:
                # 2. Fallback to standard PaddleOCR package
                from paddleocr import PaddleOCR
                self.ocr_engine = PaddleOCR(use_angle_cls=USE_ANGLE_CLS, lang=OCR_LANG, show_log=False)
                self.engine_name = "PaddleOCR Native"
                print("✓ Initialized PaddleOCR Native Engine")
            except Exception as e:
                print(f"⚠ Warning: Could not initialize PaddleOCR engine: {e}")
                self.ocr_engine = None
                self.engine_name = "None"

    def process_file(self, file_path: Union[str, Path]) -> Dict:
        """
        Main entry point. Processes an image or PDF document.
        Returns a dictionary with extracted text, metadata, page breakdown, and status.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = path.suffix.lower()
        if ext == ".pdf":
            return self._process_pdf(path)
        elif ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]:
            return self._process_image(path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")

    def _process_pdf(self, pdf_path: Path) -> Dict:
        """Process PDF: inspect text layer vs scanned pages."""
        doc = fitz.open(str(pdf_path))
        num_pages = len(doc)
        pages_result = []
        total_text = ""
        is_scanned = False
        methods_used = []

        print(f"📄 Processing PDF ({num_pages} pages): {pdf_path.name}")

        for page_num in range(num_pages):
            page = doc[page_num]
            text = page.get_text("text").strip()

            # If page text is very short (< 15 chars), treat as scanned image page
            if len(text) > 15:
                method = "PyMuPDF (Vector Text)"
                extracted = self._clean_text(text)
            else:
                is_scanned = True
                method = f"{self.engine_name} (OCR)"
                pix = page.get_pixmap(dpi=DPI_FOR_PDF)
                img_bytes = pix.tobytes("png")
                img_np = np.frombuffer(img_bytes, np.uint8)
                img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
                extracted = self._run_ocr_on_image(img)

            methods_used.append(method)
            pages_result.append({
                "page": page_num + 1,
                "method": method,
                "text": extracted
            })
            total_text += f"\n--- Page {page_num + 1} ---\n" + extracted + "\n"

        doc.close()
        full_cleaned_text = self._clean_text(total_text)

        return {
            "file_name": pdf_path.name,
            "file_type": "PDF",
            "is_scanned": is_scanned,
            "num_pages": num_pages,
            "methods_used": list(set(methods_used)),
            "full_text": full_cleaned_text,
            "pages": pages_result,
            "chunks": self.chunk_text(full_cleaned_text)
        }

    def _process_image(self, img_path: Path) -> Dict:
        """Process standard image file using OCR."""
        print(f"🖼 Processing Image: {img_path.name}")
        img = cv2.imread(str(img_path))
        if img is None:
            # Fallback PIL load
            pil_img = Image.open(img_path).convert("RGB")
            img = np.array(pil_img)[:, :, ::-1]

        extracted_text = self._run_ocr_on_image(img)
        cleaned = self._clean_text(extracted_text)

        return {
            "file_name": img_path.name,
            "file_type": "IMAGE",
            "is_scanned": True,
            "num_pages": 1,
            "methods_used": [self.engine_name],
            "full_text": cleaned,
            "pages": [{"page": 1, "method": self.engine_name, "text": cleaned}],
            "chunks": self.chunk_text(cleaned)
        }

    def _run_ocr_on_image(self, img_np: np.ndarray) -> str:
        """Runs local OCR engine on a numpy BGR image array."""
        if self.ocr_engine is None:
            return "[Error: OCR Engine not initialized]"

        try:
            # If RapidOCR
            if hasattr(self.ocr_engine, "__call__"):
                result, _ = self.ocr_engine(img_np)
                if result:
                    lines = [line[1] for line in result if len(line) >= 2]
                    return "\n".join(lines)
                return ""
            # If PaddleOCR Native
            elif hasattr(self.ocr_engine, "ocr"):
                result = self.ocr_engine.ocr(img_np, cls=USE_ANGLE_CLS)
                text_lines = []
                if result and result[0]:
                    for line in result[0]:
                        text_lines.append(line[1][0])
                return "\n".join(text_lines)
        except Exception as e:
            print(f"❌ OCR execution error: {e}")
            return ""
        return ""

    def _clean_text(self, text: str) -> str:
        """Sanitizes OCR/PDF text for optimal audio pronunciation."""
        if not text:
            return ""
        # Normalize whitespace & remove unusual glyphs
        text = re.sub(r'[\r\t]', ' ', text)
        text = re.sub(r'\n+', '\n', text)
        text = re.sub(r' +', ' ', text)
        # Fix common OCR noise like duplicate dashes
        text = re.sub(r'-{2,}', '-', text)
        return text.strip()

    @staticmethod
    def chunk_text(text: str, max_chunk_len: int = 300) -> List[str]:
        """
        Splits full document text into user-friendly audio chunks.
        This enables blind users to listen chunk-by-chunk without overwhelming speech streams.
        """
        if not text:
            return []

        paragraphs = text.split("\n\n")
        chunks = []

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # If paragraph fits in max_chunk_len
            if len(para) <= max_chunk_len:
                chunks.append(para)
            else:
                # Split paragraph by sentences
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

        return chunks

if __name__ == "__main__":
    doc_proc = DocumentProcessor()
    print(f"DocumentProcessor ready using engine: {doc_proc.engine_name}")
