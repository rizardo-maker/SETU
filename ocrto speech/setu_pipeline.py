import time
from pathlib import Path
from typing import Dict, List, Optional, Union
from document_processor import DocumentProcessor
from tts_engine import TTSEngine
from model_downloader import ensure_models_installed
from config import OUTPUT_AUDIO_DIR

class SETUPipeline:
    """
    SETU Complete Local Offline OCR-to-Speech Pipeline
    ==================================================
    Architecture:
      IMAGE / DOCUMENT ──> PyMuPDF / PaddleOCR / RapidOCR ──> TEXT ──> Text Sanitization & Chunking ──> Piper TTS ──> WAV AUDIO / SPEAKER
    
    Design for Assistive Accessibility:
      - Supports full-document speech generation.
      - Supports chunked interactive playback ("Read this", "Continue").
    """

    def __init__(self, auto_download_models: bool = True):
        if auto_download_models:
            ensure_models_installed()

        print("\n🚀 Initializing SETU Local Offline OCR-to-Speech Pipeline...")
        self.doc_processor = DocumentProcessor()
        self.tts_engine = TTSEngine()
        
        # Interactive session state
        self.current_chunks: List[str] = []
        self.current_chunk_idx: int = 0
        self.current_doc_name: str = ""

    def process_file(self, file_path: Union[str, Path], synthesize_audio: bool = True) -> Dict:
        """
        Processes document/image file and optionally synthesizes chunked speech.
        Returns detailed summary dict with text, audio file paths, and chunk breakdown.
        """
        path = Path(file_path)
        start_time = time.time()

        # Step 1: Run Document & Image Text Extraction (PyMuPDF / PaddleOCR)
        doc_result = self.doc_processor.process_file(path)
        ocr_time = time.time() - start_time

        full_text = doc_result["full_text"]
        chunks = doc_result["chunks"]
        
        # Update interactive session state
        self.current_chunks = chunks
        self.current_chunk_idx = 0
        self.current_doc_name = path.stem

        audio_files = []
        tts_time = 0.0

        if synthesize_audio and chunks:
            tts_start = time.time()
            prefix = f"{path.stem}_audio"
            audio_files = self.tts_engine.synthesize_chunks(chunks, prefix=prefix)
            tts_time = time.time() - tts_start

        total_time = time.time() - start_time

        return {
            "file_name": doc_result["file_name"],
            "file_type": doc_result["file_type"],
            "is_scanned": doc_result["is_scanned"],
            "num_pages": doc_result["num_pages"],
            "methods_used": doc_result["methods_used"],
            "ocr_engine": self.doc_processor.engine_name,
            "tts_engine": self.tts_engine.engine_type,
            "full_text": full_text,
            "num_chunks": len(chunks),
            "chunks": chunks,
            "audio_files": [str(p) for p in audio_files],
            "metrics": {
                "ocr_time_seconds": round(ocr_time, 3),
                "tts_time_seconds": round(tts_time, 3),
                "total_time_seconds": round(total_time, 3)
            }
        }

    def read_first_chunk(self, play_speaker: bool = True) -> Dict:
        """
        Voice Command: 'Read this'
        Reads only the first detected section to avoid overwhelming blind users with large text dumps.
        """
        if not self.current_chunks:
            return {"error": "No document currently loaded."}

        self.current_chunk_idx = 0
        chunk_text = self.current_chunks[0]
        out_wav = OUTPUT_AUDIO_DIR / f"{self.current_doc_name}_chunk_001.wav"
        
        wav_path = self.tts_engine.text_to_speech(chunk_text, out_wav)
        
        if play_speaker:
            print(f"🔊 Playing Chunk 1/{len(self.current_chunks)}...")
            self.tts_engine.play_audio(wav_path)

        return {
            "chunk_idx": 1,
            "total_chunks": len(self.current_chunks),
            "text": chunk_text,
            "audio_file": str(wav_path),
            "has_more": len(self.current_chunks) > 1
        }

    def read_next_chunk(self, play_speaker: bool = True) -> Dict:
        """
        Voice Command: 'Continue'
        Advances to and speaks the next section of the loaded document.
        """
        if not self.current_chunks:
            return {"error": "No document loaded."}

        if self.current_chunk_idx + 1 >= len(self.current_chunks):
            return {
                "chunk_idx": len(self.current_chunks),
                "total_chunks": len(self.current_chunks),
                "text": "End of document reached.",
                "audio_file": None,
                "has_more": False
            }

        self.current_chunk_idx += 1
        idx = self.current_chunk_idx
        chunk_text = self.current_chunks[idx]
        out_wav = OUTPUT_AUDIO_DIR / f"{self.current_doc_name}_chunk_{idx+1:03d}.wav"
        
        wav_path = self.tts_engine.text_to_speech(chunk_text, out_wav)

        if play_speaker:
            print(f"🔊 Playing Chunk {idx+1}/{len(self.current_chunks)}...")
            self.tts_engine.play_audio(wav_path)

        return {
            "chunk_idx": idx + 1,
            "total_chunks": len(self.current_chunks),
            "text": chunk_text,
            "audio_file": str(wav_path),
            "has_more": idx + 1 < len(self.current_chunks)
        }

if __name__ == "__main__":
    pipeline = SETUPipeline(auto_download_models=False)
    print("SETU Pipeline instantiated successfully.")
