# SETU: Local Offline Document OCR to English Speech Pipeline 👁️🔊

A high-performance, 100% local, offline Python pipeline designed for assistive accessibility (SETU). Converts images and document files (PDFs) into English text and synthesizes speech using pre-trained neural models without requiring internet access or cloud APIs.

---

## 🏗️ Architecture

```text
                           SETU ENGINE
                                │
               ┌────────────────┴────────────────┐
               │                                 │
          CAMERA / IMAGE                      DOCUMENT
               │                                 │
               ▼                                 ▼
          Image OCR                        PDF Document
               │                                 │
               │                   ┌─────────────┴─────────────┐
               │                   │                           │
               │               Text PDF                     Scanned
               │                   │                           │
               │               PyMuPDF                     PaddleOCR
               │                   │                           │
               └───────────────────┴─────────────┬─────────────┘
                                                 │
                                                 ▼
                                        TEXT SANITIZATION
                                        & CHUNKING ENGINE
                                                 │
                                                 ▼
                                         Piper Neural TTS
                                         (ONNX Voice Model)
                                                 │
                                                 ▼
                                          WAV AUDIO FILE
                                         & SPEAKER OUTPUT
```

---

## ⚡ Key Features

1. **Native PDF Extraction via PyMuPDF (`fitz`)**:
   - Detects if PDF contains selectable text vectors.
   - Extracts text instantly without running heavy OCR models when unneeded.
2. **Local Image & Scanned Document OCR via RapidOCR / PaddleOCR**:
   - Runs local ONNX / Paddle inference on camera photos, appointment letters, forms, and scanned pages.
3. **Assistive Speech Chunking**:
   - Prevents overwhelming blind users with 10-page text dumps.
   - Splits document text into manageable, natural sentences/paragraphs (~40-50 words per chunk).
4. **Piper Neural TTS Engine**:
   - Uses pre-trained ONNX neural voice (`en_US-lessac-medium`).
   - Generates `.wav` audio output files and supports direct device speaker playback offline.
5. **Interactive Simulated Voice Commands**:
   - `"Read this"` -> Synthesizes and plays the first section.
   - `"Continue"` -> Advances and speaks the next section on command.

---

## 🚀 Quick Start & Usage

### 1. Verification Test
Run the automated test suite to verify model downloads, image OCR, PDF parsing, text chunking, and speech generation:
```bash
python test_pipeline.py
```

### 2. Command Line Interface (CLI)
Process an image or PDF document directly from the terminal:
```bash
# Process image and play audio on speakers:
python cli.py --input sample_appointment.png --play

# Interactive chunked mode ("Read this" / "Continue"):
python cli.py --input sample_appointment.png --interactive
```

### 3. Gradio Web Interface
Launch the local web UI for drag-and-drop document upload, visual text highlighting, and audio player:
```bash
python app.py
```
Open browser at `http://127.0.0.1:7860`.

---

## 📁 Repository File Structure

- `config.py`: Directory settings, model URLs, DPI resolution, and chunking parameters.
- `model_downloader.py`: Automatic downloader for production-ready Piper voice models.
- `document_processor.py`: Unified engine for PyMuPDF text parsing & RapidOCR/PaddleOCR image processing.
- `tts_engine.py`: Offline neural speech synthesis engine (Piper ONNX + macOS `say` fallback + audio playback).
- `setu_pipeline.py`: Orchestrator class linking extraction, chunking, state management, and voice commands.
- `cli.py`: Interactive command-line terminal tool.
- `app.py`: Gradio web UI.
- `test_pipeline.py`: Comprehensive test harness.
