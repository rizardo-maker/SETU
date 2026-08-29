import os
import cv2
import numpy as np
import fitz  # PyMuPDF
from pathlib import Path
from setu_pipeline import SETUPipeline
from model_downloader import ensure_models_installed
from config import BASE_DIR, OUTPUT_AUDIO_DIR

def create_sample_image(path: Path):
    """Creates a sample appointment letter image matching the user prompt specification."""
    # Create white canvas
    img = np.ones((500, 700, 3), dtype=np.uint8) * 255
    
    # Draw border box
    cv2.rectangle(img, (30, 30), (670, 470), (0, 0, 0), 3)
    
    # Write text onto canvas
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, "APPOINTMENT LETTER", (150, 100), font, 1.2, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, "Date: 28 August 2026", (100, 220), font, 0.9, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(img, "Time: 10:30 AM", (100, 290), font, 0.9, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(img, "Location: SETU Health Center", (100, 360), font, 0.9, (0, 0, 0), 2, cv2.LINE_AA)
    
    cv2.imwrite(str(path), img)
    print(f"✓ Created sample image: {path.name}")

def create_sample_pdf(path: Path):
    """Creates a sample vector text PDF using PyMuPDF."""
    doc = fitz.open()
    page = doc.new_page()
    rect = fitz.Rect(50, 50, 550, 700)
    
    text = (
        "SETU Assistive Document Reader.\n\n"
        "Page 1: Welcome to the automated offline accessibility pipeline.\n"
        "This pipeline automatically extracts selectable text from native PDF documents "
        "and processes scanned document images via PaddleOCR.\n\n"
        "The extracted text is broken into small audio chunks and spoken clearly using Piper TTS."
    )
    
    page.insert_textbox(rect, text, fontsize=14)
    doc.save(str(path))
    doc.close()
    print(f"✓ Created sample PDF document: {path.name}")

def run_verification():
    print("=" * 70)
    print("      SETU PIPELINE END-TO-END VERIFICATION TEST      ")
    print("=" * 70)

    # 1. Download models if needed
    ensure_models_installed()

    # 2. Setup sample files
    sample_img_path = BASE_DIR / "sample_appointment.png"
    sample_pdf_path = BASE_DIR / "sample_document.pdf"
    
    create_sample_image(sample_img_path)
    create_sample_pdf(sample_pdf_path)

    # 3. Instantiate pipeline
    pipeline = SETUPipeline(auto_download_models=False)

    # 4. Test 1: Image OCR -> Speech
    print("\n--- Test 1: Image -> OCR -> Text -> TTS ---")
    res_img = pipeline.process_file(sample_img_path, synthesize_audio=True)
    print(f"Engine Used: {res_img['ocr_engine']}")
    print(f"Extracted Text:\n{res_img['full_text']}")
    print(f"Synthesized Audio Files: {res_img['audio_files']}")
    assert len(res_img['audio_files']) > 0, "Image audio synthesis failed!"

    # 5. Test 2: Native PDF -> PyMuPDF Text -> TTS
    print("\n--- Test 2: Native PDF -> PyMuPDF Text -> TTS ---")
    res_pdf = pipeline.process_file(sample_pdf_path, synthesize_audio=True)
    print(f"Extraction Method: {res_pdf['methods_used']}")
    print(f"Extracted Text:\n{res_pdf['full_text']}")
    print(f"Synthesized Audio Files: {res_pdf['audio_files']}")
    assert len(res_pdf['audio_files']) > 0, "PDF audio synthesis failed!"

    # 6. Test 3: Interactive Voice Command Simulator ('Read this' / 'Continue')
    print("\n--- Test 3: Interactive Chunk Reader ---")
    chunk1 = pipeline.read_first_chunk(play_speaker=False)
    print(f"Voice Cmd 'Read this': Chunk 1 text: \"{chunk1['text']}\"")
    print(f"Audio file generated: {chunk1['audio_file']}")

    if chunk1.get("has_more", False):
        chunk2 = pipeline.read_next_chunk(play_speaker=False)
        print(f"Voice Cmd 'Continue': Chunk 2 text: \"{chunk2['text']}\"")
        print(f"Audio file generated: {chunk2['audio_file']}")

    print("\n" + "=" * 70)
    print("✅ ALL TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    run_verification()
