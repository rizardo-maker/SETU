import cv2
import time
import numpy as np
from pathlib import Path
from setu_pipeline import SETUPipeline
from config import TEMP_DIR, OUTPUT_AUDIO_DIR

def run_realtime_camera():
    """
    SETU Real-Time Desktop Camera OCR -> TTS Engine
    ================================================
    1. Opens live camera feed via OpenCV.
    2. User presses SPACE or 'c' to capture & read text.
    3. Runs RapidOCR/PaddleOCR on captured frame.
    4. Synthesizes Piper TTS audio and plays speech immediately over device speaker.
    """
    print("=" * 60)
    print("  📷 SETU Real-Time Camera OCR to Speech Assistant  ")
    print("=" * 60)
    print("Controls:")
    print("  [SPACE] / [c]  : Capture text & speak audio")
    print("  [r]            : Read first section ('Read this')")
    print("  [n]            : Continue reading next section ('Continue')")
    print("  [q] / [ESC]    : Quit camera")
    print("=" * 60)

    # Initialize SETU pipeline
    pipeline = SETUPipeline(auto_download_models=True)

    # Open local webcam (Device Index 0)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Could not open camera device 0. Please check webcam permissions.")
        return

    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("✓ Camera feed active. Press SPACE or 'c' to capture frame...")

    last_status_msg = "Press SPACE to Capture & Read Text"
    is_processing = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠ Failed to grab frame.")
            break

        # Display preview frame with HUD overlay
        preview = frame.copy()
        h, w, _ = preview.shape

        # Semi-transparent HUD header
        cv2.rectangle(preview, (0, 0), (w, 60), (15, 23, 42), -1)
        cv2.putText(preview, "SETU REAL-TIME OCR SCANNER", (20, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (56, 189, 248), 2, cv2.LINE_AA)

        # Status footer banner
        cv2.rectangle(preview, (0, h - 50), (w, h), (30, 41, 59), -1)
        cv2.putText(preview, last_status_msg, (20, h - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (248, 250, 252), 2, cv2.LINE_AA)

        # Center target frame box
        box_w, box_h = int(w * 0.7), int(h * 0.6)
        x1, y1 = int((w - box_w) / 2), int((h - box_h) / 2)
        cv2.rectangle(preview, (x1, y1), (x1 + box_w, y1 + box_h), (56, 189, 248), 2)

        cv2.imshow("SETU Real-Time Camera Scanner", preview)

        key = cv2.waitKey(1) & 0xFF

        if key in [ord('q'), 27]:  # 'q' or ESC
            print("Closing camera.")
            break

        elif key in [32, ord('c')]:  # SPACE or 'c'
            print("\n📸 Frame captured! Running local OCR & TTS...")
            last_status_msg = "⚡ Processing frame: OCR -> Piper Speech..."
            
            # Save temporary captured frame
            cap_path = TEMP_DIR / "camera_capture.png"
            cv2.imwrite(str(cap_path), frame)

            start_t = time.time()
            res = pipeline.process_file(cap_path, synthesize_audio=True)
            elapsed = time.time() - start_t

            if not res["full_text"]:
                last_status_msg = "⚠ No text detected in frame. Try again."
                print("⚠ No text detected.")
            else:
                last_status_msg = f"✓ Detected text ({len(res['chunks'])} chunks) in {elapsed:.2f}s"
                print("\n" + "=" * 60)
                print(f"Captured Text:\n{res['full_text']}")
                print("=" * 60)

                # Play first audio chunk immediately over speaker
                if res.get("audio_files"):
                    print(f"🔊 Playing synthesized audio on speaker...")
                    pipeline.tts_engine.play_audio(res["audio_files"][0])

        elif key == ord('r'):  # 'r' -> Read first chunk
            last_status_msg = "🎙️ Voice Cmd: 'Read this'"
            chunk_res = pipeline.read_first_chunk(play_speaker=True)
            print(f"Reading: \"{chunk_res.get('text', '')}\"")

        elif key == ord('n'):  # 'n' -> Read next chunk
            last_status_msg = "⏭️ Voice Cmd: 'Continue'"
            chunk_res = pipeline.read_next_chunk(play_speaker=True)
            print(f"Reading: \"{chunk_res.get('text', '')}\"")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_realtime_camera()
