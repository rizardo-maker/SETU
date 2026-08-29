import argparse
import sys
from pathlib import Path
from setu_pipeline import SETUPipeline

def main():
    parser = argparse.ArgumentParser(description="SETU Local Offline OCR-to-Speech CLI")
    parser.add_argument("--input", "-i", type=str, help="Path to input image or PDF document", required=False)
    parser.add_argument("--play", "-p", action="store_true", help="Play audio immediately on device speaker")
    parser.add_argument("--interactive", action="store_true", help="Run in chunked interactive mode ('Read this' / 'Continue')")
    
    args = parser.parse_args()

    pipeline = SETUPipeline(auto_download_models=True)

    if not args.input:
        print("\n" + "=" * 60)
        print("   SETU Local Offline OCR-to-Speech Interactive CLI   ")
        print("=" * 60)
        print("Usage:")
        print("  python cli.py --input <path_to_image_or_pdf> [--play] [--interactive]")
        print("\nExample:")
        print("  python cli.py --input sample_appointment.png --interactive")
        sys.exit(0)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Error: File not found: {args.input}")
        sys.exit(1)

    print(f"\n📄 Processing input file: {input_path.name}")
    result = pipeline.process_file(input_path, synthesize_audio=not args.interactive)

    print("\n" + "=" * 60)
    print(f"File: {result['file_name']} ({result['file_type']})")
    print(f"OCR Method: {result['ocr_engine']} | Pages: {result['num_pages']}")
    print(f"TTS Engine: {result['tts_engine']} | Chunks: {result['num_chunks']}")
    print("=" * 60)
    print("\n--- Extracted Text ---")
    print(result['full_text'])
    print("----------------------\n")

    if args.interactive:
        print("=" * 60)
        print("🎙 SETU Assistive Reader Mode (Simulating Voice Commands)")
        print("Commands: [r] Read first chunk | [c] Continue | [q] Quit")
        print("=" * 60)
        
        # Initial trigger: Read first chunk
        chunk_res = pipeline.read_first_chunk(play_speaker=args.play)
        print(f"\n[Chunk 1/{chunk_res['total_chunks']}]")
        print(f"Text: \"{chunk_res['text']}\"")
        print(f"Audio file: {chunk_res['audio_file']}")

        while chunk_res.get("has_more", False):
            user_cmd = input("\nAction [c = Continue / q = Quit]: ").strip().lower()
            if user_cmd == 'q':
                print("Exiting reader mode.")
                break
            elif user_cmd in ['c', '', 'continue']:
                chunk_res = pipeline.read_next_chunk(play_speaker=args.play)
                print(f"\n[Chunk {chunk_res['chunk_idx']}/{chunk_res['total_chunks']}]")
                print(f"Text: \"{chunk_res['text']}\"")
                print(f"Audio file: {chunk_res['audio_file']}")
            else:
                print("Unknown command. Enter 'c' to continue or 'q' to quit.")

    elif args.play and result.get("audio_files"):
        print("\n🔊 Playing synthesized document audio...")
        for audio_file in result["audio_files"]:
            pipeline.tts_engine.play_audio(audio_file)

if __name__ == "__main__":
    main()
