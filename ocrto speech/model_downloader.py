import os
import urllib.request
import sys
from pathlib import Path
from config import PIPER_MODEL_FILE, PIPER_CONFIG_FILE, PIPER_MODEL_URL, PIPER_CONFIG_URL, MODELS_DIR

def download_file(url: str, dest_path: Path):
    """Download a file with progress reporting."""
    if dest_path.exists() and dest_path.stat().st_size > 0:
        print(f"✓ Model file already exists: {dest_path.name}")
        return True

    print(f"Downloading {dest_path.name} from {url}...")
    try:
        def progress_bar(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(100, int(downloaded * 100 / total_size))
                mb_downloaded = downloaded / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                sys.stdout.write(f"\rProgress: [{percent:3d}%] {mb_downloaded:.2f} MB / {mb_total:.2f} MB")
                sys.stdout.flush()
        
        urllib.request.urlretrieve(url, str(dest_path), reporthook=progress_bar)
        print(f"\n✓ Successfully downloaded {dest_path.name}")
        return True
    except Exception as e:
        print(f"\n❌ Error downloading {dest_path.name}: {e}")
        if dest_path.exists():
            dest_path.unlink()
        return False

def ensure_models_installed():
    """Ensure all required production models are downloaded locally."""
    print("=" * 60)
    print("SETU Pipeline: Ensuring Production Models Available")
    print("=" * 60)
    
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Download Piper TTS Model (.onnx & .json)
    onnx_ok = download_file(PIPER_MODEL_URL, PIPER_MODEL_FILE)
    json_ok = download_file(PIPER_CONFIG_URL, PIPER_CONFIG_FILE)
    
    if onnx_ok and json_ok:
        print("✓ Piper TTS ONNX voice model is ready.")
    else:
        print("⚠ Warning: Piper voice model download incomplete. Audio synthesis will fall back to local CLI/TTS.")
        
    return onnx_ok and json_ok

if __name__ == "__main__":
    ensure_models_installed()
