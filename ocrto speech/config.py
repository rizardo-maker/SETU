import os
from pathlib import Path

# Base Directory Setup
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
OUTPUT_AUDIO_DIR = BASE_DIR / "output_audio"
TEMP_DIR = BASE_DIR / "temp"

# Create required directories
MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# Piper TTS Configuration
# Using en_US-lessac-medium voice model as standard production offline voice
PIPER_MODEL_NAME = "en_US-lessac-medium"
PIPER_MODEL_FILE = MODELS_DIR / f"{PIPER_MODEL_NAME}.onnx"
PIPER_CONFIG_FILE = MODELS_DIR / f"{PIPER_MODEL_NAME}.onnx.json"

PIPER_MODEL_URL = f"https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/{PIPER_MODEL_NAME}.onnx"
PIPER_CONFIG_URL = f"https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/{PIPER_MODEL_NAME}.onnx.json"

# OCR Configuration
OCR_LANG = "en"
USE_ANGLE_CLS = True
DPI_FOR_PDF = 200  # Resolution when rendering scanned PDF pages to images

# Assistive Reader / Chunking Settings
DEFAULT_MAX_CHUNK_CHARS = 300  # Sentences grouped up to 300 characters (~40-50 words) per speech chunk
DEFAULT_CHUNK_STRATEGY = "paragraph_sentence"  # Reads block-by-block for optimal listener comprehension
