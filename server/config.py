"""
Central configuration for SETU.

Everything here is a plain constant on purpose — for a hackathon build,
a settings object you have to trace through three files is slower to
debug at 2am than a file you can read top to bottom. Promote to env
vars / pydantic-settings later if this grows.
"""
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"
CLIENT_DIR = ROOT_DIR / "client"

# ---- Server ----
HOST = os.environ.get("SETU_HOST", "0.0.0.0")
PORT = int(os.environ.get("SETU_PORT", "8443"))
USE_TLS = os.environ.get("SETU_TLS", "1") == "1"   # getUserMedia requires HTTPS on a phone
CERT_FILE = ROOT_DIR / "certs" / "cert.pem"
KEY_FILE = ROOT_DIR / "certs" / "key.pem"

# ---- Tier 1: currency classifier ----
CURRENCY_MODEL_PATH = MODELS_DIR / "currency_classifier.onnx"
CURRENCY_LABELS_PATH = MODELS_DIR / "currency_labels.json"
CURRENCY_INPUT_SIZE = 224          # square, matches MobileNetV3/EfficientNet-B0 default
CURRENCY_TEMPERATURE = 1.7         # placeholder — refit on your held-out set, see training/
CURRENCY_CONF_FLOOR = 0.80
CURRENCY_MARGIN_FLOOR = 0.25
CURRENCY_FRAMES_REQUIRED = 3       # multi-frame agreement window

# ---- Tier 1: quality gate (feeds both the abstain decision and the audio sonar) ----
GATE_SHARPNESS_FLOOR = 60.0        # variance-of-Laplacian; tune against your own camera/lighting
GATE_MIN_LUMINANCE = 40.0          # 0-255 mean brightness below which we suggest the torch
GATE_MAX_CLIPPED_FRACTION = 0.15   # fraction of pixels near-saturated before we call it glare
GATE_MAX_MOTION = 18.0             # mean abs frame diff above which we say "hold steady"

# ---- Tier 2: local VLM via Ollama ----
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.environ.get("SETU_VLM_MODEL", "gemma3:4b")
VLM_SYSTEM_PROMPT = (
    "You describe scenes for a blind listener. Two sentences maximum. "
    "Lead with what matters for safety or navigation. Plain spoken language, "
    "no markdown, no lists. If the image is too dark, too blurred, or you are "
    "not confident what it shows, reply with exactly: UNCLEAR"
)
VLM_TEMPERATURE = 0.1
VLM_NUM_PREDICT = 80
VLM_TIMEOUT_S = 20.0

# ---- Speech ----
STT_MODEL_SIZE = os.environ.get("SETU_STT_MODEL", "base")   # faster-whisper size
STT_LANGUAGE = None   # None = auto-detect; set e.g. "te", "hi", "en" to force one
TTS_VOICE = os.environ.get("SETU_TTS_VOICE", "en_US-lessac-medium")  # a Piper voice name
TTS_SAMPLE_RATE = 22050

# ---- Modes the client can request ----
# Tier 1 modes are fast + calibrated. Tier 2 modes are open-ended + local VLM.
TIER1_MODES = {"currency", "text", "obstacle"}
TIER2_MODES = {"scene", "question"}

# ---- Abstention copy — centralised so the tone stays consistent everywhere ----
PHRASES = {
    "abstain_low_conf": "I'm not sure. Move it towards the light and hold it steady.",
    "abstain_dark": "It's too dark. Turning on the torch.",
    "abstain_blurred": "That's blurry. Hold the phone a little steadier.",
    "abstain_motion": "Keep moving slowly until I say stop.",
    "vlm_unclear": "I can't tell from this image. Try moving a little closer.",
    "tier2_unavailable": "The description feature isn't available right now, but currency and text reading still work.",
    "ready": "Got it.",
}
