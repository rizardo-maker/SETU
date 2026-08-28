#!/usr/bin/env bash
# One command to get a working server: creates a venv if needed,
# installs the lightweight core dependencies, and starts uvicorn.
#
#   ./scripts/dev.sh          start the server
#   ./scripts/dev.sh --full   also install requirements-full.txt
#                             (torch, onnxruntime, paddleocr, ultralytics,
#                              faster-whisper — everything beyond the stub
#                              behaviour, several hundred MB, several minutes)
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -d .venv ]; then
  echo "Creating virtual environment..."
  PY_BIN=""
  for candidate in python3.11 /opt/homebrew/opt/python@3.11/bin/python3.11 python3.12 /opt/homebrew/bin/python3.12 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
      PY_BIN="$candidate"
      break
    fi
  done
  "${PY_BIN:-python3}" -m venv .venv
fi
source .venv/bin/activate

echo "Installing core dependencies..."
pip install --quiet --upgrade pip
pip install --quiet -r requirements-core.txt

if [ "${1:-}" = "--full" ]; then
  echo "Installing full dependencies (this can take several minutes)..."
  pip install --quiet -r requirements-full.txt
fi

mkdir -p models certs

if [ ! -f certs/cert.pem ]; then
  echo ""
  echo "No HTTPS certificate found yet. The server will fall back to plain"
  echo "HTTP, which is fine for localhost testing but will NOT let a phone's"
  echo "camera work. Run ./scripts/gen_certs.sh before demoing on a phone."
  echo ""
fi

if ! curl -sS -m 1 -o /dev/null http://127.0.0.1:11434/api/tags 2>/dev/null; then
  echo "Note: Ollama doesn't appear to be running — Tier 2 (scene"
  echo "description) will report itself unavailable. Install it from"
  echo "https://ollama.com, run 'ollama pull gemma3:4b', then 'ollama serve'."
  echo ""
fi

echo "Starting SETU..."
python -m server.main
