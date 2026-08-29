#!/usr/bin/env bash
# ==============================================================================
# SETU Network Mode Launcher
# 
# Starts the SETU server bound to all network interfaces (0.0.0.0:8443)
# with automatic HTTPS certificate setup, CORS enabled, and live LAN IP
# broadcast so phones and mobile devices on the same Wi-Fi can connect.
# ==============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

echo "============================================================"
echo "  🌐 INITIALIZING SETU IN NETWORK MODE"
echo "============================================================"

# 1. Activate / Initialize virtual environment
if [ ! -d .venv ]; then
  echo "[1/4] Creating virtual environment..."
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

# 2. Check / Generate SSL Certificates for all local network IPs
mkdir -p certs models
LAN_IPS=($(ifconfig 2>/dev/null | grep "inet " | grep -v "127.0.0.1" | grep -v "169.254" | awk '{print $2}'))

if [ ! -f certs/cert.pem ] || [ ! -f certs/key.pem ]; then
  echo "[2/4] Generating HTTPS certificates for Network Mode..."
  ./scripts/gen_certs.sh
else
  echo "[2/4] HTTPS certificates found in certs/."
fi

# 3. Check optional Tier 2 Ollama engine
echo "[3/4] Checking Ollama Tier 2 service..."
if ! curl -sS -m 1 -o /dev/null http://127.0.0.1:11434/api/tags 2>/dev/null; then
  echo "      (Note: Ollama offline; Tier 1 Edge models will operate 100% locally)"
else
  echo "      Ollama local LLM service is active."
fi

# 4. Launch SETU server in Network Mode
echo "[4/4] Starting SETU server on 0.0.0.0:8443..."
export SETU_HOST="0.0.0.0"
export SETU_PORT="8443"
export SETU_TLS="1"

python -m server.main
