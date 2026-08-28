#!/usr/bin/env bash
# Generates a locally-trusted HTTPS certificate so getUserMedia (the
# camera API) works when you open the app from a phone on the same
# LAN. Plain HTTP only works on localhost — a phone hitting
# http://192.168.x.x gets a silently dead camera with no useful error,
# which is the #1 way this kind of demo fails. Do this on day one.
set -euo pipefail
cd "$(dirname "$0")/.."

if ! command -v mkcert >/dev/null 2>&1; then
  echo "mkcert not found. Installing via Homebrew..."
  if ! command -v brew >/dev/null 2>&1; then
    echo "Homebrew not found either. Install it from https://brew.sh, then re-run this script."
    exit 1
  fi
  brew install mkcert
fi

mkcert -install

mkdir -p certs
LAN_IP=$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || echo "")

echo "Generating certificate for localhost, 127.0.0.1${LAN_IP:+, and $LAN_IP}"
if [ -n "$LAN_IP" ]; then
  mkcert -cert-file certs/cert.pem -key-file certs/key.pem localhost 127.0.0.1 "$LAN_IP"
else
  mkcert -cert-file certs/cert.pem -key-file certs/key.pem localhost 127.0.0.1
  echo "Could not detect a LAN IP automatically. If phones can't connect,"
  echo "find your Mac's IP (System Settings > Wi-Fi > Details) and re-run:"
  echo "  mkcert -cert-file certs/cert.pem -key-file certs/key.pem localhost 127.0.0.1 <your-ip>"
fi

echo ""
echo "Done. On your phone, open Wi-Fi settings and confirm it's on the SAME"
echo "network as this Mac, then visit: https://${LAN_IP:-<this-macs-ip>}:8443"
echo ""
echo "IMPORTANT: mkcert's root CA is trusted by THIS Mac only. A phone will"
echo "still show a certificate warning unless you also install mkcert's"
echo "root CA on the phone (mkcert -CAROOT shows where it lives) — for a"
echo "quick demo it's usually faster to just tap through the warning."
