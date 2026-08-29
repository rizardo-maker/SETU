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
LAN_IPS=($(ifconfig 2>/dev/null | grep "inet " | grep -v "127.0.0.1" | grep -v "169.254" | awk '{print $2}'))

if [ ${#LAN_IPS[@]} -gt 0 ]; then
  echo "Generating certificate for localhost, 127.0.0.1, and network IPs: ${LAN_IPS[*]}"
  mkcert -cert-file certs/cert.pem -key-file certs/key.pem localhost 127.0.0.1 "${LAN_IPS[@]}"
else
  echo "Generating certificate for localhost and 127.0.0.1"
  mkcert -cert-file certs/cert.pem -key-file certs/key.pem localhost 127.0.0.1
fi

echo ""
echo "============================================================"
echo "  SETU SSL Certificates Generated Successfully"
echo "============================================================"
for ip in "${LAN_IPS[@]}"; do
  echo "  • Phone / Network URL: https://${ip}:8443"
done
echo "  • Local Device URL:   https://localhost:8443"
echo "============================================================"
echo ""
echo "Note: On your phone, connect to the SAME Wi-Fi network and open"
echo "the Network URL above. (Tap through the self-signed cert warning on phone)."
