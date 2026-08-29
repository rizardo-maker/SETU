# SETU

Offline, install-free visual assistance for blind and low-vision users.
Problem Statement #4. See the full project document for the "why" —
this README is the "how to run it."

## Architecture in one paragraph

A browser (any device, no install) streams camera frames over a
WebSocket to a local FastAPI server. Currency, text, and obstacle
detection go through small, fast, calibrated models (Tier 1 —
`server/tier1/`) that can abstain when unsure. Open-ended scene
description goes through a local Ollama vision-language model (Tier 2 —
`server/tier2/vlm.py`). Nothing leaves the machine. Full detail in
`server/config.py` and the module docstrings — they're written to be
read, not just imported.

## Quickstart

```bash
./scripts/network.sh          # starts server in Network Mode with auto HTTPS & LAN IPs
# or
./scripts/dev.sh              # standard startup (creates venv, starts server)
```

Open **https://localhost:8443** on this computer, or open the displayed **Network URL** (e.g. `https://10.10.85.71:8443`) on any smartphone or tablet connected to the same Wi-Fi.

`getUserMedia` (camera) requires HTTPS on mobile browsers — `./scripts/network.sh` automatically generates trusted certificates for all your LAN IPs.

## What works out of the box vs. what you need to add

| Capability | Out of the box | To make it real |
|---|---|---|
| Camera capture, WebSocket streaming, backpressure | ✅ | — |
| Audio sonar framing guidance | ✅ | — |
| Quality gate (blur/exposure/motion) | ✅ | Tune thresholds in `config.py` against your own camera |
| Currency recognition | Reports "not trained yet" | Collect data (`training/dataset_layout.md`), run `training/train_currency_classifier.py` |
| Text reading (OCR) | Reports "unavailable" | `pip install -r requirements-full.txt` (PaddleOCR) |
| Obstacle detection | Reports "unavailable" | `pip install -r requirements-full.txt` (Ultralytics, **AGPL-3.0** — see below) |
| Scene description / free questions | Reports "unavailable" | Install [Ollama](https://ollama.com), `ollama pull moondream`, `ollama serve` |
| Speech-to-text (voice commands) | Reports "unavailable" | `pip install -r requirements-full.txt` (faster-whisper) |
| Text-to-speech | Works via macOS `say` (dev-only fallback) | Download a [Piper](https://github.com/rhasspy/piper) voice into `models/tts/` |

This "degrades honestly instead of crashing or faking it" behaviour is
not a shortcut — it's the same abstention philosophy the currency
classifier uses, applied to the whole system. Every unavailable
feature says so out loud instead of silently doing nothing.

## Run the full stack

```bash
./scripts/dev.sh --full
```

Installs PyTorch, ONNX Runtime, PaddleOCR, Ultralytics, and
faster-whisper. Several hundred MB, several minutes, needs network.

## Licensing note

`ultralytics` (obstacle detection) is **AGPL-3.0**. That's a
deliberate, accepted choice — this repo is public and AGPL. If you
ever need a permissive license instead, swap in an Apache-2.0 detector
(MMDetection) or a BSD-licensed torchvision detection model;
`server/tier1/detect.py`'s interface is what you'd keep.

## Repo layout

```
server/
  main.py           FastAPI app + WebSocket loop — start here to trace a request
  config.py         every tunable constant, in one place
  ws_protocol.py     the message schema between client and server
  arbiter.py         calibration + multi-frame abstention logic
  tier1/            currency, ocr, obstacle detection, quality gate
  tier2/            local VLM client (Ollama)
  speech/           STT (faster-whisper) and TTS (Piper / macOS say)
client/
  index.html, app.js, sonar.js, style.css, manifest.json
training/
  train_currency_classifier.py, dataset_layout.md
scripts/
  dev.sh, gen_certs.sh
models/
  (generated — see models/README.md)
```
