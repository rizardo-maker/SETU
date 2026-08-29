# SETU — Comprehensive Code Documentation & Architecture Decision Records (ADRs)

> **Document Version**: 2.0.0  
> **Status**: Active / Production  
> **Target Audience**: Core Engineers, Contributors, Hackathon Judges, and AI Coding Agents  
> **License**: AGPL-3.0  

---

## 1. Executive System Overview

**SETU** is an offline, install-free visual assistance system designed for blind and low-vision users. It provides immediate collision warnings, multi-note currency recognition, document/signboard reading, open-ended scene understanding, and accessible educational tools.

### Core Architectural Principles:

1. **100% On-Device / Edge-First (Zero Cloud Dependency)**:  
   Camera frames depicting personal environments, identity documents, and currency notes **never leave the local host machine**.
2. **Dual-Tier Reflex & Reasoning Engine**:  
   - **Tier 1 (Reflex Layer)**: Calibrated, deterministic models (<20ms latency on CPU/Metal) executing local collision scans, OCR, quality checks, and currency bounding.
   - **Tier 2 (Reasoning Layer)**: Local Vision-Language Model (**Gemma 3 4B** via Ollama) executing in ~1.2–1.5s for contextual scene understanding and query resolution.
3. **Deterministic Abstention & Honest Degradation**:  
   If a frame is blurry, dark, moving, or low-confidence, the system abstains and gives explicit verbal guidance rather than hallucinating an answer.
4. **Voice-First Fault Tolerance**:  
   Every error or failure path resolves to spoken feedback through speech synthesis, ensuring the blind user is never left in silence.

---

## 2. Architecture Decision Records (ADRs)

```
ADR-001: 100% Local Edge Compute vs. Cloud Vision APIs
ADR-002: Dual-Tier Reflex & Reasoning Architecture
ADR-003: Two-Stage Anti-Hallucination Currency Pipeline
ADR-004: Vision-Language Model Selection (Gemma 3 4B)
ADR-005: OCR Engine Selection (RapidOCR ONNX Runtime)
ADR-006: Re-entrant Speech Lock for Acoustic Mic Isolation
ADR-007: Multi-Frame Consensus Arbiter for Currency
ADR-008: Multi-SAN Local TLS Context for Mobile Camera Access
```

---

### ADR-001: 100% Local Edge Compute vs. Cloud Vision APIs

- **Status**: Accepted
- **Date**: 2026-08-28
- **Context**:  
  Commercial vision assistive tools (e.g., Be My Eyes, Envision, Seeing AI) rely on cloud APIs (GPT-4o, Claude, Google Cloud Vision). For blind users, camera frames capture private homes, confidential mail, banking documents, and location surroundings. Additionally, in regions like India with 4.95 crore visually impaired individuals, rural connectivity is frequently degraded or absent.
- **Decision**:  
  Mandate **zero external network calls at inference time**. All vision, OCR, speech recognition, speech synthesis, and language reasoning models must run entirely on the user's local edge hardware.
- **Consequences**:  
  - Complete privacy and confidentiality guarantee.
  - Immunity to network outages, API rate limits, and latency spikes.
  - Requires efficient quantization (int8 STT, 4-bit VLM, YOLO nano models) to run smoothly on consumer laptop hardware.

---

### ADR-002: Dual-Tier Reflex & Reasoning Architecture

- **Status**: Accepted
- **Date**: 2026-08-28
- **Context**:  
  Generative vision-language models take 1.2–4.0 seconds per inference. While acceptable for open-ended queries (*"Describe this room"*), this latency is unacceptable for safety-critical collision avoidance (which requires <50ms response to prevent walking into obstacles).
- **Decision**:  
  Separate the system into two distinct execution tiers:
  - **Tier 1 (Reflex Layer)**: YOLO11n, custom YOLO currency detector, RapidOCR, OpenCV Quality Gate. Executes in **<20ms** without language models.
  - **Tier 2 (Reasoning Layer)**: Gemma 3 4B running on Ollama. Handles visual question answering and text summarization in **~1.5s**.
- **Consequences**:  
  - Collision alerts are instantaneous (30 FPS capability).
  - Heavy VLM compute is only triggered when explicitly requested by user commands.

---

### ADR-003: Two-Stage Anti-Hallucination Currency Pipeline

- **Status**: Accepted
- **Date**: 2026-08-28
- **Context**:  
  A single YOLO object detector trained only on currency images has no negative/background rejection class. If pointed at a person, a wall, or a chair, it may misclassify textures as "100 rupees" with high confidence. Conversely, prompting a pure VLM to count multiple banknotes is slow and prone to arithmetic hallucinations.
- **Decision**:  
  Implement a **coordinated two-stage pipeline**:
  1. **Stage 1 (Semantic Presence Gate)**: Gemma 3 4B checks if genuine Indian currency is visible. If absent, it immediately returns *"No currency detected"*.
  2. **Stage 2 (YOLO Denomination Detector & Counter)**: If verified, Custom YOLOv11 (trained on 1,917 Indian banknote images across 10 classes) detects exact bounding boxes, calculates individual note values, and sums the total.
- **Consequences**:  
  - False positives on non-currency objects drop to 0%.
  - Supports multi-note counting (e.g. *"500 rupees note and 100 rupees note. Total 600 rupees"*).

---

### ADR-004: Vision-Language Model Selection (Gemma 3 4B)

- **Status**: Accepted
- **Date**: 2026-08-28
- **Context**:  
  We benchmarked three local vision models on identical hardware and prompts:
  - *Moondream 2*: Fast (~800ms) but hallucinated severely (e.g., misidentified complex code IDEs as basic browser windows).
  - *Qwen2.5-VL 7B*: Highly accurate but took **51 seconds** per query due to excessive image token density.
  - *Gemma 3 4B*: High spatial reasoning and factual accuracy with **~1.38s** average response time.
- **Decision**:  
  Standardize on **Gemma 3 4B** via local Ollama inference.
- **Consequences**:  
  - Sub-2-second conversational responses.
  - Pre-warmed on server startup via a $1\times 1$ dummy JPEG to eliminate cold-start lag.

---

### ADR-005: OCR Engine Selection (RapidOCR ONNX Runtime)

- **Status**: Accepted
- **Date**: 2026-08-28
- **Context**:  
  Tesseract OCR requires system binary installations (`brew install tesseract`), fails on rotated text, and on photographed mobile screen text read only 37 characters of noise at 0.15 confidence. Native PaddlePaddle requires massive Python dependencies.
- **Decision**:  
  Deploy **RapidOCR** (ONNX Runtime packaging of PaddleOCR algorithms) with CLAHE adaptive contrast enhancement and custom 2D row-bucketing line sorting.
- **Consequences**:  
  - Self-contained Python installation with zero system-level binary dependencies.
  - On the same mobile screen test, RapidOCR extracted 1,398 characters at 0.68 confidence in 42ms.

---

### ADR-006: Re-entrant Speech Lock for Acoustic Mic Isolation

- **Status**: Accepted
- **Date**: 2026-08-29
- **Context**:  
  While the Web Speech API recognizer was listening, browser audio playback triggered acoustic echo cancellation and microphone ducking (heavily attenuating the TTS speaker volume). The microphone also picked up SETU's own voice, causing recursive command loops.
- **Decision**:  
  Implement a **Re-entrant Speech Lock** (`_acquireSpeechLock()` / `_releaseSpeechLock()`):
  - When speech begins, the speech recognizer is immediately aborted and paused.
  - When speech ends, an **acoustic grace period of 280ms** allows room reverberation to decay before re-opening the microphone.
  - In-flight speech recognition packets during speech are rejected at entry.
- **Consequences**:  
  - TTS speech is loud, clear, and unattenuated.
  - Zero false voice activations from device speaker output.

---

### ADR-007: Multi-Frame Consensus Arbiter for Currency

- **Status**: Accepted
- **Date**: 2026-08-28
- **Context**:  
  Single-frame object detection during camera motion can produce momentary flickering detections (e.g., reading a tilted 500 note as 50 for 1 frame).
- **Decision**:  
  Implement `YOLOFrameArbiter` in [`server/tier1/currency.py`](file:///Users/ravi/Documents/SETU/server/tier1/currency.py), requiring $N=2$ consecutive frames to agree on the exact multiset of detected denominations before speaking.
- **Consequences**:  
  - Eliminates motion jitter without introducing noticeable latency.

---

### ADR-008: Multi-SAN Local TLS Context for Mobile Camera Access

- **Status**: Accepted
- **Date**: 2026-08-29
- **Context**:  
  Modern mobile browsers (Safari on iOS, Chrome on Android) strictly require a secure context (HTTPS) to enable `navigator.mediaDevices.getUserMedia`. Plain HTTP connections to LAN IPs (`http://192.168.x.x:8000`) silently fail with `NotAllowedError`.
- **Decision**:  
  Implement [`scripts/gen_certs.sh`](file:///Users/ravi/Documents/SETU/scripts/gen_certs.sh) using `mkcert` with dynamic discovery of all active IPv4 network interfaces (e.g. `10.10.85.71`, `localhost`, `127.0.0.1`). Serve over TLS 1.3 on port `8443` with CORS enabled.
- **Consequences**:  
  - Phone cameras work seamlessly when accessing the host Mac over local Wi-Fi.

---

## 3. Codebase Directory Structure & Module Map

```
/Users/ravi/Documents/SETU/
├── overview.md                       # High-level architecture & tech stack overview
├── README.md                         # Quickstart & runbook
├── docs/                             # Technical design docs & presentation materials
│   ├── code_documentation.md         # This comprehensive documentation & ADR index
│   ├── SETU_Pitch_Deck_Narrative.md  # Presentation script & judge Q&A guide
│   └── proximity_implementation_prompt.md # Proximity radar technical spec
├── client/                           # 100% Client-Side Accessible Application
│   ├── index.html                    # High-contrast accessible UI
│   ├── app.js                        # Master Application Logic & Gesture Engine
│   ├── style.css                     # Design system, Viewfinder HUD, & animations
│   ├── sonar.js                      # Web Audio Sonar Radar tone generator
│   └── manifest.json                 # PWA Web Application manifest
├── server/                           # FastAPI Edge Server
│   ├── main.py                       # Application Entrypoint, REST Routes & WS Stream Loop
│   ├── config.py                     # Central Hyperparameters & Tunable Constants
│   ├── ws_protocol.py                # WebSocket message schemas & serialisation
│   ├── tier1/                        # Tier 1 Edge Models (<20ms)
│   │   ├── detect.py                 # YOLO11n Obstacle & Hazard Scanner
│   │   ├── currency.py               # Custom YOLOv11 Indian Currency Detector
│   │   ├── ocr.py                    # RapidOCR ONNX with Spatial 2D Line Sorting
│   │   └── quality_gate.py           # Sharpness, Luminance, & Motion Blur Evaluator
│   ├── tier2/                        # Tier 2 Reasoning Engine (~1.5s)
│   │   └── vlm.py                    # Ollama Gemma 3 4B Client & Pre-Warming
│   └── speech/                       # Local Speech Subsystem
│       ├── stt.py                    # faster-whisper Offline Speech-to-Text
│       └── tts.py                    # Piper Neural Voice Synthesis + macOS say fallback
├── scripts/                          # Lifecycle & Automation Scripts
│   ├── dev.sh                        # One-command development server runner
│   ├── network.sh                    # Network Mode launcher (LAN/Mobile deployment)
│   └── gen_certs.sh                  # Multi-SAN SSL/TLS certificate generator
├── models/                           # Local Weights & ONNX Models (Zero Cloud)
│   ├── currency_best.pt              # Custom Trained Currency YOLO (10 classes, 5.5MB)
│   ├── yolo11n.pt                    # General Obstacle & Collision YOLO (5.6MB)
│   └── tts/                          # Piper Neural Voice Models (.onnx)
└── ocrto speech/                     # Document-to-Speech Processing Subsystem
```

---

## 4. Module Deep-Dives & Implementation Specifications

### 4.1. Server Core & Lifecycle — [`server/main.py`](file:///Users/ravi/Documents/SETU/server/main.py)

- **Purpose**: Serves static web assets, runs the WebSocket streaming loop, and routes single-shot REST API commands.
- **Key Components**:
  - `lifespan(app)`: Initializes models and runs warmup inference for YOLO detectors and Gemma 3 4B.
  - `_unhandled_exception_handler()`: Catches any unhandled server error and returns well-formed JSON with `"speak": "Something went wrong"`, ensuring client voice synthesis is never broken.
  - `get_network_ips()`: Dynamically extracts active IPv4 addresses to display Network Mode URLs in the terminal on boot.
  - `CORSMiddleware`: Permits cross-origin requests from smartphones and companion clients on the local network.

---

### 4.2. Obstacle & Collision Radar — [`server/tier1/detect.py`](file:///Users/ravi/Documents/SETU/server/tier1/detect.py)

- **Class**: `ObstacleDetector`
- **Model**: YOLO11n (`models/yolo11n.pt`, 5.6 MB)
- **Key Methods**:
  - `scan_for_collision(bgr_frame) -> list[CollisionThreat]`:
    - Evaluates frame against `HAZARD_CLASSES` (chairs, tables, doors, stairs, cars, people, street obstacles).
    - Calculates bounding box area fraction $\frac{(x_2-x_1)(y_2-y_1)}{W \times H}$.
    - If $\ge 0.40$, flags as `urgent` (*"Stop! [Object] right in front of you"*).
    - If $\ge 0.22$, flags as `warn` (*"Careful, [Object] close ahead"*).
  - `detect(bgr_frame) -> list[Detection]`:
    - Runs general object exploration (*"Close by: chair, table"*).

---

### 4.3. Currency Recognition Engine — [`server/tier1/currency.py`](file:///Users/ravi/Documents/SETU/server/tier1/currency.py)

- **Class**: `CurrencyDetector` & `YOLOFrameArbiter`
- **Model**: Custom YOLOv11 (`models/currency_best.pt`, 5.5 MB, 10 classes)
- **Denominations**: 10 (old/new), 20, 50 (old/new), 100 (old/new), 200, 500, 2000.
- **Key Methods**:
  - `detect(bgr_frame, conf_threshold=0.70)`: Returns bounding boxes, confidence scores, and denomination integer values.
  - `_speak_for(denominations, total)`: Formats compact spoken phrases (e.g. *"Two 500 rupees notes and one 100 rupees note. Total 1100 rupees"*).

---

### 4.4. High-Accuracy OCR & Spatial Sorting — [`server/tier1/ocr.py`](file:///Users/ravi/Documents/SETU/server/tier1/ocr.py)

- **Class**: `OCREngine`
- **Engine**: RapidOCR (ONNX Runtime)
- **Key Algorithms**:
  - `read_with_confidence(bgr_frame) -> OCRResult`:
    - Applies CLAHE contrast enhancement (`cv2.createCLAHE`) if initial detection produces low confidence or sparse characters.
    - Extracts 2D bounding boxes and buckets vertical coordinates by $24\text{px}$ rows:
      $$\text{Row} = \text{round}\left(\frac{y_{\text{center}}}{24.0}\right) \times 24.0$$
    - Sorts lines left-to-right within each row to reconstruct multi-column layouts into natural reading order.
  - `clean_ocr_text(text)`: Strips OCR artifacts, duplicate line breaks, and repeated hyphens.
  - `chunk_text(text, max_chunk_len=250)`: Splits long documents into discrete spoken sentences for sentence-by-sentence reading.

---

### 4.5. Quality Gate & Framing Assessment — [`server/tier1/quality_gate.py`](file:///Users/ravi/Documents/SETU/server/tier1/quality_gate.py)

- **Function**: `assess(bgr_frame, prev_gray) -> tuple[GateResult, np.ndarray]`
- **Execution Target**: Pure OpenCV / NumPy (<5ms CPU)
- **Metrics**:
  - **Sharpness**: Laplacian variance $\text{Var}(\nabla^2 I) \ge 60.0$.
  - **Exposure**: Mean gray level $40.0 \le \mu \le 220.0$, pixel clipping $<15\%$.
  - **Motion Blur**: Temporal absolute difference $\frac{1}{N}\sum |I_t - I_{t-1}| \le 18.0$.
  - **Framing Score**: Combined metric $0.0 \dots 1.0$ driving the Web Audio sonar tone frequency on the client.

---

### 4.6. Vision-Language Reasoning — [`server/tier2/vlm.py`](file:///Users/ravi/Documents/SETU/server/tier2/vlm.py)

- **Engine**: Gemma 3 4B via local Ollama HTTP REST API (`http://127.0.0.1:11434/api/generate`).
- **Key Methods**:
  - `describe(jpeg_bytes, question=None)`: Performs multimodal scene interpretation constrained to 2 concise spoken sentences prioritizing safety and navigation.
  - `answer_from_text(extracted_text, question)`: Performs pure text reasoning over OCR-extracted strings without re-encoding image tokens (<400ms latency).
  - `warm_up()`: Evaluates a $1\times 1$ dummy JPEG at startup to preload model weights into VRAM.

---

### 4.7. Speech Subsystem — [`server/speech/stt.py`](file:///Users/ravi/Documents/SETU/server/speech/stt.py) & [`server/speech/tts.py`](file:///Users/ravi/Documents/SETU/server/speech/tts.py)

- **Speech-to-Text (`SpeechToText`)**:
  - Uses `faster-whisper` (`base` model, `cpu`/`int8` quantization).
  - Silero VAD (Voice Activity Detection) filter strips silence.
  - Handles silent audio clips safely without throwing unhandled exceptions.
- **Text-to-Speech (`TextToSpeech`)**:
  - **Primary**: Piper Neural TTS (`models/tts/en_US-lessac-medium.onnx`, 22050 Hz PCM WAV).
  - **Dev Fallback**: macOS `say` + `afconvert`.

---

### 4.8. Client Application & Gesture Engine — [`client/app.js`](file:///Users/ravi/Documents/SETU/client/app.js)

- **Class**: `SetuApp`
- **Blind Accessibility Subsystem**:
  - **Gestures**:
    - *Double-Tap*: Activates voice query / speech input.
    - *Swipe Left / Right*: Cycles active modes (Currency $\leftrightarrow$ Objects $\leftrightarrow$ Proximity $\leftrightarrow$ Read $\leftrightarrow$ Describe $\leftrightarrow$ Question).
    - *Two-Finger Tap*: Immediately interrupts and silences speech playback.
    - *Long Press (650ms)*: Triggers proximity safety scan.
  - **Speech Lock (`_acquireSpeechLock` / `_releaseSpeechLock`)**:
    - Shuts off speech recognizer during speech output.
    - 280ms acoustic grace period prevents room echo feedback.
  - **Browser Fallback**: Integrates `window.speechSynthesis` if server TTS is temporarily unavailable.

---

## 5. REST & WebSocket API Specification

### 5.1. WebSocket Streaming Endpoint

```http
WS /ws/stream
```

- **Client Message Format**:
  ```json
  {
    "type": "frame",
    "mode": "collision",
    "image_b64": "<base64 JPEG>",
    "seq": 102
  }
  ```
- **Server Response Format**:
  ```json
  {
    "type": "result",
    "mode": "collision",
    "seq": 102,
    "collision_alert": "urgent" | "warn" | null,
    "speak": "Stop! Chair right in front of you.",
    "threats": [
      {
        "label": "chair",
        "confidence": 0.88,
        "area_fraction": 0.44,
        "severity": "urgent",
        "bbox": [120.0, 80.0, 520.0, 440.0]
      }
    ],
    "quality": {
      "sharpness": 142.5,
      "mean_luminance": 128.0,
      "framing_score": 0.85,
      "accept": true
    }
  }
  ```

---

### 5.2. REST Endpoints Summary

| Endpoint | Method | Input Payload | Output Schema | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| `/api/currency` | `POST` | `{"image_b64": "..."}` | `{"answered": bool, "speak": str, "denominations": int[], "total_value": int}` | Two-stage banknote verification and counting |
| `/api/proximity` | `POST` | `{"image_b64": "..."}` | `{"answered": bool, "collision_alert": str, "speak": str, "threats": obj[]}` | One-shot obstacle and hazard scan |
| `/api/objects` | `POST` | `{"image_b64": "..."}` | `{"answered": bool, "speak": str, "latency_ms": float}` | Conversational surrounding object exploration |
| `/api/ocr` | `POST` | `{"image_b64": "..."}` | `{"answered": bool, "speak": str, "mean_confidence": float, "chunks": str[]}` | RapidOCR text extraction with spatial ordering |
| `/api/vlm` | `POST` | `{"image_b64": "...", "question": "..."}` | `{"answered": bool, "speak": str, "latency_ms": float}` | Gemma 3 4B multimodal scene reasoning |
| `/api/stt` | `POST` | `{"audio_b64": "..."}` | `{"text": str, "latency_ms": float}` | Offline faster-whisper speech transcription |
| `/api/tts` | `POST` | `{"text": str}` | Binary `audio/wav` stream | Piper neural text-to-speech audio synthesis |

---

## 6. Verification, Testing & Troubleshooting Runbook

### 6.1. Verification Commands

```bash
# 1. Verify all dependencies and model imports
.venv/bin/python -c "import server.main; print('Server module verification successful')"

# 2. Test LAN IP discovery
.venv/bin/python -c "import server.main; print('Active LAN IPs:', server.main.get_network_ips())"

# 3. Validate client-side JavaScript syntax
node --check client/app.js

# 4. Generate/Verify multi-IP HTTPS certificates
./scripts/gen_certs.sh

# 5. Start SETU in Network Mode
./scripts/network.sh
```

### 6.2. Common Gotchas & Solutions

| Symptom | Root Cause | Solution |
| :--- | :--- | :--- |
| **Phone camera fails with `NotAllowedError`** | Mobile browsers reject `getUserMedia` over plain HTTP | Start with `./scripts/network.sh` (enables HTTPS on port `8443`). Connect via `https://<LAN_IP>:8443` and accept the self-signed cert. |
| **TTS speech gets cut off or muted** | Microphone active while speaker plays (acoustic echo ducking) | Fixed by `_acquireSpeechLock()` in [`client/app.js`](file:///Users/ravi/Documents/SETU/client/app.js). If reproducing, verify `this._isSpeaking` is true during playback. |
| **YOLO detects non-existent currency on furniture** | Single-detector false positive without negative class | Handled by Gemma 3 4B presence gate in `/api/currency` before YOLO detections are accepted. |
| **Whisper throws error on quiet voice** | Silero VAD strips 100% silence from audio clip | Handled in [`server/speech/stt.py`](file:///Users/ravi/Documents/SETU/server/speech/stt.py) by catching `ValueError` and returning empty string safely. |
