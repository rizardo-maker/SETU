import { Sonar } from "./sonar.js";

/* ---------------------------------------------------------------------
 * SETU client
 *
 * Design rules that are not optional (see the project document,
 * section 3.5 "audio-first interaction"):
 *   - Every control must be reachable without sight: big full-screen
 *     tap target, keyboard-operable buttons, ARIA labels throughout.
 *   - Speak before you render. The spoken result fires before any
 *     visual DOM update.
 *   - Speech is interruptible: a tap cuts current playback immediately.
 *   - Nothing here calls a browser cloud speech API. STT and TTS both
 *     go through the local server (server/speech/*.py) — see the
 *     comment in sonar usage below for why that's non-negotiable.
 * ------------------------------------------------------------------- */

const FRAME_FPS = 5;
const FRAME_INTERVAL_MS = 1000 / FRAME_FPS;
const FRAME_MAX_DIM = 640;
const JPEG_QUALITY = 0.7;

const TIER1_MODES = new Set(["currency", "text", "obstacle"]);
const TIER2_MODES = new Set(["scene", "question"]);

// Distinct haptic pattern per denomination so a user in a noisy market
// can feel the answer without hearing it. Tune freely; the point is
// that each pattern is *distinguishable* by feel, not exact timing.
const DENOMINATION_HAPTICS = {
  "10": [80],
  "20": [80, 60, 80],
  "50": [80, 60, 80, 60, 80],
  "100": [200],
  "200": [200, 80, 80],
  "500": [200, 80, 200],
  "2000": [200, 80, 200, 80, 200],
};

class SetuApp {
  constructor() {
    this.video = document.getElementById("camera");
    this.canvas = document.createElement("canvas");
    this.ctx2d = this.canvas.getContext("2d", { willReadFrequently: true });
    this.statusEl = document.getElementById("status");
    this.transcriptEl = document.getElementById("transcript");
    this.modeButtons = Array.from(document.querySelectorAll("[data-mode]"));
    this.captureArea = document.getElementById("capture-area");
    this.questionInput = document.getElementById("question-input");
    this.submitQuestionBtn = document.getElementById("submit-question-btn");
    this.describeSceneBtn = document.getElementById("describe-scene-btn");
    this.readTextBtn = document.getElementById("read-text-btn");

    this.sonar = new Sonar();
    this.ws = null;
    this.wsReady = false;
    this.reconnectDelay = 1000;

    this.mode = "currency";
    this.stream = null;
    this.track = null;
    this.streaming = false;
    this.seq = 0;
    this.inFlightSeq = null;
    this.frameTimer = null;

    this.currentAudio = null; // for interruptible playback

    this._bindUI();
  }

  // ---------------- setup ----------------

  async start() {
    console.log("🎥 [SETU Client] Initializing camera...");
    this.announce("Starting camera.");
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: "environment" } },
        audio: false,
      });
      this.video.srcObject = this.stream;
      await this.video.play();
      this.track = this.stream.getVideoTracks()[0];
      console.log("✅ [SETU Client] Camera started successfully:", this.track.label);
    } catch (err) {
      console.error("❌ [SETU Client] Camera error:", err);
      this.announce("Could not access the camera. " + this._explainCameraError(err));
      return;
    }
    this._connectWebSocket();
  }

  _explainCameraError(err) {
    if (location.protocol !== "https:" && location.hostname !== "localhost") {
      return "This page needs to be loaded over HTTPS for the camera to work. Ask whoever set this up to run scripts/gen_certs.sh.";
    }
    if (err && err.name === "NotAllowedError") {
      return "Camera permission was denied. Please allow camera access and reload.";
    }
    return "Please check camera permissions and try again.";
  }

  _connectWebSocket() {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const url = `${proto}://${location.host}/ws/stream`;
    console.log("🔌 [SETU WebSocket] Connecting to", url);
    this.ws = new WebSocket(url);

    this.ws.onopen = () => {
      this.wsReady = true;
      this.reconnectDelay = 1000;
      console.log("✅ [SETU WebSocket] Connected!");
      this.announce("Connected. " + this._modeInstructions());
    };

    this.ws.onclose = () => {
      this.wsReady = false;
      this._stopStreaming();
      console.warn("⚠️ [SETU WebSocket] Disconnected. Reconnecting in", this.reconnectDelay, "ms...");
      this.announce("Disconnected. Reconnecting.");
      setTimeout(() => this._connectWebSocket(), this.reconnectDelay);
      this.reconnectDelay = Math.min(this.reconnectDelay * 1.6, 10000);
    };

    this.ws.onerror = (err) => {
      console.error("❌ [SETU WebSocket] Error:", err);
    };

    this.ws.onmessage = (event) => this._handleServerMessage(JSON.parse(event.data));
  }

  _bindUI() {
    for (const btn of this.modeButtons) {
      btn.addEventListener("click", () => this.setMode(btn.dataset.mode));
    }

    // Direct "Describe Scene" button
    if (this.describeSceneBtn) {
      this.describeSceneBtn.addEventListener("click", () => this.triggerDescribeScene());
    }

    // Direct "Read Text Now" button
    if (this.readTextBtn) {
      this.readTextBtn.addEventListener("click", () => this.triggerReadText());
    }

    // Direct "Ask Question" button
    if (this.submitQuestionBtn) {
      this.submitQuestionBtn.addEventListener("click", () => this.triggerAskQuestion());
    }

    // Enter key in question input
    if (this.questionInput) {
      this.questionInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter") {
          e.preventDefault();
          this.triggerAskQuestion();
        }
      });
    }

    // Big full-screen capture target: tap to start/stop
    this.captureArea.addEventListener("click", () => this._onCaptureTap());
    this.captureArea.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        this._onCaptureTap();
      }
    });
  }

  triggerDescribeScene() {
    console.log("📸 [SETU UI] 'Describe Scene' button triggered!");
    this.setMode("scene");
    this._interruptSpeech();
    if (this.statusEl) this.statusEl.textContent = "Analyzing scene with Gemma3...";
    this._sendVLMRequest(null);
  }

  async triggerReadText() {
    const q = this.questionInput ? this.questionInput.value.trim() : "";
    console.log("📖 [SETU UI] 'Read Text' triggered! Question:", q);
    this.setMode("text");
    this._interruptSpeech();
    if (this.statusEl) {
      this.statusEl.textContent = q
        ? `Reading text & asking Gemma3: "${q}"...`
        : "Reading text via OCR & reasoning with Gemma3...";
    }

    const b64 = this._captureFrame();
    if (!b64) {
      console.error("❌ [SETU Text] No frame captured — camera may not be ready.");
      this.announce("Camera is not ready. Please allow camera access and try again.");
      return;
    }

    try {
      const res = await fetch("/api/ocr", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image_b64: b64, question: q || null }),
      });
      const data = await res.json();
      console.log(`📥 [SETU OCR Response] (${data.latency_ms}ms):`, data);

      if (res.ok) {
        this.announce(data.speak);
        this._renderResult({
          tier: data.tier || 1,
          mode: "text",
          speak: data.speak,
          ocr_text: data.ocr_text,
          latency_ms: data.latency_ms,
        });
      } else {
        this.announce(data.speak || data.error || "Could not read text.");
      }
    } catch (err) {
      console.error("❌ [SETU OCR] Fetch error, falling back to WebSocket frame:", err);
      this._sendSingleTextFrame(q || null);
    }
  }

  triggerAskQuestion() {
    const q = this.questionInput ? this.questionInput.value.trim() : "";
    if (this.mode === "text") {
      this.triggerReadText();
      return;
    }
    const questionText = q || "What is in front of me?";
    console.log("❓ [SETU UI] 'Ask Question' triggered! Question:", questionText);
    this.setMode("question");
    this._interruptSpeech();
    if (this.statusEl) this.statusEl.textContent = `Asking Gemma3: "${questionText}"...`;
    this._sendVLMRequest(questionText);
  }

  _sendSingleTextFrame(question) {
    if (!this.wsReady) {
      console.warn("⚠️ [SETU WebSocket] Cannot send frame — WebSocket is not open.");
      this.announce("Server is not connected. Please wait a moment.");
      return;
    }
    const b64 = this._captureFrame();
    if (!b64) {
      console.error("❌ [SETU Text] No frame captured — camera may not be ready.");
      this.announce("Camera is not ready. Please allow camera access and try again.");
      return;
    }

    this.seq += 1;
    this.inFlightSeq = this.seq;
    const payload = {
      type: "frame",
      mode: "text",
      image_b64: b64,
      seq: this.seq,
    };
    if (question) {
      payload.question = question;
    }

    console.log(`🚀 [SETU Send] One-shot Read Text frame seq=${this.seq} | Question="${question || ""}" | Image size=${b64.length} bytes`);
    this.ws.send(JSON.stringify(payload));
  }

  async _sendVLMRequest(question) {
    const b64 = this._captureFrame();
    if (!b64) {
      console.error("❌ [SETU VLM] No frame captured — camera may not be ready.");
      this.announce("Camera is not ready. Please allow camera access and try again.");
      return;
    }

    const body = { image_b64: b64 };
    if (question) body.question = question;

    console.log(`🚀 [SETU VLM] POST /api/vlm — image_b64 len=${b64.length}, question=${JSON.stringify(question)}`);
    if (this.statusEl) this.statusEl.textContent = question
      ? `Asking Gemma3: "${question}"...`
      : "Analyzing scene with Gemma3...";

    try {
      const res = await fetch("/api/vlm", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const data = await res.json();
      console.log(`📥 [SETU VLM] Response (${data.latency_ms}ms, model=${data.model}):`, data.speak);

      if (res.ok) {
        this.announce(data.speak);
        if (this.statusEl) {
          this.statusEl.textContent = `[Moondream ${data.latency_ms}ms] ${data.speak}`;
        }
      } else {
        console.error("❌ [SETU VLM] Server error:", data);
        this.announce(data.speak || data.error || "Something went wrong.");
      }
    } catch (err) {
      console.error("❌ [SETU VLM] Fetch error:", err);
      this.announce("Could not reach the server. Please try again.");
    }
  }

  _captureFrame() {
    if (this.video.readyState < 2) {
      console.warn("⚠️ [SETU Camera] Video not ready (readyState=" + this.video.readyState + ")");
      return null;
    }
    const vw = this.video.videoWidth, vh = this.video.videoHeight;
    if (!vw || !vh) {
      console.warn("⚠️ [SETU Camera] Video dimensions are 0.");
      return null;
    }
    const scale = FRAME_MAX_DIM / Math.max(vw, vh);
    this.canvas.width = Math.round(vw * scale);
    this.canvas.height = Math.round(vh * scale);
    this.ctx2d.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);
    const dataUrl = this.canvas.toDataURL("image/jpeg", JPEG_QUALITY);
    return dataUrl.split(",")[1];
  }

  setMode(mode) {
    if (this.mode !== mode) {
      console.log("👉 [SETU UI] Mode changed from", this.mode, "to", mode);
    }
    this.mode = mode;
    this._stopStreaming();
    for (const btn of this.modeButtons) {
      btn.setAttribute("aria-pressed", String(btn.dataset.mode === mode));
    }
    this.announce(this._modeInstructions());
  }

  _modeInstructions() {
    switch (this.mode) {
      case "currency": return "Currency mode. Tap anywhere and hold the note in view.";
      case "text": return "Text reading mode. Tap anywhere and point at the text.";
      case "obstacle": return "Obstacle mode. Tap anywhere to check what's ahead.";
      case "scene": return "Scene description mode. Tap Describe Scene or anywhere on screen.";
      case "question": return "Question mode. Type your question and tap Ask.";
      default: return "";
    }
  }

  // ---------------- capture loop ----------------

  _onCaptureTap() {
    this._interruptSpeech();
    if (this.streaming) {
      console.log("⏹️ [SETU UI] Capture stopped by user tap.");
      this._stopStreaming();
      this.announce("Stopped.");
      return;
    }

    if (TIER2_MODES.has(this.mode)) {
      console.log("📸 [SETU UI] Single-shot capture triggered for mode:", this.mode);
      if (this.mode === "scene") {
        this.triggerDescribeScene();
      } else {
        this.triggerAskQuestion();
      }
      return;
    }

    console.log("▶️ [SETU UI] Continuous streaming capture started for mode:", this.mode);
    this.sonar.start();
    this.streaming = true;
    this.frameTimer = setInterval(() => this._sendFrame(), FRAME_INTERVAL_MS);
  }

  _stopStreaming() {
    this.streaming = false;
    this.sonar.stop();
    if (this.frameTimer) {
      clearInterval(this.frameTimer);
      this.frameTimer = null;
    }
  }

  _sendFrame() {
    if (!this.wsReady) {
      console.warn("⚠️ [SETU WebSocket] Cannot send frame — WebSocket is not open.");
      return;
    }
    const b64 = this._captureFrame();
    if (!b64) return;

    this.seq += 1;
    this.inFlightSeq = this.seq;
    const payload = { type: "frame", mode: this.mode, image_b64: b64, seq: this.seq };
    if ((this.mode === "question" || this.mode === "text") && this.questionInput) {
      const q = this.questionInput.value.trim();
      if (q) {
        payload.question = q;
      } else if (this.mode === "question") {
        payload.question = "What is in front of me?";
      }
    }

    console.log(`🚀 [SETU Send] Frame seq=${this.seq} | Mode=${this.mode} | Question="${payload.question || ""}" | Image size=${b64.length} bytes`);
    this.ws.send(JSON.stringify(payload));
  }

  // ---------------- server messages ----------------

  _handleServerMessage(msg) {
    console.log(`📥 [SETU Received] Type: ${msg.type}`, msg);

    if (msg.type === "guidance") {
      if (msg.seq === this.inFlightSeq) this.inFlightSeq = null;
      this.sonar.update(msg.framing_score);
      if (msg.torch_suggested) this._enableTorch();
      if (msg.spoken_hint) this.announce(msg.spoken_hint, { interrupt: false });
      return;
    }

    if (msg.type === "result") {
      this.inFlightSeq = null;
      console.log(`🎯 [SETU Result] [Tier ${msg.tier}] ${msg.mode}: "${msg.speak}" (took ${msg.latency_ms}ms)`);
      if (msg.answered && TIER1_MODES.has(msg.mode) && msg.mode === "currency") {
        this._stopStreaming();
        this._vibrateForDenomination(msg.label);
      }
      this.announce(msg.speak);
      this._renderResult(msg);
      return;
    }

    if (msg.type === "transcript") {
      if (this.transcriptEl) this.transcriptEl.textContent = msg.text;
      return;
    }

    if (msg.type === "status") {
      console.log("⚡ [SETU Status]", msg);
      if (msg.message) this.announce(msg.message, { interrupt: false });
      const badge = document.getElementById("tier2-badge");
      if (badge) badge.textContent = msg.tier2_available ? "Scene description: available (Gemma3)" : "Scene description: unavailable";
      return;
    }
  }

  _renderResult(msg) {
    if (!this.statusEl) return;
    const conf = msg.confidence != null ? ` (${Math.round(msg.confidence * 100)}%)` : "";
    const latency = msg.latency_ms != null ? ` [${msg.latency_ms}ms]` : "";
    let displayText = `[Tier ${msg.tier}] ${msg.mode}: ${msg.speak}${conf}${latency}`;
    if (msg.ocr_text && msg.ocr_text !== msg.speak) {
      displayText += ` (OCR: "${msg.ocr_text}")`;
    }
    this.statusEl.textContent = displayText;
  }

  async _enableTorch() {
    if (!this.track) return;
    try {
      const caps = this.track.getCapabilities ? this.track.getCapabilities() : {};
      if (caps.torch) {
        await this.track.applyConstraints({ advanced: [{ torch: true }] });
      }
    } catch (_) { /* torch not supported on this device — fine, we tried */ }
  }

  _vibrateForDenomination(label) {
    const pattern = DENOMINATION_HAPTICS[String(label)];
    if (pattern && navigator.vibrate) navigator.vibrate(pattern);
  }

  // ---------------- speech output ----------------

  _interruptSpeech() {
    if (this.currentAudio) {
      this.currentAudio.pause();
      this.currentAudio.currentTime = 0;
      this.currentAudio = null;
    }
  }

  /**
   * Speaks text via the LOCAL server TTS endpoint (server/speech/tts.py),
   * never the browser's speechSynthesis. Reason: keeping every audio
   * path server-side means one place to guarantee nothing leaves the
   * machine, and it lets the deployed voice be a real Indic Piper voice
   * instead of whatever the browser/OS happens to ship.
   */
  async announce(text, { interrupt = true } = {}) {
    if (!text) return;
    if (interrupt) this._interruptSpeech();
    if (this.transcriptEl) this.transcriptEl.textContent = text;
    try {
      const res = await fetch("/api/tts", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      if (!res.ok) return; // TTS unavailable server-side; visual text above still updated
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const audio = new Audio(url);
      this.currentAudio = audio;
      audio.play().catch(() => {});
      audio.onended = () => URL.revokeObjectURL(url);
    } catch (_) {
      // Network/server hiccup — the on-screen transcript is still there.
    }
  }
}

window.addEventListener("DOMContentLoaded", () => {
  const app = new SetuApp();
  app.start();
  window.__setu = app; // for console debugging during development
});
