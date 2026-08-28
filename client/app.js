/* ---------------------------------------------------------------------
 * SETU voice-first client
 *
 * Default state: continuous COLLISION detection streaming to the server
 * over the WebSocket. Only speaks when there's an actual hazard.
 *
 * Voice commands: browser's Web Speech API listens continuously for
 * "hey setu <command>". On match, invokes the corresponding one-shot
 * mode (currency / describe / read / question), speaks the answer,
 * and returns to collision watch automatically.
 *
 * "hey setu question" is the only command that then records the user's
 * spoken question via MediaRecorder, ships the audio to /api/stt for
 * offline whisper transcription, and hands the transcript to Gemma
 * along with the current camera frame.
 *
 * The Web Speech API IS used here for the wake-phrase match — but only
 * for that. It never handles the user's actual question content, which
 * always goes through the local whisper server. That preserves the
 * "user data never leaves the machine" story for the load-bearing part.
 * ------------------------------------------------------------------- */

const FRAME_MAX_DIM = 640;          // collision streaming — small & fast
const JPEG_QUALITY = 0.7;
const HIRES_MAX_DIM = 1600;         // read / currency / describe — text needs detail
const HIRES_QUALITY = 0.9;
const COLLISION_FPS = 3;                                // frames per second we stream during collision watch
const COLLISION_INTERVAL_MS = 1000 / COLLISION_FPS;
const QUESTION_RECORD_MS = 5000;                        // how long we listen after "hey setu question"

// Distinct haptic pulses for the two collision severities.
const COLLISION_HAPTICS = {
  warn: [120, 60, 120],
  urgent: [400, 120, 400, 120, 400],
};

// Distinct haptic pattern per currency denomination.
const DENOMINATION_HAPTICS = {
  "10": [80],
  "20": [80, 60, 80],
  "50": [80, 60, 80, 60, 80],
  "100": [200],
  "200": [200, 80, 80],
  "500": [200, 80, 200],
  "2000": [200, 80, 200, 80, 200],
};

// Wake phrases: just the command word, spoken alone or as the last
// word of the sentence. No "hey setu" prefix — turns out that got
// misheard consistently in noisy rooms. Now:
//   "currency" / "money" / "cash"          -> currency mode
//   "describe" / "scene" / "look"          -> scene description
//   "read" / "read text" / "text"          -> OCR + summarize
//   "question" / "ask" / "ask a question"  -> record & ask
// Plus snooze/resume controls for collision alerts:
//   "stop" / "ok stop" / "quiet" / "mute"  -> stop collision voice
//   "resume" / "start again" / "unmute"    -> resume collision voice
const COMMAND_ALIASES = {
  currency:  ["currency", "money", "cash", "notes"],
  describe:  ["describe", "scene", "what do you see", "look"],
  read:      ["read text", "read this", "read", "text", "ocr"],
  question:  ["ask a question", "question", "ask"],
};
const SNOOZE_PHRASES = ["stop", "ok stop", "okay stop", "quiet", "mute", "shut up"];
const RESUME_PHRASES = ["resume", "start again", "unmute", "wake up", "listen"];


class SetuApp {
  constructor() {
    this.video      = document.getElementById("camera");
    this.canvas     = document.createElement("canvas");
    this.ctx2d      = this.canvas.getContext("2d", { willReadFrequently: true });
    this.modeCard   = document.getElementById("mode-card");
    this.modeIcon   = document.getElementById("mode-icon");
    this.modeLabel  = document.getElementById("mode-label");
    this.modeDetail = document.getElementById("mode-detail");
    this.wakeHint   = document.getElementById("wake-hint");
    this.transcript = document.getElementById("transcript");
    this.footerEl   = document.getElementById("footer-status");
    this.listenBadge = document.getElementById("listen-badge");

    this.ws = null;
    this.wsReady = false;
    this.reconnectDelay = 1000;

    this.stream = null;
    this.track = null;

    // The state machine has exactly one active state at a time.
    //   "collision"      — continuous stream of frames, only speak on hazard
    //   "recording"      — capturing the user's spoken question via MediaRecorder
    //   "processing"     — a one-shot mode is in flight (currency / scene / text / question)
    // Collision streaming is paused during recording/processing so the WebSocket
    // isn't racing with the one-shot request. Urgent collision alerts, when we
    // add mid-mode collision peek, still interrupt.
    this.state = "collision";
    this.collisionTimer = null;
    this.seq = 0;

    this.currentAudio = null;
    this.speechRecognizer = null;
    this.recognizerRestartTimer = null;
    this.mediaRecorder = null;

    this.tier2Available = false;
    this.collisionMuted = false;
    this.lastCollisionState = null;
    this._lastActionAt = 0;
    this._recognizerPaused = false;
    this._wakeWantsToRun = false;
  }

  async start() {
    this._setModeCard("boot", "🚀", "Starting up", "Requesting camera…");
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: "environment" } },
        audio: false,
      });
      this.video.srcObject = this.stream;
      await this.video.play();
      this.track = this.stream.getVideoTracks()[0];
      console.log("✅ Camera:", this.track.label);
    } catch (err) {
      console.error("❌ Camera error:", err);
      this._setModeCard("boot", "❌", "Camera blocked", this._explainCameraError(err));
      this._speak("Camera is not available. " + this._explainCameraError(err));
      return;
    }
    this._connectWebSocket();
    this._startWakeListener();
    this._bindMuteTap();
  }

  // Tap anywhere on the mode card to snooze / resume collision voice
  // alerts. Also serves as the first user gesture that satisfies
  // Chrome's autoplay + vibration policies (both need a real tap
  // before they'll fire).
  _bindMuteTap() {
    if (!this.modeCard) return;
    const toggle = () => this._toggleCollisionMute();
    this.modeCard.addEventListener("click", toggle);
    this.modeCard.setAttribute("role", "button");
    this.modeCard.setAttribute("tabindex", "0");
    this.modeCard.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); toggle(); }
    });
  }

  _toggleCollisionMute() {
    this.collisionMuted = !this.collisionMuted;
    if (this.collisionMuted) {
      this._interruptSpeech();
      this._speak("Muted.");
      this._footer("Collision voice muted. Tap or say 'resume' to unmute.");
    } else {
      this._speak("Listening for hazards.");
      this._footer("Voice + Gemma3 ready.");
    }
  }

  _explainCameraError(err) {
    if (location.protocol !== "https:" && location.hostname !== "localhost") {
      return "Load this page over HTTPS.";
    }
    if (err && err.name === "NotAllowedError") {
      return "Camera permission was denied. Reload and allow it.";
    }
    return "Check camera permissions and try again.";
  }

  // -------- WebSocket lifecycle --------

  _connectWebSocket() {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const url = `${proto}://${location.host}/ws/stream`;
    console.log("🔌 WS connecting", url);
    this.ws = new WebSocket(url);
    this.ws.onopen = () => {
      this.wsReady = true;
      this.reconnectDelay = 1000;
      this._enterCollisionMode();
    };
    this.ws.onclose = () => {
      this.wsReady = false;
      this._stopCollisionStream();
      this._footer("Disconnected — reconnecting…");
      setTimeout(() => this._connectWebSocket(), this.reconnectDelay);
      this.reconnectDelay = Math.min(this.reconnectDelay * 1.6, 10000);
    };
    this.ws.onerror = (e) => console.error("WS error", e);
    this.ws.onmessage = (ev) => this._onServerMessage(JSON.parse(ev.data));
  }

  // -------- Collision mode (default state) --------

  _enterCollisionMode() {
    this.state = "collision";
    this._setModeCard("clear", "🛡️", "Collision watch", "Path is clear.");
    this._startCollisionStream();
  }

  _startCollisionStream() {
    if (this.collisionTimer) return;
    this.collisionTimer = setInterval(() => this._sendCollisionFrame(), COLLISION_INTERVAL_MS);
  }

  _stopCollisionStream() {
    if (this.collisionTimer) {
      clearInterval(this.collisionTimer);
      this.collisionTimer = null;
    }
  }

  _sendCollisionFrame() {
    if (!this.wsReady) return;
    if (this.state !== "collision") return;   // paused during a one-shot mode
    const b64 = this._captureFrame();
    if (!b64) return;
    this.seq += 1;
    this.ws.send(JSON.stringify({
      type: "frame", mode: "collision",
      image_b64: b64, seq: this.seq,
    }));
  }

  // -------- WebSocket incoming (guidance + result for collision mode) --------

  _onServerMessage(msg) {
    if (msg.type === "status") {
      this.tier2Available = !!msg.tier2_available;
      this._footer(this.tier2Available ? "Voice + Gemma3 ready." : "Gemma3 unavailable — describe/question limited.");
      return;
    }
    if (msg.type === "guidance") return;  // ignored in collision mode (no sonar UI)
    if (msg.type === "result" && msg.mode === "collision") {
      this._handleCollisionResult(msg);
      return;
    }
  }

  _handleCollisionResult(msg) {
    const urgent = msg.collision_alert === "urgent";
    const warn = msg.collision_alert === "warn";

    // Urgent hazards ALWAYS interrupt — even mid-mode, even if muted.
    // Mute silences everyday alerts, not an actual "you're about to
    // walk into a car" moment.
    if (urgent) {
      this._setModeCard("urgent", "🛑", "STOP", msg.speak || "Hazard right in front of you.");
      if (msg.speak) this._speak(msg.speak, { interrupt: true });
      this._safeVibrate(COLLISION_HAPTICS.urgent);
      return;
    }

    // Non-urgent updates only touch the UI while collision watch is the
    // active state; otherwise the "Careful" chip would overwrite the
    // mode card mid-answer.
    if (this.state !== "collision") return;

    if (warn) {
      // De-dupe: only speak the "careful" once per hazard, not on every
      // frame while the object is still in view.
      const changed = this.lastCollisionState !== "warn";
      this.lastCollisionState = "warn";
      this._setModeCard("warn", "⚠️", "Careful", msg.speak || "Object close ahead.");
      if (changed && !this.collisionMuted && msg.speak) {
        this._speak(msg.speak, { interrupt: false });
        this._safeVibrate(COLLISION_HAPTICS.warn);
      }
    } else {
      this.lastCollisionState = "clear";
      const label = this.collisionMuted ? "Watching (muted)" : "Collision watch";
      this._setModeCard("clear", "🛡️", label, "Path is clear.");
    }
  }

  _safeVibrate(pattern) {
    // Chrome logs an "Intervention" warning if we call vibrate before
    // the user has ever tapped the frame. Wrapping it silences those
    // logs and avoids the crash on browsers with no vibration API.
    if (!navigator.vibrate) return;
    try { navigator.vibrate(pattern); } catch (_) {}
  }

  // -------- Wake-phrase listener (Web Speech API) --------

  _startWakeListener() {
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SR) {
      this._badge("Voice not supported", false);
      this._footer("This browser lacks SpeechRecognition. Try Chrome or Edge.");
      return;
    }
    this._SR = SR;
    this._wakeWantsToRun = true;
    this._spawnRecognizer();
  }

  // Create a FRESH recognizer each time. Reusing one instance across
  // Chrome's ~60s auto-kill is unreliable — a reused object often throws
  // "already started" or silently never restarts, which is exactly the
  // "commands stop working after the first" symptom. A new object each
  // cycle sidesteps all of that.
  _spawnRecognizer() {
    if (!this._wakeWantsToRun) return;
    if (this._recognizerPaused) return;   // suspended during question recording

    const r = new this._SR();
    r.continuous = true;
    r.interimResults = true;
    r.lang = "en-US";

    r.onstart = () => { this._badge("Listening", true); console.log("🎧 recognizer started"); };
    r.onresult = (ev) => this._onSpeechResult(ev);
    r.onerror = (e) => {
      console.warn("🎧 SR error:", e.error);
      if (e.error === "not-allowed") {
        this._wakeWantsToRun = false;
        this._badge("Mic blocked", false);
        this._footer("Microphone permission was denied. Allow it and reload.");
      }
      // 'no-speech' / 'aborted' / 'network' → just let onend respawn.
    };
    r.onend = () => {
      console.log("🎧 recognizer ended, respawning");
      this._badge("…", false);
      clearTimeout(this.recognizerRestartTimer);
      this.recognizerRestartTimer = setTimeout(() => this._spawnRecognizer(), 300);
    };

    this.speechRecognizer = r;
    try {
      r.start();
    } catch (e) {
      console.warn("🎧 SR start failed, retrying:", e);
      clearTimeout(this.recognizerRestartTimer);
      this.recognizerRestartTimer = setTimeout(() => this._spawnRecognizer(), 500);
    }
  }

  _pauseRecognizer() {
    // Stop the wake recognizer while question mode owns the mic, so the
    // two aren't fighting over the microphone (which kills the recognizer
    // on macOS Chrome and it never comes back).
    this._recognizerPaused = true;
    if (this.speechRecognizer) {
      try { this.speechRecognizer.abort(); } catch (_) {}
    }
  }

  _resumeRecognizer() {
    this._recognizerPaused = false;
    clearTimeout(this.recognizerRestartTimer);
    this.recognizerRestartTimer = setTimeout(() => this._spawnRecognizer(), 300);
  }

  _onSpeechResult(ev) {
    // Read only the latest result in this event. Reading the whole
    // accumulated buffer caused stale phrases to re-trigger.
    const last = ev.results[ev.results.length - 1];
    if (!last) return;
    let heard = (last[0] && last[0].transcript ? last[0].transcript : "").trim().toLowerCase();
    if (!heard) return;
    if (this.transcript) this.transcript.textContent = `heard: “${heard}”`;

    // Don't process our own TTS being picked up by the mic. If audio is
    // currently playing, ignore recognizer input entirely.
    if (this.currentAudio && !this.currentAudio.paused) return;

    // Debounce: act at most once per 1.2s window.
    const now = Date.now();
    if (now - (this._lastActionAt || 0) < 1200) return;

    // 1. Snooze / resume — always processed, even mid-mode.
    if (this._matchesAny(heard, SNOOZE_PHRASES)) {
      this._lastActionAt = now;
      console.log("🔇 snooze:", heard);
      if (!this.collisionMuted) this._toggleCollisionMute();
      return;
    }
    if (this._matchesAny(heard, RESUME_PHRASES)) {
      this._lastActionAt = now;
      console.log("🔊 resume:", heard);
      if (this.collisionMuted) this._toggleCollisionMute();
      return;
    }

    // 2. Mode commands.
    const cmd = this._parseCommand(heard);
    if (!cmd) return;

    console.log("🎙️ command matched:", cmd, "| state:", this.state);

    if (this.state === "collision") {
      this._lastActionAt = now;
      this._dispatchCommand(cmd);
    } else {
      console.log("(ignored — a mode is already running, state=" + this.state + ")");
    }
  }

  _matchesAny(heard, phrases) {
    for (const p of phrases) {
      if (heard === p || heard.startsWith(p + " ") || heard.endsWith(" " + p) || heard.includes(" " + p + " ")) {
        return true;
      }
    }
    return false;
  }

  _parseCommand(heard) {
    // Match longest alias first so "read text" wins over "text".
    // The alias may appear anywhere in the utterance — a user could
    // say "okay, describe this" and we still pick up "describe".
    let match = null;
    let matchLen = 0;
    for (const [cmd, aliases] of Object.entries(COMMAND_ALIASES)) {
      for (const alias of aliases) {
        const ok =
          heard === alias ||
          heard.startsWith(alias + " ") ||
          heard.endsWith(" " + alias) ||
          heard.includes(" " + alias + " ") ||
          heard.startsWith(alias + ".") ||   // Chrome sometimes appends punctuation
          heard.endsWith("." + alias) ||
          heard.endsWith(" " + alias + ".") ||
          heard.includes(" " + alias + ",");
        if (ok && alias.length > matchLen) {
          match = cmd;
          matchLen = alias.length;
        }
      }
    }
    return match;
  }

  async _dispatchCommand(cmd) {
    // Keep the collision stream running while a mode runs on REST — the
    // two transports don't race, and an *urgent* hazard has to be able
    // to interrupt any answer we're speaking. We just mark ourselves as
    // "processing" so non-urgent collision updates don't overwrite the
    // mode card while the user is trying to hear their answer.
    this.state = "processing";
    try {
      switch (cmd) {
        case "currency": await this._runCurrencyMode(); break;
        case "describe": await this._runDescribeMode(); break;
        case "read":     await this._runReadMode(); break;
        case "question": await this._runQuestionMode(); break;
      }
    } catch (err) {
      console.error("[mode crashed]", cmd, err);
    } finally {
      // Guarantee we return to the ready state even if the mode threw.
      // Without this a single fetch error would leave us stuck in
      // "processing" and every future command would be silently
      // ignored by the `_onSpeechResult` state check.
      if (this.state !== "collision") this._returnToCollision();
    }
  }

  _returnToCollision() {
    this.state = "collision";
    this._lastActionAt = 0;          // clear debounce so the next command fires immediately
    this.lastCollisionState = null;  // let the next warn/clear re-announce
    this._enterCollisionMode();
    console.log("↩️  returned to collision watch — ready for next command");
  }

  // -------- Mode: currency (POST /api/vlm with a currency-specific prompt) --------
  //
  // Currency detection is a Tier-1 YOLO model that streams over the
  // WebSocket, but the *voice-invoked* single-shot use case wants a
  // deterministic REST round-trip. We use /api/vlm with a currency
  // prompt so a single frame is analyzed. If the frame's blurry or
  // empty, Gemma politely says so instead of us silently returning
  // garbage.

  async _runCurrencyMode() {
    this._setModeCard("mode", "💵", "Currency", "Scanning for notes…");
    // Small delay so the user has time to steady the camera. No spoken
    // prompt — the user just said "currency", they know they've been
    // heard from the visual state change.
    await this._sleep(600);
    const b64 = this._captureFrame(HIRES_MAX_DIM, HIRES_QUALITY);
    if (!b64) {
      await this._sayAndWait("Camera not ready.");
      this._returnToCollision();
      return;
    }
    let speak = "No currency detected.";
    try {
      const res = await fetch("/api/currency", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image_b64: b64 }),
      });
      const data = await res.json();
      speak = data.speak || speak;
      this._setModeCard("mode", "💵", "Currency", speak);
    } catch (err) {
      console.error(err);
      speak = "Could not reach the server.";
    }
    try { await this._sayAndWait(speak); } catch (_) {}
    this._returnToCollision();
  }

  // -------- Mode: describe --------

  async _runDescribeMode() {
    this._setModeCard("mode", "👁️", "Describe scene", "Looking…");
    await this._sleep(400);
    const b64 = this._captureFrame(HIRES_MAX_DIM, HIRES_QUALITY);
    if (!b64) {
      await this._sayAndWait("Camera not ready.");
      this._returnToCollision();
      return;
    }
    let speak = "I could not describe the scene.";
    try {
      const res = await fetch("/api/vlm", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image_b64: b64 }),
      });
      const data = await res.json();
      speak = data.speak || speak;
      this._setModeCard("mode", "👁️", "Describe scene", speak);
    } catch (err) {
      console.error(err);
      speak = "Could not reach the server.";
    }
    try { await this._sayAndWait(speak); } catch (_) {}
    this._returnToCollision();
  }

  // -------- Mode: read (OCR + Gemma summary) --------

  async _runReadMode() {
    this._setModeCard("mode", "📖", "Read text", "Reading…");
    await this._sleep(400);
    // High-res capture — OCR needs the detail. The 640px collision
    // frame reads as garbage; text needs ~1600px to be legible.
    const b64 = this._captureFrame(HIRES_MAX_DIM, HIRES_QUALITY);
    if (!b64) {
      await this._sayAndWait("Camera not ready.");
      this._returnToCollision();
      return;
    }
    let speak = "No text found.";
    try {
      const res = await fetch("/api/ocr", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          image_b64: b64,
          question: "Read the visible text and give me a one or two sentence summary of what it says.",
        }),
      });
      const data = await res.json();
      speak = data.speak || speak;
      this._setModeCard("mode", "📖", "Read text", speak);
    } catch (err) {
      console.error(err);
      speak = "Could not reach the server.";
    }
    try { await this._sayAndWait(speak); } catch (_) {}
    this._returnToCollision();
  }

  // -------- Mode: question (record audio -> STT -> Gemma) --------

  async _runQuestionMode() {
    // Question mode is the one place we DO need a spoken prompt — the
    // user has to know that mic recording has started.
    this._setModeCard("listen", "🎤", "Ask a question", "Ask your question now…");
    await this._sayAndWait("Ask your question.");
    try {
      const audioB64 = await this._recordAudio(QUESTION_RECORD_MS);
      this._setModeCard("mode", "🧠", "Ask a question", "Transcribing…");
      const sttRes = await fetch("/api/stt", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ audio_b64: audioB64 }),
      });
      const sttData = await sttRes.json();
      const question = (sttData.text || "").trim();
      if (!question) {
        await this._sayAndWait("I could not hear a question. Try again.");
        this._returnToCollision();
        return;
      }
      this._setModeCard("mode", "🧠", "Question", `“${question}” — thinking…`);
      const b64 = this._captureFrame(HIRES_MAX_DIM, HIRES_QUALITY);
      const res = await fetch("/api/vlm", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image_b64: b64, question }),
      });
      const data = await res.json();
      const speak = data.speak || "I don't know the answer.";
      this._setModeCard("mode", "🧠", "Question", speak);
      await this._sayAndWait(speak);
    } catch (err) {
      console.error(err);
      await this._sayAndWait("Something went wrong recording your question.");
    }
    this._returnToCollision();
  }

  // -------- Audio recording for question mode --------

  async _recordAudio(durationMs) {
    // Pause the wake recognizer so it isn't fighting question mode for
    // the microphone — on macOS Chrome that fight kills the recognizer
    // permanently, which is why commands stopped working after using
    // question mode. We resume it in the finally.
    this._pauseRecognizer();
    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const rec = new MediaRecorder(stream, { mimeType: "audio/webm" });
      const chunks = [];
      rec.ondataavailable = (e) => { if (e.data && e.data.size) chunks.push(e.data); };
      const done = new Promise((resolve) => { rec.onstop = () => resolve(); });
      rec.start();
      await this._sleep(durationMs);
      rec.stop();
      stream.getTracks().forEach((t) => t.stop());
      await done;
      const blob = new Blob(chunks, { type: "audio/webm" });
      const buf = await blob.arrayBuffer();
      const bytes = new Uint8Array(buf);
      let bin = "";
      const CHUNK = 0x8000;
      for (let i = 0; i < bytes.length; i += CHUNK) {
        bin += String.fromCharCode.apply(null, bytes.subarray(i, i + CHUNK));
      }
      return btoa(bin);
    } catch (err) {
      throw new Error("Microphone permission denied.");
    } finally {
      // Give the OS a beat to release the mic before the recognizer grabs it.
      setTimeout(() => this._resumeRecognizer(), 400);
    }
  }

  // -------- Frame capture --------

  _captureFrame(maxDim = FRAME_MAX_DIM, quality = JPEG_QUALITY) {
    if (this.video.readyState < 2) return null;
    const vw = this.video.videoWidth, vh = this.video.videoHeight;
    if (!vw || !vh) return null;
    // Never upscale — cap the scale at 1.0 so we send the native
    // resolution at most.
    const scale = Math.min(1.0, maxDim / Math.max(vw, vh));
    this.canvas.width = Math.round(vw * scale);
    this.canvas.height = Math.round(vh * scale);
    this.ctx2d.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);
    return this.canvas.toDataURL("image/jpeg", quality).split(",")[1];
  }

  // -------- UI helpers --------

  _setModeCard(state, icon, label, detail) {
    if (this.modeCard) this.modeCard.dataset.state = state;
    if (this.modeIcon) this.modeIcon.textContent = icon;
    if (this.modeLabel) this.modeLabel.textContent = label;
    if (this.modeDetail) this.modeDetail.textContent = detail || "";
  }

  _badge(text, listening) {
    if (!this.listenBadge) return;
    this.listenBadge.textContent = text;
    this.listenBadge.classList.toggle("listening", !!listening);
  }

  _footer(text) {
    if (this.footerEl) this.footerEl.textContent = text;
  }

  _sleep(ms) { return new Promise((r) => setTimeout(r, ms)); }

  // -------- TTS (interruptible) --------

  _interruptSpeech() {
    if (this.currentAudio) {
      try { this.currentAudio.pause(); } catch (_) {}
      this.currentAudio = null;
    }
  }

  async _speak(text, { interrupt = true } = {}) {
    if (!text) return;
    if (interrupt) this._interruptSpeech();
    try {
      const res = await fetch("/api/tts", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      if (!res.ok) return;
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const audio = new Audio(url);
      this.currentAudio = audio;
      audio.play().catch(() => {});
      audio.onended = () => URL.revokeObjectURL(url);
    } catch (_) { /* network hiccup, transcript still shown */ }
  }

  // Speak text and only resolve after playback ends (or 8s max as a
  // safety net). Used inside modes so the "return to collision" step
  // doesn't stomp on the answer mid-sentence.
  async _sayAndWait(text) {
    if (!text) return;
    this._interruptSpeech();
    try {
      const res = await fetch("/api/tts", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      if (!res.ok) return;
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const audio = new Audio(url);
      this.currentAudio = audio;
      await new Promise((resolve) => {
        const cleanup = () => { URL.revokeObjectURL(url); resolve(); };
        audio.onended = cleanup;
        audio.onerror = cleanup;
        audio.play().catch(cleanup);
        setTimeout(cleanup, 12000);   // safety cap
      });
    } catch (_) {}
  }
}

window.addEventListener("DOMContentLoaded", () => {
  const app = new SetuApp();
  window.__setu = app;
  app.start();
});
