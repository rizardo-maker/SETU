/**
 * SETU Client Application
 * Voice-first assistive vision app for blind & low-vision navigation.
 * 
 * Direct collision detection pipeline from prox/ architecture:
 * - Default background state is continuous Collision Watch.
 * - Urgent collisions ALWAYS interrupt with top priority audio & haptics.
 * - Single-shot modes (Currency, Objects, Describe, Read, Question) execute and return to collision watch.
 */

const FRAME_MAX_DIM = 640;          // collision streaming — small & fast
const JPEG_QUALITY = 0.7;
const HIRES_MAX_DIM = 2048;         // read / currency / describe — full detail
const HIRES_QUALITY = 0.95;
const COLLISION_FPS = 3;            // frames per second during collision watch
const COLLISION_INTERVAL_MS = 1000 / COLLISION_FPS;
const QUESTION_RECORD_MS = 5000;    // listen duration after "question"

// Distinct haptic pulses for collision severities
const COLLISION_HAPTICS = {
  warn: [120, 60, 120],
  urgent: [400, 120, 400, 120, 400],
};

const COMMAND_ALIASES = {
  currency: [
    "currency detection", "detect currency", "currency", "money", "cash",
    "notes", "rupees", "note", "check currency", "count money", "how much money", "what note is this", "what note"
  ],
  objects: [
    "detect objects", "object detection", "objects", "object", "items",
    "detect items", "find objects", "what items", "what is in front of me", "what do you see", "obstacle", "obstacles"
  ],
  proximity: [
    "proximity detection", "proximity", "collision", "safety", "radar",
    "watch path", "clear path", "check path", "collision watch"
  ],
  describe: [
    "describe scene", "scene description", "tell me what's around", "describe",
    "scene", "look around", "surroundings", "describe the frame"
  ],
  read: [
    "read text", "read this", "read sign", "read document", "read label", "read", "text", "ocr"
  ],
  question: [
    "ask a question", "question", "ask", "what is this", "where is", "tell me"
  ],
  help: [
    "help", "voice help", "tutorial", "how to use", "instructions", "guide", "what can i say", "commands"
  ],
};

const SNOOZE_PHRASES = ["stop", "ok stop", "okay stop", "quiet", "mute", "shut up"];
const RESUME_PHRASES = ["resume", "start again", "unmute", "wake up", "listen"];


class SetuApp {
  constructor() {
    this.video       = document.getElementById("camera");
    this.canvas      = document.createElement("canvas");
    this.ctx2d       = this.canvas.getContext("2d", { willReadFrequently: true });
    this.modeCard    = document.getElementById("mode-card");
    this.modeIcon    = document.getElementById("mode-icon");
    this.modeLabel   = document.getElementById("mode-label");
    this.modeDetail  = document.getElementById("mode-detail");
    this.wakeHint    = document.getElementById("wake-hint");
    this.transcript  = document.getElementById("transcript");
    this.footerEl    = document.getElementById("footer-status");
    this.listenBadge = document.getElementById("listen-badge");

    this.ws = null;
    this.wsReady = false;
    this.reconnectDelay = 1000;

    this.stream = null;
    this.track = null;

    // State machine: "collision", "processing", "reading", "recording"
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
    this._recognizerGen = 0;

    this._modeEpoch = 0;
    this._modeInFlight = false;
    this._lastReadText = "";

    // Speech & microphone synchronization locks
    this._isSpeaking = false;
    this._speechLockCount = 0;
    this._micResumeTimer = null;
    this._activeUtterance = null;
  }

  _epochLive(epoch) {
    return epoch === this._modeEpoch;
  }

  async start() {
    this._setModeCard("boot", "🚀", "Starting up", "Requesting camera…");
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: { ideal: "environment" },
          width: { ideal: 1920 },
          height: { ideal: 1080 },
        },
        audio: false,
      });
      this.video.srcObject = this.stream;
      await this.video.play();
      this.track = this.stream.getVideoTracks()[0];
      console.log("✅ Camera active:", this.track.label);
    } catch (err) {
      console.error("❌ Camera error:", err);
      this._setModeCard("boot", "❌", "Camera blocked", this._explainCameraError(err));
      this._speak("Camera is not available. " + this._explainCameraError(err));
      return;
    }
    this._connectWebSocket();
    this._startWakeListener();
    this._bindMuteTap();
    this._bindActionButtons();
    this._bindHelpButton();
    this._bindBlindGestures();
    this._bindKeyboardShortcuts();
    this._bindVisibilityLogging();
  }

  _bindHelpButton() {
    const helpBtn = document.getElementById("btn-help-voice");
    if (helpBtn) {
      helpBtn.addEventListener("click", () => {
        this._tryInvokeCommand("help", "tap");
      });
    }
  }

  _bindBlindGestures() {
    let lastTapTime = 0;
    let touchStartX = 0;
    let touchStartY = 0;
    let touchStartTime = 0;
    let longPressTimer = null;

    const root = document.getElementById("app-root") || document.body;

    root.addEventListener("touchstart", (e) => {
      // Two-finger tap immediately silences audio
      if (e.touches.length === 2) {
        e.preventDefault();
        this._interruptSpeech();
        if (navigator.vibrate) navigator.vibrate(50);
        console.log("🤫 Two-finger tap: Silenced speech");
        return;
      }

      if (e.touches.length === 1) {
        touchStartX = e.touches[0].clientX;
        touchStartY = e.touches[0].clientY;
        touchStartTime = Date.now();

        // Long press (650ms) triggers Proximity / Collision safety check
        longPressTimer = setTimeout(() => {
          if (navigator.vibrate) navigator.vibrate([200, 80, 200]);
          console.log("🛑 Long press: Triggered Proximity check");
          this._tryInvokeCommand("proximity", "gesture");
        }, 650);
      }
    }, { passive: false });

    root.addEventListener("touchmove", (e) => {
      if (e.touches.length === 1 && longPressTimer) {
        const dx = Math.abs(e.touches[0].clientX - touchStartX);
        const dy = Math.abs(e.touches[0].clientY - touchStartY);
        if (dx > 15 || dy > 15) {
          clearTimeout(longPressTimer);
          longPressTimer = null;
        }
      }
    }, { passive: true });

    root.addEventListener("touchend", (e) => {
      if (longPressTimer) {
        clearTimeout(longPressTimer);
        longPressTimer = null;
      }

      if (e.changedTouches.length === 1) {
        const touchEndX = e.changedTouches[0].clientX;
        const touchEndY = e.changedTouches[0].clientY;
        const dx = touchEndX - touchStartX;
        const dy = touchEndY - touchStartY;
        const dt = Date.now() - touchStartTime;

        // Swipe horizontal (> 65px) switches mode
        if (Math.abs(dx) > 65 && Math.abs(dy) < 60 && dt < 600) {
          if (dx > 0) {
            this._cycleMode(1); // Swipe right -> Next mode
          } else {
            this._cycleMode(-1); // Swipe left -> Prev mode
          }
          return;
        }

        // Tap handling
        if (Math.abs(dx) < 20 && Math.abs(dy) < 20 && dt < 350) {
          const now = Date.now();
          if (now - lastTapTime < 320) {
            // Double Tap anywhere -> Trigger voice command
            lastTapTime = 0;
            if (navigator.vibrate) navigator.vibrate([60, 40, 60]);
            console.log("🎙️ Double tap: Listening for voice command");
            this._sayAndWait("Listening. Speak your command.");
            return;
          }
          lastTapTime = now;
        }
      }
    });
  }

  _bindKeyboardShortcuts() {
    window.addEventListener("keydown", (e) => {
      // Don't intercept if user is typing in an input
      if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;

      switch (e.key) {
        case "1": this._tryInvokeCommand("currency", "key"); break;
        case "2": this._tryInvokeCommand("objects", "key"); break;
        case "3": this._tryInvokeCommand("proximity", "key"); break;
        case "4": this._tryInvokeCommand("read", "key"); break;
        case "5": this._tryInvokeCommand("describe", "key"); break;
        case "6": this._tryInvokeCommand("question", "key"); break;
        case "ArrowRight": e.preventDefault(); this._cycleMode(1); break;
        case "ArrowLeft": e.preventDefault(); this._cycleMode(-1); break;
        case "v": case "V": case "l": case "L":
          e.preventDefault();
          this._sayAndWait("Listening. Speak your command.");
          break;
        case "s": case "S": case "Escape":
          e.preventDefault();
          this._interruptSpeech();
          break;
        case "h": case "H": case "?":
          e.preventDefault();
          this._tryInvokeCommand("help", "key");
          break;
      }
    });
  }

  _cycleMode(delta) {
    const modes = ["currency", "objects", "proximity", "read", "describe", "question"];
    if (this._currentModeIdx === undefined) this._currentModeIdx = 0;
    this._currentModeIdx = (this._currentModeIdx + delta + modes.length) % modes.length;
    const nextCmd = modes[this._currentModeIdx];
    if (navigator.vibrate) navigator.vibrate(40);
    this._tryInvokeCommand(nextCmd, "gesture");
  }

  _bindVisibilityLogging() {
    document.addEventListener("visibilitychange", () => {
      console.log(`👁️ tab visibility: ${document.visibilityState} | video.readyState=${this.video.readyState}`);
    });
  }

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

  _bindActionButtons() {
    document.querySelectorAll("[data-cmd]").forEach((btn) => {
      btn.addEventListener("click", (e) => {
        e.stopPropagation();
        const cmd = btn.getAttribute("data-cmd");
        if (cmd) {
          console.log(`🔘 Mode command button clicked: ${cmd}`);
          this._tryInvokeCommand(cmd, "tap");
        }
      });
    });
  }

  _toggleCollisionMute() {
    this.collisionMuted = !this.collisionMuted;
    if (this.collisionMuted) {
      this._footer("Collision voice muted. Tap card or say 'resume' to re-enable.");
      this._setModeCard(this.lastCollisionState || "clear", "🔇", "Watching (muted)", "Voice alerts paused.");
      this._speak("Collision alerts muted.");
    } else {
      this._footer("Collision voice active.");
      this._setModeCard(this.lastCollisionState || "clear", "🛡️", "Collision watch", "Path is clear.");
      this._speak("Collision alerts active.");
    }
  }

  _explainCameraError(err) {
    if (err.name === "NotAllowedError" || err.name === "PermissionDeniedError") {
      return "Camera permission denied. Allow access in browser settings.";
    }
    if (err.name === "NotFoundError" || err.name === "DevicesNotFoundError") {
      return "No camera found on this device.";
    }
    return err.message || "Unknown camera error.";
  }

  // -------- WebSocket Connection & Streaming --------
  _connectWebSocket() {
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    const url = `${proto}//${location.host}/ws/stream`;
    this.ws = new WebSocket(url);

    this.ws.onopen = () => {
      this.wsReady = true;
      this.reconnectDelay = 1000;
      this._footer("Connected to server.");
      this._enterCollisionMode();
    };

    this.ws.onclose = () => {
      this.wsReady = false;
      this._stopCollisionStream();
      this._footer("Disconnected — reconnecting…");
      setTimeout(() => this._connectWebSocket(), this.reconnectDelay);
      this.reconnectDelay = Math.min(this.reconnectDelay * 1.6, 10000);
    };

    this.ws.onerror = (e) => console.error("WS error:", e);
    this.ws.onmessage = (ev) => this._onServerMessage(JSON.parse(ev.data));
  }

  // -------- Proximity / Collision Mode (Default Continuous Background Watch) --------
  _enterCollisionMode() {
    this.state = "collision";
    const label = this.collisionMuted ? "Watching (muted)" : "Collision watch";
    this._setModeCard("clear", "🛡️", label, "Path is clear.");
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
    if (this.state !== "collision") return;
    const b64 = this._captureFrame();
    if (!b64) return;
    this.seq += 1;
    this.ws.send(JSON.stringify({
      type: "frame", mode: "collision",
      image_b64: b64, seq: this.seq,
    }));
  }

  _onServerMessage(msg) {
    if (msg.type === "status") {
      this.tier2Available = !!msg.tier2_available;
      this._footer(this.tier2Available ? "Voice + Gemma3 ready." : "Gemma3 unavailable — describe/question limited.");
      return;
    }
    if (msg.type === "guidance") return;
    if (msg.type === "result" && msg.mode === "collision") {
      this._handleCollisionResult(msg);
      return;
    }
  }

  _handleCollisionResult(msg) {
    const urgent = msg.collision_alert === "urgent";
    const warn = msg.collision_alert === "warn";

    // 1. Urgent hazards ALWAYS interrupt — even mid-mode, even if muted.
    if (urgent) {
      if (this.state === "processing" || this.state === "reading") {
        console.log(`🛑 urgent collision interrupts in-flight mode #${this._modeEpoch}`);
        this._modeEpoch++;       // supersede whatever mode is running
        this._modeInFlight = false;
      }
      this.state = "collision";  // an urgent alert always returns us to collision-watch framing
      this._setModeCard("urgent", "🛑", "STOP", msg.speak || "Hazard right in front of you.");
      if (msg.speak) this._speak(msg.speak, { interrupt: true });
      this._safeVibrate(COLLISION_HAPTICS.urgent);
      return;
    }

    // Non-urgent updates only touch the UI while collision watch is the active state
    if (this.state !== "collision") return;

    if (warn) {
      // De-dupe: only speak the "careful" once per hazard, not on every frame while the object is still in view
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
    if (!navigator.vibrate) return;
    try { navigator.vibrate(pattern); } catch (_) {}
  }

  // -------- Wake & Speech Recognition --------
  _startWakeListener() {
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SR) {
      this._badge("Voice not supported", false);
      this._footer("Browser lacks SpeechRecognition. Use buttons.");
      return;
    }
    this._SR = SR;
    this._wakeWantsToRun = true;
    this._spawnRecognizer();
  }

  _spawnRecognizer() {
    // Never turn on microphone while SETU is speaking, paused, or recording audio
    if (!this._wakeWantsToRun || this._recognizerPaused || this._isSpeaking || this.state === "recording") return;
    const gen = ++this._recognizerGen;
    const r = new this._SR();
    r.continuous = true;
    r.interimResults = true;
    r.lang = "en-US";

    r.onstart = () => {
      if (gen !== this._recognizerGen) return;
      if (this._isSpeaking || this.state === "recording") {
        try { r.abort(); } catch (_) {}
        return;
      }
      this._badge("Listening", true);
    };
    r.onresult = (ev) => {
      if (gen !== this._recognizerGen) return;
      if (this._isSpeaking || this.state === "recording") return;
      this._onSpeechResult(ev);
    };
    r.onerror = (e) => {
      if (gen !== this._recognizerGen) return;
      if (e.error === "not-allowed") {
        this._wakeWantsToRun = false;
        this._badge("Mic blocked", false);
      }
    };
    r.onend = () => {
      if (gen !== this._recognizerGen) return;
      if (this._isSpeaking) {
        this._badge("Speaking 🔊", false, "speaking");
        return;
      }
      this._badge("…", false);
      clearTimeout(this.recognizerRestartTimer);
      if (!this._recognizerPaused && !this._isSpeaking && this.state !== "recording") {
        this.recognizerRestartTimer = setTimeout(() => this._spawnRecognizer(), 300);
      }
    };

    this.speechRecognizer = r;
    try {
      r.start();
    } catch (e) {
      if (gen !== this._recognizerGen) return;
      clearTimeout(this.recognizerRestartTimer);
      if (!this._recognizerPaused && !this._isSpeaking) {
        this.recognizerRestartTimer = setTimeout(() => this._spawnRecognizer(), 500);
      }
    }
  }

  _pauseRecognizer() {
    this._recognizerPaused = true;
    this._recognizerGen++;
    clearTimeout(this.recognizerRestartTimer);
    if (this.speechRecognizer) {
      try { this.speechRecognizer.abort(); } catch (_) {}
      this.speechRecognizer = null;
    }
  }

  _resumeRecognizer() {
    if (this._isSpeaking || this.state === "recording" || !this._wakeWantsToRun) return;
    this._recognizerPaused = false;
    clearTimeout(this.recognizerRestartTimer);
    this.recognizerRestartTimer = setTimeout(() => this._spawnRecognizer(), 150);
  }

  _onSpeechResult(ev) {
    if (this._isSpeaking || this.state === "recording") return;
    const last = ev.results[ev.results.length - 1];
    if (!last) return;
    let heard = (last[0] && last[0].transcript ? last[0].transcript : "").trim().toLowerCase();
    if (!heard) return;
    if (this.transcript) this.transcript.textContent = `heard: “${heard}”`;

    // Only process finalized segments to prevent double-triggering
    if (!last.isFinal) return;

    const now = Date.now();

    // 1. Snooze / mute / stop (always allowed to interrupt)
    if (this._matchesAny(heard, SNOOZE_PHRASES)) {
      this._lastActionAt = now;
      this._interruptSpeech();
      this._modeEpoch++;
      this._modeInFlight = false;
      this._returnToCollision();
      this._speak("Stopped.");
      return;
    }

    // If currently processing a command, ignore new command triggers
    if (this.state === "processing") return;

    // 2. Resume / unmute
    if (this._matchesAny(heard, RESUME_PHRASES)) {
      this._lastActionAt = now;
      if (this.collisionMuted) this._toggleCollisionMute();
      return;
    }

    // 3. Mode command
    const cmd = this._parseCommand(heard);
    if (!cmd) return;

    if (now - (this._lastActionAt || 0) < 600) return;
    this._lastActionAt = now;

    console.log(`🎙️ Command matched: ${cmd} (heard: "${heard}")`);
    this._tryInvokeCommand(cmd, "voice");
  }

  _tryInvokeCommand(cmd, source) {
    this._interruptSpeech();
    this._modeInFlight = true;
    const epoch = ++this._modeEpoch;
    this._dispatchCommand(cmd, epoch);
    return true;
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
    let match = null;
    let matchLen = 0;
    for (const [cmd, aliases] of Object.entries(COMMAND_ALIASES)) {
      for (const alias of aliases) {
        const ok =
          heard === alias ||
          heard.startsWith(alias + " ") ||
          heard.endsWith(" " + alias) ||
          heard.includes(" " + alias + " ") ||
          heard.startsWith(alias + ".") ||
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

  async _dispatchCommand(cmd, epoch) {
    console.log(`▶️ Dispatch #${epoch}: ${cmd}`);
    this.state = "processing";
    try {
      switch (cmd) {
        case "currency":  await this._runCurrencyMode(epoch); break;
        case "objects":   await this._runObjectsMode(epoch); break;
        case "proximity": await this._runProximityMode(epoch); break;
        case "describe":  await this._runDescribeMode(epoch); break;
        case "read":      await this._runReadMode(epoch); break;
        case "question":  await this._runQuestionMode(epoch); break;
        case "help":      await this._runHelpMode(epoch); break;
      }
    } catch (err) {
      console.error(`❌ [Mode #${epoch} error]`, cmd, err);
    } finally {
      if (this._epochLive(epoch) && this.state !== "reading") {
        this._modeInFlight = false;
        this._returnToCollision();
      }
    }
  }

  _returnToCollision() {
    this.state = "collision";
    this._lastActionAt = 0;
    this.lastCollisionState = null;
    this._enterCollisionMode();
    console.log("↩️ Returned to collision watch — ready for next command");
  }

  // -------- Mode: Help (Audio Tutorial for Blind & Low-Vision Users) --------
  async _runHelpMode(epoch) {
    this._setModeCard("mode", "🔊", "Voice Help", "Playing audio instructions…");
    const tutorialText = (
      "Welcome to SETU. You can control this app using voice or touch gestures. " +
      "Double-tap anywhere on screen to speak a command. " +
      "Swipe left or right to switch between Currency, Objects, Proximity, Read, Describe, and Question. " +
      "Tap with two fingers to silence audio at any time."
    );
    if (this.transcript) this.transcript.textContent = tutorialText;
    if (this._epochLive(epoch)) {
      try { await this._sayAndWait(tutorialText); } catch (_) {}
    }
  }

  // -------- Mode: Currency (Dedicated 95%+ Precision Banknote YOLO) --------
  async _runCurrencyMode(epoch) {
    this._setModeCard("mode", "💵", "Currency", "Scanning notes…");
    await this._sleep(400);
    if (!this._epochLive(epoch)) return;

    const b64 = this._captureFrame(HIRES_MAX_DIM, HIRES_QUALITY);
    if (!b64) {
      if (this._epochLive(epoch)) await this._sayAndWait("Camera not ready.");
      return;
    }

    let speak = "No currency detected.";
    try {
      const res = await fetch("/api/currency", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image_b64: b64 }),
      });
      if (!this._epochLive(epoch)) return;
      const data = await res.json();
      if (!this._epochLive(epoch)) return;
      speak = data.speak || speak;
      this._setModeCard("mode", "💵", "Currency", speak);
    } catch (err) {
      console.error(err);
      speak = "Could not reach the server.";
    }

    if (this._epochLive(epoch)) {
      try { await this._sayAndWait(speak); } catch (_) {}
    }
  }

  // -------- Mode: Objects (General Everyday Objects & Items) --------
  async _runObjectsMode(epoch) {
    this._setModeCard("mode", "📦", "Objects", "Scanning objects ahead…");
    await this._sleep(400);
    if (!this._epochLive(epoch)) return;

    const b64 = this._captureFrame(HIRES_MAX_DIM, HIRES_QUALITY);
    if (!b64) {
      if (this._epochLive(epoch)) await this._sayAndWait("Camera not ready.");
      return;
    }

    let speak = "No clear objects detected.";
    try {
      const res = await fetch("/api/objects", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image_b64: b64 }),
      });
      if (!this._epochLive(epoch)) return;
      const data = await res.json();
      if (!this._epochLive(epoch)) return;
      speak = data.speak || speak;
      this._setModeCard("mode", "📦", "Objects", speak);
    } catch (err) {
      console.error(err);
      speak = "Could not reach the server.";
    }

    if (this._epochLive(epoch)) {
      try { await this._sayAndWait(speak); } catch (_) {}
    }
  }

  // -------- Mode: Proximity (Instant Path Hazard Query) --------
  async _runProximityMode(epoch) {
    this._setModeCard("mode", "🛡️", "Collision watch", "Scanning walking path…");
    this.collisionMuted = false;
    await this._sleep(300);
    if (!this._epochLive(epoch)) return;

    const b64 = this._captureFrame(HIRES_MAX_DIM, HIRES_QUALITY);
    let speak = "Path is clear.";
    if (b64) {
      try {
        const res = await fetch("/api/proximity", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ image_b64: b64 }),
        });
        if (!this._epochLive(epoch)) return;
        const data = await res.json();
        if (!this._epochLive(epoch)) return;
        speak = data.speak || speak;
        if (data.collision_alert === "urgent") {
          this._setModeCard("urgent", "🛑", "STOP", speak);
          this._safeVibrate(COLLISION_HAPTICS.urgent);
        } else if (data.collision_alert === "warn") {
          this._setModeCard("warn", "⚠️", "Careful", speak);
          this._safeVibrate(COLLISION_HAPTICS.warn);
        } else {
          this._setModeCard("clear", "🛡️", "Collision watch", speak);
        }
      } catch (err) {
        console.error(err);
      }
    }

    if (this._epochLive(epoch)) {
      try { await this._sayAndWait(speak); } catch (_) {}
    }
  }

  // -------- Mode: Describe (Scene Description) --------
  async _runDescribeMode(epoch) {
    this._setModeCard("mode", "👁️", "Describe", "Looking…");
    await this._sleep(400);
    if (!this._epochLive(epoch)) return;

    const b64 = this._captureFrame(HIRES_MAX_DIM, HIRES_QUALITY);
    if (!b64) {
      if (this._epochLive(epoch)) await this._sayAndWait("Camera not ready.");
      return;
    }

    let speak = "I could not describe the scene.";
    try {
      const res = await fetch("/api/vlm", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image_b64: b64 }),
      });
      if (!this._epochLive(epoch)) return;
      const data = await res.json();
      if (!this._epochLive(epoch)) return;
      speak = data.speak || speak;
      this._setModeCard("mode", "👁️", "Describe", speak);
    } catch (err) {
      console.error(err);
      speak = "Could not reach the server.";
    }

    if (this._epochLive(epoch)) {
      try { await this._sayAndWait(speak); } catch (_) {}
    }
  }

  // -------- Mode: Read (100% Offline RapidOCR Auto 5s Sequential Loop) --------
  async _runReadMode(epoch) {
    this.state = "reading";
    this._setModeCard("mode", "📖", "Auto Text Reader", "Scanning text… Hold text in front of camera.");
    await this._sayAndWait("Text scanner active. Hold text in view.");
    if (!this._epochLive(epoch)) return;

    while (this.state === "reading" && this._epochLive(epoch)) {
      await this._scanAndReadText(epoch);
      if (this.state !== "reading" || !this._epochLive(epoch)) break;

      this._setModeCard("mode", "📖", "Auto Text Reader", "Next scan in 5 seconds…");
      await this._sleep(5000);
    }
  }

  async _scanAndReadText(epoch) {
    if (this.state !== "reading" || !this._epochLive(epoch)) return;
    const b64 = this._captureFrame(HIRES_MAX_DIM, HIRES_QUALITY);
    if (!b64) return;

    try {
      this._setModeCard("mode", "📖", "Scanning…", "Analyzing text in view…");
      const res = await fetch("/api/ocr", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image_b64: b64 }),
      });
      if (!this._epochLive(epoch) || this.state !== "reading") return;
      const data = await res.json();
      if (!this._epochLive(epoch) || this.state !== "reading") return;

      if (data.answered && data.speak && data.speak.trim()) {
        const text = data.speak.trim();
        if (text !== this._lastReadText) {
          this._lastReadText = text;
          this._setModeCard("mode", "📖", "Reading Text", text);
          if (this.transcript) this.transcript.textContent = text;
          await this._sayAndWait(text);
        } else {
          this._setModeCard("mode", "📖", "Auto Text Reader", "Same text in view.");
        }
      } else {
        this._setModeCard("mode", "📖", "Auto Text Reader", "Looking for text in view…");
      }
    } catch (err) {
      console.error("[Auto OCR error]", err);
    }
  }

  // -------- Mode: Question (STT -> Gemma reasoning) --------
  async _runQuestionMode(epoch) {
    this.state = "recording";
    this._setModeCard("listen", "🎤", "Question", "Ask your question now…");
    await this._sayAndWait("Ask your question.");
    if (!this._epochLive(epoch)) return;

    try {
      const audioB64 = await this._recordAudio(QUESTION_RECORD_MS);
      if (!this._epochLive(epoch)) return;

      this._setModeCard("mode", "🧠", "Question", "Transcribing…");
      const sttRes = await fetch("/api/stt", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ audio_b64: audioB64 }),
      });
      if (!this._epochLive(epoch)) return;
      const sttData = await sttRes.json();
      if (!this._epochLive(epoch)) return;

      const question = (sttData.text || "").trim();
      if (!question) {
        await this._sayAndWait("I could not hear a question. Try again.");
        return;
      }

      this._setModeCard("mode", "🧠", "Question", `“${question}” — thinking…`);
      const b64 = this._captureFrame(HIRES_MAX_DIM, HIRES_QUALITY);
      const res = await fetch("/api/vlm", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image_b64: b64, question }),
      });
      if (!this._epochLive(epoch)) return;
      const data = await res.json();
      if (!this._epochLive(epoch)) return;

      const speak = data.speak || "I don't know the answer.";
      this._setModeCard("mode", "🧠", "Question", speak);
      await this._sayAndWait(speak);
    } catch (err) {
      console.error(err);
      if (this._epochLive(epoch)) {
        await this._sayAndWait("Something went wrong recording your question.");
      }
    }
  }

  // -------- Audio recording for question mode --------
  async _recordAudio(durationMs) {
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
      let binary = "";
      for (let i = 0; i < bytes.length; i++) binary += String.fromCharCode(bytes[i]);
      return btoa(binary);
    } finally {
      this._resumeRecognizer();
    }
  }

  // -------- Frame capture --------
  _captureFrame(maxDim = FRAME_MAX_DIM, quality = JPEG_QUALITY) {
    if (this.video.readyState < 2) return null;
    const vw = this.video.videoWidth, vh = this.video.videoHeight;
    if (!vw || !vh) return null;
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

  _badge(text, listening, customClass = "") {
    if (!this.listenBadge) return;
    const textEl = this.listenBadge.querySelector(".badge-text") || this.listenBadge;
    textEl.textContent = text;
    this.listenBadge.classList.toggle("listening", !!listening);
    this.listenBadge.classList.toggle("speaking", customClass === "speaking");
  }

  _footer(text) {
    if (this.footerEl) this.footerEl.textContent = text;
  }

  _sleep(ms) { return new Promise((r) => setTimeout(r, ms)); }

  // -------- Audio & TTS Engine with Strict Microphone Isolation --------
  _acquireSpeechLock() {
    this._speechLockCount++;
    this._isSpeaking = true;
    clearTimeout(this._micResumeTimer);
    this._pauseRecognizer();
    this._badge("Speaking 🔊", false, "speaking");
  }

  _releaseSpeechLock() {
    this._speechLockCount = Math.max(0, this._speechLockCount - 1);
    if (this._speechLockCount === 0) {
      this._isSpeaking = false;
      this._badge("…", false);
      clearTimeout(this._micResumeTimer);
      // Acoustic grace period: allow speaker room echo and reverberation to decay
      // before reopening microphone speech recognition.
      this._micResumeTimer = setTimeout(() => {
        if (!this._isSpeaking && this.state !== "recording" && this._wakeWantsToRun) {
          this._resumeRecognizer();
        }
      }, 280);
    }
  }

  _interruptSpeech() {
    if (this.currentAudio) {
      try { this.currentAudio.pause(); } catch (_) {}
      this.currentAudio = null;
    }
    if (window.speechSynthesis && window.speechSynthesis.speaking) {
      try { window.speechSynthesis.cancel(); } catch (_) {}
    }
    this._activeUtterance = null;
    clearTimeout(this._micResumeTimer);
    this._speechLockCount = 0;
    this._isSpeaking = false;
    if (this.state !== "recording" && this._wakeWantsToRun) {
      this._resumeRecognizer();
    }
  }

  async _speak(text, { interrupt = true } = {}) {
    if (!text || !text.trim()) return;
    if (interrupt) this._interruptSpeech();
    this._acquireSpeechLock();

    try {
      const res = await fetch("/api/tts", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      if (!res.ok) throw new Error(`TTS server error: ${res.status}`);

      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const audio = new Audio(url);
      this.currentAudio = audio;

      let ended = false;
      const cleanup = () => {
        if (ended) return;
        ended = true;
        URL.revokeObjectURL(url);
        if (this.currentAudio === audio) this.currentAudio = null;
        this._releaseSpeechLock();
      };

      audio.onended = cleanup;
      audio.onerror = cleanup;
      audio.play().catch((err) => {
        console.warn("Audio play failed, falling back to browser TTS:", err);
        ended = true;
        URL.revokeObjectURL(url);
        if (this.currentAudio === audio) this.currentAudio = null;
        this._speakWithBrowserFallback(text);
      });

      const safetyMs = Math.max(8000, Math.round((text.length / 10) * 1000) + 5000);
      setTimeout(cleanup, safetyMs);
    } catch (err) {
      console.warn("TTS server error, using browser speech fallback:", err);
      this._speakWithBrowserFallback(text);
    }
  }

  async _sayAndWait(text) {
    if (!text || !text.trim()) return;
    this._interruptSpeech();
    this._acquireSpeechLock();

    try {
      const res = await fetch("/api/tts", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      if (!res.ok) throw new Error(`TTS server error: ${res.status}`);

      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const audio = new Audio(url);
      this.currentAudio = audio;

      await new Promise((resolve) => {
        let done = false;
        const cleanup = () => {
          if (done) return;
          done = true;
          URL.revokeObjectURL(url);
          if (this.currentAudio === audio) this.currentAudio = null;
          resolve();
        };
        audio.onended = cleanup;
        audio.onerror = cleanup;
        audio.play().catch(async (err) => {
          console.warn("Audio play failed, falling back to browser TTS:", err);
          done = true;
          URL.revokeObjectURL(url);
          if (this.currentAudio === audio) this.currentAudio = null;
          await this._sayAndWaitBrowserFallback(text);
          resolve();
        });
        const safetyMs = Math.max(12000, Math.round((text.length / 10) * 1000) + 8000);
        setTimeout(cleanup, safetyMs);
      });
    } catch (err) {
      console.warn("TTS server error, using browser speech fallback:", err);
      await this._sayAndWaitBrowserFallback(text);
    } finally {
      this._releaseSpeechLock();
    }
  }

  _speakWithBrowserFallback(text) {
    if (!window.speechSynthesis) {
      this._releaseSpeechLock();
      return;
    }
    try {
      const u = new SpeechSynthesisUtterance(text);
      u.lang = "en-US";
      u.rate = 1.0;
      u.onend = () => this._releaseSpeechLock();
      u.onerror = () => this._releaseSpeechLock();
      this._activeUtterance = u;
      window.speechSynthesis.speak(u);
    } catch (_) {
      this._releaseSpeechLock();
    }
  }

  _sayAndWaitBrowserFallback(text) {
    return new Promise((resolve) => {
      if (!window.speechSynthesis) {
        resolve();
        return;
      }
      try {
        const u = new SpeechSynthesisUtterance(text);
        u.lang = "en-US";
        u.rate = 1.0;
        let resolved = false;
        const done = () => {
          if (!resolved) {
            resolved = true;
            this._activeUtterance = null;
            resolve();
          }
        };
        u.onend = done;
        u.onerror = done;
        this._activeUtterance = u;
        window.speechSynthesis.speak(u);
        const safetyMs = Math.max(6000, Math.round((text.length / 10) * 1000) + 3000);
        setTimeout(done, safetyMs);
      } catch (_) {
        resolve();
      }
    });
  }
}

window.addEventListener("DOMContentLoaded", () => {
  const app = new SetuApp();
  window.__setu = app;
  app.start();
});
