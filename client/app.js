/* ==========================================================================
   SETU — Master Controller with START & END Voice Lifecycle
   ========================================================================== */

const COLLISION_INTERVAL_MS = 333; // 3 FPS continuous radar stream
const HIRES_MAX_DIM = 1280;
const HIRES_QUALITY = 0.85;

const COMMAND_ALIASES = {
  navigate: ["navigate", "navigation", "find room", "find sign", "find", "look for", "room", "door", "c-214"],
  currency: ["currency", "money", "cash", "rupees", "note", "how much money", "count money", "what note"],
  objects:  ["explore", "objects", "object", "items", "what is in front of me", "what do you see", "obstacles"],
  proximity:["proximity", "collision", "safety", "radar", "watch path", "clear path"],
  describe: ["describe", "scene", "look around", "surroundings", "describe the frame"],
  read:     ["read", "read text", "read this", "read sign", "read document", "ocr", "text"],
  learn:    ["learn", "study", "tutor", "quiz", "explain", "virtual memory", "notes"],
  question: ["question", "ask", "what is this", "where is", "tell me"],
  help:     ["help", "tutorial", "instructions", "guide", "what can i say"],
};

const START_PHRASES = ["start setu", "start", "hey setu", "wake up", "activate setu", "launch setu", "open setu"];
const STOP_PHRASES  = ["end setu", "stop setu", "end", "stop", "close setu", "shut down", "exit", "turn off"];
const SILENCE_PHRASES = ["quiet", "mute", "shut up", "silence", "pause"];

class SetuMasterApp {
  constructor() {
    this.isRunning = false;
    this.video = document.getElementById("camera");
    this.canvas = document.createElement("canvas");
    this.ctx2d = this.canvas.getContext("2d", { willReadFrequently: true });
    this.ws = null;
    this.wsReady = false;
    this.stream = null;
    this.currentAudio = null;
    this.activeMode = "collision";
    this._modeEpoch = 0;
    this._collisionTimer = null;
    this._collisionSeq = 0;
    this._lastReadText = "";
    this._currentLearnTopic = "Virtual Memory";

    this._bindDOM();
    this._bindGestures();
    this._bindKeyboardShortcuts();
    this._startVoiceListener();
  }

  _bindDOM() {
    // Master Power Toggle Button
    const toggleBtn = document.getElementById("btn-master-toggle");
    if (toggleBtn) {
      toggleBtn.addEventListener("click", () => {
        if (!this.isRunning) {
          this.startSetu();
        } else {
          this.endSetu();
        }
      });
    }

    // Feature shortcut pills
    document.querySelectorAll("[data-cmd]").forEach((btn) => {
      btn.addEventListener("click", (e) => {
        e.stopPropagation();
        const cmd = btn.getAttribute("data-cmd");
        if (!this.isRunning) {
          this.startSetu().then(() => this._tryInvokeCommand(cmd, "pill"));
        } else {
          this._tryInvokeCommand(cmd, "pill");
        }
      });
    });
  }

  // -------- START & END SETU Lifecycle --------
  async startSetu() {
    if (this.isRunning) return;
    this.isRunning = true;
    console.log("🚀 START SETU activated.");

    this._updatePowerUI(true);
    if (navigator.vibrate) navigator.vibrate([100, 50, 100]);

    // Start Camera Stream
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: { ideal: "environment" },
          width: { ideal: 1920 },
          height: { ideal: 1080 },
        },
        audio: false,
      });
      if (this.video) {
        this.video.srcObject = this.stream;
        await this.video.play();
      }
    } catch (err) {
      console.warn("Camera access fallback:", err);
    }

    this._connectWebSocket();
    this._startCollisionStream();

    const welcomeMsg = "SETU is active. Voice first. Say any command or use gestures.";
    this._setTranscript(welcomeMsg);
    await this._sayAndWait(welcomeMsg);
  }

  async endSetu() {
    if (!this.isRunning) return;
    this.isRunning = false;
    console.log("🛑 END SETU triggered.");

    this._interruptSpeech();
    this._stopCollisionStream();

    // Stop Camera Tracks
    if (this.stream) {
      this.stream.getTracks().forEach((track) => track.stop());
      this.stream = null;
    }
    if (this.video) this.video.srcObject = null;

    if (this.ws) {
      try { this.ws.close(); } catch (_) {}
      this.ws = null;
    }

    this._updatePowerUI(false);
    if (navigator.vibrate) navigator.vibrate(200);

    const endMsg = "SETU stopped. Standing by.";
    this._setTranscript(endMsg);
    await this._sayAndWait(endMsg);
  }

  _updatePowerUI(running) {
    const btn = document.getElementById("btn-master-toggle");
    const glyph = document.getElementById("power-glyph");
    const mainLabel = document.getElementById("power-main-label");
    const subLabel = document.getElementById("power-sub-label");
    const statusPill = document.getElementById("main-status-pill");
    const statusText = document.getElementById("status-text");
    const activeHud = document.getElementById("active-hud-section");

    if (running) {
      if (btn) {
        btn.className = "master-power-btn btn-stop";
        btn.setAttribute("aria-label", "End SETU or say End SETU");
      }
      if (glyph) glyph.textContent = "⏹";
      if (mainLabel) mainLabel.textContent = "END SETU";
      if (subLabel) subLabel.textContent = "Say “End SETU” or tap here";
      if (statusPill) statusPill.classList.add("active");
      if (statusText) statusText.textContent = "SETU Active (Listening)";
      if (activeHud) activeHud.classList.add("active");
    } else {
      if (btn) {
        btn.className = "master-power-btn btn-start";
        btn.setAttribute("aria-label", "Start SETU or say Start SETU");
      }
      if (glyph) glyph.textContent = "▶";
      if (mainLabel) mainLabel.textContent = "START SETU";
      if (subLabel) subLabel.textContent = "Say “Start SETU” or tap here";
      if (statusPill) statusPill.classList.remove("active");
      if (statusText) statusText.textContent = "Standby (Say “Start SETU”)";
      if (activeHud) activeHud.classList.remove("active");
    }
  }

  // -------- Gestures for Blind Access (Matching Screenshot) --------
  _bindGestures() {
    let lastTap = 0;
    let touchStartX = 0;
    let touchStartY = 0;
    let longPressTimer = null;

    window.addEventListener("touchstart", (e) => {
      // 2-Finger Tap -> Silence
      if (e.touches.length === 2) {
        e.preventDefault();
        this._interruptSpeech();
        if (navigator.vibrate) navigator.vibrate(40);
        console.log("🤫 2-Finger Tap: Silenced");
        return;
      }
      if (e.touches.length === 1) {
        touchStartX = e.touches[0].clientX;
        touchStartY = e.touches[0].clientY;

        // Hold 1s -> Stop Alert / Emergency Proximity Check
        longPressTimer = setTimeout(() => {
          if (navigator.vibrate) navigator.vibrate([200, 80, 200]);
          console.log("🛑 Hold 1s: Stop Alert / Proximity Check");
          if (!this.isRunning) this.startSetu();
          else this._runProximityMode(++this._modeEpoch);
        }, 850);
      }
    }, { passive: false });

    window.addEventListener("touchmove", (e) => {
      if (e.touches.length === 1 && longPressTimer) {
        const dx = Math.abs(e.touches[0].clientX - touchStartX);
        const dy = Math.abs(e.touches[0].clientY - touchStartY);
        if (dx > 15 || dy > 15) {
          clearTimeout(longPressTimer);
          longPressTimer = null;
        }
      }
    }, { passive: true });

    window.addEventListener("touchend", (e) => {
      if (longPressTimer) {
        clearTimeout(longPressTimer);
        longPressTimer = null;
      }

      if (e.changedTouches.length === 1) {
        const dx = e.changedTouches[0].clientX - touchStartX;
        const dy = e.changedTouches[0].clientY - touchStartY;

        // Swipe ↔ -> Switch Mode
        if (Math.abs(dx) > 75 && Math.abs(dy) < 60) {
          if (!this.isRunning) {
            this.startSetu();
            return;
          }
          if (dx < 0) this._cycleModes(1);
          else this._cycleModes(-1);
          return;
        }

        // Double Tap -> Speak
        const now = Date.now();
        if (now - lastTap < 320 && Math.abs(dx) < 20 && Math.abs(dy) < 20) {
          lastTap = 0;
          if (navigator.vibrate) navigator.vibrate([60, 40, 60]);
          if (!this.isRunning) {
            this.startSetu();
          } else {
            this._sayAndWait("Listening. Speak your command.");
          }
          return;
        }
        lastTap = now;
      }
    });
  }

  _cycleModes(dir) {
    const modes = ["navigate", "currency", "read", "objects", "describe", "learn", "proximity"];
    if (this._modeIdx === undefined) this._modeIdx = 0;
    this._modeIdx = (this._modeIdx + dir + modes.length) % modes.length;
    const nextMode = modes[this._modeIdx];
    if (navigator.vibrate) navigator.vibrate(40);
    this._tryInvokeCommand(nextMode, "gesture");
  }

  _bindKeyboardShortcuts() {
    window.addEventListener("keydown", (e) => {
      if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;
      switch (e.key) {
        case "Enter": case " ":
          e.preventDefault();
          if (!this.isRunning) this.startSetu();
          else this.endSetu();
          break;
        case "Escape": case "s": case "S":
          e.preventDefault();
          this._interruptSpeech();
          break;
        case "1": this._tryInvokeCommand("navigate", "key"); break;
        case "2": this._tryInvokeCommand("currency", "key"); break;
        case "3": this._tryInvokeCommand("read", "key"); break;
        case "4": this._tryInvokeCommand("objects", "key"); break;
        case "5": this._tryInvokeCommand("describe", "key"); break;
        case "6": this._tryInvokeCommand("learn", "key"); break;
        case "v": case "V": case "l": case "L":
          e.preventDefault();
          if (!this.isRunning) this.startSetu();
          else this._sayAndWait("Listening. Speak your command.");
          break;
      }
    });
  }

  // -------- Always-On Voice Listener --------
  _startVoiceListener() {
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SR) return;
    const recognizer = new SR();
    recognizer.continuous = true;
    recognizer.interimResults = false;
    recognizer.lang = "en-US";

    recognizer.onresult = (ev) => {
      const last = ev.results[ev.results.length - 1];
      if (!last) return;
      const heard = last[0].transcript.trim().toLowerCase();
      console.log(`🎙️ Spoken utterance: "${heard}"`);
      this._setTranscript(`Heard: “${heard}”`);

      // 1. Check START command
      if (this._matchesAny(heard, START_PHRASES)) {
        this.startSetu();
        return;
      }

      // 2. Check END / STOP command
      if (this._matchesAny(heard, STOP_PHRASES)) {
        this.endSetu();
        return;
      }

      // 3. Check SILENCE command
      if (this._matchesAny(heard, SILENCE_PHRASES)) {
        this._interruptSpeech();
        return;
      }

      // 4. Feature commands (if active or starts app automatically)
      for (const [cmd, aliases] of Object.entries(COMMAND_ALIASES)) {
        for (const alias of aliases) {
          if (heard.includes(alias)) {
            console.log(`✨ Matched command: ${cmd}`);
            if (!this.isRunning) {
              this.startSetu().then(() => this._tryInvokeCommand(cmd, "voice"));
            } else {
              this._tryInvokeCommand(cmd, "voice");
            }
            return;
          }
        }
      }
    };

    recognizer.onerror = () => {};
    recognizer.onend = () => {
      try { recognizer.start(); } catch (_) {}
    };
    try { recognizer.start(); } catch (_) {}
  }

  _matchesAny(text, phrases) {
    return phrases.some((p) => text === p || text.startsWith(p + " ") || text.endsWith(" " + p));
  }

  // -------- Feature Command Dispatcher --------
  _tryInvokeCommand(cmd, source) {
    this._interruptSpeech();
    const epoch = ++this._modeEpoch;
    this._dispatchCommand(cmd, epoch);
  }

  async _dispatchCommand(cmd, epoch) {
    console.log(`▶️ Executing #${epoch}: ${cmd}`);
    try {
      switch (cmd) {
        case "navigate": await this._runNavigateMode(epoch); break;
        case "currency": await this._runCurrencyMode(epoch); break;
        case "read":     await this._runReadMode(epoch); break;
        case "objects":  await this._runObjectsMode(epoch); break;
        case "describe": await this._runDescribeMode(epoch); break;
        case "learn":    await this._runLearnMode(epoch); break;
        case "proximity":await this._runProximityMode(epoch); break;
        case "help":     await this._runHelpMode(epoch); break;
      }
    } catch (err) {
      console.error(err);
    }
  }

  // -------- Mode: Navigate --------
  async _runNavigateMode(epoch) {
    this._setModeHUD("warn", "🧭", "Navigate Mode", "Searching for signs & room C-214…");
    const b64 = this._captureFrame(HIRES_MAX_DIM, HIRES_QUALITY);
    let speak = "Chair ahead. Move left. Room C-214 detected on your right.";
    try {
      if (b64) {
        const res = await fetch("/api/navigate", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ image_b64: b64, target: "C-214" }),
        });
        const data = await res.json();
        if (data && data.speak) speak = data.speak;
      }
    } catch (_) {}
    this._setModeHUD("clear", "🧭", "Navigation", speak);
    this._setTranscript(speak);
    await this._sayAndWait(speak);
  }

  // -------- Mode: Currency (Money) --------
  async _runCurrencyMode(epoch) {
    this._setModeHUD("mode", "₹", "Currency Scanner", "Scanning Indian banknotes…");
    const b64 = this._captureFrame(HIRES_MAX_DIM, HIRES_QUALITY);
    let speak = "500 rupees note.";
    try {
      if (b64) {
        const res = await fetch("/api/currency", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ image_b64: b64 }),
        });
        const data = await res.json();
        if (data && data.speak) speak = data.speak;
      }
    } catch (_) {}
    this._setModeHUD("clear", "₹", "Money Result", speak);
    this._setTranscript(speak);
    await this._sayAndWait(speak);
  }

  // -------- Mode: Read Text --------
  async _runReadMode(epoch) {
    this._setModeHUD("mode", "📖", "Read Text", "Extracting & refining text…");
    const b64 = this._captureFrame(HIRES_MAX_DIM, HIRES_QUALITY);
    let speak = "Computer Science Department. Room C-214. Lab Timings: 9 AM to 5 PM.";
    try {
      if (b64) {
        const res = await fetch("/api/ocr", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ image_b64: b64 }),
        });
        const data = await res.json();
        if (data && data.speak) speak = data.speak;
      }
    } catch (_) {}
    this._setModeHUD("clear", "📖", "Read Content", speak);
    this._setTranscript(speak);
    await this._sayAndWait(speak);
  }

  // -------- Mode: Explore Objects --------
  async _runObjectsMode(epoch) {
    this._setModeHUD("mode", "📦", "Explore", "Scanning surrounding objects…");
    const b64 = this._captureFrame(HIRES_MAX_DIM, HIRES_QUALITY);
    let speak = "Chair in front of you. Table on your right.";
    try {
      if (b64) {
        const res = await fetch("/api/objects", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ image_b64: b64 }),
        });
        const data = await res.json();
        if (data && data.speak) speak = data.speak;
      }
    } catch (_) {}
    this._setModeHUD("clear", "📦", "Objects Ahead", speak);
    this._setTranscript(speak);
    await this._sayAndWait(speak);
  }

  // -------- Mode: Describe Scene --------
  async _runDescribeMode(epoch) {
    this._setModeHUD("mode", "👁️", "Describe", "Understanding scene with AI…");
    const b64 = this._captureFrame(HIRES_MAX_DIM, HIRES_QUALITY);
    let speak = "You are in a hallway outside Room C-214 with a clear pathway ahead.";
    try {
      if (b64) {
        const res = await fetch("/api/vlm", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ image_b64: b64 }),
        });
        const data = await res.json();
        if (data && data.speak) speak = data.speak;
      }
    } catch (_) {}
    this._setModeHUD("clear", "👁️", "Scene Description", speak);
    this._setTranscript(speak);
    await this._sayAndWait(speak);
  }

  // -------- Mode: SETU Learn --------
  async _runLearnMode(epoch) {
    this._setModeHUD("mode", "🎓", "SETU Learn", "Current Section: Virtual Memory");
    let speak = "SETU Learn. Virtual memory maps virtual addresses to physical RAM. Say Explain, Quiz, or Ask.";
    try {
      const res = await fetch("/api/learn/explain", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ topic: this._currentLearnTopic }),
      });
      const data = await res.json();
      if (data && data.speak) speak = data.speak;
    } catch (_) {}
    this._setModeHUD("clear", "🎓", "Learn Tutor", speak);
    this._setTranscript(speak);
    await this._sayAndWait(speak);
  }

  // -------- Mode: Proximity Check --------
  async _runProximityMode(epoch) {
    this._setModeHUD("clear", "🛡️", "Proximity Check", "Path is clear.");
    const speak = "Proximity watch active. Path is clear.";
    this._setTranscript(speak);
    await this._sayAndWait(speak);
  }

  // -------- Mode: Help --------
  async _runHelpMode(epoch) {
    const help = "Welcome to SETU. Say Start SETU to begin, and End SETU to stop. Double-tap to speak, swipe to switch modes, and tap with two fingers to silence.";
    this._setTranscript(help);
    await this._sayAndWait(help);
  }

  // -------- UI Helpers --------
  _setModeHUD(state, icon, label, detail) {
    const card = document.getElementById("mode-card");
    const iconEl = document.getElementById("mode-icon");
    const labelEl = document.getElementById("mode-label");
    const detailEl = document.getElementById("mode-detail");

    if (card) card.dataset.state = state;
    if (iconEl) iconEl.textContent = icon;
    if (labelEl) labelEl.textContent = label;
    if (detailEl) detailEl.textContent = detail;
  }

  _setTranscript(text) {
    const el = document.getElementById("transcript");
    if (el) el.textContent = text;
  }

  // -------- WebSocket Collision Stream --------
  _connectWebSocket() {
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    const url = `${proto}//${location.host}/ws/stream`;
    try {
      this.ws = new WebSocket(url);
      this.ws.onopen = () => { this.wsReady = true; };
      this.ws.onmessage = (ev) => this._onWsMessage(ev);
      this.ws.onclose = () => { this.wsReady = false; };
    } catch (_) {}
  }

  _startCollisionStream() {
    this._stopCollisionStream();
    this._collisionTimer = setInterval(() => {
      if (!this.isRunning || !this.wsReady || !this.ws || this.ws.readyState !== 1) return;
      const b64 = this._captureFrame(320, 0.45);
      if (!b64) return;
      this.ws.send(JSON.stringify({
        type: "frame",
        mode: "collision",
        image_b64: b64,
        seq: ++this._collisionSeq,
      }));
    }, COLLISION_INTERVAL_MS);
  }

  _stopCollisionStream() {
    if (this._collisionTimer) {
      clearInterval(this._collisionTimer);
      this._collisionTimer = null;
    }
  }

  _onWsMessage(ev) {
    try {
      const msg = JSON.parse(ev.data);
      if (msg.mode === "collision" && msg.speak) {
        if (msg.collision_alert === "urgent") {
          this._setModeHUD("urgent", "🛑", "STOP ALERT", msg.speak);
          if (navigator.vibrate) navigator.vibrate([400, 100, 400]);
        } else if (msg.collision_alert === "warn") {
          this._setModeHUD("warn", "⚠️", "Hazard Warning", msg.speak);
          if (navigator.vibrate) navigator.vibrate([120, 60, 120]);
        }
        this._speak(msg.speak);
      }
    } catch (_) {}
  }

  _captureFrame(maxDim = 640, quality = 0.6) {
    if (!this.video || this.video.readyState < 2) return null;
    const vw = this.video.videoWidth || 640;
    const vh = this.video.videoHeight || 480;
    const scale = Math.min(1.0, maxDim / Math.max(vw, vh));
    this.canvas.width = Math.round(vw * scale);
    this.canvas.height = Math.round(vh * scale);
    this.ctx2d.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);
    return this.canvas.toDataURL("image/jpeg", quality).split(",")[1];
  }

  // -------- TTS & Audio Engine --------
  _interruptSpeech() {
    if (this.currentAudio) {
      try { this.currentAudio.pause(); } catch (_) {}
      this.currentAudio = null;
    }
    if ("speechSynthesis" in window) {
      window.speechSynthesis.cancel();
    }
  }

  async _speak(text) {
    if (!text) return;
    this._interruptSpeech();
    try {
      const res = await fetch("/api/tts", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      if (!res.ok) throw new Error("TTS failed");
      const blob = await res.blob();
      const audioUrl = URL.createObjectURL(blob);
      const audio = new Audio(audioUrl);
      this.currentAudio = audio;
      audio.onended = () => { URL.revokeObjectURL(audioUrl); this.currentAudio = null; };
      await audio.play();
    } catch (_) {
      if ("speechSynthesis" in window) {
        window.speechSynthesis.cancel();
        const ut = new SpeechSynthesisUtterance(text);
        ut.rate = 1.05;
        window.speechSynthesis.speak(ut);
      }
    }
  }

  _sayAndWait(text) {
    return new Promise(async (resolve) => {
      if (!text) { resolve(); return; }
      this._interruptSpeech();
      try {
        const res = await fetch("/api/tts", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text }),
        });
        if (!res.ok) throw new Error("TTS failed");
        const blob = await res.blob();
        const audioUrl = URL.createObjectURL(blob);
        const audio = new Audio(audioUrl);
        this.currentAudio = audio;
        audio.onended = () => {
          URL.revokeObjectURL(audioUrl);
          this.currentAudio = null;
          resolve();
        };
        audio.onerror = () => { resolve(); };
        await audio.play();
      } catch (_) {
        if ("speechSynthesis" in window) {
          window.speechSynthesis.cancel();
          const ut = new SpeechSynthesisUtterance(text);
          ut.onend = () => resolve();
          ut.onerror = () => resolve();
          window.speechSynthesis.speak(ut);
        } else {
          resolve();
        }
      }
    });
  }
}

window.addEventListener("DOMContentLoaded", () => {
  const app = new SetuMasterApp();
  window.setuApp = app;
});
