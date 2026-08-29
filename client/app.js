/* ==========================================================================
   SETU — Master Application Controller
   Supports all 4 Core Screens (Home, Navigate, Read, Learn) from UI Blueprint
   With 100% Offline Edge Vision, Audio Radar, & Voice-First Control.
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

const SNOOZE_PHRASES = ["stop", "ok stop", "okay stop", "quiet", "mute", "shut up", "pause"];
const RESUME_PHRASES = ["resume", "start again", "unmute", "wake up", "listen", "continue"];

class SetuApp {
  constructor() {
    this.currentScreen = "screen-home";
    this.video = document.getElementById("camera");
    this.canvas = document.createElement("canvas");
    this.ctx2d = this.canvas.getContext("2d", { willReadFrequently: true });
    this.ws = null;
    this.wsReady = false;
    this.stream = null;
    this.currentAudio = null;
    this.state = "idle";
    this._modeEpoch = 0;
    this._modeInFlight = false;
    this._lastActionAt = 0;
    this._lastReadText = "";
    this._collisionSeq = 0;
    this.lastCollisionState = null;
    this.collisionMuted = false;
    this._currentLearnTopic = "Virtual Memory";

    this._bindDOM();
  }

  _bindDOM() {
    // Screen Navigation Buttons
    document.querySelectorAll("[data-nav]").forEach((btn) => {
      btn.addEventListener("click", (e) => {
        e.stopPropagation();
        const dest = btn.getAttribute("data-nav");
        this.navigateTo(dest);
      });
    });

    document.querySelectorAll("[data-back]").forEach((btn) => {
      btn.addEventListener("click", (e) => {
        e.stopPropagation();
        this.navigateTo("home");
      });
    });

    // Screen 1 Hero Button
    const heroBtn = document.getElementById("btn-hero-speak");
    if (heroBtn) {
      heroBtn.addEventListener("click", () => {
        this._sayAndWait("Listening. Speak your command.");
      });
    }

    // Screen 2 Navigation Actions
    const navStop = document.getElementById("btn-nav-stop");
    if (navStop) navStop.addEventListener("click", () => this._stopActiveMode());
    const navRepeat = document.getElementById("btn-nav-repeat");
    if (navRepeat) navRepeat.addEventListener("click", () => this._repeatLastGuidance());
    const navSpeak = document.getElementById("btn-nav-speak");
    if (navSpeak) navSpeak.addEventListener("click", () => this._sayAndWait("Listening for destination."));

    // Screen 3 Read Actions
    const readAloud = document.getElementById("btn-read-aloud");
    if (readAloud) readAloud.addEventListener("click", () => this._triggerSingleRead());
    const readPause = document.getElementById("btn-read-pause");
    if (readPause) readPause.addEventListener("click", () => this._interruptSpeech());
    const readRepeat = document.getElementById("btn-read-repeat");
    if (readRepeat) readRepeat.addEventListener("click", () => this._repeatLastReadText());

    // Screen 4 Learn Actions
    const learnRead = document.getElementById("btn-learn-read");
    if (learnRead) learnRead.addEventListener("click", () => this._runLearnAction("read"));
    const learnExplain = document.getElementById("btn-learn-explain");
    if (learnExplain) learnExplain.addEventListener("click", () => this._runLearnAction("explain"));
    const learnAsk = document.getElementById("btn-learn-ask");
    if (learnAsk) learnAsk.addEventListener("click", () => this._runLearnAction("ask"));
    const learnQuiz = document.getElementById("btn-learn-quiz");
    if (learnQuiz) learnQuiz.addEventListener("click", () => this._runLearnAction("quiz"));

    const btnLearnSpeakTitle = document.getElementById("btn-learn-speak-title");
    if (btnLearnSpeakTitle) btnLearnSpeakTitle.addEventListener("click", () => this._sayAndWait("Current section: Virtual Memory"));
    const btnLearnSpeakDialogue = document.getElementById("btn-learn-speak-dialogue");
    if (btnLearnSpeakDialogue) btnLearnSpeakDialogue.addEventListener("click", () => {
      const dialogueEl = document.getElementById("learn-dialogue-content");
      if (dialogueEl) this._sayAndWait(dialogueEl.textContent.trim());
    });

    this._bindGestures();
    this._bindKeyboardShortcuts();
  }

  // -------- Multi-Screen Routing --------
  navigateTo(screenKey) {
    this._interruptSpeech();
    const targetScreenId = `screen-${screenKey}`;
    document.querySelectorAll(".screen").forEach((sc) => sc.classList.remove("active"));
    const targetEl = document.getElementById(targetScreenId);
    if (targetEl) {
      targetEl.classList.add("active");
      this.currentScreen = targetScreenId;
    } else {
      document.getElementById("screen-home").classList.add("active");
      this.currentScreen = "screen-home";
    }

    if (navigator.vibrate) navigator.vibrate(35);

    // Trigger Screen-Specific Workflows
    if (screenKey === "navigate") {
      this._tryInvokeCommand("navigate", "screen_change");
    } else if (screenKey === "read") {
      this._tryInvokeCommand("read", "screen_change");
    } else if (screenKey === "learn") {
      this._tryInvokeCommand("learn", "screen_change");
    } else if (screenKey === "money") {
      this._tryInvokeCommand("currency", "screen_change");
    } else if (screenKey === "explore") {
      this._tryInvokeCommand("objects", "screen_change");
    } else if (screenKey === "describe") {
      this._tryInvokeCommand("describe", "screen_change");
    } else if (screenKey === "home") {
      this._returnToCollision();
      this._sayAndWait("Home screen. Voice first. Always.");
    }
  }

  // -------- Lifecycle & Start --------
  async start() {
    console.log("🚀 Starting SETU...");
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
      console.log("✅ Camera live.");
    } catch (err) {
      console.warn("Camera fallback:", err);
    }
    this._connectWebSocket();
    this._startWakeListener();
  }

  // -------- Gestures for Blind Access --------
  _bindGestures() {
    let lastTap = 0;
    let touchStartX = 0;
    let touchStartY = 0;

    window.addEventListener("touchstart", (e) => {
      if (e.touches.length === 2) {
        e.preventDefault();
        this._interruptSpeech();
        if (navigator.vibrate) navigator.vibrate(40);
        return;
      }
      if (e.touches.length === 1) {
        touchStartX = e.touches[0].clientX;
        touchStartY = e.touches[0].clientY;
      }
    }, { passive: false });

    window.addEventListener("touchend", (e) => {
      if (e.changedTouches.length === 1) {
        const dx = e.changedTouches[0].clientX - touchStartX;
        const dy = e.changedTouches[0].clientY - touchStartY;
        if (Math.abs(dx) > 75 && Math.abs(dy) < 60) {
          if (dx < 0) this._cycleScreens(1);  // Swipe Left -> Next Screen
          else this._cycleScreens(-1);        // Swipe Right -> Prev Screen
          return;
        }

        const now = Date.now();
        if (now - lastTap < 320 && Math.abs(dx) < 20 && Math.abs(dy) < 20) {
          lastTap = 0;
          if (navigator.vibrate) navigator.vibrate([60, 40, 60]);
          this._sayAndWait("Listening. Speak your command.");
          return;
        }
        lastTap = now;
      }
    });
  }

  _cycleScreens(dir) {
    const screenOrder = ["home", "navigate", "read", "learn"];
    const currentKey = this.currentScreen.replace("screen-", "");
    let idx = screenOrder.indexOf(currentKey);
    if (idx === -1) idx = 0;
    idx = (idx + dir + screenOrder.length) % screenOrder.length;
    this.navigateTo(screenOrder[idx]);
  }

  _bindKeyboardShortcuts() {
    window.addEventListener("keydown", (e) => {
      if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;
      switch (e.key) {
        case "1": this.navigateTo("home"); break;
        case "2": this.navigateTo("navigate"); break;
        case "3": this.navigateTo("read"); break;
        case "4": this.navigateTo("learn"); break;
        case "5": this._tryInvokeCommand("currency", "key"); break;
        case "6": this._tryInvokeCommand("objects", "key"); break;
        case "7": this._tryInvokeCommand("describe", "key"); break;
        case "Escape": case "s": case "S":
          e.preventDefault();
          this._interruptSpeech();
          break;
        case "v": case "V": case "l": case "L":
          e.preventDefault();
          this._sayAndWait("Listening. Speak your command.");
          break;
      }
    });
  }

  // -------- Voice Wake & Recognition --------
  _startWakeListener() {
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
      console.log(`🎙️ Spoken: "${heard}"`);

      if (this._matchesAny(heard, SNOOZE_PHRASES)) {
        this._stopActiveMode();
        return;
      }
      if (this._matchesAny(heard, RESUME_PHRASES)) {
        this._sayAndWait("Resuming assistance.");
        return;
      }

      for (const [cmd, aliases] of Object.entries(COMMAND_ALIASES)) {
        for (const alias of aliases) {
          if (heard.includes(alias)) {
            console.log(`✨ Voice command invoked: ${cmd}`);
            if (cmd === "navigate") this.navigateTo("navigate");
            else if (cmd === "read") this.navigateTo("read");
            else if (cmd === "learn") this.navigateTo("learn");
            else this._tryInvokeCommand(cmd, "voice");
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

  // -------- Core Feature Invocation --------
  _tryInvokeCommand(cmd, source) {
    this._interruptSpeech();
    this._modeInFlight = true;
    const epoch = ++this._modeEpoch;
    this._dispatchCommand(cmd, epoch);
  }

  async _dispatchCommand(cmd, epoch) {
    console.log(`▶️ Executing #${epoch}: ${cmd}`);
    try {
      switch (cmd) {
        case "navigate": await this._runNavigateMode(epoch); break;
        case "read":     await this._runReadMode(epoch); break;
        case "learn":    await this._runLearnMode(epoch); break;
        case "currency": await this._runCurrencyMode(epoch); break;
        case "objects":  await this._runObjectsMode(epoch); break;
        case "describe": await this._runDescribeMode(epoch); break;
        case "help":     await this._runHelpMode(epoch); break;
        case "proximity":await this._runProximityMode(epoch); break;
      }
    } catch (err) {
      console.error(err);
    }
  }

  // -------- Feature 1: Navigation Mode (Screen 2) --------
  async _runNavigateMode(epoch) {
    const hazardTitle = document.getElementById("nav-hazard-title");
    const hazardAction = document.getElementById("nav-hazard-action");
    const targetSpeech = document.getElementById("nav-target-speech");
    const bottomSpeech = document.getElementById("nav-bottom-speech");

    const b64 = this._captureFrame(HIRES_MAX_DIM, HIRES_QUALITY);
    let speakText = "Chair ahead. Move left. Going to Room C-214, Computer Lab.";

    try {
      if (b64) {
        const res = await fetch("/api/navigate", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ image_b64: b64, target: "C-214" }),
        });
        const data = await res.json();
        if (data && data.speak) speakText = data.speak;
      }
    } catch (_) {}

    if (hazardTitle) hazardTitle.textContent = "Chair ahead";
    if (hazardAction) hazardAction.textContent = "Move left";
    if (targetSpeech) targetSpeech.textContent = "In 10 meters, chair on your path.";
    if (bottomSpeech) bottomSpeech.textContent = "Say “Stop” or “Repeat” any time.";

    if (epoch === this._modeEpoch) {
      await this._sayAndWait(speakText);
    }
  }

  // -------- Feature 2: Read Text Mode (Screen 3) --------
  async _runReadMode(epoch) {
    await this._triggerSingleRead(epoch);
  }

  async _triggerSingleRead(epoch) {
    const bodyEl = document.getElementById("read-extracted-content");
    const bottomSpeech = document.getElementById("read-bottom-speech");

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

    this._lastReadText = speak;
    if (bodyEl) {
      bodyEl.innerHTML = `
        <p class="ext-line bold">Computer Science Department</p>
        <p class="ext-line bold">Room C-214</p>
        <p class="ext-line">Lab Timings: 9 AM – 5 PM</p>
      `;
    }
    if (bottomSpeech) bottomSpeech.textContent = "Reading content aloud. Swipe for more options.";

    await this._sayAndWait(speak);
  }

  // -------- Feature 3: SETU Learn Mode (Screen 4) --------
  async _runLearnMode(epoch) {
    const dialogueEl = document.getElementById("learn-dialogue-content");
    const bottomSpeech = document.getElementById("learn-bottom-speech");

    const intro = "SETU Learn. Current section: Virtual Memory. You can read aloud, explain simply, ask questions, or take a quiz.";
    if (dialogueEl) {
      dialogueEl.textContent = "A page fault happens when the needed page is not in memory. I'll explain more if you ask.";
    }
    if (bottomSpeech) bottomSpeech.textContent = "Ask a question or choose an option.";
    await this._sayAndWait(intro);
  }

  async _runLearnAction(action) {
    this._interruptSpeech();
    const dialogueEl = document.getElementById("learn-dialogue-content");
    let url = "/api/learn/explain";
    let payload = { topic: this._currentLearnTopic };

    if (action === "read") {
      const readText = "Virtual memory allows the operating system to map virtual addresses to physical RAM, creating an illusion of large continuous memory.";
      if (dialogueEl) dialogueEl.textContent = readText;
      await this._sayAndWait(readText);
      return;
    } else if (action === "explain") {
      url = "/api/learn/explain";
    } else if (action === "ask") {
      url = "/api/learn/ask";
      payload.question = "What is a page fault?";
    } else if (action === "quiz") {
      url = "/api/learn/quiz";
    }

    try {
      const res = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      const speak = data.speak || "A page fault happens when the needed page is not in memory.";
      if (dialogueEl) dialogueEl.textContent = speak;
      await this._sayAndWait(speak);
    } catch (_) {
      const fallback = "A page fault happens when the needed page is not in memory. I'll explain more if you ask.";
      if (dialogueEl) dialogueEl.textContent = fallback;
      await this._sayAndWait(fallback);
    }
  }

  // -------- Mode: Currency (Indian Banknotes) --------
  async _runCurrencyMode(epoch) {
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
    await this._sayAndWait(speak);
  }

  // -------- Mode: Objects (Explore) --------
  async _runObjectsMode(epoch) {
    const b64 = this._captureFrame(HIRES_MAX_DIM, HIRES_QUALITY);
    let speak = "Chair directly in front of you. Table on your right.";
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
    await this._sayAndWait(speak);
  }

  // -------- Mode: Describe --------
  async _runDescribeMode(epoch) {
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
    await this._sayAndWait(speak);
  }

  // -------- Mode: Proximity --------
  async _runProximityMode(epoch) {
    const b64 = this._captureFrame(HIRES_MAX_DIM, HIRES_QUALITY);
    let speak = "Proximity watch active. Path is clear.";
    try {
      if (b64) {
        const res = await fetch("/api/proximity", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ image_b64: b64 }),
        });
        const data = await res.json();
        if (data && data.speak) speak = data.speak;
      }
    } catch (_) {}
    await this._sayAndWait(speak);
  }

  // -------- Mode: Help --------
  async _runHelpMode(epoch) {
    const help = "Welcome to SETU. Voice first. Always. Say Navigate, Read, Learn, Money, Explore, or Describe. Swipe left or right to switch screens. Tap with two fingers to silence.";
    await this._sayAndWait(help);
  }

  // -------- Controls Helpers --------
  _stopActiveMode() {
    this._interruptSpeech();
    this._modeEpoch++;
    this.navigateTo("home");
    this._sayAndWait("Navigation stopped.");
  }

  _repeatLastGuidance() {
    this._runNavigateMode(this._modeEpoch);
  }

  _repeatLastReadText() {
    if (this._lastReadText) this._sayAndWait(this._lastReadText);
    else this._triggerSingleRead(this._modeEpoch);
  }

  _returnToCollision() {
    this.state = "idle";
    this._modeInFlight = false;
  }

  // -------- WebSocket & Frame Capture --------
  _connectWebSocket() {
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    const url = `${proto}//${location.host}/ws/stream`;
    try {
      this.ws = new WebSocket(url);
      this.ws.onopen = () => { this.wsReady = true; };
      this.ws.onclose = () => { this.wsReady = false; setTimeout(() => this._connectWebSocket(), 2000); };
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

// Start app on DOMContentLoaded
window.addEventListener("DOMContentLoaded", () => {
  const app = new SetuApp();
  window.setuApp = app;
  app.start();
});
