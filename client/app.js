/* ---------------------------------------------------------------------
 * SETU voice-first client
 *
 * Default/idle state: nothing is running. The app waits for a voice
 * command or a tapped command word. Every mode — including proximity
 * detection — is a ONE-SHOT scan: capture a frame, ask the server,
 * speak the result, return to idle. Nothing streams continuously.
 *
 * (Earlier version ran proximity/collision detection continuously in
 * the background at ~3fps as the app's default state. That's been
 * deliberately removed — it now behaves exactly like currency /
 * describe / read: invoked on demand via "detect", scans, speaks once,
 * done. The trade-off: a hazard that appears BETWEEN scans, or while
 * another mode is running, is no longer caught automatically — there's
 * no background watcher left to catch it. If you want that safety net
 * back, the old continuous-stream approach is worth revisiting instead
 * of purely on-demand scanning.)
 *
 * Voice commands: browser's Web Speech API listens continuously for a
 * bare command word ("currency", "describe", "read", "question",
 * "detect"). On match, invokes the corresponding one-shot mode, speaks
 * the answer, and returns to idle automatically.
 *
 * "question" is the only command that then records the user's spoken
 * question via MediaRecorder, ships the audio to /api/stt for offline
 * whisper transcription, and hands the transcript to Gemma along with
 * the current camera frame.
 *
 * The Web Speech API IS used here for command-word matching — but only
 * for that. It never handles the user's actual question content, which
 * always goes through the local whisper server. That preserves the
 * "user data never leaves the machine" story for the load-bearing part.
 * ------------------------------------------------------------------- */

const FRAME_MAX_DIM = 640;          // proximity scan — doesn't need fine detail, keep it fast
const JPEG_QUALITY = 0.7;
const HIRES_MAX_DIM = 1600;         // read / currency / describe — text/detail needs resolution
const HIRES_QUALITY = 0.9;
const QUESTION_RECORD_MS = 5000;    // how long we listen after "question"

// Distinct haptic pulses for the two proximity-alert severities.
const PROXIMITY_HAPTICS = {
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
// word of the sentence. No "hey setu" prefix — that got misheard
// consistently in noisy rooms.
//   "currency" / "money" / "cash"           -> currency mode
//   "describe" / "scene" / "look"           -> scene description
//   "read" / "read text" / "text"           -> OCR + summarize
//   "question" / "ask" / "ask a question"   -> record & ask
//   "detect" / "proximity" / "scan"         -> one-shot proximity/collision scan
const COMMAND_ALIASES = {
  currency:  ["currency", "money", "cash", "notes"],
  describe:  ["describe", "scene", "what do you see", "look"],
  read:      ["read text", "read this", "read", "text", "ocr"],
  question:  ["ask a question", "question", "ask"],
  detect:    ["detect", "proximity", "scan", "check surroundings"],
};


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

    this.stream = null;
    this.track = null;

    // The state machine has exactly one active state at a time.
    //   "idle"        — nothing running; waiting for a command
    //   "processing"  — a one-shot mode is in flight (currency / describe /
    //                   read / question / detect)
    this.state = "idle";

    this.currentAudio = null;
    this.speechRecognizer = null;
    this.recognizerRestartTimer = null;
    this.mediaRecorder = null;

    this.tier2Available = false;
    this._lastActionAt = 0;
    this._recognizerPaused = false;
    this._wakeWantsToRun = false;
    this._recognizerGen = 0;   // invalidates stale onend/spawn callbacks — see _spawnRecognizer

    // ---- Mode-run epoch: the actual fix for "command matched but did
    // nothing" ----
    //
    // This app has THREE independent async event sources touching shared
    // state: the SpeechRecognition callback (voice commands, firing on
    // its own timeline), fetch() promise resolution (network-timed), and
    // HTMLAudioElement events (audio-hardware-timed). None of them run
    // on a shared clock. A plain `this.state = "..."` string checked "by
    // convention" at the top of a function is NOT a lock — it only
    // protects the instant it's read, and every `await` in an async
    // mode function is a point where the world can change underneath it
    // (the user re-triggers a command) before the function resumes and
    // blindly keeps going as if nothing happened. That's what "matched,
    // but returned with no network call" looks like: a mode function
    // silently continuing past a stale check into a no-op path, or two
    // overlapping mode runs both writing `this.state` and clobbering
    // each other.
    //
    // Fix: `_modeEpoch` increments exactly once per attempted mode
    // dispatch, synchronously, in the same tick as the command match —
    // no async gap for a second command to sneak through. Every mode
    // function captures its own epoch value at entry and MUST re-check
    // `epoch === this._modeEpoch` after every single `await` before
    // touching shared state (this.state, this.currentAudio, the mode
    // card) or speaking. A mismatch means we've been superseded — abort
    // silently, do not touch anything, do not speak. This is the
    // standard "cancellation token" shape (conceptually identical to an
    // AbortController, but signalled by value-equality instead of an
    // event, since there's no long-running I/O to actually abort here —
    // fetch() will still complete, we just discard its result).
    this._modeEpoch = 0;
    this._modeInFlight = false;   // true from the instant a command is accepted until it fully resolves
  }

  // True if `epoch` is still the live one. Call after every `await`
  // inside a mode function before doing anything user-visible.
  _epochLive(epoch) {
    return epoch === this._modeEpoch;
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
    this._enterIdleMode();
    this._startWakeListener();
    this._bindCommandButtons();
    this._bindVisibilityLogging();

    // Ping the server once at startup purely to learn whether Gemma
    // (tier2) is reachable, so the footer can say so honestly. No
    // persistent connection is kept — every mode already talks to the
    // server over plain one-shot fetch() calls.
    this._checkTier2Status();
  }

  async _checkTier2Status() {
    try {
      const res = await fetch("/api/vlm", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({}),   // deliberately missing image_b64 -> cheap 400, just probing reachability
      });
      this.tier2Available = res.status !== 503;
    } catch (_) {
      this.tier2Available = false;
    }
    this._footer(this.tier2Available ? "Voice + Gemma3 ready." : "Gemma3 unavailable — describe/question limited.");
  }

  // A backgrounded tab (screen lock, app-switch, another window
  // focused) can leave the <video> element in a transient state where
  // `_captureFrame()` returns null — every mode function already logs
  // that with the video's actual readyState/dimensions when it happens
  // (see the `console.warn` calls in _run*Mode), but log the visibility
  // transition itself too so "camera not ready" in the console can be
  // correlated with "tab was hidden a moment earlier" instead of
  // looking like an unexplained one-off.
  _bindVisibilityLogging() {
    document.addEventListener("visibilitychange", () => {
      console.log(`👁️  tab visibility: ${document.visibilityState} | video.readyState=${this.video.readyState}`);
    });
  }

  // Wire the tappable "currency"/"describe"/"read"/"question"/"detect" words in
  // the hint line to the exact same dispatch path voice uses
  // (_tryInvokeCommand), so a tap and a spoken command are genuinely
  // two front doors into one system, not two separate paths that can
  // drift out of sync.
  _bindCommandButtons() {
    const buttons = document.querySelectorAll(".cmd-word[data-cmd]");
    buttons.forEach((btn) => {
      btn.addEventListener("click", () => {
        const cmd = btn.dataset.cmd;
        console.log(`👆 command word tapped: ${cmd}`);
        const accepted = this._tryInvokeCommand(cmd, "tap");
        if (!accepted) {
          // Give tactile feedback that the tap registered but was
          // declined, rather than silently doing nothing.
          this._flashButtonDeclined(btn);
        }
      });
    });
  }

  _flashButtonDeclined(btn) {
    btn.disabled = true;
    setTimeout(() => { btn.disabled = false; }, 600);
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

  // -------- Idle mode (default state — nothing running) --------

  _enterIdleMode() {
    this.state = "idle";
    this._setModeCard("idle", "🎙️", "Ready", "Say a command, or tap one below.");
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

  // Create a FRESH recognizer each cycle (Chrome kills continuous
  // recognition every ~60s, and reusing one instance across that is
  // unreliable — it often throws "already started" or silently never
  // restarts). Every recognizer instance is tagged with the generation
  // counter active when it was spawned; its own onend/onerror callbacks
  // only schedule a respawn if that generation is STILL the current one.
  // This is what actually prevents the double-recognizer race: pause →
  // abort() (queues an onend for gen N) → resume (bumps to gen N+1 and
  // spawns) → the late gen-N onend fires and is a no-op because
  // this._recognizerGen is now N+1. Without this tag, both the aborted
  // recognizer's respawn AND the explicit resume's respawn would create
  // two live SpeechRecognition instances fighting over the mic, and one
  // would silently fail to start — leaving `speechRecognizer` pointing
  // at a dead object while commands stopped working with no error shown.
  _spawnRecognizer() {
    if (!this._wakeWantsToRun || this._recognizerPaused) return;

    const gen = ++this._recognizerGen;
    const r = new this._SR();
    r.continuous = true;
    r.interimResults = true;
    r.lang = "en-US";

    r.onstart = () => {
      if (gen !== this._recognizerGen) return;
      this._badge("Listening", true);
      console.log(`🎧 recognizer #${gen} started`);
    };
    r.onresult = (ev) => {
      if (gen !== this._recognizerGen) return;   // stale instance, ignore
      this._onSpeechResult(ev);
    };
    r.onerror = (e) => {
      if (gen !== this._recognizerGen) return;
      console.warn(`🎧 recognizer #${gen} error:`, e.error);
      if (e.error === "not-allowed") {
        this._wakeWantsToRun = false;
        this._badge("Mic blocked", false);
        this._footer("Microphone permission was denied. Allow it and reload.");
      }
      // 'no-speech' / 'aborted' / 'network' → let onend respawn.
    };
    r.onend = () => {
      console.log(`🎧 recognizer #${gen} ended`);
      if (gen !== this._recognizerGen) return;   // superseded — do NOT respawn
      this._badge("…", false);
      clearTimeout(this.recognizerRestartTimer);
      this.recognizerRestartTimer = setTimeout(() => this._spawnRecognizer(), 300);
    };

    this.speechRecognizer = r;
    try {
      r.start();
    } catch (e) {
      if (gen !== this._recognizerGen) return;
      console.warn(`🎧 recognizer #${gen} start failed, retrying:`, e);
      clearTimeout(this.recognizerRestartTimer);
      this.recognizerRestartTimer = setTimeout(() => this._spawnRecognizer(), 500);
    }
  }

  _pauseRecognizer() {
    // Bump the generation FIRST so the outgoing recognizer's onend (which
    // fires asynchronously after abort()) can never schedule a respawn —
    // it checks its captured `gen` against this._recognizerGen and finds
    // a mismatch.
    this._recognizerPaused = true;
    this._recognizerGen++;
    clearTimeout(this.recognizerRestartTimer);
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

    // Gate on `_modeInFlight`, NOT on `this.state` — `this.state` is a
    // display label and not safe to use as a mutex. `_modeInFlight` is
    // set synchronously, right here, in the same tick as the match —
    // there is no `await` between reading it and setting it, so two
    // SpeechRecognition results arriving back to back cannot both pass
    // this check.
    const cmd = this._parseCommand(heard);
    if (!cmd) return;

    console.log("🎙️ command matched:", cmd, "| inFlight:", this._modeInFlight);
    this._lastActionAt = now;
    this._tryInvokeCommand(cmd, "voice");
  }

  // Single gate for invoking a mode, shared by voice (_onSpeechResult)
  // and the tappable command-word buttons in the hint line
  // (_bindCommandButtons) — both need the exact same synchronous
  // accept-or-reject-and-mint-epoch sequence, so this lives in one
  // place rather than being duplicated (and inevitably drifting) across
  // two call sites. Returns true if the command was accepted.
  _tryInvokeCommand(cmd, source) {
    if (this._modeInFlight) {
      console.log(`(ignored — a mode is already running) [source=${source}]`);
      return false;
    }
    this._modeInFlight = true;
    const epoch = ++this._modeEpoch;
    this._dispatchCommand(cmd, epoch);
    return true;
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

  async _dispatchCommand(cmd, epoch) {
    // `epoch` was minted synchronously in `_onSpeechResult` at the exact
    // moment this command was accepted. Every mode function below
    // receives it and must re-check `_epochLive(epoch)` after each
    // `await` — see the constructor comment for why a plain state
    // string can't do this job.
    console.log(`▶️  dispatch #${epoch}:`, cmd);
    this.state = "processing";
    try {
      switch (cmd) {
        case "currency": await this._runCurrencyMode(epoch); break;
        case "describe": await this._runDescribeMode(epoch); break;
        case "read":     await this._runReadMode(epoch); break;
        case "question": await this._runQuestionMode(epoch); break;
        case "detect":   await this._runDetectMode(epoch); break;
      }
    } catch (err) {
      console.error(`❌ [mode #${epoch} crashed]`, cmd, err);
    } finally {
      // Guarantee we return to the ready state even if the mode threw,
      // AND even if a later command already superseded us — but only
      // touch shared state if we're still the live epoch. If we were
      // superseded, whoever superseded us owns the return-to-idle step;
      // doing it here too would double-fire _enterIdleMode (harmless-ish,
      // but also clears `_lastActionAt` that the newer run may already
      // be relying on).
      if (this._epochLive(epoch)) {
        this._modeInFlight = false;
        this._returnToIdle();
      } else {
        console.log(`⏭️  dispatch #${epoch} finished but was superseded — not touching shared state`);
      }
    }
  }

  _returnToIdle() {
    this._lastActionAt = 0;   // clear debounce so the next command fires immediately
    this._enterIdleMode();
    console.log("↩️  returned to idle — ready for next command");
  }

  // -------- Mode: currency (POST /api/vlm with a currency-specific prompt) --------
  //
  // Currency detection is a Tier-1 YOLO model that streams over the
  // WebSocket, but the *voice-invoked* single-shot use case wants a
  // deterministic REST round-trip. We use /api/vlm with a currency
  // prompt so a single frame is analyzed. If the frame's blurry or
  // empty, Gemma politely says so instead of us silently returning
  // garbage.

  async _runCurrencyMode(epoch) {
    this._setModeCard("mode", "💵", "Currency", "Scanning for notes…");
    // Small delay so the user has time to steady the camera. No spoken
    // prompt — the user just said "currency", they know they've been
    // heard from the visual state change.
    await this._sleep(600);
    if (!this._epochLive(epoch)) return;   // superseded during the sleep

    const b64 = this._captureFrame(HIRES_MAX_DIM, HIRES_QUALITY);
    if (!b64) {
      console.warn(`[currency #${epoch}] camera not ready — video.readyState=${this.video.readyState}, dims=${this.video.videoWidth}x${this.video.videoHeight}`);
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
      if (!this._epochLive(epoch)) return;   // superseded while the request was in flight
      const data = await res.json();
      if (!this._epochLive(epoch)) return;
      speak = data.speak || speak;
      this._setModeCard("mode", "💵", "Currency", speak);
    } catch (err) {
      console.error(`[currency #${epoch}] fetch failed:`, err);
      speak = "Could not reach the server.";
    }
    if (this._epochLive(epoch)) {
      try { await this._sayAndWait(speak); } catch (_) {}
    }
  }

  // -------- Mode: describe --------

  async _runDescribeMode(epoch) {
    this._setModeCard("mode", "👁️", "Describe scene", "Looking…");
    await this._sleep(400);
    if (!this._epochLive(epoch)) return;

    const b64 = this._captureFrame(HIRES_MAX_DIM, HIRES_QUALITY);
    if (!b64) {
      console.warn(`[describe #${epoch}] camera not ready — video.readyState=${this.video.readyState}, dims=${this.video.videoWidth}x${this.video.videoHeight}`);
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
      this._setModeCard("mode", "👁️", "Describe scene", speak);
    } catch (err) {
      console.error(`[describe #${epoch}] fetch failed:`, err);
      speak = "Could not reach the server.";
    }
    if (this._epochLive(epoch)) {
      try { await this._sayAndWait(speak); } catch (_) {}
    }
  }

  // -------- Mode: read (OCR + Gemma summary) --------

  async _runReadMode(epoch) {
    this._setModeCard("mode", "📖", "Read text", "Reading…");
    await this._sleep(400);
    if (!this._epochLive(epoch)) return;

    // High-res capture — OCR needs the detail. The 640px collision
    // frame reads as garbage; text needs ~1600px to be legible.
    const b64 = this._captureFrame(HIRES_MAX_DIM, HIRES_QUALITY);
    if (!b64) {
      console.warn(`[read #${epoch}] camera not ready — video.readyState=${this.video.readyState}, dims=${this.video.videoWidth}x${this.video.videoHeight}`);
      if (this._epochLive(epoch)) await this._sayAndWait("Camera not ready.");
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
      if (!this._epochLive(epoch)) return;
      const data = await res.json();
      if (!this._epochLive(epoch)) return;
      speak = data.speak || speak;
      this._setModeCard("mode", "📖", "Read text", speak);
    } catch (err) {
      console.error(`[read #${epoch}] fetch failed:`, err);
      speak = "Could not reach the server.";
    }
    if (this._epochLive(epoch)) {
      try { await this._sayAndWait(speak); } catch (_) {}
    }
  }

  // -------- Mode: question (record audio -> STT -> Gemma) --------

  async _runQuestionMode(epoch) {
    // Question mode is the one place we DO need a spoken prompt — the
    // user has to know that mic recording has started.
    this._setModeCard("listen", "🎤", "Ask a question", "Ask your question now…");
    await this._sayAndWait("Ask your question.");
    if (!this._epochLive(epoch)) return;

    try {
      const audioB64 = await this._recordAudio(QUESTION_RECORD_MS);
      if (!this._epochLive(epoch)) return;

      this._setModeCard("mode", "🧠", "Ask a question", "Transcribing…");
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
      if (!b64) {
        console.warn(`[question #${epoch}] camera not ready — video.readyState=${this.video.readyState}`);
        if (this._epochLive(epoch)) await this._sayAndWait("Camera not ready.");
        return;
      }
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
      console.error(`[question #${epoch}] failed:`, err);
      if (this._epochLive(epoch)) await this._sayAndWait("Something went wrong recording your question.");
    }
  }

  // -------- Mode: detect (one-shot proximity/collision scan) --------
  //
  // This used to be a continuous background stream that ran as the
  // app's default state. It's now a one-shot scan, same shape as
  // currency/describe/read: capture a frame, ask the server, speak the
  // result, done. See the top-of-file note for the safety trade-off
  // this implies (no background watcher between scans anymore).

  async _runDetectMode(epoch) {
    this._setModeCard("mode", "🛡️", "Detecting", "Scanning surroundings…");
    await this._sleep(300);
    if (!this._epochLive(epoch)) return;

    const b64 = this._captureFrame();   // fast/small capture — object detection doesn't need fine detail
    if (!b64) {
      console.warn(`[detect #${epoch}] camera not ready — video.readyState=${this.video.readyState}, dims=${this.video.videoWidth}x${this.video.videoHeight}`);
      if (this._epochLive(epoch)) await this._sayAndWait("Camera not ready.");
      return;
    }

    let speak = "Path is clear.";
    let severity = null;
    try {
      const res = await fetch("/api/detect", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image_b64: b64 }),
      });
      if (!this._epochLive(epoch)) return;
      const data = await res.json();
      if (!this._epochLive(epoch)) return;
      speak = data.speak || speak;
      severity = data.collision_alert || null;
      const cardState = severity === "urgent" ? "urgent" : severity === "warn" ? "warn" : "clear";
      const icon = severity === "urgent" ? "🛑" : severity === "warn" ? "⚠️" : "🛡️";
      const label = severity === "urgent" ? "STOP" : severity === "warn" ? "Careful" : "Clear";
      this._setModeCard(cardState, icon, label, speak);
    } catch (err) {
      console.error(`[detect #${epoch}] fetch failed:`, err);
      speak = "Could not reach the server.";
    }

    if (severity === "warn") this._safeVibrate(PROXIMITY_HAPTICS.warn);
    if (severity === "urgent") this._safeVibrate(PROXIMITY_HAPTICS.urgent);

    if (this._epochLive(epoch)) {
      try { await this._sayAndWait(speak); } catch (_) {}
    }
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

  // Speak text and only resolve after playback ends (or 12s max as a
  // safety net). Used inside modes so the "return to idle" step doesn't
  // stomp on the answer mid-sentence.
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
