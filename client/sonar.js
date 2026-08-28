/**
 * Audio "sonar" — continuous guidance tone so a user who cannot see a
 * viewfinder can still aim the camera by ear.
 *
 * Pitch rises from ~220Hz to ~880Hz as the server-reported framing
 * score goes from 0 to 1. Volume drops to silence once framing is
 * good — silence means "ready", which matters because we never want
 * this tone to mask spoken results.
 */
export class Sonar {
  constructor() {
    this.ctx = null;
    this.osc = null;
    this.gain = null;
    this.running = false;
  }

  start() {
    if (this.running) return;
    this.ctx = new (window.AudioContext || window.webkitAudioContext)();
    this.osc = this.ctx.createOscillator();
    this.gain = this.ctx.createGain();
    this.osc.type = "sine";
    this.osc.frequency.value = 220;
    this.gain.gain.value = 0.0;
    this.osc.connect(this.gain).connect(this.ctx.destination);
    this.osc.start();
    this.running = true;
  }

  stop() {
    if (!this.running) return;
    try {
      this.gain.gain.setTargetAtTime(0, this.ctx.currentTime, 0.05);
      this.osc.stop(this.ctx.currentTime + 0.2);
    } catch (_) { /* already stopped */ }
    this.running = false;
  }

  /** score: 0..1 framing quality from the server's quality gate. */
  update(score) {
    if (!this.running) return;
    const clamped = Math.max(0, Math.min(1, score));
    const freq = 220 + clamped * 660;
    // Near-perfect framing goes quiet — silence itself is the "ready" signal.
    const vol = clamped > 0.85 ? 0.0 : 0.05 + (1 - clamped) * 0.05;
    const now = this.ctx.currentTime;
    this.osc.frequency.setTargetAtTime(freq, now, 0.06);
    this.gain.gain.setTargetAtTime(vol, now, 0.06);
  }
}
