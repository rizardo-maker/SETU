"""
Text-to-speech, with a three-rung fallback so the demo works on day
one and the "real" path is clear for later:

  1. Piper (preferred) — fully local, fast, small, ships proper Indic
     voices. This is what a deployed node should run.
  2. macOS `say` + afconvert — zero setup, works immediately on any
     Mac, good enough to demo the pipeline while you install Piper
     voices. NOT what you ship — it's not available on Linux/Windows
     and isn't the multilingual story the project is built around.
  3. Fail loudly with a clear message — never fail silently, per the
     project's whole design philosophy.

Every backend returns 16-bit PCM WAV bytes so the client-side
<audio>/Web Audio playback code doesn't need to know which one ran.
"""
from __future__ import annotations
import logging
import platform
import shutil
import subprocess
import tempfile
from pathlib import Path

from server import config

log = logging.getLogger("setu.tts")


class TTSUnavailable(RuntimeError):
    pass


class TextToSpeech:
    def __init__(self):
        self.backend: str | None = None
        self._piper_voice = None
        self._detect()

    def _detect(self) -> None:
        # Rung 1: Piper CLI binary + a downloaded voice model.
        voice_model = config.MODELS_DIR / "tts" / f"{config.TTS_VOICE}.onnx"
        if shutil.which("piper") and voice_model.exists():
            self.backend = "piper"
            self._piper_model_path = voice_model
            log.info("TTS backend: piper (%s)", config.TTS_VOICE)
            return

        # Rung 2: macOS `say`, only on macOS, only as a dev-time fallback.
        if platform.system() == "Darwin" and shutil.which("say") and shutil.which("afconvert"):
            self.backend = "macos_say"
            log.warning(
                "TTS backend: macOS `say` (fallback). This is for local dev only — "
                "install Piper + a voice model in models/tts/ before you deploy or demo "
                "the multilingual/offline story to a jury."
            )
            return

        self.backend = None
        log.warning("No TTS backend available. Install Piper (requirements-full.txt) "
                    "and place a voice .onnx in models/tts/.")

    @property
    def ready(self) -> bool:
        return self.backend is not None

    def synthesize(self, text: str) -> bytes:
        if self.backend == "piper":
            return self._synthesize_piper(text)
        if self.backend == "macos_say":
            return self._synthesize_macos(text)
        raise TTSUnavailable("No text-to-speech backend is configured.")

    def _synthesize_piper(self, text: str) -> bytes:
        proc = subprocess.run(
            ["piper", "--model", str(self._piper_model_path), "--output_file", "-"],
            input=text.encode("utf-8"),
            capture_output=True,
            timeout=15,
        )
        if proc.returncode != 0:
            raise TTSUnavailable(f"piper failed: {proc.stderr.decode(errors='replace')[:300]}")
        return proc.stdout

    def _synthesize_macos(self, text: str) -> bytes:
        with tempfile.TemporaryDirectory() as td:
            aiff_path = Path(td) / "out.aiff"
            wav_path = Path(td) / "out.wav"
            subprocess.run(["say", "-o", str(aiff_path), text], check=True, timeout=15)
            subprocess.run(
                ["afconvert", "-f", "WAVE", "-d", "LEI16@22050", str(aiff_path), str(wav_path)],
                check=True, timeout=15,
            )
            return wav_path.read_bytes()


engine = TextToSpeech()
