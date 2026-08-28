"""
Speech-to-text — server-side, on purpose.

Do NOT use the browser's Web Speech API for this. In Chrome it sends
audio to Google's servers for recognition, which silently breaks the
project's core "nothing leaves the room" claim in a way a technical
judge will catch. faster-whisper runs the model locally instead.

Lazy-imported so the base server boots without it installed.
"""
from __future__ import annotations
import io
import logging

import numpy as np

from server import config

log = logging.getLogger("setu.stt")


class STTUnavailable(RuntimeError):
    pass


class SpeechToText:
    def __init__(self):
        self._model = None
        self.ready = False
        self._load()

    def _load(self) -> None:
        try:
            from faster_whisper import WhisperModel  # lazy
            # int8 compute type keeps this usable on a CPU-only laptop.
            self._model = WhisperModel(config.STT_MODEL_SIZE, device="cpu", compute_type="int8")
            self.ready = True
            log.info("STT loaded: faster-whisper (%s, cpu/int8)", config.STT_MODEL_SIZE)
        except ImportError:
            log.warning("faster-whisper not installed — voice commands unavailable. "
                        "pip install -r requirements-full.txt to enable.")
        except Exception as e:
            log.warning("faster-whisper failed to load: %s", e)

    def transcribe(self, wav_bytes: bytes) -> str:
        if not self.ready:
            raise STTUnavailable(
                "Speech recognition isn't available. Install faster-whisper "
                "(requirements-full.txt) to enable voice commands."
            )
        segments, _info = self._model.transcribe(
            io.BytesIO(wav_bytes), language=config.STT_LANGUAGE, vad_filter=True
        )
        return " ".join(seg.text.strip() for seg in segments).strip()


engine = SpeechToText()
