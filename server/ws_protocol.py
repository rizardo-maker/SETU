"""
WebSocket message schema between the browser client and the server.

Deliberately JSON-only (base64 image payloads inside JSON), not a raw
binary framing protocol. That costs ~33% extra bytes per frame, which
is a non-issue at 640px/5fps on a LAN, and it buys a schema you can
read, validate, and log without a binary parser. Optimise later if
you ever need to.
"""
from __future__ import annotations
from typing import Literal, Optional, Union
from pydantic import BaseModel, Field


# ---------------- Client -> Server ----------------

class ClientFrame(BaseModel):
    """One camera frame plus what the user is asking of it."""
    type: Literal["frame"] = "frame"
    mode: Literal["currency", "text", "obstacle", "scene", "question"]
    image_b64: str                      # JPEG, base64, no data-URI prefix
    question: Optional[str] = None      # optional user question (used in "question" and "text" modes)
    seq: int = 0                        # client-assigned, echoed back for backpressure


class ClientAudio(BaseModel):
    """A recorded utterance for speech-to-text (e.g. voice mode switch)."""
    type: Literal["audio"] = "audio"
    audio_b64: str                      # 16kHz mono PCM16 WAV, base64
    seq: int = 0


class ClientControl(BaseModel):
    """Non-media control messages: repeat last answer, cancel, ping."""
    type: Literal["control"] = "control"
    action: Literal["repeat", "cancel", "ping"]


ClientMessage = Union[ClientFrame, ClientAudio, ClientControl]


def parse_client_message(raw: dict) -> ClientMessage:
    t = raw.get("type")
    if t == "frame":
        return ClientFrame.model_validate(raw)
    if t == "audio":
        return ClientAudio.model_validate(raw)
    if t == "control":
        return ClientControl.model_validate(raw)
    raise ValueError(f"unknown message type: {t!r}")


# ---------------- Server -> Client ----------------

class ServerGuidance(BaseModel):
    """
    Sent on every accepted frame, even when no answer is ready yet.
    Drives the client's audio sonar — this is the most frequently sent
    message type and must stay cheap to produce.
    """
    type: Literal["guidance"] = "guidance"
    seq: int
    framing_score: float          # 0..1, drives the sonar pitch/volume
    torch_suggested: bool = False
    spoken_hint: Optional[str] = None   # only set occasionally (cooldown-limited)


class ServerResult(BaseModel):
    """A completed answer, from either tier."""
    type: Literal["result"] = "result"
    seq: int
    tier: Literal[1, 2]
    mode: str
    answered: bool                 # False when the system abstained
    label: Optional[str] = None    # e.g. "500" for currency, or free text for scene
    ocr_text: Optional[str] = None # raw OCR text when available
    confidence: Optional[float] = None
    margin: Optional[float] = None
    speak: str                     # what should be spoken/shown regardless of answered
    latency_ms: Optional[float] = None


class ServerTranscript(BaseModel):
    type: Literal["transcript"] = "transcript"
    text: str
    seq: int = 0


class ServerStatus(BaseModel):
    """Connection/tier health, so the client never fails silently."""
    type: Literal["status"] = "status"
    tier2_available: bool
    message: Optional[str] = None


ServerMessage = Union[ServerGuidance, ServerResult, ServerTranscript, ServerStatus]
