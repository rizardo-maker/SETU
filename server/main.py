"""
SETU server — FastAPI app: serves the browser client and runs the
WebSocket loop that drives it.

Run with:  python -m server.main   (or scripts/dev.sh)

Design reminder for anyone editing this file: every failure path must
end in a spoken message to the user, never a silent drop. That rule
is more important than almost anything else in this codebase — see
the project document's "fault tolerance" section for why.
"""
from __future__ import annotations
import base64
import logging
import ssl
import time
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from server import config
from server.ws_protocol import parse_client_message, ClientFrame, ClientAudio, ClientControl
from server.tier1 import quality_gate, currency, ocr, detect
from server.tier2 import vlm
from server.speech import stt, tts
from server.arbiter import MultiFrameArbiter

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logging.getLogger("setu.vlm").setLevel(logging.DEBUG)
log = logging.getLogger("setu.main")
log.setLevel(logging.DEBUG)


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("SETU starting up.")
    log.info("Currency classifier ready: %s", currency.classifier.ready)
    log.info("OCR ready: %s (%s)", ocr.engine.ready, ocr.engine.backend)
    log.info("Obstacle detector ready: %s", detect.detector.ready)
    log.info("STT ready: %s", stt.engine.ready)
    log.info("TTS ready: %s (%s)", tts.engine.ready, tts.engine.backend)
    await vlm.warm_up()
    yield
    log.info("SETU shutting down.")


app = FastAPI(title="SETU", lifespan=lifespan)


def decode_jpeg_b64(image_b64: str) -> np.ndarray:
    raw = base64.b64decode(image_b64)
    arr = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Could not decode image — corrupt or unsupported format.")
    return frame


class ConnectionState:
    """Everything the WebSocket loop needs to remember for one client.
    One instance per connection — nothing here is shared, so there's no
    locking to worry about."""

    def __init__(self):
        self.prev_gray: np.ndarray | None = None
        self.arbiter = MultiFrameArbiter(labels=currency.classifier.labels or [])
        self.last_mode: str | None = None
        self.last_hint_time: float = 0.0
        self.last_result: dict | None = None

    def reset_for_mode(self, mode: str) -> None:
        if mode != self.last_mode:
            self.arbiter.reset()
            self.prev_gray = None
            self.last_mode = mode


HINT_COOLDOWN_S = 1.5


async def handle_frame(msg: ClientFrame, state: ConnectionState, ws: WebSocket) -> None:
    t0 = time.monotonic()
    state.reset_for_mode(msg.mode)

    log.info("📥 [Frame] mode='%s' seq=%s question=%r", msg.mode, msg.seq, msg.question)

    try:
        frame = decode_jpeg_b64(msg.image_b64)
    except ValueError as e:
        log.error("❌ Image decode failed: %s", e)
        await ws.send_json({"type": "result", "seq": msg.seq, "tier": 1, "mode": msg.mode,
                            "answered": False, "speak": str(e)})
        return

    gate, gray = quality_gate.assess(frame, state.prev_gray)
    state.prev_gray = gray

    now = time.monotonic()
    hint = gate.hint if (now - state.last_hint_time) > HINT_COOLDOWN_S else None
    if hint:
        state.last_hint_time = now

    await ws.send_json({
        "type": "guidance", "seq": msg.seq,
        "framing_score": gate.framing_score,
        "torch_suggested": gate.torch_suggested,
        "spoken_hint": hint,
    })

    # ---- Tier 2 modes (scene description & Q&A) ----
    # Single-shot requests: always run the VLM without quality-gate dropping
    if msg.mode in ("scene", "question"):
        log.info("🤖 [VLM Request] mode='%s' question=%r -> querying Moondream...", msg.mode, msg.question)
        ok, jpeg_bytes = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        try:
            speak, latency_ms = await vlm.describe(jpeg_bytes.tobytes(), question=msg.question)
            log.info("✅ [VLM Response] (%0.1fms): '%s'", latency_ms, speak)
            await ws.send_json({"type": "result", "seq": msg.seq, "tier": 2, "mode": msg.mode,
                                "answered": True, "speak": speak,
                                "latency_ms": round(latency_ms, 1)})
        except vlm.VLMUnavailable as e:
            log.warning("⚠️ VLM unavailable: %s", e)
            await ws.send_json({"type": "result", "seq": msg.seq, "tier": 2, "mode": msg.mode,
                                "answered": False, "speak": config.PHRASES["tier2_unavailable"]})
        return

    # For continuous streaming currency/obstacle mode, skip unsharp/dark frames if quality gate rejects
    if msg.mode in ("currency", "obstacle") and not gate.accept and (currency.classifier.ready or detect.detector.ready):
        return  # guidance already sent; don't waste inference on a bad frame

    # ---- Tier 1 modes (with seamless VLM fallback if specialized model not installed) ----
    if msg.mode == "currency":
        if currency.classifier.ready:
            logits = currency.classifier.predict_logits(frame)
            decision = state.arbiter.submit(logits)
            await ws.send_json({
                "type": "result", "seq": msg.seq, "tier": 1, "mode": "currency",
                "answered": decision.answered,
                "label": currency.classifier.labels[decision.label_idx] if decision.answered else None,
                "confidence": round(decision.confidence, 3),
                "margin": round(decision.margin, 3),
                "speak": decision.speak,
                "latency_ms": round((time.monotonic() - t0) * 1000, 1),
            })
            return
        elif await vlm.is_available():
            ok, jpeg_bytes = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            speak, latency_ms = await vlm.describe(
                jpeg_bytes.tobytes(),
                question="Identify the currency note or coin in this image (e.g. 10, 20, 50, 100, 200, 500 Rupees). Say only the denomination. If no currency is visible, say 'No currency detected'."
            )
            await ws.send_json({"type": "result", "seq": msg.seq, "tier": 2, "mode": "currency",
                                "answered": "no currency" not in speak.lower(), "speak": speak,
                                "latency_ms": round(latency_ms, 1)})
            return
        else:
            await ws.send_json({"type": "result", "seq": msg.seq, "tier": 1, "mode": "currency",
                                "answered": False,
                                "speak": "The currency model hasn't been trained yet."})
            return

    if msg.mode == "text":
        log.info("📖 [Text Mode] Processing frame (OCR ready: %s, question: %r)",
                 ocr.engine.ready, msg.question)
        if ocr.engine.ready:
            try:
                text = ocr.engine.read(frame)
                log.info("📖 [OCR Extracted Text] (%d chars): %r", len(text), text[:120] if text else "")
                if not text:
                    await ws.send_json({"type": "result", "seq": msg.seq, "tier": 1, "mode": "text",
                                        "answered": False, "speak": "No text found. Try moving closer.",
                                        "latency_ms": round((time.monotonic() - t0) * 1000, 1)})
                    return

                prompt_q = msg.question.strip() if (msg.question and msg.question.strip()) else "Read and clearly transcribe or summarize the key text in 1-2 sentences."
                try:
                    speak, _vlm_latency = await vlm.answer_from_text(text, prompt_q)
                    log.info("🤖 [Gemma Answer from OCR] (%0.1fms): '%s'", _vlm_latency, speak)
                except vlm.VLMUnavailable:
                    log.warning("VLM unavailable, speaking raw OCR text")
                    speak = text

                await ws.send_json({"type": "result", "seq": msg.seq, "tier": 1, "mode": "text",
                                    "answered": True, "label": text, "ocr_text": text, "speak": speak,
                                    "latency_ms": round((time.monotonic() - t0) * 1000, 1)})
            except RuntimeError as e:
                log.error("OCR runtime error: %s", e)
                await ws.send_json({"type": "result", "seq": msg.seq, "tier": 1, "mode": "text",
                                    "answered": False, "speak": str(e)})
            return
        elif await vlm.is_available():
            ok, jpeg_bytes = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            speak, latency_ms = await vlm.describe(
                jpeg_bytes.tobytes(),
                question=msg.question if (msg.question and msg.question.strip()) else "Read and transcribe all visible text or signboards in this image. Keep it concise. If there is no text, say 'No text found'."
            )
            await ws.send_json({"type": "result", "seq": msg.seq, "tier": 2, "mode": "text",
                                "answered": "no text" not in speak.lower(), "speak": speak,
                                "latency_ms": round(latency_ms, 1)})
            return
        else:
            await ws.send_json({"type": "result", "seq": msg.seq, "tier": 1, "mode": "text",
                                "answered": False, "speak": "No OCR backend installed."})
            return

    if msg.mode == "obstacle":
        if detect.detector.ready:
            try:
                dets = detect.detector.detect(frame)
                await ws.send_json({"type": "result", "seq": msg.seq, "tier": 1, "mode": "obstacle",
                                    "answered": True, "speak": detect.detector.speak(dets),
                                    "latency_ms": round((time.monotonic() - t0) * 1000, 1)})
            except RuntimeError as e:
                await ws.send_json({"type": "result", "seq": msg.seq, "tier": 1, "mode": "obstacle",
                                    "answered": False, "speak": str(e)})
            return
        elif await vlm.is_available():
            ok, jpeg_bytes = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            speak, latency_ms = await vlm.describe(
                jpeg_bytes.tobytes(),
                question="Identify any obstacles, chairs, stairs, doorways, people, or hazards directly ahead in the walking path in one sentence."
            )
            await ws.send_json({"type": "result", "seq": msg.seq, "tier": 2, "mode": "obstacle",
                                "answered": True, "speak": speak,
                                "latency_ms": round(latency_ms, 1)})
            return
        else:
            await ws.send_json({"type": "result", "seq": msg.seq, "tier": 1, "mode": "obstacle",
                                "answered": False, "speak": "Obstacle detector not available."})
            return


async def handle_audio(msg: ClientAudio, ws: WebSocket) -> None:
    try:
        wav_bytes = base64.b64decode(msg.audio_b64)
        text = stt.engine.transcribe(wav_bytes)
        await ws.send_json({"type": "transcript", "text": text, "seq": msg.seq})
    except stt.STTUnavailable as e:
        await ws.send_json({"type": "transcript", "text": "", "seq": msg.seq})
        log.warning("STT unavailable: %s", e)


async def handle_control(msg: ClientControl, state: ConnectionState, ws: WebSocket) -> None:
    if msg.action == "ping":
        await ws.send_json({"type": "status", "tier2_available": await vlm.is_available()})
    elif msg.action == "repeat" and state.last_result:
        await ws.send_json(state.last_result)
    elif msg.action == "cancel":
        state.arbiter.reset()


@app.websocket("/ws/stream")
async def ws_stream(ws: WebSocket) -> None:
    await ws.accept()
    state = ConnectionState()
    tier2_available = await vlm.is_available()
    await ws.send_json({"type": "status", "tier2_available": tier2_available,
                        "message": None if tier2_available else config.PHRASES["tier2_unavailable"]})
    try:
        while True:
            raw = await ws.receive_json()
            log.debug("[WS] Received message: type=%s mode=%s keys=%s",
                      raw.get("type"), raw.get("mode"), list(raw.keys()))
            try:
                msg = parse_client_message(raw)
            except Exception as e:
                log.error("[WS] Failed to parse message: %s — raw keys: %s", e, list(raw.keys()))
                await ws.send_json({"type": "status", "tier2_available": tier2_available,
                                    "message": f"Bad message: {e}"})
                continue

            if isinstance(msg, ClientFrame):
                await handle_frame(msg, state, ws)
            elif isinstance(msg, ClientAudio):
                await handle_audio(msg, ws)
            elif isinstance(msg, ClientControl):
                await handle_control(msg, state, ws)
    except WebSocketDisconnect:
        log.info("Client disconnected.")


@app.post("/api/tts")
async def api_tts(payload: dict):
    """Simple non-streaming TTS endpoint: {"text": "..."} -> audio/wav bytes.
    Kept as plain REST (not over the WebSocket) since it's a one-shot
    request/response, not part of the continuous frame loop."""
    from fastapi.responses import Response
    text = payload.get("text", "")
    if not text:
        return Response(status_code=400, content=b"missing 'text'")
    try:
        wav = tts.engine.synthesize(text)
        return Response(content=wav, media_type="audio/wav")
    except tts.TTSUnavailable as e:
        return Response(status_code=503, content=str(e).encode())


@app.post("/api/vlm")
async def api_vlm(payload: dict):
    """Single-shot VLM endpoint for scene description and questions.
    Accepts: {"image_b64": "...", "question": "optional question"}
    Returns: {"speak": "...", "latency_ms": ..., "model": "..."}
    """
    from fastapi.responses import JSONResponse
    image_b64 = payload.get("image_b64", "")
    question = payload.get("question")
    if not image_b64:
        return JSONResponse(status_code=400, content={"error": "missing image_b64"})

    log.info("[API/VLM] Request received — image_b64 len=%d, question=%r", len(image_b64), question)

    try:
        frame = decode_jpeg_b64(image_b64)
    except ValueError as e:
        log.error("[API/VLM] Image decode failed: %s", e)
        return JSONResponse(status_code=400, content={"error": str(e)})

    ok, jpeg_bytes = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        return JSONResponse(status_code=500, content={"error": "JPEG re-encode failed"})

    try:
        model = await vlm.get_model_name()
        speak, latency_ms = await vlm.describe(jpeg_bytes.tobytes(), question=question)
        log.info("[API/VLM] Response (%.0fms, model=%s): '%s'", latency_ms, model, speak[:100])
        return JSONResponse(content={
            "speak": speak,
            "latency_ms": round(latency_ms, 1),
            "model": model,
        })
    except vlm.VLMUnavailable as e:
        log.warning("[API/VLM] VLM unavailable: %s", e)
        return JSONResponse(status_code=503, content={"error": str(e), "speak": config.PHRASES["tier2_unavailable"]})


@app.post("/api/ocr")
async def api_ocr(payload: dict):
    """Single-shot OCR + Gemma reasoning endpoint.
    Accepts: {"image_b64": "...", "question": "optional question"}
    Returns: {"speak": "...", "ocr_text": "...", "latency_ms": ..., "tier": 1}
    """
    from fastapi.responses import JSONResponse
    image_b64 = payload.get("image_b64", "")
    question = payload.get("question")
    if not image_b64:
        return JSONResponse(status_code=400, content={"error": "missing image_b64"})

    t0 = time.monotonic()
    log.info("[API/OCR] Request received — image_b64 len=%d, question=%r", len(image_b64), question)

    try:
        frame = decode_jpeg_b64(image_b64)
    except ValueError as e:
        log.error("[API/OCR] Image decode failed: %s", e)
        return JSONResponse(status_code=400, content={"error": str(e)})

    if ocr.engine.ready:
        try:
            text = ocr.engine.read(frame)
            log.info("[API/OCR] Extracted text (%d chars): %r", len(text), text[:120] if text else "")
            if not text:
                return JSONResponse(content={
                    "speak": "No text found. Try moving closer.",
                    "ocr_text": "",
                    "answered": False,
                    "tier": 1,
                    "latency_ms": round((time.monotonic() - t0) * 1000, 1),
                })

            prompt_q = question.strip() if (question and question.strip()) else "Read and clearly transcribe or summarize the key text in 1-2 sentences."
            try:
                speak, _ = await vlm.answer_from_text(text, prompt_q)
                log.info("[API/OCR] Gemma answer: '%s'", speak[:100])
            except vlm.VLMUnavailable:
                log.warning("[API/OCR] VLM unavailable, falling back to raw OCR text")
                speak = text

            return JSONResponse(content={
                "speak": speak,
                "ocr_text": text,
                "answered": True,
                "tier": 1,
                "latency_ms": round((time.monotonic() - t0) * 1000, 1),
            })
        except Exception as e:
            log.error("[API/OCR] OCR error: %s", e)
            return JSONResponse(status_code=500, content={"error": str(e), "speak": "OCR processing failed."})
    elif await vlm.is_available():
        ok, jpeg_bytes = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        speak, latency_ms = await vlm.describe(
            jpeg_bytes.tobytes(),
            question=question if (question and question.strip()) else "Read and transcribe all visible text in this image."
        )
        return JSONResponse(content={
            "speak": speak,
            "ocr_text": speak,
            "answered": "no text" not in speak.lower(),
            "tier": 2,
            "latency_ms": round(latency_ms, 1),
        })
    else:
        return JSONResponse(status_code=503, content={"error": "No OCR or VLM available", "speak": "No OCR backend installed."})


@app.get("/")
async def index():
    return FileResponse(config.CLIENT_DIR / "index.html")


@app.get("/favicon.ico")
async def favicon():
    from fastapi.responses import Response
    return Response(status_code=204)


app.mount("/static", StaticFiles(directory=str(config.CLIENT_DIR)), name="static")


def _ssl_context() -> ssl.SSLContext | None:
    if not config.USE_TLS:
        return None
    if not (config.CERT_FILE.exists() and config.KEY_FILE.exists()):
        log.warning(
            "SETU_TLS=1 but no certs found at %s / %s. Run scripts/gen_certs.sh — "
            "without HTTPS, getUserMedia will silently fail on a phone. "
            "Falling back to plain HTTP for localhost-only testing.",
            config.CERT_FILE, config.KEY_FILE,
        )
        return None
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(str(config.CERT_FILE), str(config.KEY_FILE))
    return ctx


if __name__ == "__main__":
    import uvicorn
    ssl_ctx = _ssl_context()
    kwargs = {}
    if ssl_ctx is not None:
        kwargs["ssl_certfile"] = str(config.CERT_FILE)
        kwargs["ssl_keyfile"] = str(config.KEY_FILE)
    uvicorn.run("server.main:app", host=config.HOST, port=config.PORT, reload=False, **kwargs)
