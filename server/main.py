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
import asyncio
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

    # Warm up YOLO models — the first inference call inside a live async
    # request otherwise hangs the event loop for ~10-40s while Metal/CUDA
    # sets up its kernels. Running a throwaway prediction now moves that
    # cost to startup where it belongs.
    warmup_frame = np.zeros((640, 640, 3), dtype=np.uint8)
    if currency.classifier.ready:
        try:
            currency.classifier.detect(warmup_frame)
            log.info("[Startup] Currency YOLO warm.")
        except Exception as e:
            log.warning("[Startup] Currency YOLO warmup failed: %s", e)
    if detect.detector.ready:
        try:
            detect.detector.scan_for_collision(warmup_frame)
            log.info("[Startup] Obstacle/collision YOLO warm.")
        except Exception as e:
            log.warning("[Startup] Obstacle YOLO warmup failed: %s", e)

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
        self.currency_arbiter = currency.YOLOFrameArbiter()
        self.last_mode: str | None = None
        self.last_hint_time: float = 0.0
        self.last_collision_hint_time: float = 0.0
        self.last_result: dict | None = None

    def reset_for_mode(self, mode: str) -> None:
        if mode != self.last_mode:
            self.currency_arbiter.reset()
            self.prev_gray = None
            self.last_mode = mode


HINT_COOLDOWN_S = 1.5


async def run_text_mode(frame: np.ndarray, question: str | None) -> dict:
    """
    Shared OCR -> (optional) Gemma-reasoning pipeline for "text" mode, used by
    both the WebSocket frame handler and the single-shot /api/ocr endpoint.

    Decision tree:
      no OCR backend      -> fall back to VLM reading the image directly (tier 2)
      no text found       -> honest "no text found", no VLM call
      low OCR confidence  -> honest "too unclear", no VLM call (garbage in -> hallucinated answer out)
      question asked      -> OCR text handed to Gemma for reasoning (tier 1, extra latency)
      no question         -> raw OCR text spoken directly (tier 1, fast, no VLM round trip)

    Returns a dict of the fields callers merge into their own response shape
    (WS result / REST JSON) — always includes at least "answered", "speak",
    "tier"; includes "ocr_text" / "ocr_confidence" / "label" when applicable.
    """
    q = question.strip() if question and question.strip() else None

    if not ocr.engine.ready:
        if await vlm.is_available():
            ok, jpeg_bytes = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            speak, latency_ms = await vlm.describe(
                jpeg_bytes.tobytes(),
                question=q or "Read and transcribe all visible text or signboards in this image. Keep it concise. If there is no text, say 'No text found'."
            )
            return {"tier": 2, "answered": "no text" not in speak.lower(), "speak": speak, "latency_ms": round(latency_ms, 1)}
        return {"tier": 1, "answered": False, "speak": "No OCR backend installed."}

    result = ocr.engine.read_with_confidence(frame)
    log.info("📖 [OCR] backend=%s confidence=%.2f (%d chars): %r",
              result.backend, result.mean_confidence, len(result.text), result.text[:120])

    if not result.text:
        return {"tier": 1, "answered": False, "speak": "No text found. Try moving closer."}

    if result.mean_confidence < config.OCR_MIN_CONFIDENCE:
        log.info("📖 [OCR] confidence %.2f below floor %.2f — declining to reason over it",
                  result.mean_confidence, config.OCR_MIN_CONFIDENCE)
        return {
            "tier": 1, "answered": False, "speak": config.PHRASES["ocr_low_confidence"],
            "ocr_text": result.text, "ocr_confidence": round(result.mean_confidence, 3),
        }

    if q:
        try:
            speak, vlm_latency = await vlm.answer_from_text(result.text, q, result.mean_confidence)
            log.info("🤖 [Gemma Answer from OCR] (%.0fms): '%s'", vlm_latency, speak)
        except vlm.VLMUnavailable:
            log.warning("VLM unavailable for OCR reasoning — speaking raw OCR text instead")
            speak = result.text
        return {
            "tier": 1, "answered": True, "label": result.text, "speak": speak,
            "ocr_text": result.text, "ocr_confidence": round(result.mean_confidence, 3),
        }

    # No question: reading text back verbatim is a legitimate fast path —
    # don't spend 2-6s on a VLM round trip nobody asked for.
    return {
        "tier": 1, "answered": True, "label": result.text, "speak": result.text,
        "ocr_text": result.text, "ocr_confidence": round(result.mean_confidence, 3),
    }


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

    # For continuous streaming currency/obstacle mode, skip unsharp/dark frames if quality gate rejects.
    # Collision is deliberately NOT in this list — you can't refuse to check for a collision because the
    # user's phone shook slightly. The safety trade-off is inverted from the currency/obstacle case.
    if msg.mode in ("currency", "obstacle") and not gate.accept and (currency.classifier.ready or detect.detector.ready):
        # Also send an abstain-shaped result so the client's in-flight tracker
        # can advance to the next frame instead of blocking forever on this seq.
        await ws.send_json({
            "type": "result", "seq": msg.seq, "tier": 1, "mode": msg.mode,
            "answered": False, "speak": "",
            "latency_ms": round((time.monotonic() - t0) * 1000, 1),
        })
        return

    # ---- Tier 1 modes (with seamless VLM fallback if specialized model not installed) ----
    if msg.mode == "currency":
        if currency.classifier.ready:
            # YOLO inference is synchronous & CPU/GPU-bound — run it off the
            # event loop so we don't stall other WebSocket clients or the
            # sonar guidance stream while it runs.
            detections = await asyncio.to_thread(currency.classifier.detect, frame)
            decision = state.currency_arbiter.submit(detections)
            log.info("💵 [Currency] detections=%d denominations=%s answered=%s",
                     len(detections), decision.denominations, decision.answered)
            await ws.send_json({
                "type": "result", "seq": msg.seq, "tier": 1, "mode": "currency",
                "answered": decision.answered,
                "label": ",".join(str(d) for d in decision.denominations) if decision.denominations else None,
                "denominations": decision.denominations,
                "total_value": decision.total_value,
                "confidence": round(decision.confidence, 3),
                "detection_count": decision.detection_count,
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
                                "speak": "The currency model isn't loaded. Drop weights into models/."})
            return

    if msg.mode == "text":
        log.info("📖 [Text Mode] Processing frame (OCR ready: %s, question: %r)",
                 ocr.engine.ready, msg.question)
        try:
            result = await run_text_mode(frame, msg.question)
        except RuntimeError as e:
            log.error("OCR runtime error: %s", e)
            await ws.send_json({"type": "result", "seq": msg.seq, "tier": 1, "mode": "text",
                                "answered": False, "speak": str(e)})
            return
        result.setdefault("latency_ms", round((time.monotonic() - t0) * 1000, 1))
        await ws.send_json({"type": "result", "seq": msg.seq, "mode": "text", **result})
        return

    if msg.mode == "obstacle":
        if detect.detector.ready:
            try:
                dets = await asyncio.to_thread(detect.detector.detect, frame)
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

    if msg.mode == "collision":
        if not detect.detector.ready:
            await ws.send_json({"type": "result", "seq": msg.seq, "tier": 1, "mode": "collision",
                                "answered": False,
                                "speak": "Collision detector not available. Install ultralytics."})
            return
        try:
            threats = await asyncio.to_thread(detect.detector.scan_for_collision, frame)
        except RuntimeError as e:
            await ws.send_json({"type": "result", "seq": msg.seq, "tier": 1, "mode": "collision",
                                "answered": False, "speak": str(e)})
            return
        speak, severity = detect.detector.speak_collision(threats)
        # Cooldown-limit the audio: continuous streaming would otherwise
        # spam "chair close ahead" every 200ms while the chair sits there.
        # The visual/haptic status still updates every frame.
        speak_this_frame = None
        if severity is not None and (now - state.last_collision_hint_time) >= config.COLLISION_HINT_COOLDOWN_S:
            speak_this_frame = speak
            state.last_collision_hint_time = now
        log.info("🚧 [Collision] threats=%d severity=%s speak=%r",
                 len(threats), severity, speak_this_frame)
        await ws.send_json({
            "type": "result", "seq": msg.seq, "tier": 1, "mode": "collision",
            "answered": bool(threats),
            "collision_alert": severity,
            "detection_count": len(threats),
            "label": ", ".join(sorted({t.label for t in threats})) if threats else None,
            "speak": speak_this_frame or "",   # empty string = don't say anything this tick
            "latency_ms": round((time.monotonic() - t0) * 1000, 1),
        })
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
        state.currency_arbiter.reset()


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


@app.post("/api/stt")
async def api_stt(payload: dict):
    """Speech-to-text endpoint for voice question mode.
    Accepts: {"audio_b64": "..."} — WAV audio, base64 encoded, 16kHz mono PCM16 preferred.
    Returns: {"text": "..."}
    """
    from fastapi.responses import JSONResponse
    audio_b64 = payload.get("audio_b64", "")
    if not audio_b64:
        return JSONResponse(status_code=400, content={"error": "missing audio_b64"})
    try:
        wav_bytes = base64.b64decode(audio_b64)
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"bad base64: {e}"})
    log.info("[API/STT] Received %d bytes of audio", len(wav_bytes))
    try:
        text = await asyncio.to_thread(stt.engine.transcribe, wav_bytes)
    except stt.STTUnavailable as e:
        log.warning("[API/STT] STT unavailable: %s", e)
        return JSONResponse(status_code=503, content={"error": str(e), "text": ""})
    log.info("[API/STT] Transcribed: %r", text)
    return JSONResponse(content={"text": text})


@app.post("/api/currency")
async def api_currency(payload: dict):
    """Single-shot currency detection via the YOLO classifier.
    Accepts: {"image_b64": "..."}
    Returns: {"speak": "...", "denominations": [...], "total_value": N, "detection_count": N, "answered": bool}
    Falls back to VLM if the YOLO model isn't loaded.
    """
    from fastapi.responses import JSONResponse
    image_b64 = payload.get("image_b64", "")
    if not image_b64:
        return JSONResponse(status_code=400, content={"error": "missing image_b64"})
    try:
        frame = decode_jpeg_b64(image_b64)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

    # Gemma is the gatekeeper for "is there actually currency here?" The
    # YOLO currency model has no negative class — it confidently labels a
    # face or a bus as "100 rupees" — so we cannot trust it to decide
    # whether currency is present. Gemma reliably answers yes/no. If yes,
    # we hand off to YOLO for the exact denomination breakdown, which is
    # what YOLO is actually good at.
    ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    gemma_says_currency = True
    try:
        verdict, _ = await vlm.describe(
            jpeg.tobytes(),
            question="Is there an Indian paper currency note or coin clearly visible in this image? Answer with only the single word YES or NO.",
        )
        gemma_says_currency = "yes" in verdict.strip().lower()[:6]
        log.info("[API/CURRENCY] Gemma currency-present verdict: %r -> %s", verdict[:40], gemma_says_currency)
    except vlm.VLMUnavailable:
        gemma_says_currency = True   # no Gemma — fall through and trust YOLO

    if not gemma_says_currency:
        return JSONResponse(content={
            "answered": False, "denominations": [], "total_value": 0,
            "detection_count": 0, "speak": "No currency detected. Point the camera at the notes.",
        })

    if currency.classifier.ready:
        detections = await asyncio.to_thread(currency.classifier.detect, frame)
        if detections:
            denoms = sorted((d["denomination"] for d in detections), reverse=True)
            total = sum(denoms)
            speak = currency._speak_for(list(denoms), total)
            log.info("[API/CURRENCY] denominations=%s total=%d", denoms, total)
            return JSONResponse(content={
                "answered": True, "denominations": denoms, "total_value": total,
                "detection_count": len(detections), "speak": speak,
            })

    # Gemma confirmed currency but YOLO couldn't localise it (or isn't
    # loaded) — let Gemma read the denominations directly as a fallback.
    try:
        speak, _ = await vlm.describe(
            jpeg.tobytes(),
            question="List every Indian rupee note denomination visible and give the total in one short sentence.",
        )
    except vlm.VLMUnavailable:
        speak = "I can see currency but couldn't read the denomination."
    return JSONResponse(content={"answered": True, "speak": speak, "denominations": [], "total_value": 0})


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
    Returns: {"speak": "...", "ocr_text": "...", "ocr_confidence": ..., "answered": ..., "tier": ..., "latency_ms": ...}
    Shares its decision logic with the WebSocket "text" mode via run_text_mode().
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

    try:
        result = await run_text_mode(frame, question)
    except RuntimeError as e:
        log.error("[API/OCR] OCR error: %s", e)
        return JSONResponse(status_code=500, content={"error": str(e), "speak": "OCR processing failed."})

    result.setdefault("latency_ms", round((time.monotonic() - t0) * 1000, 1))
    log.info("[API/OCR] Response (%.0fms, tier=%s): '%s'", result["latency_ms"], result["tier"], result["speak"][:100])
    return JSONResponse(content=result)


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
