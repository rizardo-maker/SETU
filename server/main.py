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


@app.exception_handler(Exception)
async def _unhandled_exception_handler(request, exc: Exception):
    """
    Last-resort safety net for every REST route. Without this, an
    unexpected exception anywhere in a voice-command endpoint (a model
    library throwing on an edge-case input, a network hiccup mid-call)
    propagates as a bare 500 with no JSON body — the client's
    `await res.json()` then throws its own SyntaxError, and the whole
    voice command silently dies with nothing spoken and nothing logged
    client-side. This guarantees every request gets back well-formed
    JSON with a "speak" field the client can always fall back to,
    fulfilling the file's top-of-file rule: every failure path ends in
    a spoken message, never a silent drop.
    """
    from fastapi.responses import JSONResponse
    log.error("[UNHANDLED] %s %s -> %s", request.method, request.url.path, exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": str(exc),
            "speak": "Something went wrong. Please try again.",
            "answered": False,
        },
    )


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


async def run_text_mode(frame: np.ndarray, question: str | None = None) -> dict:
    """
    OCR + LLM Refinement Pipeline (RapidOCR + Local Gemma-3 LLM):
    1. Extracts high-accuracy text via RapidOCR.
    2. Sends the raw extracted text to the local LLM (Ollama) to refine OCR typos,
       reconstruct broken lines, and generate a clear, natural summary.
    3. Returns the output in both text and spoken speech formats.
    """
    if not ocr.engine.ready:
        return {"tier": 1, "answered": False, "speak": "No OCR backend installed.", "text": "No OCR backend installed."}

    result = await asyncio.to_thread(ocr.engine.read_with_confidence, frame)
    log.info("📖 [OCR] backend=%s confidence=%.2f (%d chars): %r",
              result.backend, result.mean_confidence, len(result.text), result.text[:120])

    if not result.text:
        return {"tier": 1, "answered": False, "speak": "No text detected in view.", "text": "No text detected in view."}

    if result.mean_confidence < 0.20:
        return {
            "tier": 1, "answered": False,
            "speak": "Text is too blurry or unclear. Please hold steady.",
            "text": "Text is too blurry or unclear. Please hold steady.",
            "ocr_text": result.text, "ocr_confidence": round(result.mean_confidence, 3),
        }

    # Step 2: Send extracted text to LLM to refine and correct OCR output
    final_text = result.text
    tier = 1

    if await vlm.is_available():
        try:
            llm_question = question.strip() if question and question.strip() else (
                "Read and cleanly state what this text says in 1-2 concise sentences."
            )
            refined_text, lat = await vlm.answer_from_text(
                extracted_text=result.text,
                question=llm_question,
                ocr_confidence=result.mean_confidence,
                system_override=(
                    "You are a direct text reader for a blind assistant. "
                    "Read the provided OCR text, correct obvious character typos, and output only the refined, natural text. "
                    "Do not include any conversational filler or meta-commentary."
                ),
            )
            if refined_text and not refined_text.upper().startswith("UNCLEAR"):
                log.info("🧠 [LLM-Refined OCR] (%.1fms): %r", lat, refined_text[:120])
                final_text = refined_text
                tier = 2
        except Exception as e:
            log.warning("LLM OCR refinement failed, using raw OCR text: %s", e)

    return {
        "tier": tier,
        "answered": True,
        "label": final_text,
        "text": final_text,
        "speak": final_text,
        "ocr_text": result.text,
        "ocr_confidence": round(result.mean_confidence, 3),
        "chunks": result.chunks,
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
    except Exception as e:
        # Belt-and-suspenders: whisper's internals have known edge-case
        # crashes on malformed/near-silent audio (see stt.py). Whatever
        # the cause, "question" mode must degrade to "I didn't hear a
        # question" rather than propagate a raw 500 to the browser.
        log.error("[API/STT] Unexpected transcription error: %s", e, exc_info=True)
        return JSONResponse(content={"text": ""})
    log.info("[API/STT] Transcribed: %r", text)
    return JSONResponse(content={"text": text})


@app.post("/api/currency")
async def api_currency(payload: dict):
    """Dedicated single-shot currency detection endpoint.
    Accepts: {"image_b64": "..."}
    Returns: {"speak": "...", "denominations": [...], "total_value": N, "detection_count": N, "answered": bool}
    """
    from fastapi.responses import JSONResponse
    image_b64 = payload.get("image_b64", "")
    if not image_b64:
        return JSONResponse(status_code=400, content={"error": "missing image_b64"})
    try:
        frame = decode_jpeg_b64(image_b64)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

    t0 = time.monotonic()

    # 1. High-Precision YOLO Currency Detection (95%+ Confidence)
    if currency.classifier.ready:
        try:
            detections = await asyncio.to_thread(currency.classifier.detect, frame, conf_threshold=config.CURRENCY_CONF_FLOOR)
            if detections:
                min_conf = min(float(d["confidence"]) for d in detections)
                denoms = sorted((d["denomination"] for d in detections), reverse=True)
                total = sum(denoms)
                speak = currency._speak_for(list(denoms), total)
                log.info(f"[API/CURRENCY] High Accuracy YOLO detected: {denoms} (Total: {total}, min_conf={min_conf:.2f})")
                return JSONResponse(content={
                    "answered": True, "denominations": denoms, "total_value": total,
                    "detection_count": len(detections), "confidence": min_conf, "speak": speak,
                    "latency_ms": round((time.monotonic() - t0) * 1000, 1),
                })
        except Exception as e:
            log.warning("[API/CURRENCY] YOLO error: %s", e)

    # 2. Check for lower confidence / borderline detections (abstain if doubtful)
    if currency.classifier.ready:
        try:
            borderline = await asyncio.to_thread(currency.classifier.detect, frame, conf_threshold=0.35)
            if borderline:
                log.info("[API/CURRENCY] Borderline detection (conf < 0.70) -> Safe Abstention")
                return JSONResponse(content={
                    "answered": False, "denominations": [], "total_value": 0,
                    "detection_count": len(borderline),
                    "speak": "I am not completely sure. Please hold the note a bit steadier in good lighting.",
                    "latency_ms": round((time.monotonic() - t0) * 1000, 1),
                })
        except Exception as e:
            log.warning("[API/CURRENCY] YOLO borderline check error: %s", e)

    # 3. Local Vision-Language Model Verification
    ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        return JSONResponse(status_code=500, content={"error": "JPEG encode failed"})

    try:
        speak, latency_ms = await vlm.describe(
            jpeg.tobytes(),
            question=(
                "Identify any paper currency notes or coins visible in this image (such as Indian Rupees or other currency). "
                "State the exact denomination of each note clearly and calculate the total amount in 1 concise sentence (e.g. '500 rupees' or 'Two 100 rupee notes, total 200 rupees'). "
                "If the note is too blurry or unclear to be certain, say 'I am not completely sure. Please hold the note steadier.' "
                "If no currency or money is visible at all, say exactly 'No currency detected.'"
            ),
        )
        answered = "no currency" not in speak.lower() and "unclear" not in speak.lower() and "not sure" not in speak.lower() and "not completely sure" not in speak.lower()
        log.info("[API/CURRENCY] VLM Response: %r (answered=%s)", speak, answered)
        return JSONResponse(content={
            "answered": answered,
            "speak": speak,
            "denominations": [],
            "total_value": 0,
            "latency_ms": round(latency_ms, 1),
        })
    except vlm.VLMUnavailable:
        return JSONResponse(content={
            "answered": False,
            "speak": "Currency model is not available.",
            "denominations": [],
            "total_value": 0,
            "latency_ms": round((time.monotonic() - t0) * 1000, 1),
        })


@app.post("/api/objects")
async def api_objects(payload: dict):
    """Dedicated Object Detection endpoint.
    Identifies surrounding objects, furniture, people, items, tools (ignoring currency notes).
    """
    from fastapi.responses import JSONResponse
    image_b64 = payload.get("image_b64", "")
    if not image_b64:
        return JSONResponse(status_code=400, content={"error": "missing image_b64"})
    try:
        frame = decode_jpeg_b64(image_b64)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

    t0 = time.monotonic()
    detected_parts: list[str] = []

    # 1. Run local YOLO general object detector
    if detect.detector.ready:
        try:
            o_dets = await asyncio.to_thread(detect.detector.detect, frame)
            counts: dict[str, int] = {}
            for od in o_dets:
                label = od.label
                # Exclude any generic labels that might represent currency
                if label.lower() in ("currency", "money", "banknote"):
                    continue
                counts[label] = counts.get(label, 0) + 1
            for lbl, count in sorted(counts.items(), key=lambda x: -x[1]):
                detected_parts.append(f"{count} {lbl}s" if count > 1 else f"a {lbl}")
        except Exception as e:
            log.warning("[API/OBJECTS] Object YOLO error: %s", e)

    if detected_parts:
        if len(detected_parts) == 1:
            joined = detected_parts[0]
        elif len(detected_parts) == 2:
            joined = f"{detected_parts[0]} and {detected_parts[1]}"
        else:
            joined = ", ".join(detected_parts[:-1]) + f", and {detected_parts[-1]}"
        speak = f"I can see: {joined}."
        log.info("[API/OBJECTS] Detected (YOLO Objects): %r", speak)
        return JSONResponse(content={
            "answered": True,
            "speak": speak,
            "latency_ms": round((time.monotonic() - t0) * 1000, 1),
        })

    # 2. VLM Fallback for general objects (strictly without currency)
    ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if ok and await vlm.is_available():
        try:
            speak, latency_ms = await vlm.describe(
                jpeg.tobytes(),
                question=(
                    "Identify the key physical objects, furniture, tools, electronic devices, people, or items visible in front of the camera. "
                    "List what you see in 1 concise spoken sentence (e.g. 'I see a chair, a laptop, and a water bottle'). "
                    "Do not mention any currency, money, or banknotes."
                ),
            )
            log.info("[API/OBJECTS] VLM detected: %r", speak)
            return JSONResponse(content={
                "answered": True,
                "speak": speak,
                "latency_ms": round((time.monotonic() - t0) * 1000, 1),
            })
        except Exception as e:
            log.warning("[API/OBJECTS] VLM query failed: %s", e)

    return JSONResponse(content={
        "answered": False,
        "speak": "No objects clearly detected.",
        "latency_ms": round((time.monotonic() - t0) * 1000, 1),
    })


@app.post("/api/proximity")
@app.post("/api/collision")
async def api_proximity(payload: dict):
    """Proximity & Collision Detection endpoint (from prox/ pipeline).
    Scans for physical hazards and obstacles on the walking path.
    Returns: {"answered": bool, "collision_alert": "warn"|"urgent"|None, "threats": [...], "speak": "..."}
    """
    from fastapi.responses import JSONResponse
    image_b64 = payload.get("image_b64", "")
    if not image_b64:
        return JSONResponse(status_code=400, content={"error": "missing image_b64"})
    try:
        frame = decode_jpeg_b64(image_b64)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

    t0 = time.monotonic()
    if not detect.detector.ready:
        return JSONResponse(content={
            "answered": False,
            "collision_alert": None,
            "threats": [],
            "speak": "Collision detector not available.",
            "latency_ms": round((time.monotonic() - t0) * 1000, 1),
        })

    try:
        threats = await asyncio.to_thread(detect.detector.scan_for_collision, frame)
    except Exception as e:
        log.error("[API/PROXIMITY] Error: %s", e)
        return JSONResponse(content={"answered": False, "speak": f"Proximity error: {e}"})

    speak, severity = detect.detector.speak_collision(threats)
    threat_list = [
        {
            "label": t.label,
            "confidence": round(t.confidence, 3),
            "area_fraction": round(t.area_fraction, 3),
            "distance_meters": t.distance_meters,
            "direction": t.direction,
            "severity": t.severity,
            "bbox": t.bbox,
        }
        for t in threats
    ]

    return JSONResponse(content={
        "answered": bool(threats),
        "collision_alert": severity,
        "severity": severity,
        "threats": threat_list,
        "detection_count": len(threats),
        "speak": speak,
        "latency_ms": round((time.monotonic() - t0) * 1000, 1),
    })


@app.post("/api/navigate")
async def api_navigate(payload: dict):
    """
    Core SRS Feature: Navigate Mode.
    Finds room/sign targets (e.g. 'C-214', 'Exit', 'Office', 'Washroom')
    while enforcing the strict priority hierarchy:
      1. Critical / Path-blocking obstacle warnings (e.g. "Chair ahead. Move slightly left.")
      2. Target detection (e.g. "C-214 detected on your right.")
      3. Searching status (e.g. "Searching for C-214. Path is clear.")
    """
    from fastapi.responses import JSONResponse
    image_b64 = payload.get("image_b64", "")
    target = payload.get("target", "").strip()
    if not image_b64:
        return JSONResponse(status_code=400, content={"error": "missing image_b64"})
    if not target:
        target = "signboard"

    try:
        frame = decode_jpeg_b64(image_b64)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

    t0 = time.monotonic()

    # Priority 1: Check for critical or path-blocking obstacles
    obstacle_warning = None
    if detect.detector.ready:
        threats = await asyncio.to_thread(detect.detector.scan_for_collision, frame)
        if threats:
            speak_obs, sev = detect.detector.speak_collision(threats)
            if sev == "urgent":
                obstacle_warning = f"Stop! {threats[0].label} right in front of you."
            elif sev == "warn":
                obstacle_warning = f"{threats[0].label.capitalize()} ahead. Move carefully."

    # Priority 2: Search for target room / sign in frame using RapidOCR
    target_match = None
    if ocr.engine.ready:
        target_match = await asyncio.to_thread(ocr.engine.find_target, frame, target)

    # Assemble prioritized voice response
    if obstacle_warning:
        if target_match and target_match.get("found"):
            speak = f"{obstacle_warning} {target} detected {target_match['direction']}."
        else:
            speak = obstacle_warning
        answered = True
    elif target_match and target_match.get("found"):
        speak = f"{target} detected {target_match['direction']}."
        answered = True
    else:
        speak = f"Searching for {target}. Path is clear."
        answered = False

    return JSONResponse(content={
        "answered": answered,
        "target": target,
        "target_found": bool(target_match and target_match.get("found")),
        "obstacle_warning": obstacle_warning,
        "direction": target_match.get("direction") if target_match else None,
        "speak": speak,
        "latency_ms": round((time.monotonic() - t0) * 1000, 1),
    })


@app.post("/api/detect")
async def api_detect(payload: dict):
    """Single-shot proximity/collision scan via the general YOLO detector.
    Accepts: {"image_b64": "..."}
    Returns: {"speak": "...", "answered": bool, "collision_alert": "warn"|"urgent"|None, "detection_count": N}

    On-demand only — see server/tier1/detect.py's `scan_for_collision()`
    for the hazard-class list and area-fraction thresholds. This used to
    run continuously over the WebSocket at ~3fps as the app's default
    idle state; it's now invoked explicitly (voice "detect" / tap), same
    shape as currency/describe/read, so the user controls when a scan
    happens instead of it always running in the background.
    """
    from fastapi.responses import JSONResponse
    image_b64 = payload.get("image_b64", "")
    if not image_b64:
        return JSONResponse(status_code=400, content={"error": "missing image_b64"})
    try:
        frame = decode_jpeg_b64(image_b64)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

    if not detect.detector.ready:
        return JSONResponse(content={
            "answered": False, "collision_alert": None, "detection_count": 0,
            "speak": "Proximity detection isn't available. Install ultralytics.",
        })

    try:
        threats = await asyncio.to_thread(detect.detector.scan_for_collision, frame)
    except RuntimeError as e:
        return JSONResponse(status_code=500, content={"error": str(e), "speak": "Proximity scan failed."})

    speak, severity = detect.detector.speak_collision(threats)
    log.info("[API/DETECT] threats=%d severity=%s speak=%r", len(threats), severity, speak)
    return JSONResponse(content={
        "answered": bool(threats),
        "collision_alert": severity,
        "detection_count": len(threats),
        "label": ", ".join(sorted({t.label for t in threats})) if threats else None,
        "speak": speak,
    })


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


@app.post("/api/learn/explain")
async def api_learn_explain(payload: dict):
    """Explains a topic simply and conversationally for blind learners."""
    from fastapi.responses import JSONResponse
    topic = payload.get("topic", "Virtual Memory")
    t0 = time.monotonic()
    
    if await vlm.is_available():
        try:
            model = await vlm.get_model_name()
            prompt = (
                f"You are SETU Learn, an audio-first tutor for blind students. "
                f"Explain the concept '{topic}' in 1 to 2 clear, simple, plain-English sentences without technical jargon or bullet points. "
                f"Make it immediately intuitive to listen to."
            )
            raw, _ = await vlm._generate(model, prompt, "You are a concise, helpful tutor for blind students.")
            return JSONResponse(content={
                "topic": topic,
                "speak": raw.strip(),
                "latency_ms": round((time.monotonic() - t0) * 1000, 1)
            })
        except Exception as e:
            log.warning("Learn explain error: %s", e)
            
    # Fallback explanation
    fallback_text = f"{topic} allows your computer to use secondary storage as extra RAM when physical memory runs low."
    return JSONResponse(content={
        "topic": topic,
        "speak": fallback_text,
        "latency_ms": round((time.monotonic() - t0) * 1000, 1)
    })


@app.post("/api/learn/ask")
async def api_learn_ask(payload: dict):
    """Answers student questions on the current learning topic."""
    from fastapi.responses import JSONResponse
    question = payload.get("question", "What is a page fault?")
    topic = payload.get("topic", "Virtual Memory")
    t0 = time.monotonic()
    
    if await vlm.is_available():
        try:
            model = await vlm.get_model_name()
            prompt = (
                f"Topic: {topic}\n"
                f"Student Question: {question}\n\n"
                f"Answer the student's question directly in 1-2 conversational spoken sentences."
            )
            raw, _ = await vlm._generate(model, prompt, "You are SETU Learn, an audio tutor for blind students.")
            return JSONResponse(content={
                "question": question,
                "speak": raw.strip(),
                "latency_ms": round((time.monotonic() - t0) * 1000, 1)
            })
        except Exception as e:
            log.warning("Learn ask error: %s", e)

    return JSONResponse(content={
        "question": question,
        "speak": "A page fault happens when the needed data page is not currently in physical RAM, so the operating system retrieves it from disk.",
        "latency_ms": round((time.monotonic() - t0) * 1000, 1)
    })


@app.post("/api/learn/quiz")
async def api_learn_quiz(payload: dict):
    """Generates a quick audio quiz question for active recall."""
    from fastapi.responses import JSONResponse
    topic = payload.get("topic", "Virtual Memory")
    t0 = time.monotonic()
    
    if await vlm.is_available():
        try:
            model = await vlm.get_model_name()
            prompt = (
                f"Create 1 quick True or False quiz question about '{topic}'. "
                f"Format: 'True or False: [question]. Think about it and tap to answer.'"
            )
            raw, _ = await vlm._generate(model, prompt, "You are a concise quiz tutor.")
            return JSONResponse(content={
                "topic": topic,
                "quiz": raw.strip(),
                "speak": raw.strip(),
                "latency_ms": round((time.monotonic() - t0) * 1000, 1)
            })
        except Exception as e:
            log.warning("Learn quiz error: %s", e)

    return JSONResponse(content={
        "topic": topic,
        "quiz": "True or False: Virtual memory makes your computer think it has more physical RAM than it actually does.",
        "speak": "True or False: Virtual memory makes your computer think it has more physical RAM than it actually does. Think about it and speak your answer.",
        "latency_ms": round((time.monotonic() - t0) * 1000, 1)
    })


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
