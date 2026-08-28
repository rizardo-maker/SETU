"""
Tier 2 "reason" path — a local vision-language model served by Ollama.

Deliberately not used for currency (see server/tier1/currency.py for
why). This is for open-ended description: "what's in front of me",
free-form questions, messy documents. Fluency is the point here, and
a 1-4 second wait is acceptable for an open-ended answer in a way it
isn't for "how much money am I holding".

Requires Ollama running locally (`ollama serve`, usually automatic
after install) with a vision model pulled, e.g.:
    ollama pull gemma3:4b
Runs entirely on localhost — no data leaves the machine, which is the
whole point of the offline story.
"""
from __future__ import annotations
import base64
import logging
import time

import httpx

from server import config

log = logging.getLogger("setu.vlm")


class VLMUnavailable(RuntimeError):
    pass


_active_model: str | None = None


async def get_model_name() -> str:
    global _active_model
    if _active_model:
        return _active_model
    import os

    tags: list[str] = []
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            r = await client.get(f"{config.OLLAMA_URL}/api/tags")
            if r.status_code == 200:
                tags = [m.get("name", "") for m in r.json().get("models", [])]
    except Exception:
        pass

    # Only trust SETU_VLM_MODEL if it's actually pulled — an env var left
    # over from a previous model (e.g. after `ollama rm`) must not win
    # over what's really installed, or every request 404s silently.
    wanted = os.environ.get("SETU_VLM_MODEL")
    if wanted:
        for t in tags:
            if t == wanted or t.startswith(wanted.split(":")[0]):
                _active_model = t
                return _active_model
        if not tags:
            _active_model = wanted
            return _active_model
        log.warning("SETU_VLM_MODEL=%s is not pulled in Ollama; falling back to auto-detect.", wanted)

    for candidate in ["gemma3:4b", "gemma3", "llava:latest", "llava", "moondream:latest", "moondream", "bakllava"]:
        for t in tags:
            if t == candidate or t.startswith(candidate.split(":")[0]):
                _active_model = t
                return _active_model
    if tags:
        _active_model = tags[0]
        return _active_model

    _active_model = config.OLLAMA_MODEL
    return _active_model


async def is_available() -> bool:
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            r = await client.get(f"{config.OLLAMA_URL}/api/tags")
            return r.status_code == 200
    except Exception:
        return False


async def _generate(model: str, prompt: str, system: str, images: list[str] | None = None) -> tuple[str, float]:
    """
    Shared Ollama /api/generate caller for both vision (describe) and pure-text
    (answer_from_text) reasoning. Returns (raw_response_text, latency_ms).
    Raises VLMUnavailable if Ollama isn't reachable.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "system": system,
        "stream": False,
        "keep_alive": -1,
        "options": {
            "temperature": config.VLM_TEMPERATURE,
            "num_predict": config.VLM_NUM_PREDICT,
        },
    }
    if images:
        payload["images"] = images

    log.debug("[VLM] POST %s/api/generate  model=%r  prompt=%r  image=%s",
              config.OLLAMA_URL, model, prompt[:80], "yes" if images else "no")

    t0 = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=config.VLM_TIMEOUT_S) as client:
            r = await client.post(f"{config.OLLAMA_URL}/api/generate", json=payload)
            log.debug("[VLM] Ollama responded HTTP %d in %.0fms", r.status_code, (time.monotonic() - t0) * 1000)
            if r.status_code != 200:
                log.error("[VLM] Ollama returned HTTP %d — body: %s", r.status_code, r.text[:500])
            r.raise_for_status()
            text = r.json().get("response", "").strip()
    except (httpx.ConnectError, httpx.ConnectTimeout) as e:
        log.error("[VLM] Connection failed to %s: %s", config.OLLAMA_URL, e)
        raise VLMUnavailable(
            f"Can't reach Ollama at {config.OLLAMA_URL}. Is it running? (`ollama serve`)"
        ) from e
    latency_ms = (time.monotonic() - t0) * 1000
    log.debug("[VLM] Response (%.0fms): '%s'", latency_ms, text[:200])
    return text, latency_ms


async def describe(jpeg_bytes: bytes, question: str | None = None, system_override: str | None = None) -> tuple[str, float]:
    """
    Returns (speak_text, latency_ms). Raises VLMUnavailable if Ollama
    isn't reachable.
    """
    model = await get_model_name()
    img_b64 = base64.b64encode(jpeg_bytes).decode("ascii")

    user_prompt = question.strip() if question and question.strip() else "Describe what is in front of me clearly in 1 or 2 sentences."

    log.debug("[VLM] describe() — model=%r, image_size=%d bytes, question=%r",
              model, len(jpeg_bytes), question)

    text, latency_ms = await _generate(
        model, user_prompt, system_override or config.VLM_SYSTEM_PROMPT, images=[img_b64]
    )

    if not text or text.upper().startswith("UNCLEAR"):
        return config.PHRASES["vlm_unclear"], latency_ms
    return text, latency_ms


def _truncate_ocr_text(text: str, max_chars: int) -> str:
    """Truncate on a line boundary so we don't cut a word/sentence mid-way."""
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    last_break = max(truncated.rfind("\n"), truncated.rfind(". "))
    if last_break > max_chars // 2:   # only trust the break point if it isn't absurdly early
        truncated = truncated[:last_break + 1]
    return truncated.strip()


async def answer_from_text(
    extracted_text: str,
    question: str | None = None,
    ocr_confidence: float | None = None,
    system_override: str | None = None,
) -> tuple[str, float]:
    """
    Given OCR-extracted text and an optional question, queries the local model
    (pure text reasoning, no image payload) to reason over the text.
    Returns (speak_text, latency_ms). Raises VLMUnavailable if Ollama isn't reachable.
    """
    model = await get_model_name()
    q_str = question.strip() if question and question.strip() else "Summarize this text in 1-2 sentences."

    truncated = _truncate_ocr_text(extracted_text, config.OCR_TEXT_MAX_CHARS)
    was_truncated = len(truncated) < len(extracted_text)

    confidence_note = ""
    if ocr_confidence is not None and ocr_confidence < config.OCR_MIN_CONFIDENCE + 0.15:
        # Close to (but above) the reject threshold — text is being reasoned over,
        # but the model should hedge rather than assert facts confidently.
        confidence_note = "Note: this OCR text is low-confidence and may contain errors. "

    user_prompt = (
        f"{confidence_note}Here is text extracted from an image via OCR"
        f"{' (truncated)' if was_truncated else ''} (it may contain noise or errors):\n\n"
        f"{truncated}\n\n"
        f"Question: {q_str}"
    )

    log.debug("[VLM] answer_from_text() — model=%r, text_len=%d (truncated=%s), question=%r, ocr_confidence=%s",
              model, len(extracted_text), was_truncated, question, ocr_confidence)

    text, latency_ms = await _generate(
        model, user_prompt, system_override or config.OCR_REASONING_SYSTEM_PROMPT
    )

    if not text or text.upper().startswith("UNCLEAR"):
        return config.PHRASES["vlm_unclear"], latency_ms
    return text, latency_ms


async def warm_up() -> None:
    """Call once at server startup so the first real request isn't the
    one paying the cold-load cost. Failure here is logged, not fatal —
    Tier 1 must keep working even if Ollama never comes up."""
    try:
        available = await is_available()
        if not available:
            log.warning("Ollama not reachable at %s — Tier 2 (scene description) "
                        "will be unavailable until it's running.", config.OLLAMA_URL)
            return
        model = await get_model_name()
        log.info("[VLM] Warming up model=%r ...", model)
        # A tiny 1x1 pixel warms the model into VRAM without a real image.
        tiny_jpeg = base64.b64decode(
            "/9j/4AAQSkZJRgABAQEAYABgAAD/2wBDAAMCAgICAgMCAgIDAwMDBAYEBAQEBAgGBgUGCQgKCgkI"
            "CQkKDA8MCgsOCwkJDRENDg8QEBEQCgwSExIQEw8QEBD/wAALCAABAAEBAREA/8QAFAABAAAAAAAA"
            "AAAAAAAAAAAAAAgQAf/EABQQAQAAAAAAAAAAAAAAAAAAAAD/2gAIAQEAAD8AVN//2Q=="
        )
        await describe(tiny_jpeg)
        log.info("Ollama (%s) warmed up.", model)
    except Exception as e:
        log.warning("Ollama warm-up failed (non-fatal): %s", e)
