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


async def describe(jpeg_bytes: bytes, question: str | None = None, system_override: str | None = None) -> tuple[str, float]:
    """
    Returns (speak_text, latency_ms). Raises VLMUnavailable if Ollama
    isn't reachable.
    """
    model = await get_model_name()
    img_b64 = base64.b64encode(jpeg_bytes).decode("ascii")

    if question and question.strip():
        user_prompt = question.strip()
    else:
        user_prompt = "Describe what is in front of me clearly in 1 or 2 sentences."

    log.debug("[VLM] describe() — model=%r, image_size=%d bytes, question=%r",
              model, len(jpeg_bytes), question)

    payload = {
        "model": model,
        "prompt": user_prompt,
        "system": system_override or config.VLM_SYSTEM_PROMPT,
        "images": [img_b64],
        "stream": False,
        "keep_alive": -1,
        "options": {
            "temperature": config.VLM_TEMPERATURE,
            "num_predict": config.VLM_NUM_PREDICT,
        },
    }

    log.debug("[VLM] POST %s/api/generate  model=%r  prompt=%r  image=yes",
              config.OLLAMA_URL, model, user_prompt[:80])

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
