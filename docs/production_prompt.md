# Production-grade prompt — OCR + Gemma reasoning for "text" mode

Keep this for later. This is the spec I'll implement myself once the MVP direction is validated — not meant for Antigravity.

---

## Goal

Upgrade "text" mode from raw-OCR-readback to a proper **extract → reason** pipeline: OCR pulls text out of the frame, Gemma3:4b reasons over that extracted text against the user's actual question, and the result degrades gracefully at every failure point (no OCR backend, empty OCR output, Ollama down, garbled/low-confidence text, no question asked).

## Scope of changes

### 1. `server/tier1/ocr.py`
- Extend `OCREngine.read()` (or add `read_with_confidence()`) to return not just the joined text string but also a **mean confidence score** where the backend supports it:
  - PaddleOCR gives per-line confidence (`_conf` in the existing tuple unpacking at line 58 — currently discarded). Aggregate to a mean.
  - pytesseract: use `image_to_data()` instead of `image_to_string()` to get per-word confidences, average them (ignore -1/negative confidence entries tesseract emits for whitespace).
  - Return type becomes a small dataclass or named tuple: `OCRResult(text: str, mean_confidence: float, backend: str)`.
- Add a low-confidence threshold (config-driven, see below) — if mean confidence is below it, treat as "no reliable text" rather than passing garbage into Gemma's reasoning step (garbage-in-garbage-out wastes a ~2-6s round trip and produces a hallucinated answer).
- Keep this module's existing "lazy import, report unavailable rather than crash" philosophy intact — no behavior change to `ready`/backend selection logic.

### 2. `server/tier2/vlm.py`
- Add `answer_from_text(extracted_text: str, question: str | None, ocr_confidence: float | None = None) -> tuple[str, float]`:
  - Text-only Ollama `/api/generate` call (no `images` field) using the same resolved model as `describe()`.
  - System prompt distinct from `VLM_SYSTEM_PROMPT` (that one is vision-scene-description-flavored) — add a new `config.OCR_REASONING_SYSTEM_PROMPT` tuned for: "you are given noisy OCR text, not a real document; correct obvious OCR errors silently if confident, don't invent facts not present in the text, if the question can't be answered from the given text say so plainly, keep answers spoken-language short."
  - If `ocr_confidence` is below threshold, prepend a note into the prompt telling the model the OCR quality was poor, so it hedges appropriately instead of asserting wrong facts confidently.
  - Truncate `extracted_text` to a safe token budget before sending (e.g. ~2000 chars) — OCR on a cluttered scene can return far more raw text than any reasonable question needs, and Gemma3:4b's context/latency both suffer from stuffing all of it in blindly. Truncate on line boundaries, not mid-word.
  - Return the same `(text, latency_ms)` shape everything else uses, and raise `VLMUnavailable` on connect failure exactly like `describe()` does — no special-cased exception type for this path.
  - Add logging parity with the existing `describe()` debug logs (`log.debug` at request/response boundaries) so this path is equally debuggable via the `setu.vlm` logger already wired to DEBUG in `main.py`.

### 3. `server/config.py`
- `OCR_MIN_CONFIDENCE = float(os.environ.get("SETU_OCR_MIN_CONFIDENCE", "0.35"))` (tune empirically; PaddleOCR/tesseract confidence scales differ — verify both report 0..1 or normalize in `ocr.py`)
- `OCR_TEXT_MAX_CHARS = 2000`
- `OCR_REASONING_SYSTEM_PROMPT` as described above — write full text, don't stub it
- Add a new `PHRASES` entry, e.g. `"ocr_low_confidence": "The text is too unclear to read reliably. Try moving closer or improving the lighting."`, matching the existing terse, second-person, actionable phrasing style already used for `abstain_*` entries.

### 4. `server/main.py` — `msg.mode == "text"` branch (currently ~line 166-191)
Restructure the `ocr.engine.ready` branch into this decision tree:
- Run OCR → get `(text, mean_confidence)`.
- If `not text`: existing "No text found. Try moving closer." (`answered: False`), tier 1, unchanged.
- elif `mean_confidence < config.OCR_MIN_CONFIDENCE`: speak `config.PHRASES["ocr_low_confidence"]`, `answered: False`, tier 1, log the raw text + confidence at debug level (so the failure is diagnosable, not silent) but never speak the raw low-confidence text to the user as if it were reliable.
- elif `msg.question` present: call `vlm.answer_from_text(text, msg.question, mean_confidence)`, tier 2 (a Gemma call happened), speak the Gemma answer, still include the raw OCR text under a distinct JSON field (`"ocr_text"`) so the client can display or log it separately.
- else (no question, confidence is fine): keep current behavior — speak raw OCR text directly, tier 1, no Gemma call. Reading text back verbatim with no question is a legitimate use case (e.g. "just read me this sign") and shouldn't cost an extra 2-6s VLM round trip when it's not needed.
- The VLM-only fallback path (no OCR backend installed at all) stays as today, but reuse the same "did the user ask a question or not" branching for consistency — if there's no OCR backend, degrade straight to `vlm.describe()` with the "read text" instruction prompt regardless of whether a question was asked (current behavior), since there's no extracted text to separate reasoning from in that path anyway.
- Every branch must still end in a spoken `"speak"` field — no silent drops, per the file's existing top-of-file design comment.

### 5. `server/ws_protocol.py`
- Add `ocr_text: Optional[str] = None` and `ocr_confidence: Optional[float] = None` to `ServerResult`, so the client can distinguish "this is what OCR actually saw" from "this is what Gemma concluded" — useful for debugging and for a future UI that shows both.

### 6. `client/app.js`
- Extend the question-payload inclusion (currently gated to `this.mode === "question"` in `_sendFrame()`) to also include `payload.question` when `this.mode === "text"` and the question input has a non-empty value, so a user can type a targeted question ("what's the expiry date") while in text-reading mode.
- Update `_renderResult()` to show both `ocr_text` and `speak` when both are present and differ, so a sighted collaborator watching the screen can sanity-check Gemma's answer against the raw OCR.
- No change to `_captureFrame()` / `_sendVLMRequest()` — text mode continues to use the WebSocket streaming path (`_sendFrame()`), unlike scene/question which use `/api/vlm`, since text mode is designed for continuous point-and-hold use, not single-shot capture. (If you'd rather unify text mode onto the single-shot `/api/vlm`-style REST path too, that's a bigger decision — flag it back to me rather than deciding silently, since it changes the interaction model from "hold and it keeps reading" to "tap once.")

### 7. Testing / validation before calling this done
- Test against: a clean printed label (should get answered confidently), a blurry/angled photo of small text (should trigger `ocr_low_confidence` gracefully, not a hallucinated Gemma answer), a scene with text plus a question that's unanswerable from that text (Gemma should say so, not invent an answer), and no-question "just read it" usage (should stay fast, tier 1, no Gemma round-trip).
- Confirm latency budget: OCR itself + optional Gemma call. If OCR is ~200-800ms (PaddleOCR on CPU) and Gemma adds another 2-6s when a question is asked, that compound latency needs to be reflected honestly in client UI state ("Reading text..." → "Thinking about your question...") rather than a single opaque spinner, since silence for 5+ seconds reads as broken.

## Explicitly out of scope for this pass
- No changes to PaddleOCR/tesseract backend selection or install story.
- No multi-language OCR handling beyond what PaddleOCR already does with `lang="en"` — that's a separate, larger piece of work if needed later.
- No caching/memoization of repeated OCR+Gemma calls on the same frame.
