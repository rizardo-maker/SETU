# MVP prompt — OCR + Gemma reasoning for "text" mode

Copy everything below into Antigravity.

---

In this SETU project, "text" mode currently either runs raw OCR (PaddleOCR/tesseract via `server/tier1/ocr.py`) and speaks the raw extracted text verbatim, or — if no OCR backend is installed — falls back to asking the Gemma vision model (`server/tier2/vlm.py`, `vlm.describe()`) to read the image directly with no real OCR involved.

I want a new behavior: **run OCR to extract the text, then pass that extracted text to Gemma (via Ollama, `gemma3:4b`) so it can reason about it and answer the user's actual question**, instead of just reading text back verbatim. Example: user is in "text" mode, asks "what's the total on this receipt", OCR extracts all the receipt's raw text (noisy, unordered lines), and Gemma is given that raw text plus the question and answers "The total is $42.50" instead of dumping the whole receipt.

Build this as an MVP — quick and functional, not production-hardened (I'll harden it later):

1. In `server/tier1/ocr.py`, keep the existing `OCREngine.read()` as-is (returns raw extracted text string).

2. In `server/tier2/vlm.py`, add a new function `answer_from_text(extracted_text: str, question: str | None) -> tuple[str, float]` that:
   - Builds a plain-text prompt like: `f"Here is text extracted from an image via OCR (it may contain noise or errors):\n\n{extracted_text}\n\nQuestion: {question or 'Summarize this text in 1-2 sentences.'}"`
   - Sends it to Ollama's `/api/generate` endpoint using the already-resolved model from `get_model_name()` — **text-only, no `images` field in the payload** (this is a pure text reasoning call, not vision)
   - Reuses `config.VLM_TEMPERATURE`, `config.VLM_NUM_PREDICT`, `config.VLM_TIMEOUT_S` for the request options, matching the existing `describe()` function's style
   - Returns `(text, latency_ms)` same shape as `describe()`

3. In `server/main.py`, inside the `if msg.mode == "text":` block (around line 166), change the `ocr.engine.ready` branch so that after `text = ocr.engine.read(frame)` succeeds and returns non-empty text:
   - If `msg.question` is set (user asked something specific), call `vlm.answer_from_text(text, msg.question)` and speak that answer instead of the raw OCR text
   - If no question was given, keep current behavior (speak the raw OCR text)
   - Send back `tier: 1` still, but you can add a field like `"ocr_text": text` alongside `"speak"` in the JSON result so the raw OCR output is still available to the client if needed
   - If OCR returns empty text, keep the current "No text found. Try moving closer." message — don't call Gemma in that case

4. No changes needed to the VLM-fallback branch (when no OCR backend is installed) — that can keep working as it does now for the MVP.

5. Don't touch the client (`client/app.js`) — the existing `question` field already gets sent for "text" mode frames if the user has typed something into the question input, since `payload.question` is included whenever `this.mode === "question"` currently... check `_sendFrame()` and extend the question payload inclusion to also fire for `mode === "text"` if a question is present, so a user can type "what's the price" while in text mode and have it delivered.

Keep it simple — no retries, no elaborate prompt engineering, no new config flags. Just get OCR text into Gemma's text-reasoning path and back out again correctly.
