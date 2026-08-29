# SETU — Pitch Narrative
### Team 31, Gryffindor · RGUKT Srikakulam · Problem Statement #4

This is the talk track, not the slides — what to actually say, in what order, and why each part is there. Use it to rehearse, not to read aloud.

---

## 1. The Open (30 seconds)

Start with the desktop demo running live, not a slide. Say this while it's on screen:

> "This is SETU, running right now on a normal laptop. No Wi-Fi. No hotspot. Watch —"

Turn off Wi-Fi visibly. Then say "currency" and show a note being counted, or "read" on a printed page. The point you're making without saying it: **this is not a demo that only works because the venue has good Wi-Fi.**

Then state the problem in one sentence:

> "4.95 crore people in India live with vision impairment — the largest population of any country — and almost every assistive app built for them assumes a cloud connection they don't reliably have."

That's the whole hook. Don't over-explain it. Move to the desktop proof.

---

## 2. The Desktop Build Is the Proof, Not a Placeholder

This is the part judges most often get wrong when they hear "desktop app" — they assume it's a prototype standing in for the real product. Reframe it explicitly:

> "We built this on desktop first, on purpose. A desktop is the hardest environment to fake offline-ness in — there's no 'it'll work fine on the phone's NPU' excuse. If it runs fully offline on a MacBook Air, the architecture is proven, and porting it to a phone is an engineering task, not a research question."

Say what "completely offline" actually means here, concretely:
- No API keys anywhere in the code.
- No network call at inference time — verified by literally disconnecting Wi-Fi during testing.
- Every model — vision, OCR, speech-to-text, text-to-speech — runs as a local process on the same machine as the camera.

**Judges are trained to be suspicious of "we'll deploy it to X later" claims.** The way to defuse that is to show the boundary of what you built vs. what you're proposing, clearly and without inflating either side. Say the sentence: *"Everything I'm about to show you is running, not planned."* Then don't claim anything as running that isn't.

---

## 3. What It Actually Does (keep this tight — 5 modes, one line each)

Don't read a feature list. Say it as five spoken commands, because that's the actual user experience:

| Say this | What happens | Why it matters for a blind user |
|---|---|---|
| **"currency"** | Detects Indian notes, speaks the total | Cash handling today needs a sighted second person |
| **"describe"** | Narrates the scene in front of the camera | Basic spatial awareness — what room, what's ahead |
| **"read"** | Reads signs, labels, documents aloud | Medicine labels, notices, printed forms |
| **"question"** | Free-form spoken question about what the camera sees | Handles anything the 4 fixed modes don't anticipate |
| **"detect"** | Warns of large obstacles close in the path | Collision avoidance while walking |

One line that ties it together, worth memorizing:

> "Five apps' worth of functionality, one voice interface, zero screen dependency — because the user this is built for cannot see a screen to begin with."

---

## 4. Market — Say Less Than You Think You Should

This is the section where teams over-talk and lose credibility. Judges have seen a hundred pitches with a market-size slide that's a single unsourced number pulled to sound big. Do the opposite: **give one real number, be honest about what it does and doesn't prove, and stop.**

**The one number to use:**
> "4.95 crore people with vision impairment in India — National Blindness and Visual Impairment Survey, Ministry of Health and Family Welfare. That's roughly the population of South Korea."

**What NOT to do:**
- Don't multiply that by an assumed price point to produce a big fake TAM number. Judges will ask you to justify the price assumption and you won't be able to.
- Don't claim "global market" unless you can name a second country's data. Stick to India — it's your actual context and it's a big enough number to matter on its own.

**The value framing that actually lands with judges (say this, don't skip it):**

> "The market case for offline isn't just 'nice to have' — it's that the population most affected includes rural and lower-income users, who are exactly the users least likely to have a reliable data connection. A cloud-dependent assistive app has a coverage gap in precisely the places assistive tech is needed most. Offline isn't a differentiator here — it's the qualifying feature."

If a judge pushes on "what's your business model" — answer honestly per the questions section below (§8). Don't invent a monetization story you haven't thought through; a clear "here's what we'd need to validate before pricing this" answer is more credible than a made-up number.

---

## 5. Tech Stack — Frame It as Decisions, Not a List

Don't recite "we used Python, FastAPI, YOLO, Gemma." List-recitation is boring and doesn't show engineering judgment. Instead, pick **two decisions that had a real trade-off**, and tell the story of the trade-off — that's what demonstrates you understand the system rather than assembled it.

**Decision 1 — model selection (say this one, it's your best material):**

> "We didn't pick Gemma 3 by default — we benchmarked three vision-language models on the same image and the same prompt. Moondream answered in about a second but got things wrong — it called our IDE screenshot 'a web browser.' Qwen2.5-VL was accurate but took 51 seconds per answer, because it processes way more image tokens at high resolution. Gemma 3 4B was the only one that was both correct and fast — about 1.5 seconds. That's the one we shipped."

**Decision 2 — why currency uses a trained model instead of just asking Gemma:**

> "Our currency detector is a YOLO model we trained ourselves on 1,917 images across 10 Indian note denominations — not Gemma. Here's why that split matters: our currency YOLO model has no 'this isn't currency' class, so it will confidently call a photo of a person '100 rupees' at 98% confidence. We only trust it once Gemma has independently confirmed currency is actually present. Two models, each doing the part it's actually good at — Gemma judges 'is there money here,' YOLO counts exactly how much."

**The stack, stated once, briefly, after the story:**
- Vision + reasoning: Gemma 3 4B (Ollama, local)
- Currency: custom YOLO, 10 classes, 5.5 MB
- General objects/obstacles: YOLO11n, 5.6 MB
- OCR: RapidOCR (ONNX PaddleOCR build)
- Speech-to-text: faster-whisper, local
- Backend: FastAPI + WebSocket, Python
- Frontend: vanilla JS, no framework, no build step

That's it — don't spend more than 45 seconds reading the stack itself. The two stories above are what makes it memorable.

---

## 6. Results — Numbers You Can Defend Under Questioning

Every number below is one you personally measured on this machine. Do not add numbers you haven't verified — a judge asking "how did you measure that" and getting a shrug is worse than not having the slide at all.

| Mode | Measured response time (warm) | Note |
|---|---|---|
| Proximity/detect | 0.03 s | YOLO only, no reasoning step |
| Currency | 0.30 s | Gemma gate + YOLO count |
| Describe | 1.49 s | Full Gemma vision call |
| Read (OCR + summary) | 4.34 s | OCR is fast; Gemma reasoning is the bulk of this |
| Question | ~5.9 s | Whisper transcription + Gemma vision answer |

**Say the honest caveat out loud before a judge finds it themselves:**

> "One thing worth being upfront about: the very first request after the app starts is slower — about 4 to 5 seconds slower — because the language model has to load into memory. After that first call, it stays warm and every subsequent request hits the numbers above. In a real deployment you'd warm the model at boot, which we already do in our server startup."

This kind of pre-emptive honesty is worth more in judging than hiding it — it signals you understand your own system's real behavior rather than only its best case.

**The other results worth stating, because they show iteration, not just a final number:**

> "Two of our components changed because testing told us the first choice was wrong. Our OCR engine was originally Tesseract — on a 640-pixel camera frame it read 37 characters of garbage at 15% confidence. We swapped to RapidOCR and got 1,398 usable characters at 68% confidence on the identical frame. That's not a tuning improvement, that's the feature going from broken to working."

---

## 7. Future Scope — Standalone Mobile App, and Cloud/Smart-Glasses as a Separate Track

Structure this as two branches, because they have different engineering profiles and judges will ask about them differently.

### Branch A — Standalone mobile app (the natural next step)

> "The architecture already separates cleanly into a client that captures and speaks, and models that infer. Porting means replacing the browser client with a native Android/iOS shell and running the same models — or their mobile-optimized equivalents — via on-device runtimes like ONNX Runtime Mobile or Core ML. The currency and object detection models are already under 6 MB each, which is mobile-friendly out of the box. Gemma 3 4B would need a quantized variant or a smaller vision model swap to fit comfortably on a phone — that's the one open engineering question, not the whole port."

Be precise about what's known vs. unknown here — don't claim the mobile port is trivial. It isn't. State the actual open question (model size on phone hardware) as an open question.

### Branch B — Cloud-hosted support for aided devices (smart glasses etc.)

This is a genuinely different deployment model from "offline-first," so frame it as an *additional option*, not a contradiction of your core pitch:

> "For a form factor like smart glasses — which have far less onboard compute than a phone — full on-device inference for a 4-billion-parameter vision model isn't realistic yet. There, the architecture would flip: the glasses stream a frame to a nearby paired phone or a private local server on the same network, which does the heavy inference and streams the spoken answer back. That's still not the public cloud — it's edge compute the user or institution controls — which keeps the core privacy promise intact even in a lighter-weight device."

**The line that keeps this coherent with your "offline" pitch instead of undermining it:**

> "Offline-first doesn't mean 'never networked' — it means 'never dependent on someone else's server.' A phone-to-glasses local link is still under the user's control. Uploading a blind user's camera feed to a company's cloud is the thing we're avoiding, not networking itself."

This distinction matters — say it explicitly if a judge asks "isn't glasses-to-cloud the same privacy problem you're solving?" It is not, if the "cloud" is a local/private link, and you should be ready to make that distinction cleanly.

---

## 8. Anticipated Judge Questions — With Answers

These are the questions this specific pitch invites. Prepare short, direct answers — don't pad them.

**Q: "This only works because you tested it on your own laptop with a good GPU/CPU. What about a low-end device?"**
> A: "Fair concern. Our smallest models — currency and object detection — are under 6 MB and run in milliseconds on CPU alone, no GPU required; those would work on modest hardware today. The bottleneck is the language model, Gemma 3 4B at 3.3 GB, which needs a reasonably modern machine. On a genuinely low-end device, we'd swap in a smaller quantized model for the reasoning-heavy modes and keep the fast paths as-is — the architecture doesn't require one model size for everything."

**Q: "Why not just use ChatGPT/GPT-4V or Google's cloud vision APIs? They're more accurate."**
> A: "They likely are more accurate in isolation. But every one of those requires a network call carrying a blind user's camera feed to a third party — their home, their documents, their money, their location context, all in that frame. That's the exact dependency we built this to remove. We deliberately chose 'sometimes less accurate but always available and always private' over 'more accurate but only when connected and never verifiable what happens to the data.'"

**Q: "Your currency detector was only trained on Indian notes. What about counterfeit detection, or damaged notes?"**
> A: "Correct scope limit — we trained on 10 genuine Indian denominations, old and new print variants, and we haven't built counterfeit detection. That's a materially different, harder problem — it needs security-feature-level image analysis, not denomination classification — and it's explicitly out of scope for what we built in this timeframe. We'd flag it as future work, not claim it works today."

**Q: "How do you know the answers are actually correct, especially for someone who can't verify them visually?"**
> A: "That's the core design constraint we built around, not an afterthought. We use confidence floors that make the system say 'I'm not sure' rather than guess, and a two-model verification step for currency specifically — Gemma has to independently confirm money is present before the YOLO count is trusted. We'd rather the system admit uncertainty than give a blind user false confidence in a wrong answer, because they have no way to double-check it themselves."

**Q: "What's your business model? Who pays for this?"**
> A: "We haven't validated a pricing model yet, and we'd rather say that plainly than invent one. What we can say: assistive technology in India has real precedent for institutional buyers — state disability welfare boards, NGOs, and rehabilitation centers already procure devices at scale for beneficiaries. An offline tool removes a recurring cloud API cost that a cloud-dependent competitor would carry, which changes the economics for that kind of bulk institutional deployment. That's a hypothesis, not a validated model yet."

**Q: "Isn't 4-6 seconds too slow for something like an obstacle warning? What if someone's about to walk into traffic?"**
> A: "Important distinction: our proximity/obstacle-detection mode is the fastest one we have — 30 milliseconds — specifically because that's the safety-critical path and it doesn't go through the language model at all. It's a lightweight object detector always running that check. The 4-6 second modes are the ones where the user explicitly asked a question or wants text read — those can tolerate a few seconds because the user chose to wait for that answer."

**Q: "What happens if the model gives a completely wrong answer with high confidence?"**
> A: "That risk exists for any ML system, and we don't claim to have eliminated it — we've reduced its likelihood with the verification steps described above, not solved it entirely. It's an honest limitation we'd want to keep improving, particularly by expanding the training data for the currency model and tightening the confidence thresholds based on real user feedback rather than our own test images."

**Q: "You said 'completely offline' — but doesn't the wake-word / voice recognition need internet in the browser?"**
> A: "Good catch on the nuance — the wake-word matching in our browser client uses the Web Speech API, which on some browsers does route through the browser vendor's servers. We were explicit about that boundary: it's used only to detect a fixed five-word command list, never the user's actual spoken question, which is transcribed entirely on-device by Whisper. If a judge wants a fully airtight offline story including wake-word matching, the fix is a local wake-word model instead of the browser API — that's a known, buildable swap, we just prioritized differently for this build."

---

## What to Cut If You're Running Short on Time

If you only have 3-4 minutes total, keep: the live demo opening (§1), the five-mode table (§3), one tech-decision story (§5, model selection), and the one honest caveat about cold-start latency (§6). Cut market sizing to the single sentence in §4 and skip the branch-B smart-glasses discussion unless a judge asks — it's a strong answer to have ready, not necessarily material to lead with.
