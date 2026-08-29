# Prompt: implement/fix proximity (collision) detection

Copy everything below into the other project's Claude Code session.

---

I'm implementing a "proximity" feature in this project — it's the same concept as "collision detection" in a reference project called SETU, just renamed. It currently isn't working. Here's exactly how the reference implementation works, end to end, so you can port it correctly or diff it against what's currently broken here.

## What it does

A continuously-streamed camera feed is scanned by a general-purpose YOLO object detector (COCO classes, e.g. `yolo11n.pt`). Instead of announcing everything it sees, it filters to a curated list of **hazard classes** (furniture, vehicles, people, animals — NOT small handheld objects like bottles/cups/phones, since those would false-positive constantly), and only raises an alert when a hazard object's bounding box takes up a large enough fraction of the frame to mean "this is close." Two severity tiers: **warn** ("careful, X close ahead") at a lower area threshold, **urgent** ("stop, X right in front of you") at a higher one. The common case — nothing hazardous close by — is silence; the client doesn't nag every frame.

## The exact pipeline

### 1. Detector setup (Python / Ultralytics YOLO)

```python
from ultralytics import YOLO

# Load once at startup, not per-request — model load is expensive,
# inference is cheap. Reuse a general COCO-trained model; no custom
# training needed for this feature.
model = YOLO("yolo11n.pt")  # or a locally pinned path if you ship the weights
```

### 2. The hazard class list — this is the part that actually makes it useful

```python
HAZARD_CLASSES = {
    # Furniture / structural — the "wall/bench/table" family
    "bench", "chair", "couch", "bed", "dining table", "potted plant",
    "toilet", "refrigerator", "oven", "sink", "tv",
    # Vehicles — always hazards on a walking path
    "car", "truck", "bus", "motorcycle", "bicycle", "train",
    # People and larger animals
    "person", "dog", "horse", "cow", "sheep", "cat", "bear",
    # Blockers
    "traffic light", "stop sign", "fire hydrant", "parking meter",
}
```

Deliberately excludes small handheld COCO classes (bottle, cup, keyboard, book, cell phone, laptop) — those would fire constantly for no useful reason (the user's own hands/desk fill the frame with these all day) and would erode trust in the alert.

Note: "wall" isn't a COCO class YOLO can detect directly. The area-fraction heuristic below catches wall-like obstructions anyway (a wall dominates the frame the same way a big piece of furniture would) — if you need dedicated wall detection, you'd need a depth-estimation model (MiDaS / Depth-Anything) instead of object detection, which is a materially bigger lift.

### 3. Confidence and distance thresholds — the tunable knobs

```python
CONF_FLOOR = 0.45              # YOLO detection confidence to trust at all
CLOSE_AREA_FRACTION = 0.22     # box_area / frame_area >= this -> "warn"
URGENT_AREA_FRACTION = 0.40    # box_area / frame_area >= this -> "urgent"
HINT_COOLDOWN_SECONDS = 2.0    # minimum gap between spoken alerts (see below)
```

These four numbers are the entire tuning surface. If proximity alerts fire too often on harmless things, raise `CONF_FLOOR`. If it's not catching real hazards, lower the area fractions. Distance here is a **monocular area-fraction heuristic** — bigger box = closer object — not real depth. It's imprecise but cheap and good enough for "something is about to be bumped into."

### 4. The scan function

```python
def scan_for_proximity(bgr_frame):
    """
    Returns a list of hazard objects big enough in frame to matter,
    sorted by severity (urgent first, then by size).
    Empty list = path is clear — this is the common case.
    """
    h, w = bgr_frame.shape[:2]
    frame_area = float(h * w)
    results = model.predict(bgr_frame, conf=CONF_FLOOR, verbose=False)

    threats = []
    for r in results:
        for box in r.boxes:
            label = r.names[int(box.cls[0])]
            if label not in HAZARD_CLASSES:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            area_fraction = max(0.0, (x2 - x1) * (y2 - y1)) / frame_area
            if area_fraction < CLOSE_AREA_FRACTION:
                continue  # detected, but not close enough to matter
            severity = "urgent" if area_fraction >= URGENT_AREA_FRACTION else "warn"
            threats.append({
                "label": label,
                "confidence": float(box.conf[0]),
                "area_fraction": area_fraction,
                "severity": severity,
                "bbox": [x1, y1, x2, y2],
            })

    threats.sort(key=lambda t: (t["severity"] == "warn", -t["area_fraction"]))
    return threats
```

### 5. Turning threats into speech

```python
def speak_for(threats):
    """Returns (spoken_text, severity_or_None)."""
    if not threats:
        return ("Path is clear.", None)

    by_label = {}
    for t in threats:
        by_label.setdefault(t["label"], []).append(t)
    urgent = [lbl for lbl, ts in by_label.items() if any(t["severity"] == "urgent" for t in ts)]
    warn = [lbl for lbl in by_label if lbl not in urgent]

    if urgent:
        return (f"Stop. {', '.join(sorted(urgent))} right in front of you.", "urgent")
    return (f"Careful, {', '.join(sorted(warn))} close ahead.", "warn")
```

### 6. Server-side wiring (per-frame handler, e.g. inside a WebSocket loop)

The critical detail here: **run YOLO inference off the async event loop.** Ultralytics' `predict()` is synchronous and CPU/GPU-bound; calling it directly inside an `async def` handler blocks every other connection and the guidance/heartbeat messages for the duration of inference. In SETU this is done with:

```python
import asyncio
threats = await asyncio.to_thread(scan_for_proximity, frame)
```

Also **warm the model at server startup**, not on the first real request — the very first YOLO inference call pays a one-time Metal/CUDA kernel-compilation cost that can hang for 10-40+ seconds. Run one throwaway prediction on a blank frame during your app's startup/lifespan hook so that cost is paid before any user ever sends a frame:

```python
warmup_frame = np.zeros((640, 640, 3), dtype=np.uint8)
scan_for_proximity(warmup_frame)  # discard result — just forces kernel compile
```

Then per-frame, apply a **spoken-alert cooldown** — without this, a continuous 3fps stream would re-speak "careful, chair close ahead" every ~300ms while the chair just sits there, which is unusable:

```python
now = time.monotonic()
speak_text, severity = speak_for(threats)
speak_this_frame = None
if severity is not None and (now - state.last_hint_time) >= HINT_COOLDOWN_SECONDS:
    speak_this_frame = speak_text
    state.last_hint_time = now
# state.last_hint_time must be per-connection, not global — otherwise
# multiple simultaneous users share one cooldown clock.
```

Every response back to the client must include the **visual/haptic severity every frame** (`severity`, computed fresh each frame) even when `speak_this_frame` is `None` — the client uses `speak_this_frame` to decide whether to play audio/TTS this tick, but still updates its on-screen state and haptic feedback continuously. Don't collapse these into one field — a client that only gets told about a hazard on the cooldown tick will show a stale "all clear" the rest of the time.

### 7. Client-side wiring (the parts most likely to be your actual bug)

- The client runs a **separate, always-on interval** streaming frames for this mode specifically — independent of any voice-command / one-shot request-response flow the app might also have. Typical: `setInterval` at ~3fps, sending `{type: "frame", mode: "proximity", image_b64: ..., seq: N}` over a WebSocket.
- **Check your mode-name string is identical, character-for-character, on both client and server.** If you renamed "collision" to "proximity" in this project, and the server has a strict schema validator (e.g. a Pydantic `Literal["currency", "text", "collision", ...]` or equivalent enum check) that still says `"collision"` instead of `"proximity"`, **every frame will be silently rejected by request validation before it ever reaches your detection code.** This produces exactly the symptom "nothing happens, no visible error" — the WebSocket message gets bounced back as a parse/validation error (often logged server-side only, easy to miss), and the client never receives a real detection result. **This is the single most likely cause if the rest of the pipeline above is otherwise a faithful port.** Grep your whole codebase (client AND server) for the string `"collision"` and make sure every occurrence — mode literals, schema enums, WebSocket message-type checks, JSON field names like `collision_alert` — is consistently renamed, not just the obviously-named files.
- On the client, gate the continuous stream on your app's own state — if the app has other modes (voice commands, single-shot requests) that shouldn't run concurrently with proximity streaming, make sure entering another mode actually pauses the interval (or the server-side handler at minimum ignores/short-circuits proximity frames while another mode owns the "turn"), and that returning to the idle/default state resumes it. A stream that keeps silently running (or one that never resumes after being paused) both look identical to "the feature isn't working" from the outside.
- Only fire the haptic/vibration API on a real user gesture having happened first — browsers (Chrome specifically) silently block `navigator.vibrate()` calls until the user has tapped somewhere on the page at least once. Wrap it in a try/catch and don't treat a blocked vibration as a sign the whole feature is broken.

## Debugging checklist if it's still not working after porting this

1. Confirm the YOLO model actually loads (`model.ready` / equivalent flag) — log it explicitly at startup, don't assume.
2. Add a `console.log`/server-log right where the mode-name string is checked/validated on both ends, and confirm the exact literal string matches on both sides.
3. Send one frame manually (curl / a REST test) with a known object-filled image (anything with a person or car in frame) and confirm you get threats back with the raw pipeline **before** wiring the WebSocket/streaming loop — isolate "is detection broken" from "is the transport broken."
4. Check confidence values in a raw test run — if a real hazard object is detected below `CONF_FLOOR`, you'll get zero threats even though YOLO "saw" it; log the raw detections before the confidence filter to tell the two cases apart.
5. Confirm inference is actually running off the event loop (`asyncio.to_thread` or equivalent) — if you skip this, the symptom is "everything else in the app hangs/lags whenever a proximity frame arrives," not "proximity itself doesn't work," so it can be misdiagnosed.
