# models/

Nothing in this folder is checked into git (see .gitignore) — it's
all generated artifacts, some of them hundreds of MB.

| File | From | Required for |
|---|---|---|
| `currency_classifier.onnx` | `training/train_currency_classifier.py` | Currency mode |
| `currency_labels.json` | same script | Currency mode |
| `tts/<voice>.onnx` + `.onnx.json` | download from the [Piper voices repo](https://github.com/rhasspy/piper/blob/master/VOICES.md) | Higher-quality / multilingual speech output |

Until `currency_classifier.onnx` exists, the server runs fine — it
just says "The currency model hasn't been trained yet" instead of
guessing. That's deliberate; see server/tier1/currency.py.

Until a Piper voice is here, TTS falls back to macOS `say` (dev-only,
English, Mac-only) — see server/speech/tts.py.
