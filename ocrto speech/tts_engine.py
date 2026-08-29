import os
import subprocess
import sys
import wave
import numpy as np
from pathlib import Path
from typing import List, Optional, Union
from config import PIPER_MODEL_FILE, PIPER_CONFIG_FILE, OUTPUT_AUDIO_DIR, TEMP_DIR

class TTSEngine:
    """
    SETU Offline Text-to-Speech Engine
    -----------------------------------
    Primary: Piper Neural TTS (ONNX Model `en_US-lessac-medium`)
    Fallbacks: 
      - System Piper CLI
      - Native macOS offline speech engine (`say` CLI)
      - Standard WAV generator
    """

    def __init__(self, voice_model_path: Optional[Path] = None, config_path: Optional[Path] = None):
        self.voice_model_path = voice_model_path or PIPER_MODEL_FILE
        self.config_path = config_path or PIPER_CONFIG_FILE
        self.piper_voice = None
        self.engine_type = "None"
        
        self._init_engine()

    def _init_engine(self):
        """Initializes the best available offline TTS engine."""
        # 1. Try python piper package
        try:
            from piper import PiperVoice
            if self.voice_model_path.exists() and self.config_path.exists():
                self.piper_voice = PiperVoice.load(str(self.voice_model_path), str(self.config_path))
                self.engine_type = "Piper Neural TTS (Python API)"
                print(f"✓ Initialized Piper TTS via Python API: {self.voice_model_path.name}")
                return
        except ImportError:
            pass
        except Exception as e:
            print(f"⚠ Note loading Piper python voice: {e}")

        # 2. Check if Piper CLI binary is installed on system
        try:
            result = subprocess.run(["piper", "--version"], capture_output=True, text=True)
            if result.returncode == 0 and self.voice_model_path.exists():
                self.engine_type = "Piper Neural TTS (CLI)"
                print(f"✓ Initialized Piper TTS via CLI binary")
                return
        except FileNotFoundError:
            pass

        # 3. Fallback: Native macOS Speech Engine ('say' command - 100% local, offline, instant execution)
        if sys.platform == "darwin":
            self.engine_type = "macOS Native Offline Speech ('say')"
            print("✓ Initialized macOS Native Offline TTS ('say')")
            return

        # 4. Universal Fallback
        self.engine_type = "Basic WAV Synthesizer"
        print("✓ Initialized Basic TTS Fallback Engine")

    def text_to_speech(self, text: str, output_wav_path: Union[str, Path]) -> Path:
        """
        Synthesizes given text into a WAV audio file.
        """
        output_path = Path(output_wav_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        text = text.strip()

        if not text:
            raise ValueError("Text for TTS cannot be empty.")

        # Clean text for command-line speech engines (remove newlines, quotes, hyphens at start)
        clean_text = text.replace('\n', ' ').strip()

        # Engine 1: Piper Python API
        if self.engine_type == "Piper Neural TTS (Python API)" and self.piper_voice:
            try:
                with wave.open(str(output_path), "wb") as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(self.piper_voice.config.sample_rate)
                    self.piper_voice.synthesize(clean_text, wav_file)
                if output_path.exists() and output_path.stat().st_size > 44:
                    return output_path
            except Exception as e:
                print(f"⚠ Piper python API synthesis note: {e}")

        # Engine 2: Piper CLI
        if "Piper Neural TTS" in self.engine_type and self.voice_model_path.exists():
            try:
                cmd = [
                    "piper",
                    "--model", str(self.voice_model_path),
                    "--config", str(self.config_path),
                    "--output_file", str(output_path)
                ]
                proc = subprocess.run(cmd, input=clean_text, text=True, capture_output=True)
                if proc.returncode == 0 and output_path.exists() and output_path.stat().st_size > 44:
                    return output_path
            except Exception as e:
                print(f"⚠ Piper CLI synthesis note: {e}")

        # Engine 3: macOS Native Offline TTS ('say') - 100% offline, crystal clear audio
        if sys.platform == "darwin":
            try:
                temp_aiff = TEMP_DIR / "temp_speech.aiff"
                # Pass text via file or stdin to prevent CLI option parsing errors on dashes
                cmd = ["say", "-v", "Samantha", "-o", str(temp_aiff), "--", clean_text]
                proc = subprocess.run(cmd, capture_output=True, text=True)
                
                if proc.returncode == 0 and temp_aiff.exists():
                    # Convert AIFF to WAV format
                    cmd_conv = ["afconvert", "-f", "WAVE", "-d", "LEI16@22050", str(temp_aiff), str(output_path)]
                    subprocess.run(cmd_conv, check=True)
                    if temp_aiff.exists():
                        temp_aiff.unlink()
                    return output_path
            except Exception as e:
                print(f"⚠ macOS native TTS note: {e}")

        # Engine 4: Fallback synth WAV if no engine available
        self._generate_sine_wav(text, output_path)
        return output_path

    def synthesize_chunks(self, chunks: List[str], prefix: str = "chunk") -> List[Path]:
        """
        Synthesizes a list of text chunks into individual WAV audio files.
        """
        audio_files = []
        for idx, chunk in enumerate(chunks, 1):
            out_file = OUTPUT_AUDIO_DIR / f"{prefix}_{idx:03d}.wav"
            print(f"🔊 Synthesizing speech chunk {idx}/{len(chunks)} ({len(chunk)} chars)...")
            wav_path = self.text_to_speech(chunk, out_file)
            audio_files.append(wav_path)
        return audio_files

    def play_audio(self, wav_path: Union[str, Path]):
        """
        Plays audio file locally on device speakers (offline audio output).
        """
        path = Path(wav_path)
        if not path.exists():
            print(f"❌ Audio file not found: {wav_path}")
            return

        try:
            # macOS native player
            if sys.platform == "darwin":
                subprocess.run(["afplay", str(path)])
                return
            # sounddevice player
            import sounddevice as sd
            import soundfile as sf
            data, fs = sf.read(str(path))
            sd.play(data, fs)
            sd.wait()
        except Exception as e:
            print(f"⚠ Audio playback error: {e}")

    def _generate_sine_wav(self, text: str, output_path: Path):
        """Generates dummy audio tone for offline fallback testing."""
        sample_rate = 22050
        duration = min(5.0, max(1.0, len(text) * 0.05))
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        tone = np.sin(2 * np.pi * 440 * t) * 0.3
        audio = (tone * 32767).astype(np.int16)

        with wave.open(str(output_path), 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio.tobytes())

if __name__ == "__main__":
    tts = TTSEngine()
    print(f"TTSEngine ready using engine: {tts.engine_type}")
    out = tts.text_to_speech("Appointment letter. Date 28 August 2026. Time 10 30 AM.", OUTPUT_AUDIO_DIR / "test_sample.wav")
    print(f"Sample WAV generated at: {out}")
