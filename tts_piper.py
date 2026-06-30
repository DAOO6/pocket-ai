import os
import re
import threading
import queue
import time
import urllib.request
import sys
import numpy as np

from piper.voice import PiperVoice
from piper.voice import SynthesisConfig

os.environ['ORT_LOGGING_LEVEL'] = '3'

IS_WINDOWS = sys.platform == "win32"

# On Linux/Pi the original ALSA device env var is still respected (for Pi compatibility).
# On Windows it is ignored; sounddevice uses the system default output.
TTS_ALSA_DEVICE = os.environ.get("TTS_ALSA_DEVICE", "default")

SENTENCE_END_RE = re.compile(r'(?<=[.!?\n])\s*')


def split_sentences(text):
    """Split text into sentences on . ! ? and newlines. Strips each; drops empty."""
    if not text or not text.strip():
        return []
    parts = SENTENCE_END_RE.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


class PocketAudio:
    def __init__(self, model_name="en_GB-alan-medium"):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_dir = os.path.join(self.base_dir, "models")
        self.model_path = os.path.join(self.model_dir, f"{model_name}.onnx")
        self.config_path = os.path.join(self.model_dir, f"{model_name}.onnx.json")
        self._ensure_models_exist(model_name)

        print("Loading Piper into memory... (Standby)")
        self.voice = PiperVoice.load(self.model_path, config_path=self.config_path)

        if IS_WINDOWS:
            print("System Ready. Audio output via sounddevice (Windows default device).")
        else:
            print(f"System Ready. Outputting to ALSA device: {TTS_ALSA_DEVICE}")

        self._queue = queue.Queue()
        self._queue_drained_callback = None
        self._abort = threading.Event()
        self._stt_engines = []   # STT engines to mute while speaking
        self.muted = False       # Global mute — silences TTS output entirely
        self._worker = threading.Thread(target=self._queue_worker, daemon=True)
        self._worker.start()

    def register_stt_engine(self, engine):
        """Register an STT engine to be muted during TTS playback."""
        self._stt_engines.append(engine)

    def _mute_stt(self):
        for engine in self._stt_engines:
            try:
                engine.muted = True
            except Exception:
                pass

    def _unmute_stt(self):
        for engine in self._stt_engines:
            try:
                engine.muted = False
            except Exception:
                pass

    def _ensure_models_exist(self, name):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        parts = name.split("-")
        quality = parts[-1]
        speaker = parts[-2]
        locale = parts[0]  # e.g. en_GB or en_US
        region = locale.split("_")[1].lower()  # e.g. gb or us
        lang = locale.split("_")[0]  # e.g. en
        url_base = f"https://huggingface.co/rhasspy/piper-voices/resolve/main/{lang}/{lang}_{region.upper()}/{speaker}/{quality}/"
        for ext in [".onnx", ".onnx.json"]:
            path = self.model_path if ext == ".onnx" else self.config_path
            if not os.path.exists(path):
                url = url_base + name + ext
                self._download_with_retry(url, path, name + ext)

    def _download_with_retry(self, url, dest_path, label, max_retries=5):
        """Download a file with resume support and automatic retries."""
        for attempt in range(1, max_retries + 1):
            existing = os.path.getsize(dest_path) if os.path.exists(dest_path) else 0
            headers = {}
            if existing > 0:
                headers["Range"] = f"bytes={existing}-"
                print(f"Resuming {label} from {existing // (1024*1024)}MB (attempt {attempt}/{max_retries})...")
            else:
                print(f"Downloading {label}... (attempt {attempt}/{max_retries})")
            try:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=60) as resp:
                    mode = "ab" if existing > 0 else "wb"
                    with open(dest_path, mode) as f:
                        while True:
                            chunk = resp.read(1024 * 1024)  # 1MB chunks
                            if not chunk:
                                break
                            f.write(chunk)
                print(f"Downloaded {label} successfully.")
                return
            except Exception as e:
                print(f"Download error ({e}), retrying...")
                time.sleep(2)
        raise RuntimeError(f"Failed to download {label} after {max_retries} attempts.")

    def set_queue_drained_callback(self, callback):
        self._queue_drained_callback = callback

    def clear_queue(self):
        self._abort.set()  # Signal worker to discard next_audio and stop
        try:
            while True:
                self._queue.get_nowait()
        except queue.Empty:
            pass

    def set_muted(self, muted: bool):
        """Mute or unmute TTS by setting output volume to 0/1.
        Does NOT interrupt playback or touch the queue — avoids the skip-first-line bug."""
        self.muted = muted
        try:
            import sounddevice as sd
            sd.default.latency = 'high'  # no-op but ensures sd is initialised
            # sounddevice doesn't have a global volume, so we set via stream
            # Instead we track muted and apply gain in the worker
        except Exception:
            pass

    def _synthesise_to_array(self, text):
        """Synthesise text to a numpy audio array without playing it."""
        audio_chunks = []
        for chunk in self.voice.synthesize(text, SynthesisConfig(length_scale=0.82)):
            audio_chunks.append(chunk.audio_int16_bytes)
        if not audio_chunks:
            return None
        raw_bytes = b"".join(audio_chunks)
        audio_array = np.frombuffer(raw_bytes, dtype=np.int16)
        silence = np.zeros(int(22050 * 0.3), dtype=np.int16)
        return np.concatenate([audio_array, silence])

    def _queue_worker(self):
        """
        Pre-synthesise the next sentence while the current one is playing,
        so there is minimal gap between sentences.
        """
        import sounddevice as sd

        next_audio = None
        next_text = None

        while True:
            try:
                # If aborted, discard any pre-synthesised audio and reset
                if self._abort.is_set():
                    next_audio = None
                    next_text = None
                    self._abort.clear()
                    self._unmute_stt()
                    continue

                # Get current sentence (use pre-synthesised if available)
                if next_audio is not None:
                    current_audio = next_audio
                    current_text = next_text
                    next_audio = None
                    next_text = None
                else:
                    text = self._queue.get()
                    if text is None:
                        continue
                    # Check abort again after potentially long wait on queue.get()
                    if self._abort.is_set():
                        self._abort.clear()
                        continue
                    current_audio = self._synthesise_to_array(text)
                    current_text = text

                if current_audio is None:
                    continue

                # Start playback — apply zero gain if muted
                self._mute_stt()
                playback_audio = np.zeros_like(current_audio) if self.muted else current_audio
                sd.play(playback_audio, samplerate=22050)

                # While playing, pre-synthesise the next sentence if one is queued
                # But only if not aborted
                if not self._abort.is_set():
                    try:
                        next_text = self._queue.get_nowait()
                        if next_text is not None:
                            next_audio = self._synthesise_to_array(next_text)
                    except queue.Empty:
                        next_audio = None
                        next_text = None

                # Wait for playback to finish
                sd.wait()
                self._unmute_stt()

                # If aborted during playback, discard pre-synthesised and reset
                if self._abort.is_set():
                    next_audio = None
                    next_text = None
                    self._abort.clear()
                    continue

                # Check if queue is now drained
                if next_audio is None and self._queue.empty() and self._queue_drained_callback:
                    try:
                        self._queue_drained_callback()
                    except Exception as e:
                        print(f"TTS queue drained callback error: {e}")

            except Exception as e:
                print(f"TTS worker error: {e}")

    def clean_text(self, text):
        return re.sub(r'[^a-zA-Z0-9\s.,!?;:\'\"()-]', '', text)

    def enqueue_sentence(self, sentence):
        cleaned = self.clean_text(sentence)
        if cleaned.strip():
            self._queue.put(cleaned)

    def enqueue_text(self, text):
        for s in split_sentences(text):
            self.enqueue_sentence(s)

    def speak(self, text):
        self.enqueue_text(text)

    def _speak_internal(self, text):
        print(f"Synthesizing: {text[:50]}...")
        t0 = time.perf_counter()

        if IS_WINDOWS:
            self._speak_windows(text, t0)
        else:
            self._speak_linux(text, t0)

    def _speak_windows(self, text, t0):
        """Play audio on Windows using sounddevice."""
        try:
            import sounddevice as sd

            # Collect all int16 audio chunks from Piper
            audio_chunks = []
            for chunk in self.voice.synthesize(text, SynthesisConfig(length_scale=0.82)):
                audio_chunks.append(chunk.audio_int16_bytes)

            tts_ms = (time.perf_counter() - t0) * 1000
            print(f"  text-to-speech: {tts_ms:.0f} ms")

            if not audio_chunks:
                return

            # Combine all chunks into a single numpy array
            raw_bytes = b"".join(audio_chunks)
            audio_array = np.frombuffer(raw_bytes, dtype=np.int16)

            # Pad with 0.3s of silence at the end to prevent last word clipping
            silence = np.zeros(int(22050 * 0.3), dtype=np.int16)
            audio_array = np.concatenate([audio_array, silence])

            # Piper outputs mono int16 at 22050 Hz
            sd.play(audio_array, samplerate=22050)
            sd.wait()  # Block until playback is complete

        except ImportError:
            print("ERROR: sounddevice not installed. Run: pip install sounddevice")
        except Exception as e:
            print(f"Audio Error (Windows): {e}")

    def _speak_linux(self, text, t0):
        """Play audio on Linux/Pi using aplay (original behaviour)."""
        import subprocess
        import shlex

        command = f"aplay -D {TTS_ALSA_DEVICE} -r 22050 -f S16_LE -t raw -"
        args = shlex.split(command)
        try:
            with subprocess.Popen(args, stdin=subprocess.PIPE) as play_process:
                for chunk in self.voice.synthesize(text, SynthesisConfig(length_scale=0.82)):
                    play_process.stdin.write(chunk.audio_int16_bytes)
                tts_ms = (time.perf_counter() - t0) * 1000
                print(f"  text-to-speech: {tts_ms:.0f} ms")
                play_process.stdin.close()
                play_process.wait()
                if play_process.returncode != 0:
                    print(f"aplay exited with code {play_process.returncode}")
        except Exception as e:
            print(f"Audio Error (Linux): {e}")


if __name__ == "__main__":
    models = [
        ("medium", "en_US-lessac-medium"),
    ]
    msg = "Hello, my name is Pocket. Nice to meet you."
    for label, model_name in models:
        print(f"\n{'='*60}\n Model: {label} ({model_name})\n{'='*60}")
        done = threading.Event()
        ai = PocketAudio(model_name=model_name)
        ai.set_queue_drained_callback(lambda: done.set())
        ai.speak(msg)
        print("Waiting for playback to finish...")
        if not done.wait(timeout=30):
            print("Timeout waiting for playback.")
        else:
            print("Done.")
