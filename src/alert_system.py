import io
import queue
import threading
import time
from typing import Dict

from gtts import gTTS

try:                                  # Colab/Jupyter inline audio (optional)
    from IPython.display import Audio, display
except Exception:                     # not in a notebook → no inline playback
    Audio = display = None

class VoiceAlert:
    """Non-blocking TTS alert. Throttled to avoid spam."""

    def __init__(self, min_interval_sec: float = 4.0):
        self.min_interval = min_interval_sec
        self.last_alert_time: Dict[str,float] = {}
        self._queue: queue.Queue = queue.Queue(maxsize=3)
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self):
        while True:
            try:
                text = self._queue.get(timeout=1)
                try:
                    tts = gTTS(text=text, lang="en", slow=False)
                    buf = io.BytesIO()
                    tts.write_to_fp(buf)
                    buf.seek(0)
                    self._play(buf)
                except Exception:
                    pass
            except queue.Empty:
                pass

    @staticmethod
    def _in_notebook() -> bool:
        """True only inside a Jupyter/Colab kernel (ZMQInteractiveShell).
        Import success alone is insufficient — IPython is installed as a
        dependency, so it imports fine in a plain terminal too."""
        try:
            from IPython import get_ipython
            return type(get_ipython()).__name__ == "ZMQInteractiveShell"
        except Exception:
            return False

    @staticmethod
    def _play(buf: io.BytesIO):
        """
        Production-safe playback.
        Notebook (Colab/Jupyter): inline Audio widget, non-blocking.
        Terminal: native playback via playsound if available; otherwise
        skip silently. Never prints an Audio object to stdout.
        """
        # Notebook path — inline audio widget (real kernel only)
        if Audio is not None and display is not None and VoiceAlert._in_notebook():
            display(Audio(buf.read(), autoplay=True, rate=24000))
            return
        # Terminal path — best-effort native playback, skip if unsupported
        try:
            import os
            import tempfile
            from playsound import playsound
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                f.write(buf.read())
                path = f.name
            try:
                playsound(path)
            finally:
                os.remove(path)
        except Exception:
            pass   # playback unsupported in this environment → skip

    def speak(self, message: str, key: str = "default"):
        now = time.time()
        if now - self.last_alert_time.get(key, 0) < self.min_interval:
            return
        self.last_alert_time[key] = now
        try:
            self._queue.put_nowait(message)
        except queue.Full:
            pass

    def alert_for_decision(self, action: str, reason: str, rec_speed: float):
        messages = {
            "STOP":       f"Danger ahead. Stop immediately.",
            "BRAKE HARD": f"Brake hard. Reduce speed to {rec_speed:.0f} kilometres per hour.",
            "SLOW DOWN":  f"Slow down. {reason}. Recommended speed {rec_speed:.0f}.",
            "REDUCE SPD": f"Reduce speed to {rec_speed:.0f}. {reason}.",
            "MAINTAIN":   f"Maintain current speed.",
            "ACCELERATE": f"Road clear. You may accelerate.",
            "CRUISE":     "",
        }
        msg = messages.get(action, "")
        if msg:
            self.speak(msg, key=action)
