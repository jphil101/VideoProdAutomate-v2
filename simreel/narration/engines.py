"""Text-to-speech engines.  Each `synth(text, ...)` returns
(samples float32 mono, sample_rate, native_word_starts | None).

native_word_starts is a list aligned with text.words(sentence): seconds or None.
"""
from __future__ import annotations

import asyncio
import base64
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from ..text import norm, words as split_words
from ..utils import decode_bytes, log, python_has, read_audio
from . import espeak

CACHE = Path(os.environ.get("SIMREEL_CACHE", Path.home() / ".cache" / "simreel"))


def _starts_from_char_times(sentence: str, char_starts: list[float]) -> list[float | None]:
    out, pos = [], 0
    for w in split_words(sentence):
        i = sentence.find(w, pos)
        pos = i + len(w)
        out.append(char_starts[i] if 0 <= i < len(char_starts) else None)
    return out


def _starts_from_tokens(sentence: str, tokens: list[tuple[str, float]]) -> list[float | None]:
    """Sequentially match engine word tokens to script tokens."""
    toks = split_words(sentence)
    out: list[float | None] = [None] * len(toks)
    j = 0
    for i, w in enumerate(toks):
        nw = norm(w)
        for k in range(j, min(j + 4, len(tokens))):
            if norm(tokens[k][0]) and (norm(tokens[k][0]) in nw or nw in norm(tokens[k][0])):
                out[i] = tokens[k][1]
                j = k + 1
                break
    return out


# ----------------------------------------------------------------- ElevenLabs
class ElevenLabs:
    name = "elevenlabs"

    @staticmethod
    def available() -> bool:
        return bool(os.environ.get("ELEVENLABS_API_KEY"))

    def __init__(self, voice: str | None = None):
        self.voice = voice or os.environ.get("ELEVENLABS_VOICE_ID", "JBFqnCBsd6RMkjVDRZzb")
        self.model = os.environ.get("ELEVENLABS_MODEL", "eleven_multilingual_v2")

    def synth(self, text: str, prev_text: str = "", next_text: str = ""):
        import requests

        url = f"https://api.elevenlabs.io/v1/text-to-speech/{self.voice}/with-timestamps"
        r = requests.post(url, params={"output_format": "mp3_44100_128"},
                          headers={"xi-api-key": os.environ["ELEVENLABS_API_KEY"]},
                          json={"text": text, "model_id": self.model, "previous_text": prev_text or None,
                                "next_text": next_text or None}, timeout=120)
        r.raise_for_status()
        data = r.json()
        audio, sr = decode_bytes(base64.b64decode(data["audio_base64"]), 44100)
        al = data.get("alignment") or {}
        chars = al.get("characters") or []
        starts = al.get("character_start_times_seconds") or []
        if chars and "".join(chars) == text:
            return audio, sr, _starts_from_char_times(text, starts)
        return audio, sr, None


# ----------------------------------------------------------------- Edge (online, free)
class Edge:
    name = "edge"

    @staticmethod
    def available() -> bool:
        return python_has("edge_tts")

    def __init__(self, voice: str | None = None, rate: float = 1.0):
        self.voice = voice or "en-US-AndrewMultilingualNeural"
        self.rate = rate

    def synth(self, text: str, **_):
        import edge_tts

        pct = int(round((self.rate - 1.0) * 100))
        rate = f"{pct:+d}%"

        async def go():
            try:
                comm = edge_tts.Communicate(text, self.voice, rate=rate, boundary="WordBoundary")
            except TypeError:  # older edge-tts
                comm = edge_tts.Communicate(text, self.voice, rate=rate)
            audio, toks = bytearray(), []
            async for ch in comm.stream():
                if ch["type"] == "audio":
                    audio.extend(ch["data"])
                elif ch["type"] == "WordBoundary":
                    toks.append((ch["text"], ch["offset"] / 1e7))
            return bytes(audio), toks

        data, toks = asyncio.run(go())
        audio, sr = decode_bytes(data, 24000)
        return audio, sr, (_starts_from_tokens(text, toks) if toks else None)


# ----------------------------------------------------------------- Kokoro (local neural)
class Kokoro:
    name = "kokoro"
    URLS = {
        "kokoro-v1.0.onnx": "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx",
        "voices-v1.0.bin": "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin",
    }

    @staticmethod
    def available() -> bool:
        return python_has("kokoro_onnx")

    def __init__(self, voice: str | None = None, rate: float = 1.0):
        from kokoro_onnx import Kokoro as K

        d = CACHE / "kokoro"
        d.mkdir(parents=True, exist_ok=True)
        for fn, url in self.URLS.items():
            if not (d / fn).exists():
                log(f"   downloading {fn} ...")
                import requests

                with requests.get(url, stream=True, timeout=600) as r:
                    r.raise_for_status()
                    tmp = d / (fn + ".part")
                    with open(tmp, "wb") as f:
                        for chunk in r.iter_content(1 << 20):
                            f.write(chunk)
                    tmp.rename(d / fn)
        self.k = K(str(d / "kokoro-v1.0.onnx"), str(d / "voices-v1.0.bin"))
        self.voice = voice or "am_michael"
        self.rate = rate

    def synth(self, text: str, **_):
        samples, sr = self.k.create(text, voice=self.voice, speed=self.rate, lang="en-us")
        return np.asarray(samples, dtype=np.float32), int(sr), None


# ----------------------------------------------------------------- Pico (offline, light)
class Pico:
    name = "pico"

    @staticmethod
    def available() -> bool:
        return shutil.which("pico2wave") is not None

    def __init__(self, voice: str | None = None, rate: float = 1.0):
        self.lang = voice or "en-US"
        self.rate = rate

    def synth(self, text: str, **_):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "s.wav"
            subprocess.run(["pico2wave", "-l", self.lang, "-w", str(out), text], check=True, capture_output=True)
            audio, sr = read_audio(out)
        return audio, sr, None


# ----------------------------------------------------------------- eSpeak NG (offline, exact timing)
class ESpeak:
    name = "espeak"

    @staticmethod
    def available() -> bool:
        return espeak.available()

    def __init__(self, voice: str | None = None, rate: float = 1.0):
        self.voice = voice or "en-us"
        self.wpm = int(165 * rate)

    def synth(self, text: str, **_):
        audio, sr, evs = espeak.synth(text, self.voice, self.wpm)
        b2c = []
        for ci, ch in enumerate(text):
            b2c.extend([ci] * len(ch.encode()))
        char_starts = [None] * len(text)
        for b, _ln, t in evs:
            c = b2c[min(max(b, 0), len(b2c) - 1)] if b2c else 0
            if char_starts[c] is None:
                char_starts[c] = t
        return audio, sr, _starts_from_char_times(text, char_starts)


ENGINES = {"elevenlabs": ElevenLabs, "kokoro": Kokoro, "edge": Edge, "pico": Pico, "espeak": ESpeak}
AUTO_ORDER = ["elevenlabs", "kokoro", "edge", "pico", "espeak"]
