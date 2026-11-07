"""Tiny ctypes binding to libespeak-ng that returns PCM *and* exact word events.

Used two ways:
  * as an offline TTS engine with native word timestamps;
  * as the reference voice for the DTW forced aligner (any other voice is aligned
    against espeak's rendering of the same text, whose word timings are exact).
"""
from __future__ import annotations

import ctypes as C
import ctypes.util
import threading

import numpy as np

_lock = threading.Lock()
_lib = None
_rate = None


class _Event(C.Structure):
    _fields_ = [("type", C.c_int), ("unique_identifier", C.c_uint), ("text_position", C.c_int),
                ("length", C.c_int), ("audio_position", C.c_int), ("sample", C.c_int),
                ("user_data", C.c_void_p), ("id", C.c_int * 2)]


_CB = C.CFUNCTYPE(C.c_int, C.POINTER(C.c_short), C.c_int, C.POINTER(_Event))


def available() -> bool:
    try:
        _load()
        return True
    except OSError:
        return False


def _load():
    global _lib, _rate
    if _lib is None:
        name = ctypes.util.find_library("espeak-ng") or "libespeak-ng.so.1"
        lib = C.CDLL(name)
        rate = lib.espeak_Initialize(2, 0, None, 0)  # AUDIO_OUTPUT_SYNCHRONOUS
        if rate <= 0:
            raise OSError("espeak_Initialize failed")
        _lib, _rate = lib, rate
    return _lib, _rate


def synth(text: str, voice: str = "en-us", rate_wpm: int = 165) -> tuple[np.ndarray, int, list[tuple[int, int, float]]]:
    """Return (float32 mono samples, sample_rate, [(char_start, char_len, time_s), ...])."""
    with _lock:
        lib, sr = _load()
        pcm: list[bytes] = []
        events: list[tuple[int, int, int]] = []

        def cb(wav, n, ev):
            if n > 0:
                pcm.append(C.string_at(wav, n * 2))
            i = 0
            while ev[i].type != 0:
                e = ev[i]
                if e.type == 1:  # espeakEVENT_WORD
                    events.append((e.text_position, e.length, e.audio_position))
                i += 1
            return 0

        cbf = _CB(cb)
        lib.espeak_SetSynthCallback(cbf)
        lib.espeak_SetVoiceByName(voice.encode())
        lib.espeak_SetParameter(1, int(rate_wpm), 0)  # espeakRATE
        data = text.encode("utf-8")
        lib.espeak_Synth(data, len(data) + 1, 0, 0, 0, 0x0, None, None)
        lib.espeak_Synchronize()
    audio = np.frombuffer(b"".join(pcm), dtype=np.int16).astype(np.float32) / 32768.0
    # text_position is 1-based byte offset
    words = []
    for pos, ln, ms in events:
        words.append((pos - 1, ln, ms / 1000.0))
    return audio, sr, words
