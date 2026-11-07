"""Word-level timestamp recovery for narration audio.

Priority (``aligner="auto"``):
  1. native timestamps from the TTS engine (ElevenLabs, Edge, eSpeak) -- exact;
  2. faster-whisper word timestamps, text-aligned back to the *script* words
     (so recognition typos never reach the captions);
  3. DTW forced alignment against an eSpeak rendering of the same sentence
     (aeneas-style, fully offline, works with any voice incl. recorded VO);
  4. syllable-weighted estimate inside the voiced region.
"""
from __future__ import annotations

import difflib
import math
from dataclasses import dataclass

import numpy as np

from ..text import norm, syllables, words as split_words
from . import espeak


@dataclass
class WordTime:
    text: str
    start: float
    end: float


# ----------------------------------------------------------------- features
def _resample(x: np.ndarray, sr: int, target: int) -> np.ndarray:
    if sr == target:
        return x
    n = int(round(len(x) * target / sr))
    return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)


def _mel_filterbank(sr: int, n_fft: int, n_mels: int = 32) -> np.ndarray:
    def hz2mel(f):
        return 2595 * np.log10(1 + f / 700)

    def mel2hz(m):
        return 700 * (10 ** (m / 2595) - 1)

    mels = np.linspace(hz2mel(60), hz2mel(sr / 2 * 0.95), n_mels + 2)
    bins = np.floor((n_fft + 1) * mel2hz(mels) / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(1, n_mels + 1):
        a, b, c = bins[m - 1], bins[m], bins[m + 1]
        for k in range(a, b):
            fb[m - 1, k] = (k - a) / max(1, b - a)
        for k in range(b, c):
            fb[m - 1, k] = (c - k) / max(1, c - b)
    return fb


def mfcc(x: np.ndarray, sr: int, hop_s: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
    """Return (features [frames, 24] = 12 MFCC + 12 deltas, frame rms energy)."""
    x = _resample(x, sr, 16000)
    sr = 16000
    win, hop, n_fft = int(0.025 * sr), int(hop_s * sr), 512
    if len(x) < win:
        x = np.pad(x, (0, win - len(x)))
    x = np.append(x[0], x[1:] - 0.97 * x[:-1])
    n = 1 + (len(x) - win) // hop
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    frames = x[idx] * np.hamming(win)[None, :]
    spec = np.abs(np.fft.rfft(frames, n_fft)) ** 2
    mel = np.log(spec @ _mel_filterbank(sr, n_fft).T + 1e-10)
    k = np.arange(mel.shape[1])
    dct = np.cos(np.pi / mel.shape[1] * (k[None, :] + 0.5) * np.arange(13)[:, None])
    c = mel @ dct.T
    c = c[:, 1:13]
    c = (c - c.mean(0)) / (c.std(0) + 1e-6)
    d = np.gradient(c, axis=0)
    rms = np.sqrt((frames**2).mean(1) + 1e-12)
    return np.hstack([c, d]), rms


def voiced_bounds(rms: np.ndarray, hop_s: float = 0.01, rel_db: float = -38.0) -> tuple[float, float]:
    if len(rms) == 0:
        return 0.0, 0.0
    thr = rms.max() * 10 ** (rel_db / 20)
    on = np.where(rms > thr)[0]
    if len(on) == 0:
        return 0.0, len(rms) * hop_s
    return on[0] * hop_s, (on[-1] + 1) * hop_s


def _dtw_path(cost: np.ndarray) -> np.ndarray:
    """Global DTW with vectorised rows (prefix-min trick). Returns ref->target map."""
    n, m = cost.shape
    D = np.empty((n, m), dtype=np.float64)
    D[0] = np.cumsum(cost[0])
    for i in range(1, n):
        prev = D[i - 1]
        A = prev.copy()
        A[1:] = np.minimum(prev[1:], prev[:-1])
        S = np.cumsum(cost[i])
        B = A - (S - cost[i])
        D[i] = S + np.minimum.accumulate(B)
    # backtrack
    i, j = n - 1, m - 1
    first_j = np.full(n, m - 1, dtype=int)
    while True:
        first_j[i] = min(first_j[i], j)
        if i == 0 and j == 0:
            break
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            k = np.argmin((D[i - 1, j - 1], D[i - 1, j], D[i, j - 1]))
            if k == 0:
                i, j = i - 1, j - 1
            elif k == 1:
                i -= 1
            else:
                j -= 1
    return first_j


def _char_spans(sentence: str) -> list[tuple[int, int]]:
    spans, pos = [], 0
    for w in split_words(sentence):
        i = sentence.find(w, pos)
        spans.append((i, i + len(w)))
        pos = i + len(w)
    return spans


def _byte_to_char(sentence: str) -> list[int]:
    m = []
    for ci, ch in enumerate(sentence):
        m.extend([ci] * len(ch.encode("utf-8")))
    m.append(len(sentence))
    return m


def words_from_starts(sentence: str, starts: list[float | None], duration: float, voiced_end: float | None = None) -> list[WordTime]:
    """Fill gaps by interpolation and turn word starts into [start, end) spans."""
    toks = split_words(sentence)
    n = len(toks)
    s = [None if v is None else float(v) for v in starts]
    known = [i for i, v in enumerate(s) if v is not None]
    if not known:
        return estimate(sentence, duration)
    for i in range(n):
        if s[i] is None:
            lo = max([k for k in known if k < i], default=None)
            hi = min([k for k in known if k > i], default=None)
            if lo is not None and hi is not None:
                s[i] = s[lo] + (s[hi] - s[lo]) * (i - lo) / (hi - lo)
            elif lo is not None:
                s[i] = s[lo] + 0.3 * (i - lo)
            else:
                s[i] = max(0.0, s[hi] - 0.3 * (hi - i))
    for i in range(1, n):  # monotonic
        s[i] = max(s[i], s[i - 1] + 0.03)
    end_all = voiced_end if voiced_end is not None else duration
    out = []
    for i, w in enumerate(toks):
        e = s[i + 1] if i + 1 < n else max(s[i] + 0.15, end_all)
        out.append(WordTime(w, s[i], min(e, duration)))
    return out


def estimate(sentence: str, duration: float, lead: float = 0.0, tail: float | None = None) -> list[WordTime]:
    toks = split_words(sentence)
    if not toks:
        return []
    end = duration if tail is None else tail
    weights = []
    for w in toks:
        wt = syllables(w) + 0.25
        if w.endswith((",", ";", ":")):
            wt += 0.9
        weights.append(wt)
    total = sum(weights)
    t, out = lead, []
    span = max(0.1, end - lead)
    for w, wt in zip(toks, weights):
        d = span * wt / total
        out.append(WordTime(w, t, t + d))
        t += d
    return out


def estimate_audio(sentence: str, audio: np.ndarray, sr: int) -> list[WordTime]:
    _, rms = mfcc(audio, sr)
    a, b = voiced_bounds(rms)
    return estimate(sentence, len(audio) / sr, a, b)


def dtw_align(sentence: str, audio: np.ndarray, sr: int) -> list[WordTime]:
    """Forced alignment of one sentence against an eSpeak reference rendering."""
    ref, rsr, evs = espeak.synth(sentence)
    f_ref, e_ref = mfcc(ref, rsr)
    f_tgt, e_tgt = mfcc(audio, sr)
    r0, r1 = voiced_bounds(e_ref)
    t0, t1 = voiced_bounds(e_tgt)
    hop = 0.01
    i0, i1 = int(r0 / hop), max(int(r0 / hop) + 2, int(r1 / hop))
    j0, j1 = int(t0 / hop), max(int(t0 / hop) + 2, int(t1 / hop))
    A, B = f_ref[i0:i1], f_tgt[j0:j1]
    if len(A) < 3 or len(B) < 3:
        return estimate_audio(sentence, audio, sr)
    An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
    Bn = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-9)
    cost = 1.0 - An @ Bn.T
    mapping = _dtw_path(cost)
    b2c = _byte_to_char(sentence)
    spans = _char_spans(sentence)
    starts: list[float | None] = [None] * len(spans)
    for bpos, _ln, t_ref in evs:
        c = b2c[min(max(bpos, 0), len(b2c) - 1)]
        wi = next((k for k, (a, b) in enumerate(spans) if a <= c < b), None)
        if wi is None:
            wi = next((k for k, (a, b) in enumerate(spans) if c < a), None)
        if wi is None:
            continue
        fi = int(round(t_ref / hop)) - i0
        fi = min(max(fi, 0), len(mapping) - 1)
        t_tgt = (mapping[fi] + j0) * hop
        if starts[wi] is None or t_tgt < starts[wi]:
            starts[wi] = t_tgt
    if starts and starts[0] is not None:
        starts[0] = min(starts[0], t0)
    elif starts:
        starts[0] = t0
    return words_from_starts(sentence, starts, len(audio) / sr, voiced_end=t1)


def match_tokens(sentence: str, rec_words: list[tuple[str, float, float]], duration: float) -> list[WordTime]:
    """Map recognised (word, start, end) back onto the script's own tokens."""
    toks = split_words(sentence)
    a = [norm(t) for t in toks]
    b = [norm(w) for w, _, _ in rec_words]
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    starts: list[float | None] = [None] * len(toks)
    for blk in sm.get_matching_blocks():
        for k in range(blk.size):
            starts[blk.a + k] = rec_words[blk.b + k][1]
    voiced_end = rec_words[-1][2] if rec_words else None
    return words_from_starts(sentence, starts, duration, voiced_end)


_whisper_model = None


def whisper_available(model_name: str) -> bool:
    try:
        import faster_whisper  # noqa: F401
    except Exception:
        return False
    return True


def whisper_words(audio: np.ndarray, sr: int, prompt: str, model_name: str = "base.en") -> list[tuple[str, float, float]]:
    global _whisper_model
    from faster_whisper import WhisperModel

    if _whisper_model is None:
        _whisper_model = WhisperModel(model_name, device="auto", compute_type="int8")
    x = _resample(audio.astype(np.float32), sr, 16000)
    segs, _ = _whisper_model.transcribe(x, word_timestamps=True, initial_prompt=prompt, language="en",
                                        vad_filter=False, beam_size=5)
    out = []
    for s in segs:
        for w in s.words or []:
            out.append((w.word.strip(), float(w.start), float(w.end)))
    return out


def align_sentence(sentence: str, audio: np.ndarray, sr: int, method: str = "auto",
                   whisper_model: str = "base.en", log=print) -> tuple[list[WordTime], str]:
    duration = len(audio) / sr
    order = [method] if method != "auto" else ["whisper", "dtw", "estimate"]
    for m in order:
        try:
            if m == "whisper" and whisper_available(whisper_model):
                rec = whisper_words(audio, sr, sentence, whisper_model)
                if rec:
                    return match_tokens(sentence, rec, duration), "whisper"
            elif m == "dtw" and espeak.available():
                return dtw_align(sentence, audio, sr), "dtw"
            elif m == "estimate":
                return estimate_audio(sentence, audio, sr), "estimate"
        except Exception as exc:  # fall through to the next aligner
            log(f"   aligner {m} failed: {exc}")
    return estimate_audio(sentence, audio, sr), "estimate"


def split_full_alignment(sentences: list[str], words: list[WordTime]) -> list[list[WordTime]]:
    """Distribute a whole-script word list back into sentences (in order)."""
    out, k = [], 0
    for s in sentences:
        n = len(split_words(s))
        out.append(words[k:k + n])
        k += n
    return out


def rms_db(x: np.ndarray) -> float:
    return 20 * math.log10(float(np.sqrt((x**2).mean())) + 1e-12)
