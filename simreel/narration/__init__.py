"""Narration: per-sentence TTS clips with word-level timestamps.

Sentences are synthesised one by one so that the choreographer can open up pauses
between them when the physical action needs more time than the words (a crane
cannot slew 20 m in 1.5 s).  Word timestamps are relative to each clip; the final
narration track is assembled after the timeline is solved.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import NarrationSettings
from ..text import words as split_words
from ..utils import log, read_audio, time_stretch
from . import align
from .align import WordTime
from .engines import AUTO_ORDER, ENGINES


@dataclass
class SentenceClip:
    text: str
    audio: np.ndarray
    sr: int
    words: list[WordTime]

    @property
    def duration(self) -> float:
        return len(self.audio) / self.sr


@dataclass
class Narration:
    clips: list[SentenceClip]
    sr: int
    engine: str
    aligner: str
    fixed_starts: list[float] | None = None  # recorded voice-over: timing is not negotiable
    total_fixed: float | None = None
    notes: list[str] = field(default_factory=list)

    def render(self, starts: list[float], total: float) -> np.ndarray:
        out = np.zeros(int(total * self.sr) + 1, dtype=np.float32)
        for c, s in zip(self.clips, starts):
            i = int(round(s * self.sr))
            seg = c.audio[: max(0, len(out) - i)]
            out[i:i + len(seg)] += seg
        return out


def _trim(audio: np.ndarray, sr: int, words: list[WordTime], pad: float = 0.06):
    _, rms = align.mfcc(audio, sr)
    a, b = align.voiced_bounds(rms)
    a = max(0.0, a - pad)
    b = min(len(audio) / sr, b + pad)
    seg = audio[int(a * sr):int(b * sr)]
    dur = len(seg) / sr
    out = [WordTime(w.text, min(max(0.0, w.start - a), dur), min(max(0.0, w.end - a), dur)) for w in words]
    return seg, out


def _make_engine(name: str, cfg: NarrationSettings):
    cls = ENGINES[name]
    if name == "elevenlabs":
        return cls(cfg.voice)
    return cls(cfg.voice, rate=cfg.rate)


def pick_engine(cfg: NarrationSettings):
    if cfg.tts != "auto":
        return _make_engine(cfg.tts, cfg)
    for name in AUTO_ORDER:
        cls = ENGINES[name]
        if not cls.available():
            continue
        try:
            eng = _make_engine(name, cfg)
            eng.synth("Test.")  # catches missing models / no network early
            return eng
        except Exception as exc:
            log(f"   tts '{name}' unavailable: {str(exc).splitlines()[0][:120]}")
    return None


def synthesize(sentences: list[str], cfg: NarrationSettings) -> Narration:
    if cfg.tts == "file":
        return _from_file(sentences, cfg)
    eng = None if cfg.tts == "none" else pick_engine(cfg)
    if eng is None:
        log("   narration: silent (timing from reading speed)")
        sr = 24000
        clips = []
        for s in sentences:
            n = len(split_words(s))
            dur = max(1.2, n / (2.6 * cfg.rate))
            clips.append(SentenceClip(s, np.zeros(int(dur * sr), np.float32), sr, align.estimate(s, dur)))
        return Narration(clips, sr, "none", "estimate")

    log(f"   narration engine: {eng.name}")
    clips, aligners = [], set()
    sr_common = None

    for i, s in enumerate(sentences):
        prev_t = sentences[i - 1] if i else ""
        next_t = sentences[i + 1] if i + 1 < len(sentences) else ""
        audio, sr, starts = eng.synth(s, prev_text=prev_t, next_text=next_t)
        if eng.name in ("pico", "elevenlabs") and abs(cfg.rate - 1) > 1e-3:  # no native speed control
            audio = time_stretch(audio, sr, cfg.rate)
            starts = [None if v is None else v / cfg.rate for v in starts] if starts else None
        if sr_common is None:
            sr_common = sr
        elif sr != sr_common:
            audio = align._resample(audio, sr, sr_common)
            sr = sr_common
        method = cfg.aligner
        if starts and any(v is not None for v in starts) and method in ("auto", "native"):
            _, rms = align.mfcc(audio, sr)
            words = align.words_from_starts(s, starts, len(audio) / sr, voiced_end=align.voiced_bounds(rms)[1])
            used = "native"
        else:
            words, used = align.align_sentence(s, audio, sr, "auto" if method == "native" else method,
                                               cfg.whisper_model, log)
        aligners.add(used)
        audio, words = _trim(audio, sr, words)
        clips.append(SentenceClip(s, audio, sr, words))
        log(f"   [{i + 1}/{len(sentences)}] {len(audio) / sr:5.2f}s via {used}: {s[:60]}")
    return Narration(clips, sr_common or 24000, eng.name, "+".join(sorted(aligners)))


def _from_file(sentences: list[str], cfg: NarrationSettings) -> Narration:
    if not cfg.voiceover_file:
        raise ValueError("tts=file needs narration.voiceover_file")
    audio, sr = read_audio(cfg.voiceover_file)
    full = " ".join(sentences)
    words, used = None, None
    if cfg.aligner in ("auto", "whisper") and align.whisper_available(cfg.whisper_model):
        try:
            rec = align.whisper_words(audio, sr, full, cfg.whisper_model)
            words, used = align.match_tokens(full, rec, len(audio) / sr), "whisper"
        except Exception as exc:
            log(f"   whisper failed: {exc}")
    if words is None:
        try:
            words, used = align.dtw_align(full, audio, sr), "dtw"
        except Exception as exc:
            log(f"   dtw failed: {exc}")
            words, used = align.estimate_audio(full, audio, sr), "estimate"
    per = align.split_full_alignment(sentences, words)
    clips, starts = [], []
    total = len(audio) / sr
    for i, (s, ws) in enumerate(zip(sentences, per)):
        st = max(0.0, ws[0].start - 0.05) if ws else 0.0
        en = (per[i + 1][0].start - 0.02) if i + 1 < len(per) and per[i + 1] else total
        seg = audio[int(st * sr):int(en * sr)]
        rel = [WordTime(w.text, w.start - st, w.end - st) for w in ws]
        clips.append(SentenceClip(s, seg, sr, rel))
        starts.append(st)
    return Narration(clips, sr, "file", used, fixed_starts=starts, total_fixed=total)
