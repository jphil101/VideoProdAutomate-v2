"""Procedural, physics-timed sound effects (modal synthesis -- no sample library needed).

Impacts are placed at the frame the body actually lands/releases, so sound and
picture agree with the simulation."""
from __future__ import annotations

import numpy as np

SR = 48000


def _env(n, attack, decay):
    t = np.arange(n) / SR
    return np.minimum(1.0, t / max(attack, 1e-4)) * np.exp(-t / decay)


def _modes(freqs, decays, amps, dur, rng):
    n = int(dur * SR)
    t = np.arange(n) / SR
    out = np.zeros(n)
    for f, d, a in zip(freqs, decays, amps):
        out += a * np.sin(2 * np.pi * f * (1 + rng.uniform(-0.01, 0.01)) * t + rng.uniform(0, 6.28)) * np.exp(-t / d)
    return out


def impact(kind: str, strength: float, rng) -> np.ndarray:
    if kind == "steel":
        f0 = rng.uniform(170, 210)
        ratios = [1.0, 2.76, 5.40, 8.93, 13.3]  # free-free beam bending modes
        body = _modes([f0 * r for r in ratios], [0.9, 0.55, 0.35, 0.22, 0.12], [1.0, 0.6, 0.4, 0.25, 0.15], 1.6, rng)
        n = len(body)
        click = rng.standard_normal(n) * _env(n, 0.0005, 0.012) * 0.8
        sig = body * 0.5 + click
    elif kind == "concrete" or kind == "module":
        n = int(0.9 * SR)
        noise = rng.standard_normal(n)
        k = int(SR / 900)
        low = np.convolve(noise, np.ones(k) / k, mode="same")
        t = np.arange(n) / SR
        thump = np.sin(2 * np.pi * (58 + 30 * np.exp(-t / 0.05)) * t) * np.exp(-t / 0.18)
        sig = low * _env(n, 0.001, 0.09) * 2.5 + thump * 0.9
    elif kind == "timber":
        sig = _modes([310, 720, 1300], [0.12, 0.07, 0.04], [1, 0.5, 0.3], 0.5, rng)
    elif kind == "bolt":  # impact wrench burst
        n = int(0.55 * SR)
        t = np.arange(n) / SR
        buzz = np.sign(np.sin(2 * np.pi * 38 * t)) * rng.standard_normal(n) * 0.25
        sig = buzz * np.exp(-((t - 0.25) / 0.16) ** 2) + _modes([2400, 3700], [0.05, 0.03], [0.3, 0.2], 0.55, rng)
    else:  # "rig": shackle / hook clink
        sig = _modes([2100, 3300, 5200], [0.08, 0.05, 0.03], [0.6, 0.4, 0.3], 0.3, rng)
    sig = sig / (np.abs(sig).max() + 1e-9)
    return sig * strength


def render_sfx(events: list[tuple[float, str, float]], total: float, gain_db: float = -14.0, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.zeros(int(total * SR) + SR)
    g = 10 ** (gain_db / 20)
    for t, kind, strength in events:
        s = impact(kind, strength, rng) * g
        i = int(t * SR)
        if i >= len(out):
            continue
        seg = s[: len(out) - i]
        out[i:i + len(seg)] += seg
    return out[: int(total * SR)].astype(np.float32)
