from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
import wave
from pathlib import Path

import numpy as np

_T0 = time.time()


def log(msg: str = ""):
    print(f"[{time.time() - _T0:7.1f}s] {msg}", flush=True)


def ffmpeg_bin() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:  # optional pip fallback
        import static_ffmpeg

        static_ffmpeg.add_paths()
        exe = shutil.which("ffmpeg")
        if exe:
            return exe
    except Exception:
        pass
    raise RuntimeError("ffmpeg not found (install it, or `pip install static-ffmpeg`)")


def ffprobe_bin() -> str:
    ffmpeg_bin()
    exe = shutil.which("ffprobe")
    if not exe:
        raise RuntimeError("ffprobe not found")
    return exe


def run(cmd: list[str], quiet: bool = True, check: bool = True, **kw) -> subprocess.CompletedProcess:
    if not quiet:
        log("$ " + " ".join(str(c) for c in cmd))
    p = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, **kw)
    if check and p.returncode != 0:
        tail = (p.stderr or p.stdout or "")[-4000:]
        raise RuntimeError(f"command failed ({p.returncode}): {' '.join(map(str, cmd[:6]))} ...\n{tail}")
    return p


def probe(path: str | Path) -> dict:
    p = run([ffprobe_bin(), "-v", "error", "-print_format", "json", "-show_streams", "-show_format", str(path)])
    return json.loads(p.stdout)


def media_duration(path: str | Path) -> float:
    return float(probe(path)["format"]["duration"])


def read_audio(path: str | Path, sr: int | None = None) -> tuple[np.ndarray, int]:
    """Decode any audio file to float32 mono via ffmpeg."""
    if sr is None:
        info = probe(path)
        st = next(s for s in info["streams"] if s["codec_type"] == "audio")
        sr = int(st["sample_rate"])
    p = subprocess.run([ffmpeg_bin(), "-v", "error", "-i", str(path), "-f", "f32le", "-ac", "1", "-ar", str(sr), "-"],
                       capture_output=True, check=True)
    return np.frombuffer(p.stdout, dtype=np.float32).copy(), sr


def decode_bytes(data: bytes, sr: int = 24000) -> tuple[np.ndarray, int]:
    p = subprocess.run([ffmpeg_bin(), "-v", "error", "-i", "pipe:0", "-f", "f32le", "-ac", "1", "-ar", str(sr), "-"],
                       input=data, capture_output=True, check=True)
    return np.frombuffer(p.stdout, dtype=np.float32).copy(), sr


def write_wav(path: str | Path, x: np.ndarray, sr: int):
    x = np.clip(np.asarray(x, dtype=np.float32), -1, 1)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sr))
        w.writeframes((x * 32767).astype(np.int16).tobytes())


def time_stretch(x: np.ndarray, sr: int, rate: float) -> np.ndarray:
    """Pitch-preserving tempo change via ffmpeg atempo."""
    if abs(rate - 1.0) < 1e-3:
        return x
    filters = []
    r = rate
    while r > 2.0:
        filters.append("atempo=2.0")
        r /= 2.0
    while r < 0.5:
        filters.append("atempo=0.5")
        r /= 0.5
    filters.append(f"atempo={r:.4f}")
    p = subprocess.run([ffmpeg_bin(), "-v", "error", "-f", "f32le", "-ar", str(sr), "-ac", "1", "-i", "pipe:0",
                        "-af", ",".join(filters), "-f", "f32le", "-"], input=x.astype(np.float32).tobytes(),
                       capture_output=True, check=True)
    return np.frombuffer(p.stdout, dtype=np.float32).copy()


def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def python_has(module: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(module) is not None


def die(msg: str, code: int = 2):
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(code)
