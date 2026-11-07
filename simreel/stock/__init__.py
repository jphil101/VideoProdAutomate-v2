"""Stock footage cut-ins: Pexels, Pixabay, Envato Elements (via your existing
EnvatoDownloader.py session) or a local folder.  Clips are scored for relevance,
duration and orientation, cached, then fitted (crop or blur-pad) to the output frame."""
from __future__ import annotations

import hashlib
import os
import re
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path

from ..config import StockSettings
from ..text import norm
from ..utils import log, media_duration, probe

CACHE = Path(os.environ.get("SIMREEL_CACHE", Path.home() / ".cache" / "simreel")) / "stock"
VIDEO_EXT = {".mp4", ".mov", ".m4v", ".webm", ".mkv"}


@dataclass
class Candidate:
    url_or_path: str
    width: int
    height: int
    duration: float
    title: str = ""
    provider: str = ""

    def score(self, query: str, need: float, W: int, H: int) -> float:
        q = {norm(w) for w in query.split() if len(w) > 2}
        t = {norm(w) for w in re.split(r"[\s_\-./]+", self.title) if len(w) > 2}
        rel = len(q & t) / max(1, len(q))
        orient = 1.0 if (self.width >= self.height) == (W >= H) else 0.4
        res = min(1.0, min(self.width, self.height) / max(1, min(W, H)))
        dur = 1.0 if self.duration >= need else max(0.2, self.duration / need)
        return 2.0 * rel + orient + 0.7 * res + dur


def _download(url: str, suffix=".mp4") -> Path:
    import requests

    CACHE.mkdir(parents=True, exist_ok=True)
    out = CACHE / (hashlib.sha1(url.encode()).hexdigest()[:16] + suffix)
    if out.exists() and out.stat().st_size > 0:
        return out
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        tmp = out.with_suffix(".part")
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
        tmp.rename(out)
    return out


def search_pexels(query: str, orientation: str, n: int) -> list[Candidate]:
    import requests

    key = os.environ.get("PEXELS_API_KEY")
    if not key:
        raise RuntimeError("set PEXELS_API_KEY")
    r = requests.get("https://api.pexels.com/videos/search", headers={"Authorization": key},
                     params={"query": query, "orientation": orientation, "per_page": n, "size": "medium"}, timeout=30)
    r.raise_for_status()
    out = []
    for v in r.json().get("videos", []):
        files = [f for f in v.get("video_files", []) if f.get("file_type") == "video/mp4" and f.get("width")]
        if not files:
            continue
        best = max(files, key=lambda f: min(f["width"], f["height"]) if min(f["width"], f["height"]) <= 2160 else 0)
        title = v.get("url", "").rstrip("/").split("/")[-1]
        out.append(Candidate(best["link"], best["width"], best["height"], float(v.get("duration", 0)), title, "pexels"))
    return out


def search_pixabay(query: str, orientation: str, n: int) -> list[Candidate]:
    import requests

    key = os.environ.get("PIXABAY_API_KEY")
    if not key:
        raise RuntimeError("set PIXABAY_API_KEY")
    r = requests.get("https://pixabay.com/api/videos/", params={"key": key, "q": query, "per_page": max(3, n)},
                     timeout=30)
    r.raise_for_status()
    out = []
    for h in r.json().get("hits", []):
        vids = h.get("videos", {})
        for size in ("large", "medium", "small"):
            v = vids.get(size)
            if v and v.get("url"):
                out.append(Candidate(v["url"], v.get("width", 0), v.get("height", 0), float(h.get("duration", 0)),
                                     h.get("tags", ""), "pixabay"))
                break
    return out


def search_local(query: str, folder: str, n: int) -> list[Candidate]:
    out = []
    for p in sorted(Path(folder).rglob("*")):
        if p.suffix.lower() in VIDEO_EXT:
            try:
                info = probe(p)
                st = next(s for s in info["streams"] if s["codec_type"] == "video")
                out.append(Candidate(str(p), int(st["width"]), int(st["height"]), float(info["format"]["duration"]),
                                     p.stem, "local"))
            except Exception:
                continue
    return out


def search_envato(query: str, script: str | None, n: int) -> list[Candidate]:
    """Drive the user's existing Envato Elements downloader (Playwright, logged-in session)."""
    script = script or os.environ.get("ENVATO_DOWNLOADER")
    if not script or not Path(script).exists():
        raise RuntimeError("set stock.envato_script (path to EnvatoDownloader.py) or ENVATO_DOWNLOADER")
    seg = "simreel_" + hashlib.sha1(query.encode()).hexdigest()[:10]
    folder = Path(script).parent / "downloads" / seg
    have = folder.exists() and any(p.suffix.lower() in VIDEO_EXT for p in folder.rglob("*"))
    if not have:
        log(f"   envato: downloading '{query}' (a browser window may open for login)")
        py = os.environ.get("ENVATO_PYTHON", sys.executable)  # e.g. the downloader's own venv python
        subprocess.run([py, str(Path(script).name), "--term", query, "--segment", seg, "--count", str(n)],
                       cwd=str(Path(script).parent), check=False)
    for z in folder.glob("*.zip"):
        with zipfile.ZipFile(z) as zf:
            for m in zf.namelist():
                if Path(m).suffix.lower() in VIDEO_EXT and not (folder / Path(m).name).exists():
                    (folder / Path(m).name).write_bytes(zf.read(m))
    cands = search_local(query, str(folder), n) if folder.exists() else []
    for c in cands:
        c.provider = "envato"
        c.title = query + " " + c.title  # all results came from this search
    return cands


def fetch_clip(query: str, need: float, W: int, H: int, cfg: StockSettings) -> Path | None:
    orientation = "portrait" if H > W else ("square" if W == H else "landscape")
    try:
        if cfg.provider == "pexels":
            cands = search_pexels(query, orientation, cfg.max_candidates)
        elif cfg.provider == "pixabay":
            cands = search_pixabay(query, orientation, cfg.max_candidates)
        elif cfg.provider == "envato":
            cands = search_envato(query, cfg.envato_script, cfg.max_candidates)
        elif cfg.provider == "local":
            if not cfg.local_dir:
                raise RuntimeError("stock.local_dir is not set")
            cands = search_local(query, cfg.local_dir, cfg.max_candidates)
        else:
            return None
    except Exception as exc:
        log(f"   stock search failed ({cfg.provider}): {exc}")
        return None
    if not cands:
        log(f"   no stock results for '{query}'")
        return None
    best = max(cands, key=lambda c: c.score(query, need, W, H))
    log(f"   stock '{query}' -> {best.provider}: {best.title[:50]} ({best.width}x{best.height}, {best.duration:.1f}s)")
    if best.provider in ("local", "envato"):
        return Path(best.url_or_path)
    try:
        return _download(best.url_or_path)
    except Exception as exc:
        log(f"   stock download failed: {exc}")
        return None


def fit_filter(W: int, H: int, fps: int, mode: str, dur: float, src_dur: float) -> str:
    """ffmpeg filter chain to fit a clip to the frame and the slot duration."""
    speed = ""
    if src_dur and src_dur < dur:  # slow down (up to 1.6x) rather than freeze
        factor = min(1.6, dur / src_dur)
        speed = f"setpts={factor:.4f}*PTS,"
    if mode == "blurpad":
        return (f"{speed}split[a][b];[a]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},boxblur=20:2[bg];"
                f"[b]scale={W}:{H}:force_original_aspect_ratio=decrease[fg];[bg][fg]overlay=(W-w)/2:(H-h)/2,"
                f"fps={fps},format=yuv420p,setsar=1")
    return (f"{speed}scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},fps={fps},format=yuv420p,setsar=1")


def clip_duration(p: Path) -> float:
    try:
        return media_duration(p)
    except Exception:
        return 0.0
