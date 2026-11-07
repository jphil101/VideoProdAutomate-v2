"""User-selectable job/render settings.

Every option can come from (lowest to highest precedence):
  built-in defaults  ->  job file (YAML/JSON)  ->  command-line flags.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

RESOLUTION_PRESETS = {
    # name: long edge x short edge
    "360p": (640, 360),
    "480p": (854, 480),
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    "1440p": (2560, 1440),
    "2160p": (3840, 2160),
    "4k": (3840, 2160),
}

Orientation = Literal["landscape", "portrait", "square"]
Quality = Literal["draft", "preview", "final"]


class VideoSettings(BaseModel):
    resolution: str = Field("1080p", description="Preset (360p..2160p/4k) or explicit WxH, e.g. 1080x1350")
    orientation: Orientation = "landscape"
    fps: int = 30
    quality: Quality = "preview"
    engine: Optional[Literal["cycles", "eevee", "workbench"]] = None  # None -> chosen by quality
    samples: Optional[int] = None  # Cycles samples override
    gpu: Literal["auto", "off"] = "auto"
    crf: int = 18
    x264_preset: str = "medium"
    intro_s: float = 1.2  # establishing time before first sentence
    outro_s: float = 2.5  # hold after last sentence
    transition_s: float = 0.5  # crossfade between render and stock shots

    @field_validator("resolution")
    @classmethod
    def _check_res(cls, v: str) -> str:
        v = v.strip().lower()
        if v in RESOLUTION_PRESETS:
            return v
        if "x" in v:
            w, h = v.split("x", 1)
            if int(w) > 0 and int(h) > 0:
                return v
        raise ValueError(f"resolution must be one of {list(RESOLUTION_PRESETS)} or WxH")

    def size(self) -> tuple[int, int]:
        """Final (width, height), even numbers, honouring orientation for presets."""
        if self.resolution in RESOLUTION_PRESETS:
            long_e, short_e = RESOLUTION_PRESETS[self.resolution]
            if self.orientation == "landscape":
                w, h = long_e, short_e
            elif self.orientation == "portrait":
                w, h = short_e, long_e
            else:
                w = h = short_e
        else:
            w, h = (int(x) for x in self.resolution.split("x", 1))
        return w - (w % 2), h - (h % 2)

    def resolved_engine(self) -> str:
        if self.engine:
            return self.engine
        return {"draft": "workbench", "preview": "cycles", "final": "cycles"}[self.quality]

    def resolved_samples(self) -> int:
        if self.samples:
            return self.samples
        return {"draft": 1, "preview": 24, "final": 160}[self.quality]


class NarrationSettings(BaseModel):
    tts: Literal["auto", "elevenlabs", "edge", "kokoro", "pico", "espeak", "file", "none"] = "auto"
    voice: Optional[str] = None  # engine-specific voice name / id
    voiceover_file: Optional[str] = None  # used when tts == "file"
    rate: float = 1.0  # speech-rate multiplier
    sentence_gap_s: float = 0.35  # minimum silence between sentences
    aligner: Literal["auto", "native", "whisper", "dtw", "estimate"] = "auto"
    whisper_model: str = "base.en"


class CaptionSettings(BaseModel):
    mode: Literal["burn", "soft", "both", "none"] = "both"
    style: Literal["karaoke", "plain"] = "karaoke"
    font: str = "Inter"
    font_weight_bold: bool = True
    size_scale: float = 1.0  # multiplier on the auto-computed font size
    color: str = "#FFFFFF"
    highlight_color: str = "#FFC629"
    outline_color: str = "#101010"
    box: bool = False  # opaque box behind text instead of outline
    position: Literal["bottom", "middle", "top"] = "bottom"
    max_lines: int = 2
    max_chars_per_line: Optional[int] = None  # None -> by orientation
    titles: bool = True  # lower-third element callouts


class StockSettings(BaseModel):
    provider: Literal["none", "local", "pexels", "pixabay", "envato"] = "none"
    local_dir: Optional[str] = None
    envato_script: Optional[str] = None  # path to EnvatoDownloader.py
    fit: Literal["crop", "blurpad"] = "crop"
    max_candidates: int = 6


class AudioSettings(BaseModel):
    sfx: bool = True
    music_file: Optional[str] = None
    music_gain_db: float = -20.0
    narration_gain_db: float = 0.0


class PhysicsSettings(BaseModel):
    anti_sway: bool = True  # input-shaped crane motion (as real anti-sway controllers do)
    motion_speed: float = 1.0  # >1 = faster crane (still acceleration-limited)
    release_gap_m: float = 0.005  # dynamic bodies are released this far above their seat
    substeps: int = 12  # Bullet substeps per frame
    solver_iterations: int = 30
    strict: bool = False  # fail the job if a body does not settle at its target


class PlannerSettings(BaseModel):
    planner: Literal["auto", "claude", "heuristic"] = "auto"
    model: str = "claude-opus-5-5"
    effort: Literal["low", "medium", "high", "xhigh", "max"] = "high"


class JobConfig(BaseModel):
    events_file: Optional[str] = None
    script_file: Optional[str] = None
    plan_file: Optional[str] = None  # skip planning and use an existing (possibly hand-edited) plan
    output: str = "output/video.mp4"
    work_dir: Optional[str] = None
    video: VideoSettings = VideoSettings()
    narration: NarrationSettings = NarrationSettings()
    captions: CaptionSettings = CaptionSettings()
    stock: StockSettings = StockSettings()
    audio: AudioSettings = AudioSettings()
    physics: PhysicsSettings = PhysicsSettings()
    planner: PlannerSettings = PlannerSettings()
    blender: Optional[str] = None  # path to a blender executable; default: bpy module / PATH
    time_range: Optional[tuple[float, float]] = None  # render/export only this window (seconds)

    @classmethod
    def load(cls, path: str | os.PathLike) -> "JobConfig":
        p = Path(path)
        text = p.read_text()
        if p.suffix.lower() in (".yaml", ".yml"):
            import yaml

            data = yaml.safe_load(text) or {}
        else:
            data = json.loads(text)
        cfg = cls.model_validate(data)
        # resolve relative paths against the job file location
        base = p.parent
        for attr in ("events_file", "script_file", "plan_file"):
            v = getattr(cfg, attr)
            if v and not Path(v).is_absolute():
                setattr(cfg, attr, str((base / v).resolve()))
        return cfg
