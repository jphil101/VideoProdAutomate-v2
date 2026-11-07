"""simreel command line.

  simreel make   --events events.md --script script.md [options]   full video
  simreel plan   --events events.md --script script.md -o plan.json  plan only (review / hand-edit)
  simreel wizard                                                      interactive parameter selection
  simreel check                                                       environment diagnostics
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from .config import RESOLUTION_PRESETS, JobConfig
from .utils import die, log, python_has


def _add_common(p: argparse.ArgumentParser):
    p.add_argument("--job", help="YAML/JSON job file with any of the settings below")
    p.add_argument("--events", help="description of the construction events (text/markdown)")
    p.add_argument("--script", help="narration script (text/markdown)")
    p.add_argument("--plan", help="use this plan.json instead of planning (e.g. a hand-edited one)")
    p.add_argument("--planner", choices=["auto", "claude", "heuristic"])
    p.add_argument("--model", help="Claude model for planning (default claude-opus-5-5)")


def _add_make(p: argparse.ArgumentParser):
    g = p.add_argument_group("video")
    g.add_argument("-o", "--out", help="output .mp4 path")
    g.add_argument("--resolution", help=f"{', '.join(RESOLUTION_PRESETS)} or WxH (e.g. 1080x1350)")
    g.add_argument("--orientation", choices=["landscape", "portrait", "square"])
    g.add_argument("--fps", type=int)
    g.add_argument("--quality", choices=["draft", "preview", "final"],
                   help="draft=Workbench (fast), preview=Cycles low samples, final=Cycles high samples")
    g.add_argument("--engine", choices=["cycles", "eevee", "workbench"])
    g.add_argument("--samples", type=int)
    g.add_argument("--gpu", choices=["auto", "off"])
    g.add_argument("--crf", type=int)
    g = p.add_argument_group("narration")
    g.add_argument("--tts", choices=["auto", "elevenlabs", "edge", "kokoro", "pico", "espeak", "file", "none"])
    g.add_argument("--voice")
    g.add_argument("--voiceover", help="recorded narration audio (implies --tts file)")
    g.add_argument("--rate", type=float, help="speech rate multiplier")
    g.add_argument("--aligner", choices=["auto", "native", "whisper", "dtw", "estimate"])
    g = p.add_argument_group("captions")
    g.add_argument("--captions", choices=["burn", "soft", "both", "none"])
    g.add_argument("--caption-style", choices=["karaoke", "plain"])
    g.add_argument("--caption-position", choices=["bottom", "middle", "top"])
    g.add_argument("--font")
    g.add_argument("--caption-scale", type=float)
    g.add_argument("--highlight-color")
    g.add_argument("--caption-box", action="store_true", help="opaque box behind captions")
    g.add_argument("--no-titles", action="store_true", help="no lower-third element callouts")
    g = p.add_argument_group("stock footage")
    g.add_argument("--stock", choices=["none", "local", "pexels", "pixabay", "envato"])
    g.add_argument("--stock-dir", help="folder of clips for --stock local")
    g.add_argument("--envato-script", help="path to EnvatoDownloader.py for --stock envato")
    g.add_argument("--stock-fit", choices=["crop", "blurpad"])
    g = p.add_argument_group("audio")
    g.add_argument("--music", help="background music file (auto-ducked under narration)")
    g.add_argument("--no-sfx", action="store_true")
    g = p.add_argument_group("physics")
    g.add_argument("--motion-speed", type=float, help="crane speed multiplier (still acceleration limited)")
    g.add_argument("--no-anti-sway", action="store_true", help="disable input-shaped crane motion")
    g.add_argument("--strict-physics", action="store_true", help="fail if any body does not settle on target")
    g = p.add_argument_group("misc")
    g.add_argument("--range", help="render only this part of the timeline, e.g. 20-32 (seconds)")
    g.add_argument("--work-dir")
    g.add_argument("--blender", help="path to a Blender executable (default: bpy module or `blender` on PATH)")


def build_config(a: argparse.Namespace) -> JobConfig:
    cfg = JobConfig.load(a.job) if getattr(a, "job", None) else JobConfig()
    v, n, c, s, au, ph, pl = cfg.video, cfg.narration, cfg.captions, cfg.stock, cfg.audio, cfg.physics, cfg.planner

    def put(obj, attr, val):
        if val is not None:
            setattr(obj, attr, val)

    put(cfg, "events_file", getattr(a, "events", None))
    put(cfg, "script_file", getattr(a, "script", None))
    put(cfg, "plan_file", getattr(a, "plan", None))
    put(pl, "planner", getattr(a, "planner", None))
    put(pl, "model", getattr(a, "model", None))
    if not hasattr(a, "resolution"):
        return cfg
    put(cfg, "output", a.out)
    put(cfg, "work_dir", a.work_dir)
    put(cfg, "blender", a.blender)
    if a.range:
        lo, hi = (float(x) for x in a.range.split("-", 1))
        if hi <= lo:
            raise ValueError("--range must be START-END with END > START")
        cfg.time_range = (lo, hi)
    if a.resolution:
        v.resolution = v._check_res(a.resolution)
    for attr in ("orientation", "fps", "quality", "engine", "samples", "gpu", "crf"):
        put(v, attr, getattr(a, attr))
    put(n, "tts", a.tts)
    put(n, "voice", a.voice)
    put(n, "rate", a.rate)
    put(n, "aligner", a.aligner)
    if a.voiceover:
        n.tts, n.voiceover_file = "file", a.voiceover
    put(c, "mode", a.captions)
    put(c, "style", a.caption_style)
    put(c, "position", a.caption_position)
    put(c, "font", a.font)
    put(c, "size_scale", a.caption_scale)
    put(c, "highlight_color", a.highlight_color)
    if a.caption_box:
        c.box = True
    if a.no_titles:
        c.titles = False
    put(s, "provider", a.stock)
    put(s, "local_dir", a.stock_dir)
    put(s, "envato_script", a.envato_script)
    put(s, "fit", a.stock_fit)
    put(au, "music_file", a.music)
    if a.no_sfx:
        au.sfx = False
    put(ph, "motion_speed", a.motion_speed)
    if a.no_anti_sway:
        ph.anti_sway = False
    if a.strict_physics:
        ph.strict = True
    return cfg


def cmd_check(_a):
    from .narration import espeak
    from .narration.engines import ENGINES
    from .planner.llm import credentials_available

    rows = []
    try:
        from .blender_runner import find_blender

        rows.append(("blender", " ".join(find_blender(None)[:2])))
    except Exception as exc:
        rows.append(("blender", f"MISSING - {exc}"))
    rows.append(("ffmpeg", shutil.which("ffmpeg") or "MISSING"))
    for name, cls in ENGINES.items():
        rows.append((f"tts:{name}", "available" if cls.available() else "not available"))
    rows.append(("aligner:whisper", "available" if python_has("faster_whisper") else "not installed (pip install faster-whisper)"))
    rows.append(("aligner:dtw (espeak ref)", "available" if espeak.available() else "not available (apt install espeak-ng)"))
    rows.append(("planner:claude", "credentials found" if credentials_available() else "no credentials (heuristic planner will be used)"))
    import os

    for k in ("ELEVENLABS_API_KEY", "PEXELS_API_KEY", "PIXABAY_API_KEY"):
        rows.append((k, "set" if os.environ.get(k) else "not set"))
    w = max(len(r[0]) for r in rows)
    for k, v in rows:
        print(f"  {k:<{w}}  {v}")


def cmd_plan(a):
    from .pipeline import plan_only

    cfg = build_config(a)
    plan, sentences, warnings = plan_only(cfg)
    out = Path(a.output or "plan.json")
    out.write_text(plan.model_dump_json(indent=2))
    print(f"wrote {out}")
    for i, s in enumerate(sentences):
        evs = [e for e in plan.events if e.sentence == i]
        desc = "; ".join(f"{e.action} {','.join(e.targets)}" + (f" @'{e.cue_word}'" if e.cue_word else "") for e in evs)
        print(f"  [{i}] {s[:70]}{'...' if len(s) > 70 else ''}\n       -> {desc or '(camera only)'}")
    for w in warnings:
        print(f"  warning: {w}")


def cmd_make(a):
    from .pipeline import run

    cfg = build_config(a)
    run(cfg)


def _ask(prompt, choices=None, default=None):
    sfx = f" [{'/'.join(choices)}]" if choices else ""
    d = f" (default {default})" if default is not None else ""
    while True:
        v = input(f"{prompt}{sfx}{d}: ").strip()
        if not v and default is not None:
            return default
        if not choices or v in choices:
            return v
        print("  please choose one of", ", ".join(choices))


def cmd_wizard(_a):
    print("simreel wizard -- press Enter to accept defaults\n")
    cfg = JobConfig()
    cfg.events_file = _ask("Events description file", default="examples/modular_frame/events.md")
    cfg.script_file = _ask("Narration script file", default="examples/modular_frame/script.md")
    cfg.output = _ask("Output video", default="output/video.mp4")
    cfg.video.orientation = _ask("Orientation", ["landscape", "portrait", "square"], "landscape")
    cfg.video.resolution = cfg.video._check_res(_ask("Resolution (preset or WxH)", default="1080p"))
    cfg.video.fps = int(_ask("Frames per second", ["24", "25", "30", "60"], "30"))
    cfg.video.quality = _ask("Quality", ["draft", "preview", "final"], "preview")
    cfg.narration.tts = _ask("Voice engine", ["auto", "elevenlabs", "edge", "kokoro", "pico", "espeak", "file", "none"], "auto")
    if cfg.narration.tts == "file":
        cfg.narration.voiceover_file = _ask("Voice-over audio file")
    cfg.captions.mode = _ask("Captions", ["burn", "soft", "both", "none"], "both")
    cfg.captions.style = _ask("Caption style", ["karaoke", "plain"], "karaoke")
    cfg.stock.provider = _ask("Stock footage provider", ["none", "local", "pexels", "pixabay", "envato"], "none")
    if cfg.stock.provider == "local":
        cfg.stock.local_dir = _ask("Folder with clips")
    if cfg.stock.provider == "envato":
        cfg.stock.envato_script = _ask("Path to EnvatoDownloader.py", default="EnvatoDownloader.py")
    job = Path(_ask("Save these settings as", default="job.yaml"))
    import yaml

    job.write_text(yaml.safe_dump(json.loads(cfg.model_dump_json()), sort_keys=False))
    print(f"saved {job}")
    if _ask("Render now?", ["y", "n"], "y") == "y":
        from .pipeline import run

        run(cfg)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="simreel", description="Events + script -> physics-simulated, captioned video")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("make", help="produce the full video")
    _add_common(p)
    _add_make(p)
    p.set_defaults(func=cmd_make)
    p = sub.add_parser("plan", help="plan only; review/edit the JSON and pass it back with --plan")
    _add_common(p)
    p.add_argument("-o", "--output", help="plan output path (default plan.json)")
    p.set_defaults(func=cmd_plan)
    p = sub.add_parser("wizard", help="interactive parameter selection")
    p.set_defaults(func=cmd_wizard)
    p = sub.add_parser("check", help="show which engines/providers are available")
    p.set_defaults(func=cmd_check)
    a = ap.parse_args(argv)
    try:
        a.func(a)
    except KeyboardInterrupt:
        die("interrupted", 130)
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        log(f"error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
