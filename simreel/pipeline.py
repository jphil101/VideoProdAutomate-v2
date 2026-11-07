"""End-to-end orchestration: events + script -> captioned, physics-simulated MP4."""
from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict
from pathlib import Path

from .captions import build_cards, caption_metrics, write_ass, write_srt, write_vtt
from .choreo import Choreographer
from .compose import compose
from .config import JobConfig
from .narration import synthesize
from .plan_schema import StoryPlan
from .planner import make_plan
from .sfx import SR as SFX_SR, render_sfx
from .stock import fetch_clip
from .structure import solve
from .text import split_sentences
from .utils import ensure_dir, log, write_wav

FONT_DIRS = ["/usr/share/fonts/opentype/inter", "/usr/share/fonts/truetype/inter", "/usr/share/fonts"]


def _work_dir(cfg: JobConfig) -> Path:
    if cfg.work_dir:
        return ensure_dir(cfg.work_dir)
    out = Path(cfg.output)
    return ensure_dir(out.parent / f".{out.stem}_work")


def load_inputs(cfg: JobConfig) -> tuple[str, str]:
    if not cfg.script_file:
        raise ValueError("a narration script is required (--script)")
    script = Path(cfg.script_file).read_text(encoding="utf-8")
    events = Path(cfg.events_file).read_text(encoding="utf-8") if cfg.events_file else ""
    if not events and not cfg.plan_file:
        raise ValueError("an events description (--events) or a plan (--plan) is required")
    return events, script


def plan_only(cfg: JobConfig) -> tuple[StoryPlan, list[str], list[str]]:
    events, script = load_inputs(cfg)
    sentences = split_sentences(script)
    if cfg.plan_file:
        plan = StoryPlan.model_validate_json(Path(cfg.plan_file).read_text())
        how = "file"
    else:
        plan, how = make_plan(events, script, sentences, cfg.planner)
    log(f"plan: {how} planner -> {len(plan.elements)} elements, {len(plan.events)} events")
    scene = solve(plan)
    return plan, sentences, scene.warnings


def run(cfg: JobConfig, render: bool = True) -> Path:
    W, H = cfg.video.size()
    work = _work_dir(cfg)
    out = Path(cfg.output)
    ensure_dir(out.parent)
    log(f"job: {W}x{H} {cfg.video.orientation} @ {cfg.video.fps} fps, quality={cfg.video.quality} "
        f"({cfg.video.resolved_engine()}), work dir {work}")

    # 1. plan --------------------------------------------------------------------
    log("1/7 planning")
    plan, sentences, warnings = plan_only(cfg)
    (work / "plan.json").write_text(plan.model_dump_json(indent=2))
    for w in warnings:
        log(f"   structure warning: {w}")
    scene = solve(plan)

    # 2. narration ---------------------------------------------------------------
    log(f"2/7 narration ({len(sentences)} sentences)")
    nar = synthesize(sentences, cfg.narration)

    # 3. choreography ------------------------------------------------------------
    log("3/7 choreography (crane kinematics, pendulum sway, timing, camera)")
    manifest, tl = Choreographer(plan, scene, nar, cfg, sentences).run()
    for n in tl.notes:
        log(f"   note: {n}")
    log(f"   timeline {tl.total:.1f}s, {manifest['frames']} frames, {len(tl.stock)} stock slot(s)")
    mpath = work / "manifest.json"
    mtext = json.dumps(manifest)
    mpath.write_text(mtext)
    timeline = dict(total=tl.total, fps=tl.fps, sentence_starts=tl.sentence_starts, words=tl.words, titles=tl.titles,
                    sfx=tl.sfx, stock=tl.stock, render_ranges=tl.render_ranges, notes=tl.notes,
                    windows=[dict(kind=w.kind, target=w.target, start=w.start, end=w.end) for w in tl.windows],
                    narration=dict(engine=nar.engine, aligner=nar.aligner))
    (work / "timeline.json").write_text(json.dumps(timeline, indent=2, default=float))

    # 4. stock footage -----------------------------------------------------------
    if tl.stock:
        log("4/7 stock footage")
        for st in tl.stock:
            if cfg.stock.provider == "none":
                log("   stock_clip events present but stock provider is 'none'; the 3D scene is shown instead")
                st["clip"] = None
                continue
            clip = fetch_clip(st["query"], st["end"] - st["start"], W, H, cfg.stock)
            st["clip"] = str(clip) if clip else None
    else:
        log("4/7 stock footage: none requested")
    # frames covered by missing stock clips must be rendered
    ranges = tl.render_ranges
    if any(st.get("clip") is None for st in tl.stock):
        n = manifest["frames"]
        ranges = [(0, n - 1)] if not any(st.get("clip") for st in tl.stock) else _ranges_without(
            [st for st in tl.stock if st.get("clip")], n, tl.fps, cfg.video.transition_s)
        manifest["render_ranges"] = ranges
        mtext = json.dumps(manifest)
        mpath.write_text(mtext)

    window = None
    if cfg.time_range:
        lo, hi = max(0.0, cfg.time_range[0]), min(tl.total, cfg.time_range[1])
        f0, f1 = int(round(lo * tl.fps)), int(round(hi * tl.fps)) - 1
        ranges = [(max(a, f0), min(b, f1)) for a, b in ranges if b >= f0 and a <= f1]
        tl.stock = [st for st in tl.stock if st["end"] > lo and st["start"] < hi]
        window = (lo, hi)
        manifest["render_ranges"] = ranges
        mtext = json.dumps(manifest)
        mpath.write_text(mtext)
        log(f"   rendering only {lo:.1f}-{hi:.1f}s ({f1 - f0 + 1} frames)")

    # 5. blender -----------------------------------------------------------------
    log("5/7 blender: build scene, bake rigid-body physics, render")
    # frames are keyed to the exact manifest: identical manifest -> finished frames are reused and an
    # interrupted render resumes where it stopped (Blender skips frames that already exist)
    digest = hashlib.sha1(mtext.encode()).hexdigest()
    frames = work / "frames"
    stamp = frames / ".manifest_sha1"
    complete = frames / ".complete"
    same = stamp.exists() and stamp.read_text().strip() == digest
    from .blender_runner import run_blender

    if same and complete.exists() and (work / "physics_qa.json").exists():
        log("   frames for this exact manifest already rendered; reusing them")
        qa = json.loads((work / "physics_qa.json").read_text())
    else:
        if frames.exists() and not same:
            shutil.rmtree(frames)
        elif same:
            log(f"   resuming an interrupted render ({len(list(frames.glob('f_*.png')))} frames already done)")
        ensure_dir(frames)
        stamp.write_text(digest)
        complete.unlink(missing_ok=True)
        qa = run_blender(mpath, work, cfg.blender, render=render)
        if render:
            complete.write_text("ok")
    bad = [q for q in qa["elements"] if q["status"] == "FAILED"]
    for q in qa["elements"]:
        log(f"   physics QA {q['id']}: {q['status']} ({q['error_m'] * 1000:.1f} mm, {q['error_deg']:.2f} deg)")
    if bad and cfg.physics.strict:
        raise RuntimeError(f"physics QA failed for {[q['id'] for q in bad]} (see {work / 'physics_qa.json'})")
    if not render:
        return work

    # 6. audio + captions --------------------------------------------------------
    log("6/7 audio mix and captions")
    track = nar.render(tl.sentence_starts, tl.total)
    nar_wav = work / "narration.wav"
    write_wav(nar_wav, track, nar.sr)
    sfx_wav = None
    if cfg.audio.sfx and tl.sfx:
        sfx_wav = work / "sfx.wav"
        write_wav(sfx_wav, render_sfx(tl.sfx, tl.total), SFX_SR)
    m = caption_metrics(W, H, cfg.captions)
    cards = build_cards(tl.words, m["max_chars"], cfg.captions.max_lines)
    srt = work / "captions.srt"
    write_srt(srt, cards)
    write_vtt(work / "captions.vtt", cards)
    burn = cfg.captions.mode in ("burn", "both")
    ass = work / "captions.ass"
    write_ass(ass, cards, tl.titles if cfg.captions.titles else [], W, H, cfg.captions, burn_captions=burn)
    use_ass = burn or (cfg.captions.titles and tl.titles)
    fonts = next((d for d in FONT_DIRS if Path(d).exists()), None)

    # 7. compose -----------------------------------------------------------------
    log("7/7 compositing final video")
    srt_use = srt if cfg.captions.mode in ("soft", "both") else None
    if window and srt_use:  # shift the soft track into the window's time base
        srt_use = work / "captions_window.srt"
        write_srt(srt_use, _shift_cards(cards, window))
    compose(cfg, work / "frames", ranges, tl.stock, tl.total, nar_wav, sfx_wav, ass if use_ass else None,
            srt_use, out, fonts, window=window)
    # sidecars next to the video
    shutil.copy(srt, out.with_suffix(".srt"))
    shutil.copy(work / "captions.vtt", out.with_suffix(".vtt"))
    shutil.copy(work / "plan.json", out.with_suffix(".plan.json"))
    shutil.copy(work / "physics_qa.json", out.with_suffix(".physics.json"))
    dur = (window[1] - window[0]) if window else tl.total
    log(f"done: {out}  ({dur:.1f}s, {W}x{H})")
    return out


def _shift_cards(cards, window):
    import copy

    lo, hi = window
    out = []
    for c in cards:
        if c.end <= lo or c.start >= hi:
            continue
        c2 = copy.copy(c)
        c2.start, c2.end = max(0.0, c.start - lo), min(hi, c.end) - lo
        out.append(c2)
    return out


def _ranges_without(stock, n, fps, pad):
    import numpy as np

    need = np.ones(n, bool)
    for st in stock:
        a, b = int((st["start"] + pad) * fps) + 1, int((st["end"] - pad) * fps)
        if b > a:
            need[a:b] = False
    out, i = [], 0
    while i < n:
        if need[i]:
            j = i
            while j < n and need[j]:
                j += 1
            out.append((i, j - 1))
            i = j
        else:
            i += 1
    return out
