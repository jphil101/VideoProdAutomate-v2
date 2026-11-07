"""Final assembly in a single ffmpeg pass: rendered frame ranges + fitted stock clips
(cross-faded at exact timeline offsets), narration + physics-timed SFX (+ optional
ducked music), burned-in captions and an embedded soft subtitle track."""
from __future__ import annotations

from pathlib import Path

from .config import JobConfig
from .stock import clip_duration, fit_filter
from .utils import ffmpeg_bin, log, run


def _esc_filter_path(p: str) -> str:
    return p.replace("\\", "/").replace(":", "\\:").replace("'", "\\'")


def compose(cfg: JobConfig, frames_dir: Path, render_ranges, stock_slots, total: float, narration_wav: Path,
            sfx_wav: Path | None, ass_path: Path | None, srt_path: Path | None, out_path: Path, fonts_dir: str | None,
            window: tuple[float, float] | None = None):
    W, H = cfg.video.size()
    fps = cfg.video.fps
    d = cfg.video.transition_s
    inputs: list[list[str]] = []
    segs = []  # (kind, input_index, out_start, duration, extra)
    for a, b in render_ranges:
        segs.append(("render", a / fps, (b - a + 1) / fps, (a, b)))
    for st in stock_slots:
        if st.get("clip"):
            segs.append(("stock", st["start"], st["end"] - st["start"], st))
    segs.sort(key=lambda s: s[1])

    fc = []
    vlabels = []
    for k, (kind, start, dur, extra) in enumerate(segs):
        idx = len(inputs)
        if kind == "render":
            a, b = extra
            inputs.append(["-framerate", str(fps), "-start_number", str(a + 1), "-i", str(frames_dir / "f_%05d.png")])
            fc.append(f"[{idx}:v]trim=end_frame={b - a + 1},setpts=PTS-STARTPTS,fps={fps},format=yuv420p,setsar=1[v{k}]")
        else:
            clip = Path(extra["clip"])
            sd = clip_duration(clip)
            skip = 0.4 if sd > dur + 1.0 else 0.0
            inputs.append(["-ss", f"{skip:.2f}", "-i", str(clip)])
            fit = fit_filter(W, H, fps, cfg.stock.fit, dur, max(0.1, sd - skip))
            fc.append(f"[{idx}:v]{fit},trim=duration={dur:.3f},setpts=PTS-STARTPTS[v{k}]")
        vlabels.append((f"v{k}", start, dur))
    # cross-fade chain at absolute offsets
    cur, cur_end = vlabels[0][0], vlabels[0][1] + vlabels[0][2]
    for k in range(1, len(vlabels)):
        lab, start, dur = vlabels[k]
        overlap = max(0.04, min(d, cur_end - start)) if cur_end > start else 0.04
        offset = max(0.0, cur_end - overlap)
        out = f"x{k}"
        fc.append(f"[{cur}][{lab}]xfade=transition=fade:duration={overlap:.3f}:offset={offset:.3f}[{out}]")
        cur, cur_end = out, offset + dur
    vchain = cur
    t0 = vlabels[0][1]  # timeline time of the first output frame (non-zero for --range)
    if ass_path is not None:
        fd = f":fontsdir='{_esc_filter_path(fonts_dir)}'" if fonts_dir else ""
        shift = f"setpts=PTS+{t0:.4f}/TB," if t0 > 0 else ""
        unshift = ",setpts=PTS-STARTPTS" if t0 > 0 else ""
        fc.append(f"[{vchain}]{shift}ass='{_esc_filter_path(str(ass_path))}'{fd}{unshift}[vout]")
        vchain = "vout"
    if window:
        total = window[1] - window[0]
    # audio
    a_in = len(inputs)
    inputs.append(["-i", str(narration_wav)])
    nar_lbl = "nar0" if cfg.audio.music_file else "nar"
    amix = [f"[{a_in}:a]aresample=48000,volume={cfg.audio.narration_gain_db}dB,apad[{nar_lbl}]"]
    mix_inputs = ["[nar]"]
    if sfx_wav is not None:
        s_in = len(inputs)
        inputs.append(["-i", str(sfx_wav)])
        amix.append(f"[{s_in}:a]aresample=48000,apad[sfx]")
        mix_inputs.append("[sfx]")
    if cfg.audio.music_file:
        m_in = len(inputs)
        inputs.append(["-stream_loop", "-1", "-i", cfg.audio.music_file])
        amix.append(f"[{m_in}:a]aresample=48000,volume={cfg.audio.music_gain_db}dB[mus0]")
        amix.append("[nar0]asplit=2[nar][key]")
        amix.append("[mus0][key]sidechaincompress=threshold=0.03:ratio=8:attack=20:release=400[mus]")
        mix_inputs.append("[mus]")
    a0 = window[0] if window else 0.0
    amix.append(f"{''.join(mix_inputs)}amix=inputs={len(mix_inputs)}:normalize=0:duration=longest,"
                f"atrim=start={a0:.3f}:duration={total:.3f},asetpts=PTS-STARTPTS,"
                f"loudnorm=I=-16:TP=-1.5:LRA=11,aresample=48000[aout]")
    fc.extend(amix)
    cmd = [ffmpeg_bin(), "-y", "-hide_banner", "-loglevel", "error", "-stats"]
    for inp in inputs:
        cmd += inp
    sub_idx = None
    if srt_path is not None:
        sub_idx = len(inputs)
        cmd += ["-i", str(srt_path)]
    cmd += ["-filter_complex", ";".join(fc), "-map", f"[{vchain}]", "-map", "[aout]"]
    if sub_idx is not None:
        cmd += ["-map", f"{sub_idx}:s", "-c:s", "mov_text", "-metadata:s:s:0", "language=eng",
                "-metadata:s:s:0", "title=English"]
    cmd += ["-c:v", "libx264", "-preset", cfg.video.x264_preset, "-crf", str(cfg.video.crf), "-pix_fmt", "yuv420p",
            "-r", str(fps), "-c:a", "aac", "-b:a", "192k", "-t", f"{total:.3f}", "-movflags", "+faststart",
            str(out_path)]
    log(f"   ffmpeg: {len(segs)} video segment(s), captions={'burned' if ass_path else 'no'}, "
        f"soft subs={'yes' if sub_idx is not None else 'no'}")
    run(cmd)
    return out_path
