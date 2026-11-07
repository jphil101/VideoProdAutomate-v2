"""
SubTheme processor — discovers sub-theme images, matches them to script segments,
generates parallax motion videos, and builds interleaved segment videos.

Sub-theme images are placed in theme_images/SubThemes/.  Their filenames (minus
extension, with underscores converted to spaces) are matched case-insensitively
against segment voiceover text.  Each sub-theme matches only its FIRST matching
segment.

The generated 2-second parallax clips are inserted into the segment timeline at
the temporal position where the matching words are spoken.
"""

import os
import re
import subprocess
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SubTheme:
    image_path: str
    keywords_str: str        # e.g. "calcium lactate"
    keyword_words: list      # e.g. ["calcium", "lactate"]


@dataclass
class Clip:
    clip_type: str           # 'main' or 'subtheme'
    start_time: float        # position in segment timeline (seconds)
    end_time: float
    video_path: str          # source video path
    main_video_offset: float = 0.0  # only for 'main': offset into the Envato video


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".tiff", ".tif"}
SUBTHEME_DURATION = 2.0   # each sub-theme clip is 2 seconds
MIN_CLIP_DURATION = 2.0   # ideal minimum for any clip
ABSOLUTE_MIN_CLIP = 1.0   # below this, a main clip is dropped entirely


# ---------------------------------------------------------------------------
# 1.  Discover sub-theme images
# ---------------------------------------------------------------------------

def discover_subthemes(script_hash: str = None) -> List[SubTheme]:
    """Scan SubThemes/ for images and extract keywords from filenames."""
    if script_hash:
        subthemes_dir = Path(f"theme_images_{script_hash}") / "SubThemes"
    else:
        subthemes_dir = Path("theme_images") / "SubThemes"
        
    subthemes_dir.mkdir(parents=True, exist_ok=True)

    subthemes = []
    for f in sorted(subthemes_dir.iterdir()):
        if f.is_file() and f.suffix.lower() in VALID_IMAGE_EXTS:
            stem = f.stem  # filename without extension
            # Convert underscores and multiple spaces to single spaces
            keywords_str = stem.replace("_", " ").strip()
            keywords_str = re.sub(r'\s+', ' ', keywords_str)
            keyword_words = keywords_str.lower().split()
            subthemes.append(SubTheme(
                image_path=str(f),
                keywords_str=keywords_str,
                keyword_words=keyword_words,
            ))

    return subthemes


# ---------------------------------------------------------------------------
# 2.  Match sub-themes → segments  (first match only per sub-theme)
# ---------------------------------------------------------------------------

def match_subthemes_to_segments(
    subthemes: List[SubTheme],
    segments: list,
) -> Dict[str, List[Tuple[SubTheme, int]]]:
    """Match sub-theme keywords against segment voiceover text.

    Each sub-theme matches only its FIRST matching segment.

    Returns:
        {seg_id: [(subtheme, char_start_position_in_text), ...]}
    """
    matches: Dict[str, List[Tuple[SubTheme, int]]] = {}
    matched_subtheme_ids: set = set()

    for seg in segments:
        seg_id = seg["segment_id"]
        text_lower = seg["voiceover_text"].lower()

        for st in subthemes:
            if id(st) in matched_subtheme_ids:
                continue

            pattern = re.compile(re.escape(st.keywords_str.lower()))
            match = pattern.search(text_lower)

            if match:
                matches.setdefault(seg_id, []).append((st, match.start()))
                matched_subtheme_ids.add(id(st))

    # Sort matches within each segment by position in text
    for seg_id in matches:
        matches[seg_id].sort(key=lambda x: x[1])

    return matches


# ---------------------------------------------------------------------------
# 3.  Generate parallax motion videos for matched sub-themes
# ---------------------------------------------------------------------------

def generate_subtheme_videos(
    matches: Dict[str, List[Tuple[SubTheme, int]]],
) -> Dict[str, str]:
    """Generate parallax motion videos for all matched sub-theme images.

    Returns:
        {subtheme_image_path: generated_video_path}
    """
    import subprocess

    # Collect unique sub-themes that need videos
    all_image_paths: set = set()
    for seg_matches in matches.values():
        for st, _ in seg_matches:
            all_image_paths.add(st.image_path)

    if not all_image_paths:
        return {}

    video_map: Dict[str, str] = {}

    # Check cache — skip images whose videos already exist
    to_generate: list = []
    for img_path in sorted(all_image_paths):
        stem = Path(img_path).stem
        output_dir = Path(img_path).parent / "generated_videos"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(output_dir / f"{stem}_motion.mp4")
        if os.path.exists(output_path):
            print(f"   SubTheme video already exists: {output_path}")
            video_map[img_path] = output_path
        else:
            to_generate.append((img_path, output_path))

    if to_generate:
        import sys
        current_dir = Path(__file__).resolve().parent
        parallax_repo_dir = current_dir.parent / "ImgToMotionTest"
        
        # Use the host Python environment so it automatically inherits the 
        # cross-platform pendrive_libs from the parent run_portable.sh script
        python_exec = sys.executable
        script_path = str(parallax_repo_dir / "generate_parallax.py")
        
        if not os.path.exists(script_path):
            print(f"   [!] Cannot find ImgToMotionTest script at {script_path}. Skipping parallax generation.")
            return video_map

        for img_path, output_path in to_generate:
            print(f"   Generating parallax video for {Path(img_path).name} (using Python 3.9 venv)...")
            cmd = [
                python_exec, script_path,
                "--image", img_path,
                "--out", output_path,
                "--duration", str(SUBTHEME_DURATION),
                "--shorts"
            ]
            try:
                subprocess.run(cmd, check=True)
                video_map[img_path] = output_path
            except subprocess.CalledProcessError as e:
                print(f"   [!] Parallax generation failed for {img_path}: {e}")

    return video_map


# ---------------------------------------------------------------------------
# 4.  Compute interleave plan for a segment
# ---------------------------------------------------------------------------

def _char_weight(s: str) -> float:
    """Weight a string for timing interpolation.

    Matches the weighting logic used in create_ass() so sub-theme insertion
    aligns with subtitle timing.
    """
    if not s:
        return 0
    weight = len(s)
    weight += s.count(',') * 8   # commas ≈ pause
    weight += s.count('.') * 12  # periods ≈ longer pause
    weight += s.count(' ') * 2   # spaces ≈ tiny pause
    return weight


def compute_interleave_plan(
    segment: dict,
    matches_for_seg: List[Tuple[SubTheme, int]],
    segment_duration: float,
    video_map: Dict[str, str],
) -> List[Clip]:
    """Compute how to interleave sub-theme videos into a segment's timeline.

    Returns an ordered list of Clip objects.
    """
    text = segment["voiceover_text"]

    # Calculate temporal insertion point for each sub-theme
    insertion_points: list = []
    for st, char_pos in matches_for_seg:
        before_text = text[:char_pos]
        keyword_text = text[char_pos:char_pos + len(st.keywords_str)]

        w_before = _char_weight(before_text)
        w_keyword = _char_weight(keyword_text)
        w_total = _char_weight(text)

        if w_total > 0:
            # Insert at the start of the keyword (where the word begins being spoken)
            t_insert = w_before / w_total * segment_duration
        else:
            t_insert = 0.0

        subtheme_video = video_map.get(st.image_path)
        if subtheme_video:
            insertion_points.append((t_insert, st, subtheme_video))

    if not insertion_points:
        return [Clip('main', 0, segment_duration, '', 0.0)]

    insertion_points.sort(key=lambda x: x[0])

    # Check if segment can accommodate all sub-themes
    total_subtheme_time = SUBTHEME_DURATION * len(insertion_points)
    if total_subtheme_time > segment_duration:
        # Too many sub-themes — keep only as many as fit
        max_st = max(1, int(segment_duration / SUBTHEME_DURATION))
        insertion_points = insertion_points[:max_st]
        total_subtheme_time = SUBTHEME_DURATION * len(insertion_points)

    if len(insertion_points) == 1:
        return _plan_single_insertion(
            insertion_points[0][0], segment_duration, insertion_points[0][2],
        )
    else:
        return _plan_multiple_insertions(insertion_points, segment_duration)


def _plan_single_insertion(
    t_insert: float,
    segment_duration: float,
    video_path: str,
) -> List[Clip]:
    """Plan interleaving for a single sub-theme insertion."""
    D = segment_duration
    ST = SUBTHEME_DURATION
    MIN = MIN_CLIP_DURATION

    if D < ST:
        # Segment too short for even the sub-theme — skip it
        return [Clip('main', 0, D, '', 0.0)]

    remaining = D - ST  # total time available for main clips

    # --- Case 1: not enough room for a meaningful main clip alongside sub-theme ---
    if remaining < ABSOLUTE_MIN_CLIP:
        # Sub-theme takes the entire segment
        return [Clip('subtheme', 0, min(ST, D), video_path, 0.0)]

    # --- Case 2: remaining is between ABSOLUTE_MIN and MIN ---
    # Allow one short main clip; place sub-theme at beginning or end to give
    # main the full 'remaining' as a single chunk.
    if remaining < MIN:
        if t_insert <= D / 2:
            # Word is in first half → sub-theme at beginning, main after
            return [
                Clip('subtheme', 0, ST, video_path, 0.0),
                Clip('main', ST, D, '', 0.0),
            ]
        else:
            # Word is in second half → main first, sub-theme at end
            return [
                Clip('main', 0, remaining, '', 0.0),
                Clip('subtheme', remaining, D, video_path, 0.0),
            ]

    # --- Case 3: enough room for MIN-duration main clips on each side ---
    main_a = t_insert
    main_b = D - t_insert - ST

    # Enforce minimum: if a main clip is non-zero but too short, snap sub-theme
    # to the edge so the other main clip absorbs the full remaining time.
    if 0 < main_a < MIN:
        # Main-A too short → snap sub-theme to beginning
        main_a = 0.0
        main_b = remaining

    if 0 < main_b < MIN:
        # Main-B too short → snap sub-theme to end
        main_b = 0.0
        main_a = remaining

    # Build clip list
    clips = []
    cursor = 0.0
    main_cursor = 0.0

    if main_a > 0:
        clips.append(Clip('main', cursor, cursor + main_a, '', main_cursor))
        cursor += main_a
        main_cursor += main_a

    clips.append(Clip('subtheme', cursor, cursor + ST, video_path, 0.0))
    cursor += ST

    if main_b > 0:
        clips.append(Clip('main', cursor, cursor + main_b, '', main_cursor))

    return clips


def _plan_multiple_insertions(
    insertion_points: list,
    segment_duration: float,
) -> List[Clip]:
    """Plan interleaving for multiple sub-theme insertions in one segment."""
    D = segment_duration
    ST = SUBTHEME_DURATION
    MIN = MIN_CLIP_DURATION
    n = len(insertion_points)

    # Start with natural positions, then adjust for overlap / min-gap constraints.
    adjusted = [t for t, _, _ in insertion_points]

    # Forward pass: push positions forward to avoid overlaps and tiny main gaps
    for i in range(n):
        if i == 0:
            if 0 < adjusted[i] < MIN:
                adjusted[i] = 0.0
        else:
            prev_end = adjusted[i - 1] + ST
            gap = adjusted[i] - prev_end
            if gap < 0:
                adjusted[i] = prev_end  # remove overlap
            elif 0 < gap < ABSOLUTE_MIN_CLIP:
                adjusted[i] = prev_end  # collapse tiny gap

        # Don't let sub-theme extend past segment end
        if adjusted[i] + ST > D:
            adjusted[i] = max(0.0, D - ST)

    # Backward pass: ensure trailing main clip is valid
    last_end = adjusted[-1] + ST
    trailing = D - last_end
    if 0 < trailing < ABSOLUTE_MIN_CLIP:
        adjusted[-1] = max(0.0, D - ST)

    # Build clips
    clips = []
    cursor = 0.0
    main_cursor = 0.0

    for i, (_, st, video_path) in enumerate(insertion_points):
        pos = adjusted[i]

        # Main clip before this sub-theme
        main_dur = pos - cursor
        if main_dur > 0:
            clips.append(Clip('main', cursor, cursor + main_dur, '', main_cursor))
            main_cursor += main_dur
            cursor += main_dur

        # Sub-theme clip
        clips.append(Clip('subtheme', cursor, cursor + ST, video_path, 0.0))
        cursor += ST

    # Trailing main clip after last sub-theme
    if cursor < D:
        remaining = D - cursor
        clips.append(Clip('main', cursor, D, '', main_cursor))

    return clips


# ---------------------------------------------------------------------------
# 5.  Build the interleaved segment video with FFmpeg
# ---------------------------------------------------------------------------

def build_interleaved_segment(
    clips: List[Clip],
    main_video_path: str,
    audio_path: Optional[str],
    duration: float,
    seg_id: str,
    script_hash: str,
) -> Optional[str]:
    """Build a segment video with interleaved sub-theme clips.

    Produces the same output format as process_segment_video() in
    main_workflow.py (1080×1920, 30fps, libx264, AAC audio).

    Returns:
        Path to the output video, or None on failure.
    """
    output_path = f"temp_{script_hash}_{seg_id}.mp4"

    if os.path.exists(output_path):
        print(f"   Interleaved video already exists for {seg_id} -> {output_path}")
        return output_path

    print(f"   Building interleaved segment for {seg_id} ({len(clips)} clips)...")

    temp_parts = []

    for i, clip in enumerate(clips):
        part_path = f"temp_{script_hash}_{seg_id}_part{i}.mp4"
        clip_duration = clip.end_time - clip.start_time

        if clip_duration <= 0.01:
            continue

        if clip.clip_type == 'main':
            # Trim & normalize the Envato source video
            cmd = [
                "ffmpeg", "-y",
                "-stream_loop", "-1",
                "-i", main_video_path,
                "-ss", f"{clip.main_video_offset:.3f}",
                "-t", f"{clip_duration:.3f}",
                "-vf", "scale=1080:1920:force_original_aspect_ratio=increase,"
                       "crop=1080:1920,fps=30,setsar=1,format=yuv420p",
                "-c:v", "libx264",
                "-preset", "fast",
                "-an",
                part_path,
            ]
        else:
            # Sub-theme parallax video — re-encode to ensure uniform params
            cmd = [
                "ffmpeg", "-y",
                "-stream_loop", "-1",
                "-i", clip.video_path,
                "-t", f"{clip_duration:.3f}",
                "-vf", "scale=1080:1920:force_original_aspect_ratio=increase,"
                       "crop=1080:1920,fps=30,setsar=1,format=yuv420p",
                "-c:v", "libx264",
                "-preset", "fast",
                "-an",
                part_path,
            ]

        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        temp_parts.append(part_path)

    if not temp_parts:
        print(f"   [!] No clips produced for {seg_id} — falling back to normal processing.")
        return None

    # Concatenate all parts into a single silent video
    concat_list = f"temp_{script_hash}_{seg_id}_concat.txt"
    with open(concat_list, "w") as f:
        for p in temp_parts:
            f.write(f"file '{p}'\n")

    concat_path = f"temp_{script_hash}_{seg_id}_concat.mp4"
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", concat_list,
        "-c", "copy",
        concat_path,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    if audio_path is None:
        cmd = [
            "ffmpeg", "-y",
            "-i", concat_path,
            "-t", str(duration),
            "-c:v", "copy",
            "-an",
            output_path,
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        # Mux with audio (same fade logic as process_segment_video)
        original_duration = duration - 0.4
        fade_out_start = max(0, original_duration - 0.1)
        af_filter = f"afade=t=in:ss=0:d=0.05,afade=t=out:st={fade_out_start}:d=0.1,apad"

        cmd = [
            "ffmpeg", "-y",
            "-i", concat_path,
            "-i", audio_path,
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-t", str(duration),
            "-c:v", "copy",
            "-af", af_filter,
            "-c:a", "aac",
            output_path,
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Cleanup temporary files
    for p in temp_parts:
        if os.path.exists(p):
            os.remove(p)
    if os.path.exists(concat_list):
        os.remove(concat_list)
    if os.path.exists(concat_path):
        os.remove(concat_path)

    print(f"   ✅ Interleaved segment saved: {output_path}")
    return output_path
