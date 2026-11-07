# simreel: construction events + script → physics-simulated, captioned video

`simreel` turns two plain-text inputs into a finished MP4:

* **events**: what happens on site, in the language a site engineer uses (gridlines, levels, sections, sequence).
* **script**: the narration.

The output is a Blender-rendered video. Elements are lifted by a tower crane with real pendulum dynamics, released, and they settle under Bullet rigid-body physics. The narration has word-level synced captions (burned in *and* as a soft subtitle track). You can optionally cut to stock footage (Pexels, Pixabay, Envato Elements, or a local folder).

```
events.md + script.md
   │
   ├─1 planner ───────── Claude (structured output, validated + auto-repaired) or offline heuristic
   │                      → plan.json  (gridlines, levels, sections, events → sentence + cue word)
   ├─  structure solver ─ exact geometry: beams on cap plates, stacked framing, slabs on beams,
   │                      collision/support validation (nothing floats, nothing interpenetrates)
   ├─2 narration ──────── TTS per sentence + word timestamps (native / whisper / DTW / estimate)
   ├─3 choreographer ──── crane IK, S-curve drives, ZV anti-sway input shaping, variable-length
   │                      pendulum sim, cue-word landing sync, auto-cinematography camera
   ├─4 stock ──────────── Pexels · Pixabay · Envato (your EnvatoDownloader.py) · local folder
   ├─5 Blender ────────── build scene → bake Bullet physics (2-pass lock) → physics QA → render
   ├─6 audio + captions ─ narration track, physics-timed impact SFX, karaoke ASS + SRT/VTT
   └─7 ffmpeg ─────────── one pass: frames + stock xfades + captions burned + mov_text track
```

## Why this design (lessons from the previous pipeline)

* **The LLM never does 3D maths.** It writes an engineer-level plan (e.g. "beam PB-2 from B1 to C1 at level 1, UB 406x178, lands on the word *settles*"). Deterministic code computes coordinates, bearing lengths, support relationships, masses and crane paths. Invalid plans (overlaps, unsupported elements, wrong install order) are caught by the solver and sent back to Claude for one repair round.
* **Timing is negotiated, not estimated.** Each sentence is synthesised separately with exact word times. The crane starts early enough to land *on the cue word*. If physics needs longer than the words, the pause before the next sentence is opened up, so the crane is never sped past its limits. A recorded voice-over keeps its timing, and the tool warns when a lift can't fit.
* **Physics is simulated, then checked.** Bodies are released 5 mm above their seat and settle under gravity in Bullet. Once an element is bolted, or is about to carry something heavier, it is locked at its *simulated* pose. Then `physics_qa.json` reports each body's final error against its intended position (`--strict-physics` fails the job on a miss).
* **Captions come from the same clock as the audio.** They are built from the word timestamps used to place the audio, so caption and voice share one timeline.

## Install

```bash
python3.13 -m venv .venv && source .venv/bin/activate     # bpy wheels track Blender's Python
pip install -r simreel/requirements.txt                   # includes bpy (Blender as a module)
sudo apt install ffmpeg espeak-ng fonts-inter             # ffmpeg+libass, DTW aligner reference, caption font
# optional: libttspico-utils (offline voice), pip install faster-whisper kokoro-onnx edge-tts
python -m simreel check
```

Instead of `bpy` you can use a full Blender install: `--blender /path/to/blender` or `BLENDER=/path/to/blender`.

## Use

```bash
# everything from a job file (all settings documented inline)
python -m simreel make --job examples/modular_frame/job.yaml

# or flags
python -m simreel make --events examples/modular_frame/events.md --script examples/modular_frame/script.md \
    --resolution 1080p --orientation portrait --fps 30 --quality preview -o out/reel.mp4

# review / hand-edit the plan, then render from it
python -m simreel plan --events events.md --script script.md -o plan.json
python -m simreel make --plan plan.json --script script.md -o out/video.mp4

# answer prompts instead of flags
python -m simreel wizard
```

### Parameters (most useful)

| area | flags |
|---|---|
| frame | `--resolution 360p…2160p\|WxH`, `--orientation landscape\|portrait\|square`, `--fps`, `--crf` |
| render | `--quality draft\|preview\|final` (Workbench / Cycles low / Cycles high), `--engine`, `--samples`, `--gpu auto\|off` |
| voice | `--tts auto\|elevenlabs\|edge\|kokoro\|pico\|espeak\|file\|none`, `--voice`, `--voiceover file.wav`, `--rate`, `--aligner` |
| captions | `--captions burn\|soft\|both\|none`, `--caption-style karaoke\|plain`, `--caption-position`, `--font`, `--caption-scale`, `--highlight-color`, `--caption-box`, `--no-titles` |
| stock | `--stock none\|local\|pexels\|pixabay\|envato`, `--stock-dir`, `--envato-script`, `--stock-fit crop\|blurpad` |
| audio | `--music bed.mp3` (auto-ducked), `--no-sfx` |
| physics | `--motion-speed`, `--no-anti-sway`, `--strict-physics` |
| planning | `--planner auto\|claude\|heuristic`, `--model`, `--plan plan.json` |

Environment keys: `ANTHROPIC_API_KEY` (planner), `ELEVENLABS_API_KEY`, `PEXELS_API_KEY`, `PIXABAY_API_KEY`, `ENVATO_DOWNLOADER` (path to `EnvatoDownloader.py`), `ENVATO_PYTHON` (the Python of the downloader's venv, if different).

### Outputs

`video.mp4` (burned captions plus an English `mov_text` subtitle track), and alongside it `video.srt`, `video.vtt`, `video.plan.json` and `video.physics.json`. The work directory (`.video_work/`) keeps `manifest.json`, `timeline.json`, `scene.blend` (open it in Blender to inspect the baked simulation) and the rendered frames. Re-running with an unchanged manifest reuses the frames.

## Writing events

Gridlines A, B, C… run along X; 1, 2, 3… run along Y; level 0 is the ground slab. The planner understands:

* `columns at A1, B1 … are already standing`: preplaced elements
* `beam PB-2 (UB 406x178) placed between B1 and C1 at level 1`
* `slab S-2 drops into bay B1–C2`, `volumetric module M1 in bay A1–B2 at level 0`
* `wall panel between A1 and A2`, `brace from A1 to B1`
* `the connections are bolted`

Kinds: footing, column, beam, slab, module, wall_panel, brace, box. Actions: lift_and_place, lower_into_place, reveal, bolt, highlight, camera, stock_clip, pause.

The Claude planner accepts free prose. The offline heuristic planner works best with grid references like the ones above.

## Rendering speed

Cycles on a GPU renders `preview` 1080p in roughly 1–3 s/frame. On a 4-core CPU it takes 10–50 s/frame, so use `--quality draft` (Workbench, under 1 s/frame) for iteration, and `preview`/`final` on a GPU machine.

## Limitations

* The structural vocabulary is building-frame oriented: grid-based frames, slabs, volumetric modules, panels and braces. Fluids (concrete pours), cloth and soft bodies are not simulated. Describe a pour as a `reveal` of the slab.
* There is one tower crane per scene, and lifts are sequential.
* Envato Elements has no public download API, so the `envato` provider drives your existing Playwright downloader (logged-in browser session) and picks from what it downloads.
