"""Launch the Blender stage in its own process (Blender executable or the `bpy` module)."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from .utils import log

SCRIPT = Path(__file__).parent / "blender" / "scene_build.py"


def find_blender(explicit: str | None) -> list[str]:
    if explicit:
        return [explicit, "-b", "--factory-startup", "-noaudio", "-P", str(SCRIPT), "--"]
    env = os.environ.get("BLENDER")
    if env:
        return [env, "-b", "--factory-startup", "-noaudio", "-P", str(SCRIPT), "--"]
    try:
        import importlib.util

        if importlib.util.find_spec("bpy") is not None:
            return [sys.executable, str(SCRIPT)]
    except Exception:
        pass
    exe = shutil.which("blender")
    if exe:
        return [exe, "-b", "--factory-startup", "-noaudio", "-P", str(SCRIPT), "--"]
    raise RuntimeError("Blender not found: `pip install bpy` (matching your Python), install Blender and put it on "
                       "PATH, or pass --blender /path/to/blender")


def run_blender(manifest_path: Path, work: Path, blender: str | None, render: bool = True, save_blend: bool = True) -> dict:
    job = dict(manifest=str(manifest_path), frames_dir=str(work / "frames"), qa_out=str(work / "physics_qa.json"),
               blend_out=str(work / "scene.blend") if save_blend else None, no_render=not render)
    job_path = work / "blender_job.json"
    job_path.write_text(json.dumps(job, indent=2))
    cmd = find_blender(blender) + [str(job_path)]
    log(f"   blender: {' '.join(cmd[:2])} ...")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    tail = []
    for line in proc.stdout:
        line = line.rstrip()
        tail = (tail + [line])[-40:]
        if line.startswith("[blender") or "Error" in line or "Traceback" in line:
            print("   " + line, flush=True)
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError("Blender stage failed:\n" + "\n".join(tail))
    return json.loads((work / "physics_qa.json").read_text())
