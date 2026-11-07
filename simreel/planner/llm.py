"""Claude planner: free-form event description + narration -> validated StoryPlan.

Claude only writes the engineer-level plan (gridlines, levels, sections, ordering,
which sentence shows what).  Geometry, physics and timing are computed by code, and
the plan is checked by the structural solver; problems are fed back once for repair.
"""
from __future__ import annotations

import json
import os

from ..plan_schema import StoryPlan
from ..structure import solve
from ..utils import log

SYSTEM = """You plan physically realistic construction animations. You translate a description of
construction events and a narration script into a structured plan that a deterministic engine renders
in Blender with rigid-body physics (gravity, contact, settling) and a tower crane.

How the engine interprets the plan:
- Lettered gridlines A, B, C... run along X; numbered gridlines 1, 2, 3... along Y. Node "A1" is the
  origin. x_spacings_m / y_spacings_m are the bay sizes between consecutive gridlines.
- Level 0 is the top of the ground slab. Level k is at the sum of the first k storey heights.
- column: stands on whatever is at its node (footing/beam/slab) at `level` and rises to the floor at
  level+1 (its top is set automatically so beams sit on the cap plate).
- beam: spans start->end nodes and forms floor `level` (>=1). Beams sit on column caps. If a level has
  beams in both X and Y, X-direction beams are primary and Y-direction beams sit on top of them.
- slab: precast floor unit filling the bay bay_from..bay_to at floor `level`; it rests on the beams
  around that bay, so those beams must exist (along the bay edges) before the slab is placed.
- module: volumetric unit filling a bay, standing on the level below (ground, slab or another module).
- wall_panel: precast panel between two nodes on a gridline; brace: diagonal from start node at
  `level` to end node at level+1; footing: pad at a node; box: any loose item (pallet, container).
- Elements that already exist when the video starts get preplaced=true. Everything else must be
  installed by exactly one event (lift_and_place with the crane, lower_into_place without crane,
  or reveal).
- Physical order matters: supports first (footings -> columns -> beams -> slabs -> modules above).
  Never install something before what carries it.
- Events reference narration sentences by 0-based index. Put the install on the sentence that
  describes the element arriving, and set cue_word to the word in that sentence that should coincide
  with the moment it lands ("settles", "placed", "drops"...). The engine starts the crane early enough
  to land exactly on that word and opens pauses when needed, so it is fine for a lift to be narrated in
  one sentence.  Several targets in one event are installed one after another.
- Use 'bolt' to show connections being made (it fixes those elements to their supports),
  'highlight' to draw attention, 'camera' with a shot for sentences without new action, and
  'stock_clip' (with a concise stock_query) only when the narration talks about something that is not
  part of the 3D scene (people, a factory, a city). Prefer the 3D scene when in doubt.
- Use realistic sections (e.g. UB 406x178 beams for 6 m spans, UC 305x305 columns, 200 mm hollow-core
  slabs) unless the description says otherwise. Keep labels short (<= 40 characters).
If the description is vague about positions, choose a sensible, buildable layout that matches its
intent, and keep it compact (the camera frames the whole structure)."""


def _sentences_block(sentences: list[str]) -> str:
    return "\n".join(f"[{i}] {s}" for i, s in enumerate(sentences))


def plan_with_claude(events_text: str, sentences: list[str], model: str = "claude-opus-5-5",
                     effort: str = "high", repair_rounds: int = 1, client=None) -> StoryPlan:
    import anthropic

    client = client or anthropic.Anthropic()
    user = (f"<events>\n{events_text.strip()}\n</events>\n\n"
            f"<narration_sentences>\n{_sentences_block(sentences)}\n</narration_sentences>\n\n"
            "Produce the plan.")
    messages = [{"role": "user", "content": user}]
    plan: StoryPlan | None = None
    for attempt in range(repair_rounds + 1):
        try:
            resp = client.beta.messages.parse(
                model=model,
                max_tokens=16000,
                system=SYSTEM,
                messages=messages,
                output_format=StoryPlan,
                thinking={"type": "adaptive"},
                output_config={"effort": effort},
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
        except anthropic.BadRequestError as exc:
            raise RuntimeError(f"Claude rejected the planning request: {exc}") from exc
        except anthropic.AuthenticationError as exc:
            raise RuntimeError("Claude API credentials missing/invalid (set ANTHROPIC_API_KEY)") from exc
        except (anthropic.RateLimitError, anthropic.APIConnectionError, anthropic.InternalServerError) as exc:
            raise RuntimeError(f"Claude API unavailable: {exc}") from exc
        if resp.stop_reason == "refusal":
            raise RuntimeError("Claude declined to plan this request")
        if resp.stop_reason == "max_tokens" or resp.parsed_output is None:
            raise RuntimeError("Claude's plan was truncated or unparsable")
        plan = resp.parsed_output
        problems = validate(plan, sentences)
        if not problems or attempt == repair_rounds:
            for p in problems:
                log(f"   plan warning: {p}")
            return plan
        log(f"   plan has {len(problems)} issue(s); asking Claude to repair")
        messages.append({"role": "assistant", "content": resp.content})
        messages.append({"role": "user", "content": (
            "The engine found these problems with the plan:\n- " + "\n- ".join(problems) +
            "\nReturn the full corrected plan.")})
    assert plan is not None
    return plan


def validate(plan: StoryPlan, sentences: list[str]) -> list[str]:
    problems = []
    try:
        scene = solve(plan)
        problems.extend(scene.warnings)
    except Exception as exc:  # pragma: no cover
        problems.append(f"structure could not be solved: {exc}")
        return problems
    ids = {e.id for e in plan.elements}
    installed = {}
    for ev in plan.events:
        if not 0 <= ev.sentence < len(sentences):
            problems.append(f"event references sentence {ev.sentence}, valid range is 0..{len(sentences) - 1}")
        for t in ev.targets:
            if t not in ids:
                problems.append(f"event target {t} is not an element id")
            if ev.action in ("lift_and_place", "lower_into_place", "reveal"):
                if t in installed:
                    problems.append(f"{t} is installed twice")
                installed[t] = ev.sentence
    by = {e.id: e for e in plan.elements}
    for t, si in installed.items():
        el = by.get(t)
        if el and el.preplaced:
            problems.append(f"{t} is preplaced but also installed")
    order = list(installed)
    for i, t in enumerate(order):  # supports must be installed (or preplaced) earlier
        r = scene.elements.get(t)
        if not r:
            continue
        for s in r.supports:
            if s in installed and order.index(s) > i:
                problems.append(f"{t} is installed before {s}, which carries it")
    return problems


def credentials_available() -> bool:
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return True
    return os.path.isdir(os.path.expanduser("~/.config/anthropic"))


def dump(plan: StoryPlan) -> str:
    return json.dumps(plan.model_dump(), indent=2)
