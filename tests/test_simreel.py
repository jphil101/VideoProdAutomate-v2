"""Core checks that run without Blender: geometry, motion physics, alignment, timing, captions."""
from pathlib import Path

import numpy as np
import pytest

from simreel.captions import build_cards
from simreel.config import JobConfig
from simreel.motion import move_profile, simulate_pendulum, smooth_trapezoid, zv_shape, pendulum_period
from simreel.narration import align, espeak, synthesize
from simreel.planner.heuristic import plan_heuristic
from simreel.structure import solve
from simreel.text import split_sentences

EX = Path(__file__).resolve().parents[1] / "examples" / "modular_frame"


@pytest.fixture(scope="module")
def plan():
    return plan_heuristic((EX / "events.md").read_text(), (EX / "script.md").read_text())


def test_structure_is_buildable(plan):
    scene = solve(plan)
    assert scene.warnings == []
    e = scene.elements
    # beams bear exactly on the column cap plates
    assert e["PB-2"].bottom == pytest.approx(e["C-B1"].top, abs=1e-6)
    assert e["PB-2"].bottom == pytest.approx(e["C-C1"].top, abs=1e-6)
    # slab sits on the beams of its bay
    assert set(e["S-2"].supports) == {"PB-2", "PB-4"}
    assert e["S-2"].bottom == pytest.approx(e["PB-2"].top, abs=1e-6)


def test_heuristic_maps_events_to_sentences(plan):
    by_target = {t: ev for ev in plan.events for t in ev.targets if ev.action == "lift_and_place"}
    assert by_target["PB-2"].sentence == 3 and by_target["PB-2"].cue_word == "settles"
    assert by_target["PB-4"].sentence == 5
    assert by_target["S-2"].sentence == 8 and by_target["S-2"].cue_word == "drops"


def test_smooth_trapezoid_limits():
    T, rf = move_profile(10.0, v_max=2.0, a_max=1.5)
    t = np.linspace(0, T, 4001)
    x = 10 * smooth_trapezoid(t / T, rf)
    v = np.gradient(x, t)
    a = np.gradient(v, t)
    assert x[0] == pytest.approx(0) and x[-1] == pytest.approx(10)
    assert v.max() <= 2.0 * 1.01 and np.abs(a).max() <= 1.5 * 1.03
    assert np.all(np.diff(x) >= -1e-9)


def test_zv_shaping_cancels_residual_sway():
    L = 6.0
    fps = 60
    T, rf = move_profile(12.0, 2.0, 1.5)
    t = np.arange(0, T + 12, 1 / fps)
    x = 12 * smooth_trapezoid(t / T, rf)
    raw = np.column_stack([x, np.zeros_like(x)])
    shaped = zv_shape(t, raw, pendulum_period(L))
    rope = np.full(len(t), L)
    end = t > T + pendulum_period(L)
    sway_raw = np.abs(simulate_pendulum(t, raw, rope, 0.0)[end, 0]).max()
    sway_zv = np.abs(simulate_pendulum(t, shaped, rope, 0.0)[end, 0]).max()
    assert sway_zv < 0.15 * sway_raw


@pytest.mark.skipif(not espeak.available(), reason="espeak-ng not installed")
def test_dtw_alignment_accuracy():
    s = "The tower crane lowers the beam onto the column caps between grid lines A and B."
    audio, sr, ev = espeak.synth(s, "en-us", 125)  # different rate than the reference
    truth = sorted({b: t for b, _, t in reversed(ev)}.items())
    got = align.dtw_align(s, audio, sr)
    err = np.abs(np.array([t for _, t in truth]) - np.array([w.start for w in got]))
    assert err.mean() < 0.05


@pytest.mark.skipif(not espeak.available(), reason="espeak-ng not installed")
def test_landing_happens_on_cue_word(plan):
    from simreel.choreo import Choreographer

    cfg = JobConfig()
    cfg.narration.tts = "espeak"
    sents = split_sentences((EX / "script.md").read_text())
    nar = synthesize(sents, cfg.narration)
    _, tl = Choreographer(plan, solve(plan), nar, cfg, sents).run()
    w = next(w for w in tl.windows if w.kind == "crane" and w.target == "PB-2")
    cue = next(ws for ws in tl.words[3] if ws[0].startswith("settles"))
    assert abs(w.data["t_land"] - cue[1]) < 1.0 / cfg.video.fps + 1e-6
    # sentences never overlap
    for i in range(len(sents) - 1):
        assert tl.sentence_starts[i] + nar.clips[i].duration <= tl.sentence_starts[i + 1] + 1e-6


def test_caption_cards():
    words = [[("Modular", 0.0, 0.4), ("steel", 0.4, 0.7), ("frames", 0.7, 1.1), ("grow", 1.1, 1.4),
              ("by", 1.4, 1.5), ("repeating", 1.5, 2.0), ("one", 2.0, 2.2), ("bay.", 2.2, 2.6)]]
    cards = build_cards(words, max_chars=14, max_lines=2)
    flat = [w for c in cards for w in c.words]
    assert [w[0] for w in flat] == [w[0] for w in words[0]]
    for a, b in zip(cards, cards[1:]):
        assert a.end <= b.start + 1e-9
    assert all(max(len(" ".join(c.words[i][0] for i in ln)) for ln in c.lines) <= 14 for c in cards)


def test_claude_planner_request_and_repair_loop(plan):
    """Exercise the real SDK request/parse path against a mocked transport (no network)."""
    import json

    import anthropic
    import httpx2 as httpx

    from simreel.planner.llm import plan_with_claude

    bad = plan.model_copy(deep=True)
    bad.events[1].sentence = 99  # invalid -> triggers one repair round
    replies = [bad, plan]
    seen = []

    def handler(request: httpx.Request):
        body = json.loads(request.content)
        seen.append((request.headers.get("anthropic-beta", ""), body))
        out = replies[len(seen) - 1]
        return httpx.Response(200, json={
            "id": f"msg_{len(seen)}", "type": "message", "role": "assistant", "model": body["model"],
            "content": [{"type": "text", "text": out.model_dump_json()}],
            "stop_reason": "end_turn", "stop_sequence": None,
            "usage": {"input_tokens": 10, "output_tokens": 10}})

    client = anthropic.Anthropic(api_key="test", http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    sents = split_sentences((EX / "script.md").read_text())
    result = plan_with_claude((EX / "events.md").read_text(), sents, client=client)
    assert len(seen) == 2  # first plan rejected by validation, repaired plan accepted
    beta, body = seen[0]
    assert body["model"] == "claude-opus-5-5"
    assert body["thinking"] == {"type": "adaptive"}
    assert body["output_config"]["effort"] == "high"
    assert body["output_config"]["format"]["type"] == "json_schema"
    assert body["fallbacks"] == "default" and "server-side-fallback-2026-07-01" in beta
    assert "sentence 99" in json.dumps(seen[1][1]["messages"][-1])
    assert result.model_dump() == plan.model_dump()


MODULES_EVENTS = """Site: modular housing block, 2 bays along X at 6 m (gridlines A, B, C) and 1 bay along Y at 3 m
(gridlines 1, 2). Two-storey, storey height 3.2 m. Tower crane on site.
1. Volumetric module M1 is lifted and stacked into bay A1–B2 at level 0.
2. Volumetric module M2 is placed into bay B1–C2 at level 0, next to the first.
3. Volumetric module M3 is stacked onto M1 in bay A1–B2 at level 1.
"""
MODULES_SCRIPT = """Factory-built modules arrive on site complete, ready to be stacked.
The first module is placed at the corner of the grid.
A second module is placed alongside it, matching its edge exactly.
Then a third module is stacked on top of the first.
Two storeys, in a matter of minutes.
"""


def test_heuristic_uses_ordinals_and_stacks_modules():
    p = plan_heuristic(MODULES_EVENTS, MODULES_SCRIPT)
    where = {t: ev.sentence for ev in p.events for t in ev.targets}
    assert where == {"M1": 1, "M2": 2, "M3": 3}  # not shifted onto the intro sentence
    scene = solve(p)
    assert scene.warnings == []
    m1, m3 = scene.elements["M1"], scene.elements["M3"]
    assert m3.bottom == pytest.approx(m1.top, abs=1e-6) and m3.supports == ["M1"]


def test_ordinal_must_refer_to_the_element_kind():
    from simreel.planner.heuristic import _ordinal

    assert _ordinal("A second module is placed alongside it.", "module") == 2
    assert _ordinal("So the new bay matches the first.", "beam") is None
    assert _ordinal("The first bay is already framed.", "beam") is None


def test_no_crane_mode_lowers_elements():
    ev = ("Single bay, 1 bay along X at 5 m and 1 bay along Y at 4 m. Storey height 3.5 m. No crane.\n"
          "1. Columns at A1, B1, A2 and B2 are already standing.\n2. Beam B1 is placed between A1 and B1.\n"
          "3. Beam B2 is placed between A2 and B2.\n4. A precast slab S1 drops into bay A1–B2.\n")
    sc = "A single bay.\nThe first beam is placed on grid line one.\nIts partner is placed on grid line two.\nA slab drops onto both beams.\n"
    p = plan_heuristic(ev, sc)
    assert p.site.crane == "none"
    assert {ev.action for ev in p.events if ev.targets} == {"lower_into_place"}
    assert solve(p).warnings == []


def _bake_and_qa(tmp_path, events, script):
    import importlib.util
    import json as _json

    if importlib.util.find_spec("bpy") is None:
        pytest.skip("bpy (Blender) not installed")
    from simreel.blender_runner import run_blender
    from simreel.choreo import Choreographer

    cfg = JobConfig()
    cfg.narration.tts = "none"
    cfg.video.fps = 24
    sents = split_sentences(script)
    plan = plan_heuristic(events, script)
    scene = solve(plan)
    manifest, _ = Choreographer(plan, scene, synthesize(sents, cfg.narration), cfg, sents).run()
    mp = tmp_path / "manifest.json"
    mp.write_text(_json.dumps(manifest))
    return run_blender(mp, tmp_path, None, render=False, save_blend=False)["elements"]


@pytest.mark.slow
def test_blender_physics_bodies_settle_on_target(tmp_path):
    """Real Bullet bake: every released body must end on its seat (ground, stacked module, beams)."""
    qa = _bake_and_qa(tmp_path, MODULES_EVENTS, MODULES_SCRIPT)
    assert {q["id"] for q in qa} == {"M1", "M2", "M3"}
    for q in qa:
        assert q["status"] == "settled", q
        assert q["error_m"] < 0.015 and q["error_deg"] < 1.0, q


@pytest.mark.slow
def test_blender_physics_example_frame(tmp_path):
    qa = _bake_and_qa(tmp_path, (EX / "events.md").read_text(), (EX / "script.md").read_text())
    assert [q["status"] for q in qa] == ["settled"] * 3, qa


@pytest.mark.slow
def test_blender_no_crane_mode(tmp_path):
    """Elements that are hidden until they descend must still get rigid bodies (regression)."""
    ev = ("Single bay, 1 bay along X at 5 m and 1 bay along Y at 4 m. Storey height 3.5 m. No crane.\n"
          "1. Columns at A1, B1, A2 and B2 are already standing.\n2. Beam B1 is placed between A1 and B1.\n"
          "3. Beam B2 is placed between A2 and B2.\n4. A precast slab S1 drops into bay A1–B2.\n")
    sc = "A single bay.\nThe first beam is placed on grid line one.\nIts partner is placed on grid line two.\nA slab drops onto both beams.\n"
    qa = _bake_and_qa(tmp_path, ev, sc)
    assert [q["status"] for q in qa] == ["settled"] * 3, qa


def test_titles_never_share_the_caption_band(tmp_path):
    from simreel.captions import write_ass
    from simreel.config import CaptionSettings

    words = [[("Hello", 0.0, 0.5), ("site.", 0.5, 1.0)]]
    cards = build_cards(words, 20, 2)
    for pos, title_align in (("top", "1"), ("bottom", "7"), ("middle", "7")):
        out = tmp_path / f"{pos}.ass"
        write_ass(out, cards, [(0.0, 1.0, "Beam B1")], 720, 1280, CaptionSettings(position=pos))
        styles = {l.split(",")[0]: l.split(",") for l in out.read_text().splitlines() if l.startswith("Style:")}
        assert styles["Style: Title"][18] == title_align
        assert styles["Style: Caption"][18] == {"top": "8", "bottom": "2", "middle": "5"}[pos]
