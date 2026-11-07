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
