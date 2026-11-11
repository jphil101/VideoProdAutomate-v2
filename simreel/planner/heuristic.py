"""Offline rule-based planner for construction event descriptions.

It understands the way site engineers write sequences -- grid references (A1, B2),
"between A1 and B1", "bay A1–B2", sections (UB 406x178), "already standing",
"identical", "bolted" -- and maps every described step onto the narration sentence
that talks about it.  Use the Claude planner for free-form prose; use this one
offline or as a deterministic baseline.
"""
from __future__ import annotations

import re

from ..plan_schema import Element, Event, Grid, Site, StoryPlan
from ..text import norm, split_sentences

NUM_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
             "ten": 10, "single": 1}
NODE = r"\b([A-H])\s?-?([1-9])\b"
PAIR = rf"{NODE}\s*(?:–|—|-|to|and|&)\s*{NODE}"
SECTION = r"\b(UB|UC|UKB|UKC|W|HEA|HEB|HEM|IPE|SHS|RHS|CHS)\s?\d+(?:\.\d+)?(?:\s?[x×]\s?\d+(?:\.\d+)?){0,2}"
ID = r"\b([A-Z]{1,3}-[A-Z0-9]{1,3})\b"
STOP = set("the a an and or of to in on at by with from is are be it its this that as for into onto then so each "
           "every all same up".split())
DIGIT_WORDS = {str(v): k for k, v in NUM_WORDS.items() if k != "single"}
LAND_HINTS = ["settles", "lands", "drop", "drops", "seated", "set", "placed", "slots", "stacked", "installed",
              "positioned", "bolted", "lowered", "lifted", "picks", "lifts"]


def _cue(sentence: str) -> str | None:
    """Pick the word that best marks the moment of landing (priority order)."""
    toks = [w.strip(",.;:!?") for w in sentence.split()]
    best, rank = None, 99
    for w in toks:
        n = norm(w)
        if n in LAND_HINTS and LAND_HINTS.index(n) < rank:
            best, rank = w, LAND_HINTS.index(n)
    return best


def _num(tok: str) -> int | None:
    tok = tok.lower()
    if tok.isdigit():
        return int(tok)
    return NUM_WORDS.get(tok)


def _content(s: str) -> set[str]:
    s = re.sub(r"(?i)gridlines?", "grid line", s)
    out = set()
    for w in re.findall(r"[A-Za-z0-9]+", s):
        n = norm(w)
        if not n or n in STOP:
            continue
        n = DIGIT_WORDS.get(n, n)
        if len(n) > 4 and n.endswith("s") and not n.endswith("ss"):
            n = n[:-1]  # crude plural folding: modules -> module, beams -> beam
        out.add(n)
    return out


ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8,
            "ninth": 9, "tenth": 10, "last": -1, "final": -1}


KIND_NOUNS = {
    "beam": {"beam", "girder", "joist", "rafter"},
    "column": {"column", "post", "stanchion"},
    "slab": {"slab", "unit", "plank", "deck", "floor", "panel"},
    "module": {"module", "unit", "pod", "box"},
    "wall_panel": {"wall", "panel"},
    "brace": {"brace", "bracing"},
    "box": {"pallet", "container", "load", "crate"},
}


def _ordinal(sentence: str, kind: str) -> int | None:
    """Ordinal that refers to an element of `kind` ("the second module"), not to anything else
    ("matches the first", "the first bay")."""
    nouns = KIND_NOUNS.get(kind, set())
    toks = re.findall(r"[A-Za-z]+", sentence.lower())
    for i, w in enumerate(toks):
        if w in ORDINALS:
            follow = {t.rstrip("s") for t in toks[i + 1:i + 3]}
            if follow & nouns:
                return ORDINALS[w]
    return None


def _grid(text: str) -> tuple[Grid, int]:
    t = text.replace("×", "x")
    xs = ys = None
    m = re.search(r"(\w+)\s+bays?\s+along\s+x\s+at\s+(\d+(?:\.\d+)?)\s*m", t, re.I)
    if m and _num(m[1]):
        xs = [float(m[2])] * _num(m[1])
    m = re.search(r"(\w+)\s+bays?\s+along\s+y\s+at\s+(\d+(?:\.\d+)?)\s*m", t, re.I)
    if m and _num(m[1]):
        ys = [float(m[2])] * _num(m[1])
    nodes = re.findall(NODE, t)
    max_i = max([ord(a) - 65 for a, _ in nodes] + [1])
    max_j = max([int(b) - 1 for _, b in nodes] + [1])
    metres = [float(v) for v in re.findall(r"(\d+(?:\.\d+)?)\s*m\b", t) if 2.5 <= float(v) <= 18]
    if xs is None:
        xs = [metres[0] if metres else 6.0] * max_i
    if ys is None:
        ys = [metres[1] if len(metres) > 1 else (metres[0] if metres else 6.0)] * max_j
    m = re.search(r"(?:storey|story|floor)\s+height\s+(?:of\s+)?(\d+(?:\.\d+)?)", t, re.I) or \
        re.search(r"(\d+(?:\.\d+)?)\s*m\s+(?:storey|story|floor)", t, re.I)
    h = float(m[1]) if m else 4.0
    m = re.search(r"(\w+)[- ](?:storey|story)", t, re.I)
    levels = _num(m[1]) if m and _num(m[1]) else 1
    return Grid(x_spacings_m=xs, y_spacings_m=ys, level_heights_m=[h] * levels), max_i


def plan_heuristic(events_text: str, script_text: str, title: str | None = None) -> StoryPlan:
    grid, _ = _grid(events_text)
    sentences = split_sentences(script_text)
    ev_lines = [ln.strip() for ln in re.split(r"\n\s*(?:\d+[.)]|[-*])\s+|\n\n", "\n" + events_text) if ln.strip()]
    steps = []
    for ln in ev_lines:
        if ln.lstrip().startswith("#"):
            continue
        steps.extend(split_sentences(ln))
    site = Site(crane="none" if re.search(r"no crane|without (a )?crane", events_text, re.I) else "tower")
    if re.search(r"overcast|cloudy", events_text, re.I):
        site.environment = "overcast"
    elif re.search(r"sunset|golden", events_text, re.I):
        site.environment = "golden_hour"

    elements: list[Element] = []
    by_id: dict[str, Element] = {}
    groups: list[dict] = []  # {action, targets, text}
    last_beam_section = None
    auto = {"beam": 0, "slab": 0, "module": 0, "wall_panel": 0, "brace": 0, "box": 0}

    def add(el: Element):
        if el.id in by_id:
            return by_id[el.id]
        elements.append(el)
        by_id[el.id] = el
        return el

    def next_id(kind, given: list[str]):
        if given:
            return given.pop(0)
        auto[kind] += 1
        prefix = {"beam": "B", "slab": "S", "module": "M", "wall_panel": "W", "brace": "X", "box": "P"}[kind]
        return f"{prefix}{auto[kind]}"

    n_x = len(grid.x_spacings_m) + 1
    n_y = len(grid.y_spacings_m) + 1
    all_nodes = [f"{chr(65 + i)}{j + 1}" for j in range(n_y) for i in range(n_x)]
    for step in steps:
        low = step.lower()
        ids = [i for i in re.findall(ID, step) if not re.fullmatch(r"[A-H]-?\d", i)]
        sec_m = re.search(SECTION, step, re.I)
        section = sec_m[0] if sec_m else None
        lvl_m = re.search(r"level\s+(\d)", low)
        level = int(lvl_m[1]) if lvl_m else None
        pairs = [(f"{a}{b}", f"{c}{d}") for a, b, c, d in re.findall(PAIR, step)]
        nodes = [f"{a}{b}" for a, b in re.findall(NODE, step)]
        preplaced = bool(re.search(r"already|existing|in place|standing|pre-?installed", low))
        targets: list[str] = []
        action = "lift_and_place"
        if re.search(r"\bbolt|connection|weld|\bfix(ed)?\b", low) and not pairs and not preplaced:
            named = [i for i in ids if i in by_id]
            groups.append(dict(action="bolt", targets=named or [e.id for e in elements if e.kind == "beam"], text=step))
            continue
        if "column" in low and (preplaced or not pairs):
            col_nodes = nodes or all_nodes
            if re.search(r"\ball\b|\bevery\b|\beach\b", low) and not nodes:
                col_nodes = all_nodes
            if "footing" in low or "foundation" in low or "pad" in low:
                for nd in col_nodes:
                    add(Element(id=f"F-{nd}", kind="footing", at=nd, preplaced=True))
            col_sec = section if section and section.upper().startswith(("UC", "SHS", "HE", "W", "RHS")) else None
            for nd in col_nodes:
                add(Element(id=f"C-{nd}", kind="column", at=nd, level=0, section=col_sec, preplaced=preplaced))
                targets.append(f"C-{nd}")
            if preplaced:
                continue
        elif re.search(r"slab|floor unit|hollow-?core|plank|deck", low):
            bay_pairs = pairs or []
            for a, b in bay_pairs:
                el = add(Element(id=next_id("slab", ids), kind="slab", bay_from=a, bay_to=b, level=level or 1,
                                 preplaced=preplaced))
                targets.append(el.id)
            action = "lower_into_place" if site.crane == "none" else "lift_and_place"
        elif re.search(r"volumetric|pod|modular unit|module unit|stack", low) and pairs:
            for a, b in pairs:
                el = add(Element(id=next_id("module", ids), kind="module", bay_from=a, bay_to=b, level=level or 0,
                                 preplaced=preplaced))
                targets.append(el.id)
        elif "wall" in low and pairs:
            for a, b in pairs:
                el = add(Element(id=next_id("wall_panel", ids), kind="wall_panel", start=a, end=b, level=level or 0))
                targets.append(el.id)
        elif "brac" in low and pairs:
            for a, b in pairs:
                el = add(Element(id=next_id("brace", ids), kind="brace", start=a, end=b, level=level or 0))
                targets.append(el.id)
        elif "beam" in low or "girder" in low or "joist" in low:
            if not pairs and len(nodes) >= 2:
                pairs = [(nodes[k], nodes[k + 1]) for k in range(0, len(nodes) - 1, 2)]
            if section:
                last_beam_section = section
            elif "identical" in low or "same" in low:
                section = last_beam_section
            for a, b in pairs:
                el = add(Element(id=next_id("beam", ids), kind="beam", start=a, end=b, level=level or 1,
                                 section=section or last_beam_section, preplaced=preplaced))
                targets.append(el.id)
        if targets and not preplaced:
            groups.append(dict(action=action if site.crane == "tower" else "lower_into_place", targets=targets, text=step))

    # ---- map step groups onto narration sentences (monotonic, by word overlap)
    events: list[Event] = []
    cur = 0
    sent_words = [_content(s) for s in sentences]
    used = set()
    kind_seen: dict[str, int] = {}
    n_of_kind: dict[str, int] = {}
    for g in groups:
        k = by_id[g["targets"][0]].kind if g["targets"] and g["targets"][0] in by_id else g["action"]
        n_of_kind[k] = n_of_kind.get(k, 0) + 1
    for g in groups:
        k = by_id[g["targets"][0]].kind if g["targets"] and g["targets"][0] in by_id else g["action"]
        kind_seen[k] = kind_seen.get(k, 0) + 1
        nth = kind_seen[k]  # this is the nth install of this kind of element
        gw = _content(g["text"]) | {norm(t) for t in g["targets"]}
        for tid in g["targets"]:
            el = by_id.get(tid)
            for ref in (el.start, el.end, el.bay_from, el.bay_to) if el else ():
                if ref:
                    gw.add(norm(ref[0]))
        best, best_score = None, 0.0
        for si in range(cur, len(sentences)):
            sw = sent_words[si]
            score = len(gw & sw) - 0.35 * (si - cur)
            if g["action"] == "bolt" and "bolt" in " ".join(sw):
                score += 3
            o = _ordinal(sentences[si], k)  # "the second module" -> the 2nd module install
            if o is not None and g["action"] != "bolt":
                if o == nth or (o == -1 and nth == n_of_kind[k]):
                    score += 2.0
                else:
                    score -= 1.0
            if si in used:
                score -= 3.0 if g["action"] in ("lift_and_place", "lower_into_place") else 1.0
            if score > best_score:
                best, best_score = si, score
        if best is None:
            best = min(cur, len(sentences) - 1)
        # several crane lifts described in one step but narrated over consecutive sentences
        if g["action"] in ("lift_and_place", "lower_into_place") and len(g["targets"]) > 1:
            pass
        cue = _cue(sentences[best])
        events.append(Event(sentence=best, action=g["action"], targets=g["targets"], cue_word=cue))
        used.add(best)
        cur = best if g["action"] == "bolt" else best

    # ---- split "pick up" + "lowered" pairs: if a lift's sentence only mentions picking up and the next
    # sentence talks about landing, sync the landing to the next sentence
    fixed: list[Event] = []
    for ev in events:
        s = sentences[ev.sentence].lower()
        nxt = sentences[ev.sentence + 1].lower() if ev.sentence + 1 < len(sentences) else ""
        landing = r"\b(lowered|set|placed|settles?|drops?|lands?|seated|slots?)\b"
        if ev.action in ("lift_and_place", "lower_into_place") and not re.search(landing, s) and \
                re.search(landing, nxt) and (ev.sentence + 1) not in used:
            cue = _cue(sentences[ev.sentence + 1])
            ev = ev.model_copy(update={"sentence": ev.sentence + 1, "cue_word": cue})
            used.add(ev.sentence)
        fixed.append(ev)
    events = fixed
    covered = {e.sentence for e in events}
    if sentences and 0 not in covered:
        events.insert(0, Event(sentence=0, action="camera", shot="establishing"))
    if len(sentences) > 1 and len(sentences) - 1 not in covered:
        events.append(Event(sentence=len(sentences) - 1, action="camera", shot="orbit"))
    events.sort(key=lambda e: e.sentence)
    return StoryPlan(title=title or (sentences[0][:60] if sentences else "Construction sequence"), site=site,
                     grid=grid, elements=elements, events=events)
