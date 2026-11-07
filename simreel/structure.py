"""Deterministic geometry solver: StoryPlan (gridlines, levels, sections) -> exact
3D elements with dimensions, target poses, physical roles, masses and lift points.

Conventions: metres, Z up.  Lettered gridlines (A, B, C, ...) run along X, numbered
gridlines (1, 2, 3, ...) along Y; node "A1" is at (0, 0).  Level 0 is the top of
the ground slab (z = 0).  Each element's local frame has its origin at the centre
of its bounding box and local +X along its long axis.

Framing logic (so every element rests on something real):
  * columns stand on footings / beams / slabs found below them;
  * beams sit on the column cap plates; when a level has beams in both directions
    the X-direction beams are primary and carry the Y-direction beams on their top
    flange (stacked framing), so nothing interpenetrates;
  * slabs, modules, wall panels and loose boxes drop onto whatever is below them.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from .plan_schema import Element, StoryPlan

STEEL_DENSITY = 7850.0
CONCRETE_DENSITY = 2400.0


# --------------------------------------------------------------------------- sections
@dataclass
class Section:
    shape: str  # "I", "box", "tube", "rect"
    d: float  # depth (m)
    b: float  # width (m)
    tw: float = 0.0
    tf: float = 0.0
    t: float = 0.0
    name: str = ""

    def area(self) -> float:
        if self.shape == "I":
            return 2 * self.b * self.tf + (self.d - 2 * self.tf) * self.tw
        if self.shape == "box":
            return self.d * self.b - (self.d - 2 * self.t) * (self.b - 2 * self.t)
        if self.shape == "tube":
            r = self.d / 2
            return math.pi * (r**2 - (r - self.t) ** 2)
        return self.d * self.b


def parse_section(text: str | None, default: Section) -> Section:
    if not text:
        return default
    s = text.upper().replace("×", "X").replace(" ", "")
    m = re.match(r"(UB|UC|UKB|UKC)(\d+)X(\d+)", s)
    if m:
        d, b = int(m[2]) / 1000, int(m[3]) / 1000
        return Section("I", d, b, max(0.006, d / 45), max(0.009, d / 28), name=text)
    m = re.match(r"W(\d+)X(\d+(?:\.\d+)?)", s)
    if m:
        d, w = int(m[1]) / 1000, float(m[2])
        tw, tf = max(0.0055, d / 55), max(0.008, d / 34)
        area = w / STEEL_DENSITY
        b = max(0.1, min(0.45, (area - d * tw) / (2 * tf)))
        return Section("I", d, b, tw, tf, name=text)
    m = re.match(r"(HEA|HEB|HEM|HE)(\d+)", s)
    if m:
        n = int(m[2]) / 1000
        d = n - 0.01 if m[1] == "HEA" else n
        return Section("I", d, min(n, 0.3), max(0.0065, n / 35), max(0.01, n / 24), name=text)
    m = re.match(r"IPE(\d+)", s)
    if m:
        d = int(m[1]) / 1000
        return Section("I", d, d / 2, max(0.0045, d / 45), max(0.007, d / 28), name=text)
    m = re.match(r"SHS(\d+)(?:X(\d+))?(?:X(\d+(?:\.\d+)?))?", s)
    if m:
        b = int(m[1]) / 1000
        t = float(m[3]) / 1000 if m[3] else b / 25
        return Section("box", b, b, t=t, name=text)
    m = re.match(r"RHS(\d+)X(\d+)(?:X(\d+(?:\.\d+)?))?", s)
    if m:
        d, b = int(m[1]) / 1000, int(m[2]) / 1000
        t = float(m[3]) / 1000 if m[3] else d / 25
        return Section("box", d, b, t=t, name=text)
    m = re.match(r"CHS(\d+(?:\.\d+)?)(?:X(\d+(?:\.\d+)?))?", s)
    if m:
        d = float(m[1]) / 1000
        t = float(m[2]) / 1000 if m[2] else d / 25
        return Section("tube", d, d, t=t, name=text)
    m = re.match(r"(\d+)X(\d+)", s)
    if m:  # concrete rectangle b x d (mm)
        return Section("rect", int(m[2]) / 1000, int(m[1]) / 1000, name=text)
    return default


def default_beam_section(span: float, secondary: bool, concrete: bool) -> Section:
    if concrete:
        d = max(0.35, round(span / 12 / 0.05) * 0.05)
        return Section("rect", d, 0.3, name=f"{300}x{int(d * 1000)} RC")
    d = round(span / (24 if secondary else 20) / 0.025) * 0.025
    d = min(0.9, max(0.2, d))
    b = min(0.3, max(0.12, round(d * 0.45, 3)))
    return Section("I", d, b, max(0.006, d / 45), max(0.009, d / 28), name=f"UB {int(d * 1000)}x{int(b * 1000)}")


DEFAULT_STEEL_COLUMN = Section("I", 0.305, 0.305, 0.0099, 0.0154, name="UC 305x305x97")
DEFAULT_RC_COLUMN = Section("rect", 0.4, 0.4, name="400x400 RC")


# --------------------------------------------------------------------------- data
@dataclass
class Part:
    shape: str  # box | ibeam | tube | cylinder
    size: tuple  # box:(lx,ly,lz) ibeam:(len,b,d,tw,tf) tube:(len,b,d,t) cylinder:(radius,len)
    offset: tuple = (0.0, 0.0, 0.0)
    material: str = "steel"
    axis: str = "x"  # extrusion axis for ibeam/tube/cylinder
    bevel: float = 0.0


@dataclass
class ResolvedElement:
    id: str
    kind: str
    label: str
    material: str
    color: str | None
    parts: list[Part]
    dims: tuple  # local bbox (lx, ly, lz)
    pos: tuple  # target centre (world)
    yaw: float = 0.0  # radians about Z
    pitch: float = 0.0  # radians about local Y (braces)
    role: str = "settle"  # settle (dynamic after release) | anchored (passive) | static
    mass: float = 100.0
    lift_points: list = field(default_factory=list)
    supports: list = field(default_factory=list)
    preplaced: bool = False
    section: str = ""

    def aabb(self) -> tuple:
        lx, ly, lz = self.dims
        c, s = abs(math.cos(self.yaw)), abs(math.sin(self.yaw))
        hx = (lx * c + ly * s) / 2
        hy = (lx * s + ly * c) / 2
        if self.pitch:
            cp, sp = abs(math.cos(self.pitch)), abs(math.sin(self.pitch))
            hz = (lx * sp + lz * cp) / 2
            hx, hy = hx * cp + lz * sp / 2 * c, hy * cp + lz * sp / 2 * s
        else:
            hz = lz / 2
        x, y, z = self.pos
        return (x - hx, y - hy, z - hz, x + hx, y + hy, z + hz)

    @property
    def bottom(self) -> float:
        return self.aabb()[2]

    @property
    def top(self) -> float:
        return self.aabb()[5]


@dataclass
class ResolvedScene:
    plan: StoryPlan
    xs: list[float]
    ys: list[float]
    elevations: list[float]
    elements: dict[str, ResolvedElement]
    warnings: list[str]
    footprint: tuple  # xmin, ymin, xmax, ymax of the building

    def ordered(self) -> list[ResolvedElement]:
        return list(self.elements.values())


# --------------------------------------------------------------------------- solver
def _letters_to_index(s: str) -> int:
    n = 0
    for ch in s:
        n = n * 26 + (ord(ch) - 64)
    return n - 1


def parse_node(ref: str) -> tuple[int, int]:
    m = re.match(r"^\s*([A-Za-z]{1,2})\s*[-/ ]?\s*(\d+)\s*$", ref or "")
    if not m:
        raise ValueError(f"bad grid node reference {ref!r}; expected e.g. 'A1'")
    return _letters_to_index(m[1].upper()), int(m[2]) - 1


class StructureSolver:
    def __init__(self, plan: StoryPlan):
        self.plan = plan
        g = plan.grid
        self.xs = [0.0]
        for s in g.x_spacings_m:
            self.xs.append(self.xs[-1] + s)
        self.ys = [0.0]
        for s in g.y_spacings_m:
            self.ys.append(self.ys[-1] + s)
        self.E = [0.0]
        for h in g.level_heights_m or [3.5]:
            self.E.append(self.E[-1] + h)
        self.warnings: list[str] = []
        self.out: dict[str, ResolvedElement] = {}
        self.cap_w = 0.42

    # ------------------------------------------------------------------ helpers
    def node_xy(self, ref: str, el: Element | None = None) -> tuple[float, float]:
        i, j = parse_node(ref)
        if not (0 <= i < len(self.xs)) or not (0 <= j < len(self.ys)):
            # extend gridlines with the last spacing so a slightly-off plan still works
            self.warnings.append(f"node {ref} is outside the defined grid; extrapolating")
        dx = self.plan.grid.x_spacings_m[-1] if self.plan.grid.x_spacings_m else 6.0
        dy = self.plan.grid.y_spacings_m[-1] if self.plan.grid.y_spacings_m else 6.0
        x = self.xs[i] if i < len(self.xs) else self.xs[-1] + (i - len(self.xs) + 1) * dx
        y = self.ys[j] if j < len(self.ys) else self.ys[-1] + (j - len(self.ys) + 1) * dy
        return x, y

    def elev(self, level: int) -> float:
        if level < len(self.E):
            return self.E[level]
        h = self.plan.grid.level_heights_m[-1] if self.plan.grid.level_heights_m else 3.5
        return self.E[-1] + (level - len(self.E) + 1) * h

    def support_top(self, x0, y0, x1, y1, z_limit: float, exclude: str = "") -> tuple[float, list[str]]:
        """Highest top surface under the XY rectangle that is not above z_limit."""
        best, ids = 0.0, []
        sx0, sy0, sx1, sy1 = x0 + 0.02, y0 + 0.02, x1 - 0.02, y1 - 0.02
        for e in self.out.values():
            if e.id == exclude or e.pitch:
                continue
            a = e.aabb()
            if a[3] <= sx0 or a[0] >= sx1 or a[4] <= sy0 or a[1] >= sy1:
                continue
            top = a[5]
            if top <= z_limit + 1e-6:
                if top > best + 1e-4:
                    best, ids = top, [e.id]
                elif abs(top - best) <= 1e-4:
                    ids.append(e.id)
        return best, ids

    def material(self, el: Element, default: str) -> str:
        return el.material or default

    # ------------------------------------------------------------------ level framing
    def _beam_dir(self, el: Element) -> str:
        (i0, j0), (i1, j1) = parse_node(el.start), parse_node(el.end)
        if j0 == j1:
            return "x"
        if i0 == i1:
            return "y"
        return "x"

    def _beam_section(self, el: Element) -> Section:
        x0, y0 = self.node_xy(el.start)
        x1, y1 = self.node_xy(el.end)
        span = math.hypot(x1 - x0, y1 - y0)
        concrete = (el.material or "steel") in ("concrete", "precast_concrete")
        level_beams = self.beams_by_level.get(el.level, [])
        both = {self._beam_dir(b) for b in level_beams} == {"x", "y"}
        secondary = both and self._beam_dir(el) == "y"
        return parse_section(el.section, default_beam_section(span, secondary, concrete))

    def _prepare_levels(self):
        self.beams_by_level: dict[int, list[Element]] = {}
        for el in self.plan.elements:
            if el.kind == "beam" and el.start and el.end:
                self.beams_by_level.setdefault(el.level, []).append(el)
        self.level_info = {}
        for k, beams in self.beams_by_level.items():
            dx = max((self._beam_section(b).d for b in beams if self._beam_dir(b) == "x"), default=0.0)
            dy = max((self._beam_section(b).d for b in beams if self._beam_dir(b) == "y"), default=0.0)
            nodes_x, nodes_y = set(), set()
            for b in beams:
                (nodes_x if self._beam_dir(b) == "x" else nodes_y).update(
                    {parse_node(b.start), parse_node(b.end)})
            self.level_info[k] = dict(dx=dx, dy=dy, nodes_x=nodes_x, nodes_y=nodes_y)

    def cap_z(self, node: tuple[int, int], level: int) -> float:
        """Elevation of the column cap plate top at `node` under floor `level`."""
        E = self.elev(level)
        info = self.level_info.get(level)
        if not info:
            return E
        top_x = E - info["dy"]
        if node in info["nodes_x"]:
            return top_x - info["dx"]
        if node in info["nodes_y"]:
            return E - info["dy"]
        return top_x - info["dx"]

    def beam_top(self, el: Element) -> float:
        info = self.level_info[el.level]
        E = self.elev(el.level)
        return E - info["dy"] if self._beam_dir(el) == "x" and info["dy"] > 0 else E

    # ------------------------------------------------------------------ element builders
    def build(self) -> ResolvedScene:
        self._prepare_levels()
        order = {"footing": 0, "column": 1, "beam": 2, "brace": 3, "slab": 4, "wall_panel": 5, "module": 6, "box": 7}
        # resolve in physical dependency order (lower levels first), keep plan order otherwise
        indexed = list(enumerate(self.plan.elements))
        indexed.sort(key=lambda t: (t[1].level if t[1].kind not in ("beam", "slab") else t[1].level - 0.5,
                                    order.get(t[1].kind, 9), t[0]))
        seen = set()
        for _, el in indexed:
            if el.id in seen:
                self.warnings.append(f"duplicate element id {el.id}; skipped")
                continue
            seen.add(el.id)
            try:
                r = getattr(self, f"_build_{el.kind}")(el)
            except Exception as exc:  # keep going, report
                self.warnings.append(f"element {el.id}: {exc}")
                continue
            if el.offset_m and el.kind not in ("box",):
                x, y, z = r.pos
                r.pos = (x + el.offset_m[0], y + (el.offset_m[1] if len(el.offset_m) > 1 else 0), z)
            r.preplaced = el.preplaced
            if el.color_hex:
                r.color = el.color_hex
            if el.label:
                r.label = el.label
            self.out[el.id] = r
        # keep plan order in the output dict
        ordered = {el.id: self.out[el.id] for el in self.plan.elements if el.id in self.out}
        self.out = ordered
        self._validate()
        xs = [a for e in self.out.values() for a in (e.aabb()[0], e.aabb()[3])] or [0, 1]
        ys = [a for e in self.out.values() for a in (e.aabb()[1], e.aabb()[4])] or [0, 1]
        fp = (min(min(xs), self.xs[0]), min(min(ys), self.ys[0]), max(max(xs), self.xs[-1]), max(max(ys), self.ys[-1]))
        return ResolvedScene(self.plan, self.xs, self.ys, self.E, self.out, self.warnings, fp)

    def _build_footing(self, el: Element) -> ResolvedElement:
        x, y = self._xy(el)
        sx, sy, sz = (el.size_m or [1.0, 1.0, 0.45])[:3]
        top = 0.15
        return ResolvedElement(el.id, "footing", el.label or f"Footing {el.at}", self.material(el, "concrete"), None,
                               [Part("box", (sx, sy, sz), material=self.material(el, "concrete"), bevel=0.02)],
                               (sx, sy, sz), (x, y, top - sz / 2), role="static",
                               mass=CONCRETE_DENSITY * sx * sy * sz, lift_points=[(0, 0, sz / 2)])

    def _xy(self, el: Element) -> tuple[float, float]:
        if el.position_m:
            return float(el.position_m[0]), float(el.position_m[1])
        if not el.at:
            raise ValueError("needs `at` node or `position_m`")
        return self.node_xy(el.at)

    def _build_column(self, el: Element) -> ResolvedElement:
        x, y = self._xy(el)
        node = parse_node(el.at) if el.at else None
        concrete = (el.material or "steel") in ("concrete", "precast_concrete")
        sec = parse_section(el.section, DEFAULT_RC_COLUMN if concrete else DEFAULT_STEEL_COLUMN)
        cap = self.cw = max(self.cap_w, max(sec.b, sec.d) + 0.1)
        top = self.cap_z(node, el.level + 1) if node else self.elev(el.level + 1)
        base, sup = self.support_top(x - 0.15, y - 0.15, x + 0.15, y + 0.15, self.elev(el.level) + 0.3, el.id)
        h = top - base
        if h < 0.5:
            raise ValueError(f"column height {h:.2f} m is not physical (check levels)")
        mat = self.material(el, "concrete" if concrete else "steel")
        parts: list[Part] = []
        plate = 0.025
        if sec.shape == "I":
            parts.append(Part("ibeam", (h - 2 * plate, sec.b, sec.d, sec.tw, sec.tf), material=mat, axis="z"))
        elif sec.shape == "box":
            parts.append(Part("tube", (h - 2 * plate, sec.b, sec.d, sec.t), material=mat, axis="z"))
        else:
            parts.append(Part("box", (sec.b, sec.d, h), material=mat, bevel=0.01))
        if not concrete:
            parts.append(Part("box", (cap, cap, plate), (0, 0, h / 2 - plate / 2), mat))
            parts.append(Part("box", (cap, cap, plate), (0, 0, -h / 2 + plate / 2), mat))
        dims = (cap if not concrete else sec.b, cap if not concrete else sec.d, h)
        mass = (sec.area() * h * STEEL_DENSITY) if not concrete else sec.b * sec.d * h * CONCRETE_DENSITY
        label = el.label or f"Column {el.at} · {sec.name}"
        return ResolvedElement(el.id, "column", label, mat, None, parts, dims, (x, y, base + h / 2),
                               role="anchored", mass=mass, lift_points=[(0, 0, h / 2)], supports=sup, section=sec.name)

    def _build_beam(self, el: Element) -> ResolvedElement:
        if not (el.start and el.end):
            raise ValueError("beam needs start and end nodes")
        x0, y0 = self.node_xy(el.start)
        x1, y1 = self.node_xy(el.end)
        span = math.hypot(x1 - x0, y1 - y0)
        sec = self._beam_section(el)
        direction = self._beam_dir(el)
        top = self.beam_top(el)
        info = self.level_info[el.level]
        # bearing at each end: full cap at an end node; half the cap where a collinear beam continues
        n0, n1 = parse_node(el.start), parse_node(el.end)
        full = (self.cap_w / 2 - 0.01) if (direction == "x" or info["dx"] == 0) else 0.15

        def ext(node, other):
            for b in self.beams_by_level.get(el.level, []):
                if b is el or self._beam_dir(b) != direction:
                    continue
                bn = {parse_node(b.start), parse_node(b.end)}
                if node in bn and other not in bn:
                    return -0.006
            return full

        e0, e1 = ext(n0, n1), ext(n1, n0)
        length = span + e0 + e1
        yaw = math.atan2(y1 - y0, x1 - x0)
        ux, uy = (x1 - x0) / span, (y1 - y0) / span
        shift = (e1 - e0) / 2
        mat = self.material(el, "steel")
        if sec.shape == "I":
            parts = [Part("ibeam", (length, sec.b, sec.d, sec.tw, sec.tf), material=mat)]
            mass = sec.area() * length * STEEL_DENSITY
        else:
            parts = [Part("box", (length, sec.b, sec.d), material=mat, bevel=0.01)]
            mass = sec.b * sec.d * length * (CONCRETE_DENSITY if "concrete" in mat else STEEL_DENSITY * 0.2)
        pos = ((x0 + x1) / 2 + ux * shift, (y0 + y1) / 2 + uy * shift, top - sec.d / 2)
        lift = [(-0.28 * length, 0, sec.d / 2), (0.28 * length, 0, sec.d / 2)]
        label = el.label or f"Beam {el.id} · {sec.name} · {span:.1f} m span"
        r = ResolvedElement(el.id, "beam", label, mat, None, parts, (length, sec.b, sec.d), pos, yaw=yaw,
                            role="settle", mass=mass, lift_points=lift, section=sec.name)
        _, r.supports = self.support_top(*r.aabb()[:2], *r.aabb()[3:5], r.bottom + 0.01, el.id)
        return r

    def _bay(self, el: Element) -> tuple[float, float, float, float]:
        if el.bay_from and el.bay_to:
            xa, ya = self.node_xy(el.bay_from)
            xb, yb = self.node_xy(el.bay_to)
            return min(xa, xb), min(ya, yb), max(xa, xb), max(ya, yb)
        if el.position_m and el.size_m:
            x, y = el.position_m[:2]
            return x - el.size_m[0] / 2, y - el.size_m[1] / 2, x + el.size_m[0] / 2, y + el.size_m[1] / 2
        raise ValueError("needs bay_from and bay_to nodes")

    def _build_slab(self, el: Element) -> ResolvedElement:
        x0, y0, x1, y1 = self._bay(el)
        t = (el.size_m[2] if el.size_m and len(el.size_m) > 2 else 0.2)
        base, sup = self.support_top(x0, y0, x1, y1, self.elev(el.level) + 0.01, el.id)
        lx, ly = x1 - x0 - 0.01, y1 - y0 - 0.01
        mat = self.material(el, "precast_concrete")
        yaw = 0.0
        if ly > lx:  # keep local X on the long side
            lx, ly, yaw = ly, lx, math.pi / 2
        mass = lx * ly * t * CONCRETE_DENSITY * 0.65
        lift = [(sx * 0.32 * lx, sy * 0.32 * ly, t / 2) for sx in (-1, 1) for sy in (-1, 1)]
        label = el.label or f"Precast slab {el.bay_from}–{el.bay_to} · {lx:.1f} × {ly:.1f} m"
        return ResolvedElement(el.id, "slab", label, mat, None, [Part("box", (lx, ly, t), material=mat, bevel=0.01)],
                               (lx, ly, t), ((x0 + x1) / 2, (y0 + y1) / 2, base + t / 2), yaw=yaw, role="settle",
                               mass=mass, lift_points=lift, supports=sup)

    def _build_module(self, el: Element) -> ResolvedElement:
        if el.bay_from and el.bay_to:
            x0, y0, x1, y1 = self._bay(el)
            gap = 0.03
            lx, ly = x1 - x0 - 2 * gap, y1 - y0 - 2 * gap
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        else:
            cx, cy = self._xy(el)
            lx, ly = (el.size_m or [6.0, 3.0])[:2]
        h_level = self.elev(el.level + 1) - self.elev(el.level)
        lz = el.size_m[2] if el.size_m and len(el.size_m) > 2 else h_level - 0.01
        yaw = 0.0
        if ly > lx:
            lx, ly, yaw = ly, lx, math.pi / 2
        base, sup = self.support_top(cx - lx / 2, cy - ly / 2, cx + lx / 2, cy + ly / 2, self.elev(el.level) + 0.05, el.id)
        mat = self.material(el, "module")
        p = 0.16  # corner post
        parts = [
            Part("box", (lx, ly, 0.28), (0, 0, -lz / 2 + 0.14), "steel_dark"),  # floor cassette
            Part("box", (lx, ly, 0.22), (0, 0, lz / 2 - 0.11), "steel_dark"),  # roof cassette
        ]
        for sx in (-1, 1):
            for sy in (-1, 1):
                parts.append(Part("box", (p, p, lz - 0.5), (sx * (lx / 2 - p / 2), sy * (ly / 2 - p / 2), 0), "steel_dark"))
        wall_h = lz - 0.5
        # long walls: cladding with a window band; short walls: full cladding
        for sy in (-1, 1):
            y = sy * (ly / 2 - 0.04)
            parts.append(Part("box", (lx - 2 * p, 0.06, wall_h * 0.32), (0, y, -wall_h / 2 + wall_h * 0.16), mat))
            parts.append(Part("box", (lx - 2 * p, 0.06, wall_h * 0.18), (0, y, wall_h / 2 - wall_h * 0.09), mat))
            parts.append(Part("box", (lx - 2 * p, 0.03, wall_h * 0.5), (0, y, wall_h * 0.07), "glass"))
        for sx in (-1, 1):
            parts.append(Part("box", (0.06, ly - 2 * p, wall_h), (sx * (lx / 2 - 0.04), 0, 0), mat))
        mass = lx * ly * 320.0
        lift = [(sx * 0.4 * lx, sy * 0.4 * ly, lz / 2) for sx in (-1, 1) for sy in (-1, 1)]
        label = el.label or f"Volumetric module {el.id} · {lx:.1f} × {ly:.1f} m"
        return ResolvedElement(el.id, "module", label, mat, None, parts, (lx, ly, lz), (cx, cy, base + lz / 2), yaw=yaw,
                               role="settle", mass=mass, lift_points=lift, supports=sup)

    def _build_wall_panel(self, el: Element) -> ResolvedElement:
        x0, y0 = self.node_xy(el.start)
        x1, y1 = self.node_xy(el.end)
        span = math.hypot(x1 - x0, y1 - y0)
        t = 0.15
        if el.section:
            m = re.search(r"(\d+)", el.section)
            if m:
                t = int(m[1]) / 1000
        length = span - self.cap_w - 0.02
        yaw = math.atan2(y1 - y0, x1 - x0)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        hx = abs(math.cos(yaw)) * length / 2 + abs(math.sin(yaw)) * t / 2
        hy = abs(math.sin(yaw)) * length / 2 + abs(math.cos(yaw)) * t / 2
        base, sup = self.support_top(cx - hx, cy - hy, cx + hx, cy + hy, self.elev(el.level) + 0.05, el.id)
        ceiling = self.elev(el.level + 1)
        info = self.level_info.get(el.level + 1)
        if info:
            ceiling -= info["dx"] + info["dy"]
        h = ceiling - base - 0.01
        mat = self.material(el, "precast_concrete")
        label = el.label or f"Wall panel {el.start}–{el.end}"
        return ResolvedElement(el.id, "wall_panel", label, mat, None, [Part("box", (length, t, h), material=mat, bevel=0.01)],
                               (length, t, h), (cx, cy, base + h / 2), yaw=yaw, role="anchored",
                               mass=length * t * h * CONCRETE_DENSITY,
                               lift_points=[(-0.3 * length, 0, h / 2), (0.3 * length, 0, h / 2)], supports=sup)

    def _build_brace(self, el: Element) -> ResolvedElement:
        x0, y0 = self.node_xy(el.start)
        x1, y1 = self.node_xy(el.end)
        z0, _ = self.support_top(x0 - 0.1, y0 - 0.1, x0 + 0.1, y0 + 0.1, self.elev(el.level) + 0.3, el.id)
        z0 += 0.25
        z1 = self.cap_z(parse_node(el.end), el.level + 1) - 0.25
        horiz = math.hypot(x1 - x0, y1 - y0)
        length = math.hypot(horiz, z1 - z0)
        sec = parse_section(el.section, Section("tube", 0.1683, 0.1683, t=0.008, name="CHS 168.3x8"))
        yaw = math.atan2(y1 - y0, x1 - x0)
        pitch = -math.atan2(z1 - z0, horiz)
        mat = self.material(el, "galvanized_steel")
        return ResolvedElement(el.id, "brace", el.label or f"Brace {el.start}–{el.end}", mat, None,
                               [Part("cylinder", (sec.d / 2, length), material=mat)], (length, sec.d, sec.d),
                               ((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2), yaw=yaw, pitch=pitch, role="anchored",
                               mass=sec.area() * length * STEEL_DENSITY, lift_points=[(0, 0, sec.d / 2)], section=sec.name)

    def _build_box(self, el: Element) -> ResolvedElement:
        x, y = self._xy(el)
        if el.offset_m:
            x += el.offset_m[0]
            y += el.offset_m[1] if len(el.offset_m) > 1 else 0
        lx, ly, lz = (el.size_m or [1.2, 1.0, 0.8])[:3]
        yaw = math.radians(el.yaw_deg or 0.0)
        hx = abs(math.cos(yaw)) * lx / 2 + abs(math.sin(yaw)) * ly / 2
        hy = abs(math.sin(yaw)) * lx / 2 + abs(math.cos(yaw)) * ly / 2
        base, sup = self.support_top(x - hx, y - hy, x + hx, y + hy, self.elev(el.level) + 0.05, el.id)
        mat = self.material(el, "timber")
        dens = {"timber": 550, "steel": 1500, "concrete": CONCRETE_DENSITY}.get(mat, 600)
        lift = [(sx * 0.35 * lx, sy * 0.35 * ly, lz / 2) for sx in (-1, 1) for sy in (-1, 1)]
        return ResolvedElement(el.id, "box", el.label or el.id, mat, None, [Part("box", (lx, ly, lz), material=mat, bevel=0.01)],
                               (lx, ly, lz), (x, y, base + lz / 2), yaw=yaw, role="settle", mass=lx * ly * lz * dens,
                               lift_points=lift, supports=sup)

    # ------------------------------------------------------------------ validation
    def _validate(self):
        els = [e for e in self.out.values() if not e.pitch]
        for i, a in enumerate(els):
            A = a.aabb()
            for b in els[i + 1:]:
                B = b.aabb()
                ox = min(A[3], B[3]) - max(A[0], B[0])
                oy = min(A[4], B[4]) - max(A[1], B[1])
                oz = min(A[5], B[5]) - max(A[2], B[2])
                if ox > 0.004 and oy > 0.004 and oz > 0.004:
                    # beams that rest on a column cap legitimately share the cap footprint but not volume
                    self.warnings.append(
                        f"{a.id} and {b.id} overlap by {ox:.3f}x{oy:.3f}x{oz:.3f} m -- they would collide")
        for e in els:
            if e.role == "settle" and e.bottom > 0.003 and not e.supports:
                self.warnings.append(f"{e.id} has nothing underneath it and would fall when released")


def solve(plan: StoryPlan) -> ResolvedScene:
    return StructureSolver(plan).build()
