"""Choreographer: turns (resolved structure + story events + narration timing) into a
frame-exact animation manifest for Blender plus the timeline used for captions,
titles, sound effects and stock-footage cut-ins.

Timing is negotiated in both directions:
  * the *cue word* of a sentence is when the element lands -- the crane starts its
    approach early enough to get there exactly on that word;
  * if a physical action needs longer than the words, the pause before the next
    sentence is opened up instead of speeding the crane past its limits.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np

from .config import JobConfig
from .motion import (critically_damped, min_jerk, move_duration, move_profile, pendulum_period, smooth_trapezoid, quat_from_euler_zyx, quat_mul,
                     quat_rotate, simulate_pendulum, slerp, wrap_angle, zv_shape)
from .narration import Narration
from .plan_schema import Event, StoryPlan
from .structure import ResolvedElement, ResolvedScene
from .text import norm

LAND_VERBS = {"placed", "place", "set", "sets", "lowered", "lands", "landed", "seated", "sits", "rests", "installed",
              "dropped", "positioned", "fixed", "bolted", "lowers", "settles", "settle", "slots", "slotted", "locks",
              "connects", "connected", "spans", "completes", "stacked", "stacks"}
INSTALL_ACTIONS = ("lift_and_place", "lower_into_place", "reveal")
SWAY_ZETA = 0.025  # rope + air damping of the suspended load (also used to tune the input shaper)


# =========================================================================== data
@dataclass
class Key:
    """Key moment that must line up with narration."""
    time: float
    kind: str
    target: str = ""


@dataclass
class Window:
    start: float
    end: float
    kind: str  # crane | descend | reveal | glow | camera | stock | pause
    target: str = ""
    shot: str | None = None
    data: dict = field(default_factory=dict)


@dataclass
class Timeline:
    fps: int
    total: float
    sentence_starts: list[float]
    words: list[list[tuple[str, float, float]]]  # absolute times per sentence
    windows: list[Window]
    titles: list[tuple[float, float, str]]
    sfx: list[tuple[float, str, float]]
    stock: list[dict]  # {start, end, query, sentence}
    render_ranges: list[tuple[int, int]]
    notes: list[str]


# =========================================================================== site layout
@dataclass
class SiteLayout:
    crane_base: np.ndarray | None
    crane_height: float
    jib_length: float
    yard: dict  # element id -> (pos, quat)
    dunnage: list  # (center, size)
    bounds: tuple  # whole-site xy bounds


def _q(yaw=0.0, pitch=0.0, roll=0.0):
    return quat_from_euler_zyx(yaw, pitch, roll)


def target_quat(e: ResolvedElement) -> np.ndarray:
    return _q(e.yaw, e.pitch)


def yard_pose(e: ResolvedElement) -> tuple[np.ndarray, tuple]:
    """Orientation an element is stored in at the laydown yard and its world footprint size."""
    lx, ly, lz = e.dims
    if e.kind == "column":
        return _q(0.0, math.pi / 2), (lz, ly, lx)  # lying, long axis along world X
    if e.kind == "wall_panel":
        return _q(0.0, 0.0, math.pi / 2), (lx, lz, ly)  # lying flat, height along -Y
    if e.kind == "brace":
        return _q(0.0), (lx, ly, lz)
    return _q(0.0), (lx, ly, lz)


def layout_site(scene: ResolvedScene, crane_ids: list[str], crane_kind: str) -> SiteLayout:
    fx0, fy0, fx1, fy1 = scene.footprint
    max_top = max([e.top for e in scene.elements.values()] + [3.0])
    yard: dict = {}
    dunnage = []
    if crane_kind != "tower":
        return SiteLayout(None, 0.0, 0.0, yard, dunnage, (fx0 - 8, fy0 - 8, fx1 + 8, fy1 + 8))
    # crane just outside the back-left corner; laydown yard along the left (-X) side, so a
    # pick -> place cycle needs well under a half-turn of slew
    cx = fx0 - 4.5
    cy = fy1 + 4.0
    base = np.array([cx, cy])
    x_right = fx0 - 3.0  # yard columns grow towards -X
    y_cursor = fy0 - 2.0
    row_len = max(12.0, fy1 - fy0 + 4.0)
    col_x = x_right
    col_w = 0.0
    for eid in crane_ids:
        e = scene.elements[eid]
        q, (sx, sy, sz) = yard_pose(e)
        if y_cursor + sy > fy0 - 2.0 + row_len and y_cursor > fy0 - 1.9:
            col_x -= col_w + 1.4
            col_w = 0.0
            y_cursor = fy0 - 2.0
        pos = np.array([col_x - sx / 2, y_cursor + sy / 2, 0.12 + sz / 2])
        yard[eid] = (pos, q)
        # two timber bearers under each stored element
        for f in (-0.3, 0.3):
            dunnage.append(((pos[0] + f * sx, pos[1], 0.06), (0.15, sy + 0.2, 0.12)))
        y_cursor += sy + 1.2
        col_w = max(col_w, sx)
    reach = 0.0
    for e in scene.elements.values():
        a = e.aabb()
        for x in (a[0], a[3]):
            for y in (a[1], a[4]):
                reach = max(reach, math.hypot(x - cx, y - cy))
    for pos, _ in yard.values():
        reach = max(reach, math.hypot(pos[0] - cx, pos[1] - cy) + 2)
    max_hang = max([3.0 + scene.elements[i].dims[2] if scene.elements[i].kind != "column" else 3.0 + scene.elements[i].dims[2]
                    for i in crane_ids] + [4.0])
    height = max(15.0, max_top + max_hang + 5.0)
    xs = [fx0, fx1, cx] + [p[0] for p, _ in yard.values()]
    ys = [fy0, fy1, cy] + [p[1] for p, _ in yard.values()]
    return SiteLayout(base, height, reach + 3.0, yard, dunnage, (min(xs) - 6, min(ys) - 6, max(xs) + 6, max(ys) + 6))


# =========================================================================== frame buffers
class Buffers:
    def __init__(self, n: int, ids: list[str]):
        self.n = n
        self.loc = {i: np.full((n, 3), np.nan) for i in ids}
        self.quat = {i: np.full((n, 4), np.nan) for i in ids}
        self.scale = {i: np.full(n, np.nan) for i in ids}
        self.glow = {i: np.zeros(n) for i in ids}
        self.release = {}
        self.bolt = {}
        self.visible_from = {}
        self.slew = np.full(n, np.nan)
        self.trolley = np.full(n, np.nan)
        self.hook = np.full((n, 3), np.nan)
        self.slings = [dict(vis=np.zeros(n, bool), a=np.zeros((n, 3)), b=np.zeros((n, 3))) for _ in range(4)]

    def frames(self, t0: float, n: int, fps: int) -> np.ndarray:
        f0 = int(round(t0 * fps))
        return np.arange(f0, f0 + n)


def _hold_fill(a: np.ndarray, first=None):
    """Forward-fill NaNs (hold last value); leading NaNs take `first` or the first valid value."""
    a = a.copy()
    if a.ndim == 1:
        mask = np.isnan(a)
    else:
        mask = np.isnan(a[:, 0])
    if mask.all():
        if first is not None:
            a[:] = first
        return a
    idx = np.where(~mask, np.arange(len(a)), 0)
    np.maximum.accumulate(idx, out=idx)
    a = a[idx]
    lead = np.argmax(~mask)
    a[:lead] = first if first is not None else a[lead]
    return a


# =========================================================================== crane
class CraneModel:
    def __init__(self, layout: SiteLayout, cfg: JobConfig):
        self.base = layout.crane_base
        self.z_trolley = layout.crane_height - 0.9
        self.hook_block = 0.75
        sp = cfg.physics.motion_speed
        # brisk but physically achievable drive limits (video time ~ compressed real time)
        # slew ~3.3 rpm, gentle angular acceleration (real cranes ~0.6-1 rpm; video time is compressed).
        # Faster slews swing the load harder (centripetal accel) and leave more residual sway.
        self.v_slew, self.a_slew = 0.35 * sp, 0.2 * sp
        self.v_trol, self.a_trol = 2.2 * sp, 1.5 * sp
        self.v_hoist, self.a_hoist = 2.4 * sp, 2.0 * sp
        self.anti_sway = cfg.physics.anti_sway
        self.release_gap = cfg.physics.release_gap_m

    def ik(self, p) -> tuple[float, float]:
        d = np.asarray(p[:2]) - self.base
        return math.atan2(d[1], d[0]), max(2.5, float(np.hypot(*d)))

    def xy(self, th, r):
        return np.stack([self.base[0] + r * np.cos(th), self.base[1] + r * np.sin(th)], axis=-1)


class Segments:
    """Piecewise min-jerk schedule of a scalar or vector quantity."""

    def __init__(self, v0):
        self.v0 = np.atleast_1d(np.asarray(v0, dtype=float))
        self.segs: list[tuple[float, float, np.ndarray, np.ndarray, float]] = []

    @property
    def last(self) -> np.ndarray:
        return self.segs[-1][3] if self.segs else self.v0

    def move(self, t0: float, prof, v1):
        """prof = (duration, ramp_fraction) from move_profile, or a bare duration."""
        dur, rf = prof if isinstance(prof, tuple) else (prof, 0.5)
        v1 = np.atleast_1d(np.asarray(v1, dtype=float))
        self.segs.append((t0, t0 + max(dur, 1e-6), self.last.copy(), v1, rf))
        return t0 + dur

    def sample(self, t: np.ndarray) -> np.ndarray:
        out = np.repeat(self.v0[None, :], len(t), axis=0)
        for a, b, v0, v1, rf in self.segs:
            s = smooth_trapezoid((t - a) / (b - a), rf)
            m = t >= a
            out[m] = v0 + (v1 - v0) * s[m, None]
        return out


@dataclass
class LiftPlan:
    e: ResolvedElement
    lead: float  # job start -> landing
    tail: float  # landing -> crane free
    build: object  # callable(start_time, extra_hold) -> dict
    end_hook: np.ndarray


def plan_crane_lift(crane: CraneModel, e: ResolvedElement, yard_pos, yard_q, hook_now: np.ndarray,
                    obstacles_top: float, fps: int, settle: bool, near_top: float = -1e9) -> LiftPlan:
    """near_top: highest top of already-placed elements right next to the target (not under it).
    The load hovers above it, steadies, and only then sets down vertically."""
    lc_local = np.mean(np.array(e.lift_points, dtype=float), axis=0)
    n_lift = len(e.lift_points)
    spread = max([np.linalg.norm(np.array(p[:2]) - lc_local[:2]) for p in e.lift_points] + [0.0])
    sling_h = max(1.2, 0.9 * spread) if n_lift > 1 else 0.9
    tail_up = e.kind in ("column", "wall_panel")
    q_t = target_quat(e)
    p_t = np.array(e.pos, dtype=float)

    def apex_of(p, q):
        return p + quat_rotate(q, lc_local) + np.array([0, 0, sling_h])

    if tail_up:
        # lift from the top end; in the yard the element lies flat
        top_local = np.array([0, 0, e.dims[2] / 2])
        top_world = yard_pos + quat_rotate(yard_q, top_local)
        apex_pick = top_world + np.array([0, 0, sling_h])
        u_yard = quat_rotate(yard_q, [0, 0, 1])
        u_h = np.array([u_yard[0], u_yard[1], 0.0])
        u_h /= np.linalg.norm(u_h) + 1e-9
        k_axis = np.cross(u_yard, [0, 0, 1])
        k_axis /= np.linalg.norm(k_axis) + 1e-9
        q_up = quat_mul(np.array([math.cos(math.pi / 4), *(k_axis * math.sin(math.pi / 4))]), yard_q)
    else:
        apex_pick = apex_of(yard_pos, yard_q)
        q_up = yard_q
    apex_place = apex_of(p_t, q_t)
    hang_below_apex = sling_h + (e.dims[2] if tail_up else (lc_local[2] + e.dims[2] / 2))
    z_safe = min(crane.z_trolley - crane.hook_block - 1.2,
                 max(obstacles_top + 1.2 + hang_below_apex, apex_pick[2] + 1.0, apex_place[2] + 1.0))
    z_empty = min(crane.z_trolley - crane.hook_block - 1.0, max(obstacles_top + 2.0, apex_pick[2] + 1.0))
    th_y, r_y = crane.ik(apex_pick)
    th_t, r_t = crane.ik(apex_place)
    th_h, r_h = crane.ik(hook_now)
    th_y = th_h + wrap_angle(th_y - th_h)
    th_t = th_y + wrap_angle(th_t - th_y)
    load_bottom_at_target = e.bottom
    hover = apex_place[2] + max(0.35, near_top + 0.25 - load_bottom_at_target)
    final_z = apex_place[2] + (crane.release_gap if settle else 0.0)

    def travel_prof(th0, r0, th1, r1):
        """Coordinated slew + trolley move: the slower axis sets the duration."""
        a = move_profile(abs(th1 - th0), crane.v_slew, crane.a_slew)
        b = move_profile(abs(r1 - r0), crane.v_trol, crane.a_trol)
        T, rf = max(a, b, (0.8, 0.5))
        return T, max(rf, 0.3)

    def hoist_prof(dz):
        return move_profile(abs(dz), crane.v_hoist, crane.a_hoist)

    L_empty = crane.z_trolley - z_empty
    L_travel = crane.z_trolley - (z_safe - sling_h - lc_local[2])
    T_shape_e = 0.0  # empty hook: not shaped (it is light; its sway is still simulated and damped)
    T_shape_l = pendulum_period(L_travel) / 2 if crane.anti_sway else 0.0

    # ---- schedule (relative to job start, extra hover hold inserted at build time)
    def schedule(extra_hold: float):
        V = Segments(hook_now[2])
        H = Segments([th_h, r_h])  # empty-hook approach move
        HL = Segments([th_y, r_y])  # loaded travel move (shaped separately)
        t = 0.0
        # A: approach with empty hook
        if hook_now[2] < z_empty - 0.05:
            pr = hoist_prof(z_empty - hook_now[2])
            V.move(t, pr, z_empty)
            t += 0.6 * pr[0]
        pA = travel_prof(th_h, r_h, th_y, r_y)
        H.move(t, pA, [th_y, r_y])
        tA2 = t + pA[0] + T_shape_e
        pdn = hoist_prof(V.last[0] - apex_pick[2])
        t_down0 = max(tA2 - 0.55 * pdn[0], (V.segs[-1][1] if V.segs else 0.0))
        V.move(t_down0, pdn, apex_pick[2])
        t_rig0 = t_down0 + pdn[0]
        # B: rig
        t_hoist0 = t_rig0 + 1.0
        # C: hoist
        pC = hoist_prof(z_safe - apex_pick[2])
        dC = pC[0]
        V.move(t_hoist0, pC, z_safe)
        # D: travel, starts once the load has cleared its neighbours
        tD0 = t_hoist0 + (0.8 if tail_up else 0.6) * dC  # travel once the load has cleared its neighbours
        pD = travel_prof(th_y, r_y, th_t, r_t)
        dD = pD[0]
        HL.move(tD0, pD, [th_t, r_t])
        tD1 = tD0 + dD + T_shape_l
        # E: lower to hover, overlapping the end of travel a little
        pE = hoist_prof(z_safe - hover)
        dE = pE[0]
        tE0 = max(tD1, t_hoist0 + dC)  # lower only once the horizontal move (and its shaping) is done
        V.move(tE0, pE, hover)
        tF0 = tE0 + dE
        # F: tag-line steadying, G: final set
        tG0 = tF0 + 0.9 + extra_hold
        dG = max(1.3, move_profile(hover - final_z, 0.5, 0.4)[0])  # slow, vertical final set
        V.move(tG0, dG, final_z)
        t_land = tG0 + dG
        t_free0 = t_land + (0.45 if settle else 0.8)
        V.move(t_free0, 1.2, final_z + 1.6)
        t_end = t_free0 + 1.2
        return dict(V=V, H=H, HL=HL, t_rig0=t_rig0, t_hoist0=t_hoist0, tD0=tD0, tD1=tD1, tF0=tF0, tG0=tG0, t_land=t_land,
                    t_free0=t_free0, t_end=t_end, dD=dD, tA2=tA2, t_down0=t_down0)

    base_sched = schedule(0.0)

    def build(start: float, extra_hold: float, sway0=(0.0, 0.0)) -> dict:
        S = schedule(extra_hold)
        n = int(math.ceil(S["t_end"] * fps)) + 1
        f0 = int(round(start * fps))
        t = (np.arange(n) + f0) / fps - start  # local time of each frame
        z = S["V"].sample(t)[:, 0]
        he = S["H"].sample(t)
        hl = S["HL"].sample(t)
        # Work in world space: the hanging load responds to the trolley's Cartesian acceleration
        # (including the centripetal part of a slew), so that is the path the shaper must act on.
        xy_e = crane.xy(he[:, 0], he[:, 1])
        xy_l = crane.xy(hl[:, 0], hl[:, 1])
        pick_xy = crane.xy(np.array(th_y), np.array(r_y))
        if crane.anti_sway:  # each move shaped for the pendulum length it is made with
            if T_shape_e:
                xy_e = zv_shape(t, xy_e, 2 * T_shape_e, SWAY_ZETA)
            if T_shape_l:
                xy_l = zv_shape(t, xy_l, 2 * T_shape_l, SWAY_ZETA)  # shaper tuned to the simulated damping
        pivot = xy_e + xy_l - pick_xy
        # back to crane coordinates (slew angle unwrapped continuously from the start, trolley radius)
        dxy = pivot - crane.base
        th = np.unwrap(np.arctan2(dxy[:, 1], dxy[:, 0]))
        th += th_h - th[0] + wrap_angle(th[0] - th_h) if len(th) else 0.0
        r = np.hypot(dxy[:, 0], dxy[:, 1])
        # ---- pendulum: empty-hook part and loaded part
        sway = np.zeros((n, 2))
        rope_e = crane.z_trolley - z
        m_e = t < S["t_rig0"]
        if m_e.sum() > 3:
            zeta = np.where(t[m_e] > S["t_down0"], 0.5, 0.03)
            sway[m_e] = simulate_pendulum(t[m_e], pivot[m_e], rope_e[m_e], zeta, theta0=sway0)
            k = np.where(m_e)[0]
            fade = 1 - min_jerk((t[k] - S["t_down0"]) / max(0.1, S["t_rig0"] - S["t_down0"]))
            sway[k] *= np.where(t[k] > S["t_down0"], fade, 1.0)[:, None]
        m_l = (t >= S["t_hoist0"]) & (t <= S["t_land"])
        if m_l.sum() > 3:
            com_drop = sling_h + lc_local[2]
            rope_l = crane.z_trolley - (z[m_l] - com_drop)
            zeta = np.where(t[m_l] >= S["tF0"], 0.75, SWAY_ZETA)
            s_l = simulate_pendulum(t[m_l], pivot[m_l], rope_l, zeta)
            k = np.where(m_l)[0]
            fade = 1 - min_jerk((t[k] - S["tG0"]) / (S["t_land"] - S["tG0"]))
            s_l *= np.where(t[k] >= S["tG0"], fade, 1.0)[:, None]
            sway[k] = s_l
        apex = np.column_stack([pivot, z]) + np.column_stack([sway, np.zeros(n)])
        # ---- load pose
        load_p = np.full((n, 3), np.nan)
        load_q = np.full((n, 4), np.nan)
        dD = S["dD"]
        for i in range(n):
            ti = t[i]
            if ti < S["t_hoist0"]:
                continue
            if ti > S["t_land"]:
                break
            if tail_up and ti < S["tD0"] + 0.3 * dD:
                top_z = z[i] - sling_h
                cosphi = np.clip((top_z - yard_pos[2]) / e.dims[2], 0.0, 1.0)
                phi = math.acos(cosphi)  # angle from vertical
                ang = math.pi / 2 - phi
                q_tilt = np.array([math.cos(ang / 2), *(k_axis * math.sin(ang / 2))])
                q = quat_mul(q_tilt, yard_q)
                u = quat_rotate(q, [0, 0, 1])
                top = np.array([apex[i, 0], apex[i, 1], top_z])
                p = top - u * e.dims[2] / 2
                if phi > 0.02:  # still tailing: bottom end on the ground, no sway
                    top[:2] = apex_pick[:2]
                    p = top - u * e.dims[2] / 2
                    apex[i, :2] = apex_pick[:2]
                load_p[i], load_q[i] = p, q
                continue
            s_yaw = min_jerk((ti - S["tD0"]) / dD) if dD > 0 else 1.0
            q = slerp(q_up, q_t, float(s_yaw))
            # rigid pendulum tilt about the apex
            L = max(0.5, crane.z_trolley - z[i])
            ax, ay = math.asin(np.clip(sway[i, 0] / L, -0.5, 0.5)), math.asin(np.clip(sway[i, 1] / L, -0.5, 0.5))
            q_sw = quat_mul(_q(0, -ax * 0.6, 0), _q(0, 0, ay * 0.6))  # load hangs along the rope
            q_full = quat_mul(q_sw, q)
            rel = -np.array([0, 0, sling_h]) - quat_rotate(q, lc_local)
            p = apex[i] + quat_rotate(q_sw, rel)
            load_p[i], load_q[i] = p, q_full
        # exact final pose at landing (anchored) / release gap (settle)
        i_land = int(np.searchsorted(t, S["t_land"]))
        sling_vis = (t >= S["t_rig0"]) & (t < S["t_free0"])
        return dict(n=n, f0=f0, t=t, slew=th, trolley=r, apex=apex, load_p=load_p, load_q=load_q,
                    sling_vis=sling_vis, i_land=i_land, sched=S, lc_local=lc_local, sling_h=sling_h,
                    final_p=p_t + np.array([0, 0, crane.release_gap if settle else 0.0]), final_q=q_t)

    end_hook = np.array([*crane.xy(np.array(th_t), np.array(r_t)).tolist(), final_z + 1.6])
    lead = base_sched["t_land"]
    tail = base_sched["t_end"] - base_sched["t_land"]
    return LiftPlan(e, lead, tail, build, end_hook)


# =========================================================================== camera
def lens_fov(width: int, height: int, focal: float = 30.0, sensor: float = 36.0) -> tuple[float, float]:
    """(horizontal, vertical) FOV for Blender sensor_fit AUTO (sensor spans the larger side)."""
    if width >= height:
        fh = 2 * math.atan(sensor / 2 / focal)
        fv = 2 * math.atan(math.tan(fh / 2) * height / width)
    else:
        fv = 2 * math.atan(sensor / 2 / focal)
        fh = 2 * math.atan(math.tan(fv / 2) * width / height)
    return fh, fv


# =========================================================================== main
class Choreographer:
    def __init__(self, plan: StoryPlan, scene: ResolvedScene, narration: Narration, cfg: JobConfig, sentences: list[str]):
        self.plan, self.scene, self.nar, self.cfg = plan, scene, narration, cfg
        self.sentences = sentences
        self.fps = cfg.video.fps
        self.notes: list[str] = []

    # ------------------------------------------------------------------ helpers
    def _cue_offset(self, si: int, cue: str | None, fallback_frac: float = 0.72) -> float:
        clip = self.nar.clips[si]
        ws = clip.words
        if not ws:
            return clip.duration * fallback_frac
        if cue:
            c = norm(cue)
            for w in ws:
                if norm(w.text) == c:
                    return w.start
            for w in ws:
                if c and (c in norm(w.text) or norm(w.text) in c) and len(norm(w.text)) > 2:
                    return w.start
            self.notes.append(f"cue word {cue!r} not found in sentence {si}; using a verb/default")
        for w in ws:
            if norm(w.text) in LAND_VERBS:
                return w.start
        return clip.duration * fallback_frac

    def _near_top(self, e: ResolvedElement, placed: set, margin: float = 0.6) -> float:
        """Highest top among placed elements beside e's target footprint (supports excluded)."""
        a = e.aabb()
        best = -1e9
        for pid in placed:
            if pid == e.id or pid in e.supports:
                continue
            b = self.scene.elements[pid].aabb()
            if b[3] < a[0] - margin or b[0] > a[3] + margin or b[4] < a[1] - margin or b[1] > a[4] + margin:
                continue
            if b[5] <= a[2] + 0.01:  # entirely below the load's seat: cannot be hit from above
                continue
            best = max(best, b[5])
        return best

    def _expand(self) -> tuple[list[list[tuple[Event, str]]], set]:
        per_sentence: list[list[tuple[Event, str]]] = [[] for _ in self.sentences]
        installed = set()
        for ev in self.plan.events:
            if not (0 <= ev.sentence < len(self.sentences)):
                self.notes.append(f"event refers to sentence {ev.sentence} which does not exist; moved to the last one")
                ev = ev.model_copy(update={"sentence": len(self.sentences) - 1})
            tg = [t for t in ev.targets if t in self.scene.elements]
            for t in ev.targets:
                if t not in self.scene.elements:
                    self.notes.append(f"event target {t!r} is not a known element; ignored")
            if ev.action in INSTALL_ACTIONS:
                for t in tg:
                    if t in installed:
                        self.notes.append(f"{t} is installed twice; second install ignored")
                        continue
                    installed.add(t)
                    per_sentence[ev.sentence].append((ev, t))
            else:
                per_sentence[ev.sentence].append((ev, ",".join(tg)))
        return per_sentence, installed

    # ------------------------------------------------------------------ solve
    def run(self) -> tuple[dict, Timeline]:
        cfg, fps = self.cfg, self.fps
        crane_kind = self.plan.site.crane
        per_sentence, installed = self._expand()
        els = self.scene.elements
        for eid, e in els.items():
            if eid not in installed and not e.preplaced:
                e.preplaced = True
                if e.kind not in ("footing",):
                    self.notes.append(f"{eid} has no install event; shown in place from the start")
        crane_ids = []
        for evs in per_sentence:
            for ev, t in evs:
                if ev.action == "lift_and_place" and crane_kind == "tower":
                    crane_ids.append(t)
        layout = layout_site(self.scene, crane_ids, crane_kind)
        crane = CraneModel(layout, cfg) if layout.crane_base is not None else None

        placed = {eid for eid, e in els.items() if e.preplaced}
        hook = None
        if crane:
            fx0, fy0, fx1, fy1 = self.scene.footprint
            top0 = max([els[p].top for p in placed] + [2.0])
            hook = np.array([(fx0 + fx1) / 2, (fy0 + fy1) / 2, min(crane.z_trolley - 2.0, top0 + 3.0)])
        yard_top = max([p[2] + 0.5 for p, _ in layout.yard.values()] + [0.0])

        jobs = []  # (kind, start, payload)
        titles, sfx, windows, stock = [], [], [], []
        starts: list[float] = []
        cursor = cfg.video.intro_s
        crane_free = 0.0
        fixed = self.nar.fixed_starts
        last_key = 0.0
        for si, evs in enumerate(per_sentence):
            clip = self.nar.clips[si]
            # earliest sentence start allowed by the crane for the first lift of this sentence
            first_idx = next((k for k, (ev, _) in enumerate(evs) if ev.action in ("lift_and_place", "lower_into_place")), None)
            need = cursor
            lift_plan = None
            if first_idx is not None:
                ev, t = evs[first_idx]
                cue = self._cue_offset(si, ev.cue_word)
                if ev.action == "lift_and_place" and crane:
                    obst = max([els[p].top for p in placed] + [yard_top, 0.5])
                    lift_plan = plan_crane_lift(crane, els[t], *layout.yard[t], hook, obst, fps,
                                                els[t].role == "settle", self._near_top(els[t], placed))
                    need = max(need, crane_free + lift_plan.lead - cue)
                else:
                    need = max(need, crane_free + 4.2 - cue)
            s = fixed[si] if fixed else need
            if fixed and need > s + 0.05:
                self.notes.append(f"sentence {si}: recorded voice-over leaves {need - s:.1f}s too little for the lift; "
                                  f"it will land late")
            starts.append(s)
            sent_end = s + clip.duration
            local_keys = [sent_end]
            for k, (ev, tgt) in enumerate(evs):
                cue_t = s + self._cue_offset(si, ev.cue_word)
                shot = ev.shot
                if ev.action == "lift_and_place" and crane:
                    e = els[tgt]
                    obst = max([els[p].top for p in placed] + [yard_top, 0.5])
                    lp = lift_plan if (lift_plan is not None and k == first_idx) else \
                        plan_crane_lift(crane, e, *layout.yard[tgt], hook, obst, fps, e.role == "settle",
                                        self._near_top(e, placed))
                    lift_plan = None
                    land_goal = cue_t if k == first_idx else crane_free + lp.lead
                    start = max(crane_free, land_goal - lp.lead)
                    idle = start - crane_free
                    extra = 0.0
                    if idle > 0.2:  # start earlier and hold the load steady at hover instead of idling
                        extra = min(idle, 2.5)
                        start -= extra
                    data = lp.build(start, extra)
                    land = start + lp.lead + extra
                    jobs.append(("crane", start, dict(el=tgt, data=data, settle=e.role == "settle")))
                    windows.append(Window(start, start + data["sched"]["t_end"], "crane", tgt, shot,
                                          dict(t_hoist=start + data["sched"]["t_hoist0"], t_land=land,
                                               t_travel0=start + data["sched"]["tD0"],
                                               t_lower=start + data["sched"]["tF0"] - 1.5)))
                    crane_free = start + data["sched"]["t_end"]
                    hook = lp.end_hook
                    placed.add(tgt)
                    local_keys.append(land + 0.6)
                    self._title(titles, ev, e, land)
                    sfx.append((land + (0.03 if e.role == "settle" else 0.0), self._sfx_kind(e), 1.0))
                    sfx.append((start + data["sched"]["t_rig0"] + 0.3, "rig", 0.35))
                elif ev.action in ("lift_and_place", "lower_into_place"):
                    e = els[tgt]
                    lead = 4.2
                    land_goal = cue_t if k == first_idx else crane_free + lead
                    start = max(crane_free, land_goal - lead)
                    jobs.append(("descend", start, dict(el=tgt, lead=lead, settle=e.role == "settle")))
                    land = start + lead
                    windows.append(Window(start, land + 0.8, "descend", tgt, shot, dict(t_land=land)))
                    crane_free = land + 0.4
                    placed.add(tgt)
                    local_keys.append(land + 0.6)
                    self._title(titles, ev, e, land)
                    sfx.append((land + 0.03, self._sfx_kind(e), 1.0))
                elif ev.action == "reveal":
                    e = els[tgt]
                    jobs.append(("reveal", cue_t - 0.8, dict(el=tgt)))
                    windows.append(Window(cue_t - 0.8, cue_t + 1.0, "reveal", tgt, shot, dict(t_land=cue_t)))
                    placed.add(tgt)
                    self._title(titles, ev, e, cue_t)
                elif ev.action in ("bolt", "highlight"):
                    ids = [x for x in tgt.split(",") if x]
                    jobs.append((ev.action, cue_t, dict(els=ids)))
                    windows.append(Window(cue_t - 0.6, cue_t + 1.6, "glow", tgt, shot, dict(t_land=cue_t)))
                    if ev.caption_title and cfg.captions.titles:
                        titles.append((cue_t - 0.2, cue_t + 2.6, ev.caption_title))
                    if ev.action == "bolt":
                        sfx.append((cue_t, "bolt", 0.6))
                    local_keys.append(cue_t + 0.5)
                elif ev.action == "stock_clip":
                    stock.append(dict(sentence=si, query=ev.stock_query or self.sentences[si], start=s))
                    windows.append(Window(s, sent_end, "stock", "", shot))
                elif ev.action == "camera" or shot:
                    windows.append(Window(s, sent_end, "camera", tgt, shot))
                    if ev.caption_title and cfg.captions.titles:
                        titles.append((s, min(sent_end, s + 3.0), ev.caption_title))
            last_key = max(local_keys)
            nxt_gap = cfg.narration.sentence_gap_s
            cursor = max(sent_end + nxt_gap, last_key)
        total = max(cursor + cfg.video.outro_s, crane_free + 1.0)
        if fixed and self.nar.total_fixed:
            total = max(total, self.nar.total_fixed + 0.5)
        # stock windows end where the next sentence starts
        for st in stock:
            si = st["sentence"]
            st["end"] = starts[si + 1] if si + 1 < len(starts) else total
            st["start"] = max(0.0, st["start"] - 0.15)
        self.layout, self.crane = layout, crane
        manifest = self._build_manifest(jobs, windows, total, layout, crane, stock)
        words = [[(w.text, s + w.start, s + w.end) for w in c.words] for c, s in zip(self.nar.clips, starts)]
        ranges = self._render_ranges(stock, total)
        tl = Timeline(fps, total, starts, words, windows, titles, sorted(sfx), stock, ranges, self.notes)
        manifest["render_ranges"] = ranges
        return manifest, tl

    def _title(self, titles, ev: Event, e: ResolvedElement, land: float):
        if not self.cfg.captions.titles:
            return
        text = ev.caption_title or e.label
        if text:
            titles.append((land - 0.3, land + 2.8, text))

    @staticmethod
    def _sfx_kind(e: ResolvedElement) -> str:
        if "concrete" in e.material:
            return "concrete"
        if e.kind == "module":
            return "module"
        if e.material == "timber":
            return "timber"
        return "steel"

    def _render_ranges(self, stock, total) -> list[tuple[int, int]]:
        fps = self.fps
        n = int(math.ceil(total * fps))
        need = np.ones(n, bool)
        pad = self.cfg.video.transition_s
        for st in stock:
            a = int(math.ceil((st["start"] + pad) * fps))
            b = int(math.floor((st["end"] - pad) * fps))
            if b > a:
                need[a:b] = False
        ranges, i = [], 0
        while i < n:
            if need[i]:
                j = i
                while j < n and need[j]:
                    j += 1
                ranges.append((i, j - 1))
                i = j
            else:
                i += 1
        return ranges

    # ------------------------------------------------------------------ manifest
    def _build_manifest(self, jobs, windows, total, layout: SiteLayout, crane: CraneModel | None, stock) -> dict:
        fps = self.fps
        n = int(math.ceil(total * fps)) + 1
        els = self.scene.elements
        B = Buffers(n, list(els))
        for eid, e in els.items():
            if e.preplaced:
                B.loc[eid][0] = e.pos
                B.quat[eid][0] = target_quat(e)
            elif eid in layout.yard:
                p, q = layout.yard[eid]
                B.loc[eid][0] = p
                B.quat[eid][0] = q
            else:  # appears later (reveal / descend)
                B.loc[eid][0] = e.pos
                B.quat[eid][0] = target_quat(e)
        if crane:
            fx0, fy0, fx1, fy1 = self.scene.footprint
            th0, r0 = crane.ik(np.array([(fx0 + fx1) / 2, (fy0 + fy1) / 2]))
            B.slew[0], B.trolley[0] = th0, r0
            top0 = max([e.top for e in els.values() if e.preplaced] + [2.0])
            B.hook[0] = [*crane.xy(np.array(th0), np.array(r0)).tolist(), min(crane.z_trolley - 2.0, top0 + 3.0)]

        def put(arr, f0, vals):
            a, b = max(0, f0), min(n, f0 + len(vals))
            if b > a:
                arr[a:b] = vals[a - f0:b - f0]

        prev_slew = None
        for kind, start, d in jobs:
            if kind == "crane":
                D = d["data"]
                eid = d["el"]
                f0, nn = D["f0"], D["n"]
                slew = D["slew"]
                if prev_slew is not None:
                    slew = slew + 2 * math.pi * round((prev_slew - slew[0]) / (2 * math.pi))
                prev_slew = slew[-1]
                put(B.slew, f0, slew)
                put(B.trolley, f0, D["trolley"])
                hook_pts = D["apex"].copy()
                put(B.hook, f0, hook_pts)
                lp, lq = D["load_p"], D["load_q"]
                valid = ~np.isnan(lp[:, 0])
                il = D["i_land"]
                # exact landing pose for the last frames before release
                if il < nn:
                    lp[il:] = D["final_p"]
                    lq[il:] = D["final_q"]
                    valid[il:] = True
                idx = np.where(valid)[0]
                if len(idx):
                    a = idx[0]
                    put(B.loc[eid], f0 + a, lp[a:])
                    put(B.quat[eid], f0 + a, lq[a:])
                if d["settle"]:
                    B.release[eid] = f0 + il + 1
                    # after release: physics owns the body; mark as NaN so nothing is keyed
                # slings
                e = els[eid]
                vis = D["sling_vis"]
                for si_, lpt in enumerate(e.lift_points[:4]):
                    S = B.slings[si_]
                    for i in range(nn):
                        f = f0 + i
                        if f >= n or not vis[i]:
                            continue
                        if np.isnan(lp[i, 0]):  # before hoist: load still at its yard pose
                            p0, q0 = layout.yard[eid]
                            if e.kind in ("column", "wall_panel"):
                                tip = p0 + quat_rotate(q0, [0, 0, e.dims[2] / 2])
                            else:
                                tip = p0 + quat_rotate(q0, lpt)
                        else:
                            if e.kind in ("column", "wall_panel"):
                                tip = lp[i] + quat_rotate(lq[i], [0, 0, e.dims[2] / 2])
                            else:
                                tip = lp[i] + quat_rotate(lq[i], lpt)
                        S["vis"][f] = True
                        S["a"][f] = hook_pts[i]
                        S["b"][f] = tip
                        if e.kind in ("column", "wall_panel") and si_ > 0:
                            S["vis"][f] = False
            elif kind == "descend":
                eid = d["el"]
                e = els[eid]
                f0 = int(round(start * fps))
                lead = d["lead"]
                nn = int(lead * fps) + 1
                t = np.arange(nn) / fps
                z0 = e.pos[2] + 7.0
                hover = e.pos[2] + 0.35
                zf = e.pos[2] + (self.cfg.physics.release_gap_m if d["settle"] else 0.0)
                z = np.where(t < lead - 1.6, z0 + (hover - z0) * min_jerk(t / (lead - 1.6)),
                             hover + (zf - hover) * min_jerk((t - (lead - 1.3)) / 1.3))
                # small decaying drift while it descends, faded to exactly zero at touchdown
                sway = 0.25 * np.sin(2 * math.pi * t / 3.2) * np.exp(-t / 1.2) * (1 - min_jerk(t / (lead - 1.3)))
                P = np.column_stack([np.full(nn, e.pos[0]) + sway, np.full(nn, e.pos[1]), z])
                put(B.loc[eid], f0, P)
                put(B.quat[eid], f0, np.repeat(target_quat(e)[None], nn, 0))
                B.visible_from[eid] = f0
                if d["settle"]:
                    B.release[eid] = f0 + nn
            elif kind == "reveal":
                eid = d["el"]
                f0 = int(round(start * fps))
                nn = int(0.8 * fps)
                s = 1 - (1 - np.linspace(0, 1, nn)) ** 3
                put(B.scale[eid], f0, np.maximum(0.001, s))
                B.visible_from[eid] = f0
            elif kind in ("bolt", "highlight"):
                for eid in d["els"]:
                    if kind == "bolt":
                        B.bolt[eid] = int(round(start * fps))
        # glow pulses on key moments
        for w in windows:
            if w.kind in ("crane", "descend", "reveal", "glow"):
                tk = w.data.get("t_land", w.start)
                for eid in [x for x in w.target.split(",") if x]:
                    f = int(round(tk * fps))
                    nn = int(1.4 * fps)
                    pulse = np.sin(np.linspace(0, math.pi, nn)) ** 2 * (1.0 if w.kind == "glow" else 0.6)
                    a, b = max(0, f), min(n, f + nn)
                    B.glow[eid][a:b] = np.maximum(B.glow[eid][a:b], pulse[:b - a])

        cam = self._camera(windows, total, n, B, layout)
        return self._serialize(B, n, layout, crane, cam)

    # ------------------------------------------------------------------ camera
    def _camera(self, windows: list[Window], total, n, B: Buffers, layout: SiteLayout) -> dict:
        fps = self.fps
        W, H = self.cfg.video.size()
        focal = 30.0
        fh, fv = lens_fov(W, H, focal)
        fmin = min(fh, fv)
        fx0, fy0, fx1, fy1 = self.scene.footprint
        bc = np.array([(fx0 + fx1) / 2, (fy0 + fy1) / 2, 0.0])
        top = max([e.top for e in self.scene.elements.values()] + [3.0])
        b_rad = 0.5 * math.hypot(fx1 - fx0, fy1 - fy0)
        sx0, sy0, sx1, sy1 = layout.bounds
        site_c = np.array([(sx0 + sx1) / 2, (sy0 + sy1) / 2, 0.0])
        site_r = 0.42 * math.hypot(sx1 - sx0, sy1 - sy0)
        t = np.arange(n) / fps

        tgt = np.zeros((n, 3))
        rad = np.zeros(n)
        az = np.zeros(n)
        el = np.zeros(n)
        base_az = math.radians(-58)
        # default: overview of the structure with a slow drift
        for i in range(n):
            tgt[i] = bc + np.array([0, 0, top * 0.38])
            rad[i] = b_rad * 1.05 + top * 0.3
            az[i] = base_az + math.radians(9) * math.sin(t[i] * 0.11)
            el[i] = math.radians(24)
        # establishing at the start
        intro_end = self.cfg.video.intro_s + 0.6
        first_w = min([w.start for w in windows if w.kind in ("crane", "descend")] + [total])
        m = t < max(intro_end, min(first_w, intro_end + 2.0))
        tgt[m] = site_c + np.array([0, 0, top * 0.5])
        rad[m] = site_r
        az[m] = base_az - math.radians(18) + t[m] * math.radians(3.0)
        el[m] = math.radians(30)

        def load_pos(eid, i):
            p = B.loc[eid][min(i, n - 1)]
            return p if not np.isnan(p[0]) else None

        for w in sorted(windows, key=lambda w: w.start):
            a, b = int(w.start * fps), min(n, int(w.end * fps) + 1)
            if w.kind == "crane":
                e = self.scene.elements[w.target]
                P = np.array(e.pos)
                L = max(e.dims)
                th = int(w.data["t_hoist"] * fps)
                tl = int(w.data["t_land"] * fps)
                tlow = int(w.data["t_lower"] * fps)
                travel_dir = None
                for i in range(max(a, th), min(b, n)):
                    lp = load_pos(w.target, i)
                    if lp is None:
                        continue
                    if travel_dir is None:
                        dv = P[:2] - lp[:2]
                        travel_dir = math.atan2(dv[1], dv[0]) if np.linalg.norm(dv) > 1 else base_az + math.pi / 2
                        # look across the travel direction, preferring the front of the site
                        c1, c2 = travel_dir - math.pi / 2, travel_dir + math.pi / 2
                        pick = min((c1, c2), key=lambda c: abs(wrap_angle(c - base_az)))
                        job_az = base_az + 0.55 * wrap_angle(pick - base_az)
                    if i < tlow:  # following the load across the site
                        mid = 0.55 * lp + 0.45 * P
                        dist = np.linalg.norm(lp[:2] - P[:2])
                        tgt[i] = mid
                        rad[i] = 0.5 * dist + 0.5 * L + 2.5
                        el[i] = math.radians(26)
                    else:  # landing close-up
                        tgt[i] = P + np.array([0, 0, 0.3])
                        rad[i] = max(2.8, 0.6 * L + 1.2)
                        el[i] = math.radians(22)
                    az[i] = job_az
                # hold the close-up a moment after landing, then ease back out
                for i in range(tl, min(n, tl + int(1.2 * fps))):
                    tgt[i] = P + np.array([0, 0, 0.3])
                    rad[i] = max(2.8, 0.6 * L + 1.2)
                    az[i] = az[tl - 1] if tl > 0 else base_az
                    el[i] = math.radians(22)
            elif w.kind in ("descend", "reveal", "glow"):
                ids = [x for x in w.target.split(",") if x]
                if not ids:
                    continue
                pts = np.array([self.scene.elements[x].pos for x in ids])
                c = pts.mean(0)
                r = max(3.0, 0.5 * max(max(self.scene.elements[x].dims) for x in ids)
                        + 0.5 * float(np.ptp(pts[:, :2], axis=0).max() if len(pts) > 1 else 0) + 1.5)
                tgt[a:b] = c
                rad[a:b] = r
                el[a:b] = math.radians(25)
            if w.shot:
                self._apply_shot(w, a, b, tgt, rad, az, el, bc, b_rad, top, site_c, site_r, t)

        # outro: wide orbit of the finished structure
        out0 = int(max(0, total - self.cfg.video.outro_s - 0.5) * fps)
        tgt[out0:] = bc + np.array([0, 0, top * 0.4])
        rad[out0:] = b_rad * 1.15 + top * 0.35
        az[out0:] = base_az - math.radians(20) + (t[out0:] - t[out0]) * math.radians(7)
        el[out0:] = math.radians(28)

        dt = 1 / fps
        az = np.unwrap(az)
        tgt_s = critically_damped(tgt, dt, 1.6)
        rad_s = critically_damped(rad, dt, 1.1)
        az_s = critically_damped(az, dt, 0.9)
        el_s = critically_damped(el, dt, 1.2)
        dist = rad_s / math.sin(fmin / 2)
        cam = tgt_s + np.column_stack([dist * np.cos(el_s) * np.cos(az_s), dist * np.cos(el_s) * np.sin(az_s),
                                       dist * np.sin(el_s)])
        cam[:, 2] = np.maximum(cam[:, 2], 1.7)
        d = tgt_s - cam
        yaw = np.arctan2(d[:, 1], d[:, 0]) - math.pi / 2
        pitch = math.pi / 2 + np.arctan2(d[:, 2], np.hypot(d[:, 0], d[:, 1]))
        yaw = np.unwrap(yaw)
        rot = np.column_stack([pitch, np.zeros(n), yaw])
        return dict(focal=focal, sensor=36.0, loc=cam.round(4).tolist(), rot=rot.round(5).tolist())

    def _apply_shot(self, w, a, b, tgt, rad, az, el, bc, b_rad, top, site_c, site_r, t):
        s = w.shot
        if s == "establishing":
            tgt[a:b] = site_c + np.array([0, 0, top * 0.5])
            rad[a:b] = site_r
            el[a:b] = math.radians(30)
        elif s in ("wide", "orbit"):
            tgt[a:b] = bc + np.array([0, 0, top * 0.4])
            rad[a:b] = b_rad * 1.1 + top * 0.3
            if s == "orbit":
                az[a:b] = az[a] + (t[a:b] - t[a]) * math.radians(12)
        elif s == "top":
            el[a:b] = math.radians(68)
            rad[a:b] = np.maximum(rad[a:b], b_rad * 0.9)
        elif s == "close":
            rad[a:b] = np.maximum(2.5, rad[a:b] * 0.6)
        elif s == "medium":
            rad[a:b] = np.clip(rad[a:b], b_rad * 0.45, b_rad * 0.8)

    # ------------------------------------------------------------------ serialise
    def _serialize(self, B: Buffers, n: int, layout: SiteLayout, crane: CraneModel | None, cam: dict) -> dict:
        cfg = self.cfg
        W, H = cfg.video.size()
        out_els = []
        for eid, e in self.scene.elements.items():
            loc = _hold_fill(B.loc[eid])
            quat = _hold_fill(B.quat[eid])
            rel = B.release.get(eid)
            keys = _compress_keys(loc, quat, rel)
            scale = B.scale[eid]
            scale_keys = None
            ramp = np.where(~np.isnan(scale))[0]
            if len(ramp):
                scale_keys = [[0, 0.001]] + [[int(f), round(float(scale[f]), 4)] for f in ramp] + \
                             [[int(min(n - 1, ramp[-1] + 1)), 1.0]]
            g = B.glow[eid]
            gf = np.where(g > 1e-3)[0]
            glow_keys = []
            if len(gf):
                fr = sorted(set(gf.tolist()) | {int(max(0, gf.min() - 1)), int(min(n - 1, gf.max() + 1))})
                glow_keys = [[int(f), round(float(g[f]), 4)] for f in fr[::2]] + [[int(fr[-1]), 0.0]]
            out_els.append(dict(
                id=eid, kind=e.kind, label=e.label, material=e.material, color=e.color, role=e.role, mass=e.mass,
                dims=list(e.dims), parts=[dict(shape=p.shape, size=list(p.size), offset=list(p.offset),
                                               material=p.material, axis=p.axis, bevel=p.bevel) for p in e.parts],
                target=dict(pos=list(e.pos), quat=target_quat(e).tolist()),
                keys=keys, release_frame=rel, bolt_frame=B.bolt.get(eid), supports=e.supports,
                visible_from=B.visible_from.get(eid), scale_keys=scale_keys, glow_keys=glow_keys,
                preplaced=e.preplaced))
        crane_out = None
        if crane:
            slew = _hold_fill(B.slew)
            trol = _hold_fill(B.trolley)
            hook = _hold_fill(B.hook)
            slings = []
            for S in B.slings:
                if not S["vis"].any():
                    continue
                slings.append(dict(vis=_runs(S["vis"]), a=S["a"].round(4).tolist(), b=S["b"].round(4).tolist()))
            crane_out = dict(base=layout.crane_base.tolist(), height=layout.crane_height, jib=layout.jib_length,
                             z_trolley=crane.z_trolley, hook_block=crane.hook_block,
                             slew=slew.round(5).tolist(), trolley=trol.round(4).tolist(), hook=hook.round(4).tolist(),
                             slings=slings)
        sc = self.scene
        return dict(
            fps=self.fps, frames=n, resolution=[W, H], engine=cfg.video.resolved_engine(),
            samples=cfg.video.resolved_samples(), gpu=cfg.video.gpu, quality=cfg.video.quality,
            site=dict(environment=self.plan.site.environment, ground=self.plan.site.ground,
                      footprint=list(sc.footprint), xs=sc.xs, ys=sc.ys, grid=self.plan.site.show_grid_markings,
                      bounds=list(layout.bounds)),
            physics=dict(substeps=cfg.physics.substeps, iterations=cfg.physics.solver_iterations),
            elements=out_els, crane=crane_out, camera=cam,
            dunnage=[dict(pos=list(map(float, p)), size=list(s)) for p, s in layout.dunnage],
        )


def _runs(mask: np.ndarray) -> list[list[int]]:
    out, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            out.append([i, j - 1])
            i = j
        else:
            i += 1
    return out


def _compress_keys(loc: np.ndarray, quat: np.ndarray, release: int | None) -> list:
    """Keyframes [frame, x, y, z, qw, qx, qy, qz]; static runs collapse to their endpoints.
    Keys stop at the release frame (physics takes over after it)."""
    n = len(loc) if release is None else min(len(loc), release + 1)
    keys = []
    prev_moving = True
    for f in range(n):
        moving = f == 0 or f == n - 1 or not (np.allclose(loc[f], loc[f - 1], atol=1e-5) and
                                              np.allclose(quat[f], quat[f - 1], atol=1e-6))
        nxt_moving = f + 1 < n and not (np.allclose(loc[f + 1], loc[f], atol=1e-5) and
                                        np.allclose(quat[f + 1], quat[f], atol=1e-6))
        if moving or nxt_moving or prev_moving:
            keys.append([f, *np.round(loc[f], 5).tolist(), *np.round(quat[f], 6).tolist()])
        prev_moving = moving
    return keys
