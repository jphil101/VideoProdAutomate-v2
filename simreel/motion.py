"""Physically based motion primitives (pure numpy, testable outside Blender).

* minimum-jerk point-to-point profiles honouring velocity/acceleration limits
* zero-vibration (ZV) input shaping -- what real crane anti-sway controllers do
* a damped spherical-pendulum integrator for the hook + suspended load, driven by
  the trolley acceleration and the changing hoist-rope length (variable-length
  pendulum: theta'' = -(g/L) sin(theta) - (2 L'/L) theta' - a_trolley cos(theta)/L)
"""
from __future__ import annotations

import math

import numpy as np

G = 9.81


def min_jerk(s: np.ndarray) -> np.ndarray:
    """Normalised minimum-jerk position profile on s in [0,1]."""
    s = np.clip(s, 0.0, 1.0)
    return s**3 * (10 - 15 * s + 6 * s * s)


def move_profile(distance: float, v_max: float, a_max: float, t_min: float = 0.5) -> tuple[float, float]:
    """Smooth trapezoid (raised-cosine acceleration ramps, cruise in between) respecting
    v_max and a_max.  Returns (duration, ramp_fraction).  Acceleration starts and ends at
    zero, so jerk stays finite -- like a real drive with S-curve ramps."""
    d = abs(distance)
    if d < 1e-6:
        return 0.0, 0.5
    tr = v_max * math.pi / (2 * a_max)
    if d >= v_max * tr:
        T = d / v_max + tr
        rf = tr / T
    else:
        v = math.sqrt(2 * a_max * d / math.pi)
        tr = v * math.pi / (2 * a_max)
        T, rf = 2 * tr, 0.5
    if T < t_min:
        return t_min, 0.5
    return T, rf


def move_duration(distance: float, v_max: float, a_max: float, t_min: float = 0.5) -> float:
    return move_profile(distance, v_max, a_max, t_min)[0]


def smooth_trapezoid(u: np.ndarray, rf: float) -> np.ndarray:
    """Normalised position of a smooth-trapezoid move (u, s in [0, 1])."""
    u = np.clip(np.asarray(u, dtype=float), 0.0, 1.0)
    rf = min(0.5, max(1e-3, rf))
    vn = 1.0 / (1.0 - rf)

    def acc_part(x):
        return vn * (x / 2 - rf / (2 * math.pi) * np.sin(math.pi * x / rf))

    s = np.where(u < rf, acc_part(u), vn * (rf / 2 + (u - rf)))
    s = np.where(u > 1 - rf, 1 - acc_part(1 - u), s)
    return s


def zv_shape(t: np.ndarray, x: np.ndarray, period: float, zeta: float = 0.0) -> np.ndarray:
    """Convolve a sampled command x(t) with a two-impulse ZV shaper for a pendulum of
    the given natural period.  Output is delayed by half a damped period."""
    if period <= 0:
        return x
    wd = 2 * math.pi / period
    K = math.exp(-zeta * math.pi / math.sqrt(max(1e-9, 1 - zeta * zeta)))
    a1, a2 = 1 / (1 + K), K / (1 + K)
    delay = math.pi / wd
    xd = np.empty_like(x)
    for k in range(x.shape[1] if x.ndim > 1 else 1):
        col = x[:, k] if x.ndim > 1 else x
        shifted = np.interp(t - delay, t, col, left=col[0], right=col[-1])
        if x.ndim > 1:
            xd[:, k] = a1 * col + a2 * shifted
        else:
            xd[:] = a1 * col + a2 * shifted
    return xd


def simulate_pendulum(t: np.ndarray, pivot_xy: np.ndarray, rope_len: np.ndarray, zeta: np.ndarray | float = 0.02,
                      theta0=(0.0, 0.0), omega0=(0.0, 0.0)) -> np.ndarray:
    """Integrate small-but-nonlinear sway angles (theta_x, theta_y) of a load hanging
    under a moving pivot.  Returns horizontal load offset (N, 2) relative to the pivot
    (i.e. rope_len * sin(theta)).  RK4 on a fine internal step."""
    n = len(t)
    if n < 2:
        return np.zeros((n, 2))
    dt_out = float(t[1] - t[0])
    sub = max(1, int(math.ceil(dt_out / (1 / 400.0))))
    h = dt_out / sub
    # pivot acceleration by central differences
    acc = np.gradient(np.gradient(pivot_xy, t, axis=0), t, axis=0)
    dL = np.gradient(rope_len, t)
    z = np.broadcast_to(np.asarray(zeta, dtype=float), (n,))
    th = np.array(theta0, dtype=float)
    om = np.array(omega0, dtype=float)
    out = np.zeros((n, 2))

    def deriv(th_, om_, a_, L_, Ld_, z_):
        w = math.sqrt(G / max(L_, 0.3))
        dom = (-(G / L_) * np.sin(th_) - (2 * Ld_ / L_) * om_ - 2 * z_ * w * om_ - a_ * np.cos(th_) / L_)
        return om_, dom

    for i in range(n):
        L = max(0.3, rope_len[i])
        out[i] = L * np.sin(th)
        if i == n - 1:
            break
        for s in range(sub):
            f = s / sub
            a = acc[i] * (1 - f) + acc[i + 1] * f
            Lc = max(0.3, rope_len[i] * (1 - f) + rope_len[i + 1] * f)
            Ld = dL[i] * (1 - f) + dL[i + 1] * f
            zc = z[i] * (1 - f) + z[i + 1] * f
            k1t, k1o = deriv(th, om, a, Lc, Ld, zc)
            k2t, k2o = deriv(th + h / 2 * k1t, om + h / 2 * k1o, a, Lc, Ld, zc)
            k3t, k3o = deriv(th + h / 2 * k2t, om + h / 2 * k2o, a, Lc, Ld, zc)
            k4t, k4o = deriv(th + h * k3t, om + h * k3o, a, Lc, Ld, zc)
            th = th + h / 6 * (k1t + 2 * k2t + 2 * k3t + k4t)
            om = om + h / 6 * (k1o + 2 * k2o + 2 * k3o + k4o)
    return out


def pendulum_period(L: float) -> float:
    return 2 * math.pi * math.sqrt(max(L, 0.3) / G)


# ----------------------------------------------------------------- rotations
def quat_from_euler_zyx(yaw: float, pitch: float = 0.0, roll: float = 0.0) -> np.ndarray:
    """Quaternion (w,x,y,z) for R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    return np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ])


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_rotate(q: np.ndarray, v) -> np.ndarray:
    w, x, y, z = q
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    return R @ np.asarray(v, dtype=float)


def slerp(q0: np.ndarray, q1: np.ndarray, s: float) -> np.ndarray:
    d = float(np.dot(q0, q1))
    if d < 0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + s * (q1 - q0)
        return q / np.linalg.norm(q)
    th = math.acos(d)
    return (math.sin((1 - s) * th) * q0 + math.sin(s * th) * q1) / math.sin(th)


def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


# ----------------------------------------------------------------- smoothing
def critically_damped(x: np.ndarray, dt: float, omega: float) -> np.ndarray:
    """Second-order critically damped follower (camera rigs, attention points)."""
    y = np.array(x[0], dtype=float)
    v = np.zeros_like(y)
    out = np.empty_like(x, dtype=float)
    for i in range(len(x)):
        # exact discretisation of x'' = w^2 (target - x) - 2 w x'
        a = omega * omega * (x[i] - y) - 2 * omega * v
        v = v + a * dt
        y = y + v * dt
        out[i] = y
    return out
