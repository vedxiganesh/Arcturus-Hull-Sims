"""Trim/heave equilibrium: a Newton solve of the captive residuals, one chain per speed.

Unknowns x = (theta_deg, z_m); residuals r = (R_lift_N, R_pitch_Nm) from
collect.py. Each CFD point costs hours, so every decision reuses ALL completed
points of the chain's speed:

- Jacobian: a weighted affine fit of r over the usable points near the best
  one. Weights are 1/SE^2 times a proximity factor 1/(1 + (d/rho)^2). Distances
  are in SCALED units, x / fd_step, so one unit is one stencil step in each
  direction. Needs >= 3 points with spread in both directions. Otherwise the
  next batch is a STENCIL (best, best + h*dtheta, best + h*dz).
- Step: dx = -J^-1 r(best), clipped to the trust radius rho, then to the
  envelope, then rounded to case_id resolution.
- Trust region: if the last Newton target improved the merit, rho doubles
  (up to rho_max); if not, rho halves and the next batch re-stencils around the
  best point.
- Converged: a SIMULATED point meets both tolerances and passed the quality
  check (complete run, window drift within tolerance).

Terminal states: CONVERGED, MAX_ITERS, OUTSIDE_ENVELOPE (the step hit the
same envelope bound on two consecutive steps), STALLED (the Newton step
rounds to the best point itself), FAILED.

Pure and closed-form (Cramer's rule): np.linalg crashes the local env
(see common.py).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from common import quantize_point

CONVERGED = "CONVERGED"
MAX_ITERS = "MAX_ITERS"
OUTSIDE_ENVELOPE = "OUTSIDE_ENVELOPE"
STALLED = "STALLED"
FAILED = "FAILED"
TERMINAL = (CONVERGED, MAX_ITERS, OUTSIDE_ENVELOPE, STALLED, FAILED)


@dataclass(frozen=True)
class Point:
    theta: float
    z: float
    r: tuple[float, float]  # (R_lift_N, R_pitch_Nm)
    se: tuple[float, float]  # standard errors of r (nan if unknown)
    quality_ok: bool  # complete + drift within tolerance
    case_id: str = ""

    @property
    def usable(self) -> bool:
        return all(math.isfinite(v) for v in self.r)

    @property
    def x(self) -> tuple[float, float]:
        return (self.theta, self.z)


@dataclass(frozen=True)
class Config:
    tol: tuple[float, float]  # (tol_lift_N, tol_pitch_Nm)
    fd: tuple[float, float]  # (fd_theta_deg, fd_z_m): stencil steps = scaled unit
    envelope: tuple[tuple[float, float], tuple[float, float]]  # ((th_lo, th_hi), (z_lo, z_hi))
    max_iters: int = 6
    rho0: float = 2.0
    rho_min: float = 0.25
    rho_max: float = 4.0
    min_spread: float = 0.2  # scaled RMS spread required along the weaker direction


@dataclass
class ChainState:
    """What a chain carries between decisions (stored in ledger.json)."""

    x0: tuple[float, float]
    iteration: int = 0  # batches submitted so far
    rho: float = 2.0
    last_step: dict | None = None  # {"from_merit": m, "target": [theta, z]}
    last_bound: str | None = None  # envelope bound the previous step was clipped to

    def to_dict(self) -> dict:
        return {"x0": list(self.x0), "iteration": self.iteration, "rho": self.rho,
                "last_step": self.last_step, "last_bound": self.last_bound}

    @classmethod
    def from_dict(cls, d: dict) -> "ChainState":
        return cls(x0=tuple(d["x0"]), iteration=int(d.get("iteration", 0)), rho=float(d.get("rho", 2.0)),
                   last_step=d.get("last_step"), last_bound=d.get("last_bound"))


@dataclass
class Decision:
    kind: str  # "stencil", "step", or a TERMINAL state
    points: list[tuple[float, float]] = field(default_factory=list)  # quantized, new or reused
    best: Point | None = None
    J: list[list[float]] | None = None  # physical: [[dRl/dth, dRl/dz], [dRp/dth, dRp/dz]]
    rho: float = 0.0
    note: str = ""
    state: ChainState | None = None  # updated state to store if the decision is applied

    @property
    def terminal(self) -> bool:
        return self.kind in TERMINAL


# ---------------------------------------------------------------------------
# Small closed-form linear algebra
# ---------------------------------------------------------------------------


def det3(m) -> float:
    return (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))


def solve3(a, b) -> list[float] | None:
    """Cramer's rule; None if singular."""
    d = det3(a)
    scale = max(abs(v) for row in a for v in row) or 1.0
    if abs(d) <= 1e-12 * scale ** 3:
        return None
    out = []
    for k in range(3):
        m = [list(row) for row in a]
        for i in range(3):
            m[i][k] = b[i]
        out.append(det3(m) / d)
    return out


def solve2(j, r) -> tuple[float, float] | None:
    d = j[0][0] * j[1][1] - j[0][1] * j[1][0]
    scale = max(abs(v) for row in j for v in row) or 1.0
    if abs(d) <= 1e-9 * scale ** 2:
        return None
    return ((j[1][1] * r[0] - j[0][1] * r[1]) / d, (-j[1][0] * r[0] + j[0][0] * r[1]) / d)


def weakest_spread(ds: list[tuple[float, float]]) -> float:
    """RMS spread of 2-D points along their weakest direction (sqrt of min eigenvalue)."""
    n = len(ds)
    if n < 2:
        return 0.0
    mx = sum(p[0] for p in ds) / n
    my = sum(p[1] for p in ds) / n
    sxx = sum((p[0] - mx) ** 2 for p in ds) / n
    syy = sum((p[1] - my) ** 2 for p in ds) / n
    sxy = sum((p[0] - mx) * (p[1] - my) for p in ds) / n
    tr, det = sxx + syy, sxx * syy - sxy * sxy
    lam_min = tr / 2.0 - math.sqrt(max(tr * tr / 4.0 - det, 0.0))
    return math.sqrt(max(lam_min, 0.0))


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def merit(p: Point, cfg: Config) -> float:
    return math.hypot(p.r[0] / cfg.tol[0], p.r[1] / cfg.tol[1])


def meets_tolerance(p: Point, cfg: Config) -> bool:
    return p.usable and abs(p.r[0]) <= cfg.tol[0] and abs(p.r[1]) <= cfg.tol[1]


def scaled(x, center, cfg: Config) -> tuple[float, float]:
    return ((x[0] - center[0]) / cfg.fd[0], (x[1] - center[1]) / cfg.fd[1])


def fit_jacobian(points: list[Point], center: tuple[float, float], rho: float, cfg: Config):
    """Weighted affine fit of r about `center`. Returns (J_scaled, n_used) or None.

    Uses the points within 2*rho (scaled) of the center. Each residual
    component is fitted separately, weighted by 1/SE^2 (floored at 5% of the
    tolerance, and the tolerance itself when SE is unknown) times proximity.
    """
    near = []
    for p in points:
        d = scaled(p.x, center, cfg)
        if math.hypot(*d) <= 2.0 * rho + 1e-9:
            near.append((p, d))
    if len(near) < 3 or weakest_spread([d for _, d in near]) < cfg.min_spread:
        return None
    rows = []
    for k in range(2):
        a = [[0.0] * 3 for _ in range(3)]
        b = [0.0] * 3
        for p, d in near:
            se = p.se[k]
            se = cfg.tol[k] if not math.isfinite(se) else max(se, 0.05 * cfg.tol[k])
            w = 1.0 / (1.0 + (math.hypot(*d) / rho) ** 2) / (se * se)
            phi = (1.0, d[0], d[1])
            for i in range(3):
                b[i] += w * phi[i] * p.r[k]
                for j in range(3):
                    a[i][j] += w * phi[i] * phi[j]
        coef = solve3(a, b)
        if coef is None:
            return None
        rows.append([coef[1], coef[2]])
    return rows, len(near)


def to_physical(js, cfg: Config) -> list[list[float]]:
    return [[js[k][0] / cfg.fd[0], js[k][1] / cfg.fd[1]] for k in range(2)]


# ---------------------------------------------------------------------------
# Envelope and stencil
# ---------------------------------------------------------------------------


def clip_to_envelope(x, cfg: Config) -> tuple[tuple[float, float], str | None]:
    (tl, th), (zl, zh) = cfg.envelope
    t, z = x
    bound = None
    if t < tl:
        t, bound = tl, "theta_lo"
    elif t > th:
        t, bound = th, "theta_hi"
    if z < zl:
        z, bound = zl, "z_lo"
    elif z > zh:
        z, bound = zh, "z_hi"
    return (t, z), bound


def stencil(center, h: float, cfg: Config) -> list[tuple[float, float]]:
    """center, center + h*fd_theta, center + h*fd_z; a step that leaves the envelope flips sign."""
    (tl, th), (zl, zh) = cfg.envelope
    c = quantize_point(*center)
    dt, dz = h * cfg.fd[0], h * cfg.fd[1]
    t1 = c[0] + dt if c[0] + dt <= th else c[0] - dt
    z1 = c[1] + dz if c[1] + dz <= zh else c[1] - dz
    if t1 < tl or z1 < zl:
        raise ValueError(f"envelope too small for a stencil of {h} steps around {c}")
    return [c, quantize_point(t1, c[1]), quantize_point(c[0], z1)]


def _key(x) -> tuple[float, float]:
    return quantize_point(*x)


# ---------------------------------------------------------------------------
# Decision
# ---------------------------------------------------------------------------


def decide(points: list[Point], state: ChainState, cfg: Config) -> Decision:
    """Next action for one chain, given every completed point at its speed."""
    st = ChainState(**{**state.__dict__})
    usable = [p for p in points if p.usable]
    known = {_key(p.x) for p in points}

    if not usable:
        if st.iteration == 0:
            pts = stencil(st.x0, 1.0, cfg)
            st.iteration += 1
            return Decision("stencil", pts, rho=st.rho, note="initial stencil at x0", state=st)
        return Decision(FAILED, note="no usable points", state=st)

    best = min(usable, key=lambda p: merit(p, cfg))
    if meets_tolerance(best, cfg) and best.quality_ok:
        return Decision(CONVERGED, best=best, rho=st.rho, state=st)
    if st.iteration >= cfg.max_iters:
        return Decision(MAX_ITERS, best=best, rho=st.rho, note=f"{st.iteration} batches", state=st)

    note = []
    if meets_tolerance(best, cfg):
        note.append(f"best {best.case_id} meets tolerance but failed the quality check")
    force_stencil = False
    if st.last_step:
        tgt = _key(st.last_step["target"])
        hit = next((p for p in usable if _key(p.x) == tgt), None)
        if hit is not None and merit(hit, cfg) < st.last_step["from_merit"]:
            st.rho = min(2.0 * st.rho, cfg.rho_max)
            note.append(f"step accepted, rho -> {st.rho:g}")
        else:
            st.rho = max(st.rho / 2.0, cfg.rho_min)
            force_stencil = True
            note.append(f"step rejected, rho -> {st.rho:g}, re-stencil")
        st.last_step = None

    fit = None if force_stencil else fit_jacobian(usable, best.x, st.rho, cfg)
    if fit is None and not force_stencil:
        note.append("too few well-spread points near the best for a Jacobian")
    js = fit[0] if fit else None
    dxs = solve2(js, best.r) if js else None
    if js and dxs is None:
        note.append("singular Jacobian")

    if dxs is None:
        h = min(1.0, st.rho)
        pts = stencil(best.x, h, cfg)
        if all(_key(p) in known for p in pts):
            # Every stencil point exists already, yet no fit: widen to all points.
            fit = fit_jacobian(usable, best.x, 1e6, cfg)
            dxs = solve2(fit[0], best.r) if fit else None
            if dxs is None:
                return Decision(FAILED, best=best, note="cannot build a Jacobian from the points", state=st)
            js = fit[0]
        else:
            st.iteration += 1
            return Decision("stencil", pts, best=best, rho=st.rho,
                            J=to_physical(js, cfg) if js else None,
                            note="; ".join(note + [f"stencil h={h:g} around best"]), state=st)

    ds = (-dxs[0], -dxs[1])
    n = math.hypot(*ds)
    if n > st.rho:
        ds = (ds[0] * st.rho / n, ds[1] * st.rho / n)
        note.append(f"step clipped to rho={st.rho:g} (Newton length {n:.2f})")
    target = (best.theta + ds[0] * cfg.fd[0], best.z + ds[1] * cfg.fd[1])
    target, bound = clip_to_envelope(target, cfg)
    if bound is not None:
        note.append(f"clipped to envelope {bound}")
        if st.last_bound == bound:
            st.last_bound = bound
            return Decision(OUTSIDE_ENVELOPE, best=best, J=to_physical(js, cfg), rho=st.rho,
                            note="; ".join(note + ["root appears to lie outside the envelope"]), state=st)
    st.last_bound = bound
    target = _key(target)
    if target == _key(best.x):
        return Decision(STALLED, best=best, J=to_physical(js, cfg), rho=st.rho,
                        note="; ".join(note + ["Newton step is below case-id resolution"]), state=st)
    st.last_step = {"from_merit": merit(best, cfg), "target": list(target)}
    st.iteration += 1
    return Decision("step", [target], best=best, J=to_physical(js, cfg), rho=st.rho,
                    note="; ".join(note), state=st)
