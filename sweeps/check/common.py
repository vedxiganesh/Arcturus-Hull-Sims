"""Pure helpers shared by every hullsweep stage. No Fluent, no PyFluent.

Conventions (used everywhere, stated once):

- theta_deg : trim angle, POSITIVE = BOW UP.
- z_m       : heave, POSITIVE = UP, the CG's vertical displacement from cg_ref.
- The hull is rotated about the pitch axis through cg_ref FIRST, then
  translated by (0, 0, z). The moment centre of every sweep report is
  therefore cg_ref + (0, 0, z), i.e. the displaced CG.
- Fluent's rotate-zone uses a right-hand rotation about a GLOBAL axis. The
  bow-up axis is bow x up; with the bow at -Y and up +Z that is -X, so a
  bow-up trim of theta is a Fluent rotation of -theta about +X. This is
  derived from the topology file, never hard-coded.

Numerics: closed-form only. np.linalg (lstsq/svd/solve) routes through
LAPACK, which hard-crashes the reefs-mobo env (see the vendored
_ols_slope note).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

#: Fluent pitch axis. Always +X here: the symmetry plane is x = 0, and only a
#: rotation about X keeps it invariant.
PITCH_AXIS = (1.0, 0.0, 0.0)

#: Prefix for every report definition this pipeline creates. collect.py keys
#: off it, so a template's pre-existing reports never pollute the results.
REPORT_PREFIX = "sw"
#: Name of the single report file that carries every sweep report column.
REPORT_FILE = "sweep-forces"


# ---------------------------------------------------------------------------
# Topology
# ---------------------------------------------------------------------------


@dataclass
class Topology:
    path: Path
    raw: dict

    name: str = ""
    cg_ref: np.ndarray = field(default_factory=lambda: np.zeros(3))
    bow: np.ndarray = field(default_factory=lambda: np.array([0.0, -1.0, 0.0]))
    up: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))

    @classmethod
    def load(cls, path: str | Path) -> "Topology":
        path = Path(path).resolve()
        raw = json.loads(path.read_text())
        t = cls(path=path, raw=raw)
        t.name = raw["name"]
        t.cg_ref = np.asarray(raw["cg_ref"], dtype=float)
        t.bow = _unit(raw["bow_direction"])
        t.up = _unit(raw["up_direction"])
        if abs(float(np.dot(t.bow, t.up))) > 1e-9:
            raise ValueError(f"{path.name}: bow_direction must be perpendicular to up_direction")
        return t

    def __getitem__(self, key):
        return self.raw[key]

    def get(self, key, default=None):
        return self.raw.get(key, default)

    def resolve_path(self, key: str) -> Path:
        """A path field, resolved relative to the hullsweep package root."""
        p = Path(self.raw[key])
        return p if p.is_absolute() else (HERE / p).resolve()

    def save_measured(self, **values) -> None:
        """Write measured fields (hull_zmin_at_ref, hull_length_m, ...) back."""
        self.raw.update(values)
        self.path.write_text(json.dumps(self.raw, indent=2) + "\n")

    # --- signs -------------------------------------------------------------

    @property
    def fluent_pitch_sign(self) -> float:
        """Fluent rotation angle per degree of bow-up trim about +X."""
        bow_up_axis = np.cross(self.bow, self.up)
        s = float(np.dot(bow_up_axis, PITCH_AXIS))
        if abs(abs(s) - 1.0) > 1e-9:
            raise ValueError("bow x up must be parallel to the X pitch axis")
        return s

    # --- physics -----------------------------------------------------------

    @property
    def domain_fraction(self) -> float:
        return 0.5 if self.raw.get("half_domain", False) else 1.0

    @property
    def weight_domain_N(self) -> float:
        """Weight carried by the SIMULATED domain (half the boat if half_domain)."""
        return self.raw["mass_full_kg"] * self.raw["g"] * self.domain_fraction

    @property
    def thrust_ceiling_domain_N(self) -> float:
        return self.raw["thrust_ceiling_full_N"] * self.domain_fraction

    def thrust_offset_body(self) -> float:
        """Body-frame 'up' offset of the thrust line from the CG (negative = below).

        The thrust line is thrust_offset_below_keel_m below the lowest hull
        point at theta=0, and rotates with the hull.
        """
        zmin = self.raw.get("hull_zmin_at_ref")
        if zmin is None:
            raise ValueError(
                f"{self.path.name}: hull_zmin_at_ref not measured yet; "
                "run `prepare_case.py template` first."
            )
        up_ref = float(np.dot(self.cg_ref, self.up))
        return (float(zmin) - self.raw["thrust_offset_below_keel_m"]) - up_ref


def _unit(v) -> np.ndarray:
    a = np.asarray(v, dtype=float)
    n = math.sqrt(float(np.dot(a, a)))
    if n == 0.0:
        raise ValueError("zero vector")
    return a / n


# ---------------------------------------------------------------------------
# Rigid transform
# ---------------------------------------------------------------------------


def rotation_about_axis(axis, angle_deg: float) -> np.ndarray:
    """Right-hand rotation matrix (Rodrigues), closed form."""
    k = _unit(axis)
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    kx, ky, kz = k
    K = np.array([[0.0, -kz, ky], [kz, 0.0, -kx], [-ky, kx, 0.0]])
    return np.eye(3) * c + s * K + (1.0 - c) * np.outer(k, k)


@dataclass(frozen=True)
class CaseTransform:
    """The rigid motion applied to the foreground zones for one case."""

    theta_deg: float
    z_m: float
    fluent_angle_deg: float
    origin: tuple[float, float, float]
    axis: tuple[float, float, float]
    translation: tuple[float, float, float]
    moment_center: tuple[float, float, float]

    def apply(self, pts: np.ndarray) -> np.ndarray:
        """Map template-frame points to case-frame points (rotate, then translate)."""
        R = rotation_about_axis(self.axis, self.fluent_angle_deg)
        o = np.asarray(self.origin)
        p = np.atleast_2d(np.asarray(pts, dtype=float))
        return (p - o) @ R.T + o + np.asarray(self.translation)


def case_transform(topo: Topology, theta_deg: float, z_m: float) -> CaseTransform:
    origin = tuple(float(v) for v in topo.cg_ref)
    translation = tuple(float(v) for v in (topo.up * z_m))
    mc = tuple(float(v) for v in (topo.cg_ref + topo.up * z_m))
    return CaseTransform(
        theta_deg=float(theta_deg),
        z_m=float(z_m),
        fluent_angle_deg=float(topo.fluent_pitch_sign * theta_deg),
        origin=origin,
        axis=PITCH_AXIS,
        translation=translation,
        moment_center=mc,
    )


# ---------------------------------------------------------------------------
# Case identity and run planning
# ---------------------------------------------------------------------------


def case_id(topo_name: str, speed: float, theta_deg: float, z_m: float) -> str:
    """Deterministic, filesystem- and Fluent-safe case id."""
    tag = f"{topo_name}_V{speed:.2f}_t{theta_deg:+.2f}_z{z_m * 1000.0:+.1f}mm"
    return re.sub(r"[^A-Za-z0-9_+\-]", "p", tag.replace(".", "p"))


@dataclass(frozen=True)
class RunPlan:
    dt_s: float
    steps: int
    settle_time_s: float
    end_time_s: float


def plan_run(topo: Topology, speed: float) -> RunPlan:
    """Constant convective Courant number, anchored on the validated run.

    dt = ref_dt * ref_speed / V keeps U*dt/dx fixed at the value of the
    8-hour run that worked (dt = 5e-3 s at 2.5 m/s). Duration is counted in
    hull lengths travelled: settle, then average.
    """
    if speed <= 0.0:
        raise ValueError("speed must be > 0 (a V=0 hydrostatic case needs an explicit dt)")
    L = topo.get("hull_length_m")
    if L is None:
        raise ValueError(f"{topo.path.name}: hull_length_m not measured yet")
    run = topo["run"]
    dt = topo["ref_dt_s"] * topo["ref_speed_mps"] / speed
    settle = run["settle_hull_lengths"] * L / speed
    end = settle + run["average_hull_lengths"] * L / speed
    return RunPlan(dt_s=dt, steps=int(math.ceil(end / dt)), settle_time_s=settle, end_time_s=end)


def check_envelope(topo: Topology, theta_deg: float, z_m: float) -> list[str]:
    env = topo["envelope"]
    msgs = []
    lo, hi = env["theta_deg"]
    if not lo <= theta_deg <= hi:
        msgs.append(f"theta {theta_deg} outside envelope [{lo}, {hi}]")
    lo, hi = env["z_m"]
    if not lo <= z_m <= hi:
        msgs.append(f"z {z_m} outside envelope [{lo}, {hi}]")
    return msgs


def file_sha1(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Sweep report layout (shared by prepare_case.py and collect.py)
# ---------------------------------------------------------------------------

COMPONENTS = ("x", "y", "z")


def report_groups(topo: Topology) -> dict[str, list[str]]:
    """Report group name -> wall zones. 'total' plus one group per hull zone."""
    zones = list(topo["hull_wall_zones"])
    groups = {"total": zones}
    for z in zones:
        groups[z.removeprefix("wall_")] = [z]
    return groups


def report_name(kind: str, comp: str, group: str) -> str:
    """kind 'f' (force) or 'm' (moment about the displaced CG)."""
    return f"{REPORT_PREFIX}-{kind}{comp}-{group}"
