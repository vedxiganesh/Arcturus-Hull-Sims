"""`hs topology init`: write a topology.json from the case itself.

Every field the case holds is read from it (case_reader.py, offline); the
design inputs it cannot hold (mass, CG, thrust, envelope) must be given on
the command line. Nothing is copied from another topology: that is how a
topology once kept another's name, mass, length and hull centroid.

Before writing anything it checks:
- the frame. The pipeline assumes bow -Y, up +Z, and for a half domain a
  symmetry plane at x = 0 (common.PITCH_AXIS; collect.py reads drag as Fy).
  A case in another frame is refused, with the whole-mesh rotation that fixes it.
- the envelope against the overset region and the bottom, at its four corners.
- hydrostatics at theta = 0, z = 0. The heave that floats the hull says whether
  the mass and free surface agree, and whether that heave is inside the envelope.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from case_reader import CaseFile, CaseReadError

#: Validated reference run (README "Run sizing"): dt = ref_dt * ref_speed / V.
REF_SPEED_MPS, REF_DT_S = 2.5, 0.005
RUN_DEFAULTS = {"settle_hull_lengths": 6.0, "average_hull_lengths": 4.0, "max_iter_per_step": 20}

_INLETS = ("pressure-inlet", "velocity-inlet", "mass-flow-inlet")
_PLANAR_TOL_M = 1e-6
_TOUCH_TOL_M = 1e-4
_MARGIN_WARN_M = 0.03


@dataclass
class Derived:
    raw: dict
    source: dict[str, str] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)   # refuse to write
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    hull_vertices: np.ndarray | None = None
    overset_box: tuple[np.ndarray, np.ndarray] | None = None
    symmetry_axis: int | None = None


def snap_axis(v: np.ndarray, what: str) -> np.ndarray:
    """The coordinate axis (signed unit vector) v points along; raises if it is oblique."""
    v = np.asarray(v, dtype=float)
    u = v / np.linalg.norm(v)
    i = int(np.argmax(np.abs(u)))
    if abs(u[i]) < 0.99:
        raise CaseReadError(f"{what} {np.round(u, 4).tolist()} is not along a coordinate axis")
    out = np.zeros(3)
    out[i] = math.copysign(1.0, u[i])
    return out


def rotation_to_bow_minus_y(bow: np.ndarray) -> float:
    """Angle (deg, right-hand about +Z) that turns a horizontal bow direction into -Y."""
    return math.degrees(math.atan2(-bow[0], -bow[1]))


def frame_problems(bow, up, symmetry_axis, symmetry_coord) -> list[str]:
    out = []
    if not np.allclose(up, [0, 0, 1]):
        out.append(f"up is {up.tolist()}, the pipeline needs +Z (gravity along -Z)")
        return out
    if not np.allclose(bow, [0, -1, 0]):
        ang = rotation_to_bow_minus_y(bow)
        out.append(f"bow is {bow.tolist()}, the pipeline needs [0, -1, 0] (pitch about X, drag = Fy). "
                   f"Fix: rotate the WHOLE mesh {ang:+.0f} deg about +Z through the origin "
                   f"(Fluent: Domain > Mesh > Transform > Rotate) and save the case again")
    if symmetry_axis is not None:
        # Under that rotation the symmetry normal follows the bow, so judge the rotated frame.
        normal = np.zeros(3)
        normal[symmetry_axis] = 1.0
        if not np.allclose(bow, [0, -1, 0]):
            a = math.radians(rotation_to_bow_minus_y(bow))
            R = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
            normal = np.abs(R @ normal)
        if not np.allclose(normal, [1, 0, 0]):
            out.append(f"the symmetry plane is normal to {'xyz'[symmetry_axis]}; pitch about X needs it "
                       "normal to the beam (X once the bow is -Y)")
        elif abs(symmetry_coord) > _PLANAR_TOL_M:
            out.append(f"the symmetry plane is at {'xyz'[symmetry_axis]} = {symmetry_coord:.6g}; "
                       "translate the mesh so it is at 0")
    return out


def derive(case: CaseFile, name: str) -> Derived:
    """Every topology field the case holds, with where each came from."""
    d = Derived(raw={})
    raw, src = d.raw, d.source

    # --- zones -------------------------------------------------------------
    bg_ids, comp_ids = case.overset_grids()
    if len(bg_ids) != 1 or not comp_ids:
        raise CaseReadError(f"overset interfaces: background {bg_ids}, components {comp_ids}; "
                            "need exactly one background and at least one component grid")
    bg = case.zone_by_id(bg_ids[0])
    comps = [case.zone_by_id(i) for i in comp_ids]
    others = [z for z in case.zones.values() if z.cells and z.id != bg.id and z.id not in comp_ids]
    raw["background_cell_zone"] = bg.name
    src["background_cell_zone"] = "overset interface bg-grids"
    raw["foreground_cell_zones"] = [z.name for z in comps] + sorted(z.name for z in others)
    src["foreground_cell_zones"] = "overset comp-grids" + (
        f" + {', '.join(f'{z.name} ({z.kind})' for z in others)} (every non-background cell zone moves)"
        if others else "")

    comp_names = {z.name for z in comps}
    # Largest first: run_case reads the live 6DOF state from hull_wall_zones[0].
    walls = [z.name for z in sorted(case.of_kind("wall", cells=False), key=lambda z: z.lo - z.hi)
             if case.c0_cell_zone(z.name).name in comp_names]
    if not walls:
        raise CaseReadError("no wall zone borders an overset component zone; cannot identify the hull")
    raw["hull_wall_zones"] = walls
    src["hull_wall_zones"] = ("walls bordering the component fluid, largest first (shadow walls of "
                              "solids excluded)")

    inlets, outlets = case.of_kind(*_INLETS, cells=False), case.of_kind("pressure-outlet", cells=False)
    if len(inlets) != 1 or len(outlets) != 1:
        raise CaseReadError(f"need one inlet and one pressure-outlet; found {[z.name for z in inlets]}, "
                            f"{[z.name for z in outlets]}")
    inlet, outlet = inlets[0].name, outlets[0].name
    raw["inlet_zone"], raw["outlet_zone"] = inlet, outlet
    src["inlet_zone"] = src["outlet_zone"] = "boundary type"

    # --- physics -----------------------------------------------------------
    grav = case.gravity()
    if grav is None or not np.any(grav):
        raise CaseReadError("gravity is off in the case")
    raw["g"] = float(np.linalg.norm(grav))
    up = snap_axis(-grav, "-gravity")
    raw["up_direction"] = up.tolist()
    src["g"] = src["up_direction"] = f"gravity {grav.tolist()}"

    phases = case.phases()
    dens = {p: case.material_density(m) for p, m in phases}
    liquid = [p for p, rho in dens.items() if rho is not None and rho > 500.0]
    if len(liquid) != 1:
        raise CaseReadError(f"cannot pick the water phase from {phases} (densities {dens})")
    raw["water_phase"] = liquid[0]
    rho_water = float(dens[liquid[0]])
    src["water_phase"] = f"{dict(phases)[liquid[0]]}, density {rho_water:g}"
    d.notes.append(f"water density {rho_water:g} kg/m3 (used for the hydrostatic check)")
    raw["_rho_water"] = rho_water  # dropped before writing

    if case.bc_flag(inlet, "open-channel?")[:1] != ["#t"]:
        d.problems.append(f"{inlet} is not an open-channel inlet; the flat initialization needs one")
    fs = set(case.bc_constants(inlet, "ht-local")) | set(case.bc_constants(outlet, "ht-local"))
    if not fs:
        raise CaseReadError(f"no constant free-surface level (ht-local) on {inlet}/{outlet}")
    if max(fs) - min(fs) > 1e-6:
        d.warnings.append(f"inlet/outlet free-surface levels differ: {sorted(fs)}; using the inlet's")
    raw["free_surface_z"] = case.bc_constants(inlet, "ht-local")[0]
    src["free_surface_z"] = f"open-channel free surface level on {inlet}"

    io = np.concatenate([case.vertices(inlet), case.vertices(outlet)])
    raw["bottom_z"] = float(io[:, 2].min())
    src["bottom_z"] = "lowest inlet/outlet vertex (the domain floor)"
    for z in (inlet, outlet):
        for hb in sorted(set(case.bc_constants(z, "ht-bottom"))):
            if abs(hb - raw["bottom_z"]) > 5e-3:
                d.warnings.append(f"{z} open-channel bottom level {hb:g} is not the mesh floor "
                                  f"{raw['bottom_z']:.4g}; the solver's hydrostatics use {hb:g}")

    vmag = sorted(set(case.bc_constants(inlet, "vmag")))
    if vmag:
        d.notes.append(f"case inlet speed {vmag} m/s (each case sets its own; dt is anchored on "
                       f"{REF_SPEED_MPS} m/s / {REF_DT_S} s)")

    # --- directions --------------------------------------------------------
    upstream = case.vertices(inlet).mean(axis=0) - case.vertices(outlet).mean(axis=0)
    bow = snap_axis(upstream - np.dot(upstream, up) * up, "outlet -> inlet direction")
    raw["bow_direction"] = bow.tolist()
    src["bow_direction"] = "outlet -> inlet (the bow faces the oncoming flow)"

    # --- hull geometry -----------------------------------------------------
    v = np.concatenate([case.vertices(w) for w in walls])
    d.hull_vertices = v
    lo, hi = v.min(axis=0), v.max(axis=0)
    stats = {"centroid": v.mean(axis=0).tolist(), "min": lo.tolist(), "max": hi.tolist(),
             "zmin": float(lo[2]), "length": float(np.dot(hi - lo, np.abs(bow))), "n_vertices": int(len(v))}
    raw["hull_zmin_at_ref"], raw["hull_length_m"] = stats["zmin"], stats["length"]
    raw["template_hull_stats"] = stats
    src["hull_zmin_at_ref"] = src["hull_length_m"] = src["template_hull_stats"] = \
        "hull wall vertices (as fluent_ops.hull_stats)"

    # --- half domain -------------------------------------------------------
    planes = []
    for z in case.of_kind("symmetry", cells=False):
        sv = case.vertices(z.name)
        spread = sv.max(axis=0) - sv.min(axis=0)
        ax = int(np.argmin(spread))
        if spread[ax] > _PLANAR_TOL_M:
            continue
        c = float(sv[0, ax])
        if min(abs(lo[ax] - c), abs(hi[ax] - c)) < _TOUCH_TOL_M:
            planes.append((z.name, ax, c))
    if len({(ax, round(c, 6)) for _, ax, c in planes}) > 1:
        raise CaseReadError(f"the hull touches several symmetry planes: {planes}")
    raw["half_domain"] = bool(planes)
    if planes:
        d.symmetry_axis, sym_c = planes[0][1], planes[0][2]
        src["half_domain"] = (f"hull touches symmetry plane {'xyz'[d.symmetry_axis]} = {sym_c:.3g} "
                              f"({', '.join(p[0] for p in planes)})")
    else:
        sym_c = 0.0
        src["half_domain"] = "no symmetry plane touches the hull"
    d.problems += frame_problems(bow, up, d.symmetry_axis, sym_c)

    ov = case.of_kind("overset", cells=False)
    if ov:
        ovv = np.concatenate([case.vertices(z.name) for z in ov])
        d.overset_box = (ovv.min(axis=0), ovv.max(axis=0))

    origins = case.sixdof_origins()
    if origins:
        d.notes.append(f"6DOF origin(s) in the case, a hint for --cg only: {origins}")
    blockers = case.template_blockers()
    raw["_template_ready"] = not blockers  # dropped before writing
    if blockers:
        d.notes.append("not a template yet (" + "; ".join(blockers) + "): name it <topo>_source.cas.h5 "
                       "and run prepare_case.py template")
    raw["name"] = name
    return d


# ---------------------------------------------------------------------------
# Checks that need the design inputs
# ---------------------------------------------------------------------------


def hydrostatics(case: CaseFile, raw: dict, rho: float) -> dict:
    """Displaced volume, waterplane area and centre of buoyancy at theta = 0, z = 0.

    Divergence theorem over the wetted hull walls (face centroid below the free
    surface). With F = s e_s, s the beam coordinate measured from the symmetry
    plane, the closing surfaces (symmetry plane: s = 0; waterplane: n_s = 0)
    add nothing. Fluent's face orientation is consistent per zone, so each
    zone's volume is taken positive.
    """
    up = np.asarray(raw["up_direction"])
    bow = np.asarray(raw["bow_direction"])
    s_ax = int(np.argmax(np.abs(np.cross(bow, up))))
    b_ax = int(np.argmax(np.abs(bow)))
    s0 = 0.0
    zfs = raw["free_surface_z"]
    V = Awp = Mb = 0.0
    for w in raw["hull_wall_zones"]:
        C, S = case.faces(w)
        wet = C[:, 2] < zfs
        v = float(np.sum((C[wet, s_ax] - s0) * S[wet, s_ax]))
        sign = 1.0 if v >= 0 else -1.0
        V += abs(v)
        Awp += abs(float(np.sum(S[wet, 2])))
        Mb += sign * float(np.sum(0.5 * C[wet, b_ax] ** 2 * S[wet, b_ax]))
    frac = 0.5 if raw["half_domain"] else 1.0
    W = raw["mass_full_kg"] * raw["g"] * frac
    B = rho * raw["g"] * V
    heave = (B - W) / (rho * raw["g"] * Awp) if Awp > 0 else float("nan")
    cb = Mb / V if V > 0 else float("nan")
    return {"volume_m3": V, "waterplane_m2": Awp, "buoyancy_N": B, "weight_N": W,
            "floating_mass_full_kg": rho * V / frac, "static_heave_m": heave,
            "cb_along_bow_m": cb * float(bow[b_ax]),
            "cg_along_bow_m": float(np.dot(raw["cg_ref"], bow))}


def envelope_margins(d: Derived, topo) -> list[dict]:
    """Hull clearance to the overset box and the bottom at the envelope corners."""
    from common import case_transform

    out = []
    env = d.raw["envelope"]
    for th in env["theta_deg"]:
        for z in env["z_m"]:
            p = case_transform(topo, th, z).apply(d.hull_vertices)
            row = {"theta_deg": th, "z_m": z, "bottom_m": float(p[:, 2].min() - d.raw["bottom_z"])}
            if d.overset_box is not None:
                olo, ohi = d.overset_box
                sides = {}
                for ax in range(3):
                    if ax == d.symmetry_axis:
                        continue
                    sides[f"-{'xyz'[ax]}"] = float(p[:, ax].min() - olo[ax])
                    sides[f"+{'xyz'[ax]}"] = float(ohi[ax] - p[:, ax].max())
                worst = min(sides, key=sides.get)
                row.update(overset_m=sides[worst], overset_side=worst)
            out.append(row)
    return out


def assemble(d: Derived, a) -> dict:
    """topology.json in the established key order, design inputs from the CLI."""
    r = d.raw
    frac = "half" if r["half_domain"] else "full"
    out = {
        "name": r["name"],
        "description": a.description or f"{r['name']}, {frac} domain",
        "_comment_init": "Written by `hs topology init` from the case named in _init_case; design "
                         "inputs (mass, CG, thrust, envelope, run) from its command line.",
        "_init_case": a.case_name,
        "half_domain": r["half_domain"],
        "mass_full_kg": a.mass_full_kg,
        "g": r["g"],
        "cg_ref": a.cg,
        "bow_direction": r["bow_direction"],
        "up_direction": r["up_direction"],
        "free_surface_z": r["free_surface_z"],
        "bottom_z": r["bottom_z"],
        "foreground_cell_zones": r["foreground_cell_zones"],
        "hull_wall_zones": r["hull_wall_zones"],
        "background_cell_zone": r["background_cell_zone"],
        "water_phase": r["water_phase"],
        "inlet_zone": r["inlet_zone"],
        "outlet_zone": r["outlet_zone"],
        "ref_speed_mps": a.ref_speed,
        "ref_dt_s": a.ref_dt,
        "thrust_ceiling_full_N": a.thrust_ceiling_full_n,
        "thrust_offset_below_keel_m": a.thrust_offset_below_keel_m,
        "hull_zmin_at_ref": r["hull_zmin_at_ref"],
        "hull_length_m": r["hull_length_m"],
        "envelope": {"theta_deg": a.envelope_theta, "z_m": a.envelope_z},
        "run": {"settle_hull_lengths": a.settle, "average_hull_lengths": a.average,
                "max_iter_per_step": a.max_iter},
        "template_hull_stats": r["template_hull_stats"],
    }
    return out


def _pair(s: str, what: str) -> list[float]:
    v = [float(x) for x in s.split(",")]
    if len(v) != 2 or v[0] >= v[1]:
        raise SystemExit(f"{what} must be LO,HI with LO < HI, not {s!r}")
    return v


def run(a) -> int:
    """The `hs topology init` command. Returns the exit status."""
    import layout
    from common import Topology, _unit

    name = layout.check_name(a.name, "topology")
    tdir = layout.topology_dir(name, must_exist=False)
    case_path = Path(a.case) if a.case else _find_case(layout, tdir, name)
    a.case_name = case_path.name
    a.envelope_theta = _pair(a.envelope_theta, "--envelope-theta")
    a.envelope_z = _pair(a.envelope_z, "--envelope-z")
    a.cg = [float(x) for x in a.cg.split(",")]
    if len(a.cg) != 3:
        raise SystemExit("--cg must be X,Y,Z")

    print(f"reading {case_path} (offline, h5py) ...")
    with CaseFile(case_path) as case:
        d = derive(case, name)
        rho = d.raw.pop("_rho_water")
        ready = d.raw.pop("_template_ready")
        topo_raw = assemble(d, a)
        d.raw.update(mass_full_kg=a.mass_full_kg, cg_ref=a.cg,
                     envelope=topo_raw["envelope"])

        print(f"\nfrom the case ({case.version}):")
        for k in topo_raw:
            if k in d.source:
                val = topo_raw[k]
                shown = ({kk: (np.round(vv, 5).tolist() if isinstance(vv, list) else vv)
                          for kk, vv in val.items()} if isinstance(val, dict) else val)
                print(f"  {k:<24} {shown}\n  {'':<24}   <- {d.source[k]}")
        print("\nfrom the command line:")
        for k in ("mass_full_kg", "cg_ref", "thrust_ceiling_full_N", "thrust_offset_below_keel_m",
                  "envelope", "ref_speed_mps", "ref_dt_s", "run"):
            print(f"  {k:<24} {topo_raw[k]}")
        for n in d.notes:
            print(f"  note: {n}")

        if d.symmetry_axis is not None and abs(a.cg[d.symmetry_axis]) > 1e-9:
            d.problems.append(f"--cg {a.cg} is off the symmetry plane ({'xyz'[d.symmetry_axis]} must be 0)")
        lo, hi = np.asarray(d.raw["template_hull_stats"]["min"]), np.asarray(d.raw["template_hull_stats"]["max"])
        if np.any(np.asarray(a.cg) < lo - 0.05) or np.any(np.asarray(a.cg) > hi + 0.05):
            d.warnings.append(f"--cg {a.cg} is outside the hull's bounding box {lo.round(3).tolist()} .. "
                              f"{hi.round(3).tolist()}")

        if not d.problems:
            h = hydrostatics(case, d.raw, rho)
            print("\nhydrostatics at theta = 0, z = 0 (simulated domain, no dynamic lift):")
            print(f"  displaced {h['volume_m3']:.5f} m3 -> buoyancy {h['buoyancy_N']:.1f} N vs weight "
                  f"{h['weight_N']:.1f} N; waterplane {h['waterplane_m2']:.4f} m2")
            print(f"  floats level here at a full-boat mass of {h['floating_mass_full_kg']:.1f} kg; "
                  f"static heave to float at {a.mass_full_kg:g} kg: {h['static_heave_m'] * 1000:+.0f} mm")
            print(f"  centre of buoyancy {h['cb_along_bow_m']:+.4f} m along the bow, CG {h['cg_along_bow_m']:+.4f} m")
            zlo, zhi = a.envelope_z
            if not zlo <= h["static_heave_m"] <= zhi:
                d.warnings.append(f"static heave {h['static_heave_m'] * 1000:+.0f} mm is outside the envelope "
                                  f"z {a.envelope_z}: check --mass-full-kg and the case's free surface, "
                                  "or widen the envelope")

            topo = Topology(path=tdir / layout.TOPOLOGY_FILE, raw=topo_raw, name=name,
                            cg_ref=np.asarray(a.cg, dtype=float), bow=_unit(topo_raw["bow_direction"]),
                            up=_unit(topo_raw["up_direction"]))
            print("\nenvelope corners (hull clearance, m):")
            for r in envelope_margins(d, topo):
                ov = (f"overset {r['overset_m']:+.3f} ({r['overset_side']})" if "overset_m" in r else "")
                print(f"  theta {r['theta_deg']:+6.2f} deg, z {r['z_m']:+.3f}:  bottom {r['bottom_m']:.3f}  {ov}")
                if r.get("overset_m", 1.0) < 0:
                    d.problems.append(f"the hull leaves the overset region at theta {r['theta_deg']}, "
                                      f"z {r['z_m']} ({r['overset_side']}); shrink the envelope")
                elif r.get("overset_m", 1.0) < _MARGIN_WARN_M:
                    d.warnings.append(f"only {r['overset_m'] * 1000:.0f} mm of overset clearance at theta "
                                      f"{r['theta_deg']}, z {r['z_m']} ({r['overset_side']})")

    for w in d.warnings:
        print(f"WARNING: {w}")
    if d.problems:
        for p in d.problems:
            print(f"PROBLEM: {p}")
        print("\nnot written: fix the problems above and run again.")
        return 2

    _naming_advice(layout, tdir, name, case_path, ready)
    dst = tdir / layout.TOPOLOGY_FILE
    if a.dry_run:
        print(f"\n--dry-run: {dst} not written")
        return 0
    if dst.exists() and not a.force:
        old = json.loads(dst.read_text())
        changed = sorted(k for k in set(old) | set(topo_raw) if old.get(k) != topo_raw.get(k))
        print(f"\n{dst} exists; fields that would change: {', '.join(changed) or 'none'}")
        print("not written: pass --force to replace it.")
        return 1
    tdir.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(topo_raw, indent=2) + "\n")
    print(f"\nwrote {dst}")
    return 0


def _find_case(layout, tdir: Path, name: str) -> Path:
    for role in ("template", "source"):
        try:
            p = layout.find_optional(tdir, layout.ROLE_PATTERNS[role].format(name=name), role)
        except layout.LayoutError as exc:
            raise SystemExit(str(exc))
        if p:
            return p
    cases = layout.find_all(tdir, "*.cas.h5")
    if len(cases) == 1:
        return cases[0]
    raise SystemExit(f"no case found by pattern in {tdir}; pass --case (saw: "
                     f"{', '.join(p.name for p in cases) or 'nothing'})")


def _naming_advice(layout, tdir: Path, name: str, case_path: Path, ready: bool) -> None:
    role = "template" if ready else "source"
    pat = layout.ROLE_PATTERNS[role].format(name=name)
    if case_path.resolve() in [p.resolve() for p in layout.find_all(tdir, pat)]:
        return
    want = layout.canonical_name(name, role, ".cas.h5")
    why = ("it is clean (no 6DOF zones, UDF or convergence conditions), so it can be the template as is"
           if ready else "prepare_case.py template must still strip it")
    print(f"\nnaming: {case_path.name} does not match '{pat}'. Rename it to {want} in {tdir}: {why}.")
