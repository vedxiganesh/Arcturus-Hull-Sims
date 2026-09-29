"""Fluent-facing operations for hullsweep, written introspect-first.

Every settings path here is looked up on the live tree (child_names /
command_names / get_state) before it is used, with a TUI fallback where one
exists. Nothing is assumed from another release: the local prep stack is
Fluent 25R2 + PyFluent 0.40.1, the cluster is 26R1 + 0.42.1, and paths have
moved between them before (CLAUDE.md section 8).

Each helper logs WHICH route it took, so the first real run records the
paths that work and they can be hard-coded later.
"""

from __future__ import annotations

import math
import re
import time
from typing import Any, Callable, Iterable

import numpy as np

from common import (
    COMPONENTS,
    REPORT_FILE,
    REPORT_PREFIX,
    CaseTransform,
    Topology,
    report_groups,
    report_name,
)


def log(msg: str) -> None:
    print(f"[hullsweep {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Launch / raw command plumbing
# ---------------------------------------------------------------------------


def launch_solver(cwd: str, procs: int = 4, ui_mode: str = "no_gui_or_graphics"):
    import ansys.fluent.core as pyfluent

    log(f"launching Fluent solver: procs={procs} ui_mode={ui_mode} cwd={cwd}")
    return pyfluent.launch_fluent(
        mode="solver",
        ui_mode=ui_mode,
        processor_count=procs,
        cwd=cwd,
        start_timeout=600,
        additional_arguments="",  # empty string, never None
    )


def tui(solver, command: str):
    """Run one raw TUI command line."""
    try:
        return solver.execute_tui(command)
    except AttributeError:
        return solver.scheme_eval.scheme_eval(f'(ti-menu-load-string "{command}")')


def names(obj, attr: str) -> list[str]:
    try:
        return list(getattr(obj, attr))
    except Exception:
        return []


def has_child(obj, name: str) -> bool:
    return name in names(obj, "child_names") or name in names(obj, "command_names")


def nav(obj, path: Iterable[str]):
    """Walk a settings path: attribute access, falling back to [] for named objects."""
    for part in path:
        if part in names(obj, "child_names") or part in names(obj, "command_names"):
            obj = getattr(obj, part)
        else:
            obj = obj[part]
    return obj


def find_leaves(state: Any, pred: Callable[[str], bool], prefix=()) -> list[tuple[tuple, Any]]:
    """All (path, value) pairs in a get_state() dict whose KEY satisfies pred."""
    out = []
    if isinstance(state, dict):
        for k, v in state.items():
            p = prefix + (k,)
            if pred(k):
                out.append((p, v))
            out.extend(find_leaves(v, pred, p))
    return out


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------


def read_case(solver, path: str, data: bool = False) -> None:
    ftype = "case-data" if data else "case"
    log(f"read {ftype}: {path}")
    solver.settings.file.read(file_type=ftype, file_name=path)


def write_case(solver, path: str, data: bool) -> None:
    ftype = "case-data" if data else "case"
    log(f"write {ftype}: {path}")
    f = solver.settings.file
    try:
        f.write(file_type=ftype, file_name=path)
    except Exception as exc:
        legacy = "write_case_data" if data else "write_case"
        log(f"file.write failed ({exc}); trying file.{legacy}")
        getattr(f, legacy)(file_name=path)


# ---------------------------------------------------------------------------
# Geometry queries
# ---------------------------------------------------------------------------


def surface_vertices(solver, surfaces: list[str]) -> dict[str, np.ndarray]:
    """Vertex coordinates per surface (mesh only; no solution data needed)."""
    from ansys.fluent.core.services.field_data import SurfaceDataType

    fd = solver.fields.field_data
    out = {}
    for s in surfaces:
        res = fd.get_surface_data(surfaces=[s], data_types=[SurfaceDataType.Vertices])
        entry = res[s]
        verts = getattr(entry, "vertices", None)
        if verts is None and isinstance(entry, dict):
            verts = entry.get(SurfaceDataType.Vertices, entry.get("vertices"))
        out[s] = np.asarray(verts, dtype=float).reshape(-1, 3)
    return out


def hull_stats(solver, topo: Topology) -> dict:
    """Centroid, bounding box, and length of the hull wall vertices."""
    v = surface_vertices(solver, list(topo["hull_wall_zones"]))
    allv = np.concatenate(list(v.values()), axis=0)
    lo, hi = allv.min(axis=0), allv.max(axis=0)
    if not np.allclose(topo.up, [0.0, 0.0, 1.0]):
        raise ValueError("hull_stats assumes up = +Z")
    return {
        "centroid": allv.mean(axis=0).tolist(),
        "min": lo.tolist(),
        "max": hi.tolist(),
        "zmin": float(lo[2]),
        "length": float(np.dot(hi - lo, np.abs(topo.bow))),
        "n_vertices": int(allv.shape[0]),
    }


# ---------------------------------------------------------------------------
# Rigid motion of the foreground
# ---------------------------------------------------------------------------


def _cmd_args(cmd) -> list[str]:
    return names(cmd, "argument_names")


def rotate_translate(solver, zones: list[str], xf: CaseTransform) -> str:
    """Rotate the foreground cell zones, then translate. Returns the route used."""
    mz = None
    try:
        mz = solver.settings.mesh.modify_zones
    except Exception:
        pass

    if mz is not None and has_child(mz, "rotate_zone") and has_child(mz, "translate_zone"):
        rot, tra = mz.rotate_zone, mz.translate_zone
        ra, ta = _cmd_args(rot), _cmd_args(tra)
        log(f"settings rotate_zone args={ra}  translate_zone args={ta}")
        rk = _match_args(ra, {"zones": ("zone_names", "zones", "zone_list"),
                              "angle": ("rotation_angle", "angle"),
                              "origin": ("origin",),
                              "axis": ("axis", "axis_components")})
        tk = _match_args(ta, {"zones": ("zone_names", "zones", "zone_list"),
                              "offset": ("translation", "offset", "translate")})
        if rk and tk:
            if xf.fluent_angle_deg != 0.0:
                # The settings API is SI: rotation_angle is in RADIANS (the TUI
                # takes degrees). VERIFIED 2026-09-28: passing -3 rotated the
                # foreground by -3 rad; the centroid check reproduced that
                # exactly and stopped the run.
                rot(**{rk["zones"]: zones, rk["angle"]: math.radians(xf.fluent_angle_deg),
                       rk["origin"]: list(xf.origin), rk["axis"]: list(xf.axis)})
            if any(xf.translation):
                tra(**{tk["zones"]: zones, tk["offset"]: list(xf.translation)})
            return "settings"
        log("settings rotate/translate present but argument names unrecognised; using TUI")

    # TUI fallback. UNVERIFIED prompt order: zones, angle, origin xyz, axis xyz.
    # The caller checks the result against the analytic transform, so a wrong
    # order fails loudly instead of producing a silently wrong case.
    zl = " ".join(zones)
    if xf.fluent_angle_deg != 0.0:
        o, a = xf.origin, xf.axis
        tui(solver, f"/mesh/modify-zones/rotate-zone {zl} () {xf.fluent_angle_deg} "
                    f"{o[0]} {o[1]} {o[2]} {a[0]} {a[1]} {a[2]}")
    if any(xf.translation):
        t = xf.translation
        tui(solver, f"/mesh/modify-zones/translate-zone {zl} () {t[0]} {t[1]} {t[2]}")
    return "tui"


def _match_args(available: list[str], wanted: dict[str, tuple[str, ...]]) -> dict | None:
    out = {}
    for key, candidates in wanted.items():
        hit = next((c for c in candidates if c in available), None)
        if hit is None:
            return None
        out[key] = hit
    return out


def verify_transform(before: dict, after: dict, xf: CaseTransform, tol: float = 1e-5) -> float:
    """The vertex centroid moves exactly like a point under a rigid motion."""
    expected = xf.apply(np.asarray(before["centroid"]))[0]
    err = float(np.max(np.abs(np.asarray(after["centroid"]) - expected)))
    if err > tol:
        raise RuntimeError(
            f"transform check FAILED: centroid {after['centroid']} vs expected "
            f"{expected.tolist()} (err {err:.3g} m). Wrong sign, axis, or TUI "
            f"prompt order."
        )
    return err


# ---------------------------------------------------------------------------
# Case physics edits
# ---------------------------------------------------------------------------


def disable_dynamic_mesh(solver) -> str:
    """Delete the 6DOF dynamic zones (they reference stage::libudf), then turn dynamic mesh off."""
    dm = solver.settings.setup.dynamic_mesh
    try:
        dz = dm.dynamic_zones
        for name in list(dz.get_object_names()):
            del dz[name]
            log(f"deleted dynamic zone {name}")
    except Exception as exc:
        log(f"WARNING: dynamic zones not deleted ({exc}); they may still reference libudf")
    for flag in ("enabled", "enable", "dynamic_mesh"):
        if flag in names(dm, "child_names"):
            setattr(dm, flag, False)
            log(f"dynamic mesh off via setup.dynamic_mesh.{flag}")
            return flag
    tui(solver, "/define/dynamic-mesh/dynamic-mesh? no")
    log("dynamic mesh off via TUI")
    return "tui"


def unload_udf(solver, lib: str = "libudf") -> None:
    try:
        tui(solver, f'/define/user-defined/compiled-functions unload "{lib}"')
        log(f"unloaded {lib}")
    except Exception as exc:
        log(f"UDF unload skipped ({exc})")


def clear_convergence_conditions(solver) -> None:
    try:
        cc = solver.settings.solution.monitor.convergence_conditions
        reports = cc.convergence_reports
        for name in list(reports.get_object_names()):
            del reports[name]
            log(f"removed convergence condition {name}")
    except Exception as exc:
        log(f"WARNING: could not clear convergence conditions ({exc})")


_VMAG = re.compile(r"(^vmag$|velocity_magnitude|velocity_mag|^v_mag)")


def set_inlet_speed(solver, zone: str, speed: float) -> list[str]:
    """Set every velocity-magnitude leaf of the open-channel pressure inlet."""
    bcs = solver.settings.setup.boundary_conditions
    bc = None
    for kind in ("pressure_inlet", "velocity_inlet"):
        try:
            container = getattr(bcs, kind)
            if zone in container.get_object_names():
                bc = container[zone]
                break
        except Exception:
            continue
    if bc is None:
        raise KeyError(f"inlet zone {zone!r} not found as pressure/velocity inlet")

    hits = find_leaves(bc.get_state(), lambda k: bool(_VMAG.search(k)))
    if not hits:
        raise KeyError(f"no velocity-magnitude field under {zone}; dump its state and extend _VMAG")
    done = []
    for path, cur in hits:
        obj = nav(bc, path)
        if isinstance(cur, dict) and "value" in cur:
            obj.set_state({**cur, "value": float(speed)})
        elif isinstance(cur, str):
            obj.set_state(f"{float(speed)} [m/s]")
        else:
            obj.set_state(float(speed))
        done.append("/".join(path))
    log(f"inlet speed {speed} m/s set at: {done}")
    return done


def update_init_defaults(solver, inlet: str, speed: float) -> dict:
    """Make the standard-initialization defaults match the NEW inlet speed.

    The template's defaults were computed from the inlet at the template
    speed (y-velocity 2.5, k and omega to match). Changing only the BC would
    initialize every other speed with a 2.5 m/s field. Try Fluent's own
    compute-from-inlet first; if the read-back does not show |U| = speed,
    rescale directly: velocity ~ V, and at fixed intensity and viscosity
    ratio both k ~ (I V)^2 and omega = rho k / (mu ratio) scale as V^2.
    """
    init = solver.settings.solution.initialization
    try:
        init.compute_defaults(from_zone_type="pressure-inlet", from_zone_name=inlet, phase="mixture")
        log(f"compute_defaults from {inlet}")
    except Exception as exc:
        log(f"compute_defaults failed ({exc}); rescaling defaults directly")

    d = init.defaults.get_state()
    vel = np.array([float(d.get(f"{c}-velocity", 0.0)) for c in "xyz"])
    umag = float(np.sqrt(np.dot(vel, vel)))
    if abs(umag - speed) > 1e-6 * max(speed, 1.0):
        if umag == 0.0:
            raise RuntimeError("initialization defaults have zero velocity; cannot infer direction")
        r = speed / umag
        new = {f"{c}-velocity": float(v * r) for c, v in zip("xyz", vel)}
        for key in ("k", "omega"):
            if key in d:
                new[key] = float(d[key]) * r * r
        for key, val in new.items():
            init.defaults[key] = val
        log(f"init defaults rescaled {umag:.4g} -> {speed:.4g} m/s: {new}")
    out = init.defaults.get_state()
    log(f"init defaults: {out}")
    return out


def set_time_step(solver, dt: float, max_iter: int) -> None:
    rc = solver.settings.solution.run_calculation
    for holder in ("parameters", "transient_controls"):
        try:
            h = getattr(rc, holder)
            if "time_step_size" in names(h, "child_names"):
                h.time_step_size = dt
                if "max_iter_per_time_step" in names(h, "child_names"):
                    h.max_iter_per_time_step = max_iter
                log(f"dt={dt} via run_calculation.{holder}")
                return
        except Exception:
            continue
    raise RuntimeError("time_step_size not found under run_calculation")


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

_CENTER = ("mom_center", "moment_center", "center")
_AXIS = ("mom_axis", "moment_axis", "axis")


def create_sweep_reports(solver, topo: Topology, moment_center) -> list[str]:
    """Fx,Fy,Fz and Mx,My,Mz (about the displaced CG) per group, in one file."""
    rd = solver.settings.solution.report_definitions
    created = []
    for group, zones in report_groups(topo).items():
        for i, c in enumerate(COMPONENTS):
            vec = [0.0, 0.0, 0.0]
            vec[i] = 1.0

            fname = report_name("f", c, group)
            _recreate(rd.force, fname, {"zones": zones, "force_vector": vec})
            created.append(fname)

            mname = report_name("m", c, group)
            _recreate(rd.moment, mname, {"zones": zones})
            m = rd.moment[mname]
            kids = names(m, "child_names")
            ck = next((k for k in _CENTER if k in kids), None)
            ak = next((k for k in _AXIS if k in kids), None)
            if ck is None or ak is None:
                raise KeyError(f"moment report children {kids}: no centre/axis field recognised")
            setattr(m, ck, list(moment_center))
            setattr(m, ak, vec)
            created.append(mname)
    log(f"created {len(created)} sweep report definitions")
    _one_report_file(solver, created)
    return created


def _recreate(container, name: str, state: dict) -> None:
    try:
        if name in container.get_object_names():
            del container[name]
    except Exception:
        pass
    container[name] = state


def _one_report_file(solver, defs: list[str]) -> None:
    """Group all sweep reports into ONE relative-path report file.

    Relative, so the same case writes into whatever cwd the cluster job
    uses. Falls back to per-definition files if the container differs.
    """
    try:
        rf = solver.settings.solution.monitor.report_files
        if REPORT_FILE in rf.get_object_names():
            del rf[REPORT_FILE]
        rf[REPORT_FILE] = {"report_defs": defs, "file_name": f"{REPORT_FILE}.out"}
        log(f"report file {REPORT_FILE}.out <- {len(defs)} definitions")
    except Exception as exc:
        log(f"single report file failed ({exc}); enabling per-definition files")
        rd = solver.settings.solution.report_definitions
        for d in defs:
            kind = rd.force if d.startswith("sw-f") else rd.moment
            kind[d].create_report_file = True


def relativize_report_files(solver) -> None:
    """Strip directories from every report-file path (Windows -> Linux safety)."""
    try:
        rf = solver.settings.solution.monitor.report_files
        for n in rf.get_object_names():
            fn = str(rf[n].file_name())
            base = re.split(r"[\\/]", fn)[-1]
            if base != fn:
                rf[n].file_name = base
                log(f"report file {n}: {fn} -> {base}")
    except Exception as exc:
        log(f"WARNING: could not relativize report files ({exc})")


def ensure_flat_open_channel_init(solver, inlet: str) -> dict:
    """Set flat open-channel initialization from the inlet EXPLICITLY.

    The template carries open_channel_auto_init = {boundary_zone: inlet,
    flat_init: true} from the GUI setup, but 26R1 adds an
    open_channel_initialization_method field that a 25R2-written case may not
    populate, so nothing is left to inheritance.
    """
    init = solver.settings.solution.initialization
    init.initialization_type = "standard"
    oc = init.open_channel_auto_init
    kids = names(oc, "child_names")
    oc.boundary_zone = inlet
    if "open_channel_initialization_method" in kids:  # 26R1+
        oc.open_channel_initialization_method = "Flat"
    if "flat_init" in kids:
        try:
            oc.flat_init = True
        except Exception as exc:  # may be inactive once the method field is set
            log(f"flat_init not set directly ({exc})")
    state = oc.get_state()
    log(f"open-channel init: {state}")
    return state


def initialize(solver, inlet: str) -> None:
    ensure_flat_open_channel_init(solver, inlet)
    solver.settings.solution.initialization.standard_initialize()
    log("standard initialization done (flat open-channel from inlet)")


def _first_float(obj):
    if isinstance(obj, (int, float)) and not isinstance(obj, bool):
        return float(obj)
    if isinstance(obj, dict):
        obj = list(obj.values())
    if isinstance(obj, (list, tuple)):
        for v in obj:
            f = _first_float(v)
            if f is not None:
                return f
    return None


_WATER_REPORT = f"{REPORT_PREFIX}-init-water"


def check_water_level(solver, topo: Topology, tol: float = 0.03) -> dict:
    """Volume-averaged water fraction of the background vs. a flat free surface.

    An INTERIOR check on purpose: inlet/outlet face values come from the
    open-channel BCs and read correctly even if the interior was initialized
    all-air (defaults carry phase-2-mp = 0). For a box background with a
    flat surface, the expected value is (z_fs - z_min) / (z_max - z_min),
    with z extents taken from the inlet/outlet vertices.
    """
    v = surface_vertices(solver, [topo["inlet_zone"], topo["outlet_zone"]])
    z = np.concatenate([a[:, 2] for a in v.values()])
    zmin, zmax = float(z.min()), float(z.max())
    expected = float(np.clip((topo["free_surface_z"] - zmin) / (zmax - zmin), 0.0, 1.0))

    rd = solver.settings.solution.report_definitions
    _recreate(rd.volume, _WATER_REPORT, {"report_type": "volume-average",
                                        "cell_zones": [topo["background_cell_zone"]]})
    rep = rd.volume[_WATER_REPORT]
    try:
        rep.phase = topo["water_phase"]
    except Exception as exc:
        log(f"volume report phase not set ({exc})")
    allowed = []
    try:
        allowed = list(rep.field.allowed_values())
    except Exception:
        pass
    pick = next((f for f in ("vof", "volume-fraction", f"{topo['water_phase']}-vof") if f in allowed), None)
    if pick is None:
        pick = next((f for f in allowed if re.search(r"vof|volume.fraction", f)), None)
    if pick is None:
        raise KeyError(f"no volume-fraction field among allowed values: {allowed[:40]}")
    rep.field = pick
    res = rd.compute(report_defs=[_WATER_REPORT])
    measured = _first_float(res)
    try:
        del rd.volume[_WATER_REPORT]
    except Exception:
        pass
    out = {"field": pick, "z_extent": [zmin, zmax], "expected": expected,
           "measured": measured, "ok": measured is not None and abs(measured - expected) <= tol}
    log(f"water-level check: {out}")
    if not out["ok"]:
        raise RuntimeError(
            f"initialized water fraction {measured} != flat-surface {expected:.4f} (tol {tol}); "
            "open-channel flat initialization did not take -- do not run this case."
        )
    return out
