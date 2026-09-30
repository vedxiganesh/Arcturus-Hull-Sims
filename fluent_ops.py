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
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from common import (
    COMPONENTS,
    REPORT_FILE,
    REPORT_PREFIX,
    SDOF_UDF_NAME,
    UDF_LIBRARY,
    CaseTransform,
    Release,
    Topology,
    report_groups,
    report_name,
    udf_source,
)


class CaseSetupError(RuntimeError):
    """The case itself is wrong (bad transform, all-air init, missing setting).

    Deterministic: rerunning the same case reproduces it, so the orchestrator
    does not retry it.
    """


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
    """Vertex coordinates per surface (mesh only; no solution data needed).

    Request-object API (field_data.get_field_data(SurfaceFieldDataRequest)),
    present in PyFluent 0.40.1 and the only one in 0.42.1: VERIFIED
    2026-09-29, 0.42.1's LiveFieldData has no get_surface_data. The old
    keyword API is kept as a fallback for older releases.
    """
    import ansys.fluent.core as pf

    SurfaceDataType = pf.SurfaceDataType
    fd = solver.fields.field_data
    out = {}
    for s in surfaces:
        if hasattr(fd, "get_field_data") and hasattr(pf, "SurfaceFieldDataRequest"):
            res = fd.get_field_data(pf.SurfaceFieldDataRequest(surfaces=[s], data_types=[SurfaceDataType.Vertices]))
        else:
            res = fd.get_surface_data(surfaces=[s], data_types=[SurfaceDataType.Vertices])
        entry = res[s]
        verts = getattr(entry, "vertices", None)
        if verts is None and isinstance(entry, dict):
            verts = entry.get(SurfaceDataType.Vertices, entry.get("vertices"))
        arr = np.asarray(verts if verts is not None else [], dtype=float).reshape(-1, 3)
        if arr.shape[0] == 0:
            raise CaseSetupError(f"no vertices returned for surface {s!r} (entry type {type(entry).__name__})")
        out[s] = arr
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
    """The vertex centroid moves exactly like a point under a rigid motion.

    `before` and `after` must come from the SAME session: the centroid is a
    mean over the returned surface vertices, and that set depends on the
    partitioning (VERIFIED 2026-09-29: the template's centroid measured
    locally on 4 processes and on the cluster on 21 differ by 1.9e-4 m, with
    x off although a pitch rotation cannot change x).
    """
    expected = xf.apply(np.asarray(before["centroid"]))[0]
    err = float(np.max(np.abs(np.asarray(after["centroid"]) - expected)))
    if err > tol:
        raise CaseSetupError(
            f"transform check FAILED: centroid {after['centroid']} vs expected "
            f"{expected.tolist()} (err {err:.3g} m). Wrong sign, axis, or TUI "
            f"prompt order."
        )
    return err


#: How far the template's hull centroid, measured in the running session, may
#: sit from the value stored in topology.json. Partitioning alone moves it
#: ~2e-4 m; a wrong or moved template moves it by the motion itself.
TEMPLATE_CENTROID_TOL_M = 2e-3


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
            raise CaseSetupError("initialization defaults have zero velocity; cannot infer direction")
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
    raise CaseSetupError("time_step_size not found under run_calculation")


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


FLOW_TIME_REPORT = "flow-time"


def _one_report_file(solver, defs: list[str]) -> None:
    """Group all sweep reports into ONE relative-path report file.

    Relative, so the same case writes into whatever cwd the cluster job
    uses. Falls back to per-definition files if the container differs.

    The built-in time report "flow-time" goes last in report_defs, as in the
    GUI-made report files (introspection/hull_6sec); without it the file
    has only "Time Step" and collect.py has to borrow flow time elsewhere.
    """
    rd = solver.settings.solution.report_definitions
    try:
        has_ft = FLOW_TIME_REPORT in rd.time.get_state()
    except Exception as exc:
        log(f"could not list time report definitions ({exc})")
        has_ft = False
    if has_ft:
        defs = [*defs, FLOW_TIME_REPORT]
    else:
        log(f"WARNING: no {FLOW_TIME_REPORT!r} time report; {REPORT_FILE}.out will lack flow time")
    try:
        rf = solver.settings.solution.monitor.report_files
        if REPORT_FILE in rf.get_object_names():
            del rf[REPORT_FILE]
        rf[REPORT_FILE] = {"report_defs": defs, "file_name": f"{REPORT_FILE}.out"}
        log(f"report file {REPORT_FILE}.out <- {defs}")
    except Exception as exc:
        log(f"single report file failed ({exc}); enabling per-definition files")
        for d in defs:
            if d == FLOW_TIME_REPORT:
                continue
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


# ---------------------------------------------------------------------------
# One captive case from the template
# ---------------------------------------------------------------------------


def transcript_marks(cwd: Path) -> dict[Path, int]:
    return {p: p.stat().st_size for p in cwd.glob("fluent-*.trn")}


def transcript_tail(cwd: Path, since: dict[Path, int]) -> str:
    """Text appended to Fluent's auto-transcript(s) since `since` was taken."""
    out = []
    for p in sorted(cwd.glob("fluent-*.trn")):
        with open(p, "rb") as f:
            f.seek(since.get(p, 0))
            out.append(f.read().decode(errors="replace"))
    return "".join(out)


def prepare_case(solver, topo: Topology, template: Path, theta: float, z: float, speed: float,
                 dt: float, transcript_dir: Path) -> dict:
    """Template -> initialized captive case at (theta, z, speed), in the live session.

    Reads the template, moves the foreground and CHECKS the move against the
    analytic transform, sets inlet speed, init defaults and time step, creates
    the sweep reports about the displaced CG, initializes (flat open channel)
    and checks the water level. Does not write anything; returns the facts for
    the case sidecar.
    """
    from common import case_transform, check_envelope

    for m in check_envelope(topo, theta, z):
        log(f"WARNING: {m}")
    tmpl_stats = topo.get("template_hull_stats")
    if tmpl_stats is None:
        raise CaseSetupError("topology has no template_hull_stats; run `prepare_case.py template` first")
    xf = case_transform(topo, theta, z)

    read_case(solver, str(template))
    before = hull_stats(solver, topo)
    tmpl_diff = float(np.max(np.abs(np.asarray(before["centroid"]) - np.asarray(tmpl_stats["centroid"]))))
    log(f"template hull centroid here vs topology.json: {tmpl_diff:.2e} m (partitioning moves it ~2e-4)")
    if tmpl_diff > TEMPLATE_CENTROID_TOL_M:
        raise CaseSetupError(
            f"template hull centroid {before['centroid']} is {tmpl_diff:.3g} m from topology.json's "
            f"{tmpl_stats['centroid']}: this is not the template the topology was measured on")
    route = rotate_translate(solver, list(topo["foreground_cell_zones"]), xf)
    after = hull_stats(solver, topo)
    err = verify_transform(before, after, xf)
    clearance = after["zmin"] - topo["bottom_z"]
    log(f"transform via {route}: centroid err {err:.2e} m; hull zmin {after['zmin']:.4f} "
        f"(clearance to bottom {clearance:.3f} m)")

    set_inlet_speed(solver, topo["inlet_zone"], speed)
    init_defaults = update_init_defaults(solver, topo["inlet_zone"], speed)
    set_time_step(solver, dt, topo["run"]["max_iter_per_step"])
    reports = create_sweep_reports(solver, topo, xf.moment_center)
    clear_convergence_conditions(solver)
    relativize_report_files(solver)

    marks = transcript_marks(transcript_dir)
    initialize(solver, topo["inlet_zone"])
    water = check_water_level(solver, topo)
    tail = transcript_tail(transcript_dir, marks)
    orphans = [ln.strip() for ln in tail.splitlines() if "orphan" in ln.lower()]
    for ln in orphans:
        log(f"overset: {ln}")

    return {
        "transform": xf.__dict__,
        "transform_route": route,
        "transform_centroid_err_m": err,
        "template_centroid_diff_m": tmpl_diff,
        "hull_before": before,
        "hull_after": after,
        "bottom_clearance_m": clearance,
        "initialized": True,
        "init_defaults": init_defaults,
        "water_level_check": water,
        "overset_transcript_lines": orphans,
        "report_file": f"{REPORT_FILE}.out",
        "report_definitions": reports,
        "weight_domain_N": topo.weight_domain_N,
        "thrust_ceiling_domain_N": topo.thrust_ceiling_domain_N,
        "thrust_offset_body_m": topo.thrust_offset_body(),
    }


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
        raise CaseSetupError(
            f"initialized water fraction {measured} != flat-surface {expected:.4f} (tol {tol}); "
            "open-channel flat initialization did not take -- do not run this case."
        )
    return out


# ---------------------------------------------------------------------------
# Free-running 2DOF relaxation from a converged captive case (cluster, 26R1)
# ---------------------------------------------------------------------------

#: Written into the free case dir and compiled there (per case, so concurrent jobs never
#: share a libudf; CLAUDE.md section 4).
SDOF_SOURCE = "sw_sdof.c"
#: six_dof basename. Relative: Fluent's cwd is the case dir.
MOTION_BASENAME = "sw-motion"
#: Motion sampled by run_case after every chunk, from the live rigid_body_properties.
#: It does not depend on how Fluent's own .6dof file behaves on restart.
MOTION_LOG = "sw-motion.csv"
MOTION_LOG_HEADER = "flow_time,time_step,cg_x,cg_y,cg_z,theta_x_deg"
ENSIGHT_DIR = "ensight"
#: Relative on purpose: the export is registered by one TUI line split on spaces, and case
#: dir paths contain '+'.
ENSIGHT_BASENAME = f"{ENSIGHT_DIR}/free"
ENSIGHT_OBJECT = "sw-ensight"


def compile_udf(solver, case_dir: Path, rel: Release, note: str = "") -> dict:
    """Write the release's 2DOF UDF into the case dir, compile it to libudf, and load it."""
    src = case_dir / SDOF_SOURCE
    src.write_text(udf_source(rel, note))
    lib = case_dir / UDF_LIBRARY
    if lib.exists():  # a stale build from an earlier attempt: Fluent would ask about it
        shutil.rmtree(lib)
        log(f"removed stale {lib}")
    ud = solver.settings.setup.user_defined
    route = "tui"
    if has_child(ud, "compiled_udf") and has_child(ud, "load"):
        try:
            ud.compiled_udf(library_name=UDF_LIBRARY, source_files=[SDOF_SOURCE], header_files=[])
            ud.load(udf_library_name=UDF_LIBRARY)
            route = "settings"
        except Exception as exc:
            log(f"settings compiled_udf/load failed ({exc}); trying the TUI")
    if route == "tui":
        # UNVERIFIED prompt order: library, 'yes' to add sources, source, header list.
        tui(solver, f'/define/user-defined/compiled-functions compile {UDF_LIBRARY} yes {SDOF_SOURCE} "" ""')
        tui(solver, f"/define/user-defined/compiled-functions load {UDF_LIBRARY}")
    if not lib.is_dir():
        raise CaseSetupError(f"UDF compile via {route} left no {lib}; see the Fluent transcript")
    log(f"compiled and loaded {SDOF_SOURCE} -> {UDF_LIBRARY} via {route}")
    return {"udf_source": src.name, "udf_route": route}


def _create_dynamic_zone(dz, name: str, zone: str, state: dict) -> str:
    """Create one dynamic zone. Returns the route used."""
    try:
        dz[name] = state
        return "setitem"
    except Exception as exc:
        log(f"dynamic_zones[{name!r}] = state failed ({exc}); trying create(zone=...)")
    before = set(dz.get_object_names())
    dz.create(zone=zone)
    new = set(dz.get_object_names()) - before
    if len(new) != 1:
        raise CaseSetupError(f"dynamic_zones.create(zone={zone!r}) made {sorted(new)}, expected one zone")
    dz[new.pop()].set_state({k: v for k, v in state.items() if k != "zone"})
    return "create"


def set_implicit_update(dm, implicit: dict | None) -> dict:
    """6DOF implicit update (dynamic_mesh.options.implicit_update, 26R1 fields).

    On, the 6DOF motion is updated inside the time step every update_interval iterations,
    under-relaxed by relaxation_factor. This is Fluent's remedy for the added-mass
    instability of a body that is light compared with the water it moves.
    """
    iu = dm.options.implicit_update
    if not implicit or not implicit.get("enabled"):
        iu.enabled = False
        log("6DOF implicit update off")
        return {"enabled": False}
    iu.enabled = True  # the other fields are inactive until this is set
    for k in ("update_interval", "relaxation_factor", "residual_criterion"):  # not mode/k_over_L
        if k in implicit:
            setattr(iu, k, implicit[k])
    state = iu.get_state()
    log(f"6DOF implicit update on: {state}")
    return state


def arm_six_dof(solver, topo: Topology, rel: Release, implicit: dict | None = None) -> dict:
    """Dynamic mesh on, 6DOF on, and one rigid-body dynamic zone per foreground part.

    The hull walls carry the 6DOF body (UDF stage::libudf). Every foreground cell zone
    follows it as a passive rigid body. That includes the solid fluid:1 inside the ama:
    the GUI setup left it static, but it shares nodes with the moving wall_amas.
    The layout follows introspection/hull_6sec/state.setup.dynamic_mesh.json.
    The orientation starts at 0, so the 6DOF angles are measured from the captive trim.
    `implicit` is study["free"]["implicit_update"] (see set_implicit_update).
    """
    dm = solver.settings.setup.dynamic_mesh
    dm.enabled = True
    six = dm.options.six_dof
    six.enabled = True
    six.gravity = {"x": 0.0, "y": 0.0, "z": -float(topo["g"])}
    six.write_motion_history = True
    six.basename = MOTION_BASENAME
    six.second_order = True
    implicit_state = set_implicit_update(dm, implicit)
    dz = dm.dynamic_zones
    for name in list(dz.get_object_names()):
        del dz[name]
        log(f"deleted dynamic zone {name}")

    motion_def = f"{SDOF_UDF_NAME}::{UDF_LIBRARY}"
    rbp = {"cg_position": list(rel.cg), "orientation": {"angle": 0.0, "axis": [1.0, 0.0, 0.0]},
           "cg_velocity": [0.0, 0.0, 0.0], "angular_velocity": [0.0, 0.0, 0.0]}
    specs = ([(z, False) for z in topo["hull_wall_zones"]]
             + [(z, True) for z in topo["foreground_cell_zones"]])
    routes = {}
    for i, (zone, passive) in enumerate(specs):
        state = {"zone": zone, "type": "rigid-body",
                 "motion": {"motion_def": motion_def, "six_dof": {"enabled": True, "passive": passive},
                            "rigid_body_properties": rbp}}
        routes[zone] = _create_dynamic_zone(dz, f"sw-dz-{i}", zone, state)

    got = dz.get_state()
    seen = {}
    for z in got.values():
        m = z.get("motion") or {}
        seen[z.get("zone")] = (z.get("type"), m.get("motion_def"), (m.get("six_dof") or {}).get("passive"))
    want = {zone: ("rigid-body", motion_def, passive) for zone, passive in specs}
    if seen != want:
        raise CaseSetupError(f"dynamic zones read back as {seen}, wanted {want}")
    log(f"6DOF armed: CG {list(rel.cg)}, mass {rel.mass_kg:g} kg, zones {routes}")
    return {"dynamic_zone_routes": routes, "dynamic_zones": got, "implicit_update": implicit_state}


def ensight_command(water_phase: str, every_s: float) -> str:
    """One-line TUI registration of an EnSight Gold transient export on a flow-time trigger.

    Same prompt order as run_hull_vof.py's build_export_command: name, interior surfaces,
    cell zones, scalars ending with 'q', cell-centred?, binary?, export name, trigger,
    frequency, separate files?. That string wrote the EnSight series of job 22643276
    (Trimaran_HalfD.encas: these five scalars plus velocity).
    """
    scalars = ["pressure", "wall-shear", f"{water_phase}-vof",
               "cell-convective-courant-number", "moving-mesh-courant-number"]
    return " ".join(["/file/transient-export/ensight-gold-transient", ENSIGHT_BASENAME, "()", "*", "()",
                     *scalars, "q", "no", "yes", f'"{ENSIGHT_OBJECT}"', '"flow-time"',
                     f"{float(every_s):g}", "yes"])


def configure_ensight(solver, topo: Topology, case_dir: Path, every_s: float) -> bool:
    """Replace any earlier export object with ours. Returns False (and logs) if it failed."""
    (case_dir / ENSIGHT_DIR).mkdir(exist_ok=True)
    for name in ("export-1", ENSIGHT_OBJECT):
        for cmd in (f"/solve/execute-commands/delete {name}", f"/file/transient-export/settings/delete {name}"):
            try:
                tui(solver, cmd)
            except Exception:
                pass
    cmd = ensight_command(topo["water_phase"], every_s)
    log(f"EnSight export: {cmd}")
    try:
        tui(solver, cmd)
    except Exception as exc:
        log(f"WARNING: EnSight export NOT registered ({exc})")
        return False
    try:
        tui(solver, "/file/transient-export/settings/list")
    except Exception:
        pass
    return True


def read_sdof_state(solver, zone: str) -> dict:
    """Live 6DOF state of the dynamic zone on `zone`: CG and the X rotation since release."""
    dz = solver.settings.setup.dynamic_mesh.dynamic_zones
    for z in dz.get_state().values():
        if z.get("zone") == zone:
            rbp = z["motion"]["rigid_body_properties"]
            o = rbp.get("orientation") or {}
            axis = o.get("axis") or [1.0, 0.0, 0.0]
            n = math.sqrt(sum(float(a) ** 2 for a in axis)) or 1.0
            return {"cg": [float(v) for v in rbp["cg_position"]],
                    "theta_x_deg": math.degrees(float(o.get("angle", 0.0)) * float(axis[0]) / n)}
    raise KeyError(f"no dynamic zone on {zone!r}")


def prepare_free_case(solver, topo: Topology, parent_cas: Path, parent_dat: Path, rel: Release,
                      dt: float, max_iter: int, case_dir: Path,
                      parent_hull_centroid: list | None = None, implicit: dict | None = None) -> dict:
    """Converged captive case + data -> the same state with the hull free in heave and pitch.

    No initialization: the converged flow is the initial condition. The hull centroid is
    checked against the parent's own measurement, which also catches reading the wrong parent.
    """
    for p in (parent_cas, parent_dat):
        if not p.is_file():
            raise CaseSetupError(f"parent file missing: {p}")
    read_case(solver, str(parent_cas))
    log(f"read data: {parent_dat}")
    solver.settings.file.read(file_type="data", file_name=str(parent_dat))

    here = hull_stats(solver, topo)
    diff = None
    if parent_hull_centroid is not None:
        diff = float(np.max(np.abs(np.asarray(here["centroid"]) - np.asarray(parent_hull_centroid))))
        log(f"hull centroid vs parent's: {diff:.2e} m")
        if diff > TEMPLATE_CENTROID_TOL_M:
            raise CaseSetupError(f"hull centroid {here['centroid']} is {diff:.3g} m from the parent's "
                                 f"{parent_hull_centroid}: not the parent case's geometry")

    set_time_step(solver, dt, max_iter)
    facts = compile_udf(solver, case_dir, rel, note=f"parent {parent_cas.name}")
    facts.update(arm_six_dof(solver, topo, rel, implicit))
    clear_convergence_conditions(solver)
    relativize_report_files(solver)
    return {**facts, "parent_cas": str(parent_cas), "parent_dat": str(parent_dat),
            "hull_at_release": here, "parent_centroid_diff_m": diff, "release": rel.to_dict(),
            "weight_domain_N": topo.weight_domain_N}
