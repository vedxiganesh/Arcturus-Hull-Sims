#!/usr/bin/env python3
"""
Read-only dump of a Fluent case's settings tree to JSON.

Purpose: every settings path this project uses has to be read off the live
tree, never guessed (CLAUDE.md section 11). This script loads a case (or
attaches to a running session), walks the parts of the tree the sweep
pipeline touches, and writes what it finds to disk. It never writes a case.

Usage (local, reefs-mobo env):
    python introspect.py --case <path.cas.h5> --out <dir> [--procs 4]
    python introspect.py --server-info <server_info-*.txt> --out <dir>
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import ansys.fluent.core as pyfluent

# Settings subtrees dumped with get_state(). Missing paths are recorded, not
# fatal: which paths exist in this release is itself part of the answer.
STATE_PATHS = [
    "setup.general",
    "setup.models.multiphase",
    "setup.models.viscous",
    "setup.materials",
    "setup.cell_zone_conditions",
    "setup.boundary_conditions",
    "setup.dynamic_mesh",
    "setup.reference_values",
    "setup.named_expressions",
    "solution.methods",
    "solution.controls",
    "solution.report_definitions",
    "solution.monitor.report_files",
    "solution.monitor.convergence_conditions",
    "solution.initialization",
    "solution.run_calculation",
    "solution.calculation_activity",
]

# Subtrees where only the child names matter (commands and structure).
CHILD_PATHS = [
    "",
    "setup",
    "mesh",
    "mesh.modify_zones",
    "setup.dynamic_mesh",
    "setup.overset_interfaces",
    "setup.boundary_conditions",
    "solution.report_definitions",
    "solution.initialization",
    "file",
]


def log(msg: str) -> None:
    print(f"[introspect {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def resolve(root, dotted: str):
    obj = root
    for part in filter(None, dotted.split(".")):
        obj = getattr(obj, part)
    return obj


def names_of(obj) -> dict:
    out = {}
    for attr in ("child_names", "command_names", "query_names", "argument_names"):
        try:
            out[attr] = list(getattr(obj, attr))
        except Exception:
            pass
    return out


def zone_bboxes(solver, zones: list[str]) -> dict:
    """Vertex bounding box per surface, from field data (mesh only, no data)."""
    # Shares fluent_ops' extraction so the return-shape handling (a dict keyed
    # by SurfaceDataType in PyFluent 0.40.1) lives in one place.
    from fluent_ops import surface_vertices

    out = {}
    for zone in zones:
        try:
            v = surface_vertices(solver, [zone])[zone]
            out[zone] = {"min": v.min(axis=0).tolist(), "max": v.max(axis=0).tolist(),
                         "centroid": v.mean(axis=0).tolist(), "n_vertices": int(v.shape[0])}
        except Exception as exc:
            out[zone] = {"error": repr(exc)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case")
    ap.add_argument("--server-info")
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=4)
    args = ap.parse_args()

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    if args.server_info:
        solver = pyfluent.connect_to_fluent(server_info_file_name=args.server_info)
        owned = False
    else:
        log("launching Fluent (solver, no_gui_or_graphics)")
        solver = pyfluent.launch_fluent(
            mode="solver",
            ui_mode="no_gui_or_graphics",
            processor_count=args.procs,
            cwd=str(out),
            start_timeout=600,
            additional_arguments="",
        )
        owned = True

    report: dict = {"pyfluent": pyfluent.__version__}
    try:
        try:
            report["fluent_version"] = str(solver.get_fluent_version())
        except Exception as exc:
            report["fluent_version"] = repr(exc)

        if args.case:
            log(f"reading case {args.case}")
            t0 = time.time()
            solver.settings.file.read(file_type="case", file_name=str(Path(args.case).resolve()))
            report["read_seconds"] = round(time.time() - t0, 1)

        root = solver.settings
        report["children"] = {}
        for p in CHILD_PATHS:
            try:
                report["children"][p or "<root>"] = names_of(resolve(root, p))
            except Exception as exc:
                report["children"][p or "<root>"] = {"error": repr(exc)}

        for p in STATE_PATHS:
            log(f"state: {p}")
            try:
                state = resolve(root, p).get_state()
                (out / f"state.{p}.json").write_text(json.dumps(state, indent=1, default=str))
                report.setdefault("state_ok", []).append(p)
            except Exception as exc:
                report.setdefault("state_err", {})[p] = repr(exc)

        # Surface / zone inventory and wall bounding boxes.
        try:
            info = solver.fields.field_info.get_surfaces_info()
            report["surfaces"] = {k: {kk: str(vv) for kk, vv in v.items()} for k, v in info.items()}
            walls_etc = [k for k in info if not k.startswith("interior")]
            report["bboxes"] = zone_bboxes(solver, walls_etc)
        except Exception as exc:
            report["surfaces_err"] = repr(exc)

        # Raw zone listing via TUI: goes to the transcript, which is the one
        # place zone ids and types are printed together.
        for cmd in ("/define/boundary-conditions/list-zones",
                    "/mesh/modify-zones/list-zones"):
            try:
                solver.execute_tui(cmd)
            except Exception as exc:
                report.setdefault("tui_err", {})[cmd] = repr(exc)

        (out / "introspect_summary.json").write_text(json.dumps(report, indent=1, default=str))
        log(f"wrote {out / 'introspect_summary.json'}")
    finally:
        if owned:
            try:
                solver.exit()
            except Exception:
                pass


if __name__ == "__main__":
    main()
