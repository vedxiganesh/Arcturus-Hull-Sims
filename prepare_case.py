#!/usr/bin/env python3
"""Build a topology's captive template (LOCAL: Fluent 25R2, reefs-mobo env, normal terminal).

Turns the set-up 2DOF source case into a captive template at theta=0, z=0:
optional undo of the 6DOF motion already in the case, dynamic mesh off, UDF
unloaded, old convergence conditions removed, report-file paths made
relative. Measures the hull's lowest point and length and writes them into
topology.json.

Everything is found in the topology folder (layout.py):
  source     <name>*source*.cas.h5      (from `hs topology adopt`)
  reference  *foreground*.msh*          (optional position check)
  output     <name>_template.cas.h5

Usage:
  python prepare_case.py template --topology at2_trimaran_halfd       [--undo-theta-x DEG --undo-cg-current Y Z --undo-cg-initial Y Z]

Per-case preparation runs on the cluster, inside each case job
(fluent_ops.prepare_case via run_case.py).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fluent_ops as fo  # noqa: E402
import layout  # noqa: E402
from common import CaseTransform  # noqa: E402


# ---------------------------------------------------------------------------
# template
# ---------------------------------------------------------------------------


def cmd_template(args) -> None:
    topo = layout.load_topology(args.topology)
    tdir = layout.topology_dir(args.topology)
    src = layout.topology_file(args.topology, "source") if args.source is None else Path(args.source).resolve()
    dst = tdir / layout.canonical_name(topo.name, "template", ".cas.h5")
    others = [p for p in layout.find_all(tdir, layout.ROLE_PATTERNS["template"].format(name=topo.name))
              if p != dst]
    if others:
        raise SystemExit(f"other template files would make the template ambiguous: {others}")

    solver = fo.launch_solver(str(dst.parent), procs=args.procs)
    try:
        ref_stats = None
        ref_mesh = args.reference_mesh
        if ref_mesh is None:
            found = layout.topology_file(args.topology, "foreground_mesh", required=False)
            ref_mesh = str(found) if found else None
        if ref_mesh:
            fo.log(f"reading reference (pre-motion) mesh {ref_mesh}")
            try:
                solver.settings.file.read(file_type="mesh", file_name=str(Path(ref_mesh).resolve()))
                ref_stats = fo.hull_stats(solver, topo)
                fo.log(f"reference hull stats: {ref_stats}")
            except Exception as exc:
                fo.log(f"WARNING: reference mesh check SKIPPED, could not read/measure it ({exc})")

        fo.read_case(solver, str(src))
        moved = fo.hull_stats(solver, topo)
        fo.log(f"source hull stats: {moved}")

        if args.undo_theta_x is not None:
            cur = np.array([0.0, *args.undo_cg_current])
            init = np.array([0.0, *args.undo_cg_initial])
            # The 6DOF solver rotated about the moving CG and translated it
            # from init to cur. Undo: rotate back about cur, then translate
            # cur -> init.
            undo = CaseTransform(
                theta_deg=float("nan"), z_m=float("nan"),
                fluent_angle_deg=-float(args.undo_theta_x),
                origin=tuple(cur), axis=(1.0, 0.0, 0.0),
                translation=tuple(init - cur), moment_center=tuple(init),
            )
            route = fo.rotate_translate(solver, list(topo["foreground_cell_zones"]), undo)
            after = fo.hull_stats(solver, topo)
            err = fo.verify_transform(moved, after, undo)
            fo.log(f"motion undone via {route}; centroid check err={err:.2e} m")
            moved = after

        if ref_stats is not None:
            ref_c = np.asarray(ref_stats["centroid"])
            got_c = np.asarray(moved["centroid"])
            # The Fluent Meshing export is in mm; appending it into the solver
            # scaled it to m. A standalone read may not, so accept an exact
            # 1000x and say so rather than failing on units.
            if np.max(np.abs(ref_c / 1000.0 - got_c)) < np.max(np.abs(ref_c - got_c)):
                fo.log("reference mesh appears to be in mm; comparing at 1/1000 scale")
                ref_c = ref_c / 1000.0
            d = np.abs(got_c - ref_c)
            fo.log(f"template vs reference-mesh centroid diff: {d.tolist()} m")
            if float(d.max()) > args.ref_tol:
                raise RuntimeError(
                    f"template hull is {d.max():.4g} m from the pre-motion reference mesh; "
                    "the undo parameters are wrong (or the reference mesh units differ)."
                )

        fo.disable_dynamic_mesh(solver)
        fo.unload_udf(solver)
        fo.clear_convergence_conditions(solver)
        fo.relativize_report_files(solver)

        fo.write_case(solver, str(dst), data=False)
        topo.save_measured(
            hull_zmin_at_ref=moved["zmin"],
            hull_length_m=moved["length"],
            template_hull_stats=moved,
        )
        fo.log(f"template written: {dst}")
        fo.log(f"measured hull_zmin_at_ref={moved['zmin']:.5f} m, hull_length_m={moved['length']:.5f} m")
    finally:
        solver.exit()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("template")
    t.add_argument("--topology", required=True, help="topology name (folder under hullsweep_data)")
    t.add_argument("--source", help="override the <name>*source*.cas.h5 lookup")
    t.add_argument("--undo-theta-x", type=float, help="6DOF THETA_X [deg] recorded at the source case's time")
    t.add_argument("--undo-cg-current", type=float, nargs=2, metavar=("Y", "Z"))
    t.add_argument("--undo-cg-initial", type=float, nargs=2, metavar=("Y", "Z"),
                   help="CG the 6DOF zone STARTED from (as set in the source case, not cg_ref)")
    t.add_argument("--reference-mesh", help="override the *foreground*.msh* lookup")
    t.add_argument("--ref-tol", type=float, default=2e-3)
    t.add_argument("--procs", type=int, default=4)

    args = ap.parse_args()
    if args.cmd == "template" and args.undo_theta_x is not None:
        if args.undo_cg_current is None or args.undo_cg_initial is None:
            ap.error("--undo-theta-x needs --undo-cg-current and --undo-cg-initial")
    cmd_template(args)


if __name__ == "__main__":
    main()
