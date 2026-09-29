#!/usr/bin/env python3
"""hullsweep studies: the single CLI.

LOCAL (Arcturus/hullsweep, reefs-mobo env)
  hs topology adopt --name T --source X.cas.h5 [--foreground-mesh F] [--background-mesh B]
                    [--topology-json J] [--template TPL]
  python prepare_case.py template --topology T        (normal terminal: 25R2 licensing)
  hs sync push-code [--dry-run]
  hs sync push-topology T [--with-meshes] [--force]
  hs sync pull S [--with-data]

CLUSTER (~/hullsweep_code/bin/hs)
  hs new --topology T --study S --speeds 2.0,2.5 --x0 THETA,Z [options]
  hs status S
  hs stop S | hs resume S
  hs advance S                      (the orchestrator step; normally run by Slurm)
  hs list
  hs path case-dir --study S --case C

Everything but the topology and study NAMES is found by layout.py.
Monitor a running study with `tail -f` on the progress log `hs status` names.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import layout  # noqa: E402

#: From the `check` smoketest (2026-09-29): 21 cores, ~17 s/step. Estimates only.
REF_S_PER_STEP, REF_NTASKS = 17.0, 21


def floats(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x.strip()]


# ---------------------------------------------------------------------------
# topology (local)
# ---------------------------------------------------------------------------


def cmd_topology_adopt(a) -> None:
    """Gather a topology's inputs into hullsweep_data/<name>/ under canonical names."""
    if layout.site() != "local":
        sys.exit("topology adopt is a local step")
    name = layout.check_name(a.name, "topology")
    d = layout.topology_dir(name, must_exist=False)
    d.mkdir(parents=True, exist_ok=True)

    def place(src: str | None, role: str, suffix: str, move: bool) -> None:
        if not src:
            return
        s = Path(src).resolve()
        dst = d / layout.canonical_name(name, role, suffix)
        if dst.exists():
            print(f"  exists, kept: {dst.name}")
            return
        (shutil.move if move else shutil.copy2)(str(s), str(dst))
        print(f"  {'moved' if move else 'copied'} {s} -> {dst.name}")

    place(a.source, "source", ".cas.h5", move=False)
    place(a.foreground_mesh, "foreground_mesh", ".msh.h5", move=False)
    place(a.background_mesh, "background_mesh", ".msh.h5", move=False)
    place(a.template, "template", ".cas.h5", move=True)

    tj = d / layout.TOPOLOGY_FILE
    if a.topology_json and not tj.exists():
        raw = json.loads(Path(a.topology_json).read_text())
        raw["name"] = name
        for k in ("source_case", "reference_mesh", "template_case"):  # found by pattern now
            raw.pop(k, None)
        raw["_comment_template"] = ("template is <name>_template.cas.h5 in this folder, at theta=0, z=0 "
                                    "with dynamic mesh OFF, built by `prepare_case.py template` from "
                                    "<name>_source.cas.h5 (the set-up case saved BEFORE any motion).")
        tj.write_text(json.dumps(raw, indent=2) + "\n")
        print(f"  wrote {tj}")
    elif not tj.exists():
        sys.exit(f"{tj} missing: pass --topology-json to start from an existing one")
    cmd_topology_show(argparse.Namespace(name=name))


def cmd_topology_show(a) -> None:
    topo = layout.load_topology(a.name)
    print(f"topology {topo.name}: {layout.topology_dir(a.name)}")
    for role in layout.ROLE_PATTERNS:
        try:
            p = layout.topology_file(a.name, role, required=False)
            print(f"  {role:<16} {p.name if p else '-'}")
        except layout.LayoutError as exc:
            print(f"  {role:<16} ERROR {exc}")
    print(f"  template built   {'yes' if topo.get('template_hull_stats') else 'NO'}")


# ---------------------------------------------------------------------------
# sync (local)
# ---------------------------------------------------------------------------


def cmd_sync(a) -> None:
    import sync

    if a.what == "push-code":
        sync.push_code(dry_run=a.dry_run)
    elif a.what == "push-topology":
        sync.push_topology(a.name, with_meshes=a.with_meshes, force=a.force)
    elif a.what == "pull":
        sync.pull(a.name, with_data=a.with_data)


# ---------------------------------------------------------------------------
# new (cluster)
# ---------------------------------------------------------------------------


def build_study(a, topo) -> dict:
    import ledger

    speeds = floats(a.speeds)
    if not speeds or any(s <= 0 for s in speeds):
        sys.exit("--speeds must be positive, e.g. 2.0,2.5,3.0")
    mode = "grid" if (a.theta or a.z) else "newton"
    if mode == "grid" and not (a.theta and a.z):
        sys.exit("a grid study needs both --theta and --z")
    if mode == "newton" and not a.x0:
        sys.exit("a Newton study needs --x0 THETA_DEG,Z_M (the starting point)")
    x0 = floats(a.x0) if a.x0 else None
    if x0 is not None and len(x0) != 2:
        sys.exit("--x0 is THETA_DEG,Z_M")

    slurm = {"partition": a.partition, "ntasks": a.ntasks, "lanes": a.lanes, "mem": a.mem or "120G",
             "time": a.time, "advance_partition": a.advance_partition, "advance_time": a.advance_time}
    run_override, stop_margin = None, 1500
    if a.smoke:
        # quicktest caps memory per user (QOSMaxMemoryPerUser held 3 x 120G to one at a time,
        # 2026-09-29); the 6.8M-cell case should need ~15-25G on 8 cores
        slurm.update(partition="mit_quicktest", ntasks=8, time="00:14:00", lanes=min(a.lanes, 3),
                     mem=a.mem or "40G")
        run_override, stop_margin = {"steps": 10, "settle_time_s": 0.0}, 60
    need = ledger.license_need(slurm)
    if need > ledger.HPC_POOL and not a.ignore_license_limit:
        sys.exit(f"{slurm['lanes']} lanes x {slurm['ntasks']} cores needs {need} HPC licenses > pool "
                 f"{ledger.HPC_POOL}; lower --lanes/--ntasks")

    return {
        "name": a.study, "topology": topo.name, "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "mode": mode, "speeds": speeds, "x0": x0,
        "grid": {"theta": floats(a.theta), "z": floats(a.z)} if mode == "grid" else None,
        "tol_lift": a.tol_lift, "tol_pitch": a.tol_pitch, "drift_factor": a.drift_factor,
        "fd_theta": a.fd_theta, "fd_z": a.fd_z, "max_iters": a.max_iters,
        "cg_offset": list(a.cg_offset), "max_retries": a.max_retries,
        "run_override": run_override, "stop_margin_s": stop_margin, "slurm": slurm,
        "smoke": bool(a.smoke),
    }


def qos_check(study) -> list[str]:
    """Per-user QOS limits of the case and advance partitions, and what they mean for lanes."""
    import ledger

    sc = study["slurm"]
    try:
        slurm = ledger.Slurm()
        case_lim = slurm.qos_limits(sc["partition"])
        adv_lim = (case_lim if sc["advance_partition"] == sc["partition"]
                   else slurm.qos_limits(sc["advance_partition"]))
    except Exception as exc:  # no Slurm here (tests, local)
        return [f"  qos          not checked ({type(exc).__name__})"]
    lines = []
    for lim in (case_lim, adv_lim):
        if lim:
            lines.append(f"  qos          {lim['partition']}: QOS {lim.get('qos')}  per user: "
                         f"TRES {lim.get('max_tres_pu') or '-'}  jobs {lim.get('max_jobs_pu') or '-'}  "
                         f"submit {lim.get('max_submit_pu') or '-'}  wall {lim.get('max_wall') or lim.get('max_time')}")
        if case_lim is adv_lim:
            break
    extra = 1 if sc["advance_partition"] == sc["partition"] else 0
    eff, why = ledger.effective_lanes(case_lim, sc, extra_jobs=extra)
    if eff == 0:
        sys.exit("QOS limits leave room for no case job at all: " + "; ".join(why))
    if eff < sc["lanes"]:
        lines.append(f"  WARNING      only {eff} of {sc['lanes']} lanes can run at once: " + "; ".join(why))
    sub = case_lim.get("max_submit_pu", "")
    if sub.isdigit():
        lines.append(f"  note         at most {sub} queued+running jobs per user in {sc['partition']}; "
                     "cases sbatch refuses are retried by the next advance")
    return lines


def print_plan(sp, study, topo, seeds, skipped) -> None:
    import ledger
    from common import check_envelope

    sc = study["slurm"]
    print(f"\nSTUDY {study['name']}  ({study['mode']}{', SMOKE' if study['smoke'] else ''})")
    print(f"  topology     {topo.name}  ({topo.path})")
    print(f"  template     {study['sha1']['template']['name']}  sha1 {study['sha1']['template']['sha1'][:12]}")
    print(f"  code         {study['code_version'].get('content_sha1', '?')[:12]} "
          f"(commit {study['code_version'].get('commit', '?')}"
          f"{', dirty' if study['code_version'].get('dirty') else ''}, "
          f"pushed {study['code_version'].get('pushed_at', '?')})")
    print(f"  pool dir     {sp.pool_dir}")
    print(f"  scratch dir  {sp.scratch_dir}")
    print(f"  progress     {sp.progress_log}")
    print(f"  slurm        {sc['partition']}  {sc['ntasks']} cores  {sc['mem']}  {sc['time']}  "
          f"{sc['lanes']} lanes -> {ledger.license_need(sc)}/{ledger.HPC_POOL} HPC licenses")
    for s in study["speeds"]:
        p = ledger.case_plan(topo, study, s)
        hours = p["steps"] * REF_S_PER_STEP * REF_NTASKS / sc["ntasks"] / 3600
        print(f"  V={s:<5g}      dt {p['dt_s']:.4g} s  {p['steps']} steps  settle {p['settle_time_s']:.2f} s  "
              f"~{hours:.1f} h/case (estimate)")
    if study["mode"] == "newton":
        print(f"  x0           theta {study['x0'][0]:+g} deg, z {study['x0'][1] * 1000:+g} mm")
        for m in check_envelope(topo, *study["x0"]):
            print(f"  WARNING      x0 {m}")
        print(f"  tolerances   |R_lift| <= {study['tol_lift']} N, |R_pitch| <= {study['tol_pitch']} N*m "
              f"(domain weight {topo.weight_domain_N:.1f} N); window drift <= {study['drift_factor']} x tol")
        print(f"  stencil      {study['fd_theta']} deg, {study['fd_z'] * 1000:g} mm;  max {study['max_iters']} batches")
        print(f"  envelope     theta {topo['envelope']['theta_deg']} deg, z {topo['envelope']['z_m']} m")
    else:
        g = study["grid"]
        print(f"  grid         theta {g['theta']}  z {g['z']}  -> "
              f"{len(study['speeds']) * len(g['theta']) * len(g['z'])} cases")
    if study["cg_offset"] != [0.0, 0.0]:
        print(f"  cg offset    {study['cg_offset']} m (body frame)")
    if seeds or skipped:
        print(f"  seeds        {len(seeds)} imported from {study['seed']['file']}")
        for s in skipped:
            print(f"               skipped {s}")


def cmd_new(a) -> None:
    import ledger
    from common import file_sha1

    if layout.site() != "cluster":
        sys.exit("hs new runs on the cluster login node (studies live in ~/orcd/pool/hullsweep)")
    sp = layout.new_study(a.topology, a.study)
    topo = layout.load_topology(a.topology)
    if topo.get("template_hull_stats") is None:
        sys.exit(f"topology {topo.name} has no template_hull_stats: build the template first")
    template = layout.topology_file(a.topology, "template")
    study = build_study(a, topo)

    seeds, skipped = [], []
    if a.seed:
        if study["cg_offset"] != [0.0, 0.0]:
            sys.exit("--seed rows were reduced with cg_offset 0; not valid with --cg-offset")
        study["seed"] = {"file": str(Path(a.seed).resolve())}
        with open(a.seed, newline="") as f:
            seeds, skipped = ledger.seed_rows(topo, study, list(csv.DictReader(f)))
        study["seed"]["imported"] = [r["case_id"] for r in seeds]

    print(f"hashing {template.name} ...", flush=True)
    study["sha1"] = {"topology.json": file_sha1(topo.path),
                     "template": {"name": template.name, "sha1": file_sha1(template)}}
    study["code_version"] = ledger.code_version()
    print_plan(sp, study, topo, seeds, skipped)
    for ln in qos_check(study):
        print(ln)
    if a.dry_run:
        print("\n--dry-run: nothing written")
        return
    if not a.yes:
        got = input("\nType the study name to create it and submit: ").strip()
        if got != a.study:
            sys.exit("not confirmed; nothing written")
    ledger.init_study(sp, study, seeds)
    print(ledger.advance(sp))
    print(f"\nmonitor:  tail -f {sp.progress_log}\n          hs status {sp.name}")


# ---------------------------------------------------------------------------
# cluster-side study commands
# ---------------------------------------------------------------------------


def cmd_advance(a) -> None:
    import ledger

    print(ledger.advance(layout.find_study(a.study)))


def cmd_status(a) -> None:
    import ledger

    print(ledger.status_text(layout.find_study(a.study)))


def cmd_stop(a) -> None:
    import ledger

    ids = ledger.stop(layout.find_study(a.study))
    print(f"cancelled {len(ids)} job(s): {' '.join(ids)}")


def cmd_resume(a) -> None:
    import ledger

    print(ledger.resume(layout.find_study(a.study)))


def cmd_list(a) -> None:
    r = layout.roots()
    print(f"site {layout.site()}  pool {r.pool}  scratch {r.scratch}")
    for t in layout.list_topologies():
        studies = [s for tt, s in layout.list_studies() if tt == t]
        print(f"  {t}: {', '.join(studies) or '(no studies)'}")


def cmd_path(a) -> None:
    sp = layout.find_study(a.study)
    if a.what == "case-dir":
        if not a.case:
            sys.exit("case-dir needs --case")
        print(sp.case_dir(a.case))
    elif a.what == "pool":
        print(sp.pool_dir)
    elif a.what == "scratch":
        print(sp.scratch_dir)
    elif a.what == "progress":
        print(sp.progress_log)


# ---------------------------------------------------------------------------


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="hs", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("topology", help="local: gather / inspect a topology folder")
    tsub = t.add_subparsers(dest="action", required=True)
    ta = tsub.add_parser("adopt")
    ta.add_argument("--name", required=True)
    ta.add_argument("--source", help="set-up case BEFORE any motion (copied)")
    ta.add_argument("--foreground-mesh", help="copied")
    ta.add_argument("--background-mesh", help="copied")
    ta.add_argument("--template", help="an already built template (MOVED)")
    ta.add_argument("--topology-json", help="start topology.json from this file")
    ta.set_defaults(fn=cmd_topology_adopt)
    ts = tsub.add_parser("show")
    ts.add_argument("name")
    ts.set_defaults(fn=cmd_topology_show)

    s = sub.add_parser("sync", help="local: transfers to/from the cluster")
    ssub = s.add_subparsers(dest="what", required=True)
    sc = ssub.add_parser("push-code")
    sc.add_argument("--dry-run", action="store_true")
    sc.set_defaults(fn=cmd_sync)
    st = ssub.add_parser("push-topology")
    st.add_argument("name")
    st.add_argument("--with-meshes", action="store_true")
    st.add_argument("--force", action="store_true")
    st.set_defaults(fn=cmd_sync)
    sp_ = ssub.add_parser("pull")
    sp_.add_argument("name", help="study name")
    sp_.add_argument("--with-data", action="store_true", help="also the final cas/dat of each chain result")
    sp_.set_defaults(fn=cmd_sync)

    n = sub.add_parser("new", help="cluster: create a study and submit its first batch",
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    n.add_argument("--topology", required=True)
    n.add_argument("--study", required=True)
    n.add_argument("--speeds", required=True, help="comma list, m/s")
    n.add_argument("--x0", help="Newton start THETA_DEG,Z_M")
    n.add_argument("--theta", help="grid study: comma list of theta (deg)")
    n.add_argument("--z", help="grid study: comma list of z (m)")
    n.add_argument("--tol-lift", type=float, default=1.0, help="N, simulated domain")
    n.add_argument("--tol-pitch", type=float, default=0.25, help="N*m, simulated domain")
    n.add_argument("--drift-factor", type=float, default=2.0,
                   help="a converged point's window drift must be <= this x tol")
    n.add_argument("--fd-theta", type=float, default=1.0, help="stencil step, deg")
    n.add_argument("--fd-z", type=float, default=0.010, help="stencil step, m")
    n.add_argument("--max-iters", type=int, default=6, help="batches per chain")
    n.add_argument("--cg-offset", type=float, nargs=2, default=(0.0, 0.0), metavar=("DY", "DZ"))
    n.add_argument("--seed", help="results.csv of an earlier sweep/study to import as completed points")
    n.add_argument("--partition", default="mit_preemptable")
    n.add_argument("--ntasks", type=int, default=21)
    n.add_argument("--lanes", type=int, default=4, help="max concurrent case jobs")
    n.add_argument("--mem", help="per case job (default 120G; 40G with --smoke). "
                                  "Check maxRSS in `hs status` / progress.log and size down")
    n.add_argument("--time", default="06:00:00")
    n.add_argument("--advance-partition", default="mit_quicktest")
    n.add_argument("--advance-time", default="00:10:00")
    n.add_argument("--max-retries", type=int, default=3)
    n.add_argument("--ignore-license-limit", action="store_true")
    n.add_argument("--smoke", action="store_true",
                   help="10 steps per case on mit_quicktest (8 cores): tests the machinery, not physics")
    n.add_argument("--dry-run", action="store_true")
    n.add_argument("--yes", action="store_true", help="skip the type-the-name confirmation")
    n.set_defaults(fn=cmd_new)

    for name, fn, hlp in (("advance", cmd_advance, "orchestrator step (normally run by Slurm)"),
                          ("status", cmd_status, "chains and cases"),
                          ("stop", cmd_stop, "cancel every job of the study"),
                          ("resume", cmd_resume, "resubmit what `stop` cancelled and continue")):
        p = sub.add_parser(name, help=hlp)
        p.add_argument("study")
        p.set_defaults(fn=fn)

    ls = sub.add_parser("list", help="topologies and studies on this site")
    ls.set_defaults(fn=cmd_list)

    pa = sub.add_parser("path", help="resolved paths (used by cluster/case_job.sh)")
    pa.add_argument("what", choices=("case-dir", "pool", "scratch", "progress"))
    pa.add_argument("--study", required=True)
    pa.add_argument("--case")
    pa.set_defaults(fn=cmd_path)

    a = ap.parse_args(argv)
    try:
        a.fn(a)
    except layout.LayoutError as exc:
        sys.exit(f"hs: {exc}")


if __name__ == "__main__":
    main()
