#!/usr/bin/env python3
"""Generate the Slurm array script for a prepared sweep and print transfer commands.

  python make_batch.py --sweep sweeps/pilot --ntasks 21 --concurrent 4 --time 08:00:00

Writes into the sweep dir:
  submit_sweep.sh      array job, one task per manifest row
  run_sweep_case.py    the driver (copied, so the sweep is self-contained)
Checks that every manifest row has a prepared case + sidecar before emitting.

Licensing (account-wide, Shared Web), from cluster licdebug 2026-09-12: a
75-core job took 4 cores with the CFD task (HPC_PARALLEL 4/4) and 71
anshpc per-core licenses (12 + 36 + 23, cumulative 71/71). So each job
needs (ntasks - 4) anshpc, and concurrent jobs share a pool of ~71:
    concurrent * (ntasks - included) <= hpc_pool
e.g. 4 x 21, 3 x 27, 2 x 39. The pool is shared with anyone else on the
account; a task that cannot check out dies at launch ("Unexpected license
problem") and is NOT requeued -- resubmit those indices.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "cluster" / "submit_sweep.sh.in"
DRIVER = HERE / "cluster" / "run_sweep_case.py"


def _topology_name(sweep: Path, rows: list[dict]) -> str:
    """Topology of this sweep, from a prepared sidecar (else the case_id prefix)."""
    import json

    for r in rows:
        sc = sweep / "cases" / f"{r['case_id']}.json"
        if sc.exists():
            try:
                return json.loads(sc.read_text())["topology"]
            except (KeyError, ValueError):
                pass
    return rows[0]["case_id"].split("_V")[0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep", required=True)
    ap.add_argument("--partition", default="mit_preemptable")
    ap.add_argument("--ntasks", type=int, default=21)
    ap.add_argument("--mem", default="120G")
    ap.add_argument("--time", default="08:00:00",
                    help="ask for less than the partition cap to backfill")
    ap.add_argument("--concurrent", type=int, default=4)
    ap.add_argument("--hpc-pool", type=int, default=71, help="account anshpc per-core licenses")
    ap.add_argument("--included-cores", type=int, default=4, help="cores covered by the CFD task itself")
    ap.add_argument("--max-restarts", type=int, default=12,
                    help="walltime/preemption requeues per task before giving up")
    ap.add_argument("--ignore-license-limit", action="store_true")
    ap.add_argument("--export", action="store_true", help="enable sparse EnSight export")
    ap.add_argument("--remote", default="vxg@orcd-login.mit.edu")
    ap.add_argument("--allow-missing", action="store_true")
    args = ap.parse_args()

    need = args.concurrent * max(args.ntasks - args.included_cores, 0)
    if need > args.hpc_pool and not args.ignore_license_limit:
        sys.exit(f"{args.concurrent} x {args.ntasks} cores needs {need} HPC licenses > pool "
                 f"{args.hpc_pool}; lower --ntasks/--concurrent (or --ignore-license-limit)")

    sweep = Path(args.sweep).resolve()
    name = sweep.name
    with open(sweep / "manifest.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit("empty manifest")

    missing = []
    for r in rows:
        for ext in (".cas.h5", ".json"):
            if not (sweep / "cases" / f"{r['case_id']}{ext}").exists():
                missing.append(f"{r['case_id']}{ext}")
    if missing:
        print(f"{len(missing)} prepared files missing, e.g. {missing[:3]}", file=sys.stderr)
        if not args.allow_missing:
            sys.exit("run prepare_case.py prepare first (or --allow-missing)")

    text = TEMPLATE.read_text()
    for key, val in {
        "SWEEP": name, "PARTITION": args.partition, "NTASKS": args.ntasks,
        "MEM": args.mem, "TIME": args.time, "LAST": len(rows) - 1,
        "CONCURRENT": args.concurrent, "EXPORT": 1 if args.export else 0,
        "MAX_RESTARTS": args.max_restarts,
    }.items():
        text = text.replace(f"@{key}@", str(val))
    import re
    left = re.findall(r"@[A-Z_]+@", text)
    if left:
        sys.exit(f"unfilled placeholders: {sorted(set(left))}")

    out = sweep / "submit_sweep.sh"
    out.write_text(text, newline="\n")
    shutil.copy2(DRIVER, sweep / "run_sweep_case.py")
    # The driver imports these, so the case-only init path and its water-level
    # check use the same code as local prep.
    for mod in ("common.py", "fluent_ops.py"):
        shutil.copy2(HERE / mod, sweep / mod)
    topo_src = HERE / "topologies" / f"{_topology_name(sweep, rows)}.json"
    shutil.copy2(topo_src, sweep / "topology.json")

    remote_root = f"~/orcd/pool/hullsweep/{name}"
    print(f"wrote {out}  ({len(rows)} tasks, <= {args.concurrent} at once, {args.ntasks} cores each, "
          f"{need}/{args.hpc_pool} HPC licenses at full concurrency)")
    print("\nUpload (from the Arcturus dir, Git Bash):")
    print(f"  ssh {args.remote} 'mkdir -p {remote_root}/cases {remote_root}/logs'")
    support = " ".join(f"{sweep.as_posix()}/{f}" for f in
                       ("manifest.csv", "submit_sweep.sh", "run_sweep_case.py",
                        "common.py", "fluent_ops.py", "topology.json"))
    print(f"  scp {support} {args.remote}:{remote_root}/")
    print(f"  scp {sweep.as_posix()}/cases/* {args.remote}:{remote_root}/cases/")
    print("\nSubmit (on the login node):")
    print(f"  cd {remote_root} && sbatch submit_sweep.sh")
    print("\nFetch results back:")
    print(f"  scp -r {args.remote}:~/hull_results/hullsweep/{name}/ {sweep.as_posix()}/results/")


if __name__ == "__main__":
    main()
