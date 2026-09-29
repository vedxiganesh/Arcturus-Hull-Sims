#!/usr/bin/env python3
"""Build a sweep's manifest.csv: one row per (speed, theta, z) case.

Either a full grid:
  python manifest.py --topology topologies/at2_trimaran_halfd.json --sweep sweeps/pilot \
      --speeds 1,1.5,2,2.5 --theta -1,0,1 --z -0.01,0,0.01
or an explicit list (e.g. the next refinement from the equilibrium script):
  python manifest.py --topology ... --sweep sweeps/iter2 --points points.csv
where points.csv has columns speed_mps,theta_deg,z_m.

Rows are indexed 0..N-1; the index is the Slurm array task id.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import Topology, case_id, check_envelope, plan_run  # noqa: E402

FIELDS = ["index", "case_id", "speed_mps", "theta_deg", "z_m", "dt_s", "steps", "end_time_s"]


def floats(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def build_rows(topo: Topology, points: list[tuple[float, float, float]]) -> list[dict]:
    rows, seen = [], set()
    for V, th, z in points:
        cid = case_id(topo.name, V, th, z)
        if cid in seen:
            continue
        seen.add(cid)
        for msg in check_envelope(topo, th, z):
            print(f"WARNING {cid}: {msg}", file=sys.stderr)
        plan = plan_run(topo, V)
        rows.append({
            "index": len(rows), "case_id": cid, "speed_mps": V, "theta_deg": th, "z_m": z,
            "dt_s": f"{plan.dt_s:.6g}", "steps": plan.steps, "end_time_s": f"{plan.end_time_s:.4g}",
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--topology", required=True)
    ap.add_argument("--sweep", required=True)
    ap.add_argument("--speeds")
    ap.add_argument("--theta")
    ap.add_argument("--z")
    ap.add_argument("--points", help="CSV with speed_mps,theta_deg,z_m")
    ap.add_argument("--force", action="store_true", help="overwrite an existing manifest")
    args = ap.parse_args()

    topo = Topology.load(args.topology)
    if args.points:
        with open(args.points, newline="") as f:
            pts = [(float(r["speed_mps"]), float(r["theta_deg"]), float(r["z_m"])) for r in csv.DictReader(f)]
    else:
        if not (args.speeds and args.theta and args.z):
            ap.error("give --points, or all of --speeds --theta --z")
        pts = list(itertools.product(floats(args.speeds), floats(args.theta), floats(args.z)))

    sweep = Path(args.sweep)
    sweep.mkdir(parents=True, exist_ok=True)
    out = sweep / "manifest.csv"
    if out.exists() and not args.force:
        sys.exit(f"{out} exists; pass --force to overwrite (array indices would change)")
    rows = build_rows(topo, pts)
    with open(out, "w", newline="") as f:
        # "\n": the cluster reads this with awk; CRLF would leak into fields.
        w = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    total_steps = sum(int(r["steps"]) for r in rows)
    print(f"{out}: {len(rows)} cases, {total_steps} time steps total")


if __name__ == "__main__":
    main()
