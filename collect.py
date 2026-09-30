#!/usr/bin/env python3
"""Reduce case results to one table.

Studies reduce each case automatically (`hs advance`). To re-reduce a whole
study, e.g. with another CG offset:
  python collect.py --study <name> [--cg-offset DY DZ] [--out file.csv]
Legacy sweeps:
  python collect.py --topology <topology.json> --results sweeps/pilot/results

Expects one directory per case holding <case_id>.json (sidecar) and
sweep-forces.out (plus sweep-forces.part<k>.out from preempted attempts).
If sweep-forces.out has no flow-time column, flow time is taken by time step
from the case's *-rfile*.out files.

Per case, over the averaging window t >= settle_time_s:
  mean, standard error (batch means, 10 batches) and relative drift of
  every sweep report column, then, for the TOTAL group:

  drag_N        = Fy                      (hydrodynamic force, +Y = downstream)
  lift_N        = Fz
  Mbow_Nm       = pitch moment about the CG, POSITIVE BOW-UP
  thrust_N      = Fy / cos(theta)         (thrust along the body axis that
                                           balances drag)
  R_lift_N      = Fz + T sin(theta) - W   (W = weight of the simulated domain)
  R_pitch_Nm    = Mbow + Mbow_thrust      (bow-up positive; thrust line
                                           body-fixed, thrust_offset below keel)
  x_cl_m        = centre of lift ahead of the CG along the bow axis, taking
                  the whole moment as due to lift at CG height: -Mx/Fz * bow_y
  thrust_util   = full-boat thrust / thrust ceiling

All force/moment values are for the SIMULATED DOMAIN (half the boat when
half_domain is true); W, T and the ceiling are scaled to match.

--cg-offset DY DZ moves the CG in the BODY frame relative to cg_ref (m) and
transfers moments and the thrust arm accordingly: M' = M - d x F.

Closed-form numerics only (no lstsq/svd): LAPACK crashes this env.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _vendor.reefs_postproc import ReportSeries, _ols_slope, dedupe_restarts, parse_report_file  # noqa: E402
from common import (  # noqa: E402
    COMPONENTS,
    REPORT_FILE,
    Topology,
    case_transform,
    report_groups,
    report_name,
    rotation_about_axis,
)

N_BATCHES = 10


def load_series(case_dir: Path) -> ReportSeries | None:
    """Merge part files (oldest first) and the live file; later rows win."""
    parts = sorted(case_dir.glob(f"{REPORT_FILE}.part*.out"),
                   key=lambda p: int(p.name.split(".part")[1].split(".")[0]))
    live = case_dir / f"{REPORT_FILE}.out"
    files = parts + ([live] if live.exists() else [])
    if not files:
        return _load_per_definition(case_dir)
    series = [parse_report_file(p) for p in files]
    cols = series[0].columns
    for s in series[1:]:
        if s.columns != cols:
            raise ValueError(f"{case_dir.name}: column mismatch between report file parts")
    merged = ReportSeries(
        path=files[-1], title=series[-1].title, index_name=series[0].index_name,
        columns=cols, data=np.concatenate([s.data for s in series], axis=0),
        transient=series[0].transient, n_malformed_rows=sum(s.n_malformed_rows for s in series),
    )
    merged = dedupe_restarts(merged)
    return merged if merged.transient else _attach_flow_time(case_dir, merged)


def _attach_flow_time(case_dir: Path, series: ReportSeries) -> ReportSeries:
    """Add flow-time to a report file written without it.

    The grouped sweep report file can come out with only "Time Step". The
    case's per-definition *-rfile*.out files (e.g. report-lift-total-rfile.out)
    carry flow-time against the same step numbers, so map step -> time from
    them. Rows whose step has no flow-time anywhere are dropped. Returns the
    series unchanged (still non-transient) if no such file exists.
    """
    # parts first so the live file's rows win in dedupe_restarts
    files = sorted(case_dir.glob("*-rfile*.out"), key=lambda p: (".part" not in p.name, p.name))
    pairs = []
    for p in files:
        s = parse_report_file(p)
        if s.transient and s.index_name == series.index_name:
            pairs.append(np.column_stack([s.index, s.flow_time]))
    if not pairs:
        return series
    tmap = dedupe_restarts(ReportSeries(
        path=case_dir, title="flow-time map", index_name=series.index_name,
        columns=(series.index_name, "flow-time"), data=np.concatenate(pairs), transient=True))
    steps, times = tmap.index, tmap.flow_time

    idx = series.index
    pos = np.clip(np.searchsorted(steps, idx), 0, len(steps) - 1)
    hit = steps[pos] == idx
    return ReportSeries(
        path=series.path, title=series.title, index_name=series.index_name,
        columns=(*series.columns, "flow-time"),
        data=np.column_stack([series.data[hit], times[pos[hit]]]),
        transient=True, n_malformed_rows=series.n_malformed_rows + int((~hit).sum()),
    )


def _load_per_definition(case_dir: Path) -> ReportSeries | None:
    """Fallback layout: one sw-*-rfile.out per report definition."""
    files = sorted(case_dir.glob("sw-*-rfile*.out"))
    if not files:
        return None
    by_def: dict[str, list[ReportSeries]] = {}
    for p in files:
        s = parse_report_file(p)
        for c in s.value_columns:
            by_def.setdefault(c, []).append(s)
    base = None
    cols, data_cols = [], []
    for name, parts in by_def.items():
        merged = dedupe_restarts(ReportSeries(
            path=parts[-1].path, title=name, index_name=parts[0].index_name,
            columns=parts[0].columns, data=np.concatenate([p.data for p in parts]),
            transient=parts[0].transient))
        if base is None:
            base = merged
            cols = [merged.index_name, "flow-time"]
            data_cols = [merged.index, merged.flow_time]
        n = min(len(data_cols[0]), merged.n_rows)
        data_cols = [c[:n] for c in data_cols]
        cols.append(name)
        data_cols.append(merged.column(name)[:n])
    return ReportSeries(path=case_dir, title="per-definition", index_name=cols[0],
                        columns=tuple(cols), data=np.column_stack(data_cols), transient=True)


def window_stats(y: np.ndarray) -> tuple[float, float, float]:
    """(mean, batch-means standard error, relative drift over the window)."""
    n = y.shape[0]
    mean = float(np.mean(y)) if n else float("nan")
    if n < 2 * N_BATCHES:
        return mean, float("nan"), float("nan")
    b = n // N_BATCHES
    bm = y[: b * N_BATCHES].reshape(N_BATCHES, b).mean(axis=1)
    se = float(np.std(bm, ddof=1) / math.sqrt(N_BATCHES))
    drift = abs(_ols_slope(y) * n) / abs(mean) if mean != 0.0 else float("nan")
    return mean, se, drift


NOT_SIDECARS = ("status.json", "request.json")


def find_sidecar(case_dir: Path) -> Path | None:
    """<case_id>.json, i.e. the one named after the directory, else the only other .json."""
    own = case_dir / f"{case_dir.name}.json"
    if own.exists():
        return own
    others = [p for p in case_dir.glob("*.json") if p.name not in NOT_SIDECARS]
    return others[0] if len(others) == 1 else None


def reduce_case(topo: Topology, case_dir: Path, cg_offset: tuple[float, float]) -> dict | None:
    sidecar = find_sidecar(case_dir)
    if sidecar is None:
        return None
    sc = json.loads(sidecar.read_text())
    row = {k: sc[k] for k in ("case_id", "speed_mps", "theta_deg", "z_m", "dt_s", "steps")}
    status = case_dir / "status.json"
    row["status"] = json.loads(status.read_text()).get("state", "?") if status.exists() else "?"

    series = load_series(case_dir)
    if series is None:
        row["note"] = "no report file"
        return row
    if not series.transient:
        row["note"] = "no flow-time in report file and no *-rfile.out to take it from"
        return row
    t = series.flow_time
    w = t >= float(sc["settle_time_s"])
    row["t_last_s"] = float(t[-1])
    row["complete"] = bool(t[-1] >= 0.98 * float(sc["end_time_s"]))
    row["n_avg"] = int(w.sum())
    row["malformed_rows"] = series.n_malformed_rows

    means = {}
    for group in report_groups(topo):
        for kind in ("f", "m"):
            for c in COMPONENTS:
                name = report_name(kind, c, group)
                if not series.has_column(name):
                    continue
                m, se, dr = window_stats(series.column(name)[w])
                key = f"{kind.upper()}{c}_{group}"
                row[key], row[key + "_se"], row[key + "_drift"] = m, se, dr
                means[(kind, c, group)] = m

    try:
        F = np.array([means[("f", c, "total")] for c in COMPONENTS])
        M = np.array([means[("m", c, "total")] for c in COMPONENTS])
    except KeyError:
        row["note"] = "total force/moment columns missing"
        return row

    theta = float(sc["theta_deg"])
    xf = case_transform(topo, theta, float(sc["z_m"]))
    R = rotation_about_axis(xf.axis, xf.fluent_angle_deg)
    d_body = np.array([0.0, cg_offset[0], cg_offset[1]])
    d = R @ d_body
    M = M - np.cross(d, F)  # transfer to the offset CG

    sign = topo.fluent_pitch_sign  # bow-up moment = sign * Mx
    th = math.radians(theta)
    W = topo.weight_domain_N
    T = F[1] / math.cos(th)
    arm = topo.thrust_offset_body() - cg_offset[1]  # body 'up' offset of thrust line from CG
    Mx_thrust = arm * T  # (r x F)_x for r=(0,dy,arm), F=T*(0,-1,0) in the body frame
    # Standard errors of the residuals, components treated as independent
    # (the cg_offset transfer term is neglected).
    se = {k: row.get(f"{k}_total_se", float("nan")) for k in ("Fy", "Fz", "Mx")}
    row.update({
        "drag_N": F[1],
        "lift_N": F[2],
        "Mbow_Nm": sign * M[0],
        "thrust_N": T,
        "R_lift_N": F[2] + T * math.sin(th) - W,
        "R_pitch_Nm": sign * (M[0] + Mx_thrust),
        "R_lift_se": math.hypot(se["Fz"], math.tan(th) * se["Fy"]),
        "R_pitch_se": math.hypot(se["Mx"], arm / math.cos(th) * se["Fy"]),
        "x_cl_m": (-M[0] / F[2]) * topo.bow[1] if F[2] != 0 else float("nan"),
        "thrust_util": T / topo.thrust_ceiling_domain_N,
        "weight_domain_N": W,
        "thrust_arm_m": arm,
    })
    return row


# ---------------------------------------------------------------------------
# Free-running (2DOF) relaxation cases
# ---------------------------------------------------------------------------

MOTION_LOG = "sw-motion.csv"  # = fluent_ops.MOTION_LOG (collect stays free of fluent_ops)


def _monotone(rows: list[tuple]) -> list[tuple]:
    """Rows in file order; a restart (time going back) replaces the rows it re-runs."""
    out: list[tuple] = []
    for r in rows:
        if out and r[0] <= out[-1][0] + 1e-12:
            while out and out[-1][0] >= r[0] - 1e-12:
                out.pop()
        out.append(r)
    return out


def parse_motion_history(path: Path) -> list[tuple[float, float, float]]:
    """Fluent .6dof file -> [(time, CG_Z, THETA_X_deg)]. Columns: time, CG xyz, THETA xyz (deg)."""
    rows = []
    for ln in path.read_text(errors="replace").splitlines():
        parts = ln.split()
        if not parts or ln.lstrip().startswith("#") or len(parts) < 7:
            continue
        try:
            v = [float(x) for x in parts[:7]]
        except ValueError:
            continue
        rows.append((v[0], v[3], v[4]))
    return rows


def load_motion(case_dir: Path) -> tuple[np.ndarray | None, str]:
    """(rows [time, CG_Z, THETA_X_deg], source): the .6dof history (parts oldest first, then
    the live file) or run_case's sw-motion.csv, whichever reaches the later time."""
    def part_no(p: Path) -> int:
        return int(p.name.split(".part")[1].split(".")[0])

    live = sorted(p for p in case_dir.glob("*.6dof") if ".part" not in p.name)
    hist = []
    if live:
        stem = live[0].name[: -len(".6dof")]
        parts = sorted(case_dir.glob(f"{stem}.part*.6dof"), key=part_no)
        for p in [*parts, live[0]]:
            hist += parse_motion_history(p)
    hist = _monotone(hist)

    log_rows = []
    csv_path = case_dir / MOTION_LOG
    if csv_path.exists():
        with open(csv_path, newline="") as f:
            for r in csv.DictReader(f):
                try:
                    log_rows.append((float(r["flow_time"]), float(r["cg_z"]), float(r["theta_x_deg"])))
                except (TypeError, ValueError, KeyError):
                    continue
    log_rows = _monotone(log_rows)

    cands = [(rows, name) for rows, name in ((hist, live[0].name if live else ""), (log_rows, MOTION_LOG)) if rows]
    if not cands:
        return None, ""
    rows, name = max(cands, key=lambda c: c[0][-1][0] + (1e-9 if c[1] != MOTION_LOG else 0.0))
    return np.asarray(rows, dtype=float), name


def reduce_free_case(topo: Topology, case_dir: Path) -> dict | None:
    """Motion of a free case over its averaging window, in the captive conventions.

    theta = theta_release + sign * THETA_X   (the 6DOF orientation starts at 0 at release)
    z     = z_release + (CG_Z - CG_Z at release)
    Window: t >= t_release + settle_time_s. drift = the linear trend's change over the window.
    Also the mean total Fy, Fz over the window, when the report file has them. Moments are
    left out: their centre stays at the release CG while the hull moves.
    """
    sidecar = find_sidecar(case_dir)
    if sidecar is None:
        return None
    sc = json.loads(sidecar.read_text())
    rel = sc["release"]
    row = {"case_id": sc["case_id"], "kind": "free", "parent_case_id": sc.get("parent_case_id"),
           "speed_mps": sc["speed_mps"], "theta_deg": sc["theta_deg"], "z_m": sc["z_m"],
           "dt_s": sc["dt_s"], "steps": sc["steps"], "thrust_N": rel["thrust_N"]}
    status = case_dir / "status.json"
    row["status"] = json.loads(status.read_text()).get("state", "?") if status.exists() else "?"

    m, source = load_motion(case_dir)
    if m is None:
        row["note"] = "no 6DOF motion history (*.6dof) and no sw-motion.csv"
        return row
    t0 = float(sc["t_release_s"])
    t = m[:, 0]
    theta = float(sc["theta_deg"]) + topo.fluent_pitch_sign * m[:, 2]
    z = float(sc["z_m"]) + (m[:, 1] - float(rel["cg"][2]))
    w = t >= t0 + float(sc["settle_time_s"])
    row.update({"motion_source": source, "t_last_s": float(t[-1]), "n_avg": int(w.sum()),
                "complete": bool(t[-1] >= t0 + 0.98 * float(sc["end_time_s"])),
                "theta_final_deg": float(theta[-1]), "z_final_m": float(z[-1]),
                "theta_excursion_deg": float(np.max(np.abs(theta - float(sc["theta_deg"])))),
                "z_excursion_m": float(np.max(np.abs(z - float(sc["z_m"]))))})
    for key, y in (("theta", theta[w]), ("z", z[w])):
        mean, se, _ = window_stats(y)
        drift = float(_ols_slope(y) * y.shape[0]) if y.shape[0] >= 3 else float("nan")
        unit = "deg" if key == "theta" else "m"
        row[f"{key}_mean_{unit}"], row[f"{key}_se_{unit}"], row[f"{key}_drift_{unit}"] = mean, se, drift
    row["dtheta_deg"] = row["theta_mean_deg"] - float(sc["theta_deg"])
    row["dz_m"] = row["z_mean_m"] - float(sc["z_m"])

    try:
        series = load_series(case_dir)
    except Exception as exc:
        series, row["note"] = None, f"report file unreadable: {exc!r}"
    if series is not None and series.transient:
        wf = series.flow_time >= t0 + float(sc["settle_time_s"])
        for c in ("y", "z"):
            name = report_name("f", c, "total")
            if series.has_column(name) and wf.any():
                row[f"F{c}_total"] = float(np.mean(series.column(name)[wf]))
    return row


def write_rows(out: Path, rows: list[dict]) -> None:
    keys: list[str] = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    tmp = out.with_suffix(".tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    tmp.replace(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--study", help="study name; finds the topology and case dirs itself")
    ap.add_argument("--topology", help="topology.json (legacy sweeps only)")
    ap.add_argument("--results", help="dir with one sub-dir per case (legacy sweeps only)")
    ap.add_argument("--cg-offset", type=float, nargs=2, default=None, metavar=("DY", "DZ"),
                    help="default: the study's cg_offset, else 0 0")
    ap.add_argument("--out")
    args = ap.parse_args()

    if args.study:
        import layout

        sp = layout.find_study(args.study)
        topo = layout.load_topology(sp.topology)
        root = sp.scratch_dir
        study = json.loads(sp.study_json.read_text())
        cg = tuple(args.cg_offset) if args.cg_offset else tuple(study.get("cg_offset", (0.0, 0.0)))
        out = Path(args.out) if args.out else sp.pool_dir / "results_recollected.csv"
    elif args.topology and args.results:
        topo = Topology.load(args.topology)
        root = Path(args.results)
        cg = tuple(args.cg_offset) if args.cg_offset else (0.0, 0.0)
        out = Path(args.out) if args.out else root.parent / "results.csv"
    else:
        ap.error("give --study, or --topology and --results")

    rows = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        r = reduce_case(topo, d, cg)
        if r is not None:
            rows.append(r)
    if not rows:
        sys.exit(f"no case directories with sidecars under {root}")
    write_rows(out, rows)
    print(f"{out}: {len(rows)} cases")
    for r in rows:
        print(f"  {r['case_id']}: status={r.get('status')} complete={r.get('complete')} "
              f"R_lift={r.get('R_lift_N', float('nan')):.2f} N  R_pitch={r.get('R_pitch_Nm', float('nan')):.2f} N*m")


if __name__ == "__main__":
    main()
