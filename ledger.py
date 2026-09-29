"""Study state and the orchestrator step.

study.json   frozen inputs, written once by `hs new`
ledger.json  everything that changes: cases (job ids, state, reduced row) and
             chains (Newton state, history, result). Owned by advance().

advance() is idempotent and holds the ledger lock while it runs. It:
  1. resolves outstanding cases: reduce the complete ones, resubmit dead ones
     (up to max_retries), leave live ones alone;
  2. for each chain whose batch is fully resolved, asks newton.decide() for
     the next batch (or a terminal state) and submits it;
  3. re-arms itself: one `hs advance` job with afterany on every live case
     job. It checks liveness itself, so it does not matter whether afterany
     fires when a case job requeues.

Slurm is reached only through the Slurm class so tests can substitute it.
"""

from __future__ import annotations

import contextlib
import getpass
import json
import math
import os
import subprocess
import time
from pathlib import Path

import collect
import layout
import newton
import progress
from common import case_id, file_sha1, plan_run, quantize_point

#: Shared Web entitlement (CLAUDE.md section 12): ~71 anshpc usable, 4 cores ride on the CFD task.
HPC_POOL = 71
INCLUDED_CORES = 4

LIVE_CASE = "queued"  # submitted or waiting to be submitted
DONE_CASE = "done"
BAD_CASE = "bad"
SEED_CASE = "seed"


class SubmitError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Slurm
# ---------------------------------------------------------------------------


def scrubbed_env() -> dict:
    """Environment for sbatch run from inside a job: parent SLURM_* vars would
    become defaults for the child (e.g. SLURM_MEM_PER_CPU clashing with --mem)."""
    return {k: v for k, v in os.environ.items() if not k.startswith(("SLURM_", "SBATCH_", "SRUN_"))}


class Slurm:
    def __init__(self, run=subprocess.run):
        self._run = run

    def submit(self, opts: list[str], script: list[str]) -> str:
        cmd = ["sbatch", "--parsable", *opts, *script]
        r = self._run(cmd, capture_output=True, text=True, env=scrubbed_env())
        if r.returncode != 0:
            raise SubmitError(f"{' '.join(cmd)}\n{r.stderr.strip()}")
        return r.stdout.strip().split(";")[0]

    def active(self) -> dict[str, str]:
        """job id -> state for every job of this user still known to squeue."""
        user = os.environ.get("USER") or getpass.getuser()
        r = self._run(["squeue", "-h", "-u", user, "-o", "%i %T"], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"squeue failed: {r.stderr.strip()}")
        out = {}
        for ln in r.stdout.splitlines():
            parts = ln.split()
            if len(parts) >= 2:
                out[parts[0]] = parts[1]
        return out

    def cancel(self, ids: list[str]) -> None:
        if ids:
            self._run(["scancel", *ids], capture_output=True, text=True)


# ---------------------------------------------------------------------------
# JSON / lock
# ---------------------------------------------------------------------------


def read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return default


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str) + "\n")
    tmp.replace(path)


@contextlib.contextmanager
def ledger_lock(sp: layout.StudyPaths):
    """Yields True if this process holds the study lock (non-blocking)."""
    try:
        import fcntl
    except ImportError:  # Windows: tests only
        yield True
        return
    sp.lock.parent.mkdir(parents=True, exist_ok=True)
    with open(sp.lock, "a") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def code_version() -> dict:
    v = read_json(layout.CODE_DIR / "VERSION")
    return v if v else {"content_sha1": "unversioned", "note": "not pushed with hs sync push-code"}


# ---------------------------------------------------------------------------
# Study creation
# ---------------------------------------------------------------------------


def chain_key(speed: float) -> str:
    return f"V{speed:.2f}".replace(".", "p")


def license_need(slurm_cfg: dict) -> int:
    return slurm_cfg["lanes"] * max(slurm_cfg["ntasks"] - INCLUDED_CORES, 0)


def case_plan(topo, study: dict, speed: float) -> dict:
    """dt, steps and windows for one case, with the study's run override applied."""
    plan = plan_run(topo, speed)
    dt, steps, settle, end = plan.dt_s, plan.steps, plan.settle_time_s, plan.end_time_s
    ov = study.get("run_override") or {}
    if "steps" in ov:
        steps = int(ov["steps"])
        end = steps * dt
    if "settle_time_s" in ov:
        settle = float(ov["settle_time_s"])
    return {"dt_s": dt, "steps": steps, "settle_time_s": settle, "end_time_s": end}


def _num(v):
    if isinstance(v, str):
        if v in ("True", "False"):
            return v == "True"
        try:
            return float(v)
        except ValueError:
            return v
    return v


def seed_rows(topo, study: dict, rows: list[dict]) -> tuple[list[dict], list[str]]:
    """Filter imported results.csv rows to ones valid for this study; returns (kept, reasons)."""
    kept, skipped = [], []
    speeds = [float(s) for s in study["speeds"]]
    for raw in rows:
        r = {k: _num(v) for k, v in raw.items()}
        cid = str(r.get("case_id", ""))
        why = None
        if not cid.startswith(f"{topo.name}_V"):
            why = "other topology"
        elif not any(abs(float(r["speed_mps"]) - s) < 1e-9 for s in speeds):
            why = "speed not in study"
        elif r.get("complete") is not True:
            why = "incomplete run"
        else:
            p = case_plan(topo, study, float(r["speed_mps"]))
            if abs(float(r["dt_s"]) - p["dt_s"]) > 1e-12 or int(r["steps"]) != p["steps"]:
                why = f"run plan differs (dt {r['dt_s']} steps {r['steps']} vs {p['dt_s']:.6g} {p['steps']})"
        if why:
            skipped.append(f"{cid}: {why}")
        else:
            kept.append(r)
    return kept, skipped


def init_study(sp: layout.StudyPaths, study: dict, seeds: list[dict] | None = None) -> None:
    """Write study.json and the initial ledger. Refuses to touch an existing study."""
    if sp.study_json.exists():
        raise layout.LayoutError(f"{sp.study_json} exists")
    sp.pool_dir.mkdir(parents=True, exist_ok=False)
    sp.scratch_dir.mkdir(parents=True, exist_ok=True)
    sp.logs_dir.mkdir(parents=True, exist_ok=True)
    write_json(sp.study_json, study)

    led = {"cases": {}, "chains": {}, "lane_next": 0, "advance_job": None,
           "stopped": False, "done": False, "created": _now()}
    for r in seeds or []:
        cid = r["case_id"]
        led["cases"][cid] = {"speed": float(r["speed_mps"]), "theta": float(r["theta_deg"]),
                             "z": float(r["z_m"]), "chain": None, "iteration": -1, "role": "seed",
                             "state": SEED_CASE, "job": None, "jobs": [], "failures": 0,
                             "row": r, "note": f"imported from {study.get('seed', {}).get('file')}"}
    if study["mode"] == "grid":
        led["chains"]["grid"] = {"speed": None, "status": "active", "pending": [], "history": [],
                                 "state": None, "result": None}
    else:
        for s in study["speeds"]:
            st = newton.ChainState(x0=tuple(study["x0"]), rho=2.0)
            led["chains"][chain_key(s)] = {"speed": float(s), "status": "active", "pending": [],
                                           "history": [], "state": st.to_dict(), "result": None}
    write_json(sp.ledger_json, led)
    progress.emit(sp.progress_log, "study", sp.name, "CREATED",
                  f"topology {sp.topology}; mode {study['mode']}; speeds {study['speeds']}; "
                  f"{len(seeds or [])} seed points; code {study['code_version'].get('content_sha1', '?')[:10]}")


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _emit(sp, *a, **k):
    return progress.emit(sp.progress_log, *a, **k)


def _case_subject(sp, cid: str) -> str:
    return progress.short_case(cid, sp.topology)


def _case_tag(c: dict) -> str:
    return f"{progress.chain_tag(c['speed'])}/it{c['iteration']}"


def register_case(sp, study, led, topo, speed, theta, z, chain: str, iteration: int, role: str) -> str:
    th, zz = quantize_point(theta, z)
    cid = case_id(topo.name, speed, th, zz)
    if cid not in led["cases"]:
        led["cases"][cid] = {"speed": float(speed), "theta": th, "z": zz, "chain": chain,
                             "iteration": iteration, "role": role, "state": LIVE_CASE, "job": None,
                             "jobs": [], "failures": 0, "row": None, "note": ""}
    return cid


def submit_case(sp, study, led, topo, cid: str, slurm: Slurm) -> str:
    c = led["cases"][cid]
    plan = case_plan(topo, study, c["speed"])
    req = {"case_id": cid, "study": sp.name, "topology": sp.topology,
           "speed_mps": c["speed"], "theta_deg": c["theta"], "z_m": c["z"],
           "chain": c["chain"], "iteration": c["iteration"], "role": c["role"], **plan,
           "max_iter_per_step": topo["run"]["max_iter_per_step"],
           "stop_margin_s": study.get("stop_margin_s", 1500)}
    write_json(sp.case_dir(cid) / layout.REQUEST_FILE, req)
    logs = sp.case_log_dir(cid)
    logs.mkdir(parents=True, exist_ok=True)  # Slurm does not create it

    sc = study["slurm"]
    lane = led.get("lane_next", 0) % sc["lanes"]
    led["lane_next"] = lane + 1
    opts = ["--job-name", f"{sp.name}_lane{lane}", "--dependency=singleton",
            "--partition", sc["partition"], "--nodes=1", "--ntasks", str(sc["ntasks"]),
            "--cpus-per-task=1", "--mem", sc["mem"], "--time", sc["time"], "--requeue",
            "--open-mode=append",  # a requeued job keeps its id: keep every attempt's log
            "--output", str(logs / "%j.out"), "--error", str(logs / "%j.err")]
    script = [str(layout.CODE_DIR / "cluster" / "case_job.sh"), sp.name, cid]
    job = slurm.submit(opts, script)
    c["job"] = job
    c["jobs"].append(job)
    c["submitted"] = _now()
    _emit(sp, "case", _case_subject(sp, cid), "SUBMITTED",
          f"{_case_tag(c)} {c['role']}  job {job} lane {lane}  {plan['steps']} steps dt {plan['dt_s']:.4g}")
    return job


def quality_ok(row: dict, study: dict) -> bool:
    """Complete run whose window drift stays within drift_factor x tolerance."""
    if row.get("complete") is not True:
        return False
    f = float(study.get("drift_factor", 2.0))
    for key, tol in (("Fz_total", study["tol_lift"]), ("Mx_total", study["tol_pitch"])):
        m, d = row.get(key), row.get(f"{key}_drift")
        if not isinstance(m, (int, float)) or not isinstance(d, (int, float)):
            return False
        if not (math.isfinite(m) and math.isfinite(d)) or abs(d * m) > f * tol:
            return False
    return True


def _fmt_r(row: dict) -> str:
    def g(k):
        v = row.get(k)
        return float(v) if isinstance(v, (int, float)) else float("nan")

    return (f"R_lift {g('R_lift_N'):+.2f}±{g('R_lift_se'):.2f} N  "
            f"R_pitch {g('R_pitch_Nm'):+.3f}±{g('R_pitch_se'):.3f} N·m  "
            f"drift Fz {g('Fz_total_drift') * abs(g('Fz_total')):.2f} N Mx "
            f"{g('Mx_total_drift') * abs(g('Mx_total')):.3f} N·m")


def resolve_cases(sp, study, led, topo, slurm: Slurm, active: dict[str, str]) -> None:
    cg = tuple(study.get("cg_offset", (0.0, 0.0)))
    for cid, c in led["cases"].items():
        if c["state"] != LIVE_CASE:
            continue
        cdir = sp.case_dir(cid)
        st = read_json(cdir / layout.STATUS_FILE, {}) or {}
        if st.get("state") == "complete":
            try:
                row = collect.reduce_case(topo, cdir, cg)
            except Exception as exc:  # malformed output: do not rerun blindly
                row, c["note"] = None, f"reduce failed: {exc!r}"
            if row is None or "R_lift_N" not in row:
                c["state"] = BAD_CASE
                c["note"] = c["note"] or (row or {}).get("note", "no sidecar")
                _emit(sp, "case", _case_subject(sp, cid), "BAD", f"{_case_tag(c)} {c['note']}")
            else:
                c["row"], c["state"] = row, DONE_CASE
                ok = quality_ok(row, study)
                _emit(sp, "case", _case_subject(sp, cid), "REDUCED",
                      f"{_case_tag(c)} {_fmt_r(row)}{'' if ok else '  [quality: NOT ok]'}")
            continue
        job = c.get("job")
        if job and job in active:
            continue
        if job is None:  # new, or cancelled by `hs stop`
            submit_case(sp, study, led, topo, cid, slurm)
            continue
        c["failures"] += 1
        why = st.get("state") or "no status.json (died before the driver started)"
        if st.get("error"):
            why += f": {st['error'][:200]}"
        if c["failures"] <= int(study.get("max_retries", 3)):
            _emit(sp, "case", _case_subject(sp, cid), "RETRY",
                  f"{_case_tag(c)} job {job} ended without completing ({why}); "
                  f"attempt {c['failures'] + 1}")
            submit_case(sp, study, led, topo, cid, slurm)
        else:
            c["state"] = BAD_CASE
            c["note"] = f"gave up after {c['failures']} failures; last: {why}"
            _emit(sp, "case", _case_subject(sp, cid), "BAD", f"{_case_tag(c)} {c['note']}")


# ---------------------------------------------------------------------------
# Chains
# ---------------------------------------------------------------------------


def newton_config(study: dict, topo) -> newton.Config:
    env = topo["envelope"]
    return newton.Config(tol=(study["tol_lift"], study["tol_pitch"]),
                         fd=(study["fd_theta"], study["fd_z"]),
                         envelope=(tuple(env["theta_deg"]), tuple(env["z_m"])),
                         max_iters=int(study["max_iters"]))


def points_for(led: dict, speed: float, study: dict) -> list[newton.Point]:
    pts = []
    for cid, c in led["cases"].items():
        row = c.get("row")
        if not row or abs(c["speed"] - speed) > 1e-9:
            continue

        def f(k, alt=None):
            v = row.get(k, row.get(alt) if alt else None)
            return float(v) if isinstance(v, (int, float)) else float("nan")

        pts.append(newton.Point(theta=c["theta"], z=c["z"],
                                r=(f("R_lift_N"), f("R_pitch_Nm")),
                                se=(f("R_lift_se", "Fz_total_se"), f("R_pitch_se", "Mx_total_se")),
                                quality_ok=quality_ok(row, study), case_id=cid))
    return pts


def _pending_live(led, ch) -> bool:
    return any(led["cases"][c]["state"] == LIVE_CASE for c in ch["pending"])


def _fmt_pt(x) -> str:
    return f"θ={x[0]:+.2f}° z={x[1] * 1000:+.1f}mm"


def _fmt_decision(d: newton.Decision) -> str:
    parts = []
    if d.best is not None:
        parts.append(f"best {_fmt_pt(d.best.x)} R=({d.best.r[0]:+.2f} N, {d.best.r[1]:+.3f} N·m)")
    if d.J is not None:
        parts.append("J=[[{:.3g} N/°, {:.4g} N/m], [{:.3g} N·m/°, {:.4g} N·m/m]]".format(
            d.J[0][0], d.J[0][1], d.J[1][0], d.J[1][1]))
    if d.points:
        parts.append("-> " + ", ".join(_fmt_pt(p) for p in d.points))
    parts.append(f"rho={d.rho:g}")
    if d.note:
        parts.append(d.note)
    return "  ".join(parts)


def drive_chains(sp, study, led, topo, slurm: Slurm) -> None:
    cfg = newton_config(study, topo) if study["mode"] == "newton" else None
    for key, ch in led["chains"].items():
        if ch["status"] != "active" or _pending_live(led, ch):
            continue
        if study["mode"] == "grid":
            _drive_grid(sp, study, led, topo, slurm, ch)
            continue
        for _ in range(5):
            st = newton.ChainState.from_dict(ch["state"])
            d = newton.decide(points_for(led, ch["speed"], study), st, cfg)
            ch["state"] = d.state.to_dict()
            ch["history"].append({"at": _now(), "iteration": st.iteration, "kind": d.kind,
                                  "best": d.best.case_id if d.best else None, "J": d.J,
                                  "rho": d.rho, "points": d.points, "note": d.note})
            tag = f"{progress.chain_tag(ch['speed'])}/it{d.state.iteration}"
            if d.terminal:
                ch["status"] = d.kind
                ch["result"] = _chain_result(d)
                _emit(sp, "study", progress.chain_tag(ch["speed"]), d.kind, _fmt_decision(d))
                break
            _emit(sp, "newton", tag, d.kind.upper(), _fmt_decision(d))
            ch["pending"] = [register_case(sp, study, led, topo, ch["speed"], th, z, key,
                                           d.state.iteration, d.kind) for th, z in d.points]
            if _pending_live(led, ch):
                break
        if ch["status"] == "active" and not _pending_live(led, ch):
            ch["status"] = newton.FAILED
            _emit(sp, "study", progress.chain_tag(ch["speed"]), newton.FAILED,
                  "decisions only proposed points that are already finished or bad")


def _chain_result(d: newton.Decision) -> dict | None:
    if d.best is None:
        return {"note": d.note}
    return {"case_id": d.best.case_id, "theta_deg": d.best.theta, "z_m": d.best.z,
            "R_lift_N": d.best.r[0], "R_pitch_Nm": d.best.r[1], "J": d.J, "note": d.note}


def _drive_grid(sp, study, led, topo, slurm, ch) -> None:
    if not ch["pending"]:
        g = study["grid"]
        ch["pending"] = [register_case(sp, study, led, topo, s, th, z, "grid", 0, "grid")
                         for s in study["speeds"] for th in g["theta"] for z in g["z"]]
        _emit(sp, "study", "grid", "SUBMIT", f"{len(ch['pending'])} cases")
        return
    ch["status"] = "DONE"
    _emit(sp, "study", "grid", "DONE",
          f"{sum(led['cases'][c]['state'] == DONE_CASE for c in ch['pending'])}/{len(ch['pending'])} cases reduced")


# ---------------------------------------------------------------------------
# Orchestrator step
# ---------------------------------------------------------------------------


def submit_advance(sp, study, slurm: Slurm, deps: list[str]) -> str:
    sp.advance_log_dir.mkdir(parents=True, exist_ok=True)
    sc = study["slurm"]
    base = ["--job-name", f"{sp.name}_advance", "--partition", sc["advance_partition"],
            "--nodes=1", "--ntasks=1", "--mem=4G", "--time", sc["advance_time"],
            "--output", str(sp.advance_log_dir / "%j.out")]
    script = [str(layout.CODE_DIR / "bin" / "hs"), "advance", sp.name]
    try:
        return slurm.submit([*base, "--dependency", "afterany:" + ":".join(deps)], script)
    except SubmitError as exc:
        # e.g. a dependency id already purged: poll instead of waiting on it
        _emit(sp, "study", sp.name, "WARNING", f"advance with dependency refused ({exc}); polling in 15 min")
        return slurm.submit([*base, "--begin=now+15minutes"], script)


def write_results(sp, led) -> None:
    rows = []
    for cid, c in led["cases"].items():
        if c.get("row"):
            rows.append({"case_id": cid, "chain": c["chain"], "iteration": c["iteration"],
                         "role": c["role"], "state": c["state"],
                         **{k: v for k, v in c["row"].items() if k != "case_id"}})
    if rows:
        collect.write_rows(sp.results_csv, rows)


def advance(sp: layout.StudyPaths, slurm: Slurm | None = None) -> str:
    slurm = slurm or Slurm()
    with ledger_lock(sp) as got:
        if not got:
            print("another advance holds the lock; nothing to do")
            return "locked"
        study = read_json(sp.study_json)
        led = read_json(sp.ledger_json)
        if led.get("stopped"):
            print("study is stopped; `hs resume` to continue")
            return "stopped"
        if led.get("done"):
            print("study is done")
            return "done"
        tj = layout.topology_dir(sp.topology) / layout.TOPOLOGY_FILE
        if file_sha1(tj) != study["sha1"]["topology.json"]:
            _emit(sp, "study", sp.name, "ERROR", f"{tj} changed since the study was created; not advancing")
            return "error"
        topo = layout.load_topology(sp.topology)

        active = slurm.active()
        try:
            resolve_cases(sp, study, led, topo, slurm, active)
            drive_chains(sp, study, led, topo, slurm)
            for cid, c in led["cases"].items():
                if c["state"] == LIVE_CASE and c["job"] is None:
                    submit_case(sp, study, led, topo, cid, slurm)
        finally:
            write_json(sp.ledger_json, led)
            write_results(sp, led)

        own = os.environ.get("SLURM_JOB_ID")
        prev = led.get("advance_job")
        if prev and prev != own and prev in active:
            slurm.cancel([prev])  # superseded: this run re-arms below
        live = [c["job"] for c in led["cases"].values() if c["state"] == LIVE_CASE and c["job"]]
        if live:
            led["advance_job"] = submit_advance(sp, study, slurm, live)
            outcome = f"waiting on {len(live)} case job(s); next advance {led['advance_job']}"
        else:
            led["advance_job"] = None
            if all(ch["status"] != "active" for ch in led["chains"].values()):
                led["done"] = True
                _emit(sp, "study", sp.name, "DONE", summary_line(led))
                outcome = "done"
            else:
                outcome = "idle (no live jobs, chains still active)"
                _emit(sp, "study", sp.name, "WARNING", outcome)
        write_json(sp.ledger_json, led)
        return outcome


def summary_line(led) -> str:
    parts = []
    for ch in led["chains"].values():
        res = ch.get("result") or {}
        if "theta_deg" in res:
            parts.append(f"{progress.chain_tag(ch['speed'])} {ch['status']} {_fmt_pt((res['theta_deg'], res['z_m']))}")
        else:
            parts.append(f"{ch['speed']} {ch['status']}")
    return "; ".join(parts)


# ---------------------------------------------------------------------------
# stop / resume / status
# ---------------------------------------------------------------------------


def stop(sp, slurm: Slurm | None = None) -> list[str]:
    slurm = slurm or Slurm()
    with ledger_lock(sp) as got:
        if not got:
            raise RuntimeError("an advance is running; retry in a minute")
        led = read_json(sp.ledger_json)
        ids = [c["job"] for c in led["cases"].values() if c["state"] == LIVE_CASE and c["job"]]
        if led.get("advance_job"):
            ids.append(led["advance_job"])
        slurm.cancel(ids)
        for c in led["cases"].values():
            if c["state"] == LIVE_CASE:
                c["job"] = None  # resubmitted on resume without counting as a failure
        led["stopped"], led["advance_job"] = True, None
        write_json(sp.ledger_json, led)
    _emit(sp, "study", sp.name, "STOPPED", f"cancelled {len(ids)} job(s)")
    return ids


def resume(sp, slurm: Slurm | None = None) -> str:
    with ledger_lock(sp) as got:
        if not got:
            raise RuntimeError("an advance is running; retry in a minute")
        led = read_json(sp.ledger_json)
        led["stopped"] = False
        write_json(sp.ledger_json, led)
    _emit(sp, "study", sp.name, "RESUMED")
    return advance(sp, slurm)


def status_text(sp) -> str:
    study = read_json(sp.study_json)
    led = read_json(sp.ledger_json)
    out = [f"study {sp.name}  topology {sp.topology}  mode {study['mode']}  "
           f"{'STOPPED ' if led.get('stopped') else ''}{'DONE ' if led.get('done') else ''}"
           f"advance job {led.get('advance_job')}",
           f"progress log: {sp.progress_log}", ""]
    out.append(f"{'chain':<8} {'status':<17} {'batches':>7} {'rho':>5}  best")
    for key, ch in led["chains"].items():
        st = ch.get("state") or {}
        best = ch.get("result") or {}
        if not best and ch.get("history"):
            b = ch["history"][-1].get("best")
            best = {"case_id": b} if b else {}
        out.append(f"{key:<8} {ch['status']:<17} {st.get('iteration', '-'):>7} "
                   f"{st.get('rho', float('nan')):>5.2f}  {best.get('case_id', '-')}")
    out.append("")
    out.append(f"{'case':<30} {'it':>3} {'role':<8} {'state':<7} {'job':>9} {'fail':>4}  "
               f"{'step':>11} {'s/step':>6} {'ETA':>7}  R_lift / R_pitch")
    for cid, c in sorted(led["cases"].items(), key=lambda kv: (kv[1]["iteration"], kv[0])):
        s = read_json(sp.case_dir(cid) / layout.STATUS_FILE, {}) or {}
        step = f"{s.get('time_step', '-')}/{s.get('total_steps', '-')}" if s else "-"
        sps = s.get("s_per_step")
        eta = s.get("eta_s")
        r = c.get("row") or {}
        rs = (f"{r['R_lift_N']:+.2f} / {r['R_pitch_Nm']:+.3f}"
              if isinstance(r.get("R_lift_N"), float) else (c.get("note") or "")[:40])
        out.append(f"{progress.short_case(cid, sp.topology):<30} {c['iteration']:>3} {c['role']:<8} "
                   f"{c['state']:<7} {str(c.get('job') or '-'):>9} {c['failures']:>4}  {step:>11} "
                   f"{(f'{sps:.1f}' if sps else '-'):>6} {progress.fmt_hours(eta) if eta else '-':>7}  {rs}")
    return "\n".join(out)
