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

Free-running relaxation (study["free"] set): a chain whose Newton result is in
free.relax_on (default CONVERGED) does not end there. It enters phase "free"
and runs ONE free case: the 2DOF hull released from its best captive case's
converged solution. The chain then ends with a verdict (free_verdict):
VERIFIED, DRIFTED, UNSETTLED or FREE_FAILED. A mode "free" study (`hs new
--from-study P`) runs only this phase, on chains P has already converged.

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
from common import case_id, file_sha1, free_case_id, free_release, plan_free_run, plan_run, quantize_point

#: Shared Web entitlement (CLAUDE.md section 12): ~71 anshpc usable, 4 cores ride on the CFD task.
HPC_POOL = 71
INCLUDED_CORES = 4

LIVE_CASE = "queued"  # submitted or waiting to be submitted
DONE_CASE = "done"
BAD_CASE = "bad"
SEED_CASE = "seed"

#: Terminal chain states after the free-running phase.
VERIFIED = "VERIFIED"  # settled within tolerance of the captive equilibrium
DRIFTED = "DRIFTED"  # settled, but somewhere else
UNSETTLED = "UNSETTLED"  # still moving over the averaging window
FREE_FAILED = "FREE_FAILED"  # the free case did not produce a motion history


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

    def max_rss_gb(self, job: str) -> float | None:
        """Peak memory of a finished job (largest step), from sacct."""
        r = self._run(["sacct", "-j", job, "-n", "-P", "-o", "MaxRSS"], capture_output=True, text=True)
        if r.returncode != 0:
            return None
        vals = [parse_mem_gb(v) for v in r.stdout.split()]
        vals = [v for v in vals if v is not None]
        return max(vals) if vals else None

    def qos_limits(self, partition: str) -> dict:
        """Per-user limits of the partition's QOS (best effort; {} if unavailable)."""
        r = self._run(["scontrol", "show", "partition", partition, "--oneliner"],
                      capture_output=True, text=True)
        if r.returncode != 0:
            return {}
        fields = dict(kv.split("=", 1) for kv in r.stdout.split() if "=" in kv)
        out = {"partition": partition, "qos": fields.get("QoS", "N/A"),
               "max_time": fields.get("MaxTime"), "max_mem_per_node": fields.get("MaxMemPerNode")}
        if out["qos"] in ("N/A", "", None):
            return out
        r = self._run(["sacctmgr", "-n", "-P", "show", "qos", out["qos"],
                       "format=MaxTRESPU,MaxJobsPU,MaxSubmitPU,MaxWall"], capture_output=True, text=True)
        if r.returncode == 0 and r.stdout.strip():
            tres, jobs, submit, wall = (r.stdout.strip().splitlines()[0].split("|") + ["", "", "", ""])[:4]
            out.update(max_tres_pu=tres, max_jobs_pu=jobs, max_submit_pu=submit, max_wall=wall)
        return out


def parse_mem_gb(s: str | None) -> float | None:
    """'120G', '4096M', '32679208K', '1T' -> GB (Slurm units are binary)."""
    if not s:
        return None
    s = s.strip().upper()
    mult = {"K": 1 / 1024 ** 2, "M": 1 / 1024, "G": 1.0, "T": 1024.0}
    try:
        if s[-1] in mult:
            return float(s[:-1]) * mult[s[-1]]
        return float(s) / 1024 ** 3  # bare bytes
    except ValueError:
        return None


def parse_tres(s: str | None) -> dict[str, str]:
    """'cpu=192,mem=256G' -> {'cpu': '192', 'mem': '256G'}."""
    return dict(kv.split("=", 1) for kv in (s or "").split(",") if "=" in kv)


def effective_lanes(limits: dict, slurm_cfg: dict, extra_jobs: int = 0) -> tuple[int, list[str]]:
    """How many case jobs the QOS lets run at once, and why it is fewer than the lanes."""
    lanes, why = slurm_cfg["lanes"], []
    tres = parse_tres(limits.get("max_tres_pu"))
    mem_cap, mem = parse_mem_gb(tres.get("mem")), parse_mem_gb(slurm_cfg["mem"])
    if mem_cap and mem and int(mem_cap // mem) < lanes:
        lanes = int(mem_cap // mem)
        why.append(f"QOS {limits['qos']} caps memory at {tres['mem']} per user = {lanes} x {slurm_cfg['mem']}")
    if tres.get("cpu", "").isdigit() and int(tres["cpu"]) // slurm_cfg["ntasks"] < lanes:
        lanes = int(tres["cpu"]) // slurm_cfg["ntasks"]
        why.append(f"QOS {limits['qos']} caps cpus at {tres['cpu']} per user")
    mj = limits.get("max_jobs_pu", "")
    if mj.isdigit() and int(mj) - extra_jobs < lanes:
        lanes = int(mj) - extra_jobs
        why.append(f"QOS {limits['qos']} allows {mj} running jobs per user")
    return max(lanes, 0), why


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


def free_plan(topo, study: dict, speed: float) -> dict:
    """dt, steps and windows of a free case (times from release), run override applied."""
    fc = study["free"]
    p = plan_free_run(topo, speed, fc["settle_hull_lengths"], fc["average_hull_lengths"])
    dt, steps, settle, end = p.dt_s, p.steps, p.settle_time_s, p.end_time_s
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


def free_sources(parent: layout.StudyPaths, relax_on: list[str], speeds: list[float] | None = None,
                 cases: list[str] | None = None) -> tuple[list[dict], list[str]]:
    """Captive cases of `parent` to release, and problems that block them.

    Default: the best case of every chain whose Newton status is in relax_on. `cases`
    picks finished captive cases by id instead (e.g. the best of a MAX_ITERS chain).
    Each needs its <id>_final.cas.h5 + .dat.h5 in the parent's case dir.
    """
    led = read_json(parent.ledger_json)
    picks, problems = [], []
    if cases:
        for cid in cases:
            c = led["cases"].get(cid)
            if c is None or c.get("kind") == "free" or c["state"] != DONE_CASE:
                problems.append(f"{cid}: not a finished captive case of {parent.name}"
                                f"{'' if c is None else ' (state ' + c['state'] + ')'}")
                continue
            picks.append((cid, c))
    else:
        for key, ch in led["chains"].items():
            status = ch.get("newton_status", ch["status"])
            cid = (ch.get("result") or {}).get("case_id")
            if status in relax_on and cid in led["cases"]:
                picks.append((cid, led["cases"][cid]))
            else:
                problems.append(f"chain {key}: {status} (not in {relax_on}); skipped")
    out, keys = [], set()
    for cid, c in picks:
        if speeds and not any(abs(c["speed"] - s) < 1e-9 for s in speeds):
            continue
        cas, dat = parent_final({"dir": str(parent.case_dir(cid)), "case_id": cid})
        missing = [p.name for p in (cas, dat) if not p.is_file()]
        if missing:
            problems.append(f"{cid}: missing {', '.join(missing)} in {cas.parent}")
            continue
        key = chain_key(c["speed"])
        while key in keys:
            key += "b"
        keys.add(key)
        out.append({"key": key, "speed": float(c["speed"]), "study": parent.name, "case_id": cid,
                    "theta": float(c["theta"]), "z": float(c["z"]),
                    "row": {k: c["row"].get(k) for k in SOURCE_ROW_KEYS}})
    return out, problems


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
    elif study["mode"] == "free":
        for src in study["sources"]:
            led["chains"][src["key"]] = {"speed": float(src["speed"]), "status": "active", "phase": "free",
                                         "pending": [], "history": [], "state": None, "source": src,
                                         "result": {"case_id": src["case_id"], "theta_deg": src["theta"],
                                                    "z_m": src["z"], "note": f"from study {src['study']}"}}
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
    return f"{progress.chain_tag(c['speed'])}/{'free' if c.get('kind') == 'free' else 'it' + str(c['iteration'])}"


def register_case(sp, study, led, topo, speed, theta, z, chain: str, iteration: int, role: str) -> str:
    th, zz = quantize_point(theta, z)
    cid = case_id(topo.name, speed, th, zz)
    if cid not in led["cases"]:
        led["cases"][cid] = {"speed": float(speed), "theta": th, "z": zz, "chain": chain,
                             "iteration": iteration, "role": role, "state": LIVE_CASE, "job": None,
                             "jobs": [], "failures": 0, "row": None, "note": ""}
    return cid


def register_free_case(sp, led, chain: str, src: dict, iteration: int) -> str:
    """The free case released from src (a finished captive case of this or another study)."""
    cid = free_case_id(src["case_id"])
    if cid not in led["cases"]:
        parent_sp = sp if src["study"] == sp.name else layout.find_study(src["study"])
        led["cases"][cid] = {"speed": float(src["speed"]), "theta": float(src["theta"]), "z": float(src["z"]),
                             "chain": chain, "iteration": iteration, "role": "free", "kind": "free",
                             "parent": {"study": src["study"], "case_id": src["case_id"],
                                        "dir": str(parent_sp.case_dir(src["case_id"])), "row": src["row"]},
                             "state": LIVE_CASE, "job": None, "jobs": [], "failures": 0, "row": None, "note": ""}
    return cid


def parent_final(parent: dict) -> tuple[Path, Path]:
    d = Path(parent["dir"])
    return d / f"{parent['case_id']}_final.cas.h5", d / f"{parent['case_id']}_final.dat.h5"


def free_request(sp, study, topo, cid: str, c: dict) -> dict:
    """request.json of a free case: the release (6DOF state and UDF loads) and parent files."""
    par, fc = c["parent"], study["free"]
    thrust = fc["thrust"] == "constant"
    if thrust and not all(isinstance(par["row"].get(k), (int, float)) for k in ("thrust_N", "thrust_arm_m")):
        raise ValueError(f"{cid}: parent {par['case_id']} row has no thrust_N/thrust_arm_m for --thrust constant")
    rel = free_release(topo, c["theta"], c["z"], study.get("cg_offset", (0.0, 0.0)),
                       thrust_N=float(par["row"]["thrust_N"]) if thrust else 0.0,
                       thrust_arm_m=float(par["row"]["thrust_arm_m"]) if thrust else 0.0,
                       inertia_full=fc["inertia_full_kgm2"])
    cas, dat = parent_final(par)
    side = read_json(Path(par["dir"]) / f"{par['case_id']}.json", {}) or {}
    return {"kind": "free", "parent_study": par["study"], "parent_case_id": par["case_id"],
            "parent_cas": str(cas), "parent_dat": str(dat),
            "parent_hull_centroid": (side.get("hull_after") or {}).get("centroid"),
            "release": rel.to_dict(), "export_every_s": fc["export_every_s"],
            "implicit_update": fc.get("implicit_update"),
            **free_plan(topo, study, c["speed"])}


def submit_case(sp, study, led, topo, cid: str, slurm: Slurm) -> str | None:
    c = led["cases"][cid]
    plan = case_plan(topo, study, c["speed"])
    req = {"case_id": cid, "study": sp.name, "topology": sp.topology,
           "speed_mps": c["speed"], "theta_deg": c["theta"], "z_m": c["z"],
           "chain": c["chain"], "iteration": c["iteration"], "role": c["role"], **plan,
           "max_iter_per_step": topo["run"]["max_iter_per_step"],
           "stop_margin_s": study.get("stop_margin_s", 1500)}
    if c.get("kind") == "free":
        req.update(free_request(sp, study, topo, cid, c))
        plan = {k: req[k] for k in plan}
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
    try:
        job = slurm.submit(opts, script)
    except SubmitError as exc:  # e.g. a QOS submit limit: the next advance tries again
        _emit(sp, "case", _case_subject(sp, cid), "UNSUBMITTED", f"{_case_tag(c)} sbatch refused: "
              f"{str(exc).splitlines()[-1][:200]}")
        return None
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


def free_verdict(row: dict | None, fc: dict) -> tuple[str, str]:
    """VERIFIED / DRIFTED / UNSETTLED / FREE_FAILED for a reduced free case, and why."""
    def g(k):
        v = (row or {}).get(k)
        return float(v) if isinstance(v, (int, float)) else float("nan")

    if not row or not math.isfinite(g("theta_mean_deg")):
        return FREE_FAILED, (row or {}).get("note", "not reduced")
    if row.get("complete") is not True:
        return FREE_FAILED, f"motion history ends at t={g('t_last_s'):.3f} s, before the planned end"
    tt, tz = float(fc["tol_theta_deg"]), float(fc["tol_z_m"])
    txt = (f"dtheta {g('dtheta_deg'):+.3f} deg (drift {g('theta_drift_deg'):+.3f}), "
           f"dz {g('dz_m') * 1000:+.2f} mm (drift {g('z_drift_m') * 1000:+.2f}); tol {tt} deg, {tz * 1000:g} mm")
    if not (abs(g("theta_drift_deg")) <= tt and abs(g("z_drift_m")) <= tz):
        return UNSETTLED, txt
    if abs(g("dtheta_deg")) <= tt and abs(g("dz_m")) <= tz:
        return VERIFIED, txt
    return DRIFTED, txt


def _fmt_free(row: dict) -> str:
    def g(k):
        v = row.get(k)
        return float(v) if isinstance(v, (int, float)) else float("nan")

    return (f"theta {g('theta_mean_deg'):+.3f}±{g('theta_se_deg'):.3f} deg ({g('dtheta_deg'):+.3f})  "
            f"z {g('z_mean_m') * 1000:+.2f}±{g('z_se_m') * 1000:.2f} mm ({g('dz_m') * 1000:+.2f})  "
            f"from {row.get('motion_source', '?')}")


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
            _record_mem(c, slurm, c.get("job"))
            free = c.get("kind") == "free"
            try:
                row = collect.reduce_free_case(topo, cdir) if free else collect.reduce_case(topo, cdir, cg)
            except Exception as exc:  # malformed output: do not rerun blindly
                row, c["note"] = None, f"reduce failed: {exc!r}"
            if row is None or ("theta_mean_deg" if free else "R_lift_N") not in row:
                c["state"] = BAD_CASE
                c["note"] = c["note"] or (row or {}).get("note", "no sidecar")
                _emit(sp, "case", _case_subject(sp, cid), "BAD", f"{_case_tag(c)} {c['note']}")
            elif free:
                c["row"], c["state"] = row, DONE_CASE
                _emit(sp, "case", _case_subject(sp, cid), "REDUCED", f"{_case_tag(c)} {_fmt_free(row)}"
                      f"{_fmt_mem(c, study)}")
            else:
                c["row"], c["state"] = row, DONE_CASE
                ok = quality_ok(row, study)
                _emit(sp, "case", _case_subject(sp, cid), "REDUCED",
                      f"{_case_tag(c)} {_fmt_r(row)}{'' if ok else '  [quality: NOT ok]'}"
                      f"{_fmt_mem(c, study)}")
            continue
        job = c.get("job")
        if job and job in active:
            continue
        if job is None:  # new, or cancelled by `hs stop`
            submit_case(sp, study, led, topo, cid, slurm)
            continue
        _record_mem(c, slurm, job)
        why = st.get("state") or "no status.json (died before the driver started)"
        if st.get("error"):
            why += f" in {st.get('phase', '?')}: {st['error'][:200]}"
        if st.get("retryable") is False and st.get("error_type") != "CaseSetupError":
            # A code/identity error: every case would hit it. Halt instead of burning retries.
            halt(sp, led, slurm, f"{cid}: {why}", active)
            return
        c["failures"] += 1
        if st.get("retryable") is False:
            c["state"] = BAD_CASE
            c["note"] = f"not retried (case setup error): {why}"
            _emit(sp, "case", _case_subject(sp, cid), "BAD", f"{_case_tag(c)} {c['note']}")
        elif c["failures"] <= int(study.get("max_retries", 3)):
            _emit(sp, "case", _case_subject(sp, cid), "RETRY",
                  f"{_case_tag(c)} job {job} ended without completing ({why}); "
                  f"attempt {c['failures'] + 1}")
            submit_case(sp, study, led, topo, cid, slurm)
        else:
            c["state"] = BAD_CASE
            c["note"] = f"gave up after {c['failures']} failures; last: {why}"
            _emit(sp, "case", _case_subject(sp, cid), "BAD", f"{_case_tag(c)} {c['note']}")


def _record_mem(c: dict, slurm, job: str) -> None:
    fn = getattr(slurm, "max_rss_gb", None)
    if fn and job:
        try:
            gb = fn(job)
        except Exception:
            gb = None
        if gb is not None:
            c["max_rss_gb"] = max(gb, c.get("max_rss_gb") or 0.0)


def _fmt_mem(c: dict, study: dict) -> str:
    gb = c.get("max_rss_gb")
    return f"  maxRSS {gb:.1f}G of {study['slurm']['mem']}" if gb is not None else ""


def halt(sp, led, slurm, reason: str, active: dict[str, str]) -> None:
    """Stop the study on an error every case would hit. `hs resume` after the fix."""
    ids = [c["job"] for c in led["cases"].values()
           if c["state"] == LIVE_CASE and c["job"] and c["job"] in active]
    slurm.cancel(ids)
    for c in led["cases"].values():
        if c["state"] == LIVE_CASE:
            c["job"] = None  # resubmitted by `hs resume`, not counted as failures
    led["stopped"] = True
    _emit(sp, "study", sp.name, "HALTED",
          f"non-retryable error, cancelled {len(ids)} job(s): {reason[:300]}  "
          "-> fix, `hs sync push-code`, then `hs resume " + sp.name + "`")


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
        if not row or abs(c["speed"] - speed) > 1e-9 or c.get("kind") == "free":
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
        if ch.get("phase") == "free":
            _drive_free(sp, study, led, key, ch)
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
                ch["result"] = _chain_result(d)
                fc = study.get("free")
                if fc and d.kind in fc["relax_on"] and d.best is not None:
                    # Not the end: verify the equilibrium by letting the hull go.
                    ch["newton_status"], ch["phase"] = d.kind, "free"
                    best_row = led["cases"][d.best.case_id]["row"]
                    ch["source"] = {"key": key, "speed": ch["speed"], "study": sp.name,
                                    "case_id": d.best.case_id, "theta": d.best.theta, "z": d.best.z,
                                    "row": {k: best_row.get(k) for k in SOURCE_ROW_KEYS}}
                    _emit(sp, "newton", tag, d.kind, _fmt_decision(d) + "  -> free-running relaxation")
                    _drive_free(sp, study, led, key, ch)
                    break
                ch["status"] = d.kind
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


#: What a free case needs from its parent's reduced row (thrust) plus the residuals for the record.
SOURCE_ROW_KEYS = ("thrust_N", "thrust_arm_m", "R_lift_N", "R_pitch_Nm", "drag_N")


def _drive_free(sp, study, led, key: str, ch: dict) -> None:
    """Submit the chain's free case, or, once it is resolved, end the chain with the verdict."""
    src = ch["source"]
    subject = progress.chain_tag(ch["speed"])
    if not ch["pending"] or free_case_id(src["case_id"]) not in ch["pending"]:
        it = (ch.get("state") or {}).get("iteration", 0)
        cid = register_free_case(sp, led, key, src, it)
        ch["pending"] = [cid]
        if led["cases"][cid]["state"] == LIVE_CASE:
            _emit(sp, "study", subject, "RELAX", f"release from {progress.short_case(src['case_id'], sp.topology)}"
                  f"  θ={src['theta']:+.2f}° z={src['z'] * 1000:+.1f}mm")
            return
    c = led["cases"][ch["pending"][0]]
    verdict, why = (free_verdict(c.get("row"), study["free"]) if c["state"] == DONE_CASE
                    else (FREE_FAILED, c.get("note") or c["state"]))
    row = c.get("row") or {}
    ch["status"] = verdict
    ch["result"] = {**(ch.get("result") or {}), "free": {
        "case_id": ch["pending"][0], "verdict": verdict, "note": why,
        **{k: row.get(k) for k in ("theta_mean_deg", "z_mean_m", "dtheta_deg", "dz_m",
                                   "theta_drift_deg", "z_drift_m", "motion_source")}}}
    ch["history"].append({"at": _now(), "kind": verdict, "free_case": ch["pending"][0], "note": why})
    _emit(sp, "study", subject, verdict, why)


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
    if not deps:
        return slurm.submit([*base, "--begin=now+15minutes"], script)
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
                         "role": c["role"], "state": c["state"], "kind": c.get("kind", "captive"),
                         **{k: v for k, v in c["row"].items() if k not in ("case_id", "kind")}})
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
            if led.get("stopped"):  # halted on a non-retryable error
                led["advance_job"] = None
                return "halted"
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
        unsubmitted = sum(c["state"] == LIVE_CASE and not c["job"] for c in led["cases"].values())
        if unsubmitted:
            # sbatch refused some (e.g. a submit limit): poll rather than wait on the rest
            led["advance_job"] = submit_advance(sp, study, slurm, [])
            outcome = (f"{unsubmitted} case(s) not accepted by sbatch; {len(live)} live; "
                       f"polling with advance {led['advance_job']}")
        elif live:
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
               f"{'step':>11} {'s/step':>6} {'ETA':>7} {'maxRSS':>7}  R_lift / R_pitch")
    for cid, c in sorted(led["cases"].items(), key=lambda kv: (kv[1]["iteration"], kv[0])):
        s = read_json(sp.case_dir(cid) / layout.STATUS_FILE, {}) or {}
        step = f"{s.get('time_step', '-')}/{s.get('total_steps', '-')}" if s else "-"
        sps = s.get("s_per_step")
        eta = s.get("eta_s")
        mem = f"{c['max_rss_gb']:.1f}G" if c.get("max_rss_gb") else "-"
        r = c.get("row") or {}
        if isinstance(r.get("R_lift_N"), float):
            rs = f"{r['R_lift_N']:+.2f} / {r['R_pitch_Nm']:+.3f}"
        elif isinstance(r.get("theta_mean_deg"), float):
            rs = f"free: θ {r['theta_mean_deg']:+.3f}° z {r['z_mean_m'] * 1000:+.2f}mm"
        else:
            rs = (c.get("note") or "")[:40]
        out.append(f"{progress.short_case(cid, sp.topology):<30} {c['iteration']:>3} {c['role']:<8} "
                   f"{c['state']:<7} {str(c.get('job') or '-'):>9} {c['failures']:>4}  {step:>11} "
                   f"{(f'{sps:.1f}' if sps else '-'):>6} {progress.fmt_hours(eta) if eta else '-':>7} "
                   f"{mem:>7}  {rs}")
    return "\n".join(out)
