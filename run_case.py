#!/usr/bin/env python3
"""One captive CFD case of a study (cluster: Fluent 26R1, PyFluent 0.42.1).

  python run_case.py --study <name> --case <case_id> [--prep-only]

Launched by cluster/case_job.sh. Everything is resolved from the study name
(layout.py). The point to run comes from <case dir>/request.json, written by
the orchestrator.

One Fluent session per attempt:
  FRESH   template (pool) -> fluent_ops.prepare_case (move, check, speed, dt,
          reports, init, water check) -> write <case_id>.cas.h5 + sidecar
          into the scratch case dir -> solve.
  RESUME  the case dir already holds <case_id>.cas.h5 and an autosave (a
          preempted or walltime-stopped attempt): read both, set earlier
          report files aside as *.part<k>.out, run the remaining steps.

A FREE case (request kind "free") is a 2DOF relaxation of a converged captive
case. Fresh prep reads the parent's <id>_final.cas/dat instead of the template,
then compiles the generated UDF and arms 6DOF (fluent_ops.prepare_free_case).
There is no initialization. The mesh moves, so autosaves write the case each
time, and a resume reads the newest cas+dat autosave pair. Steps count from the
release: the flow time and step counter carry on from the parent. After each
chunk the live 6DOF state is appended to sw-motion.csv.

The run advances in chunks. It stops before the Slurm end time
(HULL_DEADLINE_EPOCH minus stop_margin_s), writes sw-stop-<step>.dat.h5, and
sets its status to 'incomplete'; case_job.sh then requeues the job.

Refuses to run if the template changed since the study was created, or if
the case id does not belong to the study's topology.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fluent_ops as fo  # noqa: E402
import layout  # noqa: E402
import progress  # noqa: E402
from common import Release, file_sha1  # noqa: E402

CHUNK_STEPS = 10
AUTOSAVE_EVERY = 100
AUTOSAVE_ROOT = "sw"
AUTOSAVE_RETAIN = 2
UI_MODE = "no_gui_or_graphics"  # -g: the -nochecks install has no libOpenGL

_AUTOSAVE_RE = re.compile(rf"^{AUTOSAVE_ROOT}-.*?(\d+)\.dat\.h5$")


def log(msg):
    print(f"[case {time.strftime('%H:%M:%S')}] {msg}", flush=True)


class CaseIdentityError(RuntimeError):
    """The job does not match its study (wrong template, topology, or request)."""


#: Failures that recur on every attempt: a bug or a bad case, not the cluster.
#: status.json marks them retryable=false so the orchestrator stops at once.
#: Anything else (license denial, Fluent/gRPC death, node trouble) is retried.
DETERMINISTIC = (CaseIdentityError, AttributeError, TypeError, NameError, KeyError, ImportError,
                 ValueError, IndexError, ZeroDivisionError)


class Case:
    """Paths, request, and progress reporting for one case."""

    def __init__(self, study: str, cid: str):
        self.sp = layout.find_study(study)
        self.cid = cid
        self.dir = self.sp.case_dir(cid)
        self.study = json.loads(self.sp.study_json.read_text())
        self.req = json.loads((self.dir / layout.REQUEST_FILE).read_text())
        self.subject = progress.short_case(cid, self.sp.topology)
        self.free = self.req.get("kind", "captive") == "free"
        self.tag = (f"{progress.chain_tag(self.req['speed_mps'])}/"
                    f"{'free' if self.free else 'it' + str(self.req['iteration'])}")
        self.total = int(self.req["steps"])
        self.step0 = 0  # time-step counter at release (free cases carry the parent's on)

    def emit(self, event: str, detail: str = "") -> None:
        progress.emit(self.sp.progress_log, "case", self.subject, event, f"{self.tag} {detail}".strip(),
                      echo=False)
        log(f"{event} {detail}")

    def status(self, **kw) -> None:
        tmp = self.dir / "status.json.tmp"
        tmp.write_text(json.dumps({"case_id": self.cid, "total_steps": self.total, **kw}, indent=1))
        tmp.replace(self.dir / layout.STATUS_FILE)

    @property
    def cas(self) -> Path:
        return self.dir / f"{self.cid}.cas.h5"

    @property
    def sidecar(self) -> Path:
        return self.dir / f"{self.cid}.json"


# --- guards -------------------------------------------------------------------


def check_identity(case: Case, topo, template: Path) -> None:
    if not case.cid.startswith(f"{topo.name}_V"):
        raise CaseIdentityError(f"case id {case.cid} does not belong to topology {topo.name}")
    if case.req["case_id"] != case.cid or case.req["study"] != case.sp.name:
        raise CaseIdentityError(f"request.json names {case.req['case_id']}/{case.req['study']}, "
                           f"not {case.cid}/{case.sp.name}")
    want = case.study["sha1"]["template"]
    got = file_sha1(template)
    if template.name != want["name"] or got != want["sha1"]:
        raise CaseIdentityError(f"template {template.name} ({got[:10]}) differs from the one the study was "
                           f"created with ({want['name']}, {want['sha1'][:10]})")


# --- resume -------------------------------------------------------------------


def case_of(dat: Path) -> Path:
    """The case file written with a data file (same stem)."""
    return dat.with_name(dat.name[: -len(".dat.h5")] + ".cas.h5")


def latest_autosave(work: Path, with_case: bool = False) -> Path | None:
    """Newest autosave data file; with_case: only one whose case file exists too (moving mesh)."""
    best, best_n = None, -1
    for p in work.glob(f"{AUTOSAVE_ROOT}-*.dat.h5"):
        m = _AUTOSAVE_RE.match(p.name)
        if m and int(m.group(1)) > best_n and (not with_case or case_of(p).exists()):
            best, best_n = p, int(m.group(1))
    return best


def set_aside_report_files(work: Path) -> None:
    """Keep earlier attempts' output rows; Fluent may truncate on reopen.

    Report files, the 6DOF motion history and the EnSight index become
    <stem>.part<k><suffix>. The per-step EnSight files are numbered and never collide.
    """
    files = [*work.glob("*.out"), *work.glob("*.6dof"), *work.glob(f"{fo.ENSIGHT_DIR}/*.encas")]
    for p in files:
        if ".part" in p.name:
            continue
        k = 1
        while p.with_name(f"{p.stem}.part{k}{p.suffix}").exists():
            k += 1
        p.rename(p.with_name(f"{p.stem}.part{k}{p.suffix}"))
        log(f"set aside {p.name} -> {p.stem}.part{k}{p.suffix}")


# --- Fluent -------------------------------------------------------------------


def scheme(solver, expr):
    last = None
    for fn in (lambda: solver.scheme.eval(expr), lambda: solver.scheme_eval.scheme_eval(expr)):
        try:
            return fn()
        except Exception as exc:
            last = exc
    raise RuntimeError(f"no working scheme path; last error: {last}")


def flow_state(solver):
    try:
        t = float(scheme(solver, "(rpgetvar 'flow-time)"))
        n = int(float(scheme(solver, "(rpgetvar 'time-step)")))
        return t, n
    except Exception as exc:
        log(f"WARNING: could not read flow time ({exc})")
        return None, None


def configure(solver, case: Case, topo, resumed: bool) -> bool:
    """Time step, autosave, report files, exports. Returns False if a free case's EnSight
    export could not be registered."""
    rc = solver.settings.solution.run_calculation
    rc.parameters.time_step_size = float(case.req["dt_s"])
    rc.parameters.max_iter_per_time_step = int(case.req["max_iter_per_step"])
    log(f"dt = {case.req['dt_s']} s, max iter/step = {case.req['max_iter_per_step']}")

    auto = solver.settings.solution.calculation_activity.auto_save
    auto.data_frequency = AUTOSAVE_EVERY
    try:
        # static mesh: the case never changes; moving mesh: a .dat alone cannot restart
        auto.case_frequency = "each-time" if case.free else "if-case-is-modified"
    except Exception as exc:
        log(f"case_frequency left as is ({exc})")
    auto.root_name = str(case.dir / AUTOSAVE_ROOT)
    try:
        auto.retain_most_recent_files = True  # must precede max_files (CLAUDE.md section 8)
        auto.max_files = AUTOSAVE_RETAIN
    except Exception as exc:
        log(f"WARNING: autosave retention not set ({exc})")
    log(f"autosave: {auto.get_state()}")

    fo.relativize_report_files(solver)  # report files land in the case dir
    if case.free:
        return fo.configure_ensight(solver, topo, case.dir, float(case.req["export_every_s"]),
                                    float(case.req["dt_s"]), os.environ.get("SLURM_JOB_ID", "local"), resumed)
    for cmd in ("/solve/execute-commands/delete export-1",
                "/file/transient-export/settings/delete export-1"):
        try:
            fo.tui(solver, cmd)
        except Exception:
            pass
    return True


class MotionLog:
    """Appends the live 6DOF state of the first hull wall to sw-motion.csv after each chunk."""

    def __init__(self, solver, case: Case, topo):
        self.solver, self.zone = solver, topo["hull_wall_zones"][0]
        self.path = case.dir / fo.MOTION_LOG
        self.ok = True
        if not self.path.exists():
            self.path.write_text(fo.MOTION_LOG_HEADER + "\n")

    def __call__(self) -> None:
        if not self.ok:
            return
        try:
            t, n = flow_state(self.solver)
            s = fo.read_sdof_state(self.solver, self.zone)
            with open(self.path, "a") as f:
                f.write(f"{t!r},{n},{s['cg'][0]!r},{s['cg'][1]!r},{s['cg'][2]!r},{s['theta_x_deg']!r}\n")
        except Exception as exc:  # the .6dof history is still there; do not kill the run
            self.ok = False
            log(f"WARNING: 6DOF state sampling disabled ({exc!r})")


def advance(solver, case: Case, done: int, deadline: float, margin: float,
            on_chunk=None) -> tuple[bool, float, int]:
    """Run the remaining steps in chunks; stop early if the next chunk would cross the deadline.

    Returns (stopped_for_deadline, solve_wall_seconds, steps_run). On 26R1
    dual_time_iterate's inner-iteration argument is max_iter_per_step, so only
    time_step_count is passed and the cap comes from run_calculation.parameters.
    `done` and the returned counts are steps since the case's own start.
    """
    rc = solver.settings.solution.run_calculation
    remaining = case.total - done
    t0 = time.time()
    ran, sps = 0, None
    quarters = {q for q in (1, 2, 3) if done < q * case.total / 4}
    while ran < remaining:
        n = min(CHUNK_STEPS, remaining - ran)
        if deadline and sps is not None and time.time() + n * sps + margin > deadline:
            return True, time.time() - t0, ran
        c0 = time.time()
        try:
            rc.dual_time_iterate(time_step_count=n)
        except Exception as exc:
            if ran:
                raise
            log(f"dual_time_iterate unavailable ({exc}); falling back to calculate()")
            rc.parameters.time_step_count = n
            rc.calculate()
        ran += n
        chunk_sps = (time.time() - c0) / n
        if on_chunk is not None:
            on_chunk()
        sps = chunk_sps if sps is None else max(sps, chunk_sps)
        step = done + ran
        eta = (case.total - step) * chunk_sps
        if ran % (10 * CHUNK_STEPS) == 0 or ran == remaining:
            log(f"  {step}/{case.total} steps, {chunk_sps:.1f} s/step")
            case.status(state="running", time_step=step, s_per_step=chunk_sps, eta_s=eta,
                        job=os.environ.get("SLURM_JOB_ID"))
        for q in sorted(quarters):
            if step >= q * case.total / 4:
                quarters.discard(q)
                case.emit("PROGRESS", f"{step}/{case.total}  {chunk_sps:.1f} s/step  "
                                      f"ETA {progress.fmt_hours(eta)}")
    return False, time.time() - t0, ran


# --- main ---------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--study", required=True)
    ap.add_argument("--case", required=True)
    ap.add_argument("--prep-only", action="store_true", help="write the case, do not solve")
    ap.add_argument("--procs", type=int, default=int(os.environ.get("SLURM_NTASKS", "8")))
    args = ap.parse_args()

    case = Case(args.study, args.case)
    case.dir.mkdir(parents=True, exist_ok=True)
    topo = layout.load_topology(case.sp.topology)
    template = layout.topology_file(case.sp.topology, "template")
    job = os.environ.get("SLURM_JOB_ID", "-")
    node = os.environ.get("SLURMD_NODENAME", os.environ.get("HOSTNAME", "?"))
    deadline = float(os.environ.get("HULL_DEADLINE_EPOCH", "0") or 0)
    margin = float(case.req.get("stop_margin_s", 1500))
    restarts = os.environ.get("SLURM_RESTART_COUNT", "0")

    import ansys.fluent.core as pyfluent

    log(f"pyfluent {pyfluent.__version__}; code {json.dumps(ledger_version())}")
    log(f"request: {json.dumps(case.req)}")
    t_start = time.time()
    solver = None
    phase = "check"
    try:
        check_identity(case, topo, template)
        if case.free:
            resume = latest_autosave(case.dir, with_case=True) if case.sidecar.exists() else None
        else:
            resume = latest_autosave(case.dir) if case.cas.exists() and case.sidecar.exists() else None
        if resume is not None:
            set_aside_report_files(case.dir)

        phase = "launch"
        solver = pyfluent.launch_fluent(mode="solver", ui_mode=UI_MODE, processor_count=args.procs,
                                        cwd=str(case.dir), start_timeout=600, additional_arguments="")
        version = str(getattr(solver, "get_fluent_version", lambda: "?")())

        phase = "prep"
        if resume is not None:
            case.emit("RESUME", f"job {job} node {node} restart {restarts} from {resume.name}")
            fo.read_case(solver, str(case_of(resume) if case.free else case.cas))
            solver.settings.file.read(file_type="data", file_name=str(resume))
            if case.free:
                case.step0 = int(json.loads(case.sidecar.read_text())["step_release"])
        elif case.free:
            case.emit("PREP", f"job {job} node {node} restart {restarts}  from {case.req['parent_case_id']}")
            rel = Release.from_dict(case.req["release"])
            facts = fo.prepare_free_case(solver, topo, Path(case.req["parent_cas"]), Path(case.req["parent_dat"]),
                                         rel, float(case.req["dt_s"]), int(case.req["max_iter_per_step"]),
                                         case.dir, case.req.get("parent_hull_centroid"),
                                         case.req.get("implicit_update"))
            t0, n0 = flow_state(solver)
            if n0 is None:
                raise RuntimeError("could not read the time-step counter at release")
            case.step0 = n0
            fo.write_case(solver, str(case.cas), data=False)  # the armed case, for inspection
            sidecar = {**case.req, **facts, "t_release_s": t0, "step_release": n0,
                       "template": {"path": str(template), "sha1": case.study["sha1"]["template"]["sha1"]},
                       "fluent_version": version, "prepared_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
            case.sidecar.write_text(json.dumps(sidecar, indent=1, default=str))
            case.emit("PREPPED", f"released at t={t0:.4f} s step {n0}: CG {[round(v, 5) for v in rel.cg]}  "
                                 f"mass {rel.mass_kg:g} kg  Ixx {rel.inertia_kgm2['ixx']:.4g}  "
                                 f"thrust {rel.thrust_N:.2f} N  UDF via {facts['udf_route']}  "
                                 f"implicit 6DOF {'on' if facts['implicit_update'].get('enabled') else 'off'}")
        else:
            case.emit("PREP", f"job {job} node {node} restart {restarts}")
            facts = fo.prepare_case(solver, topo, template, float(case.req["theta_deg"]), float(case.req["z_m"]),
                                    float(case.req["speed_mps"]), float(case.req["dt_s"]), case.dir)
            fo.write_case(solver, str(case.cas), data=False)
            sidecar = {**case.req, **facts,
                       "template": {"path": str(template), "sha1": case.study["sha1"]["template"]["sha1"]},
                       "fluent_version": version, "prepared_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
            case.sidecar.write_text(json.dumps(sidecar, indent=1, default=str))
            w = facts["water_level_check"]
            case.emit("PREPPED", f"centroid err {facts['transform_centroid_err_m']:.1e} m  "
                                 f"water {w['measured']:.4f} (flat {w['expected']:.4f})  "
                                 f"clearance {facts['bottom_clearance_m']:.3f} m")
        if args.prep_only:
            case.status(state="prepped")
            return 0

        phase = "solve"
        t, n = flow_state(solver)
        done = (n or 0) - case.step0
        if resume is None and not case.free and t is not None and abs(t) > 1e-9:
            log(f"WARNING: fresh case starts at flow-time {t}, not 0")
        if not configure(solver, case, topo, resume is not None):
            case.emit("WARNING", "EnSight export NOT registered; the run continues without it")
        on_chunk = MotionLog(solver, case, topo) if case.free else None
        if on_chunk is not None and resume is None:
            on_chunk()  # the release state itself
        case.status(state="running", time_step=done, job=job)
        case.emit("ITERATING", f"job {job} {node}  steps {done}->{case.total}  dt {case.req['dt_s']:.4g}"
                  + (f"  deadline {progress.fmt_hours(deadline - time.time())}" if deadline else ""))
        stopped, wall, ran = advance(solver, case, done, deadline, margin, on_chunk)

        t, n = flow_state(solver)
        step = (n or 0) - case.step0
        sps = wall / max(ran, 1)
        if stopped:
            stop = case.dir / f"{AUTOSAVE_ROOT}-stop-{(n or 0):05d}.dat.h5"
            if case.free:  # moving mesh: the case goes with the data
                solver.settings.file.write(file_type="case-data", file_name=str(stop)[: -len(".dat.h5")])
            else:
                solver.settings.file.write(file_type="data", file_name=str(stop))
            case.status(state="incomplete", flow_time=t, time_step=step, solve_wall_s=wall, s_per_step=sps)
            case.emit("INCOMPLETE", f"walltime: stopped at {step}/{case.total}, wrote {stop.name}; requeue")
            return 0

        solver.settings.file.write(file_type="case-data", file_name=str(case.dir / f"{case.cid}_final"))
        case.status(state="complete", flow_time=t, time_step=step, solve_wall_s=wall, s_per_step=sps,
                    total_wall_s=time.time() - t_start)
        case.emit("COMPLETE", f"{step}/{case.total} steps  {progress.fmt_hours(time.time() - t_start)} "
                              f"({sps:.1f} s/step)")
        return 0
    except Exception as exc:
        retryable = not isinstance(exc, (*DETERMINISTIC, fo.CaseSetupError))
        log(f"EXCEPTION in {phase}: {exc!r} (retryable={retryable})")
        case.status(state="failed", phase=phase, error=repr(exc), error_type=type(exc).__name__,
                    retryable=retryable)
        case.emit("FAILED", f"{phase}: {repr(exc)[:300]}{'' if retryable else '  [not retryable]'}")
        raise
    finally:
        if solver is not None:
            try:
                solver.exit()
            except Exception:
                pass


def ledger_version() -> dict:
    try:
        return json.loads((layout.CODE_DIR / "VERSION").read_text())
    except FileNotFoundError:
        return {"content_sha1": "unversioned"}


if __name__ == "__main__":
    sys.exit(main())
