#!/usr/bin/env python3
"""
Captive sweep case driver (cluster: Fluent 26R1, PyFluent 0.42.1).

One Slurm array task = one case = one prepared .cas.h5 (+ optional
.dat.h5) and its sidecar <case_id>.json from prepare_case.py.

Differences from run_hull_vof.py (which stays the restart driver for free
2DOF runs):
  - FRESH START: expects flow time 0, initializes if the case was shipped
    without data (--case-only prep).
  - RESUME: HULL_WORKDIR is per CASE, not per job. If a preempted/requeued
    task finds autosaves there, it reads the case plus the newest autosaved
    .dat.h5 and runs only the remaining steps. Report files from the earlier
    attempt are moved aside as *.part<k>.out; collect.py merges them.
  - dt and step count come from the sidecar, not env vars.
  - No UDF, no dynamic mesh: the hull is fixed. EnSight export is OFF by
    default (HULL_EXPORT=1 to enable) -- report files carry the numbers.

Env:
  HULL_CASE, HULL_SIDECAR     required
  HULL_DATA                   optional explicit data file
  HULL_WORKDIR                per-case work dir (resume state lives here)
  HULL_KEEP_DIR               small artifacts copied here at the end
  SLURM_NTASKS                processor count
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import ansys.fluent.core as pyfluent

# common.py, fluent_ops.py and topology.json are copied next to this driver
# by make_batch.py, so cluster-side init uses the same code as local prep.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fluent_ops as fo  # noqa: E402
from common import Topology  # noqa: E402

TOPO = Topology.load(HERE / "topology.json")

CASE_FILE = Path(os.environ["HULL_CASE"]).resolve()
SIDECAR = json.loads(Path(os.environ["HULL_SIDECAR"]).read_text())
_data_env = os.environ.get("HULL_DATA", "")
DATA_FILE = Path(_data_env).resolve() if _data_env else None
WORK_DIR = Path(os.environ.get("HULL_WORKDIR", Path.cwd())).resolve()
_keep = os.environ.get("HULL_KEEP_DIR", "")
KEEP_DIR = Path(_keep).resolve() if _keep else None

DT = float(SIDECAR["dt_s"])
TOTAL_STEPS = int(os.environ.get("HULL_STEPS_OVERRIDE", SIDECAR["steps"]))
MAX_ITER_PER_STEP = int(os.environ.get("HULL_MAX_ITER", "20"))

CHUNK_STEPS = int(os.environ.get("HULL_CHUNK_STEPS", "10"))
# Slurm end time (epoch s), exported by submit_sweep.sh. The run stops,
# writes its data, and yields STOP_MARGIN_S before it -- a walltime kill
# is not requeued and would lose everything since the last autosave.
DEADLINE = float(os.environ.get("HULL_DEADLINE_EPOCH", "0") or 0)
STOP_MARGIN_S = float(os.environ.get("HULL_STOP_MARGIN_S", "1500"))

AUTOSAVE_EVERY = int(os.environ.get("HULL_AUTOSAVE_EVERY", "100"))
AUTOSAVE_ROOT = "sw"
AUTOSAVE_RETAIN = 2
WRITE_FINAL = os.environ.get("HULL_WRITE_FINAL", "1") not in ("0", "", "false")

EXPORT_ENABLED = os.environ.get("HULL_EXPORT", "0") not in ("0", "", "false")
EXPORT_FREQUENCY = os.environ.get("HULL_EXPORT_FREQ", "0.25")
EXPORT_SCALARS = os.environ.get("HULL_EXPORT_SCALARS", "pressure phase-2-vof").split()

PROCESSOR_COUNT = int(os.environ.get("SLURM_NTASKS", "8"))
UI_MODE = os.environ.get("HULL_UI_MODE", "no_gui_or_graphics")


def log(msg):
    print(f"[sweep {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --- TUI / Scheme plumbing (from run_hull_vof.py) ---------------------------

_TUI_PATH = None
_SCHEME_PATH = None


def tui(solver, command):
    global _TUI_PATH
    table = {
        "solver.execute_tui": lambda: solver.execute_tui(command),
        "solver.scheme.eval": lambda: solver.scheme.eval(f'(ti-menu-load-string "{command}")'),
        "solver.scheme_eval.scheme_eval":
            lambda: solver.scheme_eval.scheme_eval(f'(ti-menu-load-string "{command}")'),
    }
    if _TUI_PATH is not None:
        return table[_TUI_PATH]()
    last = None
    for name, fn in table.items():
        try:
            r = fn()
            _TUI_PATH = name
            log(f"TUI path resolved: {name}")
            return r
        except Exception as exc:
            last = exc
    raise RuntimeError(f"no working TUI path; last error: {last}")


def scheme(solver, expr):
    global _SCHEME_PATH
    table = {
        "solver.scheme.eval": lambda: solver.scheme.eval(expr),
        "solver.scheme_eval.scheme_eval": lambda: solver.scheme_eval.scheme_eval(expr),
    }
    if _SCHEME_PATH is not None:
        return table[_SCHEME_PATH]()
    last = None
    for name, fn in table.items():
        try:
            r = fn()
            _SCHEME_PATH = name
            return r
        except Exception as exc:
            last = exc
    raise RuntimeError(f"no working scheme path; last error: {last}")


def flow_state(solver):
    try:
        return float(scheme(solver, "(rpgetvar 'flow-time)")), int(float(scheme(solver, "(rpgetvar 'time-step)")))
    except Exception as exc:
        log(f"WARNING: could not read flow time ({exc})")
        return None, None


# --- resume -----------------------------------------------------------------

_AUTOSAVE_RE = re.compile(rf"^{AUTOSAVE_ROOT}-.*?(\d+)\.dat\.h5$")


def latest_autosave() -> Path | None:
    best, best_n = None, -1
    for p in WORK_DIR.glob(f"{AUTOSAVE_ROOT}-*.dat.h5"):
        m = _AUTOSAVE_RE.match(p.name)
        if m and int(m.group(1)) > best_n:
            best, best_n = p, int(m.group(1))
    return best


def set_aside_report_files() -> None:
    """Keep earlier attempts' report rows; Fluent may truncate on reopen."""
    for p in WORK_DIR.glob("*.out"):
        if ".part" in p.name:
            continue
        k = 1
        while (WORK_DIR / f"{p.stem}.part{k}.out").exists():
            k += 1
        p.rename(WORK_DIR / f"{p.stem}.part{k}.out")
        log(f"set aside {p.name} -> {p.stem}.part{k}.out")


# --- configuration ------------------------------------------------------------


def configure(solver):
    rc = solver.settings.solution.run_calculation
    rc.parameters.time_step_size = DT
    rc.parameters.max_iter_per_time_step = MAX_ITER_PER_STEP
    log(f"dt = {DT} s, max iter/step = {MAX_ITER_PER_STEP}")

    auto = solver.settings.solution.calculation_activity.auto_save
    auto.data_frequency = AUTOSAVE_EVERY
    try:
        # Static mesh: the case never changes after the first write.
        auto.case_frequency = "if-case-is-modified"
    except Exception as exc:
        log(f"case_frequency left as is ({exc})")
    auto.root_name = str(WORK_DIR / AUTOSAVE_ROOT)
    try:
        auto.retain_most_recent_files = True
        auto.max_files = AUTOSAVE_RETAIN
    except Exception as exc:
        log(f"WARNING: autosave retention not set ({exc})")
    log(f"autosave: {auto.get_state()}")

    # Report files must land in WORK_DIR whatever path the local prep stored.
    try:
        rf = solver.settings.solution.monitor.report_files
        for n in rf.get_object_names():
            fn = str(rf[n].file_name())
            base = re.split(r"[\\/]", fn)[-1]
            if base != fn:
                rf[n].file_name = base
            log(f"report file {n}: {base}")
    except Exception as exc:
        log(f"WARNING: report files not inspected ({exc})")

    for cmd in ("/solve/execute-commands/delete export-1",
                "/file/transient-export/settings/delete export-1"):
        try:
            tui(solver, cmd)
        except Exception:
            pass
    if EXPORT_ENABLED:
        base = str(WORK_DIR / SIDECAR["case_id"]).replace("\\", "/")
        cmd = " ".join(["/file/transient-export/ensight-gold-transient", base, "()", "*", "()",
                        *EXPORT_SCALARS, "q", "no", "yes", '"export-1"', '"flow-time"',
                        str(EXPORT_FREQUENCY), "yes"])
        log(f"export: {cmd}")
        tui(solver, cmd)


def advance(solver, remaining: int) -> tuple[bool, float, int]:
    """Advance in chunks; stop early if the next chunk would cross the deadline.

    Returns (stopped_for_deadline, solve_wall_seconds, steps_run). Chunking
    is cheap: dual_time_iterate continues from the current state without
    reinitializing (the pattern Reefs uses on 25R2). On 26R1 its inner-
    iteration argument is max_iter_per_step, so only time_step_count is
    passed and the cap comes from run_calculation.parameters.
    """
    rc = solver.settings.solution.run_calculation
    t0 = time.time()
    ran = 0
    sps = None
    while ran < remaining:
        n = min(CHUNK_STEPS, remaining - ran)
        if DEADLINE and sps is not None:
            if time.time() + n * sps + STOP_MARGIN_S > DEADLINE:
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
        sps = chunk_sps if sps is None else max(sps, chunk_sps)
        if ran % (10 * CHUNK_STEPS) == 0 or ran == remaining:
            log(f"  {ran}/{remaining} steps, {chunk_sps:.1f} s/step")
    return False, time.time() - t0, ran


def copy_results_out():
    if KEEP_DIR is None:
        log("HULL_KEEP_DIR unset; results stay in scratch")
        return
    KEEP_DIR.mkdir(parents=True, exist_ok=True)
    for pat in ("*.out", "*.trn", "status.json"):
        for src in WORK_DIR.glob(pat):
            shutil.copy2(src, KEEP_DIR / src.name)
    sidecar_src = Path(os.environ["HULL_SIDECAR"])
    shutil.copy2(sidecar_src, KEEP_DIR / sidecar_src.name)
    log(f"kept small artifacts in {KEEP_DIR}")


def write_status(**kw):
    (WORK_DIR / "status.json").write_text(json.dumps(
        {"case_id": SIDECAR["case_id"], "total_steps": TOTAL_STEPS, **kw}, indent=1))


def main():
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    log(f"pyfluent {pyfluent.__version__}; case {SIDECAR['case_id']}")
    log(f"theta={SIDECAR['theta_deg']} z={SIDECAR['z_m']} V={SIDECAR['speed_mps']} "
        f"steps={TOTAL_STEPS} dt={DT} procs={PROCESSOR_COUNT}")

    resume = latest_autosave()
    if resume is not None:
        set_aside_report_files()

    solver = pyfluent.launch_fluent(
        mode="solver", ui_mode=UI_MODE, processor_count=PROCESSOR_COUNT,
        cwd=str(WORK_DIR), start_timeout=600, additional_arguments="",
    )
    t_start = time.time()
    try:
        solver.settings.file.read(file_type="case", file_name=str(CASE_FILE))
        if resume is not None:
            log(f"RESUME from {resume.name}")
            solver.settings.file.read(file_type="data", file_name=str(resume))
        elif DATA_FILE is not None and DATA_FILE.exists():
            solver.settings.file.read(file_type="data", file_name=str(DATA_FILE))
        else:
            # --case-only prep: initialize here, flat open-channel from the
            # inlet, set explicitly (fluent_ops.initialize) rather than
            # inherited. Defaults were recomputed for this speed during prep.
            log("no data shipped: flat open-channel initialization from the inlet")
            fo.initialize(solver, TOPO["inlet_zone"])
            try:
                fo.check_water_level(solver, TOPO)
            except RuntimeError:
                raise  # a real mismatch: never run an all-air domain
            except Exception as exc:
                log(f"WARNING: water-level check could not run on this release ({exc!r}); "
                    "inspect the first autosave before trusting results")

        t, n = flow_state(solver)
        log(f"start state: flow-time={t} time-step={n}")
        done = n or 0
        if resume is None and t is not None and abs(t) > 1e-9:
            log(f"WARNING: fresh case starts at flow-time {t}, not 0")

        configure(solver)
        remaining = TOTAL_STEPS - done
        write_status(state="running", start_step=done, remaining=remaining)
        log(f"running {remaining} steps in chunks of {CHUNK_STEPS}"
            + (f"; deadline in {(DEADLINE - time.time())/3600:.2f} h" if DEADLINE else ""))
        stopped, wall, ran = advance(solver, remaining)

        t, n = flow_state(solver)
        sps = wall / max(ran, 1)
        if stopped:
            # Out of walltime: save state where latest_autosave() will find it;
            # submit_sweep.sh requeues the task, which resumes from here.
            stop = WORK_DIR / f"{AUTOSAVE_ROOT}-stop-{(n or 0):05d}.dat.h5"
            log(f"deadline: writing {stop.name} and yielding for requeue")
            solver.settings.file.write(file_type="data", file_name=str(stop))
            write_status(state="incomplete", flow_time=t, time_step=n,
                         solve_wall_s=wall, s_per_step=sps)
            return 0

        if WRITE_FINAL:
            solver.settings.file.write(file_type="case-data",
                                       file_name=str(WORK_DIR / f"{SIDECAR['case_id']}_final"))
        write_status(state="complete", flow_time=t, time_step=n,
                     solve_wall_s=wall, s_per_step=sps,
                     total_wall_s=time.time() - t_start)
    except Exception as exc:
        log(f"EXCEPTION: {exc!r}")
        write_status(state="failed", error=repr(exc))
        raise
    finally:
        try:
            solver.exit()
        except Exception:
            pass
        copy_results_out()


if __name__ == "__main__":
    sys.exit(main())
