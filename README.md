# hullsweep: captive (θ, z, V) studies on Engaging

A **study** finds the trim/heave equilibrium (θ*, z*) of a hull topology at one or more speeds.
It runs Newton iterations on captive CFD cases, with no human in the loop between iterations.
You give a topology name, a study name and the physics inputs. Everything else (paths,
case preparation, job submission, retries, reduction, the next Newton step) is derived.

**Conventions**
- θ is positive for **bow up**.
- z is positive for **up**; it is the CG's displacement from `cg_ref`.
- The hull is rotated about the X axis through `cg_ref`, then translated by (0, 0, z).
- Moments are taken about the displaced CG.
- All forces are for the **simulated domain** (half the boat for `half_domain`).
- Residuals: R_lift = Fz + T sin θ − W, and R_pitch = bow-up moment including thrust.
  See `collect.py`.

## Where things live (`layout.py` is the only module that knows a path)

```
CLUSTER
~/hullsweep_code/                               code (hs sync push-code), VERSION
~/orcd/pool/hullsweep/<topo>/                   topology.json, <topo>_template.cas.h5, meshes
~/orcd/pool/hullsweep/<topo>/<study>/           study.json (frozen), ledger.json, results.csv,
                                                logs/<case_id>/<jobid>.out|err, logs/_advance/
~/orcd/scratch/hullsweep/<topo>/<study>/        progress.log  <- tail -f this
~/orcd/scratch/hullsweep/<topo>/<study>/<case>/ request.json, <case>.cas.h5, sidecar, autosaves,
                                                <case>_final.cas/dat, sweep-forces.out, status.json
LOCAL
Arcturus/hullsweep/                             code only
Arcturus/hullsweep_data/<topo>/                 topology inputs (+ local-only <topo>_source.cas.h5)
Arcturus/hullsweep_data/<topo>/<study>/         pulled study (pool and scratch parts merged)
```

- Topology files are found by pattern (`<topo>*template*.cas.h5`, `*foreground*.msh*`, …).
  Each pattern must match **exactly one** file, or the command stops and lists what it found.
- A study is addressed by its **name** alone. Study names are unique across topologies.
- `study.json` records the sha1 of `topology.json` and of the template. A case job refuses to
  run if the template changed, and `hs advance` stops if `topology.json` changed.
  `hs sync push-topology` refuses to overwrite files that existing studies were created with.

## Runbook

Local commands use `C:\Users\vgnau\miniforge3\envs\reefs-mobo\python.exe` from `Arcturus/hullsweep`.
Cluster commands use `~/hullsweep_code/bin/hs` (add `alias hs=~/hullsweep_code/bin/hs`).

| Step | Where | Command |
|---|---|---|
| 1. Gather a topology (once) | local | `python hs.py topology adopt --name T --source X.cas.h5 --foreground-mesh F.msh.h5 --background-mesh B.msh.h5 --topology-json J` |
| 2. Build the template (once) | local, **normal terminal** | `python prepare_case.py template --topology T` |
| 3. Upload the topology (once) | local | `python hs.py sync push-topology T` |
| 4. Upload code (after edits) | local | `python hs.py sync push-code` |
| 5. Smoke test (new topology or code) | cluster | `hs new --topology T --study T_smoke1 --speeds 2.5 --x0 0,0 --smoke --max-iters 2` |
| 6. Study | cluster | `hs new --topology T --study S --speeds 2.0,2.5,3.0 --x0 1.5,-0.01 [--seed …/results.csv]` |
| 7. Monitor | cluster | `tail -f` the progress log it prints; `hs status S` |
| 8. Fetch | local | `python hs.py sync pull S [--with-data]` |

`hs new` prints the resolved plan: paths, run plan per speed, license use, tolerances, envelope,
and seeds. It creates nothing until you type the study name back (`--yes` skips this,
`--dry-run` only prints). Other commands: `hs stop S`, `hs resume S`, `hs list`,
`hs path case-dir --study S --case C`, and `python collect.py --study S --cg-offset DY DZ`
(re-reduce).

A grid sweep with no Newton is `hs new … --speeds … --theta -1,0,1 --z -0.01,0,0.01`.

## How a study runs

- **Orchestrator.** `hs advance S` is a 1-core, 10-min job on `mit_quicktest`, chained with
  `--dependency=afterany:<case jobs>`. Each run reduces finished cases (`collect.reduce_case`),
  resubmits dead ones (up to `--max-retries`; a license denial at launch is a dead job), and asks
  `newton.decide` for each speed's next batch. It then submits that batch plus the next
  `advance`. It checks job liveness itself (squeue), so a requeued case never confuses it.
  State is kept in `ledger.json`.
- **Case job.** `cluster/case_job.sh` (static) runs `run_case.py`. Prep and solve run in **one
  Fluent session**: template, rigid move checked against the analytic transform, inlet speed and
  init defaults, dt, sweep reports about the displaced CG, flat open-channel init, water-level
  check, write the case, solve. A resumed attempt reads the case plus the newest autosave.
  The job stops 25 min before walltime, writes `sw-stop-*.dat.h5` and requeues itself.
- **Concurrency.** Case jobs are individual `sbatch` jobs, which gives per-case log dirs. `--lanes L`
  caps how many run at once: job name `<study>_lane<k>` with `--dependency=singleton`. The license
  pool requires `lanes × (ntasks − 4) ≤ 71` (from cluster licdebug on 2026-09-12), and `hs new`
  refuses anything more.
- **Newton** (`newton.py`). One chain per speed, all chains running in parallel.
  - Iteration 0 is a 3-point stencil (x0, x0 + Δθ, x0 + Δz).
  - After that, each iteration is **one** case at the Newton point. The Jacobian is a weighted
    affine fit (1/SE² × proximity) over every completed point near the best one.
  - Trust region: if a step improves the merit, ρ doubles; if not, ρ halves and the chain
    re-stencils.
  - Steps are clipped to the topology envelope and rounded to case-id resolution
    (0.01°, 0.1 mm), so repeated points reuse finished cases.
  - **Converged** means a simulated point has |R_lift| ≤ `--tol-lift` and
    |R_pitch| ≤ `--tol-pitch`, and its averaging-window drift is ≤ `--drift-factor` × tol.
  - Other endings: MAX_ITERS, OUTSIDE_ENVELOPE, STALLED, FAILED.
- **Seeds.** `--seed results.csv` imports completed points (same topology and run plan) instead
  of rerunning them.

## Run sizing

- dt = `ref_dt · ref_speed / V`, which keeps the convective Courant number of the validated run
  (5e-3 s at 2.5 m/s).
- Duration is (`settle` + `average`) hull lengths at speed V, so every speed takes about the same
  number of steps.
- The averaging window is t ≥ settle time.
- `check` smoketest: 21 cores, ~17 s/step, 979 steps ≈ **4.6 h/case**. The default
  `--time 06:00:00` fits prep plus one case in a single allocation.

## Initialization

- Flat open-channel free surface from the inlet: standard init with `open_channel_auto_init` set
  explicitly, never inherited from the template.
- The init defaults (velocity, k, ω) are recomputed for each case's speed.
- `check_water_level` compares the volume-averaged water fraction of the background with the
  flat-surface value (z_fs − z_min)/(z_max − z_min). A mismatch fails the case.

## Licensing

- **Cluster:** Shared Web, via the CA-bundle bind mount in `case_job.sh` (copied verbatim from
  `submit_hull.sh`).
- **Local:** Fluent launched from inside VS Code or Claude Code's shell is DENIED at web checkout.
  Run `prepare_case.py template` from a regular terminal.

## Tests

Run `python -m pytest tests`. It covers conventions, transforms, run planning, collect math,
layout resolution, and Newton on synthetic noisy residuals. It also runs the orchestrator state
machine end to end against a fake Slurm, the CLI, and the code packaging. Fluent is not needed.
reefs-mobo has no pytest; use `pip install --target <dir> pytest` and `PYTHONPATH=<dir>`.

## Legacy (superseded, to delete once a study smoke test passes on the cluster)

`manifest.py`, `make_batch.py`, `cluster/submit_sweep.sh.in`, `cluster/run_sweep_case.py`,
`topologies/`, `sweeps/*/` (keep `sweeps/check/results.csv` as a seed source).
