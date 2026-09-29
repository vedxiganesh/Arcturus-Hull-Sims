"""Offline tests for the study machinery: layout, Newton, orchestrator, sync, CLI."""

from __future__ import annotations

import csv
import io
import json
import random
import sys
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import layout  # noqa: E402
import ledger  # noqa: E402
import newton  # noqa: E402
from common import case_id, quantize_point  # noqa: E402

TOPO = "at2_trimaran_halfd"
REAL_TOPO_JSON = layout.LOCAL_DATA / TOPO / layout.TOPOLOGY_FILE
CHECK_RESULTS = ROOT / "sweeps" / "check" / "results.csv"


@pytest.fixture
def site(tmp_path, monkeypatch):
    """A fake cluster: pool/scratch under tmp, one topology with a small template."""
    if not REAL_TOPO_JSON.exists():
        pytest.skip("hullsweep_data topology not present")
    pool, scratch = tmp_path / "pool", tmp_path / "scratch"
    monkeypatch.setenv("HULLSWEEP_SITE", "cluster")
    monkeypatch.setenv("HULLSWEEP_POOL", str(pool))
    monkeypatch.setenv("HULLSWEEP_SCRATCH", str(scratch))
    d = pool / TOPO
    d.mkdir(parents=True)
    (d / layout.TOPOLOGY_FILE).write_text(REAL_TOPO_JSON.read_text())
    (d / f"{TOPO}_template.cas.h5").write_bytes(b"template-bytes")
    return tmp_path


# ---------------------------------------------------------------------------
# layout
# ---------------------------------------------------------------------------


def test_find_one_exactly_one_case_insensitive(tmp_path):
    (tmp_path / "A_Foreground.MSH.h5").write_bytes(b"")
    assert layout.find_one(tmp_path, "*foreground*.msh*", "fg").name == "A_Foreground.MSH.h5"
    with pytest.raises(layout.LayoutError, match="found nothing"):
        layout.find_one(tmp_path, "*background*", "bg")
    (tmp_path / "b_foreground.msh").write_bytes(b"")
    with pytest.raises(layout.LayoutError, match="A_Foreground.MSH.h5, b_foreground.msh"):
        layout.find_one(tmp_path, "*foreground*.msh*", "fg")


def test_topology_resolution(site):
    assert layout.list_topologies() == [TOPO]
    assert layout.topology_file(TOPO, "template").name == f"{TOPO}_template.cas.h5"
    assert layout.topology_file(TOPO, "foreground_mesh", required=False) is None
    (layout.topology_dir(TOPO) / f"{TOPO}_old_template.cas.h5").write_bytes(b"")
    with pytest.raises(layout.LayoutError, match="exactly one"):
        layout.topology_file(TOPO, "template")
    with pytest.raises(layout.LayoutError, match="no topology 'nope'"):
        layout.topology_dir("nope")


def test_topology_name_must_match_folder(site):
    d = layout.roots().pool / "other"
    d.mkdir()
    (d / layout.TOPOLOGY_FILE).write_text(REAL_TOPO_JSON.read_text())
    with pytest.raises(layout.LayoutError, match="name field"):
        layout.load_topology("other")


def test_study_names_unique_across_topologies(site):
    sp = layout.new_study(TOPO, "trim_a")
    assert sp.pool_dir == layout.roots().pool / TOPO / "trim_a"
    assert sp.case_dir("x") == layout.roots().scratch / TOPO / "trim_a" / "x"
    sp.pool_dir.mkdir(parents=True)
    sp.study_json.write_text("{}")
    assert layout.find_study("trim_a") == sp
    with pytest.raises(layout.LayoutError, match="already exists"):
        layout.new_study(TOPO, "trim_a")
    with pytest.raises(layout.LayoutError, match="must match"):
        layout.new_study(TOPO, "Trim-A")
    with pytest.raises(layout.LayoutError, match="no study"):
        layout.find_study("trim_b")


def test_local_site_uses_one_root(monkeypatch):
    monkeypatch.setenv("HULLSWEEP_SITE", "local")
    monkeypatch.delenv("HULLSWEEP_POOL", raising=False)
    monkeypatch.delenv("HULLSWEEP_SCRATCH", raising=False)
    r = layout.roots()
    assert r.pool == r.scratch == layout.LOCAL_DATA


# ---------------------------------------------------------------------------
# quantize / case ids
# ---------------------------------------------------------------------------


def test_quantize_matches_case_id_resolution():
    th, z = quantize_point(1.23456, -0.0123456)
    assert (th, z) == (1.23, -0.0123)
    assert case_id(TOPO, 2.5, th, z) == case_id(TOPO, 2.5, 1.23456, -0.0123456)
    assert quantize_point(-0.0001, -0.00001) == (0.0, 0.0)
    assert case_id(TOPO, 2.5, *quantize_point(-0.0001, 0)) == f"{TOPO}_V2p50_t+0p00_z+0p0mm"


# ---------------------------------------------------------------------------
# newton
# ---------------------------------------------------------------------------

CFG = newton.Config(tol=(1.0, 0.25), fd=(1.0, 0.010), envelope=((-3.0, 6.0), (-0.06, 0.06)), max_iters=8)


def model(theta, z, root=(1.7, -0.013), noise=0.0, rng=None):
    """Hydrostatic-like residuals, mildly nonlinear, root at `root`."""
    dt, dz = theta - root[0], z - root[1]
    rl = -2500.0 * dz + 9.0 * dt + 0.8 * dt * dt
    rp = 30.0 * dz - 1.8 * dt + 0.05 * dt * abs(dt)
    if noise:
        rl += rng.gauss(0, noise)
        rp += rng.gauss(0, noise * 0.1)
    return rl, rp


def run_chain(x0, cfg=CFG, noise=0.0, root=(1.7, -0.013), max_batches=20):
    rng = random.Random(4)
    pts, st, evals = {}, newton.ChainState(x0=x0), 0
    for _ in range(max_batches):
        d = newton.decide(list(pts.values()), st, cfg)
        st = d.state
        if d.terminal:
            return d, evals, pts
        for x in d.points:
            if x not in pts:
                evals += 1
                r = model(*x, root=root, noise=noise, rng=rng)
                pts[x] = newton.Point(x[0], x[1], r, (max(noise, 0.2), max(noise * 0.1, 0.03)), True, str(x))
    raise AssertionError("no terminal decision")


def test_newton_converges_from_x0():
    d, evals, _ = run_chain((0.0, 0.0))
    assert d.kind == newton.CONVERGED
    assert abs(d.best.r[0]) <= 1.0 and abs(d.best.r[1]) <= 0.25
    assert evals <= 8  # 3-point stencil + a few 1-point steps


def test_newton_converges_with_noise():
    d, evals, _ = run_chain((0.0, 0.0), noise=0.3)
    assert d.kind == newton.CONVERGED
    assert abs(d.best.theta - 1.7) < 0.3 and abs(d.best.z + 0.013) < 1e-3


def test_newton_first_batch_is_stencil_then_single_points():
    st = newton.ChainState(x0=(0.0, 0.0))
    d = newton.decide([], st, CFG)
    assert d.kind == "stencil" and d.points == [(0.0, 0.0), (1.0, 0.0), (0.0, 0.01)]
    pts = [newton.Point(*x, model(*x), (0.2, 0.03), True, str(x)) for x in d.points]
    d2 = newton.decide(pts, d.state, CFG)
    assert d2.kind == "step" and len(d2.points) == 1
    assert d2.J[0][1] == pytest.approx(-2500.0, rel=0.05)  # dR_lift/dz recovered


def test_newton_rejected_step_shrinks_and_restencils():
    base = [newton.Point(*x, model(*x), (0.2, 0.03), True, str(x)) for x in
            [(0.0, 0.0), (1.0, 0.0), (0.0, 0.01)]]
    st = newton.ChainState(x0=(0.0, 0.0), iteration=2, rho=2.0,
                           last_step={"from_merit": 1.0, "target": [0.5, 0.005]})
    bad = newton.Point(0.5, 0.005, (500.0, 50.0), (0.2, 0.03), True, "bad")
    d = newton.decide(base + [bad], st, CFG)
    assert d.kind == "stencil" and d.state.rho == 1.0 and "rejected" in d.note


def test_newton_outside_envelope():
    d, _, _ = run_chain((0.0, 0.0), root=(9.0, 0.0))
    assert d.kind == newton.OUTSIDE_ENVELOPE


def test_newton_collinear_points_trigger_stencil():
    pts = [newton.Point(t, 0.0, model(t, 0.0), (0.2, 0.03), True, str(t)) for t in (0.0, 1.0, 2.0)]
    d = newton.decide(pts, newton.ChainState(x0=(0.0, 0.0), iteration=1), CFG)
    assert d.kind == "stencil"
    assert any(p[1] != 0.0 for p in d.points)


def test_stencil_flips_at_envelope_edge():
    assert newton.stencil((6.0, 0.06), 1.0, CFG) == [(6.0, 0.06), (5.0, 0.06), (6.0, 0.05)]


def test_newton_quality_gate_blocks_convergence():
    good = newton.Point(1.7, -0.013, (0.1, 0.01), (0.2, 0.03), False, "drifty")
    d = newton.decide([good], newton.ChainState(x0=(1.7, -0.013), iteration=1), CFG)
    assert d.kind != newton.CONVERGED and "quality" in d.note


def test_solve3_cramer():
    a = [[4.0, 1.0, 2.0], [1.0, 3.0, 0.0], [2.0, 0.0, 5.0]]
    x = [1.0, -2.0, 0.5]
    b = [sum(a[i][j] * x[j] for j in range(3)) for i in range(3)]
    assert newton.solve3(a, b) == pytest.approx(x)
    assert newton.solve3([[1, 2, 3], [2, 4, 6], [0, 0, 1]], [1, 2, 3]) is None


# ---------------------------------------------------------------------------
# orchestrator (fake Slurm, fake reduce)
# ---------------------------------------------------------------------------


class FakeSlurm:
    def __init__(self):
        self.n = 1000
        self.jobs: dict[str, dict] = {}
        self.cancelled: list[str] = []

    def submit(self, opts, script):
        self.n += 1
        jid = str(self.n)
        self.jobs[jid] = {"opts": opts, "script": script, "live": True}
        return jid

    def active(self):
        return {j: "PENDING" for j, v in self.jobs.items() if v["live"]}

    def cancel(self, ids):
        for i in ids:
            if i in self.jobs:
                self.jobs[i]["live"] = False
        self.cancelled += ids

    def case_jobs(self):
        return {j: v for j, v in self.jobs.items() if v["script"][0].endswith("case_job.sh")}


def make_study(sp_name="trim_t", **over) -> dict:
    s = {"name": sp_name, "topology": TOPO, "created": "t", "mode": "newton", "speeds": [2.5],
         "x0": [0.0, 0.0], "grid": None, "tol_lift": 1.0, "tol_pitch": 0.25, "drift_factor": 2.0,
         "fd_theta": 1.0, "fd_z": 0.01, "max_iters": 8, "cg_offset": [0.0, 0.0], "max_retries": 2,
         "run_override": None, "stop_margin_s": 1500,
         "slurm": {"partition": "p", "ntasks": 21, "lanes": 4, "mem": "120G", "time": "06:00:00",
                   "advance_partition": "q", "advance_time": "00:10:00"},
         "smoke": False, "code_version": {"content_sha1": "test"}}
    s.update(over)
    return s


def new_study(site, **over):
    from common import file_sha1

    sp = layout.new_study(TOPO, over.pop("name", "trim_t"))
    tj = layout.topology_dir(TOPO) / layout.TOPOLOGY_FILE
    tpl = layout.topology_file(TOPO, "template")
    st = make_study(sp.name, **over)
    st["sha1"] = {"topology.json": file_sha1(tj), "template": {"name": tpl.name, "sha1": file_sha1(tpl)}}
    return sp, st


@pytest.fixture
def fake_reduce(monkeypatch):
    def reduce_case(topo, case_dir, cg):
        req = json.loads((case_dir / layout.REQUEST_FILE).read_text())
        rl, rp = model(req["theta_deg"], req["z_m"])
        return {"case_id": req["case_id"], "complete": True, "R_lift_N": rl, "R_pitch_Nm": rp,
                "R_lift_se": 0.2, "R_pitch_se": 0.03, "Fz_total": 170.0, "Fz_total_drift": 0.001,
                "Mx_total": 5.0, "Mx_total_drift": 0.001}

    monkeypatch.setattr(ledger.collect, "reduce_case", reduce_case)


def finish_all(sp, slurm, state="complete"):
    led = json.loads(sp.ledger_json.read_text())
    for cid, c in led["cases"].items():
        if c["state"] == "queued" and c["job"] and slurm.jobs[c["job"]]["live"]:
            slurm.jobs[c["job"]]["live"] = False
            if state:
                (sp.case_dir(cid) / layout.STATUS_FILE).write_text(json.dumps({"state": state}))


def test_advance_runs_study_to_convergence(site, fake_reduce):
    sp, st = new_study(site, speeds=[2.5, 3.0])
    slurm = FakeSlurm()
    ledger.init_study(sp, st)
    ledger.advance(sp, slurm)

    jobs = slurm.case_jobs()
    assert len(jobs) == 6  # stencil of 3 per speed
    opts = next(iter(jobs.values()))["opts"]
    assert "--dependency=singleton" in opts and opts[opts.index("--job-name") + 1].startswith("trim_t_lane")
    assert "--open-mode=append" in opts and "--requeue" in opts
    lanes = {v["opts"][v["opts"].index("--job-name") + 1] for v in jobs.values()}
    assert lanes == {f"trim_t_lane{i}" for i in range(4)}
    adv = [v for v in slurm.jobs.values() if v["script"][-2] == "advance"]
    assert len(adv) == 1 and any(o.startswith("afterany:") for o in adv[0]["opts"])
    req = json.loads((sp.case_dir(f"{TOPO}_V2p50_t+1p00_z+0p0mm") / layout.REQUEST_FILE).read_text())
    assert req["theta_deg"] == 1.0 and req["steps"] == 979 and req["study"] == "trim_t"
    for j in jobs.values():  # Slurm does not create log dirs
        out = j["opts"][j["opts"].index("--output") + 1]
        assert Path(out).parent.is_dir()

    for _ in range(12):
        finish_all(sp, slurm)
        if ledger.advance(sp, slurm) == "done":
            break
    led = json.loads(sp.ledger_json.read_text())
    assert led["done"]
    for ch in led["chains"].values():
        assert ch["status"] == newton.CONVERGED, ch["history"][-1]
        assert abs(ch["result"]["R_lift_N"]) <= 1.0
    log = sp.progress_log.read_text()
    for word in ("CREATED", "SUBMITTED", "REDUCED", "STENCIL", "STEP", "CONVERGED", "DONE"):
        assert word in log, word
    rows = list(csv.DictReader(open(sp.results_csv, newline="")))
    assert rows and {"case_id", "chain", "iteration", "R_lift_N"} <= set(rows[0])


def test_advance_retries_then_marks_bad(site, fake_reduce):
    sp, st = new_study(site, max_retries=1)
    slurm = FakeSlurm()
    ledger.init_study(sp, st)
    ledger.advance(sp, slurm)
    finish_all(sp, slurm, state=None)  # jobs vanish without status.json (license death)
    ledger.advance(sp, slurm)
    assert "RETRY" in sp.progress_log.read_text()
    assert len(slurm.case_jobs()) == 6
    finish_all(sp, slurm, state=None)
    ledger.advance(sp, slurm)
    led = json.loads(sp.ledger_json.read_text())
    assert all(c["state"] == "bad" for c in led["cases"].values())
    assert led["chains"]["V2p50"]["status"] == newton.FAILED
    assert led["done"]


def test_advance_waits_on_live_jobs_and_supersedes_old_advance(site, fake_reduce):
    sp, st = new_study(site)
    slurm = FakeSlurm()
    ledger.init_study(sp, st)
    ledger.advance(sp, slurm)
    first_adv = json.loads(sp.ledger_json.read_text())["advance_job"]
    n = len(slurm.jobs)
    ledger.advance(sp, slurm)  # nothing finished
    assert len(slurm.case_jobs()) == 3
    assert first_adv in slurm.cancelled and len(slurm.jobs) == n + 1


def test_stop_and_resume(site, fake_reduce):
    sp, st = new_study(site)
    slurm = FakeSlurm()
    ledger.init_study(sp, st)
    ledger.advance(sp, slurm)
    ids = ledger.stop(sp, slurm)
    assert len(ids) == 4 and ledger.advance(sp, slurm) == "stopped"
    ledger.resume(sp, slurm)
    led = json.loads(sp.ledger_json.read_text())
    assert all(c["failures"] == 0 and c["job"] for c in led["cases"].values())
    assert len(slurm.case_jobs()) == 6


def test_topology_change_blocks_advance(site, fake_reduce):
    sp, st = new_study(site)
    ledger.init_study(sp, st)
    tj = layout.topology_dir(TOPO) / layout.TOPOLOGY_FILE
    tj.write_text(tj.read_text().replace('"g": 9.81', '"g": 9.8'))
    assert ledger.advance(sp, FakeSlurm()) == "error"


def test_seed_import_from_check_sweep(site):
    if not CHECK_RESULTS.exists():
        pytest.skip("check sweep results not present")
    topo = layout.load_topology(TOPO)
    st = make_study(speeds=[2.5])
    rows = list(csv.DictReader(open(CHECK_RESULTS, newline="")))
    kept, skipped = ledger.seed_rows(topo, st, rows)
    assert len(kept) == 2 and not skipped
    kept, skipped = ledger.seed_rows(topo, make_study(speeds=[3.0]), rows)
    assert not kept and all("speed" in s for s in skipped)


def test_seeded_chain_reuses_x0_seed(site, fake_reduce):
    if not CHECK_RESULTS.exists():
        pytest.skip("check sweep results not present")
    sp, st = new_study(site)
    topo = layout.load_topology(TOPO)
    seeds, _ = ledger.seed_rows(topo, st, list(csv.DictReader(open(CHECK_RESULTS, newline=""))))
    slurm = FakeSlurm()
    ledger.init_study(sp, st, seeds)
    ledger.advance(sp, slurm)
    # x0 = (0, 0) is a seed: the stencil only needs its two other points
    assert len(slurm.case_jobs()) == 2


def test_grid_study(site, fake_reduce):
    sp, st = new_study(site, mode="grid", x0=None, speeds=[2.5], grid={"theta": [0.0, 2.0], "z": [0.0]})
    slurm = FakeSlurm()
    ledger.init_study(sp, st)
    ledger.advance(sp, slurm)
    assert len(slurm.case_jobs()) == 2
    finish_all(sp, slurm)
    assert ledger.advance(sp, slurm) == "done"


def test_quality_gate_uses_absolute_drift():
    st = make_study()
    row = {"complete": True, "Fz_total": 163.3, "Fz_total_drift": 0.0112, "Mx_total": 6.22,
           "Mx_total_drift": 0.0599}  # the check smoketest's theta=0 case
    assert ledger.quality_ok(row, st)  # 1.8 N <= 2 x 1.0, 0.37 <= 2 x 0.25
    assert not ledger.quality_ok({**row, "Fz_total_drift": 0.02}, st)
    assert not ledger.quality_ok({**row, "complete": False}, st)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_hs_new_dry_run_and_create(site, monkeypatch, capsys):
    import hs

    hs.main(["new", "--topology", TOPO, "--study", "cli_t", "--speeds", "2.5", "--x0", "0,0", "--dry-run"])
    out = capsys.readouterr().out
    assert "979 steps" in out and "68/71 HPC licenses" in out and "nothing written" in out
    assert not layout.list_studies()

    fake = FakeSlurm()
    monkeypatch.setattr(ledger, "Slurm", lambda: fake)
    hs.main(["new", "--topology", TOPO, "--study", "cli_t", "--speeds", "2.5", "--x0", "0,0", "--yes"])
    sp = layout.find_study("cli_t")
    assert json.loads(sp.study_json.read_text())["sha1"]["template"]["name"] == f"{TOPO}_template.cas.h5"
    assert len(fake.case_jobs()) == 3
    hs.main(["status", "cli_t"])
    assert "V2p50" in capsys.readouterr().out
    hs.main(["path", "case-dir", "--study", "cli_t", "--case", "c1"])
    assert capsys.readouterr().out.strip() == str(sp.case_dir("c1"))


def test_hs_new_refuses_license_overuse(site):
    import hs

    with pytest.raises(SystemExit, match="HPC licenses"):
        hs.main(["new", "--topology", TOPO, "--study", "big", "--speeds", "2.5", "--x0", "0,0",
                 "--ntasks", "40", "--lanes", "3", "--dry-run"])


def test_smoke_study_overrides(site, capsys):
    import hs

    hs.main(["new", "--topology", TOPO, "--study", "smk", "--speeds", "2.5", "--x0", "0,0",
             "--smoke", "--dry-run"])
    out = capsys.readouterr().out
    assert "mit_quicktest" in out and "10 steps" in out and "SMOKE" in out


# ---------------------------------------------------------------------------
# run_case guards, sync packaging
# ---------------------------------------------------------------------------


def test_run_case_identity_guard(site, fake_reduce):
    import run_case

    sp, st = new_study(site)
    ledger.init_study(sp, st)
    ledger.advance(sp, FakeSlurm())
    cid = f"{TOPO}_V2p50_t+0p00_z+0p0mm"
    case = run_case.Case(sp.name, cid)
    topo = layout.load_topology(TOPO)
    tpl = layout.topology_file(TOPO, "template")
    run_case.check_identity(case, topo, tpl)
    tpl.write_bytes(b"rebuilt")
    with pytest.raises(RuntimeError, match="differs"):
        run_case.check_identity(case, topo, tpl)


def test_run_case_latest_autosave(tmp_path):
    import run_case

    for n in ("sw-1-00100.dat.h5", "sw-1-00200.dat.h5", "sw-stop-00250.dat.h5", "other.dat.h5"):
        (tmp_path / n).write_bytes(b"")
    assert run_case.latest_autosave(tmp_path).name == "sw-stop-00250.dat.h5"


def test_code_tarball():
    import sync

    try:
        files = sync.code_files()
    except SystemExit as exc:
        pytest.skip(f"git unavailable: {exc}")
    assert "hs.py" in files and not any(f.startswith(("tests/", "sweeps/")) for f in files)
    version = sync.version_info(files)
    tar = tarfile.open(fileobj=io.BytesIO(sync.code_tarball(files, version)), mode="r:gz")
    members = {m.name: m for m in tar.getmembers()}
    assert "VERSION" in members and members["bin/hs"].mode == 0o755
    assert members["cluster/case_job.sh"].mode == 0o755
    assert b"\r\n" not in tar.extractfile("cluster/case_job.sh").read()
    assert json.loads(tar.extractfile("VERSION").read())["content_sha1"] == version["content_sha1"]
