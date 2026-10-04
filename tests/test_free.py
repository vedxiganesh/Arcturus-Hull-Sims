"""Offline tests for the free-running 2DOF relaxation: release physics, UDF, reduction,
the Newton -> free hand-off, --from-study, and the Fluent-side setup against a fake tree."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import collect  # noqa: E402
import layout  # noqa: E402
import ledger  # noqa: E402
import newton  # noqa: E402
from common import INERTIA_2DOF_C, Release, free_case_id, free_release, udf_source  # noqa: E402
from test_studies import TOPO, FakeSlurm, fake_reduce, finish_all, new_study, site  # noqa: E402,F401

FREE = {"relax_on": ["CONVERGED"], "tol_theta_deg": 0.1, "tol_z_m": 0.002, "settle_hull_lengths": 2.0,
        "average_hull_lengths": 2.0, "export_every_s": 0.1, "thrust": "constant",
        "inertia_full_kgm2": dict(INERTIA_2DOF_C)}


@pytest.fixture
def topo(site):
    return layout.load_topology(TOPO)


# ---------------------------------------------------------------------------
# release and UDF
# ---------------------------------------------------------------------------


def test_release_reproduces_the_captive_thrust_terms(topo):
    """collect.py: R_lift has +T sin(theta), R_pitch has sign*(arm*T) in Fluent-X terms."""
    T, arm, th = 20.0, -0.4, 3.0
    rel = free_release(topo, th, -0.01, thrust_N=T, thrust_arm_m=arm)
    assert rel.load_force_local[2] == pytest.approx(T * math.sin(math.radians(th)))
    assert rel.load_moment_local[0] == pytest.approx(arm * T)
    assert rel.load_moment_local[1:] == pytest.approx((0.0, 0.0), abs=1e-12)
    assert rel.cg == pytest.approx(tuple(topo.cg_ref + np.array([0.0, 0.0, -0.01])))
    assert rel.mass_kg == pytest.approx(17.5)  # half domain
    assert rel.inertia_kgm2["ixx"] == pytest.approx(INERTIA_2DOF_C["ixx"] / 2)
    assert Release.from_dict(json.loads(json.dumps(rel.to_dict()))) == rel


def test_release_cg_offset_turns_with_the_hull(topo):
    rel = free_release(topo, 90.0, 0.0, cg_offset=(0.1, 0.0))  # bow-up 90: body +Y points up
    # body +Y is stern-ward (bow is -Y); pitched bow-up 90 deg, stern points down
    assert rel.cg[2] == pytest.approx(topo.cg_ref[2] - 0.1)


def test_udf_source_constrains_to_heave_and_pitch(topo):
    src = udf_source(free_release(topo, 1.0, 0.0, thrust_N=10.0, thrust_arm_m=-0.3))
    assert "DEFINE_SDOF_PROPERTIES(stage," in src and "prop[SDOF_MASS] = 17.5;" in src
    for dof, val in (("TRANS_X", "TRUE"), ("TRANS_Y", "TRUE"), ("TRANS_Z", "FALSE"),
                     ("ROT_X", "FALSE"), ("ROT_Y", "TRUE"), ("ROT_Z", "TRUE")):
        assert f"prop[SDOF_ZERO_{dof}] = {val};" in src
    assert "prop[SDOF_LOAD_LOCAL] = TRUE;" in src and "prop[SDOF_LOAD_M_X] = -3;" in src


# ---------------------------------------------------------------------------
# reduction
# ---------------------------------------------------------------------------


def _free_case_dir(tmp_path, topo, theta=2.0, z=-0.01):
    cid = free_case_id(f"{TOPO}_V2p50_t+2p00_z-10p0mm")
    d = tmp_path / cid
    d.mkdir()
    rel = free_release(topo, theta, z, thrust_N=12.0, thrust_arm_m=-0.4)
    (d / f"{cid}.json").write_text(json.dumps({
        "case_id": cid, "parent_case_id": cid[:-5], "speed_mps": 2.5, "theta_deg": theta, "z_m": z,
        "dt_s": 0.005, "steps": 400, "settle_time_s": 1.0, "end_time_s": 2.0, "t_release_s": 4.9,
        "release": rel.to_dict()}))
    (d / "status.json").write_text(json.dumps({"state": "complete"}))
    return d, rel


def _six_dof(path, t, cgz, thx):
    lines = ["# Fluent dynamic mesh motion history", "#    time CG_X CG_Y CG_Z THETA_X THETA_Y THETA_Z", "#"]
    lines += [f" {a:.5e}  0.0 -0.236 {b:.6e} {c:.6e} 0.0 0.0" for a, b, c in zip(t, cgz, thx)]
    path.write_text("\n".join(lines) + "\n")


def test_reduce_free_case_merges_restarts_and_converts_conventions(tmp_path, topo):
    d, rel = _free_case_dir(tmp_path, topo)
    t = 4.9 + np.arange(0, 401) * 0.005
    cgz = rel.cg[2] + 0.001 * (1 - np.exp(-(t - 4.9) / 0.2))  # settles 1 mm up
    thx = -0.05 * (1 - np.exp(-(t - 4.9) / 0.2))  # Fluent -X rotation = bow up (sign -1)
    _six_dof(d / "sw-motion_stage.part1.6dof", t[:250] , cgz[:250], thx[:250] + 9.0)  # superseded tail
    _six_dof(d / "sw-motion_stage.6dof", t[200:], cgz[200:], thx[200:])  # restart from step 200
    row = collect.reduce_free_case(topo, d)
    assert row["motion_source"] == "sw-motion_stage.6dof" and row["complete"]
    assert row["dtheta_deg"] == pytest.approx(0.05, abs=2e-3)
    assert row["dz_m"] == pytest.approx(0.001, abs=5e-5)
    assert row["n_avg"] == 201  # t >= 4.9 + 1.0
    assert abs(row["theta_drift_deg"]) < 1e-3
    assert ledger.free_verdict(row, FREE)[0] == ledger.VERIFIED


def test_reduce_free_case_falls_back_to_motion_log(tmp_path, topo):
    d, rel = _free_case_dir(tmp_path, topo)
    lines = ["flow_time,time_step,cg_x,cg_y,cg_z,theta_x_deg"]
    for k in range(0, 401, 10):
        lines.append(f"{4.9 + k * 0.005},{979 + k},0,-0.236,{rel.cg[2] - 0.005},{0.3}")
    (d / "sw-motion.csv").write_text("\n".join(lines) + "\n")
    row = collect.reduce_free_case(topo, d)
    assert row["motion_source"] == "sw-motion.csv"
    assert row["dtheta_deg"] == pytest.approx(-0.3) and row["dz_m"] == pytest.approx(-0.005)
    verdict, why = ledger.free_verdict(row, FREE)
    assert verdict == ledger.DRIFTED and "dz -5.00 mm" in why


def test_free_verdicts():
    base = {"complete": True, "theta_mean_deg": 2.0, "dtheta_deg": 0.01, "dz_m": 0.0005,
            "theta_drift_deg": 0.02, "z_drift_m": 0.0003, "t_last_s": 7.0}
    assert ledger.free_verdict(base, FREE)[0] == ledger.VERIFIED
    assert ledger.free_verdict({**base, "dtheta_deg": 0.5}, FREE)[0] == ledger.DRIFTED
    assert ledger.free_verdict({**base, "z_drift_m": 0.01}, FREE)[0] == ledger.UNSETTLED
    assert ledger.free_verdict({**base, "complete": False}, FREE)[0] == ledger.FREE_FAILED
    assert ledger.free_verdict({"note": "no 6DOF motion history"}, FREE) == (ledger.FREE_FAILED,
                                                                            "no 6DOF motion history")


# ---------------------------------------------------------------------------
# orchestrator: Newton -> free, and --from-study
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_free_reduce(monkeypatch):
    rows = {}

    def reduce_free_case(topo, case_dir):
        req = json.loads((case_dir / layout.REQUEST_FILE).read_text())
        return {"case_id": req["case_id"], "kind": "free", "complete": True,
                "theta_mean_deg": req["theta_deg"] + rows.get("dtheta", 0.02), "dtheta_deg": rows.get("dtheta", 0.02),
                "z_mean_m": req["z_m"], "dz_m": 0.0004, "theta_drift_deg": 0.01, "z_drift_m": 0.0002,
                "theta_se_deg": 0.004, "z_se_m": 1e-4, "motion_source": "sw-motion_stage.6dof"}

    monkeypatch.setattr(ledger.collect, "reduce_free_case", reduce_free_case)
    return rows


def _run_to_free(sp, slurm):
    for _ in range(12):
        finish_all(sp, slurm)
        ledger.advance(sp, slurm)
        led = json.loads(sp.ledger_json.read_text())
        if any(c.get("kind") == "free" for c in led["cases"].values()) or led["done"]:
            return led
    raise AssertionError("never reached the free phase")


def test_newton_study_relaxes_after_convergence(site, fake_reduce, fake_free_reduce):
    sp, st = new_study(site, free=FREE)
    slurm = FakeSlurm()
    ledger.init_study(sp, st)
    ledger.advance(sp, slurm)
    led = _run_to_free(sp, slurm)
    ch = led["chains"]["V2p50"]
    assert ch["status"] == "active" and ch["phase"] == "free" and ch["newton_status"] == newton.CONVERGED
    fcid = free_case_id(ch["result"]["case_id"])
    assert ch["pending"] == [fcid] and led["cases"][fcid]["kind"] == "free"

    req = json.loads((sp.case_dir(fcid) / layout.REQUEST_FILE).read_text())
    assert req["kind"] == "free" and req["parent_case_id"] == ch["result"]["case_id"]
    assert req["parent_cas"].endswith(f"{ch['result']['case_id']}_final.cas.h5")
    assert Path(req["parent_cas"]).parent == sp.case_dir(ch["result"]["case_id"])
    assert req["steps"] == math.ceil(4 * 1.2225950956344604 / 2.5 / 0.005)  # 2 + 2 hull lengths
    assert req["release"]["mass_kg"] == 17.5 and req["export_every_s"] == 0.1
    assert "RELAX" in sp.progress_log.read_text()

    finish_all(sp, slurm)
    assert ledger.advance(sp, slurm) == "done"
    led = json.loads(sp.ledger_json.read_text())
    ch = led["chains"]["V2p50"]
    assert ch["status"] == ledger.VERIFIED and ch["result"]["free"]["verdict"] == ledger.VERIFIED
    assert "R_lift_N" in ch["result"]  # the captive result is kept
    # the free point never enters Newton's fit
    assert all(p.case_id != fcid for p in ledger.points_for(led, 2.5, st))
    assert "VERIFIED" in sp.progress_log.read_text() and "free:" in ledger.status_text(sp)


def test_extend_reopens_a_finished_free_case(site, fake_reduce, fake_free_reduce):
    sp, st = new_study(site, free=FREE)
    slurm = FakeSlurm()
    ledger.init_study(sp, st)
    ledger.advance(sp, slurm)
    led = _run_to_free(sp, slurm)
    fcid = led["chains"]["V2p50"]["pending"][0]
    cdir = sp.case_dir(fcid)
    steps0 = json.loads((cdir / layout.REQUEST_FILE).read_text())["steps"]

    # not finished yet: refused
    with pytest.raises(ValueError, match="only finished"):
        ledger.extend(sp, None, 10, None, slurm)

    # the fake run: final files, a sidecar, and a complete status at step `steps0`
    for ext in (".cas.h5", ".dat.h5"):
        (cdir / f"{fcid}_final{ext}").write_bytes(b"x")
    (cdir / f"{fcid}.json").write_text(json.dumps({"steps": steps0, "end_time_s": steps0 * 0.005,
                                                   "step_release": 5000}))
    finish_all(sp, slurm, time_step=steps0)
    assert ledger.advance(sp, slurm) == "done"

    ledger.extend(sp, None, None, 1.0, slurm)  # +1 s at dt 0.005 = 200 steps
    led = json.loads(sp.ledger_json.read_text())
    c = led["cases"][fcid]
    assert c["state"] == ledger.LIVE_CASE and c["extra_steps"] == 200 and not led["done"]
    assert led["chains"]["V2p50"]["status"] == "active" and c["job"]
    n = 5000 + steps0  # the final state is now the newest autosave pair
    assert (cdir / f"sw-stop-{n:05d}.cas.h5").exists() and (cdir / f"sw-stop-{n:05d}.dat.h5").exists()
    req = json.loads((cdir / layout.REQUEST_FILE).read_text())
    assert req["steps"] == steps0 + 200 and req["end_time_s"] == pytest.approx((steps0 + 200) * 0.005)
    assert json.loads((cdir / f"{fcid}.json").read_text())["steps"] == steps0 + 200
    assert "EXTEND" in sp.progress_log.read_text()

    # and it finishes again, adding to (not replacing) an earlier extension
    finish_all(sp, slurm, time_step=steps0 + 200)
    assert ledger.advance(sp, slurm) == "done"
    ledger.extend(sp, [fcid], 50, None, slurm)
    assert json.loads(sp.ledger_json.read_text())["cases"][fcid]["extra_steps"] == 250


def test_free_plan_extra_steps(topo):
    st = {"free": FREE}
    base = ledger.free_plan(topo, st, 2.5)
    more = ledger.free_plan(topo, st, 2.5, 200)
    assert more["steps"] == base["steps"] + 200
    assert more["end_time_s"] == pytest.approx(more["steps"] * base["dt_s"])
    assert more["settle_time_s"] == base["settle_time_s"]


def test_newton_study_without_free_is_unchanged(site, fake_reduce):
    sp, st = new_study(site)  # study.json of existing studies has no "free"
    slurm = FakeSlurm()
    ledger.init_study(sp, st)
    ledger.advance(sp, slurm)
    led = _run_to_free(sp, slurm)
    assert led["done"] and led["chains"]["V2p50"]["status"] == newton.CONVERGED
    assert not any(c.get("kind") == "free" for c in led["cases"].values())


def _converged_parent(site, fake_reduce_unused=None, name="trim_p"):
    sp, st = new_study(site, name=name)
    slurm = FakeSlurm()
    ledger.init_study(sp, st)
    ledger.advance(sp, slurm)
    led = _run_to_free(sp, slurm)
    best = led["chains"]["V2p50"]["result"]["case_id"]
    return sp, best


def test_free_sources_need_final_files(site, fake_reduce):
    sp, best = _converged_parent(site)
    srcs, problems = ledger.free_sources(sp, ["CONVERGED"])
    assert not srcs and "missing" in problems[0]
    for ext in (".cas.h5", ".dat.h5"):
        (sp.case_dir(best) / f"{best}_final{ext}").write_bytes(b"x")
    srcs, problems = ledger.free_sources(sp, ["CONVERGED"])
    assert [s["case_id"] for s in srcs] == [best] and not problems
    assert srcs[0]["row"]["thrust_N"] == 12.0 and srcs[0]["theta"] == pytest.approx(1.7, abs=0.2)
    srcs, problems = ledger.free_sources(sp, ["CONVERGED"], cases=["nope"])
    assert not srcs and "not a finished captive case" in problems[0]


def test_hs_new_from_study(site, fake_reduce, fake_free_reduce, monkeypatch, capsys):
    import hs

    sp, best = _converged_parent(site)
    for ext in (".cas.h5", ".dat.h5"):
        (sp.case_dir(best) / f"{best}_final{ext}").write_bytes(b"x")
    hs.main(["new", "--topology", TOPO, "--study", "trim_p_free", "--from-study", "trim_p", "--dry-run"])
    out = capsys.readouterr().out
    assert "parent       trim_p" in out and best in out and "mass 17.5 kg" in out and "EnSight every 0.1" in out

    fake = FakeSlurm()
    monkeypatch.setattr(ledger, "Slurm", lambda: fake)
    hs.main(["new", "--topology", TOPO, "--study", "trim_p_free", "--from-study", "trim_p", "--yes"])
    fsp = layout.find_study("trim_p_free")
    study = json.loads(fsp.study_json.read_text())
    assert study["mode"] == "free" and study["speeds"] == [2.5]
    jobs = fake.case_jobs()
    assert len(jobs) == 1
    fcid = free_case_id(best)
    req = json.loads((fsp.case_dir(fcid) / layout.REQUEST_FILE).read_text())
    assert Path(req["parent_cas"]).is_file() and req["parent_study"] == "trim_p"

    finish_all(fsp, fake)
    assert ledger.advance(fsp, fake) == "done"
    assert json.loads(fsp.ledger_json.read_text())["chains"]["V2p50"]["status"] == ledger.VERIFIED


def test_hs_new_rejects_bad_free_combinations(site):
    import hs

    with pytest.raises(SystemExit, match="takes no --x0"):
        hs.main(["new", "--topology", TOPO, "--study", "x", "--from-study", "p", "--x0", "0,0", "--dry-run"])
    with pytest.raises(SystemExit, match="follows Newton"):
        hs.main(["new", "--topology", TOPO, "--study", "x", "--speeds", "2.5", "--theta", "0", "--z", "0",
                 "--relax", "--dry-run"])


def test_relax_plan_printed(site, capsys):
    import hs

    hs.main(["new", "--topology", TOPO, "--study", "rlx", "--speeds", "2.5", "--x0", "0,0", "--relax",
             "--dry-run"])
    out = capsys.readouterr().out
    assert "relax        after CONVERGED" in out and "thrust       constant" in out


# ---------------------------------------------------------------------------
# Fluent side, against a fake settings tree
# ---------------------------------------------------------------------------


class FakeZones:
    def __init__(self, motion_def_override=None):
        self.objs, self.override = {}, motion_def_override

    def get_object_names(self):
        return list(self.objs)

    def __setitem__(self, name, state):
        s = json.loads(json.dumps(state))
        if self.override:
            s["motion"]["motion_def"] = self.override
        self.objs[name] = s

    def __delitem__(self, name):
        del self.objs[name]

    def get_state(self):
        return json.loads(json.dumps(self.objs))


class Obj:
    def __init__(self, **kw):
        self.__dict__.update(kw)

    def get_state(self):
        return {}


class Group(Obj):
    def get_state(self):
        return dict(self.__dict__)


def _fake_solver(zones):
    six = Obj()
    dm = Obj(options=Obj(six_dof=six, implicit_update=Group()), dynamic_zones=zones)
    return Obj(settings=Obj(setup=Obj(dynamic_mesh=dm))), dm, six


def test_arm_six_dof_implicit_update(topo):
    import fluent_ops as fo

    rel = free_release(topo, 2.0, -0.01)
    solver, dm, _ = _fake_solver(FakeZones())
    facts = fo.arm_six_dof(solver, topo, rel, {"enabled": True, "mode": "auto", "k_over_L": 0.09,
                                              "update_interval": 1, "relaxation_factor": 0.1,
                                              "residual_criterion": 1e-5})
    iu = dm.options.implicit_update
    assert iu.enabled and iu.relaxation_factor == 0.1 and iu.update_interval == 1
    assert not hasattr(iu, "mode") and not hasattr(iu, "k_over_L")  # bookkeeping stays out of Fluent
    assert facts["implicit_update"]["enabled"]
    solver, dm, _ = _fake_solver(FakeZones())
    assert fo.arm_six_dof(solver, topo, rel)["implicit_update"] == {"enabled": False}
    assert dm.options.implicit_update.enabled is False


def test_implicit_auto_follows_gyration_radius(site, capsys):
    import hs
    from common import pitch_gyration_ratio

    topo = layout.load_topology(TOPO)
    assert pitch_gyration_ratio(topo, 0.439) == pytest.approx(0.0916, abs=1e-3)
    hs.main(["new", "--topology", TOPO, "--study", "rlx", "--speeds", "2.5", "--x0", "0,0", "--relax",
             "--dry-run"])
    out = capsys.readouterr().out
    assert "Ixx 0.2195" in out and "implicit     ON (auto; every 1 iteration(s), relaxation 0.1)" in out
    hs.main(["new", "--topology", TOPO, "--study", "rlx", "--speeds", "2.5", "--x0", "0,0", "--relax",
             "--ixx-full", "137.39", "--dry-run"])
    assert "implicit     off (auto)" in capsys.readouterr().out
    hs.main(["new", "--topology", TOPO, "--study", "rlx", "--speeds", "2.5", "--x0", "0,0", "--relax",
             "--implicit-6dof", "off", "--dry-run"])
    assert "implicit     off (off)" in capsys.readouterr().out


def test_free_request_carries_implicit_update(site, fake_reduce, fake_free_reduce):
    iu = {"enabled": True, "mode": "on", "k_over_L": 0.09, "update_interval": 2, "relaxation_factor": 0.2,
          "residual_criterion": 1e-5}
    sp, st = new_study(site, free={**FREE, "implicit_update": iu})
    slurm = FakeSlurm()
    ledger.init_study(sp, st)
    ledger.advance(sp, slurm)
    led = _run_to_free(sp, slurm)
    fcid = next(c for c, v in led["cases"].items() if v.get("kind") == "free")
    assert json.loads((sp.case_dir(fcid) / layout.REQUEST_FILE).read_text())["implicit_update"] == iu


def test_arm_six_dof_zones_and_read_back(topo):
    import fluent_ops as fo

    rel = free_release(topo, 2.0, -0.01)
    zones = FakeZones()
    solver, dm, six = _fake_solver(zones)
    facts = fo.arm_six_dof(solver, topo, rel)
    assert dm.enabled and six.enabled and six.write_motion_history and six.basename == fo.MOTION_BASENAME
    assert six.gravity == {"x": 0.0, "y": 0.0, "z": -9.81}
    st = zones.get_state()
    by_zone = {z["zone"]: z for z in st.values()}
    assert set(by_zone) == {"wall_mainhull", "wall_amas", "foreground_component_mesh", "fluid:1"}
    assert not by_zone["wall_mainhull"]["motion"]["six_dof"]["passive"]
    assert by_zone["fluid:1"]["motion"]["six_dof"]["passive"]
    assert by_zone["wall_amas"]["motion"]["rigid_body_properties"]["cg_position"] == pytest.approx(list(rel.cg))
    assert set(facts["dynamic_zone_routes"].values()) == {"setitem"}

    s = fo.read_sdof_state(Obj(settings=solver.settings), "wall_mainhull")
    assert s["cg"] == pytest.approx(list(rel.cg)) and s["theta_x_deg"] == 0.0

    solver, _, _ = _fake_solver(FakeZones(motion_def_override="none"))
    with pytest.raises(fo.CaseSetupError, match="read back"):
        fo.arm_six_dof(solver, topo, rel)


def test_ensight_command_registers_every_n_steps():
    import fluent_ops as fo

    cmd = fo.ensight_command("phase-2", 0.1, 0.005, "sw-ensight-77")
    assert cmd == ('/file/transient-export/ensight-gold-transient ensight/free () * () pressure wall-shear '
                   'phase-2-vof cell-convective-courant-number moving-mesh-courant-number q no yes '
                   '"sw-ensight-77" "time-step" 20 yes')


def test_free_resume_needs_a_case_with_the_data(tmp_path):
    import run_case

    for n in ("sw-1-01079.cas.h5", "sw-1-01079.dat.h5", "sw-1-01179.dat.h5"):
        (tmp_path / n).write_bytes(b"")
    assert run_case.latest_autosave(tmp_path).name == "sw-1-01179.dat.h5"
    assert run_case.latest_autosave(tmp_path, with_case=True).name == "sw-1-01079.dat.h5"
    assert run_case.case_of(tmp_path / "sw-stop-01200.dat.h5").name == "sw-stop-01200.cas.h5"


def test_set_aside_keeps_motion_and_ensight_index(tmp_path):
    import run_case

    (tmp_path / "ensight").mkdir()
    for n in ("sweep-forces.out", "sw-motion_stage.6dof", "ensight/free.encas", "sw-motion.csv"):
        (tmp_path / n).write_text("x")
    run_case.set_aside_report_files(tmp_path)
    names = {p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*") if p.is_file()}
    assert names == {"sweep-forces.part1.out", "sw-motion_stage.part1.6dof", "ensight/free.part1.encas",
                     "sw-motion.csv"}



def test_ensight_merge_unions_sessions(tmp_path):
    import ensight_merge as em

    def idx(path, nums, t0):
        times = " ".join(f"{t0 + 0.1 * i:.5e}" for i in range(len(nums)))
        path.write_text('FORMAT\ntype:  ensight gold\nGEOMETRY\nmodel:  1   "free*****.geo"\nTIME\n'
                        f"time set: 1 Model\nnumber of steps: {len(nums)}\nfilename start number: {nums[0]}\n"
                        f'filename increment: 20\ntime values: {times}\nSCRIPTS\nmetadata: "free.xml"\n')
        for n in nums:
            (tmp_path / f"free{n:05d}.geo").write_text("g")

    idx(tmp_path / "free.part1.encas", [20, 40, 60], 0.1)
    idx(tmp_path / "free.encas", [100, 120], 0.5)
    got = em.parse_frames(em.merge_case(tmp_path))
    assert sorted(got) == [20, 40, 60, 100, 120] and got[120] == pytest.approx(0.6)
