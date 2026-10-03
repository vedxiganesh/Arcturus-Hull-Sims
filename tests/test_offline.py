"""Offline tests: everything in hullsweep that does not need a Fluent session."""

from __future__ import annotations

import csv
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import collect  # noqa: E402
from common import (  # noqa: E402
    REPORT_FILE,
    Topology,
    case_id,
    case_transform,
    plan_run,
    report_groups,
    report_name,
)

TOPO_SRC = ROOT.parent / "hullsweep_data" / "at2_trimaran_halfd" / "topology.json"
ARCTURUS = ROOT.parent


@pytest.fixture
def topo(tmp_path) -> Topology:
    raw = json.loads(TOPO_SRC.read_text())
    raw.update(hull_zmin_at_ref=-0.20, hull_length_m=1.5)
    p = tmp_path / "topo.json"
    p.write_text(json.dumps(raw))
    return Topology.load(p)


# --- conventions ------------------------------------------------------------


def test_bow_up_is_negative_rotation_about_x(topo):
    assert topo.fluent_pitch_sign == -1.0
    xf = case_transform(topo, theta_deg=5.0, z_m=0.0)
    bow_pt = topo.cg_ref + np.array([0.0, -1.0, 0.0])  # 1 m ahead of the CG
    stern_pt = topo.cg_ref + np.array([0.0, 1.0, 0.0])
    assert xf.apply(bow_pt)[0, 2] > bow_pt[2]      # bow rises
    assert xf.apply(stern_pt)[0, 2] < stern_pt[2]  # stern drops
    assert np.allclose(xf.apply(topo.cg_ref)[0], topo.cg_ref)  # CG is the pivot


def test_symmetry_plane_preserved(topo):
    xf = case_transform(topo, theta_deg=4.0, z_m=0.03)
    pts = np.array([[0.0, -0.5, -0.1], [0.0, 0.7, 0.05]])
    assert np.allclose(xf.apply(pts)[:, 0], 0.0)


def test_heave_moves_moment_center(topo):
    xf = case_transform(topo, theta_deg=2.0, z_m=-0.02)
    assert np.allclose(xf.moment_center, topo.cg_ref + [0, 0, -0.02])
    assert np.allclose(xf.apply(topo.cg_ref)[0], xf.moment_center)


def test_half_domain_scaling(topo):
    assert topo.weight_domain_N == pytest.approx(35 * 9.81 / 2)
    assert topo.thrust_ceiling_domain_N == pytest.approx(100.0)
    # thrust line 0.238 below keel (-0.20) relative to CG z -0.076
    assert topo.thrust_offset_body() == pytest.approx(-0.20 - 0.238 + 0.076)


def test_plan_run_anchored_on_validated_run(topo):
    p = plan_run(topo, 2.5)
    assert p.dt_s == pytest.approx(0.005)
    p1 = plan_run(topo, 1.0)
    assert p1.dt_s == pytest.approx(0.0125)
    # constant Courant + duration in hull lengths -> same step count at every speed
    assert abs(p.steps - p1.steps) <= 1
    assert p1.end_time_s == pytest.approx(10 * 1.5 / 1.0)


def test_study_dt_scale_keeps_windows_and_rescales_steps(topo):
    import ledger

    base = ledger.case_plan(topo, {}, 2.5)
    half = ledger.case_plan(topo, {"dt_scale": 0.5}, 2.5)
    assert half["dt_s"] == pytest.approx(base["dt_s"] / 2)
    assert half["end_time_s"] == pytest.approx(base["end_time_s"])
    assert half["settle_time_s"] == pytest.approx(base["settle_time_s"])
    assert abs(half["steps"] - 2 * base["steps"]) <= 1


def test_study_absolute_dt_overrides_speed_rule(topo):
    import ledger

    for v in (1.0, 2.5):
        p = ledger.case_plan(topo, {"dt_s": 0.002, "dt_scale": 1.0}, v)
        assert p["dt_s"] == pytest.approx(0.002)
        assert p["steps"] == math.ceil(p["end_time_s"] / 0.002)


def test_case_id_is_safe_and_unique(topo):
    a = case_id(topo.name, 1.5, -1.0, 0.005)
    b = case_id(topo.name, 1.5, -1.0, 0.0)
    assert a != b
    assert all(ch.isalnum() or ch in "_+-" for ch in a)


# --- collect ----------------------------------------------------------------


def _write_report(path: Path, cols: list[str], rows: np.ndarray) -> None:
    head = ['"sweep-forces-rfile"', '"Time Step" "x etc.."',
            "(" + " ".join(f'"{c}"' for c in ["Time Step", *cols, "flow-time"]) + ")"]
    body = [" ".join(f"{v:.10g}" for v in r) for r in rows]
    path.write_text("\n".join(head + body) + "\n")


def _make_case(topo, tmp: Path, F, M, theta=3.0, z=0.01, V=2.0, n=400, noise=0.0, split=None):
    sc = {
        "case_id": case_id(topo.name, V, theta, z), "speed_mps": V, "theta_deg": theta, "z_m": z,
        "dt_s": 0.01, "steps": n, "settle_time_s": 1.0, "end_time_s": n * 0.01,
    }
    d = tmp / sc["case_id"]
    d.mkdir(parents=True)
    (d / f"{sc['case_id']}.json").write_text(json.dumps(sc))
    (d / "status.json").write_text(json.dumps({"state": "complete"}))
    cols, vals = [], []
    for g in report_groups(topo):
        share = 1.0 if g == "total" else 0.5
        for i, c in enumerate("xyz"):
            cols.append(report_name("f", c, g)); vals.append(F[i] * share)
            cols.append(report_name("m", c, g)); vals.append(M[i] * share)
    rng = np.random.default_rng(0)
    steps = np.arange(1, n + 1)
    data = np.column_stack([steps, np.tile(vals, (n, 1)) + noise * rng.standard_normal((n, len(vals))),
                            steps * 0.01])
    if split is None:
        _write_report(d / f"{REPORT_FILE}.out", cols, data)
    else:
        # attempt 1 ran to split+30, was preempted, and resumed from the
        # autosave at `split`; its rows split+1..split+30 are stale.
        part1 = data[: split + 30].copy()
        part1[split:, 1] += 999.0
        _write_report(d / f"{REPORT_FILE}.part1.out", cols, part1)
        _write_report(d / f"{REPORT_FILE}.out", cols, data[split:])
    return d, sc


def test_collect_residual_formulas(topo, tmp_path):
    F = np.array([0.0, 40.0, 180.0])   # drag +Y, lift +Z (half domain)
    M = np.array([-5.0, 0.0, 0.0])      # Fluent Mx about the displaced CG
    d, sc = _make_case(topo, tmp_path, F, M)
    r = collect.reduce_case(topo, d, (0.0, 0.0))
    th = math.radians(sc["theta_deg"])
    T = 40.0 / math.cos(th)
    arm = topo.thrust_offset_body()
    assert r["complete"] and r["status"] == "complete"
    assert r["drag_N"] == pytest.approx(40.0)
    assert r["thrust_N"] == pytest.approx(T)
    assert r["R_lift_N"] == pytest.approx(180.0 + T * math.sin(th) - topo.weight_domain_N)
    # bow-up positive: -Mx; a thrust line BELOW the CG adds bow-up moment
    assert r["Mbow_Nm"] == pytest.approx(5.0)
    assert r["R_pitch_Nm"] == pytest.approx(-(-5.0 + arm * T))
    assert -(arm * T) > 0
    assert r["thrust_util"] == pytest.approx(T / 100.0)
    assert r["Fz_mainhull"] == pytest.approx(90.0)


def test_collect_cg_offset_transfer(topo, tmp_path):
    F = np.array([0.0, 40.0, 180.0])
    M = np.array([-5.0, 0.0, 0.0])
    d, sc = _make_case(topo, tmp_path, F, M, theta=0.0)
    r0 = collect.reduce_case(topo, d, (0.0, 0.0))
    r1 = collect.reduce_case(topo, d, (0.1, 0.0))  # CG 0.1 m aft (+Y)
    # M' = M - d x F ; (d x F)_x = dy*Fz - dz*Fy = 0.1*180
    assert r1["Mbow_Nm"] == pytest.approx(r0["Mbow_Nm"] + 18.0)


def test_collect_merges_resume_parts(topo, tmp_path):
    F = np.array([0.0, 40.0, 180.0]); M = np.array([-5.0, 0.0, 0.0])
    d, _ = _make_case(topo, tmp_path, F, M, split=200)
    s = collect.load_series(d)
    assert s.n_rows == 400
    assert np.all(np.diff(s.index) == 1)
    # corrupted rows from attempt 1 (steps 201..230) must be overwritten by the resume
    r = collect.reduce_case(topo, d, (0.0, 0.0))
    assert r["drag_N"] == pytest.approx(40.0)


def test_collect_takes_flow_time_from_rfile(topo, tmp_path):
    F = np.array([0.0, 40.0, 180.0]); M = np.array([-5.0, 0.0, 0.0])
    d, _ = _make_case(topo, tmp_path, F, M)
    # rewrite sweep-forces.out without flow-time, as the cluster run produced it
    lines = (d / f"{REPORT_FILE}.out").read_text().splitlines()
    lines[2] = lines[2].replace(' "flow-time"', "")
    lines[3:] = [ln.rsplit(" ", 1)[0] for ln in lines[3:]]
    (d / f"{REPORT_FILE}.out").write_text("\n".join(lines) + "\n")
    # lift rfile stops 5 steps short: those sweep rows have no time and are dropped
    steps = np.arange(1, 396)
    (d / "report-lift-total-rfile.out").write_text("\n".join(
        ['"report-lift-total-rfile"', '"Time Step" "report-lift-total etc.."',
         '("Time Step" "report-lift-total" "flow-time")']
        + [f"{s} 180.0 {s * 0.01:.10g}" for s in steps]) + "\n")
    s = collect.load_series(d)
    assert s.transient and s.n_rows == 395
    assert s.flow_time == pytest.approx(steps * 0.01)
    r = collect.reduce_case(topo, d, (0.0, 0.0))
    assert r["t_last_s"] == pytest.approx(3.95)
    assert r["drag_N"] == pytest.approx(40.0)


def test_window_stats_noise():
    rng = np.random.default_rng(1)
    y = 10.0 + rng.standard_normal(2000)
    m, se, drift = collect.window_stats(y)
    assert m == pytest.approx(10.0, abs=0.1)
    assert 0.005 < se < 0.06
    assert drift < 0.05


def test_vendored_parser_reads_real_run_file():
    p = ARCTURUS / "Trimaran_Coarse_Halfd_run_22616142" / "report-lift-total-rfile.out"
    if not p.exists():
        pytest.skip("reference run file not present")
    from _vendor.reefs_postproc import dedupe_restarts, parse_report_file
    s = dedupe_restarts(parse_report_file(p))
    assert s.columns == ("Time Step", "report-lift-total", "flow-time")
    assert s.n_rows > 800


# --- manifest + batch ---------------------------------------------------------


def test_manifest_and_batch(topo, tmp_path):
    sweep = tmp_path / "sweeps" / "pilot"
    py = sys.executable
    subprocess.run([py, str(ROOT / "manifest.py"), "--topology", str(topo.path), "--sweep", str(sweep),
                    "--speeds", "1,2.5", "--theta", "0,2", "--z", "0"], check=True)
    raw = (sweep / "manifest.csv").read_bytes()
    assert b"\r\n" not in raw
    rows = list(csv.DictReader(open(sweep / "manifest.csv", newline="")))
    assert [int(r["index"]) for r in rows] == [0, 1, 2, 3]

    (sweep / "cases").mkdir()
    for r in rows:
        (sweep / "cases" / f"{r['case_id']}.cas.h5").write_bytes(b"")
        (sweep / "cases" / f"{r['case_id']}.json").write_text("{}")
    subprocess.run([py, str(ROOT / "make_batch.py"), "--sweep", str(sweep), "--concurrent", "3"],
                   check=True, capture_output=True)
    sh = (sweep / "submit_sweep.sh").read_text()
    assert "#SBATCH --array=0-3%3" in sh
    assert "@" not in "".join(l for l in sh.splitlines() if l.startswith("#SBATCH"))
    assert "export SWEEP_ROOT=\"$HOME/orcd/pool/hullsweep/pilot\"" in sh
    for f in ("run_sweep_case.py", "common.py", "fluent_ops.py", "topology.json"):
        assert (sweep / f).exists(), f
    assert b"\r\n" not in (sweep / "submit_sweep.sh").read_bytes()


def test_make_batch_enforces_license_pool(topo, tmp_path):
    sweep = tmp_path / "sweeps" / "big"
    py = sys.executable
    subprocess.run([py, str(ROOT / "manifest.py"), "--topology", str(topo.path), "--sweep", str(sweep),
                    "--speeds", "2.5", "--theta", "0", "--z", "0"], check=True)
    r = subprocess.run([py, str(ROOT / "make_batch.py"), "--sweep", str(sweep), "--allow-missing",
                        "--ntasks", "40", "--concurrent", "3"], capture_output=True, text=True)
    assert r.returncode != 0 and "HPC licenses" in r.stderr
    r = subprocess.run([py, str(ROOT / "make_batch.py"), "--sweep", str(sweep), "--allow-missing",
                        "--ntasks", "21", "--concurrent", "4"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    sh = (sweep / "submit_sweep.sh").read_text()
    assert "MAX_RESTARTS=12" in sh and "scontrol requeue" in sh


# --- fluent_ops against the REAL dumped inlet state ---------------------------


class FakeNode:
    """Minimal stand-in for a PyFluent settings node built from get_state()."""

    def __init__(self, state, log):
        self._state, self._log = state, log

    @property
    def child_names(self):
        return list(self._state) if isinstance(self._state, dict) else []

    def __getattr__(self, k):
        if k.startswith("_"):
            raise AttributeError(k)
        return FakeNode(self._state[k], self._log)

    def __getitem__(self, k):
        return FakeNode(self._state[k], self._log)

    def __setitem__(self, k, v):
        self._log.append(("set", k, v))
        self._state[k] = v

    def get_state(self):
        return self._state

    def get_object_names(self):
        return list(self._state)

    def set_state(self, v):
        self._log.append(("set_state", v))
        if isinstance(self._state, dict) and isinstance(v, dict):
            self._state.update(v)


def _dumped(name):
    p = ROOT / "introspection" / "hull_6sec" / name
    if not p.exists():
        pytest.skip("introspection dump not present")
    return json.loads(p.read_text())


def test_set_inlet_speed_on_dumped_state():
    import fluent_ops as fo
    bc = _dumped("state.setup.boundary_conditions.json")
    log = []
    solver = type("S", (), {})()
    solver.settings = FakeNode({"setup": {"boundary_conditions": bc}}, log)
    done = fo.set_inlet_speed(solver, "inlet", 1.5)
    assert done == ["phase/mixture/multiphase/vmag"]
    assert bc["pressure_inlet"]["inlet"]["phase"]["mixture"]["multiphase"]["vmag"] == {"option": "value", "value": 1.5}


def test_init_defaults_rescale_on_dumped_state():
    import fluent_ops as fo
    init = _dumped("state.solution.initialization.json")
    k0, w0 = init["defaults"]["k"], init["defaults"]["omega"]
    log = []

    class Init(FakeNode):
        def compute_defaults(self, **kw):  # simulate the command being unavailable
            raise RuntimeError("no such command")

    solver = type("S", (), {})()
    solver.settings = type("T", (), {})()
    solver.settings.solution = type("U", (), {})()
    solver.settings.solution.initialization = Init(init, log)
    out = fo.update_init_defaults(solver, "inlet", 1.0)
    assert out["y-velocity"] == pytest.approx(1.0)
    assert out["x-velocity"] == 0 and out["z-velocity"] == 0
    assert out["k"] == pytest.approx(k0 * (1.0 / 2.5) ** 2)
    assert out["omega"] == pytest.approx(w0 * (1.0 / 2.5) ** 2)


def test_rotate_zone_arg_mapping_matches_252_api():
    import fluent_ops as fo
    got = fo._match_args(["zone_names", "rotation_angle", "origin", "axis"],
                         {"zones": ("zone_names", "zones"), "angle": ("rotation_angle", "angle"),
                          "origin": ("origin",), "axis": ("axis",)})
    assert got == {"zones": "zone_names", "angle": "rotation_angle", "origin": "origin", "axis": "axis"}


def test_driver_autosave_regex():
    import re
    rx = re.compile(r"^sw-.*?(\d+)\.dat\.h5$")
    assert rx.match("sw-1-00200.dat.h5").group(1) == "00200"
    assert rx.match("sw-stop-00350.dat.h5").group(1) == "00350"


def test_settings_rotate_zone_gets_radians(topo):
    """Regression: settings rotate_zone is SI (radians); degrees flipped the hull."""
    import fluent_ops as fo

    calls = {}

    class Cmd:
        def __init__(self, name, args):
            self.name, self.argument_names = name, args

        def __call__(self, **kw):
            calls[self.name] = kw

    class MZ:
        command_names = ["rotate_zone", "translate_zone"]
        child_names = []
        rotate_zone = Cmd("rotate", ["zone_names", "rotation_angle", "origin", "axis"])
        translate_zone = Cmd("translate", ["zone_names", "offset"])

    solver = type("S", (), {})()
    solver.settings = type("T", (), {})()
    solver.settings.mesh = type("M", (), {})()
    solver.settings.mesh.modify_zones = MZ()
    xf = case_transform(topo, theta_deg=3.0, z_m=-0.02)
    assert fo.rotate_translate(solver, ["a", "b"], xf) == "settings"
    assert calls["rotate"]["rotation_angle"] == pytest.approx(math.radians(-3.0))
    assert calls["translate"]["offset"] == pytest.approx([0.0, 0.0, -0.02])

    # and the analytic check agrees with a -3 deg rotation, not -3 rad
    c = np.array(topo.cg_ref) + [0.1, -0.1, -0.02]
    before = {"centroid": c.tolist()}
    fo.verify_transform(before, {"centroid": xf.apply(c)[0].tolist()}, xf)
    wrong = case_transform(topo, theta_deg=math.degrees(3.0), z_m=-0.02).apply(c)[0]
    with pytest.raises(RuntimeError):
        fo.verify_transform(before, {"centroid": wrong.tolist()}, xf)


def test_check_water_level_logic(topo):
    """Fake solver: box z in [-2, 1], free surface -0.076 -> expected 0.6413."""
    import fluent_ops as fo

    raw = dict(topo.raw, background_cell_zone="background_mesh", water_phase="phase-2")
    topo.raw.update(raw)
    exp = (topo["free_surface_z"] + 2.0) / 3.0

    def run(measured):
        class Field:
            def allowed_values(self):
                return ["pressure", "vof", "velocity-magnitude"]

        class Rep:
            field = Field()
            phase = None

        class Vol(dict):
            def get_object_names(self):
                return list(self)

            def __setitem__(self, k, v):
                dict.__setitem__(self, k, Rep())

        class RD:
            volume = Vol()

            def compute(self, report_defs):
                return [{report_defs[0]: [measured, 0]}]

        solver = type("S", (), {})()
        solver.settings = type("T", (), {})()
        solver.settings.solution = type("U", (), {})()
        solver.settings.solution.report_definitions = RD()
        box = np.array([[0, 0, -2.0], [0, 0, 1.0]])
        orig = fo.surface_vertices
        fo.surface_vertices = lambda s, zones: {z: box for z in zones}
        try:
            return fo.check_water_level(solver, topo)
        finally:
            fo.surface_vertices = orig

    out = run(exp + 0.01)
    assert out["ok"] and out["field"] == "vof" and out["expected"] == pytest.approx(exp)
    with pytest.raises(RuntimeError):
        run(0.0)  # all-air interior


def test_report_file_includes_flow_time():
    """The grouped report file lists the built-in flow-time report last, as the
    GUI-made files in the dumped 6 s case do."""
    import fluent_ops as fo

    dump = ROOT / "introspection" / "hull_6sec"
    rd_state = json.loads((dump / "state.solution.report_definitions.json").read_text())
    gui = json.loads((dump / "state.solution.monitor.report_files.json").read_text())
    assert all(f["report_defs"][-1] == fo.FLOW_TIME_REPORT for f in gui.values())

    class Time:
        def get_state(self):
            return rd_state["time"]

    class RF(dict):
        def get_object_names(self):
            return list(self)

    solver = type("S", (), {})()
    solver.settings = type("T", (), {})()
    solver.settings.solution = type("U", (), {})()
    solver.settings.solution.report_definitions = type("RD", (), {"time": Time()})()
    solver.settings.solution.monitor = type("M", (), {"report_files": RF()})()
    fo._one_report_file(solver, ["sw-fx-total", "sw-mx-total"])
    rf = solver.settings.solution.monitor.report_files[REPORT_FILE]
    assert rf["report_defs"] == ["sw-fx-total", "sw-mx-total", "flow-time"]
    assert rf["file_name"] == f"{REPORT_FILE}.out"
