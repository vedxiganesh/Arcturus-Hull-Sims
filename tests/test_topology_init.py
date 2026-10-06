"""`hs topology init`: Scheme parsing, frame rules, and derivation from a real case."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from case_reader import scheme_pairs  # noqa: E402
from topology_init import frame_problems, rotation_to_bow_minus_y, snap_axis  # noqa: E402

AT2 = ROOT.parent / "hullsweep_data" / "at2_trimaran_halfd"


def test_scheme_pairs_keeps_nested_values_and_strings():
    text = '(37 (\n(a/b 1.5)\n(c? #t)\n(d ((x . "a (b) \\" c") (y 2)))\n(e "s")\n))'
    got = scheme_pairs(text)
    assert got == {"a/b": "1.5", "c?": "#t", "d": '((x . "a (b) \\" c") (y 2))', "e": '"s"'}


def test_snap_axis():
    assert snap_axis(np.array([0.0, -12.2, 0.01]), "v").tolist() == [0.0, -1.0, 0.0]
    with pytest.raises(Exception):
        snap_axis(np.array([1.0, 1.0, 0.0]), "v")


@pytest.mark.parametrize("bow, angle", [([0, -1, 0], 0.0), ([1, 0, 0], -90.0), ([-1, 0, 0], 90.0)])
def test_rotation_to_bow_minus_y(bow, angle):
    assert rotation_to_bow_minus_y(np.array(bow, dtype=float)) == pytest.approx(angle)
    a = np.radians(angle)
    R = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    assert np.allclose(R @ np.array(bow, dtype=float), [0, -1, 0])


def test_frame_rules():
    up = np.array([0.0, 0.0, 1.0])
    assert frame_problems(np.array([0.0, -1.0, 0.0]), up, 0, 0.0) == []
    # middle_foil as first meshed: bow +X, symmetry y = 0 -> one fix, the -90 deg rotation.
    p = frame_problems(np.array([1.0, 0.0, 0.0]), up, 1, 0.0)
    assert len(p) == 1 and "-90 deg about +Z" in p[0]
    assert any("normal to y" in s for s in frame_problems(np.array([0.0, -1.0, 0.0]), up, 1, 0.0))
    assert any("translate" in s for s in frame_problems(np.array([0.0, -1.0, 0.0]), up, 0, 0.01))
    assert "up is" in frame_problems(np.array([0.0, -1.0, 0.0]), np.array([0.0, 1.0, 0.0]), None, 0.0)[0]


@pytest.fixture(scope="module")
def at2_case():
    pytest.importorskip("h5py")
    from case_reader import CaseFile

    hits = sorted(AT2.glob("*template*.cas.h5"))
    if not hits:
        pytest.skip("at2 template case not present")
    with CaseFile(hits[0]) as c:
        yield c


def test_derive_reproduces_at2_topology(at2_case):
    """The fields at2's topology.json got by hand and by prepare_case (Fluent), read offline."""
    from topology_init import derive

    want = json.loads((AT2 / "topology.json").read_text())
    d = derive(at2_case, "at2_trimaran_halfd")
    assert not d.problems
    for k in ("half_domain", "g", "bow_direction", "up_direction", "free_surface_z", "hull_wall_zones",
              "background_cell_zone", "water_phase", "inlet_zone", "outlet_zone"):
        assert d.raw[k] == pytest.approx(want[k]) if isinstance(want[k], float) else d.raw[k] == want[k], k
    assert sorted(d.raw["foreground_cell_zones"]) == sorted(want["foreground_cell_zones"])
    got, ref = d.raw["template_hull_stats"], want["template_hull_stats"]
    assert got["n_vertices"] == ref["n_vertices"]
    assert np.allclose(got["centroid"], ref["centroid"], atol=1e-6)
    assert got["length"] == pytest.approx(ref["length"], abs=1e-6)
    assert d.raw["bottom_z"] == pytest.approx(-1.9855, abs=1e-3)  # mesh floor, not ht-bottom
    assert any("-1.69" in w for w in d.warnings)


def test_at2_hydrostatics_near_its_weight(at2_case):
    from topology_init import derive, hydrostatics

    d = derive(at2_case, "at2_trimaran_halfd")
    d.raw.update(mass_full_kg=35.0, cg_ref=[0.0, -0.236, -0.076])
    h = hydrostatics(at2_case, d.raw, d.raw["_rho_water"])
    assert h["volume_m3"] == pytest.approx(0.0185, abs=5e-4)
    assert abs(h["static_heave_m"]) < 0.02
