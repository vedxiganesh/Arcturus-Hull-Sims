"""Read a Fluent .cas.h5 offline: zones, boundary settings, wall geometry.

h5py only: no Fluent session and no license, so it runs from any shell,
including the one Fluent's web licensing refuses (CLAUDE.md §12). Used by
`hs topology init` (topology_init.py) to take topology.json fields from the
case itself instead of copying them from another topology.

Layout of a Fluent 25R2 HDF5 case (VERIFIED on the at2, at3 and middle_foil
cases, 2026-10-05):

  meshes/1/nodes/coords/<id>             (n, 3) node coordinates (one node zone)
  meshes/1/faces/zoneTopology            id, name (';'-joined), minId, maxId per face zone
  meshes/1/faces/nodes/1/{nnodes,nodes}  flat face -> node lists, 1-based node ids
  meshes/1/faces/c0/1                    cell on the c0 side of each face, 1-based
  meshes/1/cells/zoneTopology            the same for cell zones
  settings/Rampant Variables             '(37 ((key value) ...))' Scheme
  settings/Thread Variables              '(39 (id type name 1)( (key . value) ...))', one block
                                         per zone and phase domain (a zone repeats)

Face polygons are mixed (3 to 8 nodes) on the hull walls.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_THREAD_HEADER = re.compile(r"\(39 \((\d+) ([a-z0-9\-]+) (\S+) \d+\)\(")
_NUM = r"([-+0-9.eE]+)"


class CaseReadError(RuntimeError):
    pass


@dataclass(frozen=True)
class Zone:
    id: int
    name: str
    kind: str      # Fluent type: fluid, solid, wall, pressure-inlet, symmetry, overset, ...
    cells: bool    # cell zone (True) or face zone (False)
    lo: int        # first face/cell id (1-based)
    hi: int        # last face/cell id


def scheme_pairs(text: str) -> dict[str, str]:
    """Top-level '(key value)' entries of a '(NN ((k v) (k v) ...))' settings blob.

    Values are kept as raw Scheme text; callers pick out what they need.
    """
    i = text.index("(", text.index("(") + 1) + 1
    out: dict[str, str] = {}
    n = len(text)
    while i < n:
        while i < n and text[i] != "(":
            if text[i] == ")":
                return out
            i += 1
        j, depth, in_str = i, 0, False
        while j < n:
            c = text[j]
            if in_str:
                if c == "\\":
                    j += 1
                elif c == '"':
                    in_str = False
            elif c == '"':
                in_str = True
            elif c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        key, _, val = text[i + 1:j].partition(" ")
        out[key] = val.strip()
        i = j + 1
    return out


class CaseFile:
    """One .cas.h5, read lazily. Coordinates are loaded once (~0.3 GB for 14M nodes)."""

    def __init__(self, path: str | Path):
        import h5py

        self.path = Path(path)
        self._f = h5py.File(self.path, "r")
        self.rampant = scheme_pairs(self._setting("Rampant Variables"))
        self.version = self._setting("Version")
        self._threads = self._thread_blocks()
        self.zones = self._read_zones()
        self._coords = None
        self._nn = None
        self._off = None

    def close(self) -> None:
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # --- settings ----------------------------------------------------------

    def _setting(self, name: str) -> str:
        return self._f[f"settings/{name}"][0].decode(errors="replace")

    def _thread_blocks(self) -> dict[int, list[str]]:
        text = self._setting("Thread Variables")
        heads = list(_THREAD_HEADER.finditer(text))
        blocks: dict[int, list[str]] = {}
        self._thread_kind: dict[int, tuple[str, str]] = {}
        for k, m in enumerate(heads):
            end = heads[k + 1].start() if k + 1 < len(heads) else len(text)
            zid = int(m.group(1))
            blocks.setdefault(zid, []).append(text[m.end():end])
            self._thread_kind.setdefault(zid, (m.group(2), m.group(3)))
        return blocks

    def _read_zones(self) -> dict[str, Zone]:
        zones = {}
        for group, cells in (("cells", True), ("faces", False)):
            zt = self._f[f"meshes/1/{group}/zoneTopology"]
            names = zt["name"][0].decode().split(";")
            for zid, name, lo, hi in zip(zt["id"][:], names, zt["minId"][:], zt["maxId"][:]):
                kind = self._thread_kind.get(int(zid), ("?", name))[0]
                zones[name] = Zone(int(zid), name, kind, cells, int(lo), int(hi))
        return zones

    def zone(self, name: str) -> Zone:
        if name not in self.zones:
            raise CaseReadError(f"{self.path.name}: no zone {name!r}; zones: {sorted(self.zones)}")
        return self.zones[name]

    def zone_by_id(self, zid: int) -> Zone:
        for z in self.zones.values():
            if z.id == zid:
                return z
        raise CaseReadError(f"{self.path.name}: no zone with id {zid}")

    def of_kind(self, *kinds: str, cells: bool | None = None) -> list[Zone]:
        return [z for z in self.zones.values()
                if z.kind in kinds and (cells is None or z.cells == cells)]

    def bc_constants(self, zone: str, key: str) -> list[float]:
        """Every constant value of a boundary setting, across the zone's phase blocks.

        Non-constant settings (profiles, expressions) are not returned.
        """
        pat = re.compile(r"\n \(" + re.escape(key) + r" \(constant \. " + _NUM + r"\)")
        return [float(v) for block in self._threads.get(self.zone(zone).id, [])
                for v in pat.findall(block)]

    def bc_flag(self, zone: str, key: str) -> list[str]:
        pat = re.compile(r"\n \(" + re.escape(key) + r" \. ([^)\s]+)\)")
        return [v for block in self._threads.get(self.zone(zone).id, []) for v in pat.findall(block)]

    def overset_grids(self) -> tuple[list[int], list[int]]:
        """(background zone ids, component zone ids) of the overset interfaces."""
        text = self.rampant.get("overset/interfaces", "")

        def ids(tag: str) -> list[int]:
            out = []
            for m in re.finditer(tag + r"((?: \(\d+ -?\d+ -?\d+\))+)", text):
                out += [int(v) for v in re.findall(r"\((\d+) ", m.group(1))]
            return out

        return ids("bg-grids"), ids("comp-grids")

    def gravity(self) -> np.ndarray | None:
        if self.rampant.get("gravity?") != "#t":
            return None
        return np.array([float(self.rampant[f"gravity/{c}"]) for c in "xyz"])

    def phases(self) -> list[tuple[str, str]]:
        """(phase name, material) for each phase domain."""
        text = self.rampant.get("domains", "")
        out = []
        for m in re.finditer(r"\(\d+ phase-domain ([^)\s]+)\)", text):
            mat = re.search(r"\(material \. ([^)\s]+)\)", text[m.end():])
            out.append((m.group(1), mat.group(1) if mat else "?"))
        return out

    def material_density(self, material: str) -> float | None:
        text = self.rampant.get("materials", "")
        m = re.search(r"\(" + re.escape(material) + r" fluid ", text)
        if not m:
            return None
        d = re.search(r"\(density \(constant \. " + _NUM + r"\)", text[m.end():])
        return float(d.group(1)) if d else None

    def sixdof_origins(self) -> dict[str, list[float]]:
        """6DOF dynamic-zone origins (the CG the GUI setup used), by zone name."""
        text = self.rampant.get("dynamesh/dynamic-zones", "")
        out = {}
        for zid, o in re.findall(r"\((\d+) \(origin \(([^)]*)\)\)", text):
            try:
                out[self.zone_by_id(int(zid)).name] = [float(v) for v in o.split()]
            except CaseReadError:
                pass
        return out

    def template_blockers(self) -> list[str]:
        """What `prepare_case.py template` would still have to strip from this case."""
        out = []
        if self.rampant.get("dynamesh/dynamic-zones", "()") not in ("()", ""):
            out.append("6DOF dynamic zones are defined")
        if self.rampant.get("udf/libname", "()") not in ("()", ""):
            out.append("a UDF library is loaded")
        if re.search(r"\(conv-reports \(\(", self.rampant.get("monitor/convergencesets", "")):
            out.append("convergence conditions are set")
        return out

    # --- geometry ----------------------------------------------------------

    @property
    def coords(self) -> np.ndarray:
        if self._coords is None:
            g = self._f["meshes/1/nodes/coords"]
            self._coords = g[list(g.keys())[0]][:]
        return self._coords

    def _face_index(self):
        if self._nn is None:
            fn = self._f["meshes/1/faces/nodes/1"]
            self._nn = fn["nnodes"][:].astype(np.int64)
            self._off = np.concatenate([[0], np.cumsum(self._nn)])
        return self._nn, self._off

    def _face_node_ids(self, z: Zone) -> tuple[np.ndarray, np.ndarray]:
        """(nnodes per face, flat 0-based node ids) of a face zone."""
        if z.cells:
            raise CaseReadError(f"{z.name} is a cell zone")
        nn, off = self._face_index()
        ids = self._f["meshes/1/faces/nodes/1/nodes"][off[z.lo - 1]:off[z.hi]].astype(np.int64) - 1
        return nn[z.lo - 1:z.hi], ids

    def vertices(self, zone: str) -> np.ndarray:
        """Unique node coordinates of a face zone (what Fluent's surface vertices return)."""
        _, ids = self._face_node_ids(self.zone(zone))
        return self.coords[np.unique(ids)]

    def faces(self, zone: str) -> tuple[np.ndarray, np.ndarray]:
        """(centroids, area vectors) of a face zone's polygons. Orientation is Fluent's."""
        k, ids = self._face_node_ids(self.zone(zone))
        starts = np.concatenate([[0], np.cumsum(k)[:-1]])
        C = np.zeros((len(k), 3))
        S = np.zeros((len(k), 3))
        for m in np.unique(k):
            sel = np.where(k == m)[0]
            P = self.coords[ids[starts[sel][:, None] + np.arange(m)]]
            C[sel] = P.mean(axis=1)
            S[sel] = 0.5 * sum(np.cross(P[:, i], P[:, (i + 1) % m]) for i in range(m))
        return C, S

    def c0_cell_zone(self, face_zone: str) -> Zone:
        """The cell zone on the c0 side of a face zone (from its first face)."""
        z = self.zone(face_zone)
        cell = int(self._f["meshes/1/faces/c0/1"][z.lo - 1])
        for cz in self.zones.values():
            if cz.cells and cz.lo <= cell <= cz.hi:
                return cz
        raise CaseReadError(f"{face_zone}: c0 cell {cell} is in no cell zone")
