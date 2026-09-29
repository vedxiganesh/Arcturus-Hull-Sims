"""Where every hullsweep file lives. The ONLY module that knows a path.

One convention, two sites:

  CLUSTER  ~/hullsweep_code/                       code (pushed by `hs sync push-code`)
           ~/orcd/pool/hullsweep/<topo>/           topology.json, <topo>_template.cas.h5, meshes
           ~/orcd/pool/hullsweep/<topo>/<study>/   study.json, ledger.json, results.csv, logs/
           ~/orcd/scratch/hullsweep/<topo>/<study>/ progress.log, one dir per CFD case

  LOCAL    Arcturus/hullsweep/                     code
           Arcturus/hullsweep_data/<topo>/         same shape; pool and scratch share one root,
                                                   so a pulled study is one directory

Files are found by pattern, never by a stored path: a lookup that matches zero
or several files raises and lists what it saw. Matching is case-insensitive
on both sites so Windows and Linux agree.

After `hs new`, a study is addressed by its NAME alone: find_study() locates
the one topology folder that holds it. Study names are unique across
topologies (enforced by new_study()).

Env overrides (tests, unusual setups): HULLSWEEP_SITE=cluster|local,
HULLSWEEP_POOL, HULLSWEEP_SCRATCH.
"""

from __future__ import annotations

import fnmatch
import os
import re
from dataclasses import dataclass
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent

REMOTE_HOST = "vxg@orcd-login.mit.edu"
#: Relative to the cluster home directory.
REMOTE_CODE = "hullsweep_code"
REMOTE_POOL = "orcd/pool/hullsweep"
REMOTE_SCRATCH = "orcd/scratch/hullsweep"

LOCAL_DATA = CODE_DIR.parent / "hullsweep_data"

TOPOLOGY_FILE = "topology.json"
STUDY_FILE = "study.json"
LEDGER_FILE = "ledger.json"
RESULTS_FILE = "results.csv"
PROGRESS_FILE = "progress.log"
REQUEST_FILE = "request.json"
STATUS_FILE = "status.json"

_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_]*$")

#: File-role patterns inside a topology folder, relative to the topology name.
ROLE_PATTERNS = {
    "template": "{name}*template*.cas.h5",
    "source": "{name}*source*.cas.h5",
    "foreground_mesh": "*foreground*.msh*",
    "background_mesh": "*background*.msh*",
}


class LayoutError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Sites and roots
# ---------------------------------------------------------------------------


def site() -> str:
    s = os.environ.get("HULLSWEEP_SITE", "").strip().lower()
    if s:
        if s not in ("cluster", "local"):
            raise LayoutError(f"HULLSWEEP_SITE must be 'cluster' or 'local', not {s!r}")
        return s
    return "cluster" if (Path.home() / "orcd" / "pool").is_dir() else "local"


@dataclass(frozen=True)
class Roots:
    pool: Path
    scratch: Path


def roots() -> Roots:
    if site() == "cluster":
        pool, scratch = Path.home() / REMOTE_POOL, Path.home() / REMOTE_SCRATCH
    else:
        pool = scratch = LOCAL_DATA
    pool = Path(os.environ.get("HULLSWEEP_POOL", pool))
    scratch = Path(os.environ.get("HULLSWEEP_SCRATCH", scratch))
    return Roots(pool=pool, scratch=scratch)


def check_name(name: str, what: str) -> str:
    if not _NAME_RE.match(name or ""):
        raise LayoutError(f"{what} name {name!r} must match {_NAME_RE.pattern} (lowercase, digits, _)")
    return name


# ---------------------------------------------------------------------------
# Pattern lookup
# ---------------------------------------------------------------------------


def find_all(directory: Path, pattern: str) -> list[Path]:
    """Files in `directory` (not recursive) whose name matches, case-insensitively."""
    if not directory.is_dir():
        return []
    pat = pattern.lower()
    return sorted(p for p in directory.iterdir() if p.is_file() and fnmatch.fnmatch(p.name.lower(), pat))


def find_one(directory: Path, pattern: str, what: str) -> Path:
    hits = find_all(directory, pattern)
    if len(hits) != 1:
        seen = ", ".join(p.name for p in hits) or "nothing"
        raise LayoutError(f"{what}: expected exactly one '{pattern}' in {directory}, found {seen}")
    return hits[0]


def find_optional(directory: Path, pattern: str, what: str) -> Path | None:
    hits = find_all(directory, pattern)
    if len(hits) > 1:
        raise LayoutError(f"{what}: '{pattern}' is ambiguous in {directory}: "
                          f"{', '.join(p.name for p in hits)}")
    return hits[0] if hits else None


# ---------------------------------------------------------------------------
# Topologies
# ---------------------------------------------------------------------------


def topology_dir(name: str, must_exist: bool = True) -> Path:
    d = roots().pool / check_name(name, "topology")
    if must_exist and not (d / TOPOLOGY_FILE).is_file():
        known = ", ".join(list_topologies()) or "none"
        raise LayoutError(f"no topology '{name}' ({d / TOPOLOGY_FILE} missing); known: {known}")
    return d


def list_topologies() -> list[str]:
    pool = roots().pool
    if not pool.is_dir():
        return []
    return sorted(p.name for p in pool.iterdir() if (p / TOPOLOGY_FILE).is_file())


def topology_file(name: str, role: str, required: bool = True) -> Path | None:
    """A topology input by role: template, source, foreground_mesh, background_mesh."""
    d = topology_dir(name)
    pat = ROLE_PATTERNS[role].format(name=name)
    what = f"topology '{name}' {role}"
    return find_one(d, pat, what) if required else find_optional(d, pat, what)


def canonical_name(name: str, role: str, suffix: str) -> str:
    """The file name `hs topology adopt` and `prepare_case.py template` write."""
    tag = {"template": "template", "source": "source",
           "foreground_mesh": "foreground", "background_mesh": "background"}[role]
    return f"{name}_{tag}{suffix}"


def load_topology(name: str):
    """Topology object for `name`; its 'name' field must equal the folder name."""
    from common import Topology

    topo = Topology.load(topology_dir(name) / TOPOLOGY_FILE)
    if topo.name != name:
        raise LayoutError(f"{topo.path}: name field '{topo.name}' != folder '{name}'")
    return topo


# ---------------------------------------------------------------------------
# Studies
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StudyPaths:
    name: str
    topology: str
    pool_dir: Path
    scratch_dir: Path

    @property
    def study_json(self) -> Path:
        return self.pool_dir / STUDY_FILE

    @property
    def ledger_json(self) -> Path:
        return self.pool_dir / LEDGER_FILE

    @property
    def results_csv(self) -> Path:
        return self.pool_dir / RESULTS_FILE

    @property
    def lock(self) -> Path:
        return self.pool_dir / "ledger.lock"

    @property
    def logs_dir(self) -> Path:
        return self.pool_dir / "logs"

    def case_log_dir(self, case_id: str) -> Path:
        return self.logs_dir / case_id

    @property
    def advance_log_dir(self) -> Path:
        return self.logs_dir / "_advance"

    @property
    def progress_log(self) -> Path:
        return self.scratch_dir / PROGRESS_FILE

    def case_dir(self, case_id: str) -> Path:
        return self.scratch_dir / case_id


def _study_paths(topology: str, name: str) -> StudyPaths:
    r = roots()
    return StudyPaths(name=name, topology=topology,
                      pool_dir=r.pool / topology / name, scratch_dir=r.scratch / topology / name)


def list_studies() -> list[tuple[str, str]]:
    """(topology, study) for every study.json under the pool root."""
    pool = roots().pool
    if not pool.is_dir():
        return []
    return sorted((p.parent.parent.name, p.parent.name) for p in pool.glob(f"*/*/{STUDY_FILE}"))


def find_study(name: str) -> StudyPaths:
    check_name(name, "study")
    hits = [(t, s) for t, s in list_studies() if s == name]
    if not hits:
        known = ", ".join(s for _, s in list_studies()) or "none"
        raise LayoutError(f"no study '{name}' under {roots().pool}; known: {known}")
    if len(hits) > 1:
        raise LayoutError(f"study '{name}' exists under several topologies: "
                          f"{', '.join(t for t, _ in hits)}; rename one")
    return _study_paths(*hits[0])


def new_study(topology: str, name: str) -> StudyPaths:
    """Paths for a NEW study. Refuses a name already used under any topology."""
    check_name(name, "study")
    topology_dir(topology)
    if name in list_topologies():
        raise LayoutError(f"study name '{name}' collides with a topology name")
    taken = [t for t, s in list_studies() if s == name]
    if taken:
        raise LayoutError(f"study '{name}' already exists (topology {taken[0]}); pick a new name")
    sp = _study_paths(topology, name)
    if sp.pool_dir.exists() or sp.scratch_dir.exists():
        raise LayoutError(f"{sp.pool_dir} or {sp.scratch_dir} already exists without a study.json; "
                          "remove it or pick a new name")
    return sp


# ---------------------------------------------------------------------------
# Remote (for sync.py)
# ---------------------------------------------------------------------------


def remote_pool(topology: str | None = None) -> str:
    """Shell path on the cluster, '~'-relative, for ssh command strings."""
    return f"~/{REMOTE_POOL}" + (f"/{topology}" if topology else "")


def remote_scratch(topology: str | None = None) -> str:
    return f"~/{REMOTE_SCRATCH}" + (f"/{topology}" if topology else "")


def remote_code() -> str:
    return f"~/{REMOTE_CODE}"
