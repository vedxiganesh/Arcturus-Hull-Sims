"""The study's single monitoring file: scratch/<topo>/<study>/progress.log.

Append-only, one line per event, written by every case job and every
orchestrator step. `tail -f` it. Each write is one line under an exclusive
lock (fcntl, where the filesystem supports it), so lines from concurrent
jobs never interleave.

  <ISO time> <kind> <subject> <EVENT> <detail>

kind is 'case', 'newton' or 'study'. ledger.json in pool stays the durable
record; this file is for humans.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

try:
    import fcntl
except ImportError:  # Windows: local tests only
    fcntl = None


def short_case(case_id: str, topology: str) -> str:
    """Case id without the topology prefix (the log already belongs to one topology)."""
    p = f"{topology}_"
    return case_id[len(p):] if case_id.startswith(p) else case_id


def chain_tag(speed: float) -> str:
    return f"V{speed:.2f}"


def line(kind: str, subject: str, event: str, detail: str = "", when: float | None = None) -> str:
    ts = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(when))
    return f"{ts} {kind:<6} {subject:<32} {event:<10} {detail}".rstrip()


def emit(path: Path, kind: str, subject: str, event: str, detail: str = "", echo: bool = True) -> str:
    text = line(kind, subject, event, detail)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (text.replace("\n", " ") + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        if fcntl is not None:
            try:
                fcntl.lockf(fd, fcntl.LOCK_EX)
            except OSError:
                pass  # no lock support on this filesystem; single-line appends
        os.write(fd, data)
    finally:
        os.close(fd)
    if echo:
        print(text, flush=True)
    return text


def fmt_hours(seconds: float | None) -> str:
    return "?" if seconds is None else f"{seconds / 3600.0:.2f} h"
