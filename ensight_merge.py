"""Merge the per-session EnSight Gold indexes of one case into a single continuous series.

Each Fluent session writes a free.encas that lists only its own frames, and
run_case.set_aside_report_files renames earlier ones to free.part<k>.encas. The per-step
.geo/.scl*/.vel files are numbered by time step and share one directory, so the whole run
is recoverable: union the (file number, flow time) pairs of every index, keep the frames
whose .geo exists, and write free.merged.encas.
"""
from __future__ import annotations

import re
from pathlib import Path

MERGED = "free.merged.encas"


def _floats(text: str) -> list[float]:
    return [float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)]


def parse_frames(encas: Path) -> dict[int, float]:
    """{file number: flow time} from one index (handles start/increment and explicit numbers)."""
    text = encas.read_text()
    out: dict[int, float] = {}
    for block in re.split(r"(?m)^time set:", text)[1:]:
        block = block.split("\nSCRIPTS")[0]
        times = _floats(block.split("time values:", 1)[1]) if "time values:" in block else []
        m = re.search(r"filename numbers:(.*?)time values:", block, re.S)
        if m:
            nums = [int(x) for x in m.group(1).split()]
        else:
            start = re.search(r"filename start number:\s*(\d+)", block)
            inc = re.search(r"filename increment:\s*(\d+)", block)
            if not (start and inc):
                continue
            nums = [int(start.group(1)) + i * int(inc.group(1)) for i in range(len(times))]
        out.update(zip(nums, times))
    return out


def merge_case(ens: Path, stem: str = "free") -> Path | None:
    """Write <ens>/free.merged.encas; returns it, or None when there are no indexes."""
    parts = sorted(ens.glob(f"{stem}.part*.encas"), key=lambda p: int(re.search(r"part(\d+)", p.name).group(1)))
    final = ens / f"{stem}.encas"
    indexes = [*parts, *([final] if final.exists() else [])]
    if not indexes:
        return None
    frames: dict[int, float] = {}
    for p in indexes:  # later sessions win on overlap
        frames.update(parse_frames(p))
    frames = {n: t for n, t in frames.items() if (ens / f"{stem}{n:05d}.geo").exists()}
    if not frames:
        return None
    nums = sorted(frames)
    template = indexes[-1].read_text().split("\nTIME")[0]
    lines = ["TIME"]
    for ts in (1, 3):  # geometry and variable time sets, as Fluent writes them
        lines += [f"time set: {ts} Model", f"number of steps: {len(nums)}",
                  "filename numbers: " + " ".join(f"{n:05d}" for n in nums),
                  "time values: " + " ".join(f"{frames[n]:.5e}" for n in nums)]
    out = ens / MERGED
    out.write_text(template + "\n" + "\n".join(lines) + f'\nSCRIPTS\nmetadata: "{stem}.xml"\n')
    return out
