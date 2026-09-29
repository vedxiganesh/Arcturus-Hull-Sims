"""Vendored from the Reefs codebase -- DO NOT EDIT IN PLACE.

Source : C:/Users/vgnau/Documents/MIT/Reefs/Reefs_Code/cfd/postproc.py
Commit : 5e44ddf79f7f3407c574716c676f932d4cf1a9ff
Copied : 2026-09-27

Only the pure report-file pieces are copied (ReportSeries, parse_report_file,
dedupe_restarts, _ols_slope). The Reefs package itself is not importable here:
it has no pyproject, uses absolute ``Reefs_Code.cfd`` imports, and its
config pulls in torch, which the cluster's pyfluent env does not have.
Refresh by re-copying the same line ranges if the source changes.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

#: Name Fluent gives the flow-time column in a transient report file.
FLOW_TIME_COLUMN = "flow-time"

#: Index column names Fluent uses. "Time Step" for transient runs, "Iteration"
#: for steady ones.
TIME_STEP_INDEX = "Time Step"
ITERATION_INDEX = "Iteration"


class ReportFileError(ValueError):
    """Raised when a Fluent report file cannot be parsed into a usable series."""


@dataclass(frozen=True)
class ReportSeries:
    """A parsed Fluent report (``-rfile.out``) file.

    Columns are addressed BY NAME, never by position. Fluent's column order is
    not stable across report definitions: a steady file has no ``flow-time``
    column at all, in a transient force file ``flow-time`` sits at index 2
    *before* the per-zone columns, and at least one real file on disk has the
    degenerate header ``("Time Step" "flow-time")`` with no value column
    whatsoever (``Data/oscillatory_benchmark/oscillatory-data-FROZEN/
    report-def-drag-power-rfile.out``). Indexing positionally silently reads
    the wrong physical quantity in each of those cases.

    Attributes:
        path: Source file.
        title: Report title from header line 1, e.g. ``"total_reef_force-rfile"``.
        index_name: ``"Time Step"`` (transient) or ``"Iteration"`` (steady).
        columns: Column names in file order, exactly as they appear in the
            Scheme tuple on header line 3, including ``index_name``.
        data: ``(n_rows, n_columns)`` float array in the same column order.
        transient: True iff a ``flow-time`` column is present.
        n_malformed_rows: Rows dropped because their field count disagreed
            with the header, or they held non-numeric text. Non-zero usually
            means the file was truncated mid-write by a crash.
    """

    path: Path
    title: str
    index_name: str
    columns: tuple[str, ...]
    data: NDArray[np.float64]
    transient: bool
    n_malformed_rows: int = 0

    def has_column(self, name: str) -> bool:
        """Whether ``name`` is a column of this report.

        Args:
            name: Column name to test.

        Returns:
            True if present.
        """
        return name in self.columns

    def column(self, name: str) -> NDArray[np.float64]:
        """Return one column by name.

        Args:
            name: Column name as it appears in :attr:`columns`.

        Returns:
            1-D array of that column's values.

        Raises:
            KeyError: If ``name`` is not a column of this report. The message
                lists the available columns, since the usual cause is assuming
                a column name that this particular report definition did not
                emit.
        """
        try:
            idx = self.columns.index(name)
        except ValueError:
            raise KeyError(
                f"{self.path.name!r} has no column {name!r}; "
                f"available columns: {list(self.columns)}"
            ) from None
        return self.data[:, idx]

    @property
    def index(self) -> NDArray[np.float64]:
        """The index column (time step or iteration number).

        Returns:
            1-D array of index values.
        """
        return self.column(self.index_name)

    @property
    def flow_time(self) -> NDArray[np.float64]:
        """Flow time in seconds.

        Returns:
            The ``flow-time`` column.

        Raises:
            ReportFileError: If this is a steady report, which has no flow
                time. Callers doing transient analysis should let this
                propagate rather than substituting the iteration number --
                iterations are not seconds.
        """
        if not self.transient:
            raise ReportFileError(
                f"{self.path.name!r} is a steady report (indexed by "
                f"{self.index_name!r}) and has no {FLOW_TIME_COLUMN!r} column."
            )
        return self.column(FLOW_TIME_COLUMN)

    @property
    def value_columns(self) -> tuple[str, ...]:
        """Column names excluding the index and ``flow-time``.

        Returns:
            Tuple of physical-quantity column names. EMPTY for a degenerate
            header such as ``("Time Step" "flow-time")``, which is a real case
            on disk -- a report definition that produced only a time axis.
        """
        return tuple(
            c for c in self.columns if c not in (self.index_name, FLOW_TIME_COLUMN)
        )

    @property
    def n_rows(self) -> int:
        """Number of data rows.

        Returns:
            Row count.
        """
        return int(self.data.shape[0])


def _parse_header_tuple(line: str) -> tuple[str, ...]:
    """Extract column names from the Scheme tuple on header line 3.

    Args:
        line: Header line 3, e.g. ``("Time Step" "total_reef_force" "flow-time")``.

    Returns:
        Column names in file order.

    Raises:
        ReportFileError: If the line is not a parenthesized tuple of quoted
            names.
    """
    stripped = line.strip()
    if not (stripped.startswith("(") and stripped.endswith(")")):
        raise ReportFileError(
            f"Expected a parenthesized column tuple on header line 3, got: {line!r}"
        )
    names = re.findall(r'"([^"]*)"', stripped)
    if not names:
        raise ReportFileError(f"No quoted column names found on header line 3: {line!r}")
    return tuple(names)


def parse_report_file(path: Path) -> ReportSeries:
    """Parse a Fluent ``-rfile.out`` report file.

    Format (verified against the real files in ``Data/``)::

        line 1: "<title>"
        line 2: "<index>" "<first-col> etc.."      <- redundant, NOT parsed
        line 3: ("Time Step" "total_reef_force" "flow-time" ...)
        line 4+: whitespace-separated floats, one row per step/iteration

    Line 3 is authoritative for both names and order. Line 2 is deliberately
    ignored: it abbreviates multi-column reports as ``"<first> etc.."`` and so
    cannot be parsed reliably.

    Tolerated without special-casing by the caller:

    - steady files indexed by ``"Iteration"`` with no ``flow-time`` column
    - degenerate headers carrying no value column at all
    - rows truncated mid-write by a crash (dropped, counted in
      ``n_malformed_rows``, not raised) -- a run killed at hour 7 must still
      yield its completed rows
    - non-uniform time steps from adaptive stepping
    - indices that start at nonzero or repeat, from restart appends (see
      :func:`dedupe_restarts`)

    Args:
        path: Path to the ``.out`` file.

    Returns:
        Parsed :class:`ReportSeries`.

    Raises:
        ReportFileError: If the file has fewer than 4 lines, line 3 is not a
            column tuple, or no numeric rows survive parsing.
    """
    path = Path(path)
    try:
        text = path.read_text()
    except OSError as exc:
        raise ReportFileError(f"Cannot read report file {path}: {exc}") from exc

    lines = text.splitlines()
    if len(lines) < 4:
        raise ReportFileError(
            f"{path.name!r} has {len(lines)} lines; a report file needs 3 header "
            f"lines plus at least one data row."
        )

    title = lines[0].strip().strip('"')
    columns = _parse_header_tuple(lines[2])

    rows: list[list[float]] = []
    n_malformed = 0
    for raw in lines[3:]:
        fields = raw.split()
        if not fields:
            continue
        if len(fields) != len(columns):
            n_malformed += 1
            continue
        try:
            rows.append([float(x) for x in fields])
        except ValueError:
            n_malformed += 1

    if not rows:
        raise ReportFileError(
            f"{path.name!r} yielded no numeric data rows "
            f"({n_malformed} malformed) for columns {list(columns)}."
        )

    data = np.asarray(rows, dtype=np.float64)
    index_name = columns[0]
    transient = FLOW_TIME_COLUMN in columns

    return ReportSeries(
        path=path,
        title=title,
        index_name=index_name,
        columns=columns,
        data=data,
        transient=transient,
        n_malformed_rows=n_malformed,
    )


def dedupe_restarts(series: ReportSeries) -> ReportSeries:
    """Collapse restart-appended duplicate indices, keeping the LAST occurrence.

    Fluent APPENDS to an existing report file when a solve is continued rather
    than truncating it, which is what makes the chunked
    ``run_until_periodic`` loop cheap. If a continuation re-runs steps already
    written -- which happens when a solve resumes from an autosave older than
    the last written report row -- the same index value appears twice. The
    later row came from the run that actually continued, so it wins.

    Rows are then sorted by index ascending. Every downstream consumer assumes
    a monotonic series, so this is not cosmetic.

    Args:
        series: Parsed series, possibly containing duplicate indices.

    Returns:
        New :class:`ReportSeries` with unique, ascending indices. Returns
        ``series`` unchanged if it is already unique and sorted.
    """
    idx = series.index
    # Stable sort by index; for equal indices the later row sorts last, so
    # taking the last of each duplicate group keeps the continuation's value.
    order = np.argsort(idx, kind="stable")
    sorted_idx = idx[order]
    # Keep the final occurrence of each index value.
    keep_last = np.ones(sorted_idx.shape[0], dtype=bool)
    keep_last[:-1] = sorted_idx[:-1] != sorted_idx[1:]
    selected = order[keep_last]

    if selected.shape[0] == idx.shape[0] and np.array_equal(selected, np.arange(idx.shape[0])):
        return series

    return ReportSeries(
        path=series.path,
        title=series.title,
        index_name=series.index_name,
        columns=series.columns,
        data=series.data[selected],
        transient=series.transient,
        n_malformed_rows=series.n_malformed_rows,
    )



def _ols_slope(y: NDArray[np.float64]) -> float:
    """Least-squares slope of ``y`` against its own index, in closed form.

    Deliberately avoids ``numpy.polyfit``/``lstsq``: those route through
    LAPACK, which HARD-CRASHES this project's environment (Windows,
    numpy 2.2.6) with 0xC0000409 rather than raising -- a stack buffer
    overrun that no ``try/except`` can catch. The closed form below needs
    only sums, is exact for the evenly-spaced integer abscissa used here,
    and is faster besides.

    slope = sum((k - k_bar) * (y - y_bar)) / sum((k - k_bar)^2)

    Args:
        y: Series to fit; abscissa is ``0, 1, ..., len(y)-1``.

    Returns:
        Slope per unit index. 0.0 for fewer than 2 points, where slope is
        undefined.
    """
    n = y.shape[0]
    if n < 2:
        return 0.0
    k = np.arange(n, dtype=np.float64)
    k_centered = k - k.mean()
    denom = float(np.sum(k_centered * k_centered))
    if denom <= 0.0:
        return 0.0
    return float(np.sum(k_centered * (y - y.mean())) / denom)

