"""The stored trajectory sampled for drawing: dense grid, vertical steps.

Shared by the detrend workbench (its fit panels) and ``gpsplot ts --model``
(the model line over a series). Evaluation is ``gps_analysis.evaluate_record``;
this module only decides WHERE the model is sampled, which is a drawing
choice (see :func:`trajectory_curve`).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

#: Days per year used to turn ``step_days`` into a fractional-year step.
#: Julian, matching the fractional-year convention the records are in.
DAYS_PER_YEAR: float = 365.25

#: Half-width of the sample pair placed either side of a step epoch, so the
#: discontinuity is drawn vertical rather than ramped.  ~32 s: far below any
#: plotted resolution, far above float64 spacing at year 2026 (~4e-13).
STEP_BRACKET_YEARS: float = 1e-6


def curve_grid(record: Mapping[str, Any], yearf: Any, *, step_days: float = 1.0) -> Any:
    """The drawing grid of :func:`trajectory_curve`: dense, steps bracketed.

    Spans exactly ``[nanmin(yearf), nanmax(yearf)]`` at ``step_days``
    spacing, plus a sample pair at ``step ± STEP_BRACKET_YEARS`` for every
    step epoch inside it (see :func:`trajectory_curve` for why). Finite,
    sorted, strictly increasing; empty when ``yearf`` has no finite epoch.
    """
    y = np.asarray(yearf, dtype=float)
    finite = y[np.isfinite(y)]
    if finite.size == 0:
        return np.empty(0, dtype=float)

    t0, t1 = float(finite.min()), float(finite.max())
    step = step_days / DAYS_PER_YEAR
    n = max(int(np.ceil((t1 - t0) / step)) + 1, 2) if t1 > t0 else 2
    grid = np.linspace(t0, t1, n)

    brackets = [
        v
        for s in (record.get("step_epochs") or ())
        for v in (float(s) - STEP_BRACKET_YEARS, float(s) + STEP_BRACKET_YEARS)
        if t0 < v < t1
    ]
    if brackets:
        # unique() sorts and de-duplicates in one call, which also keeps the
        # grid strictly increasing if a bracket lands on a grid point.
        grid = np.unique(np.concatenate([grid, np.asarray(brackets, dtype=float)]))

    return grid


def trajectory_curve(
    record: Mapping[str, Any],
    yearf: Any,
    *,
    terms: str = "all",
    step_days: float = 1.0,
) -> tuple[Any, Any]:
    """The fitted trajectory sampled for DRAWING, not at the data epochs.

    Evaluating the record at ``yearf`` and joining the dots draws a straight
    chord across every data gap, which is a claim the model does not make:
    the model says the secular trend continued and the seasonal kept
    oscillating through the gap, and the chord says the station moved
    linearly. On a station with a months-long outage the chord is the most
    visually prominent feature of the figure and it is the one part of the
    blue line that is not the model. ``evaluate_record`` is explicitly valid
    at arbitrary epochs, so the honest curve is the model on a dense grid.

    Each step epoch is BRACKETED so the jump stays vertical. A
    ``step_epochs`` entry is a discontinuity, and on a bare uniform grid the
    two samples flanking it are joined by one sloped segment — normally a
    day wide and invisible, but if the step falls inside a data gap that
    segment ramps across the whole outage and reads as slow motion that
    never happened. Two extra samples at ``step ± STEP_BRACKET_YEARS`` pin
    the pre- and post-step levels a few seconds apart, so the jump renders
    vertical at any zoom and in any gap.

    Bracketing rather than breaking is deliberate: a NaN would render the
    discontinuity as a discontinuity, which is arguably more honest but
    changes how every stepped figure in this package has always looked. The
    ask here was about gaps, and sampling density is cosmetic — how steps
    are drawn is not, so it stays as it was.

    The grid spans exactly ``[nanmin(yearf), nanmax(yearf)]``. Changing the
    sampling density is cosmetic; extending the domain would be a new claim.

    Args:
        record: A stored detrend record.
        yearf: The observed epochs [decimal yr]; only their extent is used.
        terms: Apply-time term selection, passed to ``evaluate_record``.
        step_days: Grid spacing [days]. The default matches the daily data
            cadence, so the curve is as smooth as the series can justify.

    Returns:
        ``(grid, fit)`` — the grid is finite, sorted and strictly
        increasing; ``fit`` is ``(3, len(grid))``. Both are empty when
        ``yearf`` holds no finite epoch.
    """
    from gps_analysis import evaluate_record

    grid = curve_grid(record, yearf, step_days=step_days)
    if grid.size == 0:
        # No span means no curve. Returning empties rather than raising keeps
        # this the same non-event that an all-NaN series already is upstream.
        return grid, np.empty((3, 0), dtype=float)
    fit = np.asarray(evaluate_record(record, grid, terms=terms), dtype=float)
    return grid, fit
