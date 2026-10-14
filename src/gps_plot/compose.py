"""Figure composition for multi-series plots (``gpsplot stack``).

Draws any number of :class:`~gps_plot.series.Series` into the standard
N/E/U frame of ``gpsplot ts`` (``timesmatplt.stdFrame``, same style, axes
and tick labels), one colour per series, with a legend. Nothing here reads
data: sources build the Series, this only places them.
"""

from __future__ import annotations

import datetime
import zlib
from collections.abc import Sequence
from typing import Any

import numpy as np
from matplotlib.figure import Figure

from gps_plot.series import Series

#: colours for overlaid series: distinct, readable on white, no pure red
#: (red is the single-series data colour of gpsplot ts)
STACK_PALETTE: tuple[str, ...] = (
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
    "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#393b79", "#637939",
)  # fmt: skip


def series_colors(labels: Sequence[str]) -> list[str]:
    """One palette colour per label, keyed on the label itself.

    A label hashes to a fixed palette slot, so a station keeps its colour
    from one plot to the next; when two labels in the SAME figure land on
    one slot, the later one takes the next free slot (and only then does
    its colour depend on the company it keeps). More series than colours
    reuse colours in order.
    """
    n = len(STACK_PALETTE)
    taken: set[int] = set()
    out = []
    for label in labels:
        slot = zlib.crc32(label.encode()) % n
        if len(taken) < n:
            while slot in taken:
                slot = (slot + 1) % n
            taken.add(slot)
        out.append(STACK_PALETTE[slot])
    return out


def offset_label(label: str, k: int, offset: float) -> str:
    """The legend text of series ``k``: its label, plus its offset if any."""
    if not offset or k == 0:
        return label
    return f"{label} ({k * offset:+g} mm)"


def stack_figure(
    series: Sequence[Series],
    *,
    title: Any,
    labels: Sequence[str] | None = None,
    start: datetime.datetime | None = None,
    end: datetime.datetime | None = None,
    ylim: Sequence[Any] = (),
    offset: float = 0.0,
    styles: Sequence[str] | None = None,
    colors: Sequence[str] | None = None,
    fig: Figure | None = None,
) -> Figure:
    """Overlay ``series`` on the N/E/U frame; series ``k`` is moved by k·offset.

    Args:
        series: What to draw, in legend order; each ``y``/``sigma`` (3, N) mm.
        title: As ``timesmatplt.stdFrame`` takes it (a StationTitle).
        labels: Legend texts (default: each series' label).
        start, end: Axis span (None: the union of the series' epochs).
        ylim: ``--ylim`` semantics of ``timesmatplt.setYlim``.
        offset: Vertical spacing [mm] between consecutive series.
        styles: Per series ``"points"`` (markers with error bars, the
            default), ``"marked"`` (larger, dark-edged points, to stand out
            over lines) or ``"line"`` (a thin line, no error bars: dense
            series such as 1-minute RTK medians). Lines are drawn first.
        colors: Per series colours (default: :func:`series_colors`).
    """
    from gps_plot import timesmatplt as tplt

    if not series:
        raise ValueError("nothing to stack")
    labels = list(labels or [s.label for s in series])
    colors = list(colors or series_colors([s.label for s in series]))
    styles = list(styles or ["points"] * len(series))
    xs = [tplt._as_datetime_list(s.t) for s in series]

    fig, axes = tplt.stdFrame(None, title, fig=fig)
    first = min(x[0] for x in xs)
    last = max(x[-1] for x in xs)
    period = tplt.setXlim(axes, first, last, start=start, end=end)
    fig = tplt.tsTickLabels(fig, axes, period=period)  # type: ignore[no-untyped-call]

    shifted = [
        np.asarray(s.y, dtype=float) + k * float(offset) for k, s in enumerate(series)
    ]
    ymin = [min(float(np.nanmin(y[i])) for y in shifted) for i in range(3)]
    ymax = [max(float(np.nanmax(y[i])) for y in shifted) for i in range(3)]
    fig = tplt.setYlim(fig, ymin=ymin, ymax=ymax, ylim=ylim)

    # lines first, then markers: dense line series must not hide points
    order = sorted(range(len(series)), key=lambda k: styles[k] != "line")
    for k in order:
        s, x, y, color, style = series[k], xs[k], shifted[k], colors[k], styles[k]
        label = offset_label(labels[k], k, offset)
        if style == "line":
            for i, ax in enumerate(axes[:3]):
                ax.plot(
                    x, y[i], color=color, linewidth=1.0,
                    label=label if i == 0 else None,
                )  # fmt: skip
            continue
        tplt.addData(
            x,
            y,
            s.sigma,
            fig,
            ecolor="0.3" if style == "marked" else "0.75",
            markersize=5.0 if style == "marked" else 3.0,
            markerfacecolor=color,
            markeredgecolor="black" if style == "marked" else color,
            label=label,
        )
    axes[0].legend(
        loc="upper left",
        fontsize="medium",
        markerscale=2.0,
        ncol=max(1, (len(series) + 5) // 6),
        framealpha=0.85,
    )
    return fig
