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
#: (red is the single-series data colour of gpsplot ts) and no grey (the
#: error bars and the set-aside days are grey; slot 7 is black since
#: 2026-10-30, so the other slots keep their stations)
STACK_PALETTE: tuple[str, ...] = (
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
    "#e377c2", "#000000", "#bcbd22", "#17becf", "#393b79", "#637939",
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


#: a line is broken where its epochs are further apart than this many
#: median spacings (an RTK outage must not be drawn as motion)
GAP_FACTOR = 5.0


def break_gaps(x: Sequence[Any], y: Any) -> tuple[list[Any], Any]:
    """``(x, y)`` with a NaN point in each gap wider than :data:`GAP_FACTOR`
    median spacings, so a plotted line stops there instead of bridging it."""
    y = np.asarray(y, dtype=float)
    if len(x) < 3:
        return list(x), y
    t = np.array(x, dtype="datetime64[ns]").astype(np.int64).astype(float)
    step = np.diff(t)
    gaps = np.flatnonzero(step > GAP_FACTOR * np.median(step)) + 1
    if gaps.size == 0:
        return list(x), y
    xs = list(x)
    for k in gaps[::-1]:
        xs.insert(int(k), xs[int(k) - 1])
    return xs, np.insert(y, gaps, np.nan, axis=1)


def stack_figure(
    series: Sequence[Series],
    *,
    title: Any,
    labels: Sequence[str] | None = None,
    start: datetime.datetime | None = None,
    end: datetime.datetime | None = None,
    ylim: Sequence[Any] = (),
    offset: float = 0.0,
    show_outliers: int | None = None,
    show_obstruction: int | None = None,
    show_network: int | None = None,
    styles: Sequence[str] | None = None,
    colors: Sequence[str] | None = None,
    fig: Figure | None = None,
    robust_lines: bool = False,
) -> Figure:
    """Overlay ``series`` on the N/E/U frame; series ``k`` is moved by k·offset.

    Args:
        series: What to draw, in legend order; each ``y``/``sigma`` (3, N) mm.
        title: As ``timesmatplt.stdFrame`` takes it (a StationTitle).
        labels: Legend texts (default: each series' label).
        start, end: Axis span (None: the union of the series' epochs).
        ylim: ``--ylim`` semantics of ``timesmatplt.setYlim``.
        offset: Vertical spacing [mm] between consecutive series.
        show_outliers: Draw each series' outliers (cleaned view: its
            ``outliers`` overlay) in the series' colour with a bold black
            outline — 0 all of them, N only among its last N epochs; None
            (default) none. They widen the y range, so none is cut off.
        show_obstruction: The same for the icing and snow-burial days
            (``iced``, ``buried``): light blue, edged in the series' colour.
        show_network: The same for the network days (``netday``): dark
            grey, edged in the series' colour.
        styles: Per series ``"points"`` (markers with error bars, the
            default), ``"marked"`` (larger, dark-edged points, to stand out
            over lines) or ``"line"`` (a thin line, no error bars: dense
            series such as 1-minute RTK medians). Lines are drawn first.
        colors: Per series colours (default: :func:`series_colors`).
        robust_lines: ``"line"`` series set the y range by their
            2–98 percentiles, not their extremes: the spikes of a dense
            RTK line run off the axis instead of squashing the points
            (takes effect where ``ylim`` sets the range: one number).
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

    def overlay(k: int, layer: Any, tail: int | None) -> Any:
        """Series k's set-aside layer, shifted with it; None when empty."""
        if tail is None or layer is None:
            return None
        oy, ody = layer
        if tail:
            oy, ody = tplt.last_flagged(oy, ody, tail)
        oy = np.asarray(oy, dtype=float) + k * float(offset)
        return (oy, ody) if np.isfinite(oy).any() else None

    def merged(a: Any, b: Any) -> Any:  # icing on N/E/U, snow on Up only
        if a is None or b is None:
            return a if b is None else b
        keep = np.isfinite(np.asarray(a[0], dtype=float))
        return (np.where(keep, a[0], b[0]), np.where(keep, a[1], b[1]))

    flagged = [overlay(k, s.outliers, show_outliers) for k, s in enumerate(series)]
    obstructed = [
        merged(
            overlay(k, s.iced, show_obstruction), overlay(k, s.buried, show_obstruction)
        )
        for k, s in enumerate(series)
    ]
    netdays = [overlay(k, s.netday, show_network) for k, s in enumerate(series)]

    def low(k: int, i: int) -> float:
        if robust_lines and styles[k] == "line":
            return float(np.nanpercentile(shifted[k][i], 2.0))
        return float(np.nanmin(shifted[k][i]))

    def high(k: int, i: int) -> float:
        if robust_lines and styles[k] == "line":
            return float(np.nanpercentile(shifted[k][i], 98.0))
        return float(np.nanmax(shifted[k][i]))

    ymin = [min(low(k, i) for k in range(len(series))) for i in range(3)]
    ymax = [max(high(k, i) for k in range(len(series))) for i in range(3)]
    for f in [*flagged, *obstructed, *netdays]:
        if f is not None:
            ymin = [min(v, float(np.nanmin(f[0][i]))) for i, v in enumerate(ymin)]
            ymax = [max(v, float(np.nanmax(f[0][i]))) for i, v in enumerate(ymax)]
    for k, s in enumerate(series):  # a backcast line starts at the zero
        if "backcast" in s.meta:
            ymin = [min(v, k * float(offset)) for v in ymin]
            ymax = [max(v, k * float(offset)) for v in ymax]
    fig = tplt.setYlim(fig, ymin=ymin, ymax=ymax, ylim=ylim)

    # lines first, then markers: dense line series must not hide points;
    # among the markers the densest first, so a station with a day a week
    # is not buried under a daily one (SVIN under SVIE)
    order = sorted(range(len(series)), key=lambda k: (styles[k] != "line", -len(xs[k])))
    for k in order:
        s, x, y, color, style = series[k], xs[k], shifted[k], colors[k], styles[k]
        label = offset_label(labels[k], k, offset)
        if style == "line":
            xl, yl = break_gaps(x, y)
            for i, ax in enumerate(axes[:3]):
                ax.plot(
                    xl, yl[i], color=color, linewidth=1.0,
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
    # outliers: the series' colour, a bold black outline, drawn over the data
    # (most cancel in a baseline; the few left must not go unseen)
    for k, f in enumerate(flagged):
        if f is None:
            continue
        tplt.addData(
            xs[k],
            f[0],
            f[1],
            fig,
            ecolor="0.5",
            markersize=5.0,
            markerfacecolor=colors[k],
            markeredgecolor="black",
            markeredgewidth=1.6,
        )
    if any(f is not None for f in flagged):  # one legend key for all of them
        axes[0].plot(
            [], [], linestyle="none", marker="o", markersize=5.0,
            markerfacecolor="white", markeredgecolor="black",
            markeredgewidth=1.6, label="outlier (its series' colour)",
        )  # fmt: skip
    # icing/snow and network days: the class's colour, edged in the series'
    # colour (which station it was), as gpsplot ts draws them
    for layers, face, bar, key in (
        (obstructed, tplt.ICING_FACE_COLOR, tplt.ICING_ERRORBAR_COLOR,
         "icing / snow suspected (set aside)"),
        (netdays, tplt.NETWORK_FACE_COLOR, tplt.NETWORK_ERRORBAR_COLOR,
         "network day (set aside)"),
    ):  # fmt: skip
        for k, f in enumerate(layers):
            if f is not None:
                tplt.addData(
                    xs[k], f[0], f[1], fig, ecolor=bar, markersize=5.0,
                    markerfacecolor=face, markeredgecolor=colors[k],
                    markeredgewidth=1.6,
                )  # fmt: skip
        if any(f is not None for f in layers):
            axes[0].plot(
                [], [], linestyle="none", marker="o", markersize=5.0,
                markerfacecolor=face, markeredgecolor="black",
                markeredgewidth=1.6, label=key,
            )  # fmt: skip
    # a late series' backcast (stack.backcast_late_starts): its fitted rate,
    # dashed, from the stack's first epoch (zero) to where its data begin
    for k, s in enumerate(series):
        if "backcast" not in s.meta:
            continue
        t0, t1, at_t1 = s.meta["backcast"]
        xb = tplt._as_datetime_list(np.array([t0, t1]))
        for i, ax in enumerate(axes[:3]):
            ax.plot(
                xb, [k * float(offset), at_t1[i] + k * float(offset)],
                color=colors[k], linestyle="--", linewidth=1.0,
            )  # fmt: skip
    axes[0].legend(
        loc="upper left",
        fontsize="medium",
        markerscale=2.0,
        ncol=max(1, (len(series) + 5) // 6),
        framealpha=0.85,
    )
    return fig
