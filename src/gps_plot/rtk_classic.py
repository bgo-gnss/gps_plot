"""``gpsplot rtk --classic``: the rtk_gps look, for the real-time RTK plots.

The figure rtk_gps (``rtk_gps.plot_rtk_neu``) has drawn for years: title
``SENG-ELDC 6h``, three stacked panels in cm with Icelandic labels
(Norður/Austur/Upp), one line per baseline labelled by its rover (the whole
pair when the bases differ), the Veðurstofa logo in the top corner and a
vertical line at the time the plot was made, with that time written down the
right edge.

Two changes from rtk_gps: the line's colour MEANS something — green while the
newest epoch is under :data:`CURRENT_FOR` old, red (and the last epoch
written beside the time) when the stream has stopped — and a line breaks at
a gap in the data instead of drawing an outage as motion.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from importlib import resources
from typing import Any

import numpy as np

#: the newest epoch younger than this: the stream is updating (green)
CURRENT_FOR = dt.timedelta(hours=1)
CURRENT_COLOR = "green"
STALE_COLOR = "#d62728"
YLABELS = ("Norður", "Austur", "Upp")
LOGO = "VI_Two_Line_Blue.png"


def status_color(last: dt.datetime, now: dt.datetime) -> str:
    """Green while the newest epoch is recent, red once the stream stopped."""
    return CURRENT_COLOR if now - last < CURRENT_FOR else STALE_COLOR


def now_line(
    axes: Sequence[Any],
    *,
    now: dt.datetime,
    last: dt.datetime,
    start: dt.datetime | None = None,
    stale: str = "last",
    fontsize: float = 18,
) -> str:
    """The plot-time line on each of ``axes``, with the time down the right edge.

    Green while the stream is current, red (and ``stale`` + the last epoch
    written beside the time) once it stopped. The x-axis is widened a little
    past ``now`` (from ``start``, else the current left limit), so the frame
    does not hide the line. Returns the colour.
    """
    from matplotlib import dates as mdates

    color = status_color(last, now)
    stamp = now.strftime("%Y-%m-%d %H:%M:%S")
    if color == STALE_COLOR:
        stamp += f"   ({stale} {last:%Y-%m-%d %H:%M})"
    t_now = float(mdates.date2num(now))  # type: ignore[no-untyped-call]
    for ax in axes:
        left, right = ax.get_xlim()
        if start is not None:
            left = float(mdates.date2num(start))  # type: ignore[no-untyped-call]
        span = max(right, t_now) - left
        ax.set_xlim(left, max(right, t_now + 0.015 * span))
        ax.axvline(x=now, color=color, zorder=2, linewidth=2)
        ax.text(
            1.01, 0.5, stamp, transform=ax.transAxes, rotation="vertical",
            va="center", ha="left", color=color, fontsize=fontsize,
        )  # fmt: skip
    return color


def line_labels(names: Sequence[str]) -> list[str]:
    """The rover alone when every baseline shares one base, else the pair."""
    mixed = len({n[5:] for n in names}) > 1
    return [n if mixed else n[:4] for n in names]


def classic_title(names: Sequence[str], special: str | None) -> str:
    """``SENG-ELDC 6h``: the first baseline and the span, as rtk_gps."""
    return f"{names[0]} {special}" if special else names[0]


def classic_figure(
    series: Sequence[Any],
    *,
    title: str,
    start: dt.datetime,
    end: dt.datetime,
    now: dt.datetime,
    last: dt.datetime,
    interactive: bool = False,
    names: Sequence[str] | None = None,
    ylabels: Sequence[str] | None = None,
) -> Any:
    """The rtk_gps figure of RTK ``series`` (y in mm, zeroed; drawn in cm).

    ``ylabels``: the three y-axis labels (default ``Norður [cm]`` …; see
    :func:`gps_plot.labels.axis_labels`).

    ``names``: every baseline asked for, in order. A baseline without data
    keeps its legend entry ("engin gögn": a silent station is visible) and
    its colour slot, so colours stay put when a station drops out.
    A plain ``Figure`` (no pyplot, no GUI backend) unless ``interactive``.
    """
    from matplotlib import image
    from matplotlib.figure import Figure
    from matplotlib.ticker import AutoMinorLocator

    from gps_plot.compose import break_gaps

    if interactive:
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(13, 20))
    else:
        fig = Figure(figsize=(13, 20))
    axs = fig.subplots(nrows=3, ncols=1)
    names = list(names or [s.label for s in series])
    by_name = {s.label: s for s in series}
    for name, label in zip(names, line_labels(names), strict=True):
        s = by_name.get(name)
        if s is None:  # its colour slot and legend entry, no line
            for i, ax in enumerate(axs):
                ax.plot([], [], label=f"{label} (engin gögn)" if i == 0 else None)
            continue
        x = list(np.asarray(s.t, dtype="datetime64[us]").astype(dt.datetime))
        xs, ys = break_gaps(x, np.asarray(s.y, dtype=float) / 10.0)  # mm -> cm
        for i, ax in enumerate(axs):
            ax.plot(xs, ys[i], label=label if i == 0 else None)
    if ylabels is None:
        ylabels = [f"{name} [cm]" for name in YLABELS]
    for ax, ylabel in zip(axs, ylabels, strict=True):
        ax.grid(which="major", color="#DDDDDD", linewidth=2.0)
        ax.grid(which="minor", color="#EEEEEE", linestyle=":", linewidth=1.5)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.set_ylabel(ylabel, fontsize=16)
        ax.tick_params(axis="both", labelsize=14)
        ax.set_xlim(start, end)
    now_line(axs, now=now, last=last, start=start, stale="síðast")
    axs[0].set_title(title, fontdict={"fontsize": 30, "verticalalignment": "bottom"})
    axs[0].legend(loc="lower left", fontsize=16)
    logo = resources.files("gps_plot").joinpath("assets", LOGO)
    with resources.as_file(logo) as path:
        box = axs[0].get_position()
        inset = fig.add_axes(
            (box.xmin + 0.005, box.ymax - 0.045, 0.16, 0.04), frameon=False
        )
        inset.imshow(image.imread(path), alpha=0.8)
        inset.set_axis_off()
    return fig
