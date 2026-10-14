"""The stored-model line over the N/E/U panels (``gpsplot ts --model``)."""

from __future__ import annotations

from typing import Any

import numpy as np
from matplotlib.figure import Figure

from gps_plot.series import Series

#: the model line: dark blue, distinct from the red data and grey outliers
MODEL_COLOR = "#1f4e9c"
MODEL_LINEWIDTH = 1.3


def fit_window_mask(t: Any, window: Any) -> Any:
    """True where ``t`` lies inside the record's fit window (all if unknown)."""
    t = np.asarray(t, dtype=float)
    if not window or len(window) != 2:
        return np.ones(t.shape, dtype=bool)
    return (t >= float(window[0])) & (t <= float(window[1]))


def add_model_curve(fig: Figure, model: Series) -> Figure:
    """Draw ``model`` on the first three axes: solid inside the fit window.

    Outside ``model.meta["window"]`` the record is extrapolated, so the line
    is dashed there; the solid part is drawn over it. Axis limits are kept:
    the data sets the scale, the model is drawn into it.
    """
    import geo_dataread.gps_read as gpsr

    if len(model.t) == 0:
        return fig
    x = np.asarray(list(gpsr.toDateTime(model.t)))
    inside = fit_window_mask(model.t, model.meta.get("window"))
    for ax, row in zip(fig.axes[:3], np.asarray(model.y, dtype=float), strict=True):
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        ax.plot(
            x,
            row,
            color=MODEL_COLOR,
            linewidth=MODEL_LINEWIDTH * 0.8,
            linestyle="--",
            zorder=5,
        )
        ax.plot(
            x,
            np.where(inside, row, np.nan),
            color=MODEL_COLOR,
            linewidth=MODEL_LINEWIDTH,
            zorder=6,
        )
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        note = off_scale_note(row, ylim)
        if note:
            ax.text(
                0.99,
                0.97 if np.nanmedian(row) > ylim[1] else 0.03,
                note,
                transform=ax.transAxes,
                fontsize="small",
                color=MODEL_COLOR,
                ha="right",
                va="top" if np.nanmedian(row) > ylim[1] else "bottom",
            )
    return fig


def off_scale_note(row: Any, ylim: tuple[float, float]) -> str | None:
    """A note when the whole line lies outside the data's y range.

    A model fitted long ago (e.g. a pre-unrest background) can sit hundreds
    of mm from a windowed series; that offset is the signal, so it is
    stated rather than hidden, and the data keep their scale.
    """
    row = np.asarray(row, dtype=float)
    finite = row[np.isfinite(row)]
    if finite.size == 0:
        return None
    lo, hi = min(ylim), max(ylim)
    if np.all(finite > hi) or np.all(finite < lo):
        return f"model line off scale, at {float(np.median(finite)):.0f} mm"
    return None


def model_note(model: Series) -> str:
    """One line for the figure: what the line is and where it was fitted."""
    terms = model.meta.get("terms")
    what = "stored model" + (f" ({' + '.join(terms)})" if terms else "")
    window = model.meta.get("window")
    if window and len(window) == 2:
        return (
            f"line: {what}, fitted {float(window[0]):.2f} to "
            f"{float(window[1]):.2f}, dashed outside"
        )
    return f"line: {what}"
