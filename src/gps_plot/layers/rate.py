"""Velocity-product rates on the N/E/U panels (``gpsplot ts --rate``).

The rate is the velocity product's (:func:`geo_dataread.velocity.
estimate_station_velocity`), for the same station, frame, view and
reference as the plot, so the number on a time-series plot is the number
on the map and in the velocity files. Each window gets:

- a line with the product's slope, through the plotted points' weighted
  mean inside the window (the plotted series has its own datum; the slope
  is what the product says);
- a label per panel, ``v = rate ± σ mm/yr``, σ honest (formal × k(T)), or
  the reason the window was refused.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from matplotlib.figure import Figure

#: one colour per window, in order; distinct from the red data, the grey
#: outliers and the dark-blue model line
RATE_COLORS = ("#d95f02", "#1b9e77", "#7570b3", "#e7298a", "#66a61e")
RATE_LINEWIDTH = 1.6


def rate_settings(
    *,
    ref: str | None,
    detrend_terms: Sequence[str] | None,
    view: str,
    pos_id: str | None,
    uncert: float,
    Dir: str | None,
) -> Any:
    """The velocity product's settings matching a ``gpsplot ts`` plot.

    ``--ref plate`` → ``plate``; ``--ref detrend`` with the seasonal alone
    removed (``--detrend-terms periodic``) → ``plate-periodic``, with the
    whole model removed → ``background``.

    Raises:
        ValueError: a plot the product has no counterpart for (an ITRF
            series, other detrend terms, the detrended view of old).
    """
    from geo_dataread.pos import pos_frame
    from geo_dataread.velocity import VelocitySettings

    if ref == "plate":
        vref = "plate"
    elif ref == "detrend":
        terms = set(detrend_terms or ("all",))
        # steps outside a window do not change its slope, and a window with
        # one inside is refused: linear + periodic (± steps) is background
        if "all" in terms or {"linear", "periodic"} <= terms:
            vref = "background"
        elif terms == {"periodic"}:
            vref = "plate-periodic"
        else:
            raise ValueError(
                f"--rate: --detrend-terms {','.join(sorted(terms))} has no velocity "
                "product reference (periodic, or all)"
            )
    else:
        raise ValueError(
            f"--rate: --ref {ref} has no velocity product reference; use "
            "--ref plate or --ref detrend"
        )
    if view not in ("raw", "cleaned"):
        raise ValueError(f"--rate: --view {view} (raw or cleaned)")
    return VelocitySettings(
        frame=pos_frame(pos_id), ref=vref, view=view, uncert=float(uncert),
        pos_dir=Dir,
    )  # fmt: skip


def _tex(text: str) -> str:
    """Plain text safe for the plots' usetex rendering."""
    out = text
    for ch in ("\\", "_", "%", "&", "#", "$", "{", "}"):
        out = out.replace(ch, "\\" + ch)
    return out.replace("<", "$<$").replace(">", "$>$")


def window_label(window: tuple[float, float]) -> str:
    """``2026.62–2026.79`` of a window [yr]."""
    return f"{window[0]:.2f}--{window[1]:.2f}"


def add_rate_lines(
    fig: Figure,
    sta: str,
    t: Any,
    y: Any,
    sigma: Any,
    windows: Sequence[tuple[float, float]],
    settings: Any,
) -> list[Any]:
    """Draw the product rate of each window on the first three axes.

    ``t``/``y``/``sigma`` are the plotted series (yearf, (3, N) mm); the
    lines are anchored on them. Returns the product records, in window
    order.
    """
    import geo_dataread.gps_read as gpsr
    from geo_dataread.velocity import estimate_station_velocity

    tt = np.asarray(t, dtype=float)
    yy = np.atleast_2d(np.asarray(y, dtype=float))
    ss = np.atleast_2d(np.asarray(sigma, dtype=float))
    records = []
    for i, window in enumerate(windows):
        color = RATE_COLORS[i % len(RATE_COLORS)]
        r = estimate_station_velocity(sta, window[0], window[1], settings)
        records.append(r)
        inside = (tt >= window[0]) & (tt <= window[1])
        tag = window_label(window) + ": " if len(windows) > 1 else ""
        for c, ax in enumerate(fig.axes[:3]):
            xlim, ylim = ax.get_xlim(), ax.get_ylim()
            if not r.accepted:
                text = f"{tag}rate refused: {_tex(r.refused or '')}"
            else:
                ok = inside & np.isfinite(yy[c]) & np.isfinite(ss[c]) & (ss[c] > 0)
                if ok.sum() >= 2:
                    w = 1.0 / ss[c, ok] ** 2
                    tm = float(np.sum(w * tt[ok]) / np.sum(w))
                    ym = float(np.sum(w * yy[c, ok]) / np.sum(w))
                    ends = np.array([tt[ok].min(), tt[ok].max()])
                    line = ym + r.rate[c] * (ends - tm)
                    ax.plot(list(gpsr.toDateTime(ends)), line, color=color,
                            linewidth=RATE_LINEWIDTH, zorder=7)  # fmt: skip
                text = f"{tag}$v = {r.rate[c]:.1f} \\pm {r.sigma[c]:.1f}$ mm/yr"
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.text(
                0.01, 0.97 - 0.09 * i, text, transform=ax.transAxes,
                fontsize="small", color=color, va="top",
            )  # fmt: skip
    return records


def rate_note(records: Sequence[Any]) -> str:
    """One line for the figure: what the rates are relative to and their σ."""
    if not records:
        return ""
    r = records[0]
    refs = {x.ref for x in records}
    ref = ", ".join(sorted(refs))
    sig = {x.sigma_method for x in records}
    return _tex(
        f"rates: velocity product, ref {ref}, {r.view}, sigma {'/'.join(sorted(sig))}"
    )
