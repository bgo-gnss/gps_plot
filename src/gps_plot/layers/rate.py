"""Velocity-product rates on the N/E/U panels (``gpsplot ts --rate``).

The rate is the velocity product's (:func:`geo_dataread.velocity.
estimate_station_velocity`), for the same station, frame, view and
reference as the plot, so the number on a time-series plot is the number
on the map and in the velocity files for the same ref (the map and
``gps-velocities`` default to plate-periodic + declared; the ref follows
the plot's ``--ref``, the view is always the product's default, declared,
whatever the plot's ``--view``). Each window gets:

- a line with the product's slope, through the plotted points' weighted
  mean inside the window (the plotted series has its own datum; the slope
  is what the product says);
- a label per panel, ``v = rate ± σ mm/yr``, σ honest (formal × k(T)), or
  the reason the window was refused.
"""

from __future__ import annotations

import datetime as dt
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
    from geo_dataread.velocity import DEFAULT_VIEW, VelocitySettings

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
    # the product's view, not the plot's: the rate is the map's number
    # (declared epochs out, no detection) on a raw plot (the ts default) and
    # on a cleaned one; the raw-view rate is gps-velocities --view raw
    return VelocitySettings(
        frame=pos_frame(pos_id), ref=vref, view=DEFAULT_VIEW, uncert=float(uncert),
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


def default_window(
    t: Any, fstart: float | None, fend: float | None
) -> tuple[float, float]:
    """The ``--rate`` window without ``--rate-window``: whole days, as the
    map's and ``gps-velocities``' windows, from the plot's read start (else
    its first epoch) to its read end (else its last epoch). Same T, so the
    same k(T) and σ as ``map --window`` over the same dates."""
    import geo_dataread.gps_read as gpsr
    from geo_dataread.velocity import window_bounds

    tt = np.asarray(t, dtype=float)
    a = fstart if fstart is not None else float(np.nanmin(tt))
    b = fend if fend is not None else float(np.nanmax(tt))
    # +~1 min: a midnight read bound must not round back into the day before
    first, last = (d.date() for d in gpsr.toDateTime(np.array([a, b]) + 2e-6))
    t0, t1 = window_bounds(first, last)
    return float(t0), float(t1)


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
        inside = (tt >= window[0]) & (tt < window[1])
        tag = window_label(window) + ": " if len(windows) > 1 else ""
        for c, ax in enumerate(fig.axes[:3]):
            xlim, ylim = ax.get_xlim(), ax.get_ylim()
            if not r.accepted:
                text = f"{tag}rate refused: {_tex(r.refused or '')}"
            else:
                ok = inside & np.isfinite(yy[c]) & np.isfinite(ss[c]) & (ss[c] > 0)
                if ok.sum() >= 2 and np.isfinite(r.rate[c]):
                    w = 1.0 / ss[c, ok] ** 2
                    tm = float(np.sum(w * tt[ok]) / np.sum(w))
                    ym = float(np.sum(w * yy[c, ok]) / np.sum(w))
                    ends = np.array([tt[ok].min(), tt[ok].max()])
                    line = ym + r.rate[c] * (ends - tm)
                    ax.plot(list(gpsr.toDateTime(ends)), line, color=color,
                            linewidth=RATE_LINEWIDTH, zorder=7)  # fmt: skip
                text = f"{tag}$v = {r.rate[c]:.1f} \\pm {r.sigma[c]:.1f}$ mm/yr"
                if not np.isfinite(r.rate[c]):  # a short-tier part refused
                    part = "up" if c == 2 else "horizontal"
                    why = r.parts_refused.get(part, "not estimated")
                    text = f"{tag}rate refused: {_tex(part + ' ' + why)}"
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.text(
                0.01, 0.97 - 0.09 * i, text, transform=ax.transAxes,
                fontsize="small", color=color, va="top",
            )  # fmt: skip
    return records


def rate_note(records: Sequence[Any], *, segments: bool = False) -> str:
    """One line for the figure: what the rates are relative to and their σ
    (``segments``: also what the bare segment labels are)."""
    if not records:
        return ""
    r = records[0]
    refs = {x.ref for x in records}
    ref = ", ".join(sorted(refs))
    sig = {x.sigma_method for x in records}
    seg = "; labels rate ± sigma mm/yr between events, x refused" if segments else ""
    return _tex(
        f"rates: velocity product, ref {ref}, {r.view}, sigma {'/'.join(sorted(sig))}"
        + seg
    )


# ---------------------------------------------------------------------------
# rates between events (gpsplot ts --rate-segments)
# ---------------------------------------------------------------------------

#: --rate-segments sources: the station's declared steps, or the events
SEGMENT_SOURCES = ("steps", "events")
#: shortest span between two events that gets a rate [days]
MIN_SEGMENT_DAYS = 30


def segment_spans(
    first: dt.date,
    last: dt.date,
    gaps: Sequence[tuple[dt.date, dt.date]],
    *,
    min_days: int = MIN_SEGMENT_DAYS,
) -> list[tuple[dt.date, dt.date]]:
    """The spans of ``[first, last]`` (whole days, both in) between gaps.

    Each gap ``(a, b)`` takes out its days a..b, both in: a step's day, or
    an event from its onset to its end. Gaps may overlap and need not be
    sorted. Spans shorter than ``min_days`` days are dropped.
    """
    spans: list[tuple[dt.date, dt.date]] = []
    cursor = first
    one = dt.timedelta(days=1)
    for a, b in sorted((min(g), max(g)) for g in gaps):
        if b < cursor:
            continue
        if a > last:
            break
        if a > cursor:
            spans.append((cursor, min(a - one, last)))
        cursor = max(cursor, b + one)
    if cursor <= last:
        spans.append((cursor, last))
    return [(a, b) for a, b in spans if (b - a).days + 1 >= min_days]


def step_gaps(epochs: Any) -> list[tuple[dt.date, dt.date]]:
    """Gaps of declared steps: each step's day (fractional year, noon)."""
    import geo_dataread.gps_read as gpsr

    ee = np.asarray(epochs, dtype=float)
    if ee.size == 0:
        return []
    return [(d.date(), d.date()) for d in gpsr.toDateTime(ee)]


def event_gaps(
    events: Sequence[Any], *, cut: str = "span"
) -> list[tuple[dt.date, dt.date]]:
    """Gaps of events (``geo_dataread.events.Event``): onset to last day
    (``span``), or the onset day alone (``onset``)."""
    if cut == "onset":
        return [(e.onset, e.onset) for e in events]
    return [(e.onset, e.last_day) for e in events]


def segment_windows(
    t: Any,
    fstart: float | None,
    fend: float | None,
    gaps: Sequence[tuple[dt.date, dt.date]],
    *,
    min_days: int = MIN_SEGMENT_DAYS,
) -> list[tuple[float, float]]:
    """The velocity-product windows [yr] between ``gaps`` over the plotted
    span (as :func:`default_window`): whole days, as ``--rate-window``."""
    import geo_dataread.gps_read as gpsr
    from geo_dataread.velocity import window_bounds

    t0, t1 = default_window(t, fstart, fend)
    first, last = (d.date() for d in gpsr.toDateTime(np.array([t0, t1]) + 2e-6))
    last = last - dt.timedelta(days=1)  # window_bounds' end is exclusive
    return [
        tuple(map(float, window_bounds(a, b)))  # type: ignore[misc]
        for a, b in segment_spans(first, last, gaps, min_days=min_days)
    ]


def add_segment_rates(
    fig: Figure,
    sta: str,
    t: Any,
    y: Any,
    sigma: Any,
    windows: Sequence[tuple[float, float]],
    settings: Any,
) -> list[Any]:
    """Draw the product rate of each segment on the first three axes.

    Like :func:`add_rate_lines`, but for many windows: each line is labelled
    at its own middle (``rate ± σ``, mm/yr), alternating two heights, and a
    refused segment gets a ``×`` there (its reason is in the returned
    record). Returns the product records, in window order.
    """
    import geo_dataread.gps_read as gpsr
    from geo_dataread.velocity import estimate_station_velocity
    from matplotlib.transforms import blended_transform_factory

    tt = np.asarray(t, dtype=float)
    yy = np.atleast_2d(np.asarray(y, dtype=float))
    ss = np.atleast_2d(np.asarray(sigma, dtype=float))
    records = []
    for i, window in enumerate(windows):
        color = RATE_COLORS[i % 2]
        r = estimate_station_velocity(sta, window[0], window[1], settings)
        records.append(r)
        inside = (tt >= window[0]) & (tt < window[1])
        mid = list(gpsr.toDateTime(np.array([0.5 * (window[0] + window[1])])))[0]
        for c, ax in enumerate(fig.axes[:3]):
            xlim, ylim = ax.get_xlim(), ax.get_ylim()
            text = "$\\times$"
            if r.accepted and np.isfinite(r.rate[c]):
                ok = inside & np.isfinite(yy[c]) & np.isfinite(ss[c]) & (ss[c] > 0)
                if ok.sum() >= 2:
                    w = 1.0 / ss[c, ok] ** 2
                    tm = float(np.sum(w * tt[ok]) / np.sum(w))
                    ym = float(np.sum(w * yy[c, ok]) / np.sum(w))
                    ends = np.array([tt[ok].min(), tt[ok].max()])
                    line = ym + r.rate[c] * (ends - tm)
                    ax.plot(list(gpsr.toDateTime(ends)), line, color=color,
                            linewidth=RATE_LINEWIDTH, zorder=7)  # fmt: skip
                text = f"${r.rate[c]:.0f}\\pm{r.sigma[c]:.0f}$"
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            trans = blended_transform_factory(ax.transData, ax.transAxes)
            ax.text(
                mid, 0.97 - 0.06 * (i % 2), text, transform=trans,
                fontsize="x-small", color=color, va="top", ha="center",
            )  # fmt: skip
    return records


def segment_table(sta: str, records: Sequence[Any]) -> str:
    """The segments' rates as text (one line each), for the terminal."""
    rows = [f"{sta}: rates between events (mm/yr, N E U ± σ)"]
    for r in records:
        span = f"{r.window[0]:.3f}–{r.window[1]:.3f}"
        if not r.accepted:
            rows.append(f"  {span}  refused: {r.refused}")
            continue
        vals = "  ".join(
            f"{v:7.1f} ±{s:5.1f}" for v, s in zip(r.rate, r.sigma, strict=True)
        )
        flags = f"  [{', '.join(r.flags)}]" if r.flags else ""
        rows.append(f"  {span}  {vals}{flags}")
    return "\n".join(rows)


def parse_segment_sources(value: str) -> tuple[str, ...]:
    """``steps``, ``events`` or both (comma-separated), de-duplicated.

    Raises:
        ValueError: An unknown or no source.
    """
    out: list[str] = []
    for v in (p.strip() for p in value.split(",")):
        if v not in SEGMENT_SOURCES:
            raise ValueError(f"--rate-segments {v!r}: one of {SEGMENT_SOURCES}")
        if v not in out:
            out.append(v)
    if not out:
        raise ValueError(f"--rate-segments: one of {SEGMENT_SOURCES}")
    return tuple(out)


def segment_gaps(
    sta: str,
    sources: Sequence[str],
    *,
    kinds: Sequence[str] = ("dike", "eruption"),
    cut: str = "span",
    steps: str | None = None,
    events: str | None = None,
) -> list[tuple[dt.date, dt.date]]:
    """The gaps that split ``sta``'s series for ``--rate-segments``.

    ``cut``: an event is out from onset to end (``span``) or on its onset
    day only (``onset``).

    Raises:
        ValueError: An unknown source, kind or cut, or no catalog to read (a
            missing catalog must not pass as "no events").
    """
    if cut not in ("span", "onset"):
        raise ValueError(f"--segment-cut {cut!r}: span or onset")
    gaps: list[tuple[dt.date, dt.date]] = []
    for source in sources:
        if source == "steps":
            from geo_dataread.gps_views import read_step_catalog

            try:
                catalog = read_step_catalog(steps)
            except FileNotFoundError as exc:
                raise ValueError(f"--rate-segments steps: {exc}") from None
            gaps += step_gaps(catalog.get(sta, ()))
        elif source == "events":
            from geo_dataread.events import (
                read_events,
                read_station_areas,
                station_events,
            )

            try:
                catalog, areas = read_events(events), read_station_areas()
            except FileNotFoundError as exc:
                raise ValueError(
                    f"--rate-segments events: {exc} (deploy events.yaml from "
                    "gps-config-data analysis-lane/, or pass --events-catalog PATH)"
                ) from None
            lost = sorted(
                {e.region or "(none)" for e in catalog if e.region not in areas}
            )
            if lost:  # such an event splits no station's series: say so
                import warnings

                warnings.warn(
                    f"events with region(s) {lost} not in station_areas.yaml "
                    "split no series",
                    UserWarning,
                    stacklevel=2,
                )
            evs = station_events(sta, catalog, areas, kinds=kinds)
            gaps += event_gaps(evs, cut=cut)
        else:
            raise ValueError(f"--rate-segments {source!r}: one of {SEGMENT_SOURCES}")
    return gaps
