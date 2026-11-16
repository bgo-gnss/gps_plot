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

import dataclasses
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
    common_mode: bool = True,
) -> Any:
    """The velocity product's settings matching a ``gpsplot ts`` plot
    (``common_mode``: the plot's ``--common-mode``, the product's default).

    ``--ref plate`` → ``plate``; ``--ref detrend`` with the seasonal alone
    removed (``--detrend-terms periodic``) → ``plate-periodic``, with the
    whole model removed → ``background``.

    Raises:
        ValueError: a plot the product has no counterpart for (an ITRF
            series, other detrend terms, the detrended view of old).
    """
    from geo_dataread.pos import frame_name
    from geo_dataread.products.velocity_core import DEFAULT_VIEW, VelocitySettings

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
    # the frame by its full name and the session from the product: the same
    # settings — and fingerprint — as gpsprod velocity and the map (gpsprod
    # step 6)
    session = "8h" if pos_id and ".08hr_" in pos_id else "24h"
    return VelocitySettings(
        frame=frame_name(pos_id), ref=vref, view=DEFAULT_VIEW, uncert=float(uncert),
        pos_dir=Dir, session=session, common_mode=common_mode,
    )  # fmt: skip


def _tex(text: str) -> str:
    """Plain text safe for the plots' usetex rendering."""
    out = text
    for ch in ("\\", "_", "%", "&", "#", "$", "{", "}"):
        out = out.replace(ch, "\\" + ch)
    return out.replace("<", "$<$").replace(">", "$>$")


#: a rate window at least this long [yr] averages the seasonal out
SEASONAL_SAFE_YEARS = 1.0


def seasonal_kept(ref: str | None, detrend_terms: Sequence[str] | None) -> bool:
    """True when the plotted series still carries the seasonal signal:
    ``--ref plate`` (or an ITRF series), or ``--ref detrend`` whose terms
    leave ``periodic`` in (the default terms remove it)."""
    if ref != "detrend":
        return True
    return detrend_terms is not None and "periodic" not in detrend_terms


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


def reference_rate(sta: str, window: tuple[float, float], settings: Any) -> Any:
    """The product rate of ``window`` for ``--rate-ref`` (the record).

    Raises:
        ValueError: the product refused the window or a component is NaN
            (a reference that is not there cannot be subtracted).
    """
    from geo_dataread.velocity import estimate_station_velocity

    r = estimate_station_velocity(sta, window[0], window[1], settings)
    if not r.accepted:
        raise ValueError(
            f"{sta}: --rate-ref {window_label(window)}: rate refused ({r.refused})"
        )
    if not np.all(np.isfinite(r.rate)):
        raise ValueError(
            f"{sta}: --rate-ref {window_label(window)}: a component was not "
            f"estimated ({getattr(r, 'parts_refused', {})})"
        )
    return r


def subtract_rate(t: Any, y: Any, rate: Sequence[float], t0: float) -> Any:
    """``y − v·(t − t0)`` per component: the series relative to rate ``v``.

    ``t`` (N,) [yr], ``y`` (3, N) [mm], ``v`` (3,) [mm/yr]; ``t0`` is where
    the subtracted line is zero (the plot's first epoch keeps its zero).
    Returns a new array; ``y`` is not changed.
    """
    tt = np.asarray(t, dtype=float)
    yy = np.atleast_2d(np.asarray(y, dtype=float))
    v = np.asarray(rate, dtype=float)[:, None]
    return yy - v * (tt - t0)[None, :]


def relative_record(record: Any, ref: Any | None) -> Any:
    """``record`` with ``ref``'s rate subtracted, σ in quadrature.

    The two windows are taken as independent; for overlapping windows the
    σ is an upper bound. The reference window itself keeps its own σ (rate
    0 exactly; quadrature would count the same noise twice). ``ref`` None
    returns ``record`` unchanged.
    """
    if ref is None or not record.accepted:
        return record
    rate = tuple(float(a - b) for a, b in zip(record.rate, ref.rate, strict=True))
    if tuple(record.window) == tuple(ref.window):
        return dataclasses.replace(record, rate=rate)
    sigma = tuple(
        float(np.hypot(a, b)) for a, b in zip(record.sigma, ref.sigma, strict=True)
    )
    return dataclasses.replace(record, rate=rate, sigma=sigma)


def rotate_record(record: Any, axes: Any | None) -> Any:
    """``record``'s N/E/U rate and σ along ``axes`` (``--along``): R·v, and
    σ from R·C·Rᵀ with C built from σ and the record's N–E correlation
    (``corr_en``; Up uncorrelated). ``axes`` None returns ``record``."""
    if axes is None or not record.accepted:
        return record
    from geofunc.axes import rotate

    s = np.asarray(record.sigma, dtype=float)
    rho = float(getattr(record, "corr_en", 0.0) or 0.0)
    cov = np.diag(s**2)
    if np.isfinite(rho) and np.all(np.isfinite(s[:2])):
        cov[0, 1] = cov[1, 0] = rho * s[0] * s[1]
    r = rotate(np.asarray(record.rate, dtype=float), axes, cov=cov)
    assert r.sigma is not None
    return dataclasses.replace(
        record,
        rate=tuple(float(x) for x in r.values),
        sigma=tuple(float(x) for x in r.sigma),
    )


def add_rate_lines(
    fig: Figure,
    sta: str,
    t: Any,
    y: Any,
    sigma: Any,
    windows: Sequence[tuple[float, float]],
    settings: Any,
    *,
    minus: Any | None = None,
    axes: Any | None = None,
) -> list[Any]:
    """Draw the product rate of each window on the first three axes.

    ``t``/``y``/``sigma`` are the plotted series (yearf, (3, N) mm); the
    lines are anchored on them. ``minus``: a ``--rate-ref`` record whose
    rate is subtracted from each (the plotted series is relative to it too);
    ``axes``: the ``--along`` frame the rates are rotated into.
    Returns the product records, in window order.
    """
    import geo_dataread.gps_read as gpsr
    from geo_dataread.velocity import estimate_station_velocity

    tt = np.asarray(t, dtype=float)
    yy = np.atleast_2d(np.asarray(y, dtype=float))
    ss = np.atleast_2d(np.asarray(sigma, dtype=float))
    records = []
    for i, window in enumerate(windows):
        color = RATE_COLORS[i % len(RATE_COLORS)]
        r = rotate_record(
            relative_record(
                estimate_station_velocity(sta, window[0], window[1], settings), minus
            ),
            axes,
        )
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
SEGMENT_SOURCES = ("steps", "events", "auto")
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
    *,
    minus: Any | None = None,
    axes: Any | None = None,
) -> list[Any]:
    """Draw the product rate of each segment on the first three axes.

    Like :func:`add_rate_lines`, but for many windows: each line is labelled
    at its own middle (``rate ± σ``, mm/yr), alternating two heights, and a
    refused segment gets a ``×`` there (its reason is in the returned
    record). ``minus`` as in :func:`add_rate_lines`. Returns the product
    records, in window order.
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
        r = rotate_record(
            relative_record(
                estimate_station_velocity(sta, window[0], window[1], settings), minus
            ),
            axes,
        )
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
        elif source == "auto":
            continue  # from the plotted data: auto_break_gaps
        else:
            raise ValueError(f"--rate-segments {source!r}: one of {SEGMENT_SOURCES}")
    return gaps


#: colour of the --rate-segments auto break lines
BREAK_COLOR = "saddlebrown"


@dataclasses.dataclass(frozen=True)
class AutoBreaks:
    """``--rate-segments auto``: where the rate changes, and on what."""

    breaks: tuple[float, ...]  # fractional years
    bic_gains: tuple[float, ...]  # BIC drop of each accepted break, in order
    series: str  # what was searched: "along 284.0°", "motion 284.0°"
    min_gap: float
    min_bic_gain: float


def break_component(
    sta: str, t: Any, y: Any, sigma: Any, axes: Any | None
) -> tuple[Any, str]:
    """The one series the breaks are looked for in.

    With ``--along`` the plotted first component (motion toward its
    azimuth); otherwise the horizontals projected on the direction of the
    plotted motion (``--along opt``'s estimator), so a station gets one set
    of breaks for all its panels.
    """
    yy = np.atleast_2d(np.asarray(y, dtype=float))
    if axes is not None:
        return yy[0], str(axes.keys[0])
    from gps_plot.direction import along_azimuth

    az, _ = along_azimuth(sta, "opt", t, yy, sigma)
    a = np.radians(az)
    return yy[0] * np.cos(a) + yy[1] * np.sin(a), f"motion {az:.1f}°"


def auto_breaks(
    sta: str,
    t: Any,
    y: Any,
    sigma: Any,
    axes: Any | None = None,
    *,
    min_bic_gain: float | None = None,
) -> AutoBreaks:
    """Rate changes of the plotted series (:func:`gps_analysis.breaks.
    rate_breaks`, its defaults unless ``min_bic_gain`` is given)."""
    from gps_analysis.breaks import MIN_BIC_GAIN, rate_breaks

    comp, label = break_component(sta, t, y, sigma, axes)
    tt = np.asarray(t, dtype=float)
    ok = np.isfinite(tt) & np.isfinite(comp)
    gain = MIN_BIC_GAIN if min_bic_gain is None else float(min_bic_gain)
    rb = rate_breaks(tt[ok], comp[ok], min_bic_gain=gain)
    gains = tuple(float(a - b) for a, b in zip(rb.bic[:-1], rb.bic[1:], strict=True))
    return AutoBreaks(tuple(rb.breaks), gains, label, rb.min_gap, gain)


def break_gaps(found: AutoBreaks) -> list[tuple[dt.date, dt.date]]:
    """The breaks as one-day gaps (each break's day splits the series)."""
    return step_gaps(found.breaks)


def break_note(found: AutoBreaks) -> str:
    """Console line: the breaks, their BIC gains and the settings."""
    if not found.breaks:
        return (
            f"rate breaks ({found.series}): none (BIC gain >= {found.min_bic_gain:g})"
        )
    bits = ", ".join(f"{b:.2f}" for b in found.breaks)
    gains = ", ".join(f"{g:.0f}" for g in found.bic_gains)
    return (
        f"rate breaks ({found.series}): {bits}; BIC gains {gains} "
        f"(>= {found.min_bic_gain:g}, min gap {found.min_gap:.2f} yr)"
    )
