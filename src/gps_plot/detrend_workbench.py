#!/usr/bin/env python
"""Operator workbench for detrending — one station, one PDF, two views.

Choosing a detrend model is **curation, not computation**: whether a station
supports only periodic terms or only a rate, which window is pre-unrest, whose
parameters to borrow, which events deserve an offset term — these are
judgements about a site's history that no estimator makes from the data alone.
The math is already done (:mod:`gps_analysis.detrend`); this is the loop that
lets a human drive it: look, adjust, look, commit.

Design notes:

- **Estimation is not reimplemented.**  The record comes from
  ``geo_dataread.detrend_estimate.station_record_from_arrays`` — the same
  function ``gps-estimate-detrend`` uses — so the workbench and the batch
  estimator can never disagree about what a record means.
- **Full detection by default, with a LOUD S0 fallback.**  An earlier
  version defaulted to S0-only, reasoning that S0 is robust to undeclared
  steps.  That reasoning was wrong here, and measurably so: §14 justified S0
  for *flagging*, where the virtue is a verdict that does not move with the
  window's composition.  For *estimation* the inlier set DEFINES the fitted
  trajectory — S0 leaves the model-visible outliers in, the WLS absorbs them,
  and the record encodes the bias.  Measured: RHOF vertical rate 0.642
  (S0) vs 0.105 mm/yr (full), n_rejected [4,6,2] vs [48,31,23].
  And the fallback case is rarer than it looked: the full pipeline aborted on
  1 of 28 stations tested, and on that one (SELF) it aborts only while the
  143 mm coseismic is UNdeclared — a state whose record is visibly garbage
  and would never be committed.  Once the operator declares the step, full
  runs and beats S0 on every component (rms [1.97, 3.49, 6.54] vs
  [2.31, 3.98, 10.87]).  So S0 protects only records nobody should commit.
  It remains available, and the fallback is announced rather than silent.
- **Two pages, not two files.**  ``timesmatplt.saveFig`` writes one Figure;
  a two-view PDF needs ``PdfPages``.  This is the one place the workbench
  departs from the production drawing seam, deliberately.
- **The text summary is half the tool.**  A figure shows whether the fit
  tracks; the printed record (model, window, span, n_rejected, RMS, step
  amplitudes) is what you actually judge before committing.

This module never writes configuration.  Committing a record is a separate
ticket and a separate, explicit flag.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.figure import Figure

__all__ = [
    "build_record",
    "borrow_record",
    "commit_record",
    "screen_outside_window",
    "split_outliers",
    "render",
    "main",
]

#: Colour of the fitted-trajectory overlay on the plate-frame panel.  Blue
#: against the red/grey/gold marker vocabulary of the cleaned view, so the
#: model never reads as a data state.
FIT_COLOR: str = "royalblue"
FIT_WIDTH: float = 1.8

#: Apply-time term selection the READ path uses.  ``evaluate_record`` /
#: ``apply_stored_detrend`` default to ``"all"``, and the record has no field
#: to say otherwise, so this is the only value a committed record can be
#: reproduced under.  Anything else is a look-only choice -- see ``main``.
APPLY_TERMS_DEFAULT: str = "all"

#: ``uncert`` default of :func:`geo_dataread.gps_read.getData`, which is what
#: the batch estimator ``gps-estimate-detrend`` uses when ``--uncert`` is not
#: passed.  The workbench screens harder by default (see ``--uncert``), so a
#: commit says how to reproduce the record in batch rather than assuming it.
BATCH_UNCERT_DEFAULT: int = 15


def _stage_params(stages: str | None, extra: list[str] | None = None) -> Any:
    """``OutlierParams`` for a stage selection, via the CLI's own mapping.

    ``stages=None`` means the full pipeline (catalog/spec defaults decide).
    Reuses :func:`plot_gps_timeseries._stage_overrides` /
    :func:`_build_outlier_params` rather than restating which fields a stage
    corresponds to — one definition of "S0", in one place.
    """
    from gps_plot.plot_gps_timeseries import _build_outlier_params, _stage_overrides

    return _build_outlier_params(extra or [], base=_stage_overrides(stages))


def _declared_step_epochs(sta: str, *sources: Any) -> tuple[float, ...]:
    """Union of every step declaration in play, the deployed catalog included.

    ``steps.csv`` is a FLOOR, never a fallback: the fit-catalog column and a
    CLI ``--step`` add to it.  Consulting the catalog only when the other
    sources are empty is the precise bug this exists to prevent —
    ``station_record_from_arrays`` does exactly that, so declaring one
    equipment offset from the CLI used to silently drop a station's declared
    coseismic (measured on SENG, 2 declared rows: rms N/E 112.85/178.73 ->
    271.94/592.92 mm).

    Args:
        sta: Station name; "" skips the catalog lookup.
        *sources: Any iterables of epochs [yr]; None entries are ignored.

    Returns:
        The sorted union.  A catalog failure degrades to "nothing declared"
        rather than raising — catalogs are enhancements on this path.
    """
    declared: set[float] = set()
    for source in sources:
        if source is not None:
            declared |= {float(v) for v in np.atleast_1d(np.asarray(source)).ravel()}
    if sta:
        try:
            from geo_dataread.gps_views import station_step_epochs

            epochs, _src = station_step_epochs(sta)
            declared |= {float(v) for v in np.atleast_1d(epochs)}
        except Exception:  # catalogs are enhancements, never a hard failure
            pass
    return tuple(sorted(declared))


def _override_settings(settings: Any, sta: str = "", **over: Any) -> Any:
    """Apply CLI curation levers on top of the resolved catalog row.

    Only non-None values override, so an unset flag defers to the catalog,
    which defers to the defaults -- the same precedence the rest of the
    ecosystem uses.  ``window_source`` is rewritten whenever anything is
    overridden, because the record's provenance must say the operator chose
    this, not a catalog.

    Why this exists at all: `fit_windows.csv` has no model column and no way
    to say "these gates, just for this look", so without CLI overrides the
    only route to a different window is editing a deployed catalog between
    iterations -- which is exactly the loop the workbench is meant to remove.
    """
    import dataclasses

    changed = {k: v for k, v in over.items() if v is not None}
    if not changed:
        return settings
    if "steps" in changed:
        # MERGE with whatever is already declared, never replace -- the help
        # text reads additive, and _declared_step_epochs documents the cost of
        # getting it wrong.  The sources differ: settings.steps is the
        # fit-catalog column, the lookup inside is steps.csv; both fold in.
        declared = _declared_step_epochs(sta, settings.steps)
        merged = _declared_step_epochs(sta, settings.steps, changed["steps"])
        changed["steps"] = merged
        if declared:
            print(
                f"note: --step merged with {len(declared)} already-declared "
                f"step(s); fitting {len(merged)} in total: {merged}",
                file=sys.stderr,
            )
    if "window" in changed:
        changed["window"] = tuple(changed["window"])
    prior = settings.window_source or "defaults"
    changed["window_source"] = f"workbench-cli(+{prior})"
    return dataclasses.replace(settings, **changed)


def build_record(
    sta: str,
    *,
    tot_dir: str | None = None,
    uncert: int = 10,
    outlier_param: list[str] | None = None,
    fit_catalog: str | Path | None = None,
    model: str | None = None,
    stages: str | None = None,
    window: tuple[float | None, float | None] | None = None,
    steps: Sequence[float] | None = None,
    max_gap_years: float | None = None,
    min_epochs: int | None = None,
    min_span_years: float | None = None,
) -> tuple[dict[str, Any], Any, Any, Any, Any]:
    """Read the plate-frame series and estimate one station's record.

    Returns:
        ``(record, yearf, data, sigma, estimate)`` — ``estimate`` is the
        :class:`geo_dataread.detrend_estimate.StationEstimate` of the
        WINNING fit (the S0 fallback's, on a fallback), carrying the
        inlier mask the record itself cannot.

    Raises:
        RuntimeError: When estimation returns None.  That is NOT a bug: it
            is the validity gates or an outlier abort refusing to store a
            record, and the workbench must say which, loudly, rather than
            render a figure that implies success.
    """
    import geo_dataread.gps_read as gpsr
    from geo_dataread.detrend_estimate import (
        FitDefaults,
        default_fit_catalog_path,
        read_fit_catalog,
        resolve_fit_settings,
        station_estimate_from_arrays,
    )

    yearf, data, sigma, _offset = gpsr.getData(
        sta, ref="plate", Dir=tot_dir, tType="TOT", uncert=uncert
    )
    if yearf is None or len(yearf) == 0:
        raise RuntimeError(f"no data for station {sta}")

    catalog = None
    source = None
    path = Path(fit_catalog) if fit_catalog else default_fit_catalog_path()
    if path and Path(path).is_file():
        catalog = read_fit_catalog(path)
        source = str(path)

    settings = resolve_fit_settings(sta, catalog, FitDefaults(), catalog_source=source)
    settings = _override_settings(
        settings,
        sta=sta,
        window=window,
        steps=steps,
        max_gap_years=max_gap_years,
        min_epochs=min_epochs,
        min_span_years=min_span_years,
    )
    kwargs: dict[str, Any] = {}
    if model:
        kwargs["model"] = model

    gates = (
        f"settings from {settings.window_source}; window={settings.window}, "
        f"max_gap_years={settings.max_gap_years}, "
        f"min_epochs={settings.min_epochs}, "
        f"min_span_years={settings.min_span_years}"
    )

    def _estimate(params: Any) -> Any:
        # `uncert` screens sigma at READ time (getData above), so the estimator
        # never sees it and a stored record would be silent about it -- yet the
        # workbench default (10) differs from the batch estimator's (15), which
        # made workbench and batch records indistinguishable but unequal.
        kwargs["refs"] = {"uncert": uncert}
        return station_estimate_from_arrays(
            sta,
            yearf,
            data,
            sigma,
            settings=settings,
            outlier_params=params,
            **kwargs,
        )

    try:
        if stages:
            # explicit operator override -- no fallback, they asked for this
            estimate = _estimate(_stage_params(stages, outlier_param))
        else:
            estimate = _estimate(_stage_params(None, outlier_param))
            if estimate is None:
                # Outlier ABORT, not a gate failure. Fall back to S0 so the
                # station is estimable at all -- but say so, because an S0
                # record carries absorbed-outlier bias the full one does not.
                print(
                    f"note: {sta}: full detection aborted (excess-candidate "
                    f"rule); falling back to S0-only. An S0 record leaves "
                    f"model-visible outliers in the fit, so its parameters "
                    f"are biased relative to a full-detection record. If this "
                    f"station has an undeclared step, --step is the better "
                    f"fix: declaring it usually stops the abort.",
                    file=sys.stderr,
                )
                # REBIND, so the mask that reaches the figure is the one the
                # surviving record was fitted with -- keeping the aborted run's
                # estimate here would grey out epochs this record never rejected.
                estimate = _estimate(_stage_params("S0", outlier_param))
    except ValueError as exc:
        # The leaf RAISES on a failed validity gate and RETURNS None on an
        # outlier abort -- two different refusals for the same operator
        # question ("why is there no record?"). Normalise them here so the
        # workbench answers that question the same way either time.
        raise RuntimeError(f"{sta}: {exc} ({gates})") from None
    if estimate is None:
        raise RuntimeError(
            f"{sta}: outlier stage aborted — no record stored ({gates}). "
            f"the S0-only fallback aborted too; the series needs a narrower "
            f"window or a declared step (--step)."
        )
    return estimate.record, yearf, data, sigma, estimate


#: Sentinel ``time_from`` TOS uses for "since forever"; not an event.
TOS_SENTINEL_YEAR = 1900

#: Colour of TOS equipment-change lines.  Distinct from the fit overlay
#: (royalblue) and from any data-state colour.
TOS_COLOR: str = "darkgreen"


def tos_equipment_epochs(
    sta: str, *, url: str | None = None, payload: Any = None
) -> list[tuple[float, str]]:
    """TOS equipment-change epochs for one station, coalesced.

    "No detection — a lookup" undersells the work.  A station's
    ``children_connections`` is one row per DEVICE join, so a single site
    visit that swapped antenna + receiver + radome appears as three rows
    sharing a ``time_from`` (SELF: 17 joins, 7 distinct epochs).  There is
    also a ``1000-01-01`` sentinel meaning "since forever", which is not an
    event.  Coalescing to distinct calendar days is the whole job.

    Args:
        sta: station code.
        url: TOS REST endpoint; None uses ``tostools``' own default.
        payload: pre-fetched entity dict (a cached fixture in tests) —
            when given, no network call is made.

    Returns:
        ``[(yearf, label), …]`` sorted, one per distinct day, the label
        naming how many devices changed together.

    Raises:
        RuntimeError: on any TOS failure.  Callers degrade to a warning —
        the workbench must stay usable off-VPN.
    """
    from gtimes.timefunc import TimetoYearf

    if payload is None:
        try:
            # lazy: tostools is not a declared gps_plot dependency, and the
            # workbench must import cleanly without it
            from tostools.gps_metadata_qc import URL_REST_TOS, get_station_metadata

            _contact, payload = get_station_metadata(sta, url or URL_REST_TOS)
        except (Exception, SystemExit) as exc:
            # SystemExit is not academic: tostools' get_station_metadata calls
            # sys.exit() on a connection failure rather than raising, and
            # SystemExit derives from BaseException — so a plain `except
            # Exception` lets a TOS outage kill the workbench outright. Found
            # by test_tos_failure_degrades_to_a_warning, which is the whole
            # reason that test points at a dead port.
            raise RuntimeError(
                f"TOS lookup failed for {sta} ({type(exc).__name__}: {exc})"
            ) from None

    joins = (payload or {}).get("children_connections") or []
    by_day: dict[str, int] = {}
    for join in joins:
        raw = str(join.get("time_from") or "")[:10]
        if not raw:
            continue
        try:
            y, m, d = (int(v) for v in raw.split("-"))
        except ValueError:
            continue
        if y <= TOS_SENTINEL_YEAR:  # "since forever", not an event
            continue
        by_day[raw] = by_day.get(raw, 0) + 1

    out: list[tuple[float, str]] = []
    for day, n in sorted(by_day.items()):
        y, m, d = (int(v) for v in day.split("-"))
        out.append(
            (float(TimetoYearf(y, m, d)), f"{day} ({n} device{'s' if n > 1 else ''})")
        )
    return out


def add_event_lines(
    fig: Figure, events: Sequence[tuple[float, str]], color: str
) -> Figure:
    """Vertical lines with a label on the top axis.

    Lines go through ``timesmatplt.addEvent`` (the existing primitive —
    ``axvline`` on every axis); only the text is new, and only on axis 0,
    because repeating it on all three is noise.
    """
    import gps_plot.timesmatplt as tplt
    from gtimes.timefunc import TimefromYearf

    if not events:
        return fig
    tplt.addEvent({TimefromYearf(e): [color] for e, _ in events}, fig, linestyle=":")
    ax = fig.axes[0]
    lo, hi = ax.get_ylim()
    for epoch, label in events:
        ax.text(
            TimefromYearf(epoch),
            hi,
            label.split(" ")[0],
            rotation=90,
            va="top",
            ha="right",
            fontsize=8,
            color=color,
            zorder=6,
        )
    return fig


#: Colour of seismic-event lines.  Distinct from equipment (darkgreen) and
#: from the fit overlay (royalblue): the whole value of tier A is telling the
#: two apart at a glance.
SEISMIC_COLOR: str = "darkred"

#: ``steps.csv`` ``kind`` values treated as seismic rather than equipment.
SEISMIC_KINDS: tuple[str, ...] = ("earthquake", "coseismic", "seismic")


def declared_event_epochs(
    sta: str, *, steps_catalog: str | Path | None = None
) -> tuple[list[tuple[float, str]], list[tuple[float, str]]]:
    """Declared steps for one station, split seismic vs other.

    Reads ``steps.csv`` through :func:`gps_parser.outlier_catalogs.read_steps`
    rather than ``gps_views.station_step_epochs``, because the latter flattens
    to bare epochs and DROPS ``kind``/``source``/``comment`` — and ``kind`` is
    exactly what distinguishes an earthquake from an antenna swap.

    There is deliberately no seismic-catalogue client here: none exists
    anywhere in the ecosystem (skjálftalísa appears only as a *planned*
    source in ``analysis.yaml`` and the ``steps.csv`` header), and writing one
    is its own project.  The seismic half is therefore served from what an
    operator has already declared, plus ``--events`` for anything not yet
    declared.

    Returns:
        ``(seismic, other)``, each ``[(yearf, label), …]`` sorted.  Rows are
        de-duplicated by epoch: a step declared for N, E and U separately is
        one event, not three.
    """
    try:
        from gps_parser.outlier_catalogs import read_steps

        catalog = read_steps(steps_catalog)
    except Exception:
        # catalogs are enhancements; a missing one is not an error
        return [], []

    seismic: dict[float, str] = {}
    other: dict[float, str] = {}
    for row in catalog.get(sta.upper(), ()):  # type: ignore[union-attr]
        epoch = float(row.epoch_yearf)
        kind = (row.kind or "").strip().lower()
        note = (row.comment or "").strip()
        label = f"{kind or 'step'}"
        if note:
            label = f"{label}: {note[:40]}"
        bucket = seismic if kind in SEISMIC_KINDS else other
        bucket.setdefault(epoch, label)
    return sorted(seismic.items()), sorted(other.items())


def parse_events(specs: Sequence[str]) -> list[tuple[float, str]]:
    """``--event YYYYMMDD[,label]`` -> ``[(yearf, label), …]``.

    The escape hatch for an event that is real but not yet declared in
    ``steps.csv`` — which, while the catalogs are still templates, is nearly
    all of them.
    """
    from gtimes.timefunc import TimetoYearf

    out: list[tuple[float, str]] = []
    for spec in specs or ():
        day, _, label = spec.partition(",")
        day = day.strip()
        if len(day) != 8 or not day.isdigit():
            raise SystemExit(f"--event expects YYYYMMDD[,label], got {spec!r}")
        y, m, d = int(day[:4]), int(day[4:6]), int(day[6:])
        out.append(
            (float(TimetoYearf(y, m, d)), label.strip() or f"{y}-{m:02d}-{d:02d}")
        )
    return sorted(out)


def borrow_record(
    donor: str, target: str, *, params_path: str | Path | None = None
) -> dict[str, Any]:
    """Fetch a donor station's record to apply to ``target`` (UseSTA).

    Records are SELF-CONTAINED, so a donor's parameters evaluate at any
    station's epochs — that is what makes borrowing possible at all.

    This is a **duplicate, not a pointer**, and the distinction matters:
    ``UseSTA`` (``station_detrend_record(use_sta=)``) is a READ-time switch,
    so borrowing at estimate time is not expressible in the schema.  Copying
    the donor's record into the target's slot therefore means it will NOT
    follow the donor if the donor is later re-estimated, and nothing detects
    that staleness.  ``borrowed`` records where it came from and when the
    donor was fitted, so at least the drift is legible.

    The upside of a duplicate: it round-trips.  ``plot-gps-timeseries``
    exposes no ``use_sta``, so a pointer would be invisible to the very path
    that has to consume it.
    """
    from geo_dataread.gps_views import read_detrend_params, station_detrend_record

    doc = read_detrend_params(params_path)
    record, source = station_detrend_record(doc, donor)
    if record is None:
        raise RuntimeError(
            f"donor {donor!r} has no record in the parameter document "
            f"(looked up as {source!r}). Estimate and commit the donor first."
        )
    borrowed = dict(record)
    borrowed["borrowed"] = {
        "from": donor,
        "donor_fitted_at": record.get("fitted_at"),
        "applied_to": target,
    }
    return borrowed


def summarise(record: dict[str, Any], sta: str) -> str:
    """One-screen record summary — the half of the workbench that is text."""
    comps = record.get("components") or []
    lines = [
        f"  station        {sta}",
        f"  model          {record.get('model')}",
        f"  method         {record.get('detrend_method')}",
        f"  stages         {(record.get('refs') or {}).get('outlier_stages')}",
        f"  window         {record.get('window')}",
        f"  span_used      {record.get('span_used')}",
        f"  n_epochs       {record.get('n_epochs')}",
        f"  n_rejected     {record.get('n_rejected')}",
        f"  rms [mm]       {[round(float(v), 2) for v in record.get('rms', [])]}",
        f"  step_epochs    {record.get('step_epochs')}",
        f"  borrowed       {record.get('borrowed')}",
    ]
    names = record.get("param_names") or []
    if "rate" in names:
        i = names.index("rate")
        rates = [round(float(c["params"][i]), 2) for c in comps]
        lines.append(f"  rate [mm/yr]   {rates}")
    steps = [n for n in names if n.startswith("step_amp")]
    for name in steps:
        i = names.index(name)
        amps = [round(float(c["params"][i]), 1) for c in comps]
        lines.append(f"  {name:14s} {amps} mm")
    return "\n".join(lines)


def split_outliers(
    data: Any, sigma: Any, outliers: Any
) -> tuple[Any, tuple[Any, Any] | None]:
    """Split a series into the kept points and a rejected-point overlay.

    The same MASK-not-filter convention as
    :func:`gps_plot.timesmatplt._mask_outliers`: the returned series is
    NaN at the rejected epochs and the overlay is NaN everywhere else, so
    nothing is deleted and the two arrays stay index-aligned with ``x``.

    The mask itself comes from a different place, and that is the point.
    ``timesmatplt`` re-runs ``detect_view_outliers`` over the whole
    series; here the mask is the FIT's own inlier verdict, lifted by
    ``station_estimate_from_arrays``.  Running the view detector instead
    would disagree with the record by construction — it sees neither the
    fit window nor a ``--step`` declared on the command line — and the
    figure would then contradict the ``n_rejected`` printed beside it.

    Returns:
        ``(kept, overlay)``; ``overlay`` is ``(values, sigmas)`` or None
        when nothing was rejected.
    """
    flags = np.asarray(outliers, dtype=bool)
    if not flags.any():
        return data, None
    kept = np.where(flags, np.nan, data)
    overlay = (np.where(flags, data, np.nan), np.where(flags, sigma, np.nan))
    return kept, overlay


#: Marker fill of the OUT-OF-WINDOW outlier lane.  Hollow, sharing the grey
#: edge and errorbar of the fit's own rejections but not their fill, because
#: the two greys are different claims.  Inside the window an epoch was WEIGHED
#: by the fit and rejected, and the count of those marks is the record's
#: ``n_rejected``; outside it the fit passed no verdict at all and the mark is
#: the view detector's opinion about data the record never saw.  One shared
#: marker would break the very invariant :func:`split_outliers` exists to
#: protect — that what the figure greys out equals the ``n_rejected`` printed
#: beside it.
OUTSIDE_FACE_COLOR: str = "none"


def screen_outside_window(
    sta: str,
    yearf: Any,
    data: Any,
    sigma: Any,
    estimate: Any,
    *,
    steps: Sequence[float] | None = None,
    outlier_params: Any = None,
    outlier_overrides: str | None = None,
    provisional_days: float | None = None,
) -> tuple[Any, Any]:
    """View-detector verdicts on the epochs the fit window left unjudged.

    A curated fit window is usually a small pre-unrest slice of a long
    series — RHOF's is 1452 of 4789 epochs — so most of what the operator
    LOOKS at on the detrended page was never weighed by the fit.  Drawn
    plain, those epochs claim "clean" (the flaw :func:`_add_window_edges`
    can only warn about), and a single blunder among them sets the y-axis:
    RHOF north spans 73 mm out-of-window, 27 mm once screened.

    So the fit's silence is filled by the *view* detector — the same
    station-aware chain ``plot-gps-timeseries --view cleaned`` uses, via
    :func:`gps_plot.timesmatplt.view_flags`, restricted to
    ``~estimate.in_window``.  The two lanes are disjoint by construction
    and neither touches the other: inside the window the fit's verdict
    stands alone, so ``n_rejected``, the stored record and everything
    ``--commit`` writes are unchanged by this.  It is a reading aid on
    data the record has no opinion about, not a second opinion on data it
    does.

    Two details make the lanes agree rather than argue:

    - the detector is fed every step DECLARED for the station, not just
      the ones the fit used.  ``record["step_epochs"]`` is the wrong
      source and instructively so: it keeps only epochs inside the window
      (outside it there is no amplitude to estimate), which is exactly the
      set this lane does not care about.  A step in the screened stretch
      is invisible to the record and over-flags around itself at ANY
      threshold — that hazard is why :func:`split_outliers` refuses to
      re-run the view detector *inside* the window.  Pass ``steps`` for a
      CLI ``--step``; ``steps.csv`` is folded in either way;
    - detection still runs over the whole series and only the verdict is
      narrowed, because a detector handed the post-window fragment alone
      would fit it worse.

    Expect these flags to differ from ``plot-gps-timeseries --view
    cleaned`` on the same station: ``uncert`` screens σ at READ time and
    the workbench defaults to 10 against the plot driver's 15, so the two
    tools are not even looking at the same epochs.

    Returns:
        ``(flags, provisional)`` — bool arrays shaped like ``data``, False
        everywhere inside the window — or ``(None, None)`` when the
        window is open (nothing to screen) or the screen failed.  A
        detection abort degrades inside ``geo_dataread`` to all-False
        flags plus a ``UserWarning``; the figure is never lost over it.
    """
    import gps_plot.timesmatplt as tplt

    in_window = np.asarray(estimate.in_window, dtype=bool)
    outside = ~in_window
    if not outside.any():
        return None, None

    declared = _declared_step_epochs(sta, estimate.record.get("step_epochs"), steps)
    try:
        flags, pflags, _aborted = tplt.view_flags(
            sta,
            yearf,
            data,
            sigma,
            outlier_params=outlier_params,
            outlier_overrides=outlier_overrides,
            provisional_days=provisional_days,
            step_epochs=np.asarray(declared, dtype=float),
            restrict=outside,
        )
    except Exception as exc:  # the screen is an aid; never lose the figure
        print(
            f"note: {sta}: out-of-window screen failed "
            f"({type(exc).__name__}: {exc}); those epochs are drawn unjudged.",
            file=sys.stderr,
        )
        return None, None
    return flags, pflags


def _add_window_edges(fig: Figure, record: dict[str, Any], yearf: Any) -> None:
    """Mark the fit window, but only where it actually clips the series.

    Epochs outside the window got NO verdict from this fit — they are
    neither kept nor rejected, they were never judged.  Drawn as plain
    points they would claim "clean", so the boundary has to be visible.
    Royalblue, like the trajectory, because the window is a property of
    the fit and not a fourth data state.  Suppressed when the window is
    open, where two lines at the plot edges would say nothing.

    :func:`screen_outside_window` now fills that silence with the view
    detector's verdicts, but the edges stay just as necessary: they are
    what tells the operator WHICH grey they are looking at, and that the
    fit itself weighed only the epochs between them.
    """
    from gtimes.timefunc import TimefromYearf

    window = record.get("window") or ()
    if len(window) != 2:
        return
    t = np.asarray(yearf, dtype=float)
    for bound in window:
        edge = float(bound)
        if edge <= t.min() + 1e-6 or edge >= t.max() - 1e-6:
            continue  # open bound: the data itself is the boundary
        for ax in fig.axes:
            ax.axvline(
                TimefromYearf(edge),
                color=FIT_COLOR,
                linestyle="--",
                lw=1.0,
                zorder=4,
            )


def render(
    sta: str,
    record: dict[str, Any],
    yearf: Any,
    data: Any,
    sigma: Any,
    out: str | Path,
    *,
    terms: str = "all",
    events: dict[Any, Any] | None = None,
    tos_events: Sequence[tuple[float, str]] | None = None,
    seismic_events: Sequence[tuple[float, str]] | None = None,
    outliers: Any = None,
    outside_outliers: Any = None,
    outside_provisional: Any = None,
    hide_outliers: bool = False,
) -> Path:
    """Two-page PDF: plate frame + fitted trajectory, then the detrended view.

    ``outliers`` is the fit's own (3, N) rejection mask.  Its epochs are
    ALWAYS masked out of the plotted series — they are not in the fit, so
    drawing them as data would misrepresent what the record was built
    from — and by default they are redrawn as the grey overlay of the
    ``plot-gps-timeseries`` cleaned view, using that module's own colour
    constants so the two tools cannot drift apart.  ``hide_outliers``
    drops only the overlay: DISPLAY ONLY, exactly as in ``plotTime``, and
    the y-axis then tightens to the kept series.

    ``outside_outliers`` is the second grey lane
    (:func:`screen_outside_window`): the view detector's verdict on epochs
    the fit window excluded.  Also masked, but for a different reason —
    "not in the fit" cannot justify it, since NO out-of-window epoch is in
    the fit; it is masked because this is a *view* verdict and the cleaned
    view masks what it flags.  Drawn hollow (:data:`OUTSIDE_FACE_COLOR`)
    so the two greys stay countable apart, and suppressed by
    ``hide_outliers`` alongside the first.

    ``outside_provisional`` is gold, and only ever appears outside the
    window.  The fit has no provisional category — every windowed epoch is
    an inlier or not — but the view detector does, and with a pre-unrest
    window the newest epochs are precisely the out-of-window ones.
    Dropping them there would render a recent, genuinely undecided epoch
    as plain red: "clean", the one claim nobody can make about it yet.
    Like ``plotTime``, these stay IN the series and survive
    ``hide_outliers``: hiding a decided outlier declutters, hiding an
    undecided one hides the thing most worth looking at.
    """
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.backends.backend_pdf import PdfPages

    import gps_plot.timesmatplt as tplt
    from gps_analysis import evaluate_record
    from geo_dataread.gps_views import apply_stored_detrend

    x = list(_to_datetime(yearf))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    def _lanes(values: Any) -> tuple[Any, dict[str, tuple[Any, Any] | None]]:
        """Mask every judged epoch out of ``values``, then build the overlays.

        :func:`split_outliers` is called once per lane instead of being
        reimplemented: the combined call produces the series the operator
        reads, and the per-lane calls produce overlays that stay
        separately countable — which is the whole point of having two
        greys.  A lane whose mask is empty comes back None from
        ``split_outliers`` and simply is not drawn.
        """
        lanes: dict[str, tuple[Any, Any] | None] = {}
        masks = [
            np.asarray(m, dtype=bool)
            for m in (outliers, outside_outliers)
            if m is not None
        ]
        kept_ = values
        if masks:
            kept_, _ = split_outliers(values, sigma, np.logical_or.reduce(masks))
        for name, mask in (
            ("fit", outliers),
            ("outside", outside_outliers),
            # provisional epochs are NOT in `masks`: they stay in the series
            ("provisional", outside_provisional),
        ):
            lanes[name] = (
                split_outliers(values, sigma, mask)[1] if mask is not None else None
            )
        return kept_, lanes

    def _draw_lanes(fig: Figure, lanes: dict[str, tuple[Any, Any] | None]) -> None:
        if lanes["fit"] is not None and not hide_outliers:
            tplt.addData(
                x,
                *lanes["fit"],
                fig,
                ecolor=tplt.OUTLIER_ERRORBAR_COLOR,
                markerfacecolor=tplt.OUTLIER_FACE_COLOR,
                markeredgecolor=tplt.OUTLIER_EDGE_COLOR,
            )
        if lanes["outside"] is not None and not hide_outliers:
            tplt.addData(
                x,
                *lanes["outside"],
                fig,
                ecolor=tplt.OUTLIER_ERRORBAR_COLOR,
                markerfacecolor=OUTSIDE_FACE_COLOR,
                markeredgecolor=tplt.OUTLIER_EDGE_COLOR,
            )
        if lanes["provisional"] is not None:
            # larger, to cover the red marker of the point still underneath
            tplt.addData(
                x,
                *lanes["provisional"],
                fig,
                markersize=5.5,
                ecolor=tplt.PROVISIONAL_ERRORBAR_COLOR,
                markerfacecolor=tplt.PROVISIONAL_FACE_COLOR,
                markeredgecolor=tplt.PROVISIONAL_EDGE_COLOR,
            )

    with PdfPages(out) as pdf:
        # page 1 -- observed (plate frame) with the fitted trajectory over it
        fit = np.asarray(evaluate_record(record, yearf, terms=terms))
        kept, lanes = _lanes(data)
        title = tplt.make_title(sta, x[-1], ref="Plate (workbench)")
        fig = tplt.stdTimesPlot(x, kept, sigma, Title=title)
        for c in range(3):
            fig.axes[c].plot(
                x,
                fit[c],
                "-",
                color=FIT_COLOR,
                lw=FIT_WIDTH,
                zorder=5,
                label="stored-record trajectory",
            )
        _draw_lanes(fig, lanes)
        _add_window_edges(fig, record, yearf)
        if events:
            tplt.addEvent(events, fig)
        if tos_events:
            add_event_lines(fig, tos_events, TOS_COLOR)
        if seismic_events:
            add_event_lines(fig, seismic_events, SEISMIC_COLOR)
        pdf.savefig(fig, bbox_inches="tight")

        # page 2 -- the same series with that trajectory removed.  Detrend the
        # FULL series, then mask: apply_stored_detrend is a pure evaluation, so
        # feeding it the NaN-masked series would only propagate the NaNs and
        # lose the overlay values.
        det = np.asarray(
            apply_stored_detrend(
                record, yearf, data, terms=terms, frame="plate_removed"
            )
        )
        kept2, lanes2 = _lanes(det)
        title2 = tplt.make_title(sta, x[-1], ref=f"Detrended (terms={terms})")
        fig2 = tplt.stdTimesPlot(x, kept2, sigma, Title=title2)
        for c in range(3):
            fig2.axes[c].axhline(0.0, color=FIT_COLOR, lw=1.0, zorder=5)
        _draw_lanes(fig2, lanes2)
        _add_window_edges(fig2, record, yearf)
        if events:
            tplt.addEvent(events, fig2)
        if tos_events:
            add_event_lines(fig2, tos_events, TOS_COLOR)
        if seismic_events:
            add_event_lines(fig2, seismic_events, SEISMIC_COLOR)
        pdf.savefig(fig2, bbox_inches="tight")

    return out


def _to_datetime(yearf: Any) -> Any:
    from geo_dataread.gps_read import toDateTime

    return toDateTime(np.asarray(yearf, dtype=float))


def _validate_record(sta: str, record: Any, doc: Any) -> None:
    """Reject a record the apply path would choke on, at commit time.

    Two checks the document-level reader cannot make:

    - the record must actually evaluate — ``trajectory_from_record`` is the
      same constructor the apply path uses, so if it raises here it would
      have raised (or degraded to raw) on every read;
    - the record's ``frame`` must match the document's.  A frame mismatch is
      the one condition design §2.5 says to refuse rather than fudge.
    """
    from gps_analysis import trajectory_from_record

    if not isinstance(record, dict):
        raise RuntimeError(
            f"{sta}: record must be a mapping, got {type(record).__name__}"
        )
    doc_frame = doc.get("frame")
    rec_frame = record.get("frame")
    if doc_frame and rec_frame and rec_frame != doc_frame:
        raise RuntimeError(
            f"{sta}: record frame {rec_frame!r} != document frame {doc_frame!r}. "
            f"Refusing — applying parameters across frames is a hard error "
            f"(design §2.5), and committing it would hide that until read time."
        )
    try:
        trajectory_from_record(record)
    except Exception as exc:
        raise RuntimeError(
            f"{sta}: record is not usable by the apply path "
            f"({type(exc).__name__}: {exc}). Refusing to commit it — it would "
            f"degrade this station to raw at every read."
        ) from None


SCRATCH_FIGDIR: str = "tmp-figdir"
"""Scratch figure dir, shared with ``tools/local-plot/figview.sh``.

That script defaults ``FIGDIR`` to ``<gps_plot>/tmp-figdir`` and injects
``-d "$FIGDIR"`` into ``plot-gps-timeseries``, so figures from the whole local
workflow land in one gitignored place.  The workbench predated it and wrote to
CWD, which litters the repo root on every iteration.
"""


def default_figdir() -> Path:
    """Where an unqualified ``--out`` lands.

    Precedence mirrors ``figview.sh``: ``$FIGDIR`` wins so that script can
    redirect us, then the source checkout's ``tmp-figdir/``, then CWD.

    The checkout is identified by ``pyproject.toml`` beside the package root --
    without that check a wheel install would resolve into ``site-packages``,
    which is the wrong place to write a figure and may not be writable.
    """
    env = os.environ.get("FIGDIR")
    if env:
        return Path(env).expanduser()
    root = Path(__file__).resolve().parents[2]
    if (root / "pyproject.toml").is_file():
        return root / SCRATCH_FIGDIR
    return Path.cwd()


def resolve_out(out: str | None, sta: str) -> Path:
    """Resolve ``--out`` against :func:`default_figdir`.

    A bare filename (``--out SELF-iter1.pdf``) is placed IN the figdir; anything
    carrying a path separator or ``~`` is honoured verbatim, so an explicit
    location is never silently relocated.
    """
    if out is None:
        return default_figdir() / f"{sta}-detrend-workbench.pdf"
    if os.sep in out or out.startswith("~") or (os.altsep and os.altsep in out):
        return Path(out).expanduser()
    return default_figdir() / out


def commit_record(
    sta: str,
    record: dict[str, Any],
    *,
    params_path: str | Path | None = None,
    init: bool = False,
    force: bool = False,
) -> tuple[Path, int, int]:
    """Merge one station's record into ``detrend_params.json``.

    READ-MODIFY-WRITE, never build-from-scratch.  ``gps-estimate-detrend``
    assembles a fresh document with ``build_document`` because it owns every
    station in one run; doing that here would silently drop the other 32.

    ``schema_version`` / ``frame`` / ``units`` / ``phase_convention`` are
    carried over VERBATIM.  A changed ``schema_version`` makes
    :func:`gps_views.read_detrend_params` reject the document outright, which
    degrades the ENTIRE fleet to raw — a one-station edit must not be able to
    do that.

    Refuses to CREATE a document unless ``init``.  This is the sharp edge:
    ``default_params_path()`` resolves to ``~/.config/gpsconfig/
    detrend_params.json``, which does not exist on this machine — the live
    33-station document is reached via ``GPS_CONFIG_PATH``.  A naive commit
    would therefore create a brand-new ONE-station document at the default
    path, and every consumer would silently start reading that instead.

    Returns:
        ``(path, n_before, n_after)``.
    """
    import json
    import os
    import tempfile

    from geo_dataread.gps_views import default_params_path, read_detrend_params

    path = Path(params_path) if params_path else default_params_path()
    if path is None:
        raise RuntimeError(
            "no gpsconfig available to resolve detrend_params.json; pass "
            "--params PATH or set GPS_CONFIG_PATH"
        )
    # Follow symlinks BEFORE writing. The deployed config tree is a symlink
    # farm (~/gps-data/testcfg links into ~/.config/gpsconfig), and os.replace
    # on a symlink replaces the LINK with a regular file -- the two trees then
    # diverge silently and permanently. Demonstrated by review, not theorised.
    if path.is_symlink() or path.exists():
        path = path.resolve()

    if path.is_file():
        doc = read_detrend_params(path)
    elif init:
        from geo_dataread.detrend_estimate import build_document

        doc = build_document({})
    else:
        raise RuntimeError(
            f"{path} does not exist. Refusing to create it: a new one-station "
            f"document at this path would become what every consumer reads, "
            f"silently replacing the real one. Point --params at the live "
            f"document (or set GPS_CONFIG_PATH), or pass --init if you really "
            f"do mean to start a new one."
        )

    # Validate BEFORE writing, where the operator is looking. read_detrend_params
    # only checks the document envelope, so a malformed record commits happily
    # and then degrades that station to raw at every read, with a warning
    # nobody is watching for.
    _validate_record(sta, record, doc)

    stations = doc.setdefault("stations", {})
    n_before = len(stations)
    if sta in stations and not force:
        raise RuntimeError(
            f"{sta} already has a record in {path} "
            f"(fitted_at={stations[sta].get('fitted_at')}). Pass --force to "
            f"replace it."
        )
    stations[sta] = record
    n_after = len(stations)

    # atomic: write beside the target, then rename. No helper exists anywhere
    # in the ecosystem, and a half-written parameter document is worse than a
    # stale one -- read_detrend_params would reject it and the fleet would
    # degrade to raw.
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, indent=2, sort_keys=False)
            fh.write("\n")
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return path, n_before, n_after


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="gps-detrend-workbench",
        description="Operator workbench for detrending: one station, one PDF, "
        "two views (plate frame + fitted trajectory, and the detrended "
        "series). Estimation reuses gps-estimate-detrend's own code path, so "
        "a record made here means the same thing as a batch-made one.",
        epilog="Detection runs the FULL pipeline by default, falling back to "
        "S0-only (loudly) if the excess-candidate rule aborts. An S0 record "
        "leaves model-visible outliers in the fit, so its parameters are "
        "biased relative to a full-detection one — on a fallback, prefer "
        "declaring the missing step with --step over accepting the S0 record.",
    )
    p.add_argument("station", help="four-letter station code")
    p.add_argument("--tot-dir", default=None, help="TOT directory (default: config)")
    p.add_argument("--uncert", type=int, default=10, help="sigma screen [mm]")
    p.add_argument(
        "--out",
        default=None,
        metavar="PDF",
        help="output PDF. A bare filename lands in the scratch figdir "
        "($FIGDIR, else <gps_plot>/tmp-figdir, else CWD); a path with a "
        "separator is used verbatim. Default: "
        "<figdir>/<STA>-detrend-workbench.pdf",
    )
    p.add_argument(
        "--model",
        default=None,
        choices=("linear", "periodic", "lineperiodic"),
        help="FIT-time model. Distinct from --terms: this is stored "
        "in the record and decides what was estimated",
    )
    p.add_argument(
        "--terms",
        default=APPLY_TERMS_DEFAULT,
        choices=("all", "secular", "periodic"),
        help="APPLY-time term selection for the figures. Distinct "
        "from --model: NOT stored, chosen per call — so anything other "
        f"than {APPLY_TERMS_DEFAULT!r} is refused with --commit",
    )
    p.add_argument(
        "--hide-outliers",
        action="store_true",
        help="drop the grey overlay of the epochs the fit rejected. They are "
        "masked out of the plotted series either way — this decides only "
        "whether the figure still SHOWS them, and hiding them lets the "
        "y-axis tighten to the fitted series. Same name and meaning as "
        "plot-gps-timeseries --hide-outliers",
    )
    p.add_argument(
        "--event",
        action="append",
        default=[],
        metavar="YYYYMMDD[,LABEL]",
        help="draw a seismic/other event line (repeatable). For events not "
        "yet declared in steps.csv — which, while the catalogs are "
        "templates, is nearly all of them",
    )
    p.add_argument(
        "--no-tos",
        dest="tos",
        action="store_false",
        help="skip the TOS equipment-change lookup. On by default; any TOS "
        "failure already degrades to a warning, so the workbench stays "
        "usable off-VPN either way",
    )
    p.add_argument(
        "--tos-url",
        default=None,
        metavar="URL",
        help="TOS REST endpoint (default: tostools' own)",
    )
    p.add_argument(
        "--donor",
        default=None,
        metavar="STA",
        help="apply another station's stored record instead of estimating "
        "one (UseSTA). A DUPLICATE, not a pointer: it will not follow "
        "the donor if the donor is re-estimated. Both RMS figures are "
        "printed so the cost of borrowing is visible",
    )
    p.add_argument(
        "--stages",
        default=None,
        metavar="LIST",
        help="force a detection stage set (e.g. S0, or S0,S4). Default is the "
        "FULL pipeline, falling back to S0 only if it aborts — an S0 "
        "record leaves model-visible outliers in the fit and its "
        "parameters are biased relative to a full-detection one",
    )
    p.add_argument(
        "--window-start",
        type=float,
        default=None,
        metavar="YEARF",
        help="fit-window start [fractional year]",
    )
    p.add_argument(
        "--window-end",
        type=float,
        default=None,
        metavar="YEARF",
        help="fit-window end [fractional year]",
    )
    p.add_argument(
        "--step",
        action="append",
        type=float,
        default=[],
        metavar="YEARF",
        help="declare an offset epoch (repeatable). DECLARE-AND-FIT: the "
        "epoch is yours, the amplitude is estimated and printed with "
        "the record. No epoch detection happens here",
    )
    p.add_argument(
        "--max-gap-years",
        type=float,
        default=None,
        help="validity gate: largest tolerated data gap. The 0.5 default "
        "rejects most long-history stations (RHOF 1.35, VMEY 1.21, "
        "HOFN 1.08, SELF 0.73 yr), so this is usually required rather "
        "than optional",
    )
    p.add_argument(
        "--min-epochs",
        type=int,
        default=None,
        help="validity gate: minimum epochs in the window",
    )
    p.add_argument(
        "--min-span-years",
        type=float,
        default=None,
        help="validity gate: minimum window span [yr]",
    )
    p.add_argument(
        "--commit",
        action="store_true",
        help="merge this record into detrend_params.json. WRITES CONFIG "
        "outside the repos; the resolved path is printed before writing",
    )
    p.add_argument(
        "--params",
        default=None,
        metavar="JSON",
        help="explicit detrend_params.json (default: the gpsconfig resolver)",
    )
    p.add_argument(
        "--init",
        action="store_true",
        help="allow --commit to CREATE the document. Off by default because "
        "a new one-station file at the default path would become what "
        "every consumer reads",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="allow --commit to replace an existing record for this station",
    )
    p.add_argument(
        "--fit-catalog",
        default=None,
        metavar="CSV",
        help="fit_windows.csv override (default: deployed)",
    )
    p.add_argument(
        "--outlier-param",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="repeatable OutlierParams override, applied on top of the stage set in force",
    )
    p.add_argument(
        "--no-screen-outside-window",
        dest="screen_outside_window",
        action="store_false",
        help="stop screening the epochs OUTSIDE the fit window. On by "
        "default: the fit judges only its window (RHOF: 1452 of 4789 "
        "epochs), so without this the rest is drawn as if clean and one "
        "blunder sets the y-axis. Those flags come from the view "
        "detector, are drawn HOLLOW grey to stay countable apart from "
        "the fit's own solid grey, and change NOTHING about the record",
    )
    p.add_argument(
        "--provisional-days",
        type=float,
        default=None,
        metavar="DAYS",
        help="recency bound [days] of the PROVISIONAL marker (gold), same "
        "meaning as in plot-gps-timeseries. The bound is what makes the "
        "marker useful here: a pre-unrest window leaves a decade of "
        "screened epochs, and indeterminate clusters also sit at old "
        "mid-series gaps, which would otherwise dominate the lane. 0 "
        "disables it; unset uses the geo_dataread default (14). Only the "
        "out-of-window lane has this state — a fit has no provisional "
        "category",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    sta = args.station.upper()

    if args.commit and args.terms != APPLY_TERMS_DEFAULT:
        # Refused BEFORE any work: the figure you would judge is not the
        # series production would render. `terms` is an apply-time argument
        # that the record has no field for, so committing under a non-default
        # one stores a record whose figure nobody can reproduce -- and the
        # divergence is not cosmetic (measured 16.48 mm max on RHOF between
        # terms="all" and terms="secular" on the same record).
        print(
            f"error: --terms {args.terms} cannot be committed. It is an "
            f"APPLY-time choice, evaluated per call and NOT stored, so the "
            f"record would be read back with terms={APPLY_TERMS_DEFAULT!r} "
            f"and render a different series than the one you judged. To make "
            f"that decision part of the record, fit it: --model periodic (or "
            f"linear) is stored and does round-trip. To keep looking under "
            f"--terms {args.terms}, drop --commit.",
            file=sys.stderr,
        )
        return 5

    out = resolve_out(args.out, sta)

    try:
        record, yearf, data, sigma, estimate = build_record(
            sta,
            tot_dir=args.tot_dir,
            uncert=args.uncert,
            outlier_param=args.outlier_param,
            fit_catalog=args.fit_catalog,
            model=args.model,
            stages=args.stages,
            window=(
                (args.window_start, args.window_end)
                if (args.window_start or args.window_end)
                else None
            ),
            steps=args.step or None,
            max_gap_years=args.max_gap_years,
            min_epochs=args.min_epochs,
            min_span_years=args.min_span_years,
        )
    except RuntimeError as exc:
        # A refused record is a RESULT, not a crash -- report it plainly and
        # exit non-zero so a loop cannot mistake it for success.
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.donor:
        try:
            borrowed = borrow_record(args.donor, sta, params_path=args.params)
        except RuntimeError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 4
        own_rms = [round(float(v), 2) for v in record.get("rms", [])]
        record = borrowed
        print(f"\n{sta} — BORROWED record from {args.donor}\n")
        print(summarise(record, sta))
        print(f"  own-fit rms    {own_rms}   <- the cost of borrowing")
        print(
            f"  note: the grey epochs are {sta}'s OWN detection verdict; "
            f"{args.donor} contributes only the trajectory."
        )
    else:
        print(f"\n{sta} — detrend record\n")
        print(summarise(record, sta))
    seismic, declared_other = declared_event_epochs(sta)
    seismic = sorted(seismic + parse_events(args.event))
    tos_events: list[tuple[float, str]] = []
    if args.tos:
        try:
            tos_events = tos_equipment_epochs(sta, url=args.tos_url)
        except RuntimeError as exc:
            # never fatal: an operator off-VPN still needs the workbench
            print(
                f"warning: {exc}; continuing without equipment lines", file=sys.stderr
            )
    if tos_events:
        print(f"\n  TOS equipment changes ({len(tos_events)}):")
        for epoch, label in tos_events:
            print(f"    {label}   yearf {epoch:.4f}")

    if seismic or declared_other:
        print(f"\n  declared / supplied events ({len(seismic) + len(declared_other)}):")
        for epoch, label in seismic:
            print(f"    {epoch:9.4f}  seismic   {label}")
        for epoch, label in declared_other:
            print(f"    {epoch:9.4f}  other     {label}")

    outside = outside_prov = None
    if args.screen_outside_window and args.donor:
        # The two annotations would contradict each other: _add_window_edges
        # draws the DONOR's window (it comes off the record) while in_window is
        # the own fit's, so under --donor the dashes would no longer delimit
        # the hollow grey -- and the dashes are what says which grey is which.
        print(
            "  note: out-of-window screen skipped under --donor; the dashed "
            "edges are the donor's window while the unjudged epochs are this "
            "station's own, so the two would not line up."
        )
    elif args.screen_outside_window:
        outside, outside_prov = screen_outside_window(
            sta,
            yearf,
            data,
            sigma,
            estimate,
            steps=args.step or None,
            outlier_params=_stage_params(args.stages, args.outlier_param),
            provisional_days=args.provisional_days,
        )

    path = render(
        sta,
        record,
        yearf,
        data,
        sigma,
        out,
        terms=args.terms,
        tos_events=list(tos_events) + list(declared_other),
        seismic_events=seismic,
        outliers=estimate.outliers,
        outside_outliers=outside,
        outside_provisional=outside_prov,
        hide_outliers=args.hide_outliers,
    )
    n_out = [int(v) for v in np.asarray(estimate.outliers).sum(axis=1)]
    n_unjudged = int(np.count_nonzero(~np.asarray(estimate.in_window)))
    state = "hidden" if args.hide_outliers else "grey"
    print(f"\n  rejected by the fit  {n_out} ({state}; masked from the series)")
    if n_unjudged:
        print(
            f"  outside the fit window {n_unjudged} epoch(s) — the fit passed "
            f"no verdict on them (window edges dashed)"
        )
        if outside is None:
            print("    not screened — those epochs are drawn unjudged")
        else:
            # A SECOND count, never folded into n_out: the number above is the
            # record's own n_rejected, and a reader has to be able to match the
            # figure back to it.
            n_screened = [int(v) for v in np.asarray(outside).sum(axis=1)]
            print(
                f"    flagged by the view detector {n_screened} "
                f"({'hidden' if args.hide_outliers else 'hollow grey'}; "
                f"masked, but NOT part of the record's n_rejected)"
            )
            n_prov = [int(v) for v in np.asarray(outside_prov).sum(axis=1)]
            if any(n_prov):
                print(
                    f"    provisional {n_prov} (gold; kept in the series — "
                    f"too recent for the detector to rule on)"
                )
    print(f"\nwrote {path}")

    if args.commit:
        record.setdefault("refs", {})["generator"] = "gps-detrend-workbench"
        try:
            target, before, after = commit_record(
                sta,
                record,
                params_path=args.params,
                init=args.init,
                force=args.force,
            )
        except RuntimeError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 3
        print(f"committed {sta} -> {target}   stations {before} -> {after}")
        if args.uncert != BATCH_UNCERT_DEFAULT:
            # `uncert` screens sigma at read time, so it changes WHICH epochs
            # were fitted without appearing in any fitted quantity. The record
            # carries it in refs; this line is what makes it actionable, since
            # a batch re-run at the default would silently replace this record
            # with one fitted on a different set of epochs.
            print(
                f"  note: fitted with --uncert {args.uncert}; "
                f"gps-estimate-detrend defaults to {BATCH_UNCERT_DEFAULT}. "
                f"Re-run it as `gps-estimate-detrend {sta} --uncert "
                f"{args.uncert}` to reproduce this record in batch."
            )
    else:
        print("(not committed — pass --commit to store this record)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
