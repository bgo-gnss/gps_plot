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
- **S0-only detection by default.**  §14 stage isolation: S0 works on local
  first differences with no global fit, so its verdict does not depend on
  the window's composition.  That matters here because a station with an
  undeclared step (SELF: a 143 mm 2008 coseismic) drives the full pipeline's
  candidate fraction past ``max_flag_fraction`` and aborts — the very
  stations that most need curation are the ones the full detector refuses.
  The stage set is built by reusing ``plot_gps_timeseries._stage_overrides``,
  so no detection default is restated here.
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
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["build_record", "render", "main"]

#: Colour of the fitted-trajectory overlay on the plate-frame panel.  Blue
#: against the red/grey/gold marker vocabulary of the cleaned view, so the
#: model never reads as a data state.
FIT_COLOR: str = "royalblue"
FIT_WIDTH: float = 1.8


def _s0_params(extra: list[str] | None = None) -> Any:
    """S0-only ``OutlierParams``, built from the CLI's own stage mapping.

    Reuses :func:`plot_gps_timeseries._stage_overrides` /
    :func:`_build_outlier_params` rather than restating which fields the S0
    stage corresponds to — one definition of "S0", in one place.
    """
    from gps_plot.plot_gps_timeseries import _build_outlier_params, _stage_overrides

    return _build_outlier_params(extra or [], base=_stage_overrides("S0"))


def _override_settings(settings: Any, **over: Any) -> Any:
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
        changed["steps"] = tuple(float(v) for v in changed["steps"])
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
    window: tuple[float | None, float | None] | None = None,
    steps: Sequence[float] | None = None,
    max_gap_years: float | None = None,
    min_epochs: int | None = None,
    min_span_years: float | None = None,
) -> tuple[dict[str, Any], Any, Any, Any]:
    """Read the plate-frame series and estimate one station's record.

    Returns:
        ``(record, yearf, data, sigma)``.

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
        station_record_from_arrays,
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
    try:
        record = station_record_from_arrays(
            sta,
            yearf,
            data,
            sigma,
            settings=settings,
            outlier_params=_s0_params(outlier_param),
            **kwargs,
        )
    except ValueError as exc:
        # The leaf RAISES on a failed validity gate and RETURNS None on an
        # outlier abort -- two different refusals for the same operator
        # question ("why is there no record?"). Normalise them here so the
        # workbench answers that question the same way either time.
        raise RuntimeError(f"{sta}: {exc} ({gates})") from None
    if record is None:
        raise RuntimeError(
            f"{sta}: outlier stage aborted — no record stored ({gates}). "
            f"S0-only detection is already the default here; if this still "
            f"aborts the series needs a narrower window or a declared step."
        )
    return record, yearf, data, sigma


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
) -> Path:
    """Two-page PDF: plate frame + fitted trajectory, then the detrended view."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.backends.backend_pdf import PdfPages

    import gps_plot.timesmatplt as tplt
    from gps_analysis import evaluate_record
    from geo_dataread.gps_views import apply_stored_detrend

    x = list(_to_datetime(yearf))
    out = Path(out)

    with PdfPages(out) as pdf:
        # page 1 -- observed (plate frame) with the fitted trajectory over it
        fit = np.asarray(evaluate_record(record, yearf, terms=terms))
        title = tplt.make_title(sta, x[-1], ref="Plate (workbench)")
        fig = tplt.stdTimesPlot(x, data, sigma, Title=title)
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
        if events:
            tplt.addEvent(events, fig)
        pdf.savefig(fig, bbox_inches="tight")

        # page 2 -- the same series with that trajectory removed
        det = np.asarray(
            apply_stored_detrend(
                record, yearf, data, terms=terms, frame="plate_removed"
            )
        )
        title2 = tplt.make_title(sta, x[-1], ref=f"Detrended (terms={terms})")
        fig2 = tplt.stdTimesPlot(x, det, sigma, Title=title2)
        for c in range(3):
            fig2.axes[c].axhline(0.0, color=FIT_COLOR, lw=1.0, zorder=5)
        if events:
            tplt.addEvent(events, fig2)
        pdf.savefig(fig2, bbox_inches="tight")

    return out


def _to_datetime(yearf: Any) -> Any:
    from geo_dataread.gps_read import toDateTime

    return toDateTime(np.asarray(yearf, dtype=float))


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
        epilog="Detection defaults to S0 ONLY (§14): S0 works on local first "
        "differences with no global fit, so unlike the full pipeline its "
        "verdict does not depend on the window's composition — which is what "
        "lets a station with an undeclared step be estimated at all.",
    )
    p.add_argument("station", help="four-letter station code")
    p.add_argument("--tot-dir", default=None, help="TOT directory (default: config)")
    p.add_argument("--uncert", type=int, default=10, help="sigma screen [mm]")
    p.add_argument(
        "--out",
        default=None,
        metavar="PDF",
        help="output PDF (default: <STA>-detrend-workbench.pdf in CWD)",
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
        default="all",
        choices=("all", "secular", "periodic"),
        help="APPLY-time term selection for the figures. Distinct "
        "from --model: NOT stored, chosen per call",
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
        help="repeatable OutlierParams override, applied on top of the S0 stage set",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    sta = args.station.upper()
    out = Path(args.out) if args.out else Path(f"{sta}-detrend-workbench.pdf")

    try:
        record, yearf, data, sigma = build_record(
            sta,
            tot_dir=args.tot_dir,
            uncert=args.uncert,
            outlier_param=args.outlier_param,
            fit_catalog=args.fit_catalog,
            model=args.model,
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

    print(f"\n{sta} — detrend record\n")
    print(summarise(record, sta))
    path = render(sta, record, yearf, data, sigma, out, terms=args.terms)
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
    else:
        print("(not committed — pass --commit to store this record)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
