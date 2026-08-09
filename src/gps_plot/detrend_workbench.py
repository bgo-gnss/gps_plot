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
import textwrap
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
from matplotlib.figure import Figure

__all__ = [
    "estimate_record",
    "build_record",
    "borrow_record",
    "commit_record",
    "resolve_devices",
    "tos_equipment_epochs",
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


def _resolve_cli_segments(
    args: Any,
) -> tuple[tuple[float | None, float | None], ...] | None:
    """``--segment`` else ``--window-start/--window-end``, never both.

    Returns None when the operator asked for neither, so the station's
    catalog row (else the open default) still decides — the same
    "unset defers" precedence every other workbench lever follows.

    Raises:
        SystemExit: When both spellings are given.  They are two ways to
            say which epochs are fitted, and letting one silently win
            would change stored science without saying so.
    """
    windowed = args.window_start is not None or args.window_end is not None
    if args.segment and windowed:
        raise SystemExit(
            "--segment cannot be combined with --window-start/--window-end; "
            "--segment is the general form (pass one to get a single window)"
        )
    if args.segment:
        out: list[tuple[float | None, float | None]] = []
        for spec in args.segment:
            if spec.count(":") != 1:
                raise SystemExit(
                    f"--segment expects START:END (either side may be empty "
                    f"for an open bound), got {spec!r}"
                )
            lo, _, hi = spec.partition(":")
            try:
                out.append(
                    (
                        float(lo) if lo.strip() else None,
                        float(hi) if hi.strip() else None,
                    )
                )
            except ValueError:
                raise SystemExit(
                    f"--segment {spec!r}: bounds must be fractional years"
                ) from None
        return tuple(out)
    if windowed:
        return ((args.window_start, args.window_end),)
    return None


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
    if "segments" in changed:
        changed["segments"] = tuple(tuple(s) for s in changed["segments"])
    prior = settings.window_source or "defaults"
    changed["window_source"] = f"workbench-cli(+{prior})"
    return dataclasses.replace(settings, **changed)


def estimate_record(
    sta: str,
    yearf: Any,
    data: Any,
    sigma: Any,
    *,
    settings: Any,
    outlier_params: Any = None,
    stage_plan: object | None = None,
    lookup_donor: object | None = None,
    terms: Sequence[str] | None = None,
    model: str | None = None,
    refs: Mapping[str, Any] | None = None,
) -> Any:
    """Estimate one station's detrend record — gps_plot's single call site.

    Thin wrapper over
    :func:`geo_dataread.detrend_estimate.station_estimate_from_arrays` that
    every consumer in this package routes through, so the workbench
    (:func:`build_record`) and the Qt picker (``PickerWindow.refit``) cannot
    drift apart in which kwargs they forward.  The picker passes a SUBSET
    (no ``outlier_params`` / ``refs`` — it does not commit, so the
    round-trip-fidelity concerns that force the workbench to carry
    ``uncert`` in ``refs`` do not apply to it); the workbench passes the
    full set.  That asymmetry is now visible at the two call sites rather
    than buried in two independent ``station_estimate_from_arrays(...)``
    blocks that could (and did) diverge.

    This is also the seam at which a second estimator will enter.  The
    leaf already carries a method tag —
    :attr:`gps_analysis.DetrendEstimate.detrend_method`, serialized into
    every stored record as ``detrend_method`` — so "which algorithm
    produced this" is already provenance-tracked fleet-wide.  Today that
    vocabulary distinguishes OUTLIER-HANDLING strategy
    (:data:`gps_analysis.detrend.DETREND_METHOD_ROBUST` vs
    :data:`gps_analysis.detrend.DETREND_METHOD_PLAIN`); the estimator
    machinery itself is a single WLS closed-form.  When a second estimator
    is wanted, the house pattern (mirroring :mod:`gps_analysis.velocity`'s
    ``method`` tag) is a sibling function returning
    :class:`~gps_analysis.DetrendEstimate` under an extended tag, and THIS
    function gains a ``method=`` selector that dispatches to it — not a
    Protocol, which the estimator lane deliberately does not use (the
    ``Term`` Protocol is for the additive-math-unit abstraction, a
    different axis).

    Returns:
        The :class:`~geo_dataread.detrend_estimate.StationEstimate`, or
        ``None`` on an outlier-stage abort (the leaf's convention for
        "refused to store a record"; a validity gate instead RAISES,
        which :func:`build_record` normalises to ``RuntimeError``).
    """
    from geo_dataread.detrend_estimate import station_estimate_from_arrays

    extra: dict[str, Any] = {}
    if model is not None:
        extra["model"] = model
    if refs is not None:
        extra["refs"] = refs
    return station_estimate_from_arrays(
        sta,
        yearf,
        data,
        sigma,
        settings=settings,
        outlier_params=outlier_params,
        stage_plan=stage_plan,
        lookup_donor=lookup_donor,
        terms=terms,
        **extra,
    )


def build_record(
    sta: str,
    *,
    tot_dir: str | None = None,
    uncert: int = 10,
    outlier_param: list[str] | None = None,
    fit_catalog: str | Path | None = None,
    model: str | None = None,
    stages: str | None = None,
    stage_plan: object | None = None,
    lookup_donor: object | None = None,
    terms_spec: Sequence[str] | None = None,
    segments: Sequence[tuple[float | None, float | None]] | None = None,
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
        segments=segments,
        steps=steps,
        max_gap_years=max_gap_years,
        min_epochs=min_epochs,
        min_span_years=min_span_years,
    )
    # `uncert` screens sigma at READ time (getData above), so the estimator
    # never sees it and a stored record would be silent about it -- yet the
    # workbench default (10) differs from the batch estimator's (15), which
    # made workbench and batch records indistinguishable but unequal.
    kwargs: dict[str, Any] = {"refs": {"uncert": uncert}}
    if model:
        kwargs["model"] = model

    gates = (
        f"settings from {settings.window_source}; "
        f"segments={settings.segments}, "
        f"max_gap_years={settings.max_gap_years}, "
        f"min_epochs={settings.min_epochs}, "
        f"min_span_years={settings.min_span_years}"
    )

    def _estimate(params: Any) -> Any:
        return estimate_record(
            sta,
            yearf,
            data,
            sigma,
            settings=settings,
            outlier_params=params,
            stage_plan=stage_plan,
            lookup_donor=lookup_donor,
            terms=terms_spec,
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

#: Shortest join that counts as STATION equipment [days].
#:
#: A campaign measurement is registered exactly like a permanent install —
#: same subtype, same attributes — and only its duration tells them apart.
#: SELF carries three: 2001-07-01, 2001-07-16 and 2001-09-14, all antenna +
#: receiver, all 3-4 days, all before the station's real life began on
#: 2002-02-05.  They put four green lines into a two-year span where the
#: operator expects two, and their labels overlap into an unreadable smear
#: at any full-series zoom.
#:
#: 30 days is not tuned, it is a gap: the working set has 3-4 day campaigns
#: and 3040-day-or-open installs, with nothing between, so any threshold in
#: that range separates them identically.  A join still OPEN always counts
#: however young it is — the alternative is a fresh install that draws no
#: line until a month has passed.
#:
#: The cost is real and worth stating: a receiver that genuinely failed and
#: was replaced inside a month draws nothing, and that IS a coordinate
#: event.  ``--event YYYYMMDD`` is the escape hatch when it happens.
MIN_DEPLOYMENT_DAYS: int = 30

#: The ONLY device subtypes that earn a line, and the label each gets.
#:
#: A green line is a claim that the antenna's phase centre may have moved.
#: A station's ``children_connections`` carries far more than that: RHOF's
#: ten joins include ``monument``, ``radome``, ``sim_card`` and
#: ``modem_gsm``, and its 2023-08-16 line was a GSM modem plus a SIM card —
#: a telecoms visit that cannot displace the antenna by a micron, drawn
#: across all three components as if it could.
#:
#: Checked against the canonical list (153 subtypes, TOS
#: ``/entity_subtypes/``): ``antenna`` and ``gnss_receiver`` are the only
#: codes that can be one; ``gps_clock`` is timing, ``radome`` does not move
#: the phase centre on its own.  A WHITELIST fails silent by nature — a
#: receiver registered under some future variant code would draw no line at
#: all — so the constant is the place to look when a known swap is missing.
#:
#: ``monument`` is deliberately absent per the operator rule, and it is the
#: one excluded subtype that physically could matter (replacing a monument
#: moves the antenna).  On the working set every monument join shares its
#: day with an antenna or receiver join, so nothing is lost today; an
#: unexplained step at a monument-only date is the case to remember.
EQUIPMENT_SUBTYPES: dict[str, str] = {
    "antenna": "antenna",
    "gnss_receiver": "receiver",
}


class DeviceInfo(NamedTuple):
    """What a green line needs to know about one joined device."""

    subtype: str
    model: str | None = None
    serial: str | None = None

    def describe(self) -> str:
        """Label fragment: the MODEL, because that is what an operator reads.

        Serial is deliberately not in the label — two Trimble 4000SSIs are
        the same instrument to a reader scanning a figure, and the string is
        already drawn rotated against a crowded axis.  It stays on the
        record for :func:`is_same_unit`, which is where identity matters.
        """
        return self.model or EQUIPMENT_SUBTYPES.get(self.subtype, self.subtype)


def _span_days(start: str, until: str) -> int | None:
    """How long a join lasted, or None while it is still open.

    Open is NOT "zero days": an install registered yesterday has no end and
    must count, or the newest equipment on every station would be invisible.
    """
    if not until:
        return None
    from datetime import date

    try:
        a = date(*(int(v) for v in start.split("-")))  # type: ignore[arg-type]
        b = date(*(int(v) for v in until.split("-")))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None  # unparseable end -> treat as open, never as instant
    return (b - a).days


def is_same_unit(a: DeviceInfo, b: DeviceInfo) -> bool:
    """Whether two joins are the same physical instrument.

    Model AND serial, not the entity id: TOS can hold one unit under more
    than one id, and a serial alone repeats across manufacturers.
    """
    return (a.subtype, a.model, a.serial) == (b.subtype, b.model, b.serial)


#: Resolved ``id_entity`` -> :class:`DeviceInfo`, for the life of the
#: process.  A device's identity is immutable, so this can never go stale
#: within a run.  Measured cost without it: 28-34 ms per device, 0.34 s for
#: RHOF's ten and 0.48 s for SELF's seventeen — small enough that nothing is
#: persisted to disk, which would only add a staleness question with no payoff.
_DEVICE_CACHE: dict[int, DeviceInfo] = {}

#: Attribute codes carrying the model name, most specific first.  TOS is not
#: uniform about this across subtypes, so the first hit wins rather than one
#: code being assumed.
_MODEL_CODES: tuple[str, ...] = ("model", "antenna_type", "receiver_type")


def resolve_devices(
    ids: Sequence[int], *, url: str | None = None
) -> dict[int, DeviceInfo]:
    """Look up subtype, model and serial for each device id.

    ``children_connections`` carries only ``id_entity_child`` — no type, no
    model, no serial — and the station payload has no ``children`` block at
    all, so naming what changed takes one entity call per device.  That is
    the whole reason the epoch labels once said "2 devices", then
    "(antenna, receiver)", instead of which antenna and which receiver.

    Args:
        ids: device ``id_entity`` values; duplicates are fine.
        url: TOS REST endpoint; None uses ``tostools``' own default.

    Returns:
        ``{id: DeviceInfo}``, omitting ids that could not be resolved.  The
        caller decides what an omission means — under
        :data:`EQUIPMENT_SUBTYPES` it means "not whitelisted", i.e. no
        line, because a line we cannot justify is worse than a missing one.

    Raises:
        RuntimeError: When the client cannot be built at all.  A blanket
            failure must NOT read as "this station never had a swap", so
            :func:`tos_equipment_epochs` turns a total loss into the
            caller's existing warning rather than a silently bare figure.
    """
    wanted = {int(i) for i in ids if i is not None}
    missing = sorted(wanted - _DEVICE_CACHE.keys())
    if missing:
        try:
            from tostools.api.tos_client import TOSClient

            client = TOSClient() if url is None else TOSClient(url)
        except (Exception, SystemExit) as exc:
            raise RuntimeError(
                f"TOS device lookup unavailable ({type(exc).__name__}: {exc})"
            ) from None
        for dev in missing:
            try:
                history = client.get_entity_history(dev)
            except (Exception, SystemExit):
                continue  # unresolved -> not whitelisted, counted by the caller
            subtype = (history or {}).get("code_entity_subtype")
            if not subtype:
                continue
            attrs = {}
            for attr in (history or {}).get("attributes") or []:
                code = attr.get("code")
                # attributes are a HISTORY: the same code repeats with
                # different validity, and the first row is the one TOS
                # returns as current. Later rows must not overwrite it.
                if code and code not in attrs:
                    attrs[code] = attr.get("value")
            model = next((attrs[c] for c in _MODEL_CODES if attrs.get(c)), None)
            _DEVICE_CACHE[dev] = DeviceInfo(
                subtype=str(subtype),
                model=str(model) if model else None,
                serial=str(attrs["serial_number"])
                if attrs.get("serial_number")
                else None,
            )
    return {i: _DEVICE_CACHE[i] for i in wanted if i in _DEVICE_CACHE}


def tos_equipment_epochs(
    sta: str,
    *,
    url: str | None = None,
    payload: Any = None,
    devices: Mapping[int, DeviceInfo] | None = None,
) -> list[tuple[float, str]]:
    """Epochs where a new antenna or receiver was INSTALLED, coalesced.

    A green line is a claim that the antenna's phase centre may have moved,
    so three filters stand between a TOS row and a line:

    - **subtype.**  Only :data:`EQUIPMENT_SUBTYPES`.  A station's
      ``children_connections`` is one row per DEVICE join of ANY kind, and
      most kinds cannot displace anything: RHOF's 2023-08-16 line was a GSM
      modem plus a SIM card, drawn across all three components as if a
      telecoms visit were a coordinate event.  Resolving the subtype takes
      one entity call per device (:func:`resolve_device_subtypes`) — the
      join row carries only ``id_entity_child``, which is why the label used
      to say "2 devices" rather than what they were.
    - **installations only.**  ``time_to`` is read by nothing here.  A
      removal is not a line: the operator rule is new equipment, and a swap
      already announces itself through the incoming device's ``time_from``.
    - **actually NEW.**  A join whose instrument is the same model AND
      serial as the one it directly continues is a re-registration, not an
      installation.  SELF proves the case: antenna TRM29659.00 / 263955 ran
      to 2010-06-03 and re-joins the same day, so the label read
      ``(antenna, receiver)`` when only the receiver changed (TRIMBLE 5700
      -> NETRS).  Invisible while labels said "antenna"; obvious the moment
      they name the unit.
    - **the ``1000-01-01`` sentinel** meaning "since forever", which is a
      registration artefact rather than an event.

    Then coalescing to distinct calendar days, because one site visit that
    swaps antenna and receiver is two rows sharing a ``time_from`` — and,
    on RHOF's 2023-08-16, two rows timestamped 13:20 and 15:30 that are
    plainly the same visit.  One day is one line, always.

    Net effect on the working set: RHOF 4 lines -> 3, SELF 6 -> 5.

    Args:
        sta: station code.
        url: TOS REST endpoint; None uses ``tostools``' own default.
        payload: pre-fetched entity dict (a cached fixture in tests) —
            when given, no station call is made.
        devices: pre-resolved ``{id_entity: DeviceInfo}`` — when given, no
            device calls are made either.  This is what lets the fixture
            test run the whole filter offline.

    Returns:
        ``[(yearf, label), …]`` sorted, one per distinct day, the label
        naming the instruments — ``2010-06-03 (rx TRIMBLE NETRS)``,
        ``2002-02-05 (rx TRIMBLE 5700, ant TRM29659.00)``.

    Raises:
        RuntimeError: on any TOS failure, including a device lookup that
            resolves NOTHING.  That last case matters: an empty list draws
            a figure with no green lines, which is indistinguishable from a
            station that never had a swap, so a blanket failure has to
            surface as the caller's "continuing without equipment lines"
            warning instead.  Partial failures are survivable and are
            reported as a count.
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

    # Only rows that could still become a line are worth a lookup: dropping
    # the sentinel first keeps a registration artefact from costing a call.
    dated: list[tuple[str, str, int]] = []
    for join in joins:
        raw = str(join.get("time_from") or "")[:10]
        until = str(join.get("time_to") or "")[:10]
        dev = join.get("id_entity_child")
        if not raw or dev is None:
            continue
        try:
            y, _m, _d = (int(v) for v in raw.split("-"))
        except ValueError:
            continue
        if y <= TOS_SENTINEL_YEAR:  # "since forever", not an event
            continue
        dated.append((raw, until, int(dev)))
    dated.sort()

    if devices is None:
        devices = resolve_devices([d for _f, _t, d in dated], url=url)
    unresolved = {d for _f, _t, d in dated if d not in devices}
    if dated and len(unresolved) == len({d for _f, _t, d in dated}):
        raise RuntimeError(
            f"TOS device lookup resolved nothing for {sta} "
            f"({len(unresolved)} device(s)); refusing to report 'no equipment "
            f"changes', which is what an empty result would look like"
        )
    if unresolved:
        print(
            f"warning: {sta}: {len(unresolved)} device(s) could not be typed; "
            f"excluded — a line that cannot be justified is worse than a "
            f"missing one.",
            file=sys.stderr,
        )

    # What each subtype was running immediately before this join, so a
    # re-registration of the SAME unit can be told from an installation.
    running: dict[str, tuple[str, DeviceInfo]] = {}
    by_day: dict[str, dict[str, str]] = {}
    for start, until, dev in dated:
        info = devices.get(dev)
        if info is None or info.subtype not in EQUIPMENT_SUBTYPES:
            continue  # radome, monument, SIM card, GSM modem, unresolved, …
        span = _span_days(start, until)
        if span is not None and span < MIN_DEPLOYMENT_DAYS:
            # campaign gear, never this station's equipment -- so it must not
            # become the baseline the NEXT install is compared against either
            continue
        prior = running.get(info.subtype)
        running[info.subtype] = (until, info)
        if prior is not None and prior[0] == start and is_same_unit(prior[1], info):
            continue  # continues the very unit it replaced — not new
        by_day.setdefault(start, {})[EQUIPMENT_SUBTYPES[info.subtype]] = info.describe()

    out: list[tuple[float, str]] = []
    for day, kinds in sorted(by_day.items()):
        y, m, d = (int(v) for v in day.split("-"))
        # receiver first: the operator asked for it by name, and it is the
        # part that changes most often
        order = list(EQUIPMENT_SUBTYPES.values())
        what = ", ".join(
            f"{'rx' if k == 'receiver' else 'ant'} {kinds[k]}"
            for k in sorted(kinds, key=lambda k: -order.index(k))
        )
        out.append((float(TimetoYearf(y, m, d)), f"{day} ({what})"))
    return out


def clip_events_to_span(
    events: Sequence[tuple[float, str]], yearf: Any
) -> tuple[list[tuple[float, str]], list[tuple[float, str]]]:
    """Split events into those inside the plotted span and those outside.

    Event epochs come from catalogs that describe the STATION, not this
    figure: TOS knows when the receiver was installed, ``steps.csv`` knows
    every declared offset, and neither has an opinion about how much of
    the series has been processed.  BJTV is the case — the antenna went up
    2021-08-09 and the solution starts 2025-02, so the install sits 3.5
    years off the left edge.

    Left in, such an event is not merely useless, it is *misdrawn*:
    ``axvline`` is clipped to the axes and vanishes, but the label is
    :class:`matplotlib.text.Text`, which does not clip by default — so the
    line disappears and its caption is left stranded in the margin, beside
    an axis it does not mark.  A reader sees an annotation pointing at
    nothing.

    Dropping is the honest half; the caller prints what was dropped, so
    "this station was installed long before the series starts" stays
    available as TEXT, where it does not have to be positioned to be true.

    Args:
        events: ``(epoch [yr], label)`` pairs.
        yearf: The plotted epochs; the span is their finite min/max, which
            is exactly what ``stdTimesPlot`` hands to ``setXlim``.

    Returns:
        ``(inside, outside)``, each in the input's order.  An empty or
        all-non-finite ``yearf`` keeps everything — with no span there is
        nothing to be outside of, and silently emptying the annotation
        would be the worse failure.
    """
    finite = np.asarray(yearf, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return list(events), []
    lo, hi = float(finite.min()), float(finite.max())
    inside: list[tuple[float, str]] = []
    outside: list[tuple[float, str]] = []
    for event in events:
        (inside if lo <= float(event[0]) <= hi else outside).append(event)
    return inside, outside


def add_event_lines(
    fig: Figure, events: Sequence[tuple[float, str]], color: str
) -> Figure:
    """Vertical lines with a label on the top axis.

    Lines go through ``timesmatplt.addEvent`` (the existing primitive —
    ``axvline`` on every axis); only the text is new, and only on axis 0,
    because repeating it on all three is noise.

    The label is drawn in FULL.  It used to be ``label.split(" ")[0]`` —
    the date and nothing else — which was right while the rest of the
    string was a device count, and silently threw away the equipment names
    the moment they existed.  Date and equipment go on two rotated lines
    so the identifying part stays at the axis edge and the detail runs
    beside it rather than after it.
    """
    import gps_plot.timesmatplt as tplt
    from gtimes.timefunc import TimefromYearf

    if not events:
        return fig
    tplt.addEvent({TimefromYearf(e): [color] for e, _ in events}, fig, linestyle=":")
    ax = fig.axes[0]
    _lo, hi = ax.get_ylim()
    for epoch, label in events:
        head, sep, tail = label.partition(" (")
        text = f"{head}\n{tail.rstrip(')')}" if sep else label
        ax.text(
            TimefromYearf(epoch),
            hi,
            text,
            rotation=90,
            va="top",
            ha="right",
            fontsize=7,
            linespacing=0.95,
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
        *(
            # only when it says something the hull does not
            [
                f"  segments       {record.get('segments')}",
                f"  excised        {[round(g, 4) for g in record.get('segment_gaps') or []]} yr",
            ]
            if len(record.get("segments") or []) > 1
            else []
        ),
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

#: Marker colours of the FLAGGED lanes under ``--show-outliers``, where the
#: emphasis is inverted: the verdicts are what the operator came to look at
#: and the series itself becomes the backdrop.  Only the marker swaps — the
#: error bars stay grey on every lane, as they already are for kept and
#: rejected epochs alike, so nothing about the swap is load-bearing beyond
#: colour.  The out-of-window lane keeps :data:`OUTSIDE_FACE_COLOR`: hollow
#: red against solid red preserves exactly the distinction the two greys
#: carry, and it can stay a true "none" fill because the base series is
#: masked at those epochs — there is never a marker underneath to show
#: through.
INVERTED_FACE_COLOR: str = "red"
INVERTED_EDGE_COLOR: str = "red"


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


def build_pages(
    sta: str,
    record: dict[str, Any],
    yearf: Any,
    data: Any,
    sigma: Any,
    *,
    terms: str = "all",
    events: dict[Any, Any] | None = None,
    tos_events: Sequence[tuple[float, str]] | None = None,
    seismic_events: Sequence[tuple[float, str]] | None = None,
    outliers: Any = None,
    outside_outliers: Any = None,
    outside_provisional: Any = None,
    hide_outliers: bool = False,
    show_outliers: bool = False,
) -> list[Figure]:
    """Build the workbench's two pages as live figures.

    Split out of :func:`render` so a caller can DISPLAY these rather than
    only write them -- the interactive pickers need the production figure,
    overlays and all, and rebuilding it elsewhere is how two figures drift
    apart. ``render`` is now the thin "save them" wrapper.

    Two-page: plate frame + fitted trajectory, then the detrended view.

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

    ``show_outliers`` INVERTS that emphasis — flagged epochs red, the
    unflagged series grey — and is display-only in exactly the sense
    ``hide_outliers`` is: same masks, same counts, same record.  The two
    are mutually exclusive at the CLI because they are contradictory.
    The swap is done by OVERPAINTING the kept series grey rather than by
    colouring it grey at the source: ``stdTimesPlot`` draws red and has
    no colour knob, and giving it one would reach into
    ``plot-gps-timeseries``' path for a workbench display option.  Two
    consequences are deliberate, not oversights — the "last datapoint"
    highlight keeps its green rim (it marks the EPOCH, not a verdict,
    and a 3.5 pt grey dot sits inside its 5.5 pt ring), and the y-axis
    widens to hold the flagged epochs, which is the whole point.

    The inversion answers a question the normal view cannot: an
    all-grey component is one where NOTHING was flagged, and on a
    station whose detection aborted (the excess-candidate rule serves a
    component raw) that is the only on-figure trace of it — measured on
    NYLA, where north and east abort at 7.9 / 9.2 % candidates and a
    +48 mm one-day blunder in east therefore renders as ordinary red
    data in the normal view.
    """
    import gps_plot.timesmatplt as tplt
    from gps_analysis import evaluate_record
    from geo_dataread.gps_views import apply_stored_detrend

    x = list(_to_datetime(yearf))

    # Clip HERE, not in the caller: both pages draw from these lists, and a
    # caller that forgot would produce a figure whose annotations point off
    # the axes.  `main` reports what this drops -- see clip_events_to_span.
    tos_events, _ = clip_events_to_span(tos_events or (), yearf)
    seismic_events, _ = clip_events_to_span(seismic_events or (), yearf)

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
        judged = np.logical_or.reduce(masks) if masks else None
        if judged is not None:
            kept_, _ = split_outliers(values, sigma, judged)
        for name, mask in (
            ("fit", outliers),
            ("outside", outside_outliers),
            # provisional epochs are NOT in `masks`: they stay in the series
            ("provisional", outside_provisional),
            # the inversion's grey lane -- the complement of everything
            # judged, i.e. the series MINUS both flagged lanes.  Built from
            # the same one-line split as its opposites so the two colourings
            # cannot cover different epoch sets.
            ("unflagged", ~judged if (show_outliers and judged is not None) else None),
        ):
            lanes[name] = (
                split_outliers(values, sigma, mask)[1] if mask is not None else None
            )
        if show_outliers and judged is None:
            # nothing judged at all: every epoch is the grey lane
            lanes["unflagged"] = (values, sigma)
        return kept_, lanes

    def _draw_lanes(fig: Figure, lanes: dict[str, tuple[Any, Any] | None]) -> None:
        if lanes["unflagged"] is not None:
            # FIRST, so the flagged lanes land on top of it: this repaints the
            # red series stdTimesPlot has already drawn, marker for marker.
            tplt.addData(
                x,
                *lanes["unflagged"],
                fig,
                ecolor=tplt.OUTLIER_ERRORBAR_COLOR,
                markerfacecolor=tplt.OUTLIER_FACE_COLOR,
                markeredgecolor=tplt.OUTLIER_EDGE_COLOR,
            )
        fit_face = INVERTED_FACE_COLOR if show_outliers else tplt.OUTLIER_FACE_COLOR
        edge = INVERTED_EDGE_COLOR if show_outliers else tplt.OUTLIER_EDGE_COLOR
        if lanes["fit"] is not None and not hide_outliers:
            tplt.addData(
                x,
                *lanes["fit"],
                fig,
                ecolor=tplt.OUTLIER_ERRORBAR_COLOR,
                markerfacecolor=fit_face,
                markeredgecolor=edge,
            )
        if lanes["outside"] is not None and not hide_outliers:
            tplt.addData(
                x,
                *lanes["outside"],
                fig,
                ecolor=tplt.OUTLIER_ERRORBAR_COLOR,
                markerfacecolor=OUTSIDE_FACE_COLOR,
                markeredgecolor=edge,
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

    # page 2 -- the same series with that trajectory removed.  Detrend the
    # FULL series, then mask: apply_stored_detrend is a pure evaluation, so
    # feeding it the NaN-masked series would only propagate the NaNs and
    # lose the overlay values.
    det = np.asarray(
        apply_stored_detrend(record, yearf, data, terms=terms, frame="plate_removed")
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

    return [fig, fig2]


def render(
    sta: str,
    record: dict[str, Any],
    yearf: Any,
    data: Any,
    sigma: Any,
    out: str | Path,
    **kwargs: Any,
) -> Path:
    """Write :func:`build_pages`' figures to a two-page PDF.

    Kept as its own function with an unchanged signature: every existing
    caller and test targets this, and the split exists to let the pickers
    DISPLAY the same figures, not to change what the workbench writes.
    """
    import matplotlib

    # Agg belongs HERE, not in build_pages: writing a PDF wants a headless
    # backend, but the pickers DISPLAY the same figures and forcing Agg in
    # the shared builder would silently give them nothing to click on.
    matplotlib.use("Agg")
    from matplotlib.backends.backend_pdf import PdfPages

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    figs = build_pages(sta, record, yearf, data, sigma, **kwargs)
    with PdfPages(out) as pdf:
        for fig in figs:
            pdf.savefig(fig, bbox_inches="tight")
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
    from gps_plot.plot_gps_timeseries import outlier_param_help

    # Raw*Description* leaves description and epilog verbatim (argument help is
    # still auto-wrapped), which is what lets the generated --outlier-param
    # table keep its columns -- so the prose either side has to be wrapped here
    # rather than by argparse.
    p = argparse.ArgumentParser(
        prog="gps-detrend-workbench",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.fill(
            "Operator workbench for detrending: one station, one PDF, "
            "two views (plate frame + fitted trajectory, and the detrended "
            "series). Estimation reuses gps-estimate-detrend's own code path, "
            "so a record made here means the same thing as a batch-made one.",
            width=78,
        ),
        epilog="\n\n".join(
            block
            for block in (
                textwrap.fill(
                    "Detection runs the FULL pipeline by default, falling back "
                    "to S0-only (loudly) if the excess-candidate rule aborts. "
                    "An S0 record leaves model-visible outliers in the fit, so "
                    "its parameters are biased relative to a full-detection "
                    "one — on a fallback, prefer declaring the missing step "
                    "with --step over accepting the S0 record.",
                    width=78,
                ),
                outlier_param_help(),
            )
            if block
        ),
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
    # Contradictory by definition: one drops the verdicts from the figure, the
    # other makes them the only thing on it.
    emphasis = p.add_mutually_exclusive_group()
    emphasis.add_argument(
        "--hide-outliers",
        action="store_true",
        help="drop the grey overlay of the epochs the fit rejected. They are "
        "masked out of the plotted series either way — this decides only "
        "whether the figure still SHOWS them, and hiding them lets the "
        "y-axis tighten to the fitted series. Same name and meaning as "
        "plot-gps-timeseries --hide-outliers",
    )
    emphasis.add_argument(
        "--show-outliers",
        action="store_true",
        help="INVERT the emphasis: flagged epochs red, everything else grey. "
        "The counts, the masks and the record are identical — only the "
        "colours swap, so this is for auditing WHAT was flagged (and, on "
        "an all-grey component, what was not). The solid/hollow "
        "distinction survives the swap: solid red is the fit's own "
        "rejection, hollow red the out-of-window screen. Gold stays gold "
        "— provisional has no inverse. The y-axis widens, the exact "
        "inverse of --hide-outliers",
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
        help="skip the TOS lookup for new antenna / receiver installs. On by "
        "default; any TOS failure already degrades to a warning, so the "
        "workbench stays usable off-VPN either way. Costs one entity call "
        "per joined device (~0.5 s a station) because the join row names no "
        "device type, and typing it is what keeps a SIM-card swap from "
        "drawing a line",
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
        "--segment",
        action="append",
        default=[],
        metavar="START:END",
        help="fit the UNION of these intervals [fractional year], repeatable; "
        "either side may be empty for an open bound. This is how you excise "
        "a post-seismic transient and still estimate the coseismic offset: "
        "the flanks on both sides of the excision constrain the level, so a "
        "--step inside the gap is estimable. Mutually exclusive with "
        "--window-start/--window-end, which is the one-segment spelling",
    )
    p.add_argument(
        "--stage",
        action="append",
        default=[],
        metavar="NAME:GROUP,GROUP[@START:END]",
        help="declare one stage of a STAGED estimation, repeatable and "
        "executed in order: NAME labels it, the comma list names the term "
        "groups it estimates, and the optional @START:END (same union "
        "spelling as --segment) is its fit domain. OMITTING @ inherits the "
        "caller's domain; '@:' is the full span; a bare trailing '@' is "
        "refused because those two differ. Terms are best constrained on "
        "different domains -- the seasonal where the data is clean, the "
        "rate where the baseline is longest",
    )
    p.add_argument(
        "--hold",
        action="append",
        default=[],
        metavar="[STAGE:]GROUP=stage:NAME|donor:STA",
        help="hold a term group fixed instead of estimating it: "
        "'stage:NAME' reuses what an EARLIER stage fitted, 'donor:STA' "
        "borrows station STA's stored value. The kind is never inferred -- "
        "the two are stored with different provenance, so a bare "
        "'GROUP=clean' is refused. The STAGE: prefix is required once two "
        "or more stages are declared, since binding to the last --stage "
        "seen would make flag ORDER change the science",
    )
    p.add_argument(
        "--term",
        action="append",
        default=[],
        metavar="KIND@EPOCH,tau=VALUE",
        help="add a transient term to the model, repeatable: "
        "'log@2008.4085,tau=1.0' is A*log(1+dt/T) (afterslip / rate-and-"
        "state, Bevis & Brown eq. 9), 'exp@2010.0,tau=0.33' is "
        "Phi*(1-exp(-dt/tau)) (relaxation toward a steady state, Reverso "
        "eq. 20). tau [yr] is REQUIRED and operator-fixed, which keeps the "
        "model linear in its parameters; the amplitude is estimated. A "
        "transient is the only thing that fixes an excess-candidate abort "
        "on a deforming station -- re-judging the epochs does not, because "
        "the abort is a model-adequacy problem",
    )
    p.add_argument(
        "--analysis-yaml",
        default=None,
        metavar="PATH",
        help="where --commit stores a --stage/--hold plan (default: the "
        "deployed analysis.yaml). The plan is config, not a fitted quantity, "
        "so it lives beside the record where a batch re-run will find it",
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
        help="repeatable OutlierParams override, applied on top of the stage "
        "set in force (e.g. max_flag_fraction=0.15 to stop the "
        "excess-candidate abort from serving a component raw). Every valid "
        "NAME with its spec default is listed at the END of this help; an "
        "unknown one also lists them",
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

    stage_plan = None
    if args.stage or args.hold:
        # Parsed BEFORE any data is read, like the --terms refusal above: a
        # grammar error should cost nothing, and every message below names the
        # spelling that would have worked.
        from geo_dataread.stage_plan import build_stage_plan

        try:
            stage_plan = build_stage_plan(args.stage, args.hold)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 6

        if args.donor:
            # --donor replaces the WHOLE record with the donor's; a donor hold
            # replaces ONE term group and fits the rest. Silently letting the
            # first win would discard the plan the operator just wrote.
            print(
                "error: --donor cannot be combined with --stage/--hold. "
                "--donor swaps in the donor's entire record, whereas "
                "'--hold GROUP=donor:STA' borrows ONE group and estimates the "
                "rest -- which is what a stage plan is for. Drop --donor and "
                "name the group you meant to borrow.",
                file=sys.stderr,
            )
            return 6

    resolved_stages = None
    donor_lookup = None
    if stage_plan is not None:
        from geo_dataread.gps_views import (
            default_params_path,
            read_detrend_params,
            station_detrend_record,
        )
        from geo_dataread.stage_plan import resolve_stage_plan

        def _donor(code: str) -> dict:
            doc = read_detrend_params(args.params or default_params_path())
            rec, _src = station_detrend_record(doc, code)
            if rec is None:
                raise RuntimeError(
                    f"donor {code} has no stored record; estimate and commit "
                    f"{code} before borrowing from it"
                )
            return dict(rec)

        try:
            # component 0 only: the plan is resolved per component inside the
            # estimator, this is the up-front existence check so a missing
            # donor fails before any data is read.
            # Up-front existence check only: a missing donor must fail
            # before any data is read. The estimator re-resolves per
            # component, since a donor hold borrows THAT component's numbers.
            resolve_stage_plan(stage_plan, lookup_donor=_donor, component=0)
            resolved_stages = stage_plan
            donor_lookup = _donor
        except (RuntimeError, ValueError, KeyError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 4

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
            stage_plan=resolved_stages,
            lookup_donor=donor_lookup,
            terms_spec=args.term or None,
            segments=_resolve_cli_segments(args),
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

    # Say what the figure will NOT show. `render` clips events to the plotted
    # span (a label outside it is stranded in the margin beside an axis it does
    # not mark), and a station whose equipment predates the processed solution
    # loses every line -- BJTV: installed 2021-08-09, series starts 2025-02.
    # Printed, that fact survives without needing a position on the figure.
    _kept, off_figure = clip_events_to_span(
        list(tos_events) + list(declared_other) + list(seismic), yearf
    )
    if off_figure:
        print(
            f"\n  outside the plotted span {yearf.min():.4f}–{yearf.max():.4f}, "
            f"so NOT drawn ({len(off_figure)}):"
        )
        for epoch, label in sorted(off_figure):
            print(f"    {epoch:9.4f}  {label}")

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
        show_outliers=args.show_outliers,
    )
    n_out = [int(v) for v in np.asarray(estimate.outliers).sum(axis=1)]
    n_unjudged = int(np.count_nonzero(~np.asarray(estimate.in_window)))
    # The text names the colour the figure actually uses -- under --show-outliers
    # "grey" would describe the epochs this count does NOT cover.
    if args.hide_outliers:
        state, outside_state = "hidden", "hidden"
    elif args.show_outliers:
        state, outside_state = "red", "hollow red"
    else:
        state, outside_state = "grey", "hollow grey"
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
                f"({outside_state}; masked, but NOT part of the record's "
                f"n_rejected)"
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
        if stage_plan is not None:
            # The plan is config, not a fitted quantity, so it is stored
            # ALONGSIDE the record rather than inside it: analysis.yaml is
            # what a batch re-run reads, and a plan living only in the record
            # would be invisible to gps-estimate-detrend.
            from geo_dataread.stage_plan import (
                default_analysis_yaml_path,
                write_stage_plan,
            )

            yaml_path = args.analysis_yaml or default_analysis_yaml_path()
            if yaml_path is None:
                print(
                    "  warning: stage plan NOT stored — no analysis.yaml is "
                    "reachable (no gpsconfig on this host). The record was "
                    "committed, but a batch re-run will re-fit this station "
                    "single-stage. Pass --analysis-yaml to say where.",
                    file=sys.stderr,
                )
            else:
                write_stage_plan(yaml_path, sta, stage_plan)
                print(
                    f"  stage plan -> {yaml_path}   ({len(stage_plan.stages)} stages)"
                )
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
