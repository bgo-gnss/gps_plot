"""GNSS source: one station, or a baseline ``AAAA-BBBB``, as a :class:`Series`.

The per-station pipeline is :func:`gps_plot.timesmatplt.station_series`
(read → clean → remove steps → detrend); a baseline runs both ends through
it and differences them with :func:`geo_dataread.baseline.baseline`. Both
the main series and the 8 h session overlay of ``gpsplot ts`` come from
:func:`fetch`, and so will ``gpsplot stack``.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Any

import numpy as np

from gps_plot.series import Series

#: the refs a model line can be drawn over: the stored record is fitted to
#: the plate-removed series ("detrend" reads that same series)
MODEL_REFS = ("plate", "detrend")

#: --model-terms names -> staged term groups (gps_analysis.terms.GROUP_ORDER)
MODEL_TERM_GROUPS: dict[str, str] = {
    "linear": "secular",
    "periodic": "periodic",
    "steps": "step",
    "transient": "transient",
}


@dataclasses.dataclass(frozen=True)
class SeriesRequest:
    """What to read and which view: the keywords of ``station_series``."""

    fstart: float | None
    fend: float | None
    ref: str
    Dir: str | None
    tType: str
    uncert: int
    view: str
    outlier_params: Any = None
    outlier_overrides: str | None = None
    provisional_days: float | None = None
    remove_kinds: Sequence[str] | None = None
    detrend_terms: Sequence[str] | None = None
    steps_catalog: str | None = None
    params: str | None = None
    ref_position: Any = None
    convention: str = "geodetic"
    pos_id: str | None = None
    icing: bool = True
    icing_ratio: float | None = None
    icing_mode: str = "remove"
    snow: bool = True
    snow_mode: str = "remove"

    def replace(self, **changes: Any) -> SeriesRequest:
        return dataclasses.replace(self, **changes)

    def kwargs(self) -> dict[str, Any]:
        """The fields as keywords, values untouched.

        Not ``dataclasses.asdict``: that recurses into dataclass values and
        would hand ``station_series`` a dict instead of the OutlierParams.
        """
        return {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}


@dataclasses.dataclass(frozen=True)
class BaselineOptions:
    """How a baseline is formed (ignored for a single station).

    Attributes:
        tol: Epoch-matching tolerance [yr]; None = the geo_dataread default.
        empirical: Use the empirical baseline noise record for σ
            (``--baseline-sigma empirical``); formal quadrature otherwise.
        rho: Common-mode correlation; given, it overrides the record.
        sigma_scale: Per-end σ scale factors; given with ``rho``.
        noise_path: Baseline noise record path; None = deployed default.
        check_view: Refuse a detrended baseline with an end that has no
            stored record (it would difference detrended against plate).
    """

    tol: float | None = None
    empirical: bool = False
    rho: Any = None
    sigma_scale: Any = None
    noise_path: str | None = None
    check_view: bool = True


def fetch_station(sta: str, req: SeriesRequest) -> Series:
    """One station through the per-station view pipeline."""
    from gps_plot import timesmatplt as tplt

    one = tplt.station_series(sta, **req.kwargs())
    return Series(
        label=sta,
        t=one.yearf,
        y=one.data,
        sigma=one.Ddata,
        outliers=one.outliers,
        provisional=one.provisional,
        aborted=one.aborted,
        detrend_applied=one.detrend_applied,
        iced=one.iced,
        buried=one.buried,
    )


def fetch(
    spec: str,
    req: SeriesRequest,
    baseline: BaselineOptions | None = None,
) -> Series:
    """A station (``HOFN``) or a baseline (``VFLN-VFLS``) as a Series.

    A baseline's per-station overlays do not survive differencing (their
    epochs belong to one end; masked epochs are NaN in the difference), so
    it carries no ``outliers``/``provisional``; a component is ``aborted``
    if it aborted at either end. ``ends`` holds both stations zeroed at the
    baseline's first common epoch. ``meta`` holds the ``rho`` and
    ``sigma_scale`` the baseline σ was formed with (None = formal
    quadrature) and a ``sigma_note`` saying which σ an empirical request
    ended up with.
    """
    from geo_dataread.baseline import baseline as compute_baseline
    from geo_dataread.baseline import check_same_view, split_baseline

    pair = split_baseline(spec)
    if pair is None:
        return fetch_station(spec, req)
    if req.tType == "JOIN":
        raise ValueError(f"baseline {spec}: tType JOIN is not supported")
    opts = baseline or BaselineOptions()
    ends = [fetch_station(s, req) for s in pair]
    if opts.check_view and (req.view == "detrended" or req.ref == "detrend"):
        check_same_view(pair, [e.detrend_applied for e in ends], spec=spec)
    rho, sigma_scale, sigma_note = opts.rho, opts.sigma_scale, None
    if opts.empirical and rho is None:
        from gps_plot import timesmatplt as tplt

        rho, sigma_scale, sigma_note = tplt._empirical_pair_inputs(
            pair, [_as_station_tuple(e) for e in ends], opts.noise_path
        )
    tol_kw = {} if opts.tol is None else {"tol": opts.tol}
    bl = compute_baseline(
        (ends[0].t, ends[0].y, ends[0].sigma),
        (ends[1].t, ends[1].y, ends[1].sigma),
        stations=pair,
        rho=rho,
        sigma_scale=sigma_scale,
        **tol_kw,
    )
    aborts = [e.aborted for e in ends if e.aborted is not None]
    zeroed = tuple(
        Series(label=e.label, t=t, y=d, sigma=e.sigma)
        for (t, d), e in zip(bl.ends, ends, strict=True)
    )
    return Series(
        label=spec,
        t=bl.yearf,
        y=bl.data,
        sigma=bl.sigma,
        aborted=[any(c) for c in zip(*aborts, strict=True)] if aborts else None,
        detrend_applied=all(e.detrend_applied for e in ends),
        ends=(zeroed[0], zeroed[1]),
        meta={"rho": rho, "sigma_scale": sigma_scale, "sigma_note": sigma_note},
    )


def station_record(sta: str, req: SeriesRequest) -> Any:
    """The stored detrend record a request's series is detrended with.

    Raises:
        ValueError: no record for the station, or one in another frame.
    """
    from geo_dataread import gps_views as gv

    from gps_plot import timesmatplt as tplt

    series_frame, params = tplt.record_context(req.tType, req.pos_id, req.params)
    record, _src = gv.station_detrend_record(gv.read_detrend_params(params), sta)
    if record is None:
        raise ValueError(f"{sta}: no stored detrend record to draw")
    frame = record.get("frame")
    if frame is not None and frame != series_frame:
        raise ValueError(
            f"{sta}: record frame {frame!r} != series frame {series_frame!r}"
        )
    return record


def model_values(
    sta: str,
    req: SeriesRequest,
    t: Any,
    record: Any,
    terms: Sequence[str] | None = None,
) -> Any:
    """The stored model line at ``t``, in the plotted window's datum.

    Default (``terms`` None): what is still in the plotted series,
    ``model − removed − shift`` — the full trajectory minus what the view
    takes out of the data (``--remove-steps`` kinds, the ``--ref detrend``
    terms; found by running the same :func:`trajectory_views` on zeros). So
    ``data − line`` is the same in every view: the residual from the model.

    With ``terms`` (``--model-terms``, names of :data:`MODEL_TERM_GROUPS`)
    the line is exactly those groups, ``model[terms] − shift``, whatever was
    removed from the data, e.g. the seasonal line over data with the
    seasonal removed.

    The shift (``window_datum_shift``) moves the record, fitted to the full
    series, to the window's zero (a windowed read is zeroed at its start).

    Returns:
        ``(3, len(t))`` [mm].
    """
    from geo_dataread.gps_views import window_datum_shift
    from gps_analysis import evaluate_record

    from gps_plot import timesmatplt as tplt

    if req.ref not in MODEL_REFS:
        raise ValueError(
            f"{sta}: a model line needs ref plate or detrend, not {req.ref!r}"
        )
    t = np.asarray(t, dtype=float)
    shift = window_datum_shift(
        sta,
        fstart=req.fstart,
        fend=req.fend,
        ref="plate",
        Dir=req.Dir,
        tType=req.tType,
        uncert=req.uncert,
        **tplt.pos_read_kwargs(req.ref_position, req.convention, req.pos_id),
    )
    if terms:
        bad = set(terms) - set(MODEL_TERM_GROUPS)
        if bad:
            raise ValueError(f"unknown model term(s) {sorted(bad)}")
        groups = tuple(MODEL_TERM_GROUPS[x] for x in terms)
        line = np.asarray(evaluate_record(record, t, groups=groups), dtype=float)
        return line - shift[:, None]
    full = np.asarray(evaluate_record(record, t), dtype=float)
    removed, _ = tplt.trajectory_views(
        sta,
        t,
        np.zeros_like(full),
        ref=req.ref,
        view=req.view,
        tType=req.tType,
        pos_id=req.pos_id,
        remove_kinds=req.remove_kinds,
        detrend_terms=req.detrend_terms,
        steps_catalog=req.steps_catalog,
        params=req.params,
    )
    return full + np.asarray(removed, dtype=float) - shift[:, None]


def fetch_model(
    spec: str, req: SeriesRequest, t: Any, terms: Sequence[str] | None = None
) -> Series:
    """The model line over a station's series, sampled for drawing.

    Dense over the extent of ``t`` (the plotted epochs) with step epochs
    bracketed (:func:`gps_plot.trajectory.curve_grid`). ``meta["window"]``
    is the record's fit window, so a layer can mark where it extrapolates;
    ``meta["terms"]`` the ``--model-terms`` drawn (None = what is left in
    the data; see :func:`model_values`).

    Raises:
        ValueError: a baseline (not supported yet), no record, a frame
            mismatch, or a ref the record does not apply to.
    """
    from geo_dataread.baseline import split_baseline

    from gps_plot.trajectory import curve_grid

    if split_baseline(spec) is not None:
        raise ValueError(f"{spec}: a model line over a baseline is not supported yet")
    record = station_record(spec, req)
    grid = curve_grid(record, t)
    return Series(
        label=f"{spec} model",
        t=grid,
        y=model_values(spec, req, grid, record, terms),
        sigma=None,
        meta={"window": record.get("window"), "terms": tuple(terms or ()) or None},
    )


#: samples averaged for the zero level: vshift's ``Period`` (getData's own)
ZERO_SAMPLES = 5


def zero_window(spec: str, req: SeriesRequest) -> list[tuple[float, float]] | None:
    """The fit segments of ``spec``'s record when its LINEAR term was removed.

    A record's rate is the station's velocity over its fit window, so with
    the rate removed that window is the station's zero-velocity period and
    the natural zero line (BGÓ 2026-10-20). None when the view keeps the
    rate (``--detrend-terms periodic``), for a baseline, without a record,
    and for a BORROWED record (its rate is another station's; this
    station's own window may be all deformation -- KASC holding OLAC's).
    """
    from geo_dataread import gps_views as _gv
    from geo_dataread.baseline import split_baseline

    from gps_plot import timesmatplt as tplt

    if split_baseline(spec) is not None:
        return None
    terms = req.detrend_terms or tplt.DEFAULT_DETREND_TERMS
    if not (req.view == "detrended" or (req.ref == "detrend" and "linear" in terms)):
        return None
    try:
        _frame, params = tplt.record_context(req.tType, req.pos_id, req.params)
        record, _src = _gv.station_detrend_record(_gv.read_detrend_params(params), spec)
    except Exception:  # noqa: BLE001 - no record: the start zero, as before
        return None
    if record is None or record.get("borrowed"):
        return None
    segs = record.get("segments") or (
        [record["window"]] if record.get("window") else None
    )
    return [(float(a), float(b)) for a, b in segs] if segs else None


def detrended_shift(spec: str, req: SeriesRequest, series: Series) -> Any:
    """What re-zeroes a detrended station, or None.

    With the record's linear term removed, the zero is the station's level
    over the record's fit window (:func:`zero_window`, its zero-velocity
    period: ``estimate_offset`` over every finite sample in it), when at
    least ``ZERO_SAMPLES`` of them are in the plotted span. Otherwise, at
    its start:

    ``getData`` zeroes a series at its start (``vshift``: the 1/σ-weighted
    mean of the first 5 kept samples, ``gps_analysis.prep_plot_series``),
    and the stored model is subtracted AFTER that, so a detrended station
    starts wherever the model sat at the window start (HOFN, one year:
    ~65/−150/−375 mm). The same zeroing applied again after the detrend
    (``gps_analysis.baseline.estimate_offset`` over the first 5 finite
    samples per component) puts it back on zero, like every other plot.

    Returns ``−ŷ₀`` (3,), to add. None where it does not apply: baselines
    (already zeroed at their first common epoch), no applied detrend, and a
    chosen reference position (``--session both`` pins the 8 h overlay to
    the main series' zero).
    """
    from geo_dataread.baseline import split_baseline
    from gps_analysis.baseline import estimate_offset

    if (
        split_baseline(spec) is not None
        or not series.detrend_applied
        or req.ref_position is not None
    ):
        return None
    t = np.asarray(series.t, dtype=float)
    y = np.asarray(series.y, dtype=float)
    sigma = np.asarray(series.sigma, dtype=float)
    level = np.zeros(y.shape[0])
    in_window = np.zeros(t.shape, dtype=bool)
    for a, b in zero_window(spec, req) or ():
        in_window |= (t >= a) & (t <= b)
    for c in range(y.shape[0]):
        ok = np.isfinite(y[c]) & np.isfinite(sigma[c])
        idx = np.flatnonzero(ok & in_window)
        if idx.size < ZERO_SAMPLES:
            idx = np.flatnonzero(ok)[:ZERO_SAMPLES]
        if idx.size:
            level[c] = float(
                np.asarray(estimate_offset(t[idx], y[c, idx], sigma[c, idx]))
            )
    return -level


def rezero_detrended(spec: str, req: SeriesRequest, series: Series) -> Series:
    """A detrended station re-zeroed (see :func:`detrended_shift`).

    The outlier and provisional overlays (same datum as the values) move
    with it, and ``meta["zero_shift"]`` records the shift so a model line
    can follow. Anything the shift does not apply to is returned unchanged.
    """
    shift = detrended_shift(spec, req, series)
    if shift is None:
        return series
    col = np.asarray(shift, dtype=float)[:, None]

    def moved(pair: Any) -> Any:
        if pair is None:
            return None
        return (np.asarray(pair[0], dtype=float) + col, pair[1])

    return dataclasses.replace(
        series,
        y=np.asarray(series.y, dtype=float) + col,
        outliers=moved(series.outliers),
        provisional=moved(series.provisional),
        iced=moved(series.iced),
        buried=moved(series.buried),
        meta={**series.meta, "zero_shift": shift},
    )


@dataclasses.dataclass(frozen=True)
class _StationTuple:
    """The (yearf, data, Ddata) view the empirical-σ helper reads."""

    yearf: Any
    data: Any
    Ddata: Any


def _as_station_tuple(s: Series) -> _StationTuple:
    return _StationTuple(s.t, s.y, s.sigma)
