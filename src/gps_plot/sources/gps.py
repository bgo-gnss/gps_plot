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

from gps_plot.series import Series


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


@dataclasses.dataclass(frozen=True)
class _StationTuple:
    """The (yearf, data, Ddata) view the empirical-σ helper reads."""

    yearf: Any
    data: Any
    Ddata: Any


def _as_station_tuple(s: Series) -> _StationTuple:
    return _StationTuple(s.t, s.y, s.sigma)
