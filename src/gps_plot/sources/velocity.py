"""Velocity and displacement sources for the maps: one value per station.

Three ways to a station's velocity (mm/yr, N/E/U, 1-σ):

* :func:`window_velocity` — a straight line fitted to the station's series
  over a time window (``gps_analysis.estimate_velocity``, model
  ``linear``): the windowed rate of the gps_data_analyses Reykjanes maps
  (``gpsvelo``), on the series ``gpsplot ts`` reads. On the detrended
  series (the default) it is the rate relative to the stored background.
* :func:`record_velocity` — the secular rate of the stored detrend record
  (``gps_analysis.trajectory_from_record``): the long-term rate, plate
  velocity removed.
* the gps_api velocities GeoJSON, read by
  ``gps_plot.maps.velocity_vectors_from_geojson``.

:func:`window_displacement` is the level at the window end minus the level
at its start, each a weighted mean of the first/last samples
(``gps_analysis.estimate_offset``, as the zero of every time-series plot).

Adapters only: the series come from :mod:`gps_plot.sources.gps`, the
fits from gps_analysis.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

from gps_plot.sources import gps as gps_source

#: samples averaged for each end of a displacement (cf. ZERO_SAMPLES)
END_SAMPLES = 5


@dataclasses.dataclass(frozen=True)
class StationRate:
    """One station's N/E/U value with its 1-σ (mm/yr or mm).

    ``n`` is the number of epochs it rests on (0 for a stored record).
    """

    station: str
    neu: Any
    sigma: Any
    n: int = 0


def _finite(t: Any, y: Any, sigma: Any) -> tuple[Any, Any, Any]:
    ok = np.isfinite(t) & np.all(np.isfinite(y), axis=0)
    ok &= np.all(np.isfinite(sigma) & (sigma > 0), axis=0)
    return t[ok], y[:, ok], sigma[:, ok]


def _series(sta: str, req: gps_source.SeriesRequest) -> Any:
    s = gps_source.fetch(sta, req)
    if req.ref == "detrend" and not s.detrend_applied:
        raise ValueError("no stored detrend record (the series is not detrended)")
    t = np.asarray(s.t, dtype=float)
    return _finite(t, np.asarray(s.y, dtype=float), np.asarray(s.sigma, dtype=float))


def window_velocity(
    sta: str, req: gps_source.SeriesRequest, *, min_epochs: int = 3
) -> StationRate:
    """The rate of a line fitted to ``sta``'s series over ``req``'s window.

    Raises:
        ValueError: fewer than ``min_epochs`` epochs in the window, or (with
            ``ref="detrend"``) no stored record.
    """
    from gps_analysis import estimate_velocity

    t, y, sigma = _series(sta, req)
    if t.size < min_epochs:
        raise ValueError(f"{t.size} epochs in the window (need {min_epochs})")
    est = estimate_velocity(t, y, sigma, model="linear")
    return StationRate(sta, np.asarray(est.rates), np.asarray(est.sigmas), t.size)


def window_displacement(
    sta: str, req: gps_source.SeriesRequest, *, samples: int = END_SAMPLES
) -> StationRate:
    """Level at the window end minus the level at its start [mm].

    Each level is the 1/σ-weighted mean of ``samples`` epochs; its σ the
    formal σ of that mean, ``√n / Σ(1/σᵢ)``.

    Raises:
        ValueError: fewer than ``2·samples`` epochs in the window.
    """
    from gps_analysis.baseline import estimate_offset

    t, y, sigma = _series(sta, req)
    if t.size < 2 * samples:
        raise ValueError(f"{t.size} epochs in the window (need {2 * samples})")
    level, err = [], []
    for sl in (slice(None, samples), slice(-samples, None)):
        lv = np.array(
            [float(np.asarray(estimate_offset(t[sl], y[c, sl], sigma[c, sl])))
             for c in range(3)]
        )  # fmt: skip
        level.append(lv)
        err.append(np.sqrt(samples) / np.sum(1.0 / sigma[:, sl], axis=1))
    return StationRate(sta, level[1] - level[0], np.hypot(err[0], err[1]), t.size)


def record_velocity(sta: str, req: gps_source.SeriesRequest) -> StationRate:
    """The secular rate of ``sta``'s stored detrend record (plate removed).

    Raises:
        ValueError: no record, a record in another frame, or a borrowed
            secular (the donor's rate, not this station's).
    """
    from gps_analysis import trajectory_from_record

    record = gps_source.station_record(sta, req)
    if record.get("borrowed"):
        # which groups were borrowed is not checked: refuse rather than
        # draw a donor's rate as this station's
        raise ValueError(f"record borrowed from {record['borrowed']}")
    names = list(record.get("param_names") or ())
    if "rate" not in names:
        raise ValueError(f"record has no secular rate (params {names})")
    k = names.index("rate")
    _model, fits = trajectory_from_record(record)
    by_name = {f.component: f for f in fits}
    if set(by_name) != {"north", "east", "up"}:
        raise ValueError(f"record components {sorted(map(str, by_name))}, not N/E/U")
    rows = [by_name[c] for c in ("north", "east", "up")]
    rate = np.array([float(f.params[k]) for f in rows])
    sigma = np.array([float(f.uncertainties[k]) for f in rows])
    return StationRate(sta, rate, sigma, int(record.get("n_epochs") or 0))
