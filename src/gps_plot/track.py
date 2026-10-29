"""Station tracks since an event: daily positions as dots on a map.

``gpsplot map station ... --track EVENT`` (BGÓ 2026-10-28, the Svínafellsheiði
landslide; after the time-coloured GNSS dots of Sigmundsson et al. 2015).
Per station the horizontal series has the line + seasonal of a window
before the event removed, so before the onset the dots sit on the station
and after it they walk away. The dots are coloured by days since the onset
(newest red), joined by a fine dotted path in time order. A station that
clearly moved gets two arrows with 1-σ ellipses (the rate since the onset,
blue; the last ``recent`` days, red) and its numbers in the legend; one with
too few days for a rate gets its displacement there instead.

The numbers (this module, numpy only) are kept apart from the drawing
(:func:`draw_tracks`, PyGMT), so they are tested without GMT.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

#: days of the pre-onset trend window, ending ``PRE_GAP_DAYS`` before the onset
PRE_DAYS = 3 * 365
PRE_GAP_DAYS = 7
#: fewest days in the pre-onset window for a trend
MIN_PRE_DAYS = 100
#: dots drawn from this many days before the onset
BEFORE_DAYS = 30
#: the recent rate's window [days]
RECENT_DAYS = 14
#: the dots' zero: the median of this many days before the onset
ZERO_DAYS = 14
#: fewest days for a line rate
MIN_RATE_DAYS = 5
#: the newest position: the mean of the days at most this many days before
#: the newest (two at most), not of two days a week apart
END_SPAN_DAYS = 3
#: the few-days station's displacement arrow and its 1-σ ellipse (dashed)
DISPLACEMENT_COLOR = "red"
DISPLACEMENT_ELLIPSE_PEN = "0.6p,red,4_2:0"
#: moved = the newest position off the station by more than this × its scatter
MOVED_SIGMA = 3.0
#: largest horizontal formal σ kept [mm]
TRACK_UNCERT = 10.0
#: the farthest track drawn this long on the map [cm] (automatic dot scale)
TRACK_CM = 1.2
#: the longest arrow [cm] (automatic arrow scale)
ARROW_CM = 2.0
DOT_STYLE = "c0.13c"
DOT_PEN = "0.2p,black"
PATH_PEN = "0.5p,40/40/40,1_1.5:0"
#: the dotted path follows the running median of this many days (odd; 1 or
#: 0 = the daily positions themselves): the direction without the scatter
PATH_MEDIAN_DAYS = 5
#: a reference station whose motion since the onset, against the mean of
#: the others, is above this many σ (either component) is left out
REF_REJECT_SIGMA = 5.0
ARROW_HEAD = "0.25c"
ARROW_PEN = "0.9p"
ELLIPSE_PEN = "0.6p,black"
SINCE_COLOR = "blue"
BEFORE_COLOR = "80/80/80"
RECENT_COLOR = "black"
#: the newest day in the legend: through yesterday green, else red (as the
#: time series' "Last datapoint"; a darker green, readable on white)
FRESH_COLOR = "0/160/0"
STALE_COLOR = "red"
#: the dots' scale: two dots joined by a line
DOT_PAIR_PEN = "0.8p,black"
DOT_PAIR_FILL = "gray70"
#: the track legend's box: a little less see-through than the vector maps'
TRACK_LEGEND_FILL = "white@8"
LEGEND_FONT = "7p,Helvetica-Bold"
#: the legend's line spacing [cm] (0.04 of the first 8.25 cm high map):
#: the same on a tall map as on a wide one
LEGEND_LINE_CM = 0.33
DEFAULT_CMAP = "turbo"
#: the colour bar: a fifth of the map wide, inside its upper right
CBAR_WIDTH = 0.2
CBAR_FONT = "7p,Helvetica"
CBAR_BOX = "+gwhite@15+p0.4p,gray40+c0.1c/0.1c"
#: the colour bar's box with its label, from the map's top [cm]
CBAR_BOX_HEIGHT_CM = 1.4
#: a zoom inset's longest arrow [cm] and the font of its scales
ZOOM_ARROW_CM = 1.5
INSET_FONT = "6p,Helvetica-Bold"
#: days before the onset a station's series (track_series) shows
SERIES_BEFORE_DAYS = 90
YEAR = 365.25


@dataclasses.dataclass(frozen=True)
class Rate:
    """A horizontal line rate [mm/yr] and its 1-σ (white noise, floored at the
    pre-onset daily scatter)."""

    north: float
    east: float
    sigma_north: float
    sigma_east: float

    @property
    def speed(self) -> float:
        return math.hypot(self.north, self.east)

    @property
    def azimuth(self) -> float:
        """Direction of motion [degrees clockwise from north]."""
        return math.degrees(math.atan2(self.east, self.north)) % 360

    @property
    def sigma_speed(self) -> float:
        """σ of :attr:`speed`, along the rate's direction."""
        s = self.speed
        un, ue = (self.north / s, self.east / s) if s > 0 else (1.0, 0.0)
        return math.hypot(un * self.sigma_north, ue * self.sigma_east)


@dataclasses.dataclass(frozen=True)
class Track:
    """One station's horizontal positions since ``BEFORE_DAYS`` before the
    onset [mm, trend removed, zero at the onset]."""

    marker: str
    lon: float
    lat: float
    t: FloatArray
    north: FloatArray
    east: FloatArray
    days: FloatArray  # since the onset
    scatter: float  # horizontal daily scatter before the onset [mm]
    since: Rate | None
    recent: Rate | None
    n_after: int  # days of data since the onset
    #: the trend kept (``keep_trend``): the rate in the window before the
    #: onset, relative to the reference (creep before the event)
    before: Rate | None = None
    #: the daily scatter before the onset per component (N, E) [mm]
    scatter_ne: tuple[float, float] = (0.0, 0.0)
    #: σ of the zero (N, E) [mm]: of the median of the days before the onset,
    #: or, with too few of them, of the trend extrapolated to the onset
    zero_sigma: tuple[float, float] = (0.0, 0.0)
    #: no days just before the onset: the zero is the extrapolated trend
    zero_extrapolated: bool = False

    @property
    def end_count(self) -> int:
        """Days averaged for :attr:`end`: the newest one, and the one before
        it if within ``END_SPAN_DAYS``."""
        if self.n_after == 0:
            return 0
        if self.n_after >= 2 and self.days[-1] - self.days[-2] <= END_SPAN_DAYS:
            return 2
        return 1

    @property
    def end(self) -> tuple[float, float]:
        """The newest position (N, E): the mean of the newest days after
        the onset (:attr:`end_count`)."""
        m = self.end_count
        if m == 0:
            return (0.0, 0.0)
        return (float(np.mean(self.north[-m:])), float(np.mean(self.east[-m:])))

    @property
    def moved(self) -> bool:
        """Its rate since the onset above ``MOVED_SIGMA`` σ; with too few days
        for a rate, its newest position as far off the station."""
        if self.n_after == 0:
            return False
        if self.since is not None:
            return self.since.speed > MOVED_SIGMA * self.since.sigma_speed
        return math.hypot(*self.end) > MOVED_SIGMA * self.end_sigma

    @property
    def end_sigmas(self) -> tuple[float, float]:
        """σ of :attr:`end` (N, E): the daily scatter over the days averaged
        and the zero's σ, in quadrature."""
        m = math.sqrt(max(1, self.end_count))
        daily = (
            self.scatter_ne
            if any(self.scatter_ne)
            else (self.scatter / math.sqrt(2),) * 2
        )
        n, e = (
            math.hypot(d / m, z) for d, z in zip(daily, self.zero_sigma, strict=True)
        )
        return (n, e)

    @property
    def end_sigma(self) -> float:
        """Horizontal σ of :attr:`end`."""
        return math.hypot(*self.end_sigmas)


def yearf(day: dt.date) -> float:
    """Noon of ``day`` as a fractional year (the .pos epochs' convention)."""
    start = dt.date(day.year, 1, 1)
    length = (dt.date(day.year + 1, 1, 1) - start).days
    return day.year + ((day - start).days + 0.5) / length


def day_key(t: FloatArray) -> npt.NDArray[np.int64]:
    """The calendar day (proleptic ordinal) of each fractional-year epoch,
    :func:`yearf`'s inverse to the day. ``round(t·365.25)`` is not one: in
    a leap year a day is 365.25/366 of a unit, and two days can share it."""
    year = np.floor(t).astype(np.int64)
    leap = (year % 4 == 0) & ((year % 100 != 0) | (year % 400 == 0))
    length = np.where(leap, 366, 365)
    doy = np.floor((t - year) * length).astype(np.int64)
    first = np.array([dt.date(int(y), 1, 1).toordinal() for y in year], dtype=np.int64)
    return first + doy


def from_day_key(key: npt.NDArray[np.int64]) -> FloatArray:
    """Noon of each :func:`day_key` day as a fractional year."""
    return np.array([yearf(dt.date.fromordinal(int(k))) for k in key])


def pre_window(onset: dt.date) -> tuple[dt.date, dt.date]:
    """The default trend window: ``PRE_DAYS`` up to ``PRE_GAP_DAYS`` before
    the onset."""
    end = onset - dt.timedelta(days=PRE_GAP_DAYS)
    return end - dt.timedelta(days=PRE_DAYS), end


def _design(t: FloatArray, t_ref: float) -> FloatArray:
    w = 2.0 * np.pi * t
    return np.column_stack(
        [np.ones_like(t), t - t_ref, np.cos(w), np.sin(w), np.cos(2 * w), np.sin(2 * w)]
    )


def detrend(
    t: FloatArray,
    y: FloatArray,
    pre: npt.NDArray[np.bool_],
    *,
    keep_line: bool = False,
) -> tuple[FloatArray, FloatArray]:
    """``y`` (2, N) minus the line + annual + semiannual fitted on ``pre``;
    and the per-component scatter of the fit [same unit]. ``keep_line``:
    the line is fitted with the seasonal (a seasonal fitted without it
    would be biased) but only the seasonal and the offset are removed, so
    the motion before the onset stays in."""
    if int(pre.sum()) < MIN_PRE_DAYS:
        raise ValueError(
            f"{int(pre.sum())} days in the trend window (< {MIN_PRE_DAYS})"
        )
    g = _design(t, float(t[pre].mean()))
    r = np.empty_like(y)
    rms = np.empty(y.shape[0])
    for c in range(y.shape[0]):
        m, *_ = np.linalg.lstsq(g[pre], y[c, pre], rcond=None)
        rms[c] = float(np.std(y[c, pre] - g[pre] @ m))
        if keep_line:
            m = m.copy()
            m[1] = 0.0
        r[c] = y[c] - g @ m
    return r, rms


def pre_rate(t: FloatArray, y: FloatArray, pre: npt.NDArray[np.bool_]) -> Rate:
    """The N/E rate of the line + seasonal fit on ``pre`` [mm/yr]: a
    station's motion before the onset. Its σ is the fit's white-noise σ
    times the block-mean scale of the residuals
    (:func:`gps_analysis.shared_tau.block_mean_scale`), as the transient
    product's: daily noise wanders, and three years of it are not
    independent days."""
    from gps_analysis.shared_tau import block_mean_scale

    tp = t[pre]
    g = _design(tp, float(tp.mean()))
    out = []
    for c in (0, 1):
        m, *_ = np.linalg.lstsq(g, y[c, pre], rcond=None)
        res = y[c, pre] - g @ m
        cov = np.linalg.inv(g.T @ g) * float(np.var(res, ddof=g.shape[1]))
        k = block_mean_scale(tp, res, step=1.0 / YEAR).k
        out.append((float(m[1]), k * math.sqrt(float(cov[1, 1]))))
    return Rate(out[0][0], out[1][0], out[0][1], out[1][1])


def extrapolation_sigma(
    t: FloatArray, y: FloatArray, pre: npt.NDArray[np.bool_], t_at: float
) -> FloatArray:
    """σ (N, E) of the line + seasonal fitted on ``pre`` as the position at
    ``t_at``, the zero of a station with no days just before the onset: the
    fit's prediction σ there and how far a ``ZERO_DAYS`` mean of the days
    strays from the fit (the residuals' block means before the onset), in
    quadrature. The second is what the median of the days before would
    have carried; across a long gap the first grows."""
    tp = t[pre]
    t_ref = float(tp.mean())
    g = _design(tp, t_ref)
    g0 = _design(np.array([t_at]), t_ref)[0]
    block = np.floor((tp - tp[0]) * YEAR / ZERO_DAYS)
    out = np.empty(2)
    for c in (0, 1):
        m, *_ = np.linalg.lstsq(g, y[c, pre], rcond=None)
        res = y[c, pre] - g @ m
        cov = np.linalg.inv(g.T @ g) * float(np.var(res, ddof=g.shape[1]))
        means = [res[block == b].mean() for b in np.unique(block)
                 if int((block == b).sum()) >= 5]  # fmt: skip
        wander = float(np.std(means, ddof=1)) if len(means) >= 3 else 0.0
        out[c] = math.hypot(math.sqrt(float(g0 @ cov @ g0)), wander)
    return out


def line_rate(
    t: FloatArray, r: FloatArray, a: float, b: float, floor: FloatArray
) -> Rate | None:
    """The N/E line rate over ``[a, b]`` [mm/yr]; None with fewer than
    ``MIN_RATE_DAYS`` days. Its σ uses the larger of the fit's scatter and
    ``floor`` (the pre-onset scatter): a few days can fit a line too well."""
    k = (t >= a) & (t <= b)
    if int(k.sum()) < MIN_RATE_DAYS:
        return None
    tm = t[k] - t[k].mean()
    sxx = float(np.sum(tm**2))
    out = []
    for c in (0, 1):
        p = np.polyfit(tm, r[c, k], 1)
        res = r[c, k] - np.polyval(p, tm)
        noise = max(float(np.std(res, ddof=2)), float(floor[c]))
        out.append((float(p[0]), noise / math.sqrt(sxx)))
    return Rate(out[0][0], out[1][0], out[0][1], out[1][1])


def wander_scale(
    t: FloatArray, r: FloatArray, pre: npt.NDArray[np.bool_], days: float
) -> FloatArray:
    """How much more a rate over ``days`` days scatters than white daily
    noise allows, per component (N, E), at least 1: the block-mean scale
    (:func:`gps_analysis.shared_tau.block_mean_scale`) of the residuals
    ``r`` before the onset, blocks of ``days``/8 days (eight blocks to a
    window, the transient product's rule). Noise that wanders over days
    makes a short window's line steeper than white noise would."""
    from gps_analysis.shared_tau import BLOCK_MIN_BLOCKS, block_mean_scale

    n = max(2, round(days / BLOCK_MIN_BLOCKS))
    return np.array([block_mean_scale(t[pre], r[c, pre], step=1.0 / YEAR, n=n).k
                     for c in (0, 1)])  # fmt: skip


def scaled(rate: Rate | None, k: FloatArray) -> Rate | None:
    """``rate`` with its σ times ``k`` (N, E)."""
    if rate is None:
        return None
    return dataclasses.replace(rate, sigma_north=rate.sigma_north * float(k[0]),
                               sigma_east=rate.sigma_east * float(k[1]))  # fmt: skip


def running_median(y: FloatArray, n: int) -> FloatArray:
    """The centred running median of ``n`` points (odd; shorter at the
    ends); ``n`` <= 1: ``y`` itself."""
    if n <= 1 or y.size == 0:
        return y
    half = n // 2
    return np.array(
        [np.median(y[max(0, i - half) : i + half + 1]) for i in range(y.size)]
    )


def relative_to(
    t: FloatArray, y: FloatArray, t_ref: FloatArray, y_ref: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """``y`` minus the reference station's ``y_ref`` on the days both have
    (matched to the day): the station's motion relative to the reference,
    a baseline, so motion they share (the network's, the plate's) drops
    out."""
    common, i, j = np.intersect1d(day_key(t), day_key(t_ref), return_indices=True)
    return t[i], y[:, i] - y_ref[:, j]


def reference_mean(
    series: Sequence[tuple[FloatArray, FloatArray]],
    pre: tuple[dt.date, dt.date],
    *,
    keep_trend: bool = False,
) -> tuple[FloatArray, FloatArray]:
    """One reference from several stations: each one's trend and seasonal
    (fitted on ``pre``) removed, then their weighted mean per day (weights
    1/scatter², the fit's per component), on the days at least half of
    them have. Like GLOBK's frame realisation by many reference stations,
    translation only: one station's own wander (a bad day, a local
    motion) moves the reference by its share, not wholly. ``keep_trend``:
    only the seasonal removed (:func:`detrend` ``keep_line``), so the
    mean carries the stations' common motion, the local frame."""
    if not series:
        raise ValueError("no reference stations")
    a, b = yearf(pre[0]), yearf(pre[1])
    days, values, weights = [], [], []
    for t, y in series:
        r, rms = detrend(t, y, (t >= a) & (t < b), keep_line=keep_trend)
        days.append(day_key(t))
        values.append(r)
        weights.append(1.0 / np.maximum(rms, 1e-3) ** 2)
    grid = np.unique(np.concatenate(days))
    total = np.zeros((2, grid.size))
    wsum = np.zeros((2, grid.size))
    count = np.zeros(grid.size)
    for d, r, w in zip(days, values, weights, strict=True):
        i = np.searchsorted(grid, d)
        total[:, i] += w[:, None] * r
        wsum[:, i] += w[:, None]
        count[i] += 1
    keep = count >= math.ceil(len(series) / 2)
    return from_day_key(grid[keep]), total[:, keep] / wsum[:, keep]


def screen_references(
    series: Mapping[str, tuple[FloatArray, FloatArray]],
    pre: tuple[dt.date, dt.date],
    onset: dt.date,
    *,
    n_sigma: float = REF_REJECT_SIGMA,
) -> tuple[list[str], list[tuple[str, float]]]:
    """Leave out reference stations that move: GLOBK's n-σ deletion of
    reference sites, one at a time. Each station's own rate since the onset
    (its trend and seasonal before removed) is compared with the median
    rate of the stations kept, in units of √(σ² + median σ²); the worst one
    above ``n_sigma`` (either component) goes and the rest are checked
    again, while three or more are left (with two, which one moved cannot
    be told). A median, not a mean, so the mover cannot pull the yardstick.
    Returns the kept names and ``(name, its speed against the median
    [mm/yr])`` of those left out."""
    t_on = yearf(onset)
    a, b = yearf(pre[0]), yearf(pre[1])
    rates: dict[str, Rate] = {}
    for name, (t, y) in series.items():
        r, rms = detrend(t, y, (t >= a) & (t < b))
        rate = line_rate(t, r, t_on, float(t[-1]), rms) if t.size else None
        if rate is not None:
            rates[name] = rate
    kept = list(series)
    out: list[tuple[str, float]] = []
    while len(kept) >= 3:
        have = [rates[k] for k in kept if k in rates]
        if len(have) < 3:
            break
        mid = np.median([[q.north, q.east] for q in have], axis=0)
        sig = np.median([[q.sigma_north, q.sigma_east] for q in have], axis=0)
        worst, worst_z, worst_v = None, n_sigma, 0.0
        for name in kept:
            q = rates.get(name)
            if q is None:
                continue
            d = np.array([q.north, q.east]) - mid
            s = np.hypot([q.sigma_north, q.sigma_east], sig)
            z = float(np.max(np.abs(d) / s))
            if z > worst_z:
                worst, worst_z, worst_v = name, z, float(np.hypot(*d))
        if worst is None:
            break
        kept.remove(worst)
        out.append((worst, worst_v))
    return kept, out


def station_track(
    marker: str,
    lon: float,
    lat: float,
    t: FloatArray,
    y: FloatArray,
    onset: dt.date,
    pre: tuple[dt.date, dt.date],
    *,
    before_days: int = BEFORE_DAYS,
    recent_days: int = RECENT_DAYS,
    keep_trend: bool = False,
) -> Track:
    """A station's :class:`Track` from its horizontal series ``y`` (N, E)
    [mm] at epochs ``t`` [fractional years, finite]. ``keep_trend``: only
    the seasonal before the onset removed, its trend kept and measured
    (:attr:`Track.before`)."""
    t_on = yearf(onset)
    in_pre = (t >= yearf(pre[0])) & (t < yearf(pre[1]))
    r, rms = detrend(t, y, in_pre, keep_line=keep_trend)
    before = pre_rate(t, y, in_pre) if keep_trend else None
    near = (t >= t_on - ZERO_DAYS / YEAR) & (t < t_on)
    n_near = int(near.sum())
    if n_near >= 3:
        zero = np.median(r[:, near], axis=1)
        zero_sigma = 1.25 * rms / math.sqrt(n_near)  # σ of a median
    else:
        zero = np.zeros(2)
        zero_sigma = extrapolation_sigma(t, y, in_pre, t_on)
    r = r - zero[:, None]
    shown = t >= t_on - before_days / YEAR
    if not shown.any() or t[-1] < t_on:
        raise ValueError(f"no data since {onset:%Y-%m-%d}")
    resid = detrend(t, y, in_pre)[0] if keep_trend else r
    since = scaled(
        line_rate(t, r, t_on, float(t[-1]), rms),
        wander_scale(t, resid, in_pre, (float(t[-1]) - t_on) * YEAR),
    )
    recent = scaled(
        line_rate(t, r, float(t[-1]) - recent_days / YEAR, float(t[-1]), rms),
        wander_scale(t, resid, in_pre, recent_days),
    )  # fmt: skip
    return Track(
        marker, lon, lat, t[shown], r[0, shown], r[1, shown],
        (t[shown] - t_on) * YEAR, float(math.hypot(*rms)), since, recent,
        int((t >= t_on).sum()), before, (float(rms[0]), float(rms[1])),
        (float(zero_sigma[0]), float(zero_sigma[1])), n_near < 3,
    )  # fmt: skip


def load_series(
    marker: str,
    pre: tuple[dt.date, dt.date],
    *,
    frame: str = "itrf14",
    pos_dir: str | None = None,
    uncert: float = TRACK_UNCERT,
    common_mode: bool = True,
    end: dt.date | None = None,
) -> tuple[FloatArray, FloatArray]:
    """A station's daily horizontal series (cleaned view, plate removed)
    from the trend window on: ``(t, y)``, ``y`` (N, E) [mm].

    Screened on the horizontal formal σ only (a map of horizontal motion
    keeps a day whose Up σ is large); the daily network common mode is
    subtracted where it was estimated."""
    from geo_dataread.gps_views import resolve_common_mode, subtract_common_mode
    from geo_dataread.products.loader import Source, load
    from geo_dataread.products.velocity_core import window_bounds

    t0, t1 = window_bounds(pre[0], end or dt.date.today())
    v = load(marker, t0, t1, Source(frame=frame, view="cleaned", ref="plate",
                                    uncert=1e9, pos_dir=pos_dir))  # fmt: skip
    ok = np.all(np.isfinite(v.y[:2]), axis=0) & (np.max(v.sigma[:2], axis=0) <= uncert)
    t, y = v.t[ok], v.y[:2, ok]
    if common_mode:
        epochs, values, _src = resolve_common_mode(pos_dir if pos_dir else None)
        if epochs.size:
            y, _n = subtract_common_mode(t, y, epochs, values[:2])
    return t, y


def load_track(
    marker: str,
    lon: float,
    lat: float,
    onset: dt.date,
    pre: tuple[dt.date, dt.date],
    *,
    frame: str = "itrf14",
    pos_dir: str | None = None,
    uncert: float = TRACK_UNCERT,
    common_mode: bool = True,
    before_days: int = BEFORE_DAYS,
    recent_days: int = RECENT_DAYS,
    end: dt.date | None = None,
    reference: tuple[FloatArray, FloatArray] | None = None,
    keep_trend: bool = False,
) -> Track:
    """Read a station (:func:`load_series`) and make its track; with
    ``reference`` (another station's ``load_series``) relative to it."""
    t, y = load_series(marker, pre, frame=frame, pos_dir=pos_dir, uncert=uncert,
                       common_mode=common_mode, end=end)  # fmt: skip
    if reference is not None:
        t, y = relative_to(t, y, *reference)
    return station_track(marker, lon, lat, t, y, onset, pre,
                         before_days=before_days, recent_days=recent_days,
                         keep_trend=keep_trend)  # fmt: skip


def legend_entries(
    tracks: Sequence[Track], onset: dt.date, recent_days: int
) -> list[tuple[str, str]]:
    """``(text, colour)`` per legend line, top down: per moved station its
    rate since the onset (blue) and recent rate (red), or its displacement
    (black) when it has too few days for a rate; with the trend kept, its
    rate before the onset (grey)."""
    pm, deg = "±", "°"
    out: list[tuple[str, str]] = []
    for tr in tracks:
        if tr.n_after == 0 or (tr.since is not None and not tr.moved):
            continue
        if tr.since is not None:
            s = tr.since
            moved_mm = math.hypot(*tr.end)
            out.append((f"{tr.marker} {s.speed:.0f} {pm} {s.sigma_speed:.0f} mm/yr "
                        f"toward {s.azimuth:.0f}{deg} since {onset:%d.%m.%Y}, "
                        f"moved {moved_mm:.0f} {pm} {tr.end_sigma:.0f} mm",
                        SINCE_COLOR))  # fmt: skip
            if tr.recent is not None:
                r, f = tr.recent, recent_days / YEAR
                out.append((f"{tr.marker} {r.speed:.0f} {pm} {r.sigma_speed:.0f} "
                            f"mm/yr last {recent_days} days, moved "
                            f"{r.speed * f:.0f} {pm} {r.sigma_speed * f:.0f} mm",
                            RECENT_COLOR))  # fmt: skip
            if tr.before is not None:
                out.append((before_text(tr.marker, tr.before), BEFORE_COLOR))
        else:
            n, e = tr.end
            az = math.degrees(math.atan2(e, n)) % 360
            gap = ", none just before" if tr.zero_extrapolated else ""
            out.append((f"{tr.marker} moved {math.hypot(n, e):.0f} {pm} "
                        f"{tr.end_sigma:.0f} mm toward {az:.0f}{deg} since "
                        f"{onset:%d.%m} ({tr.n_after} day{'s' if tr.n_after != 1 else ''}"
                        f"{gap})",
                        DISPLACEMENT_COLOR))  # fmt: skip
    return out


def newest_day(tracks: Sequence[Track]) -> dt.date | None:
    """The day of the newest position of any of ``tracks``."""
    ends = [float(tr.t[-1]) for tr in tracks if tr.t.size]
    if not ends:
        return None
    return dt.date.fromordinal(int(day_key(np.array([max(ends)]))[0]))


def newest_text(day: dt.date, today: dt.date | None = None) -> tuple[str, str]:
    """The legend's newest-day text and its colour: green when the newest
    day is yesterday or today (the time series' "Last datapoint" rule),
    else red (stale)."""
    today = today or dt.date.today()
    fresh = day >= today - dt.timedelta(days=1)
    return f"newest {day:%d.%m.%Y}", (FRESH_COLOR if fresh else STALE_COLOR)


def draw_dot_pair(fig: Any, x: float, y: float, length_deg: float, *,
                  style: str = DOT_STYLE) -> None:  # fmt: skip
    """The dots' scale: two dots ``length_deg`` apart (a reference length
    in mm at the dots' scale), joined by a line."""
    fig.plot(x=[x, x + length_deg], y=[y, y], pen=DOT_PAIR_PEN)
    fig.plot(x=[x, x + length_deg], y=[y, y], style=style, fill=DOT_PAIR_FILL,
             pen=DOT_PEN)  # fmt: skip


def before_text(marker: str, rate: Rate) -> str:
    """The legend's line for a rate before the onset: north and east, not
    a speed (a small speed is biased by its noise)."""
    pm = "±"
    return (f"{marker} before: N {rate.north:.1f} {pm} {rate.sigma_north:.1f}, "
            f"E {rate.east:.1f} {pm} {rate.sigma_east:.1f} mm/yr")  # fmt: skip


def nice(x: float) -> float:
    """The 1-2-5 number at or below ``x`` (> 0)."""
    k = math.floor(math.log10(x))
    for m in (5, 2, 1):
        if m * 10**k <= x:
            return float(m * 10**k)
    return float(10**k)


def dot_scale(tracks: Sequence[Track], km_per_cm: float) -> tuple[float, float]:
    """``(m_per_mm, ref_mm)``: the farthest position of a moved station (else
    of any) drawn ``TRACK_CM`` from its station, and a round reference."""
    moved = [tr for tr in tracks if tr.moved] or list(tracks)
    far = max((float(np.max(np.hypot(tr.north, tr.east))) for tr in moved
               if tr.north.size), default=10.0)  # fmt: skip
    far = max(far, 1.0)
    return TRACK_CM * km_per_cm * 1000.0 / far, nice(far / 2)


def offset_lonlat(lon: float, lat: float, north_mm: Any, east_mm: Any,
                  m_per_mm: float) -> tuple[FloatArray, FloatArray]:  # fmt: skip
    """Map positions of displacements drawn ``m_per_mm`` metres per mm."""
    k = m_per_mm / 1000.0 / 111.195
    n = np.asarray(north_mm, dtype=float)
    e = np.asarray(east_mm, dtype=float)
    return lon + e * k / max(math.cos(math.radians(lat)), 0.01), lat + n * k


@dataclasses.dataclass(frozen=True)
class Drawn:
    """What :func:`draw_tracks` put on the map, for the insets to keep off:
    the legend's top [cm above the bottom edge], the legend and colour bar
    boxes (x0, x1, y0, y1) [cm] and the dots' and arrow tips' positions."""

    legend_top: float
    boxes: tuple[tuple[float, float, float, float], ...]
    lons: FloatArray
    lats: FloatArray
    cpt: str | None = None  # the dots' colour scale, for a zoom inset


def km_per_cm(region: Sequence[float], width_cm: float) -> float:
    """The map's horizontal scale [km per cm] at its middle latitude."""
    w, e, s, n = region[:4]
    return (e - w) * 111.32 * math.cos(math.radians((s + n) / 2)) / width_cm


def arrow_scale_of(tracks: Sequence[Track], length_cm: float = ARROW_CM
                   ) -> tuple[float, float]:  # fmt: skip
    """``(cm per mm/yr, round reference [mm/yr])``: the fastest moved
    station's rate since the onset drawn ``length_cm`` long."""
    speeds = [tr.since.speed for tr in tracks if tr.moved and tr.since is not None]
    if not speeds:
        return 0.01, 100.0
    return length_cm / max(speeds), nice(max(speeds) / 2)


def draw_track_marks(
    fig: Any,
    tracks: Sequence[Track],
    *,
    m_per_mm: float,
    arrow_scale: float,
    arrows: bool = True,
    path_days: int = PATH_MEDIAN_DAYS,
    region: Sequence[float] | None = None,
    width_cm: float | None = None,
    cpt: str | None = None,
    cm_per_mm: float | None = None,
) -> tuple[FloatArray, FloatArray]:
    """The dotted paths, the dots (coloured by ``cpt``, else the current
    CPT) and the
    arrows of ``tracks``; on the map or in a zoom inset at its own scale.
    A station with too few days for a rate gets a displacement arrow to
    its newest position, at the dots' scale (``cm_per_mm``), with a dashed
    1-σ ellipse. Returns the dots' and (with ``region`` and ``width_cm``)
    the arrow tips' positions (lon, lat)."""
    from gps_plot import maps

    lons, lats, days = [], [], []
    for tr in tracks:
        x, y = offset_lonlat(tr.lon, tr.lat, tr.north, tr.east, m_per_mm)
        lons.append(x)
        lats.append(y)
        days.append(tr.days)
        pn, pe = running_median(tr.north, path_days), running_median(tr.east, path_days)
        px, py = offset_lonlat(tr.lon, tr.lat, pn, pe, m_per_mm)
        fig.plot(x=px, y=py, pen=PATH_PEN)
    x = np.concatenate(lons) if lons else np.zeros(0)
    y = np.concatenate(lats) if lats else np.zeros(0)
    z = np.concatenate(days) if days else np.zeros(0)
    if z.size:
        order = np.argsort(z, kind="stable")  # newest on top
        fig.plot(x=x[order], y=y[order], fill=z[order], cmap=cpt or True,
                 style=DOT_STYLE, pen=DOT_PEN)  # fmt: skip
    moved = [tr for tr in tracks if tr.moved and tr.since is not None]
    tips_x, tips_y = [], []
    if arrows and moved:
        vecs = []
        for tr in moved:
            for tag, rt in (("since", tr.since), ("recent", tr.recent)):
                if rt is not None:
                    vecs.append(maps.VelocityVector(
                        tr.marker, tr.lon, tr.lat, east=rt.east, north=rt.north,
                        sigma_east=rt.sigma_east, sigma_north=rt.sigma_north,
                        method=tag))  # fmt: skip
        maps.draw_velocity_vectors(
            fig, vecs, colors={"since": SINCE_COLOR, "recent": RECENT_COLOR},
            scale=arrow_scale, confidence=0.39, vector_head=ARROW_HEAD,
            pen_width=ARROW_PEN, ellipse_pen=ELLIPSE_PEN, markers=False,
            labels=False,
        )  # fmt: skip
        if region is not None and width_cm:
            deg_per_cm = (region[1] - region[0]) / width_cm
            for v in vecs:
                for f in (0.5, 1.0):
                    tips_x.append(v.lon + f * v.east * arrow_scale * deg_per_cm)
                    tips_y.append(v.lat + f * v.north * arrow_scale * deg_per_cm
                                  * math.cos(math.radians(v.lat)))  # fmt: skip
    few = [tr for tr in tracks if tr.since is None and tr.n_after > 0]
    if arrows and few and cm_per_mm:
        disp = []
        for tr in few:
            (n, e), (sn, se) = tr.end, tr.end_sigmas
            disp.append(maps.VelocityVector(
                tr.marker, tr.lon, tr.lat, east=e, north=n, sigma_east=se,
                sigma_north=sn, method="moved"))  # fmt: skip
        maps.draw_velocity_vectors(
            fig, disp, colors={"moved": DISPLACEMENT_COLOR}, scale=cm_per_mm,
            confidence=0.39, vector_head=ARROW_HEAD, pen_width=ARROW_PEN,
            ellipse_pen=DISPLACEMENT_ELLIPSE_PEN, markers=False, labels=False,
        )  # fmt: skip
    return np.concatenate([x, tips_x]), np.concatenate([y, tips_y])


def colour_range(tracks: Sequence[Track]) -> tuple[float, float]:
    """The dots' colour scale [days since the onset]: from the first dot
    (at least a day before), at least a day long."""
    z = np.concatenate([tr.days for tr in tracks]) if tracks else np.zeros(1)
    lo = float(min(z.min(), -1.0)) if z.size else -1.0
    return lo, max(float(z.max()) if z.size else 0.0, lo + 1.0)


def draw_tracks(
    fig: Any,
    tracks: Sequence[Track],
    *,
    onset: dt.date,
    region: Sequence[float],
    projection: str,
    recent_days: int = RECENT_DAYS,
    m_per_mm: float | None = None,
    arrows: bool = True,
    arrow_scale: float | None = None,
    cmap: str = DEFAULT_CMAP,
    legend_floor_cm: float = 1.0,
    path_days: int = PATH_MEDIAN_DAYS,
    note: str | None = None,
) -> Drawn:
    """Draw the tracks, arrows, legend and colour bar on ``fig``.

    The legend sits in the lower left, its foot ``legend_floor_cm`` above the
    map's bottom edge (clear of the longitude annotations). The colour scale
    is kept in a file (:attr:`Drawn.cpt`): a GMT inset does not see the
    figure's current CPT, and the zoom inset colours its dots alike.
    """
    from gps_plot import maps

    pygmt = maps._import_pygmt()
    width, height = maps._map_size_cm(region, projection)
    auto_m, dot_ref = dot_scale(tracks, km_per_cm(region, width))
    m_per_mm = m_per_mm or auto_m
    if m_per_mm != auto_m:
        far = TRACK_CM * km_per_cm(region, width) * 1000.0 / m_per_mm
        dot_ref = nice(max(far / 2, 0.5))
    auto_scale, ref = arrow_scale_of(tracks)
    scale = arrow_scale or auto_scale

    boxes: list[tuple[float, float, float, float]] = []
    cpt = None
    if tracks:
        cpt = _cpt_file()
        pygmt.makecpt(cmap=cmap, series=list(colour_range(tracks)), output=cpt)
    lons, lats = draw_track_marks(
        fig, tracks, m_per_mm=m_per_mm, arrow_scale=scale, arrows=arrows,
        path_days=path_days, region=region, width_cm=width, cpt=cpt,
        cm_per_mm=m_per_mm / (km_per_cm(region, width) * 1000.0),
    )  # fmt: skip
    if tracks:
        # small, inside the upper right, on a white box (BGÓ 2026-10-28)
        with pygmt.config(FONT_ANNOT_PRIMARY=CBAR_FONT, FONT_LABEL=CBAR_FONT,
                          MAP_ANNOT_OFFSET_PRIMARY="1p", MAP_LABEL_OFFSET="2p",
                          MAP_TICK_LENGTH_PRIMARY="2p"):  # fmt: skip
            fig.colorbar(
                cmap=cpt,
                frame=[f"xaf+ldays since {onset:%d.%m.%Y}"],
                position=f"JTR+jTR+o0.45c/0.35c+w{CBAR_WIDTH * width:.1f}c/0.2c+h",
                box=CBAR_BOX,
            )
        boxes.append((width - 0.45 - CBAR_WIDTH * width - 0.4, width,
                      height - CBAR_BOX_HEIGHT_CM, height))  # fmt: skip

    moved = any(tr.moved and tr.since is not None for tr in tracks)
    entries = legend_entries(tracks, onset, recent_days)
    if note:  # e.g. the reference station: what the motion is relative to
        entries = [(note, "black"), *entries]
    top, right = _draw_legend(
        fig, entries, region=region,
        projection=projection, arrow_ref=ref if arrows and moved else None,
        arrow_scale=scale, dot_ref=dot_ref, m_per_mm=m_per_mm,
        floor_cm=legend_floor_cm, newest=newest_day(tracks),
    )  # fmt: skip
    boxes.append((0.0, right, 0.0, top))
    return Drawn(top, tuple(boxes), lons, lats, cpt)


def _cpt_file() -> str:
    """A temporary .cpt file, removed when the program ends."""
    import atexit
    import os
    import tempfile

    fd, path = tempfile.mkstemp(prefix="gpsplot-track-", suffix=".cpt")
    os.close(fd)

    def remove() -> None:
        if os.path.exists(path):
            os.remove(path)

    atexit.register(remove)
    return path


def draw_inset_scales(
    fig: Any,
    box: Sequence[float],
    *,
    dot_ref: float,
    m_per_mm: float,
    arrow_ref: float | None,
    arrow_scale: float,
) -> None:
    """A zoom inset's own scales, lower left: the dot bar and, when arrows
    are drawn, the reference arrow."""
    from gps_plot import maps

    font = f"{INSET_FONT},black"
    x, y = maps.region_point(box, 0.04, 0.07)
    bar = dot_ref * m_per_mm / 1000.0 / (111.195 * math.cos(math.radians(y)))
    draw_dot_pair(fig, x, y, bar, style="c0.1c")
    xl, yl = maps.region_point(box, 0.04, 0.13)
    fig.text(x=xl, y=yl, text=f"{dot_ref:g} mm (dots)", font=font, justify="LM",
             fill="white")  # fmt: skip
    if arrow_ref is None:
        return
    x, y = maps.region_point(box, 0.04, 0.22)
    maps._draw_vector_layer(
        fig, np.array([x]), np.array([y]), np.array([float(arrow_ref)]),
        np.array([0.0]), color="black", scale=arrow_scale, confidence=0.39,
        head=ARROW_HEAD, pen_width=ARROW_PEN,
    )  # fmt: skip
    xl, yl = maps.region_point(box, 0.04, 0.28)
    fig.text(x=xl, y=yl, text=f"{arrow_ref:g} mm/yr", font=font, justify="LM",
             fill="white")  # fmt: skip


def draw_zoom_tracks(
    fig: Any,
    tracks: Sequence[Track],
    box: Sequence[float],
    width_cm: float,
    *,
    arrows: bool = True,
    path_days: int = PATH_MEDIAN_DAYS,
    cpt: str | None = None,
) -> None:
    """The tracks inside a zoom inset's ``box`` (``width_cm`` wide), at
    the inset's own dot and arrow scales, with those scales drawn; the
    dots coloured by the map's ``cpt``."""
    w, e, s, n = box[:4]
    inside = [tr for tr in tracks if w <= tr.lon <= e and s <= tr.lat <= n]
    if not inside:
        return
    m_per_mm, dot_ref = dot_scale(inside, km_per_cm(box, width_cm))
    scale, ref = arrow_scale_of(inside, ZOOM_ARROW_CM)
    draw_track_marks(fig, inside, m_per_mm=m_per_mm, arrow_scale=scale,
                     arrows=arrows, path_days=path_days, cpt=cpt,
                     cm_per_mm=m_per_mm / (km_per_cm(box, width_cm) * 1000.0))  # fmt: skip
    moved = any(tr.moved and tr.since is not None for tr in inside)
    draw_inset_scales(fig, box, dot_ref=dot_ref, m_per_mm=m_per_mm,
                      arrow_ref=ref if arrows and moved else None,
                      arrow_scale=scale)  # fmt: skip


def _draw_legend(
    fig: Any,
    entries: Sequence[tuple[str, str]],
    *,
    region: Sequence[float],
    projection: str,
    arrow_ref: float | None,
    arrow_scale: float,
    dot_ref: float,
    m_per_mm: float,
    floor_cm: float,
    newest: dt.date | None = None,
) -> tuple[float, float]:
    """The lower-left legend box: on top the arrow reference (when arrows are
    drawn), the dot scale and the ``newest`` day side by side, under them
    ``entries`` (top down). Returns its top and right edge [cm from the
    map's lower left]."""
    from gps_plot import maps

    width, height = maps._map_size_cm(region, projection)
    pad, sp = maps.LEGEND_BOX_PAD_CM, LEGEND_LINE_CM / height
    anchor = (0.02, 0.5 * sp + (pad + floor_cm) / height)
    n = len(entries)

    def at(fx: float, line: float) -> tuple[float, float]:
        return maps.region_point(region, anchor[0] + fx / width, anchor[1] + line * sp)

    ref_text = f"{arrow_ref:g} mm/yr" if arrow_ref is not None else ""
    dot_text = f"{dot_ref:g} mm (dots)"
    arrow_cm = arrow_ref * arrow_scale if arrow_ref is not None else 0.0
    dot_x = (max(arrow_cm, maps.text_width_cm(ref_text, LEGEND_FONT)) + 1.0
             if arrow_ref is not None else 0.0)  # fmt: skip
    bar_cm = dot_ref * m_per_mm / (km_per_cm(region, width) * 1000.0)
    new_x = dot_x + max(bar_cm, maps.text_width_cm(dot_text, LEGEND_FONT)) + 1.0
    new = newest_text(newest) if newest is not None else None
    top_wide = (new_x + maps.text_width_cm(new[0], LEGEND_FONT) if new
                else dot_x + maps.text_width_cm(dot_text, LEGEND_FONT))  # fmt: skip
    lines = n + 1.8
    box = maps.vector_legend_box(
        region, projection, anchor, sp, lines,
        [label for label, _ in entries] + [" " * 2], LEGEND_FONT, top_wide,
    )  # fmt: skip
    fig.plot(x=box[0], y=box[1], fill=TRACK_LEGEND_FILL, pen=maps.LEGEND_BOX_PEN,
             close=True)  # fmt: skip
    for i, (label, color) in enumerate(reversed(entries)):  # bottom up
        x, y = at(0.0, i)
        fig.text(x=x, y=y, text=label, font=f"{LEGEND_FONT},{color}", justify="LM")
    if arrow_ref is not None:
        x, y = at(0.0, n)
        maps._draw_vector_layer(
            fig, np.array([x]), np.array([y]), np.array([float(arrow_ref)]),
            np.array([0.0]), color="black", scale=arrow_scale, confidence=0.39,
            head=ARROW_HEAD, pen_width=ARROW_PEN,
        )  # fmt: skip
        x, y = at(0.0, n + 0.8)
        fig.text(x=x, y=y, text=ref_text, font=f"{LEGEND_FONT},black", justify="LM")
    x0, y0 = at(dot_x, n)
    bar_deg = dot_ref * m_per_mm / 1000.0 / (111.195 * math.cos(math.radians(y0)))
    draw_dot_pair(fig, x0, y0, bar_deg)
    x, y = at(dot_x, n + 0.8)
    fig.text(x=x, y=y, text=dot_text, font=f"{LEGEND_FONT},black", justify="LM")
    if new is not None:
        x, y = at(new_x, n + 0.4)
        fig.text(x=x, y=y, text=new[0], font=f"{LEGEND_FONT},{new[1]}", justify="LM")
    w = region[0]
    right = (max(box[0]) - w) / (region[1] - w) * width
    return (anchor[1] + (lines - 0.5) * sp) * height + pad, right


def trimmed(tr: Track, before_days: int) -> Track:
    """``tr`` with only the days from ``before_days`` before the onset (the
    rates and the zero are the whole track's: they do not depend on it)."""
    keep = tr.days >= -before_days
    return dataclasses.replace(
        tr, t=tr.t[keep], north=tr.north[keep], east=tr.east[keep],
        days=tr.days[keep],
    )  # fmt: skip


def rate_line(
    tr: Track, comp: int, rate: Rate, first_day: float
) -> tuple[FloatArray, FloatArray]:
    """(days, mm) of ``rate`` drawn over the days since ``first_day``: the
    line through the window's mean, as :func:`line_rate` fitted it."""
    y = (tr.north, tr.east)[comp]
    k = tr.days >= first_day
    slope = (rate.north, rate.east)[comp] / YEAR  # mm/day
    d = tr.days[k]
    mid = float(d.mean())
    ends = np.array([float(d[0]), float(d[-1])])
    return ends, float(y[k].mean()) + slope * (ends - mid)


def series_figure(
    tr: Track,
    onset: dt.date,
    *,
    recent_days: int = RECENT_DAYS,
    before_days: int = BEFORE_DAYS,
    days_range: tuple[float, float] | None = None,
    cmap: str = DEFAULT_CMAP,
    common_mode: bool = True,
    reference: str | None = None,
    keep_trend: bool = False,
) -> Any:
    """A station's track as a time series: North and East (the trend and
    seasonal before the onset removed, zero at the onset, as on the map),
    with the rate since the onset (blue) and of the last ``recent_days``
    (red) drawn on the data: the numbers of the map's legend. Days the map
    shows as dots are coloured as there (``days_range`` = the map's colour
    scale), earlier ones grey."""
    import matplotlib.figure

    fig = matplotlib.figure.Figure(figsize=(9, 6.5), layout="constrained")
    axes = fig.subplots(2, 1, sharex=True)
    t0 = dt.datetime.combine(onset, dt.time(12))
    when = [t0 + dt.timedelta(days=float(d)) for d in tr.days]
    lo, hi = days_range or (-float(before_days), float(max(tr.days.max(), 1.0)))
    on_map = tr.days >= -before_days
    rates = ((tr.since, SINCE_COLOR, 0.0, f"since {onset:%d.%m}"),
             (tr.recent, RECENT_COLOR, float(tr.days[-1]) - recent_days,
              f"last {recent_days} days"))  # fmt: skip
    for comp, (ax, name) in enumerate(zip(axes, ("North", "East"), strict=True)):
        y = (tr.north, tr.east)[comp]
        ax.plot(when, y, color="0.6", lw=0.5, ls=":", zorder=1)
        ax.scatter([w for w, m in zip(when, on_map, strict=True) if not m],
                   y[~on_map], s=12, color="0.6", zorder=2)  # fmt: skip
        dots = ax.scatter([w for w, m in zip(when, on_map, strict=True) if m],
                          y[on_map], c=tr.days[on_map], cmap=cmap, vmin=lo, vmax=hi,
                          s=18, edgecolors="black", linewidths=0.3, zorder=3)  # fmt: skip
        ax.axvline(t0, color="0.3", lw=0.8, ls="--", zorder=0)
        text = []
        for rate, color, first, label in rates:
            if rate is None:
                continue
            days, mm = rate_line(tr, comp, rate, first)
            ax.plot([t0 + dt.timedelta(days=float(d)) for d in days], mm,
                    color=color, lw=2, zorder=4)  # fmt: skip
            v = (rate.north, rate.east)[comp]
            sv = (rate.sigma_north, rate.sigma_east)[comp]
            text.append((f"{label}: v = {v:.0f} ± {sv:.0f} mm/yr", color))
        if tr.since is None:
            text.append((f"{tr.n_after} days since {onset:%d.%m}: too few for a "
                         "rate", "black"))  # fmt: skip
        if tr.before is not None:
            v = (tr.before.north, tr.before.east)[comp]
            sv = (tr.before.sigma_north, tr.before.sigma_east)[comp]
            text.append((f"before the onset: v = {v:.1f} ± {sv:.1f} mm/yr", "0.3"))
        for i, (line, color) in enumerate(text):
            ax.text(0.01, 0.95 - 0.09 * i, line, transform=ax.transAxes,
                    color=color, va="top", fontsize=10)  # fmt: skip
        ax.set_ylabel(f"{name} [mm]")
        ax.grid(True, color="0.85", lw=0.5)
    fig.colorbar(dots, ax=axes, label=f"days since {onset:%d.%m.%Y}",
                 shrink=0.6, aspect=30)  # fmt: skip
    rel = f" relative to {reference}" if reference else ""
    removed = ("seasonal before the onset removed, its trend kept" if keep_trend
               else "trend and seasonal before the onset removed")  # fmt: skip
    # two lines: a long reference name must not run off the figure
    fig.suptitle(f"{tr.marker}: horizontal motion{rel}\n{removed}", fontsize=12)
    if common_mode:
        axes[1].text(0.99, 0.02, "network common mode removed",
                     transform=axes[1].transAxes, ha="right", va="bottom",
                     fontsize=8, color="0.3")  # fmt: skip
    fig.autofmt_xdate()
    return fig
