"""``gpsplot polar --rtk``: the RTK tail on the daily polar figure.

The figure is the daily one (:mod:`gps_plot.polar`): the daily (24 h)
points since the onset, the movement since the onset, the 14-day arrow.
On top of it (BGÓ 2026-11-12):

* the newest ``--rtk-days`` of the RTK baseline, fixed solutions only
  (RTKLIB Q = 1: float and single dropped), as hourly (or 10-minute)
  medians, moved onto the daily series (:func:`align_shift`): the RTK
  continues the daily path to now;
* the RTK rate over the newest 10 and 30 minutes (1-minute values) and 6,
  12, 24 and 72 hours (``--rtk-rate``, :data:`RATE_WINDOWS`), a straight
  line per component, each with an EMPIRICAL σ (:func:`estimate_rate`).
  The SHORTEST window that passes its test is drawn on the map as an arrow
  with its 1-σ ellipse (:func:`pick_rate`, BGÓ 2026-11-12): the longest
  window when its speed is above :data:`RTK_SIGMA` σ; a shorter one when
  its rate has CHANGED from the background (the rate over the
  :data:`BG_WINDOW` before it) by more than its own threshold
  (:func:`change_test`). So a long, slow signal shows as the 72 h arrow
  until a sudden speed-up takes over. Testing the speed of each window
  against 2 σ instead fired nearly every hour (a 2-D speed passes 2 σ in
  exp(−2) ≈ 13 % of pure noise, and six windows compound it; measured
  9–24 % per window on SVIN, 2026-11-12).

The change test: d = v_window − v_background (N, E) at every end over the
newest :data:`CAL_DAYS`, its robust σ per component (1.4826 · MAD), and
z = |d / σ| (both components). The threshold is the larger of a floor and
a high quantile of z over those days (:data:`CHANGE_LIMITS`: under an
hour 4 and the 99.9 %, else 3 and the 99 %), so each window fires on
~0.1 % / 1 % of quiet ends whatever its noise, and the minute windows,
whose σ is ~0.3–2.5 m/d, only on a collapse.

The σ: RTKLIB's formal σ knows nothing of the RTK noise that matters for a
rate, which is correlated in time (multipath repeating with the sidereal
day, the atmosphere over hours). Two empirical estimates, the larger is
taken (the velocity product's rule: formal-like × k, floored by what the
data show, BGÓ 2026-10-16):

1. in the window: the scatter of the residuals from the line (white-noise
   σ of the slope) × the block-mean scale k of those residuals
   (:func:`gps_analysis.shared_tau.block_mean_scale`, blocks of an eighth
   of the window), which sees correlation shorter than the window;
2. repeatability: the rate over every window-long span ending on the
   hour over the newest :func:`history_days`, and the robust scatter
   (1.4826 · MAD) of those rates about a straight line through them in
   time. It sees what a single window cannot: a diurnal multipath cycle
   biases a 24 h line by an amount that depends on the hour the window
   ends, and the atmosphere wanders from day to day. The line through the
   rates takes out a steady rate and a steady acceleration; motion that
   changes faster than that counts as noise (σ errs large). Differences
   of back-to-back windows (an Allan variance) would miss the multipath:
   it repeats with the sidereal day, so windows one day apart share it.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import math
import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np

#: the RTKLIB quality flags dropped: 2 float, 5 single (fixed only)
DROP_Q = (2, 5)
#: days of RTK drawn over the daily points (``--rtk-days``)
RTK_DAYS = 3.0
#: the RTK rate's windows [days] (``--rtk-rate``), shortest first: the
#: first that passes its test is drawn (BGÓ 2026-11-12)
RATE_WINDOWS = (10.0 / 1440.0, 30.0 / 1440.0, 0.25, 0.5, 1.0, 3.0)
#: windows shorter than this are fitted on the 1-minute values
MINUTE_BELOW = 1.0 / 24.0
#: a shorter window's background: the rate over this [days] before it
BG_WINDOW = 3.0
#: the change thresholds are calibrated over this many days back
CAL_DAYS = 7.0
#: (up to window [days], floor, quantile) of the change z, the first that
#: holds: the minute windows high (only a collapse, BGÓ 2026-11-12); 12 h
#: and 24 h floors raised (BGÓ 2026-11-12): seven days hold only 14 / 7
#: independent 12 h / 24 h windows, so their quantile is a poor guide (24 h
#: on SVIN-SKFC fired at 20 mm/d on noise with the floor at 3)
CHANGE_LIMITS = (
    (MINUTE_BELOW, 4.0, 0.999),
    (0.25, 3.0, 0.99),
    (0.5, 3.5, 0.99),
    (math.inf, 4.0, 0.99),
)
#: the 1-minute values (:func:`clean_minutes`): a spike is further than
#: SPIKE_K robust σ (≥ SPIKE_FLOOR mm) from the CENTRED median of
#: SPIKE_POINTS minutes (a ramp, however fast, passes: its centred median
#: is itself; a trailing Hampel as rtk_accel's would cut a collapse's
#: first hour); the robust σ over SPIKE_SIGMA_H hours
SPIKE_POINTS = 5
SPIKE_K = 5.0
SPIKE_FLOOR = 2.0
SPIKE_SIGMA_H = 3.0
#: the sidereal multipath model: the median of the high-passed (24 h
#: centred median off) values one, two, … SIDEREAL_DAYS sidereal days
#: before (as rtk_accel), at least SIDEREAL_MIN of them
SIDEREAL_S = 86164.0
SIDEREAL_DAYS = 7
SIDEREAL_MIN = 3
#: one window's default (estimate_rate)
RATE_WINDOW = 1.0
#: the repeatability σ looks back this many days
HISTORY_DAYS = 7.0
#: the RTK is moved onto the daily series over up to this many common days
ALIGN_DAYS = 14.0
#: an RTK day median needs bins over at least this fraction of the day; a
#: window rate, bins over this fraction of its window
MIN_DAY_FRAC = 0.5
MIN_WINDOW_FRAC = 0.5
#: a median bin needs at least this fraction of the median bin's epochs
MIN_BIN_FRAC = 0.25
#: the RTK rate's arrow is drawn above this many σ (as the 14-day arrow)
RTK_SIGMA = 2.0
#: a shorter window that passed its change test must also stand this many σ
#: above zero on its own (BGÓ 2026-11-13: a 6 h rate of 48.9 ± 24.5 mm/d,
#: a half-hour weather excursion, passed the change test and set the scale)
SHORT_SIGMA = 3.0
#: the median intervals offered (``--rtk-interval``) and their length [days]
INTERVALS = {"1h": 1.0 / 24.0, "10min": 1.0 / 144.0, "1min": 1.0 / 1440.0}
#: the RTK path drawn through the bins: their running mean over this [days]
#: (a running median steps from bin to bin; the bins are medians already)
PATH_DAYS = 12.0 / 24.0
#: per pair of a --together figure: the RTK colour (beside polar.PAIR_STYLES)
#: the level change since an episode's onset (:func:`level_change`): the
#: median of this many days at each end; its σ from the scatter of
#: consecutive such medians (BGÓ 2026-11-13: rate × days overshot a drop
#: that had stopped)
LEVEL_DAYS = 0.5
#: each pair's RTK path and arrow colour, in pair order; none near the
#: colour bar's red end (BGÓ 2026-11-13: maroon vanished on the newest
#: dark-red daily points)
RTK_COLOURS = ("teal", "indigo", "chocolate", "maroon")


@dataclasses.dataclass(frozen=True)
class RtkRate:
    """The RTK rate over ``window`` days [mm/d] and its empirical 1-σ (N, E,
    U); ``white``/``k``/``repeat`` the two estimates behind it
    (:func:`estimate_rate`), ``n`` the bins in the window."""

    north: float
    east: float
    up: float
    sigma_north: float
    sigma_east: float
    sigma_up: float
    window: float
    n: int
    white: tuple[float, float, float]
    k: tuple[float, float, float]
    repeat: tuple[float, float, float]
    #: the change test (:func:`change_test`): z of the change from the
    #: background, its threshold, the background speed [mm/d]; ``trigger``
    #: its verdict (None: no change test, the 2 σ speed rule)
    z: float = float("nan")
    threshold: float = float("nan")
    background: float = float("nan")
    trigger: bool | None = None

    @property
    def speed(self) -> float:
        return math.hypot(self.north, self.east)

    @property
    def azimuth(self) -> float:
        """Direction of motion [° clockwise from north]."""
        return math.degrees(math.atan2(self.east, self.north)) % 360.0

    @property
    def sigma_speed(self) -> float:
        """σ of :attr:`speed`, along the rate's direction."""
        s = self.speed
        un, ue = (self.north / s, self.east / s) if s > 0 else (1.0, 0.0)
        return math.hypot(un * self.sigma_north, ue * self.sigma_east)

    @property
    def significant(self) -> bool:
        """The change test's verdict and the speed above
        :data:`SHORT_SIGMA` σ, else the horizontal speed above
        :data:`RTK_SIGMA` σ."""
        if self.trigger is not None:
            return self.trigger and self.speed > SHORT_SIGMA * self.sigma_speed
        return self.speed > RTK_SIGMA * self.sigma_speed

    def as_track_rate(self) -> Any:
        """As a :class:`gps_plot.track.Rate` [mm/yr] (the map's arrows)."""
        from gps_plot.track import YEAR, Rate

        return Rate(self.north * YEAR, self.east * YEAR,
                    self.sigma_north * YEAR, self.sigma_east * YEAR)  # fmt: skip


@dataclasses.dataclass(frozen=True)
class Alignment:
    """The RTK moved onto the daily series: ``shift`` (3,) [mm] added to
    it, over ``days`` common days, ``scatter`` (3,) the robust scatter of
    (daily − RTK day median) about the shift [mm]."""

    shift: Any
    days: int
    scatter: Any


@dataclasses.dataclass
class RtkOverlay:
    """A pair's RTK tail: ``t`` [days since 1970], ``d`` (3, N) [mm] on the
    daily displacement since the onset, ``rate`` (None: too few bins),
    ``interval`` its median interval, ``days`` the tail drawn; ``rates``
    every window's rate (shortest first), ``rate`` the one drawn
    (:func:`pick_rate`)."""

    t: Any
    d: Any
    alignment: Alignment
    rate: RtkRate | None
    interval: str
    days: float
    #: the rate's window [days]
    window: float = RATE_WINDOW
    #: (3, N) the running mean of ``d`` over :data:`PATH_DAYS` (the path)
    path: Any = None
    #: every window's rate, shortest first (None: too few bins)
    rates: tuple[RtkRate | None, ...] = ()
    #: the windows tried [days]
    windows: tuple[float, ...] = ()
    #: the average rate since an episode's onset (``since``, days since
    #: 1970), the RTK figure's black arrow (BGÓ 2026-11-12); None: not asked
    since: RtkRate | None = None
    since_t: float | None = None
    #: the movement since ``since_t`` (:func:`level_change`): the change
    #: (3,) [mm], its 1-σ (3,; at least √2 × the alignment's day scatter),
    #: and the newest level (3,) on ``d``'s scale
    since_change: Any = None
    since_sigma: Any = None
    since_end: Any = None


@dataclasses.dataclass(frozen=True)
class Anchor:
    """The other end of the RTK pair, in the figure's frame (BGÓ 2026-11-13:
    the polar figure in the reference-mean frame, as the track map and the
    time series): its daily positions ``t`` [days since 1970], ``y`` (3, N)
    [mm], its newest rate ``rate`` (3,) [mm/d] and that rate's σ (3,).

    The station in the frame = (station − anchor) by RTK + the anchor in the
    frame (:meth:`at`). The anchor moves slowly next to the RTK's noise, so
    its daily positions, linear between them, carry it; after its newest day
    it goes on at ``rate`` (held flat it would put its own motion, SVIE
    ≈ 0.65 mm/d, into the station's newest rates). The pre-onset trends of
    the two ends differ by a few mm/yr: left in."""

    name: str
    t: Any
    y: Any
    rate: Any
    sigma: Any

    def at(self, t: Any) -> Any:
        """(3, len(t)) the anchor's position at ``t`` [days since 1970]."""
        t = np.asarray(t, dtype=float)
        out = np.full((3, t.size), np.nan)
        for c in range(3):
            ok = np.isfinite(self.y[c])
            if not ok.any():
                continue
            tc, yc = np.asarray(self.t, float)[ok], np.asarray(self.y[c], float)[ok]
            v = np.interp(t, tc, yc)
            after = t > tc[-1]
            rate = float(self.rate[c]) if np.isfinite(self.rate[c]) else 0.0
            v[after] = yc[-1] + rate * (t[after] - tc[-1])
            out[c] = v
        return out


def with_anchor_sigma(rate: RtkRate | None, anchor: Anchor) -> RtkRate | None:
    """``rate`` with the anchor's rate σ added in quadrature (the RTK's own
    empirical σ does not hold the anchor's daily noise)."""
    if rate is None:
        return None
    a = [float(x) if np.isfinite(x) else 0.0 for x in anchor.sigma]
    return dataclasses.replace(
        rate,
        sigma_north=math.hypot(rate.sigma_north, a[0]),
        sigma_east=math.hypot(rate.sigma_east, a[1]),
        sigma_up=math.hypot(rate.sigma_up, a[2]),
    )


def history_days(window: float) -> float:
    """The repeatability's look-back for a ``window``-long rate: at least
    :data:`HISTORY_DAYS`, and four windows (three of back-to-back pairs)."""
    return max(HISTORY_DAYS, 4.0 * window)


def day_medians(
    t: Any, y: Any, step: float, min_frac: float = MIN_DAY_FRAC
) -> tuple[Any, Any]:
    """(UTC day [days since 1970, at 0 h], (3, M) median) of the bins
    ``t`` (centred, days since 1970) per UTC day, for days with bins over
    at least ``min_frac`` of the day (``step``: one bin [days])."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    if t.size == 0:
        return np.empty(0), np.empty((3, 0))
    day = np.floor(t)
    days = np.unique(day)
    need = min_frac / step
    keep, med = [], []
    for d in days:
        m = day == d
        if np.count_nonzero(m) >= need:
            keep.append(d)
            med.append(np.nanmedian(y[:, m], axis=1))
    if not keep:
        return np.empty(0), np.empty((3, 0))
    return np.array(keep), np.array(med).T


def align_shift(
    t_rtk: Any,
    y_rtk: Any,
    t_day: Any,
    y_day: Any,
    step: float,
    *,
    days: float = ALIGN_DAYS,
    min_frac: float = MIN_DAY_FRAC,
) -> Alignment:
    """The (3,) shift [mm] that puts the RTK on the daily series: the median
    over the newest ``days`` common UTC days of (daily point − RTK day
    median, :func:`day_medians`). Not :func:`gps_plot.sources.rtk.align`:
    that interpolates single bins at the daily epochs, ±10–20 mm for the
    noisy pairs; a day median is the RTK's own 24 h solution.

    Raises:
        ValueError: no common day.
    """
    td, md = day_medians(t_rtk, y_rtk, step, min_frac)
    t_day = np.asarray(t_day, dtype=float)
    y_day = np.asarray(y_day, dtype=float)
    daily_day = np.floor(t_day)
    common = np.intersect1d(td, daily_day)
    if common.size:
        common = common[common > common.max() - days]
    if not common.size:
        raise ValueError("no day common to the RTK and the daily series")
    diffs = []
    for d in common:
        i = int(np.flatnonzero(daily_day == d)[-1])
        j = int(np.flatnonzero(td == d)[0])
        diffs.append(y_day[:, i] - md[:, j])
    diff = np.array(diffs).T  # (3, n)
    shift = np.nanmedian(diff, axis=1)
    scatter = 1.4826 * np.nanmedian(np.abs(diff - shift[:, None]), axis=1)
    return Alignment(shift, int(common.size), scatter)


def running_mean(t: Any, y: Any, span: float = PATH_DAYS) -> Any:
    """(3, N): the centred running mean of ``y`` over ``span`` days of
    ``t`` (the bins within ±``span``/2 of each)."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    out = np.full_like(y, np.nan)
    for i, ti in enumerate(t):
        m = np.abs(t - ti) <= span / 2.0 + 1e-9
        out[:, i] = np.nanmean(y[:, m], axis=1)
    return out


def line_fit(t: Any, y: Any) -> tuple[float, float, Any]:
    """(slope, its white-noise σ, residuals) of a straight line through
    ``y`` against ``t`` (least squares); σ NaN with fewer than 3 points."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    tc = t - t.mean()
    sxx = float((tc**2).sum())
    if t.size < 2 or sxx == 0:
        return float("nan"), float("nan"), np.full_like(y, np.nan)
    slope = float((tc * (y - y.mean())).sum() / sxx)
    res = y - y.mean() - slope * tc
    if t.size < 3:
        return slope, float("nan"), res
    s = float(np.sqrt((res**2).sum() / (t.size - 2)))
    return slope, s / math.sqrt(sxx), res


def window_rates(
    t: Any,
    y: Any,
    window: float,
    step: float,
    min_frac: float = MIN_WINDOW_FRAC,
    ends: Any = None,
) -> tuple[Any, Any]:
    """(ends, rates): the slope of ``y`` over every ``window``-long span
    (t in (end − window, end]) with its end on the ``step`` grid from the
    first full window to the newest bin (or at the given ``ends``); NaN for
    a span with bins over less than ``min_frac`` of it (``step`` the bin
    length). Running sums, so 10 000 minute windows cost no more than one."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(t) & np.isfinite(y)
    t, y = t[ok], y[ok]
    if ends is None:
        if t.size < 3:
            return np.empty(0), np.empty(0)
        n_end = int((t.max() - t.min() - window) / step) + 1
        ends = (t.max() - step * np.arange(max(n_end, 0)))[::-1]
    ends = np.asarray(ends, dtype=float)
    rates = np.full(ends.size, np.nan)
    if t.size < 3 or ends.size == 0:
        return ends, rates
    order = np.argsort(t)
    t, y = t[order], y[order]
    t0, y0 = t[0], y[0]
    tc, yc = t - t0, y - y0  # small numbers: the sums keep their digits
    cum = [
        np.r_[0.0, np.cumsum(v)] for v in (np.ones_like(tc), tc, yc, tc * tc, tc * yc)
    ]
    lo = np.searchsorted(t, ends - window + 1e-9, side="right")
    hi = np.searchsorted(t, ends + 1e-9, side="right")
    n, st, sy, stt, sty = (c[hi] - c[lo] for c in cum)
    need = max(min_frac * window / step, 3)
    with np.errstate(invalid="ignore", divide="ignore"):
        sxx = stt - st * st / n
        slope = (sty - st * sy / n) / sxx
    good = (n >= need) & (sxx > 0)
    rates[good] = slope[good]
    return ends, rates


def repeat_sigma(t: Any, y: Any, window: float, step: float) -> float:
    """The repeatability σ of a ``window``-long rate: the robust scatter
    (1.4826 · MAD) of the rates :func:`window_rates` gives about a straight
    line through them in time (which takes out a steady acceleration); NaN
    when the rates span under two windows."""
    ends, rates = window_rates(t, y, window, step)
    ok = np.isfinite(rates)
    if np.count_nonzero(ok) < 2 or np.ptp(ends[ok]) < 2.0 * window - 1e-9:
        return float("nan")
    _, _, res = line_fit(ends[ok], rates[ok])
    return 1.4826 * float(np.median(np.abs(res - np.median(res))))


def estimate_rate(
    t: Any,
    y: Any,
    window: float = RATE_WINDOW,
    step: float = INTERVALS["1h"],
    history: float | None = None,
) -> RtkRate | None:
    """The rate [mm/d] over the newest ``window`` days of ``y`` (3, N) [mm]
    at ``t`` [days], each component's σ the larger of the in-window
    white × k and the repeatability over ``history`` days (default
    :func:`history_days`; the module's docstring); None when the window holds bins over less than
    :data:`MIN_WINDOW_FRAC` of it."""
    from gps_analysis.shared_tau import block_mean_scale

    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    if t.size == 0:
        return None
    if history is None:
        history = history_days(window)
    newest = float(np.nanmax(t))
    inside = t > newest - window + 1e-9  # (newest − window, newest]
    past = t > newest - window - history + 1e-9
    rate, sigma, white, ks, rep = [], [], [], [], []
    n = 0
    for c in range(3):
        ok = inside & np.isfinite(y[c])
        n = max(n, int(np.count_nonzero(ok)))
        if np.count_nonzero(ok) < max(MIN_WINDOW_FRAC * window / step, 3):
            return None
        v, sw, res = line_fit(t[ok], y[c, ok])
        sd = float(np.std(res))
        k = 1.0
        if sd > 0:
            blocks = max(2, int(np.count_nonzero(ok)) // 8)
            k = float(block_mean_scale(t[ok], res / sd, step=step, n=blocks).k)
        r = repeat_sigma(t[past], y[c, past], window, step)
        rate.append(v)
        white.append(sw)
        ks.append(k)
        rep.append(r)
        sigma.append(float(np.nanmax([sw * k, r])))
    return RtkRate(rate[0], rate[1], rate[2], sigma[0], sigma[1], sigma[2], window, n,
                   (white[0], white[1], white[2]), (ks[0], ks[1], ks[2]),
                   (rep[0], rep[1], rep[2]))  # fmt: skip


def change_test(
    rate: RtkRate,
    t: Any,
    y: Any,
    step: float,
    t_bg: Any,
    y_bg: Any,
    bg_step: float,
) -> RtkRate:
    """``rate`` (over its window, fitted on ``(t, y)`` at ``step``) with the
    change test: the change d = v − v_bg of its N/E rate from the
    background rate over the :data:`BG_WINDOW` before the window (fitted
    on ``(t_bg, y_bg)``), at every ``step`` end over :data:`CAL_DAYS`; z =
    |d/σ| with σ = 1.4826 · MAD per component; ``trigger`` when the newest
    z passes max(floor, quantile of the earlier z) (:data:`CHANGE_LIMITS`).
    No calibration (too few ends): ``trigger`` False."""
    w = rate.window
    floor, q = change_limit(w)
    t = np.asarray(t, dtype=float)
    newest = float(np.nanmax(t))
    ends = newest - step * np.arange(int(round(CAL_DAYS / step)) + 1)[::-1]
    d = []
    for c in range(2):
        _, v = window_rates(t, np.asarray(y)[c], w, step, ends=ends)
        _, vb = window_rates(t_bg, np.asarray(y_bg)[c], BG_WINDOW, bg_step,
                             ends=ends - w)  # fmt: skip
        d.append(v - vb)
    dn, de = d
    ok = np.isfinite(dn) & np.isfinite(de)
    past = ok & (ends <= newest - w + 1e-9)  # the calibration: before this window
    nan = float("nan")
    if np.count_nonzero(past) < 20 or not ok[-1]:
        return dataclasses.replace(rate, z=nan, threshold=nan, trigger=False)
    sig = [1.4826 * float(np.median(np.abs(x[past] - np.median(x[past])))) for x in d]
    if min(sig) <= 0:
        return dataclasses.replace(rate, trigger=False)
    z = np.hypot(dn / sig[0], de / sig[1])
    limit = max(floor, float(np.quantile(z[past], q)))
    bg = float(np.hypot(rate.north - dn[-1], rate.east - de[-1]))
    return dataclasses.replace(rate, z=float(z[-1]), threshold=limit, background=bg,
                               trigger=bool(z[-1] > limit))  # fmt: skip


def change_limit(window: float) -> tuple[float, float]:
    """(floor, quantile) of a ``window``'s change test
    (:data:`CHANGE_LIMITS`)."""
    for upto, floor, q in CHANGE_LIMITS:
        if window <= upto + 1e-9:
            return floor, q
    return CHANGE_LIMITS[-1][1], CHANGE_LIMITS[-1][2]


def clean_minutes(t: Any, y: Any) -> tuple[Any, Any]:
    """The 1-minute values for the minute windows: isolated spikes dropped
    (centred Hampel, :data:`SPIKE_K`), then minus the sidereal multipath
    model (:data:`SIDEREAL_DAYS`; where it has under
    :data:`SIDEREAL_MIN` days, nothing is taken off). ``t`` [days], ``y``
    (3, N) [mm]."""
    import pandas as pd  # type: ignore[import-untyped]

    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float).copy()
    if t.size == 0:
        return t, y
    idx = pd.to_datetime(np.round(t * 86400.0), unit="s")
    df = pd.DataFrame(y.T, index=idx)
    med = df.rolling(SPIKE_POINTS, center=True, min_periods=3).median()
    dev = (df - med).abs()
    sig = (
        1.4826 * dev.rolling(pd.Timedelta(hours=SPIKE_SIGMA_H), min_periods=30).median()
    )
    lim = np.maximum(SPIKE_K * sig.to_numpy(), SPIKE_FLOOR)
    y = np.where(dev.to_numpy() > lim, np.nan, df.to_numpy()).T
    hp = pd.DataFrame(y.T, index=idx)
    hp = (
        (hp - hp.rolling(pd.Timedelta(hours=24), center=True, min_periods=360).median())
        .to_numpy()
        .T
    )
    sid = SIDEREAL_S / 86400.0
    half = 0.5 / 1440.0
    shifted = []
    for k in range(1, SIDEREAL_DAYS + 1):
        want = t - k * sid
        j = np.clip(np.searchsorted(t, want), 1, t.size - 1)
        j = np.where(np.abs(t[j - 1] - want) < np.abs(t[j] - want), j - 1, j)
        near = np.abs(t[j] - want) <= half + 1e-9
        shifted.append(np.where(near, hp[:, j], np.nan))
    stack = np.stack(shifted)  # (days, 3, N)
    enough = np.count_nonzero(np.isfinite(stack), axis=0) >= SIDEREAL_MIN
    with warnings.catch_warnings():  # all-NaN columns: "too few days" below
        warnings.simplefilter("ignore", RuntimeWarning)
        model = np.nanmedian(stack, axis=0)
    model = np.where(enough & np.isfinite(model), model, 0.0)
    return t, y - model


def windows_of(window: float | Sequence[float]) -> tuple[float, ...]:
    """The rate windows as a sorted tuple (one window or several)."""
    if isinstance(window, (int, float)):
        return (float(window),)
    return tuple(sorted(float(w) for w in window))


def pick_rate(rates: Sequence[RtkRate | None]) -> RtkRate | None:
    """The rate drawn: the shortest window's that passes its test
    (``rates`` shortest first; :attr:`RtkRate.significant`), so a sudden
    speed-up overrides a slow, long-running signal; none: the longest
    window's (the most precise, for the legend's "below 2σ"), None when no
    window has one."""
    for r in rates:
        if r is not None and r.significant:
            return r
    for r in reversed(rates):
        if r is not None:
            return r
    return None


def bins(t: Any, y: Any, count: Any, step: float) -> tuple[Any, Any]:
    """The median bins kept: their epochs moved to the bin centre (the
    reader labels a bin by its start) and the thin bins dropped (under
    :data:`MIN_BIN_FRAC` of the median bin's epochs)."""
    t = np.asarray(t, dtype=float) + step / 2.0
    count = np.asarray(count, dtype=float)
    keep = count >= MIN_BIN_FRAC * float(np.median(count)) if count.size else count > 0
    return t[keep], np.asarray(y, dtype=float)[:, keep]


def read_rtk(
    name: str, start: dt.datetime, end: dt.datetime, config: Any, interval: str
) -> tuple[Any, Any]:
    """(t [days since 1970, bin centres], (3, N) [mm]) of the fixed RTK
    solutions of pair ``A-B`` (A − B; a pair processed the other way read
    flipped), medians over ``interval``."""
    from geo_dataread.rtk_read import list_baselines, read_rtk_baseline, rtk_dir

    from gps_plot.cli import common
    from gps_plot.sources.rtk import rtk_direction

    root = rtk_dir(common.config_dir(config, "rtk_dir"))
    directory, flipped = rtk_direction(name, list_baselines(root))
    r = read_rtk_baseline(directory, start, end, root=root, interval=interval,
                          drop_q=DROP_Q)  # fmt: skip
    t = np.asarray(r.time, dtype="datetime64[s]").astype(np.int64) / 86400.0
    y = np.asarray(r.neu, dtype=float) * (-1000.0 if flipped else 1000.0)
    return bins(t, y, r.count, INTERVALS[interval])


def overlay(
    t: Any,
    y: Any,
    t_day: Any,
    d_day: Any,
    *,
    interval: str = "1h",
    days: float = RTK_DAYS,
    window: float | Sequence[float] = RATE_WINDOWS,
    after: float | None = None,
    minutes: tuple[Any, Any] | None = None,
    since: float | None = None,
) -> RtkOverlay:
    """The RTK tail of ``(t, y)`` (:func:`read_rtk`) on the daily
    displacement ``d_day`` at ``t_day``: shifted onto it
    (:func:`align_shift`), the newest ``days`` kept (and none before
    ``after``, the onset), the rate over each of the newest ``window``
    (one or several, :func:`pick_rate`): windows under
    :data:`MINUTE_BELOW` on ``minutes`` (the 1-minute ``(t, y)``; none:
    those windows are skipped), every window but the longest with its
    :func:`change_test`; with ``since`` [days since 1970] also the rate
    from then to the newest bin (empirical σ, the 2 σ speed rule).

    Raises:
        ValueError: no RTK, or no day common to the two.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    if t.size == 0:
        raise ValueError("no fixed RTK solutions")
    step = INTERVALS[interval]
    al = align_shift(t, y, t_day, d_day, step)
    keep = t > t.max() - days
    if after is not None:
        keep &= t >= after
    windows = windows_of(window)
    rates = tuple(
        _window_rate(t, y, step, w, minutes, change=w < windows[-1]) for w in windows
    )
    rate = pick_rate(rates)
    d = y[:, keep] + al.shift[:, None]
    since_rate = moved = None
    if since is not None and since < t.max():
        since_rate = estimate_rate(t, y, window=float(t.max()) - since, step=step)
        moved = level_change(t, y, since)
        if moved is not None:  # nor below the RTK's day-level error: its
            # day medians' scatter about the daily solutions, at both ends
            # (BGÓ 2026-11-13: SVIE's RTK Up drifts +45 mm, the daily not)
            floor = math.sqrt(2.0) * np.asarray(al.scatter, dtype=float)
            moved = (moved[0], np.fmax(moved[1], floor), moved[2])
    return RtkOverlay(t[keep], d, al, rate, interval, days,
                      rate.window if rate is not None else windows[-1],
                      running_mean(t[keep], d), rates, windows,
                      since_rate, since,
                      *((moved[0], moved[1], moved[2] + al.shift) if moved
                        else (None, None, None)))  # fmt: skip


def level_change(
    t: Any, y: Any, t0: float, block: float = LEVEL_DAYS
) -> tuple[Any, Any, Any] | None:
    """The movement from ``t0`` to the newest epoch: the median of the
    ``block`` days from ``t0`` subtracted from the median of the newest
    ``block`` days, per component (3,) [mm]; its 1-σ from the scatter of
    consecutive ``block`` medians over the span (robust, of their
    differences / √2: the time-correlated noise within a block is in it),
    × √2 for the two ends; and the newest level (3,). None: an end
    without data or fewer than 3 blocks."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    if not t.size or t0 >= t.max():
        return None
    first = (t >= t0) & (t < t0 + block)
    last = t > t.max() - block
    if not first.any() or not last.any():
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        start = np.nanmedian(y[:, first], axis=1)
        end = np.nanmedian(y[:, last], axis=1)
        edges = np.arange(t0, t.max() + block, block)
        meds = [np.nanmedian(y[:, (t >= a) & (t < a + block)], axis=1)
                for a in edges[:-1] if ((t >= a) & (t < a + block)).any()]  # fmt: skip
    if len(meds) < 3:
        return None
    diff = np.diff(np.asarray(meds), axis=0) / math.sqrt(2.0)
    mad = np.nanmedian(np.abs(diff - np.nanmedian(diff, axis=0)), axis=0)
    sigma = math.sqrt(2.0) * 1.4826 * mad
    return end - start, sigma, end


def _window_rate(
    t: Any,
    y: Any,
    step: float,
    window: float,
    minutes: tuple[Any, Any] | None,
    change: bool,
) -> RtkRate | None:
    """One window's rate (:func:`estimate_rate`) on the bins, or the
    1-minute values under :data:`MINUTE_BELOW`; with its
    :func:`change_test` against the bins' background when ``change``."""
    src, src_step = (t, y), step
    if window < MINUTE_BELOW:
        if minutes is None:
            return None
        src, src_step = minutes, INTERVALS["1min"]
    rate = estimate_rate(src[0], src[1], window=window, step=src_step)
    if rate is None or not change:
        return rate
    return change_test(rate, src[0], src[1], src_step, t, y, step)


def read_overlay(
    name: str,
    end: dt.datetime,
    config: Any,
    t_day: Any,
    d_day: Any,
    *,
    interval: str = "1h",
    days: float = RTK_DAYS,
    window: float | Sequence[float] = RATE_WINDOWS,
    after: float | None = None,
    since: float | None = None,
    anchor: Anchor | None = None,
) -> RtkOverlay:
    """:func:`overlay` of the pair's RTK read back far enough for the tail,
    the rates' windows (``since``'s too) and history, and the alignment;
    with ``anchor`` the pair's station in the anchor's frame (the pair read
    station − anchor, plus :meth:`Anchor.at`), the rates' σ with the
    anchor's rate σ."""
    windows = windows_of(window)
    longest = max(windows)
    if since is not None:  # its window, to the end
        longest = max(longest, (end - dt.datetime(1970, 1, 1)).total_seconds() / 86400.0
                      - since)  # fmt: skip
    back = max(days, longest + history_days(longest), ALIGN_DAYS,
               CAL_DAYS + longest + BG_WINDOW) + 1.0  # fmt: skip
    t, y = read_rtk(name, end - dt.timedelta(days=back), end, config, interval)
    if anchor is not None:
        y = y + anchor.at(t)
    minutes = None
    if min(windows) < MINUTE_BELOW:
        back_min = CAL_DAYS + SIDEREAL_DAYS * SIDEREAL_S / 86400.0 + 1.5
        tm, ym = clean_minutes(*read_rtk(name, end - dt.timedelta(days=back_min),
                                         end, config, "1min"))  # fmt: skip
        if anchor is not None:  # after the cleaning: the anchor is smooth
            ym = ym + anchor.at(tm)
        minutes = (tm, ym)
    ov = overlay(t, y, t_day, d_day, interval=interval, days=days, window=window,
                 after=after, minutes=minutes, since=since)  # fmt: skip
    if anchor is None:
        return ov
    rates = tuple(with_anchor_sigma(r, anchor) for r in ov.rates)
    rate = pick_rate(rates)  # again: the 72 h speed test sees the larger σ
    return dataclasses.replace(
        ov,
        rate=rate,
        rates=rates,
        window=rate.window if rate is not None else ov.windows[-1],
        since=with_anchor_sigma(ov.since, anchor),
    )


def span_text(days: float) -> str:
    """``10 min`` / ``24 h`` / ``3 d``."""
    if days < 1.0 / 24.0:
        return f"{days * 1440.0:.0f} min"
    hours = days * 24.0
    return (
        f"{hours:g} h"
        if hours <= 72 and abs(hours - round(hours)) < 1e-6
        else f"{days:g} d"
    )


def rate_label(rate: RtkRate) -> str:
    """The RTK arrow's legend text (usetex-safe: σ and ± as math; short:
    the map's legend; Up in the log); a change-tested window says its
    speed-up against its limit."""
    text = (f"RTK {span_text(rate.window)}: {rate.speed:.1f} $\\pm$ "
            f"{rate.sigma_speed:.1f} mm/d toward {rate.azimuth:.0f}°")  # fmt: skip
    if rate.trigger is not None and math.isfinite(rate.z):
        text += f", speed-up {rate.z:.1f} > {rate.threshold:.1f}"
    return text


def windows_text(windows: Sequence[float]) -> str:
    """The windows tried, shortest–longest: ``10 min–72 h``."""
    if not windows:
        return ""
    lo, hi = span_text(min(windows)), span_text(max(windows))
    return lo if lo == hi else f"{lo}–{hi}"


def quiet_label(
    rate: RtkRate | None, window: float, windows: Sequence[float] = ()
) -> str:
    """The legend text when the RTK rate is not drawn (``windows``: all
    the windows tried: no speed-up in the shorter ones, the longest below
    2 σ)."""
    many = len(windows) > 1
    if rate is None:
        span = windows_text(windows) if many else span_text(window)
        return f"RTK {span}: too few fixed solutions"
    if many:  # short: the map's legend sits beside its colour bar
        return (f"RTK no speed-up; {span_text(rate.window)} "
                f"{rate.speed:.1f} $\\pm$ {rate.sigma_speed:.1f} mm/d")  # fmt: skip
    return (f"RTK {span_text(rate.window)}: {rate.speed:.1f} $\\pm$ "
            f"{rate.sigma_speed:.1f} mm/d, below {RTK_SIGMA:g}$\\sigma$")  # fmt: skip


def note(name: str, ov: RtkOverlay) -> str:
    """The log line of one pair's RTK."""
    al = ov.alignment
    text = (f"{name} RTK: fixed {ov.interval} medians, last {span_text(ov.days)}; "
            f"aligned on {al.days} days, shift N/E/U "
            f"{al.shift[0]:+.1f}/{al.shift[1]:+.1f}/{al.shift[2]:+.1f} mm, day-median "
            f"scatter {al.scatter[0]:.1f}/{al.scatter[1]:.1f}/{al.scatter[2]:.1f} mm")  # fmt: skip
    rates = [r for r in ov.rates if r is not None] or (
        [ov.rate] if ov.rate is not None else []
    )
    if not rates:
        return text + "; rate: too few bins"
    if ov.rate is not None and ov.rate.significant:
        text += (
            f"; drawn: {span_text(ov.rate.window)} (the shortest that passes its test)"
        )
    elif len(ov.windows) > 1:
        text += "; drawn: none (no window passes its test)"
    return text + "".join(_rate_note(r) for r in rates)


def _rate_note(r: RtkRate) -> str:
    change = (f", change z {r.z:.1f} / limit {r.threshold:.1f}"
              if r.trigger is not None else "")  # fmt: skip
    return (
        f"; rate {span_text(r.window)} N/E/U {r.north:+.2f}/{r.east:+.2f}/{r.up:+.2f} "
        f"± {r.sigma_north:.2f}/{r.sigma_east:.2f}/{r.sigma_up:.2f} mm/d "
        f"(white×k {_triple([w * k for w, k in zip(r.white, r.k, strict=True)])}, repeat {_triple(r.repeat)}), "
        f"{'significant' if r.significant else 'not significant'}{change}"
    )


def _triple(v: Sequence[float]) -> str:
    return "/".join(f"{x:.2f}" for x in v)
