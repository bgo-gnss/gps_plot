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
RECENT_COLOR = "red"
LEGEND_FONT = "7p,Helvetica-Bold"
LEGEND_SPACING = 0.04
DEFAULT_CMAP = "turbo"
#: the colour bar: a fifth of the map wide, inside its upper right
CBAR_WIDTH = 0.2
CBAR_FONT = "7p,Helvetica"
CBAR_BOX = "+gwhite@15+p0.4p,gray40+c0.1c/0.1c"
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

    @property
    def end(self) -> tuple[float, float]:
        """The newest position (N, E): the mean of the last two days after
        the onset (one if only one)."""
        m = min(2, self.n_after)
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
    def end_sigma(self) -> float:
        """σ of :attr:`end`: the daily scatter over the days averaged."""
        return self.scatter / math.sqrt(max(1, min(2, self.n_after)))


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
    t: FloatArray, y: FloatArray, pre: npt.NDArray[np.bool_]
) -> tuple[FloatArray, FloatArray]:
    """``y`` (2, N) minus the line + annual + semiannual fitted on ``pre``;
    and the per-component scatter of the fit [same unit]."""
    if int(pre.sum()) < MIN_PRE_DAYS:
        raise ValueError(
            f"{int(pre.sum())} days in the trend window (< {MIN_PRE_DAYS})"
        )
    g = _design(t, float(t[pre].mean()))
    r = np.empty_like(y)
    rms = np.empty(y.shape[0])
    for c in range(y.shape[0]):
        m, *_ = np.linalg.lstsq(g[pre], y[c, pre], rcond=None)
        r[c] = y[c] - g @ m
        rms[c] = float(np.std(r[c, pre]))
    return r, rms


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
) -> tuple[FloatArray, FloatArray]:
    """One reference from several stations: each one's trend and seasonal
    (fitted on ``pre``) removed, then their weighted mean per day (weights
    1/scatter², the fit's per component), on the days at least half of
    them have. Like GLOBK's frame realisation by many reference stations,
    translation only: one station's own wander (a bad day, a local
    motion) moves the reference by its share, not wholly."""
    if not series:
        raise ValueError("no reference stations")
    a, b = yearf(pre[0]), yearf(pre[1])
    days, values, weights = [], [], []
    for t, y in series:
        r, rms = detrend(t, y, (t >= a) & (t < b))
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
) -> Track:
    """A station's :class:`Track` from its horizontal series ``y`` (N, E)
    [mm] at epochs ``t`` [fractional years, finite]."""
    t_on = yearf(onset)
    in_pre = (t >= yearf(pre[0])) & (t < yearf(pre[1]))
    r, rms = detrend(t, y, in_pre)
    near = (t >= t_on - ZERO_DAYS / YEAR) & (t < t_on)
    zero = np.median(r[:, near], axis=1) if int(near.sum()) >= 3 else np.zeros(2)
    r = r - zero[:, None]
    shown = t >= t_on - before_days / YEAR
    if not shown.any() or t[-1] < t_on:
        raise ValueError(f"no data since {onset:%Y-%m-%d}")
    since = line_rate(t, r, t_on, float(t[-1]), rms)
    recent = line_rate(t, r, float(t[-1]) - recent_days / YEAR, float(t[-1]), rms)
    return Track(
        marker, lon, lat, t[shown], r[0, shown], r[1, shown],
        (t[shown] - t_on) * YEAR, float(math.hypot(*rms)), since, recent,
        int((t >= t_on).sum()),
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
) -> Track:
    """Read a station (:func:`load_series`) and make its track; with
    ``reference`` (another station's ``load_series``) relative to it."""
    t, y = load_series(marker, pre, frame=frame, pos_dir=pos_dir, uncert=uncert,
                       common_mode=common_mode, end=end)  # fmt: skip
    if reference is not None:
        t, y = relative_to(t, y, *reference)
    return station_track(marker, lon, lat, t, y, onset, pre,
                         before_days=before_days, recent_days=recent_days)  # fmt: skip


def legend_entries(
    tracks: Sequence[Track], onset: dt.date, recent_days: int
) -> list[tuple[str, str]]:
    """``(text, colour)`` per legend line, top down: per moved station its
    rate since the onset (blue) and recent rate (red), or its displacement
    (black) when it has too few days for a rate."""
    pm, deg = "±", "°"
    out: list[tuple[str, str]] = []
    for tr in tracks:
        if not tr.moved:
            continue
        if tr.since is not None:
            s = tr.since
            moved_mm = math.hypot(*tr.end)
            out.append((f"{tr.marker} {s.speed:.0f} {pm} {s.sigma_speed:.0f} mm/yr "
                        f"toward {s.azimuth:.0f}{deg} since {onset:%d.%m.%Y}, "
                        f"moved {moved_mm:.0f} {pm} {tr.end_sigma:.0f} mm",
                        SINCE_COLOR))  # fmt: skip
            if tr.recent is not None:
                r = tr.recent
                out.append((f"{tr.marker} {r.speed:.0f} {pm} {r.sigma_speed:.0f} "
                            f"mm/yr last {recent_days} days", RECENT_COLOR))  # fmt: skip
        else:
            n, e = tr.end
            az = math.degrees(math.atan2(e, n)) % 360
            out.append((f"{tr.marker} moved {math.hypot(n, e):.0f} {pm} "
                        f"{tr.end_sigma:.0f} mm toward {az:.0f}{deg} since "
                        f"{onset:%d.%m} ({tr.n_after} days of data)", "black"))  # fmt: skip
    return out


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
) -> float:
    """Draw the tracks, arrows, legend and colour bar on ``fig``.

    The legend sits in the lower left, its foot ``legend_floor_cm`` above the
    map's bottom edge (clear of the longitude annotations). Returns the
    legend's top [cm above the bottom edge], for the locator to sit on.
    """
    from gps_plot import maps

    pygmt = maps._import_pygmt()
    width, height = maps._map_size_cm(region, projection)
    w, e, s, n = region[:4]
    km_per_cm = (e - w) * 111.32 * math.cos(math.radians((s + n) / 2)) / width
    auto_m, dot_ref = dot_scale(tracks, km_per_cm)
    m_per_mm = m_per_mm or auto_m
    if m_per_mm != auto_m:
        far = TRACK_CM * km_per_cm * 1000.0 / m_per_mm
        dot_ref = nice(max(far / 2, 0.5))

    paths = []
    for tr in tracks:
        x, y = offset_lonlat(tr.lon, tr.lat, tr.north, tr.east, m_per_mm)
        paths.append((x, y, tr.days))
        pn, pe = running_median(tr.north, path_days), running_median(tr.east, path_days)
        px, py = offset_lonlat(tr.lon, tr.lat, pn, pe, m_per_mm)
        fig.plot(x=px, y=py, pen=PATH_PEN)
    x = np.concatenate([p[0] for p in paths]) if paths else np.zeros(0)
    y = np.concatenate([p[1] for p in paths]) if paths else np.zeros(0)
    z = np.concatenate([p[2] for p in paths]) if paths else np.zeros(0)
    if z.size:
        order = np.argsort(z, kind="stable")  # newest on top
        lo = float(min(z.min(), -1.0))
        pygmt.makecpt(cmap=cmap, series=[lo, max(float(z.max()), lo + 1.0)])
        fig.plot(x=x[order], y=y[order], fill=z[order], cmap=True,
                 style=DOT_STYLE, pen=DOT_PEN)  # fmt: skip
        # small, inside the upper right, on a white box (BGÓ 2026-10-28)
        with pygmt.config(FONT_ANNOT_PRIMARY=CBAR_FONT, FONT_LABEL=CBAR_FONT,
                          MAP_ANNOT_OFFSET_PRIMARY="1p", MAP_LABEL_OFFSET="2p",
                          MAP_TICK_LENGTH_PRIMARY="2p"):  # fmt: skip
            fig.colorbar(
                frame=[f"xaf+ldays since {onset:%d.%m.%Y}"],
                position=f"JTR+jTR+o0.45c/0.35c+w{CBAR_WIDTH * width:.1f}c/0.2c+h",
                box=CBAR_BOX,
            )

    moved = [tr for tr in tracks if tr.moved and tr.since is not None]
    speeds = [tr.since.speed for tr in moved if tr.since is not None]
    scale = arrow_scale or (ARROW_CM / max(speeds) if speeds else 0.01)
    ref = nice(max(speeds) / 2) if speeds else 100.0
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
            scale=scale, confidence=0.39, vector_head=ARROW_HEAD,
            pen_width=ARROW_PEN, ellipse_pen=ELLIPSE_PEN, markers=False,
            labels=False,
        )  # fmt: skip

    entries = legend_entries(tracks, onset, recent_days)
    if note:  # e.g. the reference station: what the motion is relative to
        entries = [(note, "black"), *entries]
    return _draw_legend(
        fig, entries, region=region,
        projection=projection, arrow_ref=ref if arrows and moved else None,
        arrow_scale=scale, dot_ref=dot_ref, m_per_mm=m_per_mm,
        floor_cm=legend_floor_cm,
    )  # fmt: skip


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
) -> float:
    """The lower-left legend box: on top the arrow reference (when arrows are
    drawn) and the dot scale side by side, under them ``entries`` (top
    down). Returns its top [cm above the map's bottom edge]."""
    from gps_plot import maps

    width, height = maps._map_size_cm(region, projection)
    pad, sp = maps.LEGEND_BOX_PAD_CM, LEGEND_SPACING
    anchor = (0.02, 0.5 * sp + (pad + floor_cm) / height)
    n = len(entries)

    def at(fx: float, line: float) -> tuple[float, float]:
        return maps.region_point(region, anchor[0] + fx / width, anchor[1] + line * sp)

    ref_text = f"{arrow_ref:g} mm/yr" if arrow_ref is not None else ""
    dot_text = f"{dot_ref:g} mm (dots)"
    arrow_cm = arrow_ref * arrow_scale if arrow_ref is not None else 0.0
    dot_x = (max(arrow_cm, maps.text_width_cm(ref_text, LEGEND_FONT)) + 1.0
             if arrow_ref is not None else 0.0)  # fmt: skip
    top_wide = dot_x + maps.text_width_cm(dot_text, LEGEND_FONT)
    lines = n + 1.8
    box = maps.vector_legend_box(
        region, projection, anchor, sp, lines,
        [label for label, _ in entries] + [" " * 2], LEGEND_FONT, top_wide,
    )  # fmt: skip
    fig.plot(x=box[0], y=box[1], fill=maps.LEGEND_BOX_FILL, pen=maps.LEGEND_BOX_PEN,
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
    fig.plot(x=[x0, x0 + bar_deg], y=[y0, y0], pen="2p,black")
    x, y = at(dot_x, n + 0.8)
    fig.text(x=x, y=y, text=dot_text, font=f"{LEGEND_FONT},black", justify="LM")
    return (anchor[1] + (lines - 0.5) * sp) * height + pad


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
        for i, (line, color) in enumerate(text):
            ax.text(0.01, 0.95 - 0.09 * i, line, transform=ax.transAxes,
                    color=color, va="top", fontsize=10)  # fmt: skip
        ax.set_ylabel(f"{name} [mm]")
        ax.grid(True, color="0.85", lw=0.5)
    fig.colorbar(dots, ax=axes, label=f"days since {onset:%d.%m.%Y}",
                 shrink=0.6, aspect=30)  # fmt: skip
    rel = f" relative to {reference}" if reference else ""
    fig.suptitle(f"{tr.marker}: horizontal motion{rel}, trend and seasonal before "
                 "the onset removed", fontsize=12)  # fmt: skip
    if common_mode:
        axes[1].text(0.99, 0.02, "network common mode removed",
                     transform=axes[1].transAxes, ha="right", va="bottom",
                     fontsize=8, color="0.3")  # fmt: skip
    fig.autofmt_xdate()
    return fig
