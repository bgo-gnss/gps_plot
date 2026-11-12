"""gpsplot rtk-tie: a landslide station in a stable frame, within the day.

RTK gives only the pairs rtk.vedur.is computes; the daily (24 h) GAMIT
solutions give the frame (BGÓ 2026-11-12). The station's daily positions
are taken against the reference stations' weighted mean (the track map's
``--track-ref``, :func:`gps_plot.track.reference_mean`), and an RTK
baseline from it to a neighbour adds the motion WITHIN the days:

    station(t) = daily(t) + [b(t) − b_day(t)]

``daily(t)``, ``b_day(t)`` = the daily positions and the RTK baseline's own
daily medians, at noon, linear between noons. After the newest daily
solution (≈ the last 24–32 h) the station is that day's position plus the
RTK change since that day's tie: provisional, shaded on the figure.

Route 1 (the default, ``--via SVIN-SVIE``): the RTK pair to a neighbour on
the slide, the quietest RTK (fixed, 1-h medians, 7-day sidereal filter as
:mod:`gps_plot.rtk_accel`); it assumes the neighbour moves evenly within
the day (its motion comes from its own daily positions only). ``--compare``
draws route 2, the pair to the stable end (SKFC), ~10x noisier.

Experiment and hold-out check: ``~/gps-data/curation/rtk-daily-tie``.

    gpsplot rtk-tie SVIN --via SVIN-SVIE --compare SVIN-SKFC \\
        --refs SKFC,SLEC,FAGC,KVSK --event svinafellsheidi-2026-09-landslide \\
        --area svinafellsheidi --days 2,7 --out DIR
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

COMP = ("N", "E", "U")
NOON = pd.Timedelta(hours=12)
#: data before the plotted span: the sidereal filter's 7 days + its 24 h median
RTK_WARMUP = pd.Timedelta(days=9)


# --- the series -------------------------------------------------------------


def daily_in_frame(
    stations: Sequence[str], refs: Sequence[str], onset: dt.date
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Each station's daily N/E/U [mm] minus the reference stations'
    weighted mean (trend + seasonal of each fitted before ``onset``), one
    row per day at noon; and its empirical σ [mm] (:func:`empirical_sigma`:
    the scatter of this series, so the references' noise is in it)."""
    from gps_plot import track

    pre = track.pre_window(onset)
    hor, ups = [], []
    for r in refs:
        t, y = track.load_series(r, pre, up=True)
        hor.append((t, y[:2]))
        ups.append(track.up_series(t, y))
    th, yh = track.reference_mean(hor, pre)
    tu, yu = track.reference_mean(ups, pre)
    ref = pd.DataFrame(
        {
            "N": pd.Series(yh[0], track.day_key(th)),
            "E": pd.Series(yh[1], track.day_key(th)),
            "U": pd.Series(yu[0], track.day_key(tu)),
        }
    )
    values, sigmas = {}, {}
    for s in stations:
        t, y = track.load_series(s, pre, up=True)
        d = pd.DataFrame(y.T, index=track.day_key(t), columns=list(COMP))
        d = (d.groupby(level=0).mean() - ref).dropna(how="all").sort_index()
        sig = empirical_sigma(d, _sigma(s, pre).reindex(d.index))
        for frame in (d, sig):
            frame.index = pd.DatetimeIndex(
                [pd.Timestamp(dt.date.fromordinal(int(o))) + NOON for o in frame.index]
            ).as_unit("ns")
        values[s], sigmas[s] = d, sig
    return values, sigmas


def empirical_sigma(values: pd.DataFrame, formal: pd.DataFrame) -> pd.DataFrame:
    """Daily σ from the scatter of the series itself (BGÓ 2026-11-12, as the
    baseline stacks' ``--baseline-sigma empirical``): per component the
    day-to-day noise 1.4826·MAD(Δ)/√2 over consecutive days (slow motion
    cancels in Δ), and each day's formal σ scaled so their median is that
    noise (a day's relative quality kept). ``values``/``formal`` indexed by
    proleptic day ordinal. Formal σ where too few consecutive days (< 10)."""
    out = formal.copy()
    days = np.asarray(values.index, dtype=float)
    consecutive = np.diff(days) == 1.0
    for c in COMP:
        dv = np.diff(values[c].to_numpy(float))[consecutive]
        dv = dv[np.isfinite(dv)]
        med_formal = float(np.nanmedian(formal[c].to_numpy(float)))
        if dv.size < 10 or not np.isfinite(med_formal) or med_formal <= 0:
            continue
        noise = 1.4826 * float(np.median(np.abs(dv - np.median(dv)))) / np.sqrt(2.0)
        out[c] = formal[c] * (noise / med_formal)
    return out


def _sigma(marker: str, pre: tuple[dt.date, dt.date]) -> pd.DataFrame:
    """The station's daily formal σ N/E/U [mm] by day (proleptic ordinal),
    read as :func:`gps_plot.track.load_series` reads its values."""
    from geo_dataread.products.loader import Source, load
    from geo_dataread.products.velocity_core import window_bounds

    from gps_plot import track

    t0, t1 = window_bounds(pre[0], dt.date.today())
    v = load(marker, t0, t1, Source(frame="itrf14", view="cleaned", ref="plate",
                                    uncert=1e9, pos_dir=None))  # fmt: skip
    df = pd.DataFrame(np.asarray(v.sigma[:3], float).T, index=track.day_key(v.t),
                      columns=list(COMP))  # fmt: skip
    return df.groupby(level=0).mean()


def rtk_binned(
    read: Any, pair: str, start: dt.datetime, end: dt.datetime, every: str = "1h"
) -> pd.DataFrame:
    """Medians over ``every`` (end-labelled; 10 min or 1 h) [mm] of the
    fixed RTK solutions of ``pair`` (A − B), despiked and sidereal-filtered
    as the acceleration warning, from ``start`` on (the filter warms up on
    the days before)."""
    from gps_plot import rtk_accel

    t, y = read(pair, (pd.Timestamp(start) - RTK_WARMUP).to_pydatetime(), end)
    if not np.size(t):
        return pd.DataFrame(columns=list(COMP), dtype=float)
    out = rtk_accel.sidereal(rtk_accel.despike_coarse(rtk_accel.ten_min(t, y)))
    if pd.Timedelta(every) != pd.Timedelta("10min"):
        out = out.resample(every, label="right", closed="right").median()
    out.columns = list(COMP)
    out.index = out.index.as_unit("ns")
    return out[out.index >= pd.Timestamp(start)]


def rtk_series(
    read: Any, pair: str, start: dt.datetime, end: dt.datetime
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(one-minute, 1-h medians)`` [mm] of ``pair`` from ``start`` on, one
    read: the 1-h series as :func:`rtk_binned`; the one-minute values
    despiked and minus the 10-min chain's multipath model, linear between
    its epochs (BGÓ 2026-11-12: the 6 h / 12 h figures at the RTK plots'
    own resolution)."""
    from gps_plot import rtk_accel

    empty = pd.DataFrame(columns=list(COMP), dtype=float)
    t, y = read(pair, (pd.Timestamp(start) - RTK_WARMUP).to_pydatetime(), end)
    if not np.size(t):
        return empty, empty
    ten = rtk_accel.despike_coarse(rtk_accel.ten_min(t, y))
    filtered = rtk_accel.sidereal(ten)
    hour = filtered.resample("1h", label="right", closed="right").median()
    model = (ten - filtered).astype(float)
    one = pd.DataFrame(np.asarray(y, float).T, index=pd.DatetimeIndex(t),
                       columns=["n", "e", "u"])  # fmt: skip
    one = rtk_accel.despike_coarse(one)
    both = model.reindex(model.index.union(one.index)).interpolate(
        method="time", limit_area="inside"
    )  # fmt: skip
    minute = one - both.reindex(one.index)
    out = []
    for df in (minute, hour):
        df = df.copy()
        df.columns = list(COMP)
        df.index = pd.DatetimeIndex(df.index).as_unit("ns")
        out.append(df[df.index >= pd.Timestamp(start)])
    return out[0], out[1]


def rtk_hourly(
    read: Any, pair: str, start: dt.datetime, end: dt.datetime
) -> pd.DataFrame:
    """:func:`rtk_binned` over 1 h."""
    return rtk_binned(read, pair, start, end, "1h")


def day_medians(b: pd.DataFrame, min_frac: float = 0.5) -> pd.DataFrame:
    """The baseline's median per UTC day (bins over at least ``min_frac``
    of the day), at noon."""
    step = pd.Series(b.index).diff().median() if len(b) > 1 else pd.Timedelta("1h")
    per_day = pd.Timedelta("1D") / (step if pd.notna(step) and step > pd.Timedelta(0)
                                    else pd.Timedelta("1h"))  # fmt: skip
    day = (b.index - pd.Timedelta("1ns")).floor("1D")  # bins are end-labelled
    g = b.groupby(day)
    m = g.median()[g.count().min(axis=1) >= min_frac * per_day]
    m.index = (m.index + NOON).as_unit("ns")
    return m


def _interp(daily: pd.DataFrame, t: pd.DatetimeIndex) -> pd.DataFrame:
    x = pd.DatetimeIndex(daily.index).as_unit("ns").asi8.astype(float)
    xt = pd.DatetimeIndex(t).as_unit("ns").asi8.astype(float)
    return pd.DataFrame(
        {
            c: np.interp(xt, x, daily[c].to_numpy(float), left=np.nan, right=np.nan)
            for c in COMP
        },
        index=t,
    )


def tie(
    daily: pd.DataFrame, b: pd.DataFrame
) -> tuple[pd.DataFrame, pd.Timestamp] | None:
    """The station within the day: ``(series, newest daily noon)``; None
    without a common day."""
    bd = day_medians(b)
    common = daily.dropna().index.intersection(bd.index)
    if not len(common) or b.empty:
        return None
    d, bd = daily.loc[common], bd.loc[common]
    t = b.index
    base = _interp(d, t)
    anom = b - _interp(bd, t)
    tail = t > d.index[-1]
    base.loc[tail] = d.iloc[-1].to_numpy()
    anom.loc[tail] = (b.loc[tail] - bd.iloc[-1]).to_numpy()
    return base + anom, d.index[-1]


def rotate(df: pd.DataFrame, azimuth: float) -> pd.DataFrame:
    """N/E/U → along ``azimuth``, across (90° clockwise of it), Up."""
    a = np.radians(azimuth)
    along = df["N"] * np.cos(a) + df["E"] * np.sin(a)
    across = -df["N"] * np.sin(a) + df["E"] * np.cos(a)
    return pd.DataFrame({"along": along, "across": across, "U": df["U"]})


# --- the figure -------------------------------------------------------------


def _series(
    label: str,
    df: pd.DataFrame,
    sig: pd.DataFrame | None,
    azimuth: float | None,
    source: str,
) -> Any:
    """A :class:`gps_plot.series.Series` (t datetime64, y/sigma (3, N) mm)
    on N/E/U or, with ``azimuth``, along/across/Up."""
    from gps_plot.series import Series

    v = rotate(df, azimuth) if azimuth is not None else df
    y = v.to_numpy(float).T
    if sig is None:
        sigma = np.full_like(y, np.nan)
    else:
        s = sig.reindex(df.index).to_numpy(float).T
        if azimuth is not None:  # σ of a rotated pair: the larger horizontal one
            h = np.nanmax(s[:2], axis=0)
            s = np.stack([h, h, s[2]])
        sigma = s
    t = pd.DatetimeIndex(df.index).as_unit("ns").to_numpy()
    return Series(label=label, t=t, y=y, sigma=sigma, meta={"source": source})


def _zeroed(df: pd.DataFrame, start: pd.Timestamp, days: int = 7) -> pd.DataFrame:
    """Minus the median of the first ``days`` days from ``start``, at least
    its first three solutions (each station on its own zero, as the
    baseline stacks)."""
    inside = df[df.index >= start].dropna(how="all")
    first = inside[inside.index < start + pd.Timedelta(days=days)]
    if len(first) < 3:  # a station solved on few days (SVIN): its first three
        first = inside.iloc[:3]
    return df - (first if len(first) else df).median()


def _clip_to(main: pd.DataFrame, other: pd.DataFrame) -> pd.DataFrame:
    """``other`` with values outside ``main``'s range (+ a margin) as NaN:
    the noisy comparison must not set the scale."""
    lo, hi = main.min(), main.max()
    pad = np.maximum(3.0, 0.5 * (hi - lo))
    return other.where((other >= lo - pad) & (other <= hi + pad))


#: days averaged at either end of the movement since the onset (the
#: polar figure's: 7 after the onset, the newest 2)
ONSET_DAYS, END_DAYS = 7.0, 2.0
#: the movement must stand this many σ above its noise for a direction
MOTION_SIGMA = 2.0


def onset_direction(
    values: pd.DataFrame, sigma: pd.DataFrame, onset: dt.date
) -> float | None:
    """The azimuth [°] of the horizontal movement since ``onset``: the
    median of the newest ``END_DAYS`` minus that of the first
    ``ONSET_DAYS`` after it (at least three solutions each end), when it
    stands ``MOTION_SIGMA`` σ out of the noise; None otherwise."""
    on = pd.Timestamp(onset)
    after = values[values.index >= on][["N", "E"]].dropna()
    if len(after) < 6:
        return None
    first = after[after.index < on + pd.Timedelta(days=ONSET_DAYS)]
    first = first if len(first) >= 3 else after.iloc[:3]
    last = after[after.index > after.index[-1] - pd.Timedelta(days=END_DAYS)]
    last = last if len(last) >= 3 else after.iloc[-3:]
    d = (last.median() - first.median()).to_numpy(float)
    size = float(np.hypot(*d))
    noise = sigma[["N", "E"]].median().to_numpy(float)
    sig = float(np.hypot(*noise)) * np.sqrt(1.0 / len(first) + 1.0 / len(last))
    if not np.isfinite(size) or size <= MOTION_SIGMA * sig:
        return None
    return float(np.degrees(np.arctan2(d[1], d[0])) % 360.0)


def _ylabels(azimuths: Sequence[float | None]) -> list[str] | None:
    """Motion / Across / Up labels (gps_plot.labels, as the baseline
    figures' ``--axes motion``), the motion azimuth(s) in the first."""
    from gps_plot.labels import component

    if all(a is None for a in azimuths):
        return None  # the frame's own North/East/Up
    which = ", ".join(f"{a:.0f}°" for a in azimuths if a is not None)
    return [
        f"{component('motion', 'en')} {which} [mm]",
        f"{component('across', 'en')} [mm]",
        f"{component('up', 'en')} [mm]",
    ]


def figure(
    series: Sequence[Any],
    styles: Sequence[str],
    colors: Sequence[str],
    *,
    main: str,
    sub: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    last: dt.datetime,
    now: dt.datetime,
    rtk: bool,
    ylabels: Sequence[str] | None = None,
    ylim: Sequence[Any] = (),
    note: str | None = None,
) -> Any:
    """The figure in the RTK / baseline-stack template (compose.stack_figure:
    the stack frame, logo, last-epoch title; with ``rtk`` the plot-time line)."""
    from gps_plot import compose, rtk_classic
    from gps_plot import timesmatplt as tplt

    current = now - last < (dt.timedelta(hours=1) if rtk else dt.timedelta(days=2))
    title = tplt.StationTitle(
        main=r"\Huge %s \ \Large %s" % (main, sub),
        status=(r"\LARGE Last epoch: %s UTC" % last.strftime("%d %b %Y %H:%M"))
        if rtk else (r"\LARGE Last datapoint: %s" % last.strftime("%d %b %Y")),
        created=r"\large (Plot created on %s UTC)" % now.strftime("%b %d %Y %H:%M"),
        status_color=tplt.STATUS_CURRENT_COLOR if current else tplt.STATUS_STALE_COLOR,
    )  # fmt: skip
    fig = compose.stack_figure(
        list(series), title=title, labels=[s.label for s in series],
        start=start.to_pydatetime(), end=end.to_pydatetime(), ylim=ylim,
        styles=list(styles), colors=list(colors), fig=tplt._reusable_figure(),
        robust_lines=rtk, ylabels=ylabels,
    )  # fmt: skip
    if note:
        fig.axes[2].text(0.01, 0.98, note, transform=fig.axes[2].transAxes,
                         fontsize="x-small", color="0.3", va="top", wrap=True)  # fmt: skip
    if rtk:
        rtk_classic.now_line(fig.axes[:3], now=min(now, end.to_pydatetime()), last=last,
                             fontsize=12)  # fmt: skip
    tplt.inpLogo(fig)  # type: ignore[no-untyped-call]
    return fig


# --- the command ------------------------------------------------------------


def build_parser(prog: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog=prog, description=(__doc__ or "").split("\n\n")[0])
    p.add_argument("stations", nargs="+", help="landslide stations, e.g. SVIN SVIE")
    p.add_argument("--via", action="append", default=[],
                   help="RTK pair tying one of the stations, repeatable, e.g. "
                   "--via SVIN-SVIE --via SVIE-SKFC: the station is the end "
                   "listed in the stations (A first), the other end its anchor. "
                   "One RTK figure per station, and one of them all together")  # fmt: skip
    p.add_argument("--compare", action="append", default=[],
                   help="a second RTK pair of a station to draw in its own "
                   "figure, repeatable, e.g. SVIN-SKFC")  # fmt: skip
    p.add_argument("--refs", required=True, help="reference stations, e.g. SKFC,SLEC")
    p.add_argument("--event", default=None,
                   help="events.yaml id whose onset ends the references' trend window")  # fmt: skip
    p.add_argument("--onset", default=None, help="that onset instead, YYYY-MM-DD")
    p.add_argument("--area", default=None,
                   help="station_areas.yaml area: directions: motion of "
                   "<station>-<first ref> for --axes motion")  # fmt: skip
    p.add_argument("--axes", choices=("neu", "motion"), default="neu",
                   help="N/E/U (default) or along the station's directions: "
                   "motion, across it (90° clockwise), Up; files get -motion")  # fmt: skip
    p.add_argument("--days", default="2,7",
                   help="spans of the RTK figures, days (comma separated; under "
                   "a day, e.g. 0.25 = 6 h, with the one-minute values, named -6h)")  # fmt: skip
    p.add_argument("--daily-days", default="",
                   help="spans of the daily-only figures of all the stations, "
                   "days, e.g. 90,365 (none by default)")  # fmt: skip
    p.add_argument("--colors", default="",
                   help="station colours, e.g. SVIN=red,SVIE=#2ca02c")  # fmt: skip
    p.add_argument("--ylim", type=float, default=None,
                   help="daily figures: ± this many mm around the data at least")  # fmt: skip
    p.add_argument("--out", required=True, help="directory of the figures")
    p.add_argument("--end", default=None, help="UTC 'YYYY-MM-DD HH:MM' (default now)")
    return p


def _onset(args: argparse.Namespace, config: Any) -> dt.date:
    """The onset DATE (the trend window ends a week before it)."""
    from gps_plot import products

    if args.onset:
        return dt.date.fromisoformat(args.onset)
    if not args.event:
        raise ValueError("--event or --onset")
    events = products.load_yaml(Path(config.config_path).expanduser() / "events.yaml")
    for ev in events.get("events") or []:
        if ev.get("id") == args.event and ev.get("onset"):
            return dt.date.fromisoformat(str(ev["onset"]))
    raise ValueError(f"events.yaml: no onset for {args.event}")


def _azimuth(config: Any, area: str | None, pair: str) -> float | None:
    from gps_plot import products

    if not area:
        return None
    table = products.area_table(
        products.load_yaml(Path(config.config_path).expanduser() / "station_areas.yaml")
    )
    got: Mapping[str, Any] = (table.get(area) or {}).get("directions") or {}
    motion = (got.get(pair.upper()) or {}).get("motion")
    return None if motion is None else float(motion)


def _event_lines(config: Any, area: str | None) -> dict[str, list[str]]:
    """The area's events.yaml onsets as ``{"YYYYMMDD": [colour]}`` for
    ``timesmatplt.addEvent``, coloured as the baseline figures (plots.yaml
    ``event_colours``; BGÓ 2026-11-12: the 23.9. onset on the daily
    figures); empty without an area or file."""
    from gps_plot import products

    if not area:
        return {}
    base = Path(config.config_path).expanduser()
    try:
        events = products.load_yaml(base / "events.yaml").get("events") or []
        plots = products.load_yaml(base / "plots.yaml")
    except OSError:
        return {}
    colours = {**products.DEFAULT_EVENT_COLOURS, **(plots.get("event_colours") or {})}
    out: dict[str, list[str]] = {}
    for spec in products.event_specs(events, [area], colours):
        day, colour = spec.split(",", 1)
        out[day] = [colour]
    return out


def _save(fig: Any, path: Path, prog: str) -> None:
    import matplotlib.pyplot as plt

    from gps_plot import timesmatplt as tplt

    tmp = path.with_name(f".{path.stem}")  # saveFig adds .png
    tplt.saveFig(str(tmp), "png", fig)
    os.replace(f"{tmp}.png", path)
    plt.close(fig)
    print(f"{prog}: {path}")


def _route(pair: str, stations: Sequence[str]) -> tuple[str, str, float]:
    """``(station, anchor, sign)`` of an RTK pair A-B: the station is the end
    among ``stations`` (A first), the anchor the other end; sign −1 when the
    station is B (station − anchor = −(A − B))."""
    a, _, b = pair.upper().partition("-")
    if a in stations:
        return a, b, 1.0
    if b in stations:
        return b, a, -1.0
    raise ValueError(f"--via/--compare {pair}: neither end is one of {list(stations)}")


def _span_name(days: float) -> str:
    return f"{days * 24:g}h" if days < 1 else f"{days:g}d"


def run(prog: str, argv: list[str]) -> int:
    import matplotlib

    matplotlib.use("Agg")
    from geo_dataread.rtk_read import list_baselines, rtk_dir

    from gps_plot import compose, rtk_status
    from gps_plot.cli import common
    from gps_plot.plot_gps_timeseries import _load_config

    args = build_parser(prog).parse_args(argv)
    config = _load_config()
    stations = [s.upper() for s in args.stations]
    refs = [r.strip().upper() for r in args.refs.split(",") if r.strip()]
    rtk_spans = sorted(float(x) for x in args.days.split(",") if x.strip())
    daily_spans = sorted(float(x) for x in args.daily_days.split(",") if x.strip())
    now = dt.datetime.now(dt.UTC).replace(tzinfo=None, second=0, microsecond=0)
    end = pd.Timestamp(args.end) if args.end else pd.Timestamp(now)
    out = Path(args.out).expanduser()
    if str(out.resolve()).startswith("/mnt_data"):
        print(f"{prog}: never write under /mnt_data", file=sys.stderr)
        return 2
    given = dict(kv.split("=", 1) for kv in args.colors.split(",") if "=" in kv)
    palette = compose.series_colors(stations)
    colors = {s: given.get(s, c) for s, c in zip(stations, palette, strict=True)}
    along = args.axes == "motion"
    tail = "-motion" if along else ""
    frame_name = "mean of " + ", ".join(refs)
    #: station -> {"1h"|"1min": (series, newest daily noon)}, its pair
    tied: dict[str, dict[str, Any]] = {}
    via_of: dict[str, str] = {}
    compare: dict[str, tuple[str, dict[str, Any]]] = {}
    try:
        onset = _onset(args, config)
        routes = [_route(p, stations) for p in args.via]
        anchors = [a for _, a, _ in routes if a not in stations]
        daily, sigma = daily_in_frame(stations + sorted(set(anchors)), refs, onset)
        # the motion axis: the movement since the onset in this frame (as the
        # polar figure's, BGÓ 2026-11-12: the long-term direction, not the
        # 14-day rate's); station_areas.yaml directions: motion when there
        # is no movement above its noise yet
        azim: dict[str, float | None] = {s: None for s in stations}
        how: dict[str, str] = {}
        if along:
            for s in stations:
                got_az = onset_direction(daily[s], sigma[s], onset)
                if got_az is not None:
                    azim[s], how[s] = got_az, "since onset"
                else:
                    azim[s] = _azimuth(config, args.area, f"{s}-{refs[0]}")
                    how[s] = "directions: motion"
            if any(a is None for a in azim.values()):
                raise ValueError("--axes motion: a station has neither movement "
                                 "since the onset nor a directions: motion "
                                 f"<station>-{refs[0]} in --area")  # fmt: skip
        if routes and rtk_spans:
            read = rtk_status.rtk_reader(
                list_baselines(rtk_dir(common.config_dir(config, "rtk_dir")))
            )
            span0 = (end - pd.Timedelta(days=max(rtk_spans) + 1)).to_pydatetime()
            for pair, (sta, anchor, sign) in zip(args.via, routes, strict=True):
                b1, b = (
                    sign * x for x in rtk_series(read, pair, span0, end.to_pydatetime())
                )
                # the station's own days; on days without one (SVIN is solved
                # on fewer days than SVIE) the anchor's + the pair's daily
                # median: station = anchor + (station − anchor)
                plus = daily[anchor] + day_medians(b).reindex(daily[anchor].index)
                base = daily[sta].combine_first(plus)
                # 1-h medians for the days-long figures, the one-minute
                # values for those under a day (BGÓ 2026-11-12)
                got = {"1h": tie(base, b), "1min": tie(base, b1)}
                if got["1h"] is None:
                    print(f"{prog}: {pair}: no day in common with the daily solutions",
                          file=sys.stderr)  # fmt: skip
                    continue
                tied[sta], via_of[sta] = got, pair.upper()
            for pair in args.compare:
                sta, _, sign = _route(pair, stations)
                c1, c1h = (
                    sign * x for x in rtk_series(read, pair, span0, end.to_pydatetime())
                )
                compare[sta] = (pair.upper(), {
                    k: (g[0] if (g := tie(daily[sta], c)) else None)
                    for k, c in (("1h", c1h), ("1min", c1))
                })  # fmt: skip
    except (OSError, ValueError, KeyError) as exc:
        print(f"{prog}: {exc}", file=sys.stderr)
        return 1
    out.mkdir(parents=True, exist_ok=True)

    def motion_note(names: Sequence[str]) -> str:
        """The motion axis of ``names``, on its own line of the note."""
        if not along:
            return ""
        parts = [f"{n} {azim[n]:.0f}°" + ("" if how.get(n) == "since onset"
                                         else " (directions: motion)")
                 for n in names]  # fmt: skip
        return "\nmotion axis = the movement since the onset: " + ", ".join(parts)

    def piece(
        sta: str, days: float
    ) -> tuple[pd.DataFrame, pd.Series, pd.Timestamp] | None:
        """The station's tied series over the span's keep window, its zero
        (its first hour, or six for days-long spans) and newest daily noon."""
        got = tied[sta]["1min" if days < 1 else "1h"]
        if got is None:
            return None
        series, newest = got
        start = end - pd.Timedelta(days=days)
        main = series.loc[start - pd.Timedelta(days=1) : end]
        head = main[(main.index >= start)
                    & (main.index < start + pd.Timedelta(hours=1 if days < 1 else 6))]  # fmt: skip
        return main, (head if len(head) else main).median(), newest

    def resolution(days: float) -> str:
        return "1-min values" if days < 1 else "1-h medians"

    for days in rtk_spans:
        start = end - pd.Timedelta(days=days)
        keep = slice(start - pd.Timedelta(days=1), end)
        made: list[tuple[str, pd.DataFrame, pd.Series, pd.Timestamp]] = []
        for sta in stations:
            if sta not in tied or (span_got := piece(sta, days)) is None:
                continue
            main, zero, newest = span_got
            made.append((sta, main, zero, newest))
            ser, sty, col = [], [], []
            cmp = compare.get(sta)
            if cmp is not None and cmp[1].get("1min" if days < 1 else "1h") is not None:
                other = cmp[1]["1min" if days < 1 else "1h"].loc[keep] - zero
                ser.append(_series(f"{sta} via {cmp[0]} (comparison)",
                                   _clip_to(main - zero, other), None, azim[sta], "rtk"))  # fmt: skip
                sty.append("line")
                col.append("0.7")
            ser += [
                _series(f"{sta} (RTK {via_of[sta]})", main - zero, None, azim[sta], "rtk"),
                _series(f"{sta} daily", daily[sta].loc[keep] - zero,
                        sigma[sta].loc[keep], azim[sta], "daily"),
            ]  # fmt: skip
            sty += ["line", "marked"]
            col += [colors[sta], colors[sta]]
            note = (
                f"RTK {via_of[sta]} ({resolution(days)}) tied to the daily "
                f"solutions (daily medians, linear between noons); after "
                f"{newest:%d %b}: provisional{motion_note([sta])}"
            )
            last = min(main.dropna(how="all").index[-1].to_pydatetime(), now)
            fig = figure(ser, sty, col, main=f"{sta}: RTK + daily",
                         sub=f"frame: {frame_name}", start=start, end=end, last=last,
                         now=now, rtk=True,
                         ylabels=_ylabels([azim[sta]]) if along else None,
                         note=note)  # fmt: skip
            _save(fig, out / f"{sta}-tied-{_span_name(days)}{tail}.png", prog)
        if len(made) < 2:
            continue
        # all the stations together, each on its own zero and motion azimuth
        ser, sty, col = [], [], []
        for sta, main, zero, _newest in made:
            what = f"{sta} (RTK {via_of[sta]}" + (
                f", {azim[sta]:.0f}°)" if azim[sta] is not None else ")"
            )
            ser.append(_series(what, main - zero, None, azim[sta], "rtk"))
            sty.append("line")
            col.append(colors[sta])
        for sta, _main, zero, _newest in made:
            ser.append(_series(f"{sta} daily", daily[sta].loc[keep] - zero,
                               sigma[sta].loc[keep], azim[sta], "daily"))  # fmt: skip
            sty.append("marked")
            col.append(colors[sta])
        names = [m[0] for m in made]
        newest_all = min(m[3] for m in made)
        note = (
            f"RTK ({resolution(days)}) "
            + ", ".join(f"{s} via {via_of[s]}" for s in names)
            + f", tied to the daily solutions; after {newest_all:%d %b}: provisional"
            + motion_note([m[0] for m in made])
        )
        last = min(
            max(m[1].dropna(how="all").index[-1] for m in made).to_pydatetime(), now
        )
        fig = figure(ser, sty, col, main=f"{', '.join(names)}: RTK + daily",
                     sub=f"frame: {frame_name}", start=start, end=end, last=last,
                     now=now, rtk=True,
                     ylabels=_ylabels([azim[s] for s in names]) if along else None,
                     note=note)  # fmt: skip
        _save(fig, out / f"{'_'.join(names)}-tied-{_span_name(days)}{tail}.png", prog)

    ylabels = _ylabels([azim[s] for s in stations]) if along else None
    event_lines = _event_lines(config, args.area) if daily_spans else {}
    for days in daily_spans:
        start = end - pd.Timedelta(days=days)
        keep = slice(start - pd.Timedelta(days=1), end)
        # each station on its own motion azimuth: said in the legend
        ser = [_series(s if azim[s] is None else f"{s} ({azim[s]:.0f}°)",
                       _zeroed(daily[s].loc[keep], start), sigma[s].loc[keep],
                       azim[s], "daily")
               for s in stations]  # fmt: skip
        last_day = max(daily[s].dropna(how="all").index[-1] for s in stations)
        fig = figure(ser, ["points"] * len(ser), [colors[s] for s in stations],
                     main=f"{', '.join(stations)}",
                     sub=f"frame: {frame_name}, plate removed",
                     start=start, end=end, last=last_day.to_pydatetime(), now=now,
                     rtk=False, ylabels=ylabels,
                     ylim=[args.ylim] if args.ylim else [],
                     note=f"daily (24 h), cleaned, minus the weighted daily mean of "
                     f"{', '.join(refs)} (trend + seasonal removed); empirical "
                     f"$\\sigma$"
                     f"{motion_note(stations)}")  # fmt: skip
        # on the three data panels only (addEvent would draw on the logo's
        # axes too)
        for day, (colour, *_rest) in event_lines.items():
            when = dt.datetime.strptime(day, "%Y%m%d")
            for ax in fig.axes[:3]:
                ax.axvline(x=when, color=colour, zorder=2)
        _save(fig, out / f"{'_'.join(stations)}-frame-{days:g}d{tail}.png", prog)
    return 0
