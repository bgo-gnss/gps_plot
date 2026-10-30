"""``gpsplot direction``: the direction a pair moved, as a config line.

Reads a baseline (rover − reference, unsigned) or a station, daily (the
cleaned plate series) or RTK (``--rtk``), and estimates the direction of
its motion between -s and -e with :func:`gps_analysis.direction.
motion_direction`. Prints one ``station_areas.yaml`` ``directions:`` line
per series, the azimuth to fix for ``--axes motion``, with its σ, plunge
and size in a comment, e.g.::

    SVIE-SKFC: {motion: 101.6}  # ±0.8°, plunge 12.0° ±3.1°, |d_h| 23.4 mm, ...
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
import sys
import textwrap
from collections.abc import Sequence
from typing import Any

import numpy as np

from gps_plot.cli import common

#: a --window: a number and d (days) or h (hours)
WINDOW = re.compile(r"^(\d+(?:\.\d+)?)([dh])$")

EXAMPLES = """examples:
  gpsplot direction SVIN-SKFC SVIE-SKFC -s 20260920 -e 20261029
  gpsplot direction SEY1-SEYD --rtk -s 20261029-12:00 -e 20261029-18:00 --window 30min
"""


def window_days(value: str) -> float:
    """``7d`` → 7, ``6h`` → 0.25 (days); ``30min`` → 1/48."""
    if value.endswith("min") and value[:-3].replace(".", "", 1).isdigit():
        return float(value[:-3]) / 1440.0
    m = WINDOW.match(value)
    if m is None:
        raise argparse.ArgumentTypeError(f"{value!r}: use e.g. 7d, 6h or 30min")
    n = float(m.group(1))
    return n if m.group(2) == "d" else n / 24.0


def build_parser(
    parser: argparse.ArgumentParser | None = None, config: Any = None
) -> argparse.ArgumentParser:
    from gps_plot.rtk import _when

    if parser is None:
        parser = argparse.ArgumentParser(prog="gpsplot direction")
    parser.description = textwrap.fill(
        "The direction each series moved between -s and -e (robust end "
        "levels: the median over --window at each end), printed as a "
        "station_areas.yaml directions: line to fix for --axes motion. A "
        "baseline is rover minus reference, unsigned.",
        width=78,
    )
    parser.formatter_class = argparse.RawDescriptionHelpFormatter
    parser.epilog = EXAMPLES
    parser.add_argument("Series", nargs="+", help="AAAA-BBBB baselines or stations")
    parser.add_argument(
        "-s", "--start", type=_when, required=True, help="YYYYMMDD[-HH:MM]"
    )
    parser.add_argument(
        "-e", "--end", type=_when, required=True, help="YYYYMMDD[-HH:MM]"
    )
    parser.add_argument(
        "--window", type=window_days, default=None,
        help="how far into the interval each end level reaches (default 7d "
        "daily, 1h RTK; 30min, 6h … also)",
    )  # fmt: skip
    parser.add_argument(
        "--rtk", action="store_true",
        help="from the RTK baseline (10-minute medians) instead of daily",
    )  # fmt: skip
    parser.add_argument(
        "--frame", default=None,
        help=f"daily .pos frame (default {common.DEFAULT_PLOT_FRAME})",
    )  # fmt: skip
    parser.add_argument(
        "--slope-from-dem", nargs="?", type=float, const=50.0, default=None,
        metavar="RADIUS_M",
        help="also the DEM slope at the rover (plane over RADIUS_M, default "
        "50 m), as slope: [downslope azimuth, dip] in the line",
    )  # fmt: skip
    return parser


def _daily(
    spec: str, start: dt.datetime, end: dt.datetime, config: Any, frame: str | None
) -> Any:
    """(t [days since 1970], y, σ) of the cleaned daily series, unsigned."""
    from geo_dataread.pos import pos_product
    from gtimes.timefunc import TimetoYearf

    from gps_plot.sources import gps as source

    req = source.SeriesRequest(
        fstart=TimetoYearf(*start.timetuple()[:6]),
        fend=TimetoYearf(*end.timetuple()[:6]),
        ref="plate",
        Dir=common.config_dir(config, "pos_dir"),
        tType="POS",
        uncert=10,
        view="cleaned",
        pos_id=pos_product(frame or common.DEFAULT_PLOT_FRAME, "24h"),
    )
    from geo_dataread.gps_read import toDateTime

    s = source.fetch(spec, req, source.BaselineOptions(sign="raw"))
    t = np.array(list(toDateTime(s.t)), dtype="datetime64[s]").astype(np.int64)
    return t / 86400.0, s.y, s.sigma


def _rtk(spec: str, start: dt.datetime, end: dt.datetime, config: Any) -> Any:
    """(t [days since 1970], y, σ) of the RTK baseline (10-minute medians)."""
    from geo_dataread.rtk_read import list_baselines, rtk_dir

    from gps_plot.sources import rtk as rtk_source

    root = rtk_dir(common.config_dir(config, "rtk_dir"))
    directory, flipped = rtk_source.rtk_direction(spec, list_baselines(root))
    s = rtk_source.fetch_rtk(
        spec, start, end, root=str(root), interval="10min",
        directory=directory, flipped=flipped,
    )  # fmt: skip
    t = np.asarray(s.t, dtype="datetime64[s]").astype(np.int64) / 86400.0
    return t, s.y, s.sigma


def _date(days: float) -> str:
    """Days since 1970 (UTC) as a date and time."""
    return f"{dt.datetime(1970, 1, 1) + dt.timedelta(days=days):%Y-%m-%d %H:%M}"


def line(spec: str, d: Any, what: str, slope: Any = None) -> str:
    """The ``directions:`` line of one result (``slope``: a
    :class:`gps_plot.terrain.Slope` to add)."""
    note = (
        f"±{d.sigma_azimuth:.1f}°, plunge {d.plunge:.1f}° ±{d.sigma_plunge:.1f}°, "
        f"|d_h| {d.horizontal:.1f} mm, {what}"
    )
    if not d.significant:
        note = "NOT significant (|d_h| < 3σ): " + note
    entry = f"motion: {d.azimuth:.1f}"
    if slope is not None:
        entry += f", slope: [{slope.azimuth:.0f}, {slope.dip:.0f}]"
        note += f"; DEM slope over {slope.radius:g} m (rms {slope.rms:.1f} m)"
    return f"{spec}: {{{entry}}}  # {note}"


def estimate(
    spec: str,
    start: dt.datetime,
    end: dt.datetime,
    config: Any,
    *,
    rtk: bool = False,
    window: float | None = None,
    frame: str | None = None,
) -> tuple[Any, float, float]:
    """(:class:`gps_analysis.direction.Direction`, first, last [days since
    1970]) of ``spec``'s motion between ``start`` and ``end``.

    The ends are where the data are: a series that starts late or stops
    early is measured over its own span (returned). ``window`` [days]
    defaults to 7 (daily) or 1 h (RTK), at most half the span.

    Raises:
        ValueError: no data, or too little for an end level.
    """
    from gps_analysis.direction import motion_direction

    if rtk:
        t, y, s = _rtk(spec, start, end, config)
    else:
        t, y, s = _daily(spec, start, end, config, frame)
    have = np.isfinite(np.asarray(y, dtype=float)).all(axis=0)
    if not have.any():
        raise ValueError("no data between -s and -e")
    first, last = float(t[have].min()), float(t[have].max())
    w = window or (1.0 / 24.0 if rtk else 7.0)
    return (
        motion_direction(t, y, s, first, last, min(w, (last - first) / 2)),
        first,
        last,
    )


#: START:END of --direction-from (each YYYYMMDD or YYYYMMDD-HH:MM)
SPAN = re.compile(r"^(\d{8}(?:-\d\d:\d\d)?):(\d{8}(?:-\d\d:\d\d)?)$")


def parse_span(value: str) -> tuple[dt.datetime, dt.datetime]:
    """``--direction-from START:END`` (YYYYMMDD[-HH:MM] each)."""
    from gps_plot.rtk import _when

    m = SPAN.match(value)
    if m is None:
        raise argparse.ArgumentTypeError(f"{value!r}: use YYYYMMDD:YYYYMMDD")
    start, end = _when(m.group(1)), _when(m.group(2))
    if start >= end:
        raise argparse.ArgumentTypeError(f"{value!r}: start is not before end")
    return start, end


def parse_slope(value: str) -> tuple[float, float]:
    """``--slope AZ,DIP`` [°]."""
    try:
        az, dip = (float(x) for x in value.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r}: use AZIMUTH,DIP") from None
    if not (0 <= az < 360 and 0 <= dip < 90):
        raise argparse.ArgumentTypeError(f"{value!r}: azimuth 0..360, dip 0..90")
    return az, dip


def config_directions(config: Any) -> dict[str, dict[str, Any]]:
    """station_areas.yaml ``directions:`` of the deployed config ({} if none).

    A malformed table is an error (it is configuration, not data).
    """
    from pathlib import Path

    from gps_plot import products

    path_dir = getattr(config, "config_path", None)
    if path_dir is None:
        return {}
    path = Path(path_dir) / "station_areas.yaml"
    if not path.is_file():
        return {}
    return products.directions_table(products.load_yaml(path))


def resolve_directions(
    pairs: Sequence[str],
    mode: str,
    *,
    motion: float | None = None,
    slope: tuple[float, float] | None = None,
    span: tuple[dt.datetime, dt.datetime] | None = None,
    config: Any = None,
    frame: str | None = None,
    rtk: bool = False,
) -> dict[str, dict[str, Any]]:
    """Each pair's directions for ``--axes motion|slope``.

    Per key, the first of: the command line (``motion``/``slope``, for every
    pair), the motion estimated over ``span`` (``--direction-from``; motion
    only, reported on stderr), the config's ``directions:``. Pairs with none
    are left out (they are dropped from the plot). ``{}`` for other modes.
    """
    from gps_plot.axes import GIVEN_MODES

    if mode not in GIVEN_MODES:
        return {}
    table = config_directions(config) if config is not None else {}
    out: dict[str, dict[str, Any]] = {}
    for pair in pairs:
        got = dict(table.get(pair.upper(), {}))
        if span is not None and mode == "motion" and motion is None:
            try:
                d, first, last = estimate(
                    pair, span[0], span[1], config, rtk=rtk, frame=frame
                )
            except Exception as exc:
                print(f"{pair}: --direction-from: {exc}", file=sys.stderr)
            else:
                got["motion"] = d.azimuth
                mark = "" if d.significant else " (NOT significant)"
                print(
                    f"{pair}: motion {d.azimuth:.1f}° ±{d.sigma_azimuth:.1f}° over "
                    f"{_date(first)}…{_date(last)}{mark}",
                    file=sys.stderr,
                )
        if motion is not None:
            got["motion"] = float(motion)
        if slope is not None:
            got["slope"] = slope
        if got.get(mode) is not None:
            out[pair] = got
    return out


def add_direction_options(parser: argparse.ArgumentParser, dest: str = "") -> None:
    """``--direction`` / ``--slope`` / ``--direction-from`` (``dest`` prefix)."""
    parser.add_argument(
        "--direction", dest=f"{dest}direction", type=float, default=None,
        metavar="AZ",
        help="--axes motion: the motion azimuth [deg] for every pair, instead "
        "of station_areas.yaml directions:",
    )  # fmt: skip
    parser.add_argument(
        "--slope", dest=f"{dest}slope", type=parse_slope, default=None,
        metavar="AZ,DIP",
        help="--axes slope: downslope azimuth and dip [deg] for every pair",
    )  # fmt: skip
    parser.add_argument(
        "--direction-from", dest=f"{dest}direction_from", type=parse_span,
        default=None, metavar="START:END",
        help="--axes motion: estimate each pair's motion direction over "
        "START:END (YYYYMMDD) from its daily series, as gpsplot direction",
    )  # fmt: skip


def run(args: argparse.Namespace, parser: argparse.ArgumentParser, config: Any) -> int:
    if args.start >= args.end:
        parser.error(f"start {args.start} is not before end {args.end}")
    window = args.window or (1.0 / 24.0 if args.rtk else 7.0)
    kind = "RTK" if args.rtk else "daily"
    failed = 0
    for spec in args.Series:
        try:
            d, start, end = estimate(
                spec, args.start, args.end, config,
                rtk=args.rtk, window=window, frame=args.frame,
            )  # fmt: skip
        except Exception as exc:  # one bad series must not sink the rest
            print(f"# {spec}: {exc}", file=sys.stderr)
            failed += 1
            continue
        slope = None
        if args.slope_from_dem is not None:
            from gps_plot import terrain
            from gps_plot.map import gis_dir

            try:
                slope = terrain.station_slope(
                    spec[:4], gis_dir(config), radius=args.slope_from_dem
                )
            except (OSError, ValueError) as exc:
                print(f"# {spec}: no DEM slope ({exc})", file=sys.stderr)
        what = f"{_date(start)}…{_date(end)} {kind}, {window:g} d ends"
        print(line(spec, d, what, slope))
    return 1 if failed == len(args.Series) else 0
