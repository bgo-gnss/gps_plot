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
    return parser


def _daily(
    spec: str, start: dt.datetime, end: dt.datetime, args: Any, config: Any
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
        pos_id=pos_product(args.frame or common.DEFAULT_PLOT_FRAME, "24h"),
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


def line(spec: str, d: Any, what: str) -> str:
    """The ``directions:`` line of one result."""
    note = (
        f"±{d.sigma_azimuth:.1f}°, plunge {d.plunge:.1f}° ±{d.sigma_plunge:.1f}°, "
        f"|d_h| {d.horizontal:.1f} mm, {what}"
    )
    if not d.significant:
        note = "NOT significant (|d_h| < 3σ): " + note
    return f"{spec}: {{motion: {d.azimuth:.1f}}}  # {note}"


def run(args: argparse.Namespace, parser: argparse.ArgumentParser, config: Any) -> int:
    from gps_analysis.direction import motion_direction

    if args.start >= args.end:
        parser.error(f"start {args.start} is not before end {args.end}")
    window = args.window or (1.0 / 24.0 if args.rtk else 7.0)
    kind = "RTK" if args.rtk else "daily"
    failed = 0
    for spec in args.Series:
        try:
            if args.rtk:
                t, y, s = _rtk(spec, args.start, args.end, config)
            else:
                t, y, s = _daily(spec, args.start, args.end, args, config)
            # the ends are where the data are: a series that starts late or
            # stops early is measured over its own span, which is printed
            have = np.isfinite(np.asarray(y, dtype=float)).all(axis=0)
            if not have.any():
                raise ValueError("no data between -s and -e")
            start, end = float(t[have].min()), float(t[have].max())
            d = motion_direction(t, y, s, start, end, min(window, (end - start) / 2))
        except Exception as exc:  # one bad series must not sink the rest
            print(f"# {spec}: {exc}", file=sys.stderr)
            failed += 1
            continue
        print(line(spec, d, f"{_date(start)}…{_date(end)} {kind}, {window:g} d ends"))
    return 1 if failed == len(args.Series) else 0
