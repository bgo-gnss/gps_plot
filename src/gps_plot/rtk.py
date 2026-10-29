"""``gpsplot rtk``: RTK baselines (rtk.vedur.is), optionally with the daily ones.

RTK baselines ``ROVER-BASE`` are the 1 Hz RTKLIB solutions of
``geo_dataread.rtk_read`` as interval medians (1 minute by default),
drawn as lines in mm, each zeroed on its first hour: the plots ``rtk_gps``
makes, on the ``gpsplot`` frame. ``--daily`` adds the daily GLOBK baseline
of the same pair (``sources.gps``, the ``ts`` baseline) as points, moved
onto the RTK line by the median over their overlap. ``--longterm`` turns it
round for monitoring against the long record: the daily points over 90 days,
a year or the whole series, with the RTK tail of the last ``--rtk-span``
moved onto them (left on its own zero, and said so, when they do not
overlap). RTK and daily stay independent sources; only that placement joins
them. Float solutions (Q=2) are used by default (BGÓ 2026-10-29: long
baselines often cannot fix); ``--fixed-only`` drops them.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import os
import sys
import textwrap
from typing import Any

import numpy as np

from gps_plot.cli import common

#: span ending at -e (default now, UTC), as rtk_gps's specials (+ month)
SPECIALS: dict[str, dt.timedelta] = {
    "6h": dt.timedelta(hours=6),
    "12h": dt.timedelta(hours=12),
    "day": dt.timedelta(days=1),
    "twodays": dt.timedelta(days=2),
    "threedays": dt.timedelta(days=3),
    "week": dt.timedelta(days=7),
    "month": dt.timedelta(days=30),
}
DEFAULT_SPECIAL = "twodays"
#: --longterm spans of the daily record (full = the whole series)
LONGTERM: dict[str, dt.timedelta | None] = {
    "week": dt.timedelta(days=7),
    "14d": dt.timedelta(days=14),
    "30d": dt.timedelta(days=30),
    "90d": dt.timedelta(days=90),
    "year": dt.timedelta(days=365),
    "full": None,
}
#: --longterm defaults: the RTK tail and its median interval
LONGTERM_RTK_SPAN = "week"
LONGTERM_INTERVAL = "10min"
LONGTERM_YMARGIN = 10  # mm around the daily + robust RTK range
FULL_START = dt.datetime(1995, 1, 1)  # before any IMO daily solution
TIME_FORMATS = ("%Y%m%d-%H:%M", "%Y%m%d")

EXAMPLES = """\
examples:
  gpsplot rtk "SEY?-SEYD" --save pdf      the last two days, all SEY baselines
  gpsplot rtk SEY1-SEYD --special 6h --save png
  gpsplot rtk "SEY[1-4]-SEYD" --special month --interval 10min --daily --save pdf
                                          a month of 10-min medians + the daily
                                          GLOBK baselines on top
  gpsplot rtk SENG-SUDV -s 20261001 -e 20261005-12:00 --save pdf
  gpsplot rtk SVIN-SKFC --longterm 14d --rtk-overlap 2 --save pdf
                                          two weeks of daily GLOBK, RTK from 2
                                          days before the last daily point on
Times are UTC. Patterns match the RTK baseline directories; quote them. A
plain pair processed the other way round (B-A) is read negated, as A-B."""


def _when(value: str) -> dt.datetime:
    for fmt in TIME_FORMATS:
        try:
            return dt.datetime.strptime(value, fmt)
        except ValueError:
            pass
    raise argparse.ArgumentTypeError(f"{value!r}: use YYYYMMDD or YYYYMMDD-HH:MM (UTC)")


def build_parser(
    parser: argparse.ArgumentParser | None = None, config: Any = None
) -> argparse.ArgumentParser:
    if parser is None:
        parser = argparse.ArgumentParser(prog="gpsplot rtk")
    parser.description = textwrap.fill(
        "RTK GPS baselines (rtk.vedur.is): interval medians of the 1 Hz RTKLIB "
        "solutions, rover minus base in mm, each zeroed on its first hour. "
        "--daily adds the daily GLOBK baseline of each pair as points.",
        width=78,
    )
    parser.formatter_class = argparse.RawDescriptionHelpFormatter
    parser.epilog = EXAMPLES
    parser.add_argument(
        "Baselines", nargs="+", help='ROVER-BASE, or a pattern ("SEY?-SEYD")'
    )
    parser.add_argument("-s", "--start", type=_when, default=None, help="start, UTC")
    parser.add_argument(
        "-e", "--end", type=_when, default=None, help="end, UTC (default now)"
    )
    parser.add_argument(
        "--special",
        choices=list(SPECIALS),
        default=None,
        help=f"span ending at -e (default {DEFAULT_SPECIAL} when -s is not given)",
    )
    parser.add_argument(
        "--longterm",
        choices=list(LONGTERM),
        default=None,
        help="the daily GLOBK baseline over this span ending at -e, with the RTK "
        "tail (--rtk-span) moved onto it by the median over their overlap",
    )
    parser.add_argument(
        "--rtk-span",
        choices=list(SPECIALS),
        default=None,
        help=f"--longterm: the RTK tail ending at -e (default {LONGTERM_RTK_SPAN})",
    )
    parser.add_argument(
        "--rtk-overlap",
        type=float,
        default=None,
        metavar="DAYS",
        help="--longterm: start the RTK tail this many days before the last "
        "daily solution, so it just overlaps the daily points (instead of "
        "--rtk-span); the short monitoring plot, e.g. --longterm 14d "
        "--rtk-overlap 2",
    )
    parser.add_argument(
        "--interval",
        default=None,
        help="median interval, e.g. 60s (default; --longterm "
        f"{LONGTERM_INTERVAL}), 10min, 1h",
    )
    parser.add_argument(
        "--offset", type=float, default=0.0, metavar="MM",
        help="vertical spacing between consecutive baselines [mm]",
    )  # fmt: skip
    parser.add_argument(
        "--daily",
        action="store_true",
        help="also the daily GLOBK baseline of each pair (points), placed on the "
        "RTK line by the median over their overlap",
    )
    parser.add_argument(
        "--frame",
        default=None,
        help="frame of the daily series (--daily; default itrf2014)",
    )
    parser.add_argument(
        "--max-sigma", type=float, default=0.1, metavar="M",
        help="drop RTK epochs with sdn/sde/sdu above this [m] (default 0.1, 0 = keep)",
    )  # fmt: skip
    quality = parser.add_mutually_exclusive_group()
    quality.add_argument(
        "--fixed-only",
        action="store_true",
        help="only the fixed solutions (Q=1); by default float (Q=2) is used "
        "too, as rtk_gps plots them: less accurate, but long baselines often "
        "cannot fix",
    )
    quality.add_argument(  # the default since 2026-10-29; kept for old scripts
        "--keep-float", action="store_true", help=argparse.SUPPRESS
    )
    quality.add_argument(
        "--keep-single",
        action="store_true",
        help="every solution, single (Q=5) included",
    )
    parser.add_argument(
        "--rtk-dir",
        default=None,
        help="RTK data directory (default: rtk_dir in postprocess.cfg [PATHS], "
        "then $GPS_RTK_DIR, then /mnt_data/rtgpsdata)",
    )
    common.add_event_options(parser)
    common.add_output_options(parser, config)
    return parser


def resolve_span(
    args: argparse.Namespace, now: dt.datetime
) -> tuple[dt.datetime, dt.datetime, str | None]:
    """``(start, end, special)``: -s/-e, or the special span ending at -e/now.

    With ``--longterm`` the span of the RTK tail (``--rtk-span``).
    """
    end = args.end or now
    if getattr(args, "longterm", None):
        special = args.rtk_span or LONGTERM_RTK_SPAN
        return end - SPECIALS[special], end, special
    if args.start is not None and args.special is None:
        return args.start, end, None
    special = args.special or DEFAULT_SPECIAL
    return end - SPECIALS[special], end, special


def longterm_start(longterm: str, end: dt.datetime) -> dt.datetime:
    """Start of the daily record shown by ``--longterm``."""
    span = LONGTERM[longterm]
    return FULL_START if span is None else end - span


def _title(
    n: int,
    bases: set[str],
    last: dt.datetime,
    now: dt.datetime,
    longterm: str | None = None,
) -> Any:
    from gps_plot import timesmatplt as tplt

    status_color = (
        tplt.STATUS_CURRENT_COLOR if now - last < dt.timedelta(hours=1)
        else tplt.STATUS_STALE_COLOR
    )  # fmt: skip
    base = f"base {next(iter(bases))}" if len(bases) == 1 else "mixed bases"
    what = r"RTK + daily" if longterm else "RTK"
    return tplt.StationTitle(
        main=r"\Huge %s \ \Large %d baseline%s, %s"
        % (what, n, "" if n == 1 else "s", base),
        status=r"\LARGE Last epoch: %s UTC" % last.strftime("%d %b %Y %H:%M"),
        created=r"\large (Plot created on %s UTC)" % now.strftime("%b %d %Y %H:%M"),
        status_color=status_color,
    )


def _daily(
    pair: str,
    start: dt.datetime,
    end: dt.datetime,
    args: argparse.Namespace,
    config: Any,
) -> Any:
    """The daily GLOBK baseline of ``pair`` over the span, t as datetime64."""
    from geo_dataread.gps_read import toDateTime
    from geo_dataread.pos import pos_product
    from gtimes.timefunc import TimetoYearf

    from gps_plot.sources import gps as source

    pid = pos_product(args.frame or common.DEFAULT_PLOT_FRAME, "24h")
    pad = dt.timedelta(days=1)  # daily epochs are at noon: keep the edge days
    req = source.SeriesRequest(
        fstart=TimetoYearf(*(start - pad).timetuple()[:6]),
        fend=TimetoYearf(*(end + pad).timetuple()[:6]),
        ref="plate",
        Dir=common.config_dir(config, "pos_dir"),
        tType="POS",
        uncert=10,
        view="raw",
        pos_id=pid,
    )
    s = source.fetch(pair, req)
    t = np.array(list(toDateTime(s.t)), dtype="datetime64[ns]")
    return dataclasses.replace(s, t=t, label=f"{pair} daily")


def rtk_start(
    start: dt.datetime,
    plot_start: dt.datetime,
    daily: Any,
    overlap: float | None,
) -> dt.datetime:
    """Start of a ``--longterm`` RTK tail.

    ``overlap`` days before the last daily epoch (not before the plot),
    so RTK only just overlaps the daily points; ``start`` (``--rtk-span``)
    when no overlap is asked or there is no daily series.
    """
    if overlap is None or daily is None or len(daily.t) == 0:
        return start
    last = as_datetime(np.asarray(daily.t)[-1])
    return max(plot_start, last - dt.timedelta(days=overlap))


def _longterm_pair(
    name: str,
    plot_start: dt.datetime,
    start: dt.datetime,
    end: dt.datetime,
    args: argparse.Namespace,
    config: Any,
    root: Any,
    dirs: list[str],
    interval: str,
) -> tuple[list[tuple[Any, str]], dict[str, list[str]]]:
    """One ``--longterm`` pair: its daily points, then its RTK tail on them.

    The daily series is the anchor; the RTK tail (``start``–``end``) is
    moved onto it by :func:`~gps_plot.sources.rtk.align`, or left on its
    own first-hour zero, noted, when the two do not overlap (a daily
    series that stopped before the RTK came back must not be hidden by
    pinning). Either part may be missing; ``([], ...)`` when both are.
    """
    from gps_plot.sources import rtk as rtk_source

    got: list[tuple[Any, str]] = []
    out: dict[str, list[str]] = {"dropped": [], "notes": []}
    daily = None
    try:
        daily = _daily(name, plot_start, end, args, config)
        daily = dataclasses.replace(daily, meta={**daily.meta, "pair": name})
        got.append((daily, "marked"))
    except Exception as exc:
        out["dropped"].append(f"{name} daily: {exc}")
    start = rtk_start(start, plot_start, daily, getattr(args, "rtk_overlap", None))
    try:
        directory, flipped = rtk_source.rtk_direction(name, dirs)
        s = rtk_source.fetch_rtk(
            name, start, end, root=str(root), interval=interval,
            drop_q=drop_q(args), max_sigma=args.max_sigma,
            directory=directory, flipped=flipped,
        )  # fmt: skip
    except Exception as exc:
        out["dropped"].append(f"{name}: {exc}")
        return got, out
    if len(s.t) == 0:
        out["dropped"].append(f"{name}: no RTK solutions in the span")
        return got, out
    s = dataclasses.replace(s, meta={**s.meta, "pair": name})
    if daily is not None:
        try:
            s = rtk_source.align(s, daily)
            sh = s.meta["aligned"]["shift"]
            out["notes"].append(
                f"{name} {sh[0]:+.0f}/{sh[1]:+.0f}/{sh[2]:+.0f}"
                f" ({s.meta['aligned']['days']} d)"
            )
        except ValueError:
            out["notes"].append(f"{name} NOT aligned (no overlap, own zero)")
    got.append((s, "line"))
    return got, out


def run(args: argparse.Namespace, parser: argparse.ArgumentParser, config: Any) -> int:
    try:  # bad --events fail before the data are read
        events = common.resolve_events(args)
    except ValueError as exc:
        parser.error(str(exc))
    import matplotlib as mpl
    from geo_dataread.rtk_read import list_baselines, rtk_dir

    from gps_plot import compose
    from gps_plot import timesmatplt as tplt
    from gps_plot.sources import rtk as rtk_source

    now = dt.datetime.now(dt.UTC).replace(tzinfo=None)
    root = rtk_dir(args.rtk_dir or common.config_dir(config, "rtk_dir"))
    try:
        baselines = rtk_source.expand_rtk(args.Baselines, list_baselines(root))
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    start, end, special = resolve_span(args, now)
    if start >= end:
        parser.error(f"start {start} is not before end {end}")
    if args.longterm and args.daily:
        parser.error("--longterm already draws the daily baseline; drop --daily")
    interval = args.interval or (LONGTERM_INTERVAL if args.longterm else "60s")
    dirs = list_baselines(root)

    series, styles, colors, dropped, notes = [], [], [], [], []
    palette = compose.series_colors(baselines)
    plot_start = start
    if args.longterm:
        plot_start = longterm_start(args.longterm, end)
        for name, color in zip(baselines, palette, strict=True):
            got, note = _longterm_pair(
                name, plot_start, start, end, args, config, root, dirs, interval
            )
            for s, style in got:
                series.append(s)
                styles.append(style)
                colors.append(color)
            dropped.extend(note["dropped"])
            notes.extend(note["notes"])
    for name, color in zip(baselines, palette, strict=True):
        if args.longterm:
            break
        try:
            directory, flipped = rtk_source.rtk_direction(name, dirs)
            s = rtk_source.fetch_rtk(
                name, start, end, root=str(root), interval=interval,
                drop_q=drop_q(args), max_sigma=args.max_sigma,
                directory=directory, flipped=flipped,
            )  # fmt: skip
        except Exception as exc:  # one bad baseline must not sink the plot
            dropped.append(f"{name}: {exc}")
            continue
        if len(s.t) == 0:
            dropped.append(f"{name}: no RTK solutions in the span")
            continue
        s = dataclasses.replace(s, meta={**s.meta, "pair": name})
        series.append(s)
        styles.append("line")
        colors.append(color)
        if args.daily:
            try:
                d = rtk_source.align_daily(_daily(name, start, end, args, config), s)
                d = dataclasses.replace(d, meta={**d.meta, "pair": name})
            except Exception as exc:
                dropped.append(f"{name} daily: {exc}")
                continue
            series.append(d)
            styles.append("marked")
            colors.append(color)
            sh = d.meta["aligned"]["shift"]
            notes.append(
                f"{name} {sh[0]:+.0f}/{sh[1]:+.0f}/{sh[2]:+.0f}"
                f" ({d.meta['aligned']['days']} d)"
            )
    for line in dropped:
        print(f"dropped {line}", file=sys.stderr)
    if not series:
        print("nothing plotted", file=sys.stderr)
        return 1

    if args.longterm == "full":  # the axis starts with the record
        plot_start = min(as_datetime(s.t[0]) for s in series)
    if not args.save:
        mpl.use("WebAgg")
    rtk_only = [s for s in series if s.meta.get("source") == "rtk"]
    last = max((as_datetime(s.t[-1]) for s in rtk_only), default=start)
    bases = {s.meta.get("pair", s.label)[5:] for s in series}
    index = baseline_index(series)
    # --offset moves each BASELINE: its RTK line and its daily points together
    moved = [
        dataclasses.replace(s, y=np.asarray(s.y) + k * args.offset)
        for k, s in zip(index, series, strict=True)
    ]
    labels = [
        compose.offset_label(s.label, k, args.offset)
        for k, s in zip(index, series, strict=True)
    ]
    fig = compose.stack_figure(
        moved,
        title=_title(
            len({s.meta.get("pair", s.label) for s in series}),
            bases,
            last,
            now,
            args.longterm,
        ),
        labels=labels,
        start=plot_start,
        end=end,
        # --longterm: a margin around the robust range, so RTK spikes clip
        ylim=args.ylim or ([LONGTERM_YMARGIN] if args.longterm else []),
        styles=styles,
        colors=colors,
        fig=tplt._reusable_figure() if args.save else None,
        robust_lines=bool(args.longterm),
    )
    if notes:
        what = (
            "RTK placed on daily, N/E/U mm (median over the overlap): "
            if args.longterm
            else "daily placed on RTK, N/E/U mm (median over days): "
        )
        fig.axes[2].text(
            0.01, 0.02, what + "; ".join(notes), transform=fig.axes[2].transAxes,
            fontsize="x-small", color="0.3", va="bottom",
        )  # fmt: skip
    if events:
        tplt.addEvent(events, fig)
    tplt.inpLogo(fig)  # type: ignore[no-untyped-call]

    if not args.save:
        import matplotlib.pyplot as plt

        plt.show()
        return 0
    if args.name:
        stem = args.name
    else:
        first = baselines[0].replace("-", "_")
        n = len({s.meta.get("pair", s.label) for s in series})
        span = special if special else f"{start:%Y%m%d%H%M}-{end:%Y%m%d%H%M}"
        if args.longterm:
            tail = special
            if args.rtk_overlap is not None:
                tail = f"overlap{args.rtk_overlap:g}d"
            span = f"long{args.longterm}-{tail}"
        stem = f"rtk_{first}_{n}-{span}"
        if args.interval:
            stem += f"-{args.interval}"
        if args.fixed_only:
            stem += "-fixed"
        elif args.keep_single:
            stem += "-allq"
        if args.daily:
            stem += "-daily"
        if args.offset:
            stem += f"-offset{args.offset:g}"
    fig_dir = (
        args.figDir
        if args.figDir is not None
        else common.config_dir(config, "fig_dir", "")
    )
    tplt.saveFig(os.path.join(fig_dir, stem), args.save, fig)
    return 0


def drop_q(args: argparse.Namespace) -> tuple[int, ...]:
    """RTKLIB quality flags dropped: single (Q=5) by default, as rtk_gps."""
    if args.keep_single:
        return ()
    if getattr(args, "fixed_only", False):
        return (2, 3, 4, 5, 6)
    return (5,)


def as_datetime(t: Any) -> dt.datetime:
    """A datetime64 / datetime epoch as a naive datetime."""
    return np.datetime64(t, "us").astype(dt.datetime)  # type: ignore[no-any-return]


def baseline_index(series: Any) -> list[int]:
    """Per series, the index of the baseline (``meta["pair"]``) it belongs to.

    Series without a pair start a new baseline at each RTK line, the
    daily points following it.
    """
    out: list[int] = []
    seen: dict[str, int] = {}
    k = -1
    for s in series:
        pair = s.meta.get("pair")
        if pair is None:
            if s.meta.get("source") == "rtk":
                k += 1
            out.append(k)
        else:
            if pair not in seen:
                k += 1
                seen[pair] = k
            out.append(seen[pair])
    return out
