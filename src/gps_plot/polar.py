"""``gpsplot polar``: where a pair's rover went, since an onset.

Two panels side by side, one point per epoch, coloured by the time since
the onset:

* a polar sector: the azimuth (compass, clockwise from north) and size of
  the horizontal displacement since the onset;
* a hodograph: north against east of the same displacement, with the
  pair's motion direction (red) and the across direction (blue).

The displacement is rover − reference, unsigned (where the rover went),
from its level at the onset (the median of the first ``--window``). The
onset is ``--onset`` (a date, or an events.yaml event id), else the start
of the period. The motion direction is the pair's (station_areas.yaml
``directions:``, ``--direction``, ``--direction-from``), else the one
estimated over the plotted span. Daily by default (the cleaned plate
series), ``--rtk`` for the RTK baseline (10-minute medians). Files:
``A_B-polar-<period>`` (``-rtk``).
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
import textwrap
from typing import Any, cast

import numpy as np

from gps_plot.cli import common

#: the routine periods (as gpsplot ts), plus an RTK week
SPECIALS = ("90d", "year", "full", "week", "month")

#: horizontal displacement [mm] below which an azimuth is noise: such
#: points are drawn at the centre of the sector, not at a random angle
MIN_RADIUS_MM = 1.0

#: points past this fraction of the largest displacement set the sector
SECTOR_FRACTION = 0.15

EXAMPLES = """examples:
  gpsplot polar SVIE-SKFC --special year --save png
  gpsplot polar SVIN-SKFC SVIE-SKFC --onset 20260920 -e 20261029 --save pdf
  gpsplot polar SVIE-SKFC --rtk --special week --save png
"""


def build_parser(
    parser: argparse.ArgumentParser | None = None, config: Any = None
) -> argparse.ArgumentParser:
    from gps_plot.direction import add_direction_options, window_days
    from gps_plot.rtk import _when

    if parser is None:
        parser = argparse.ArgumentParser(prog="gpsplot polar")
    parser.description = textwrap.fill(
        "Where each pair's rover went since an onset: a polar sector "
        "(azimuth, horizontal size) and a hodograph (north against east) "
        "side by side, coloured by the time since the onset, with the "
        "pair's motion direction. One figure per pair.",
        width=78,
    )
    parser.formatter_class = argparse.RawDescriptionHelpFormatter
    parser.epilog = EXAMPLES
    parser.add_argument("Baselines", nargs="+", help="AAAA-BBBB (rover-reference)")
    parser.add_argument(
        "-s", "--start", type=_when, default=None, help="YYYYMMDD[-HH:MM]"
    )
    parser.add_argument(
        "-e", "--end", type=_when, default=None, help="YYYYMMDD[-HH:MM]"
    )
    parser.add_argument(
        "--special", choices=SPECIALS, default=None,
        help="the period ending -e (default yesterday): 90d, year, full; "
        "week, month (for --rtk). Default year (week with --rtk)",
    )  # fmt: skip
    parser.add_argument(
        "--onset", default=None,
        help="the onset: YYYYMMDD[-HH:MM] or an events.yaml event id "
        "(default: the start of the period)",
    )  # fmt: skip
    parser.add_argument(
        "--window", type=window_days, default=None,
        help="the onset level is the median over this (default 7d, 1h RTK)",
    )  # fmt: skip
    parser.add_argument("--rtk", action="store_true", help="from the RTK baseline")
    parser.add_argument(
        "--frame", default=None,
        help=f"daily .pos frame (default {common.DEFAULT_PLOT_FRAME})",
    )  # fmt: skip
    add_direction_options(parser)
    parser.add_argument(
        "--no-dem", action="store_true",
        help="no DEM downslope line (default: the 5 m DEM's slope at the rover)",
    )  # fmt: skip
    parser.add_argument(
        "--slope-radius", type=float, default=None, metavar="M",
        help="radius of the DEM plane fitted at the rover [m] (default 50)",
    )  # fmt: skip
    parser.add_argument(
        "--save", nargs="?", const="png", default=None,
        help="save as png (default) / pdf / eps …, comma separated",
    )  # fmt: skip
    parser.add_argument("-d", "--figDir", default=None, help="output directory")
    parser.add_argument(
        "--manifest", default=None, metavar="FILE",
        help="append the files written to FILE, one absolute path per line",
    )  # fmt: skip
    return parser


def resolve_onset(value: str | None) -> dt.datetime | None:
    """``--onset``: a date (YYYYMMDD[-HH:MM]) or an events.yaml event id."""
    from gps_plot.rtk import _when

    if value is None:
        return None
    try:
        return _when(value)
    except argparse.ArgumentTypeError:
        pass
    from geo_dataread.events import read_events

    for event in read_events():
        if event.id == value:
            return dt.datetime.combine(event.onset, dt.time())
    raise ValueError(f"--onset {value!r}: no such date or events.yaml event id")


def resolve_span(args: argparse.Namespace, now: dt.datetime) -> tuple[Any, Any, str]:
    """(start, end, the period's name for the file) of the plotted span."""
    end = args.end or (now if args.rtk else dt.datetime.combine(now.date(), dt.time()))
    if args.start is not None and args.special is None:
        return args.start, end, f"{args.start:%Y%m%d}-{end:%Y%m%d}"
    special = args.special or ("week" if args.rtk else "year")
    days = {"90d": 91, "year": 366, "week": 7, "month": 31}.get(special)
    start = None if days is None else end - dt.timedelta(days=days)
    return start, end, special


def displacement(
    t: Any, y: Any, sigma: Any, onset: float, window: float
) -> tuple[Any, Any]:
    """(t, d) from ``onset`` on: ``y`` less its robust level over the first
    ``window`` after the onset (:func:`gps_analysis.direction.end_level`).

    Raises:
        ValueError: no samples within ``window`` of the onset.
    """
    from gps_analysis.direction import end_level

    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = t >= onset
    first = keep & (t <= onset + window)
    level, _, n = end_level(y[:, first], np.asarray(sigma, dtype=float)[:, first])
    if n == 0 or not np.isfinite(level).all():
        raise ValueError("no data at the onset")
    return t[keep], y[:, keep] - level[:, None]


def sector_limits(azimuth_deg: Any, pad: float = 15.0) -> tuple[float, float]:
    """(min, max) [°] of a sector holding every azimuth, padded; may start
    below 0 (a sector across north). The full circle past 270°."""
    az = np.asarray(azimuth_deg, dtype=float) % 360.0
    if az.size == 0:
        return 0.0, 360.0
    mean = np.degrees(np.arctan2(np.sin(np.radians(az)).mean(),
                                 np.cos(np.radians(az)).mean()))  # fmt: skip
    off = (az - mean + 180.0) % 360.0 - 180.0
    lo, hi = mean + off.min() - pad, mean + off.max() + pad
    if hi - lo >= 270.0:
        return 0.0, 360.0
    # the upper end in (0, 360]; the lower may be below 0 (across north)
    while hi > 360.0:
        lo, hi = lo - 360.0, hi - 360.0
    while hi <= 0.0:
        lo, hi = lo + 360.0, hi + 360.0
    return float(lo), float(hi)


def polar_figure(
    name: str,
    t_days: Any,
    d: Any,
    *,
    onset: dt.datetime,
    motion: float | None,
    note: str,
    downslope: tuple[float, float] | None = None,
    title_status: Any = None,
    status_color: Any = "k",
    interactive: bool = False,
) -> Any:
    """The figure: the sector (left) and the hodograph (right).

    ``t_days`` days since 1970, ``d`` (3, N) mm since the onset; ``motion``
    the direction line [°] (None: none drawn), with the across direction
    (± 90°) in blue; ``downslope`` the DEM slope (azimuth, dip) [°] at the
    rover, a brown dashed ray (None: none). The sector holds every line.
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib import cm

    from gps_plot import timesmatplt as tplt

    tplt.init_plot_style()
    t0 = (onset - dt.datetime(1970, 1, 1)).total_seconds() / 86400.0
    since = np.asarray(t_days, dtype=float) - t0
    dn, de = d[0], d[1]
    ok = np.isfinite(dn) & np.isfinite(de)
    since, dn, de = since[ok], dn[ok], de[ok]
    r = np.hypot(dn, de)
    az = np.degrees(np.arctan2(de, dn)) % 360.0

    if interactive:
        fig = plt.figure(figsize=(18, 9))
    else:  # no pyplot (and no GUI backend) for a saved figure
        from matplotlib.figure import Figure

        fig = Figure(figsize=(18, 9))
    from matplotlib.projections.polar import PolarAxes

    pax = cast(PolarAxes, fig.add_subplot(1, 2, 1, projection="polar"))
    hax = fig.add_subplot(1, 2, 2)
    fig.subplots_adjust(left=0.04, right=0.9, bottom=0.08, top=0.84, wspace=0.15)
    cmap = mpl.colormaps["viridis"]
    norm = mpl.colors.Normalize(
        vmin=0.0, vmax=max(float(since.max()) if since.size else 1.0, 1e-9)
    )

    pax.set_theta_zero_location("N")
    pax.set_theta_direction(-1)
    # the sector holds the points that carry a direction: past SECTOR_FRACTION
    # of the largest displacement (and MIN_RADIUS_MM); a smaller point outside
    # it is drawn at the centre (counted in the note), never hidden
    big = r >= max(MIN_RADIUS_MM, SECTOR_FRACTION * (float(r.max()) if r.size else 0.0))
    lines = [] if motion is None else [motion, motion + 90.0, motion - 90.0]
    if downslope is not None:
        lines.append(downslope[0])
    lo, hi = sector_limits(np.r_[az[big], lines])
    if not big.any() and not lines:
        lo, hi = 0.0, 360.0
    pax.set_thetamin(lo)
    pax.set_thetamax(hi)
    # into the sector's own range (a sector across north starts below 0)
    shifted = (az - lo) % 360.0 + lo
    inside = (shifted <= hi) & (r >= MIN_RADIUS_MM)
    centred = int(np.count_nonzero(~inside & (r >= MIN_RADIUS_MM)))
    pax.scatter(np.where(inside, np.radians(shifted), 0.0), np.where(inside, r, 0.0),
                c=since, cmap=cmap, norm=norm, s=14, zorder=3)  # fmt: skip
    from matplotlib.ticker import FuncFormatter

    pax.xaxis.set_major_formatter(
        FuncFormatter(lambda x, _pos: f"{np.degrees(x) % 360:.0f}°")
    )
    rmax = float(r.max()) * 1.1 if r.size and r.max() > 0 else 1.0
    pax.set_rmax(rmax)
    pax.set_rlabel_position(lo + 2.0)
    pax.set_title("azimuth, horizontal size [mm]", pad=20)

    hax.scatter(de, dn, c=since, cmap=cmap, norm=norm, s=14, zorder=3)
    hax.axhline(0.0, color="0.6", lw=0.8)
    hax.axvline(0.0, color="0.6", lw=0.8)
    if downslope is not None:
        b = np.radians((downslope[0] - lo) % 360.0 + lo)
        pax.plot([b, b], [0.0, rmax], color="saddlebrown", lw=1.4, ls="--",
                 label=f"downslope {downslope[0]:.0f}°, dip {downslope[1]:.0f}° (DEM)")  # fmt: skip
    if motion is not None:
        a = np.radians((motion - lo) % 360.0 + lo)
        pax.plot([a, a], [0.0, rmax], color="r", lw=1.2, label=f"motion {motion:.0f}°")
        for side in (90.0, -90.0):
            c = np.radians((motion + side - lo) % 360.0 + lo)
            pax.plot([c, c], [0.0, 0.3 * rmax], color="b", lw=1.2,
                     label="across" if side > 0 else None)  # fmt: skip
    if motion is not None or downslope is not None:
        pax.legend(loc="lower left", bbox_to_anchor=(-0.05, -0.08), fontsize="small")
        hax.plot([0.0, rmax * np.sin(a)], [0.0, rmax * np.cos(a)], color="r", lw=1.2,
                 label=f"motion {motion:.0f}°")  # fmt: skip
        across = a + np.pi / 2
        q = 0.3 * rmax
        hax.plot([-q * np.sin(across), q * np.sin(across)],
                 [-q * np.cos(across), q * np.cos(across)], color="b", lw=1.2,
                 label="across")  # fmt: skip
        hax.legend(loc="upper left", fontsize="small")
    hax.set_aspect("equal", adjustable="datalim")
    hax.set_xlabel("East [mm]")
    hax.set_ylabel("North [mm]")
    hax.set_title("north against east [mm]")

    cax = fig.add_axes((0.92, 0.12, 0.015, 0.68))
    bar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    bar.set_label(f"days since {onset:%Y-%m-%d %H:%M}")

    rover, ref = name.split("-")
    fig.suptitle(
        rf"\Huge {rover} -- {ref} \ \Large where {rover} went since "
        rf"{onset:%Y-%m-%d}",
        y=0.96,
    )
    if title_status is not None:
        fig.text(0.5, 0.905, title_status, ha="center", fontsize="large",
                 color=status_color)  # fmt: skip
    if centred:
        note += (
            # no "%": the figure text goes through TeX, where it starts a comment
            f"; {centred} point(s) under {100 * SECTOR_FRACTION:.0f} per cent of the largest "
            "displacement and outside the sector drawn at the centre"
        )
    fig.text(0.01, 0.015, note, fontsize="x-small", color="0.3")
    return fig


def run(args: argparse.Namespace, parser: argparse.ArgumentParser, config: Any) -> int:
    import matplotlib as mpl

    from gps_plot import direction as dr
    from gps_plot import timesmatplt as tplt

    now = dt.datetime.now(dt.UTC).replace(tzinfo=None)
    start, end, period = resolve_span(args, now)
    if start is not None and start >= end:
        parser.error(f"start {start} is not before end {end}")
    try:
        onset = resolve_onset(args.onset)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    if onset is not None and start is not None and onset < start:
        start = onset
    if not args.save:
        mpl.use("WebAgg")
    window = args.window or (1.0 / 24.0 if args.rtk else 7.0)
    directions = dr.resolve_directions(
        args.Baselines, "motion", motion=args.direction,
        span=args.direction_from, config=config, frame=args.frame, rtk=args.rtk,
    )  # fmt: skip
    read_from = start or dt.datetime(1990, 1, 1)
    failed = 0
    for name in args.Baselines:
        try:
            if args.rtk:
                t, y, s = dr._rtk(name, read_from, end, config)
            else:
                t, y, s = dr._daily(name, read_from, end, config, args.frame)
            have = np.isfinite(np.asarray(y, dtype=float)).all(axis=0)
            if not have.any():
                raise ValueError("no data in the period")
            first = float(np.asarray(t)[have].min())
            t_on = first if onset is None else dr_days(onset)
            t_on = max(t_on, first)
            when = dt.datetime(1970, 1, 1) + dt.timedelta(days=t_on)
            td, d = displacement(t, y, s, t_on, window)
            motion = (directions.get(name.upper()) or {}).get("motion")
            how = "config/option"
            if motion is None:
                est, _, _ = dr.estimate(name, when, end, config, rtk=args.rtk,
                                        window=window, frame=args.frame)  # fmt: skip
                if est.significant:
                    motion, how = est.azimuth, "estimated over the plot"
                else:  # within its own noise: no direction to draw
                    how = "no significant motion over the plot"
        except Exception as exc:  # one bad pair must not sink the rest
            print(f"{name}: {exc}", file=sys.stderr)
            failed += 1
            continue
        kind = "RTK 10-min medians" if args.rtk else "daily, cleaned plate series"
        span = f"{window:g} d" if window >= 1 else f"{24 * window:g} h"
        direction = how if motion is None else f"motion {motion:.0f} deg ({how})"
        note = (
            f"{name} rover - reference, {kind}; displacement from the median of "
            f"the first {span} after the onset; {direction}"
        )
        last = dt.datetime(1970, 1, 1) + dt.timedelta(days=float(np.max(td)))
        colour: Any
        status: str
        if args.rtk:  # minutes old, not days: no daily currency colour
            status, colour = f"Last epoch: {last:%d %b %Y %H:%M} UTC", "k"
        else:
            title = tplt.make_title(name, last)
            status, colour = title.status, title.status_color
        downslope = None
        if not args.no_dem:
            from gps_plot import terrain
            from gps_plot.map import gis_dir

            radius = args.slope_radius or terrain.SLOPE_RADIUS_M
            try:
                sl = terrain.station_slope(name[:4], gis_dir(config), radius=radius)
            except (OSError, ValueError) as exc:
                print(f"{name}: no DEM slope ({exc})", file=sys.stderr)
            else:
                downslope = (sl.azimuth, sl.dip)
                note += f"; DEM slope at {name[:4]} over {radius:g} m"
        fig = polar_figure(name, td, d, onset=when, motion=motion, note=note,
                           downslope=downslope,
                           title_status=status, status_color=colour,
                           interactive=not args.save)  # fmt: skip
        tplt.inpLogo(fig)  # type: ignore[no-untyped-call]
        if not args.save:
            import matplotlib.pyplot as plt

            plt.show()
            continue
        stem = f"{name.replace('-', '_')}-polar-{period}" + ("-rtk" if args.rtk else "")
        fig_dir = (
            args.figDir
            if args.figDir is not None
            else common.config_dir(config, "fig_dir", "")
        )
        tplt.saveFig(os.path.join(fig_dir, stem), args.save, fig)
        if args.manifest:
            with open(args.manifest, "a") as fh:
                fh.writelines(
                    os.path.abspath(f) + "\n" for f in getattr(fig, "_gps_files", ())
                )
        print(f"plotted {name}: {os.path.join(fig_dir, stem)}")
    return 1 if failed == len(args.Baselines) else 0


def dr_days(when: dt.datetime) -> float:
    """A naive UTC datetime as days since 1970."""
    return (when - dt.datetime(1970, 1, 1)).total_seconds() / 86400.0
