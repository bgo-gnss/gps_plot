"""``gpsplot polar``: where a pair's rover went, since an onset.

Panels (``--panels``, default sector,map,section), one point per epoch,
coloured by the time since the onset:

* sector: the azimuth (compass, clockwise from north) and size of the
  horizontal displacement, with the motion direction (red), the across
  direction (blue) and the DEM downslope at the rover;
* map: the paths on the terrain (slope colour, hillshade, contours, fall
  lines) around the rover, scaled up to show;
* section: the displacement along the motion direction against Up;
* hodograph: north against east (optional).

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
import dataclasses
import datetime as dt
import os
import sys
import textwrap
from collections.abc import Sequence
from typing import Any

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
        "(azimuth, horizontal size), the paths on the terrain and a "
        "section along the motion, coloured by the time since the onset, "
        "with the pair's motion direction and the DEM downslope. One "
        "figure per pair (--together: one for all).",
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
        "--together", action="store_true",
        help="all pairs in one figure (their rovers on one map; e.g. SVIN-SKFC "
        "SVIE-SKFC): one marker shape per pair",
    )  # fmt: skip
    parser.add_argument(
        "--no-dem", action="store_true",
        help="no DEM (default: the DEM slope at the rover, see --dem-res)",
    )  # fmt: skip
    parser.add_argument(
        "--panels", default=",".join(DEFAULT_PANELS),
        help=f"the panels, left to right, of {', '.join(PANELS)} (default "
        f"{','.join(DEFAULT_PANELS)})",
    )  # fmt: skip
    parser.add_argument(
        "--map-half", type=float, default=MAP_HALF_M, metavar="M",
        help=f"half-width of the map panel around the rover [m] (default {MAP_HALF_M:g})",
    )  # fmt: skip
    parser.add_argument(
        "--dem-res", choices=("auto", "2m", "5m"), default="auto",
        help="the DEM: auto (default) = a 2 m ArcticDEM area grid where one "
        "covers (gis_dir dem/arcticdem_2m), else the 5 m DEM",
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


#: the panels of the figure, left to right (``--panels``)
PANELS = ("sector", "map", "section", "hodograph")
DEFAULT_PANELS = ("sector", "map", "section")
#: half-width [m] of the map panel's DEM window around the rover
MAP_HALF_M = 120.0
#: the map panel is this much wider than the others
MAP_WIDTH = 1.6
#: per pair of a --together figure: marker, motion colour, downslope colour
PAIR_STYLES = (
    ("o", "red", "saddlebrown"),
    ("^", "magenta", "darkolivegreen"),
    ("s", "darkorange", "sienna"),
    ("D", "deeppink", "olive"),
)


@dataclasses.dataclass
class PairPlot:
    """One pair's part of a polar figure.

    ``since`` days since the onset, ``d`` (3, N) mm since the onset (finite
    horizontal only), ``motion``/``downslope`` [°] or None, ``lonlat`` the
    rover, ``pos`` (x, y) [m] of the rover in the map (set by the figure),
    ``ref`` (name, azimuth from the rover [°], distance [m]) or None.
    """

    name: str
    since: Any
    d: Any
    motion: float | None
    downslope: tuple[float, float] | None = None
    lonlat: tuple[float, float] | None = None
    ref: tuple[str, float, float] | None = None
    pos: tuple[float, float] = (0.0, 0.0)
    marker: str = "o"
    colour: str = "red"
    slope_colour: str = "saddlebrown"
    #: the onset epoch of ``since`` [days since 1970]
    t_on: float = 0.0

    @property
    def rover(self) -> str:
        return self.name[:4]


def _nice(value: float) -> float:
    """The 1-2-5 number nearest ``value`` (in ratio)."""
    if value <= 0:
        return 1.0
    exp = np.floor(np.log10(value))
    best = min((1.0, 2.0, 5.0, 10.0), key=lambda m: abs(np.log(value / (m * 10**exp))))
    return float(best * 10**exp)


def _label(pp: PairPlot, many: bool, text: str) -> str:
    return f"{pp.rover}: {text}" if many else text


def _sector(ax: Any, pairs: Sequence[PairPlot], style: dict[str, Any]) -> int:
    """The polar sector; returns how many points were drawn at the centre."""
    from matplotlib.ticker import FuncFormatter

    many = len(pairs) > 1
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    rs = [np.hypot(pp.d[0], pp.d[1]) for pp in pairs]
    azs = [np.degrees(np.arctan2(pp.d[1], pp.d[0])) % 360.0 for pp in pairs]
    rtop = max((float(r.max()) for r in rs if r.size), default=0.0)
    # the sector holds the points that carry a direction: past SECTOR_FRACTION
    # of the largest displacement (and MIN_RADIUS_MM); a smaller point outside
    # it is drawn at the centre (counted in the note), never hidden
    floor = max(MIN_RADIUS_MM, SECTOR_FRACTION * rtop)
    lines: list[float] = []
    for pp in pairs:
        if pp.motion is not None:
            lines += [pp.motion, pp.motion + 90.0, pp.motion - 90.0]
        if pp.downslope is not None:
            lines.append(pp.downslope[0])
    big = np.concatenate([az[r >= floor] for az, r in zip(azs, rs, strict=True)])
    lo, hi = sector_limits(np.r_[big, lines])
    if not big.size and not lines:
        lo, hi = 0.0, 360.0
    ax.set_thetamin(lo)
    ax.set_thetamax(hi)
    rmax = rtop * 1.1 if rtop > 0 else 1.0
    centred = 0
    for pp, az, r in zip(pairs, azs, rs, strict=True):
        # into the sector's own range (a sector across north starts below 0)
        shifted = (az - lo) % 360.0 + lo
        inside = (shifted <= hi) & (r >= MIN_RADIUS_MM)
        centred += int(np.count_nonzero(~inside & (r >= MIN_RADIUS_MM)))
        ax.scatter(np.where(inside, np.radians(shifted), 0.0), np.where(inside, r, 0.0),
                   c=pp.since, s=16, marker=pp.marker, zorder=3, edgecolors="k",
                   linewidths=0.3, **style)  # fmt: skip

        def ray(deg: float, length: float, **kw: Any) -> None:
            a = np.radians((deg - lo) % 360.0 + lo)
            ax.plot([a, a], [0.0, length], **kw)

        if pp.downslope is not None:
            ray(pp.downslope[0], rmax, color=pp.slope_colour, lw=1.4, ls="--",
                label=_label(pp, many, f"downslope {pp.downslope[0]:.0f}°, dip "
                             f"{pp.downslope[1]:.0f}° (DEM)"))  # fmt: skip
        if pp.motion is not None:
            ray(pp.motion, rmax, color=pp.colour, lw=1.2,
                label=_label(pp, many, f"motion {pp.motion:.0f}°"))  # fmt: skip
            ray(pp.motion + 90.0, 0.3 * rmax, color="b", lw=1.0,
                label="across" if pp is pairs[0] else None)  # fmt: skip
            ray(pp.motion - 90.0, 0.3 * rmax, color="b", lw=1.0)
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda x, _p: f"{np.degrees(x) % 360:.0f}°")
    )
    ax.set_rmax(rmax)
    ax.set_rlabel_position(lo + 2.0)
    ax.set_title("azimuth, horizontal size [mm]", pad=20)
    if lines:
        ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize="small")
    return centred


def _hodograph(ax: Any, pairs: Sequence[PairPlot], style: dict[str, Any]) -> None:
    """North against east [mm]."""
    many = len(pairs) > 1
    rmax = max((float(np.hypot(pp.d[0], pp.d[1]).max()) for pp in pairs
                if pp.d.shape[1]), default=1.0) * 1.1  # fmt: skip
    for pp in pairs:
        ax.scatter(pp.d[1], pp.d[0], c=pp.since, s=16, marker=pp.marker, zorder=3,
                   edgecolors="k", linewidths=0.3, **style)  # fmt: skip
        if pp.motion is not None:
            a = np.radians(pp.motion)
            ax.plot([0.0, rmax * np.sin(a)], [0.0, rmax * np.cos(a)], color=pp.colour,
                    lw=1.2, label=_label(pp, many, f"motion {pp.motion:.0f}°"))  # fmt: skip
            q = 0.3 * rmax
            ax.plot([-q * np.cos(a), q * np.cos(a)], [q * np.sin(a), -q * np.sin(a)],
                    color="b", lw=1.0, label="across" if pp is pairs[0] else None)  # fmt: skip
    ax.axhline(0.0, color="0.6", lw=0.8)
    ax.axvline(0.0, color="0.6", lw=0.8)
    if any(pp.motion is not None for pp in pairs):
        ax.legend(loc="upper left", fontsize="small")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("East [mm]")
    ax.set_ylabel("North [mm]")
    ax.set_title("north against east [mm]")


def _map(ax: Any, pairs: Sequence[PairPlot], style: dict[str, Any], dem: Any,
         half: float, path_half: float) -> str:  # fmt: skip
    """The DEM with each rover's path since the onset scaled up, as velocity
    arrows on a map; returns the scale for the note. ``path_half``: the half
    width the longest path is scaled to fill about half of."""
    from gps_plot import terrain

    many = len(pairs) > 1
    if dem is not None:
        dx = float(abs(dem.x[1] - dem.x[0]))
        extent = (dem.x[0], dem.x[-1], dem.y[0], dem.y[-1])
        # the slope angle in colour, the relief shaded over it
        angle, fall = terrain.slope_field(dem.z, dx)
        im = ax.imshow(angle, extent=extent, origin="lower", cmap="YlOrBr", vmin=0,
                       vmax=45, zorder=0, interpolation="nearest")  # fmt: skip
        shade = terrain.hillshade(dem.z, dx)
        ax.imshow(shade, extent=extent, origin="lower", cmap="gray", vmin=0, vmax=1,
                  alpha=0.35, zorder=0.5)  # fmt: skip
        bar = ax.figure.colorbar(im, ax=ax, fraction=0.04, pad=0.01)
        bar.set_label("terrain slope [°]")
        # fall lines: which way the ground falls, on a grid of about 16 a side
        k = max(1, int(round(half / 8.0 / dx)))
        fx, fy = np.meshgrid(dem.x[::k], dem.y[::k])
        a = np.radians(fall[::k, ::k])
        steep = angle[::k, ::k] > 3.0
        ax.quiver(fx[steep], fy[steep], np.sin(a)[steep], np.cos(a)[steep],
                  color="0.25", scale=30, width=0.0025, headwidth=4, zorder=1.5,
                  alpha=0.7)  # fmt: skip
        relief = float(np.nanmax(dem.z) - np.nanmin(dem.z))
        step = 2.0 if relief < 40 else 5.0 if relief < 120 else 10.0
        levels = np.arange(np.floor(np.nanmin(dem.z) / step) * step,
                           np.nanmax(dem.z) + step, step)  # fmt: skip
        cs = ax.contour(dem.x, dem.y, dem.z, levels=levels, colors="0.35",
                        linewidths=0.5, zorder=1)  # fmt: skip
        ax.clabel(
            cs, levels[:: max(1, len(levels) // 6)], fontsize="x-small", fmt="%.0f"
        )
    hmax = max((float(np.hypot(pp.d[0], pp.d[1]).max()) for pp in pairs
                if pp.d.shape[1]), default=1.0)  # fmt: skip
    scale = _nice(0.55 * path_half / max(hmax, 1e-6))  # m on the map per mm
    arrow = 0.45 * path_half
    for pp in pairs:
        x0, y0 = pp.pos
        px, py = x0 + pp.d[1] * scale, y0 + pp.d[0] * scale
        ax.plot(px, py, color="k", lw=0.6, ls=":", zorder=2)
        ax.scatter(px, py, c=pp.since, s=18, marker=pp.marker, zorder=3,
                   edgecolors="k", linewidths=0.3, **style)  # fmt: skip
        ax.plot(x0, y0, marker="^", ms=11, color="w", mec="k", zorder=4)
        if many:
            ax.annotate(pp.rover, (x0, y0), xytext=(8, -14), textcoords="offset points",
                        fontsize="small", fontweight="bold", zorder=6,
                        bbox={"boxstyle": "round", "fc": "w", "alpha": 0.8})  # fmt: skip

        def draw(
            deg: float, length: float, color: str, label: str, ls: str = "-"
        ) -> None:
            a = np.radians(deg)
            ax.annotate("", xy=(x0 + length * np.sin(a), y0 + length * np.cos(a)),
                        xytext=(x0, y0), zorder=5,
                        arrowprops={"arrowstyle": "-|>", "color": color, "lw": 1.5,
                                    "ls": ls})  # fmt: skip
            ax.plot([], [], color=color, ls=ls, label=label)

        if pp.motion is not None:
            draw(
                pp.motion,
                arrow,
                pp.colour,
                _label(pp, many, f"motion {pp.motion:.0f}°"),
            )
        if pp.downslope is not None:
            draw(pp.downslope[0], 0.8 * arrow, pp.slope_colour,
                 _label(pp, many, "downslope (DEM)"), "--")  # fmt: skip
    refs = {pp.ref[0]: pp for pp in pairs if pp.ref is not None}
    for name, pp in refs.items():  # toward each reference station, at the edge
        assert pp.ref is not None
        _, az_ref, dist = pp.ref
        a = np.radians(az_ref)
        ax.annotate(f"{name} {dist / 1000:.1f} km",
                    xy=(0.9 * half * np.sin(a), 0.9 * half * np.cos(a)),
                    ha="center", va="center", fontsize="small", color="navy",
                    bbox={"boxstyle": "round", "fc": "w", "alpha": 0.8})  # fmt: skip
    ax.set_xlim(-half, half)
    ax.set_ylim(-half, half)
    ax.set_aspect("equal")
    ax.set_xlabel("East [m]")
    ax.set_ylabel("North [m]")
    ax.legend(loc="lower left", fontsize="small")
    what = "paths" if many else "path"
    src = f"{dem.source} DEM, " if dem is not None else ""
    ax.set_title(f"terrain ({src}{what} {scale:g} m per mm)")
    return f"map paths scaled {scale:g} m per mm"


def _section(ax: Any, pairs: Sequence[PairPlot], style: dict[str, Any]) -> None:
    """Along each pair's motion against Up [mm], with the terrain's apparent
    dip along it."""
    many = len(pairs) > 1
    drawn = False
    for pp in pairs:
        if pp.motion is None:
            continue
        drawn = True
        a = np.radians(pp.motion)
        along = pp.d[0] * np.cos(a) + pp.d[1] * np.sin(a)
        ok = np.isfinite(pp.d[2])
        ax.scatter(along[ok], pp.d[2][ok], c=pp.since[ok], s=16, marker=pp.marker,
                   zorder=3, edgecolors="k", linewidths=0.3, **style)  # fmt: skip
        if pp.downslope is not None:
            # the terrain along the motion: its dip seen in that direction
            app = np.arctan(np.tan(np.radians(pp.downslope[1]))
                            * np.cos(np.radians(pp.motion - pp.downslope[0])))  # fmt: skip
            xs = np.array([min(0.0, float(np.nanmin(along))),
                           max(0.0, float(np.nanmax(along)))])  # fmt: skip
            ax.plot(xs, -np.tan(app) * xs, color=pp.slope_colour, ls="--", lw=1.4,
                    label=_label(pp, many, f"terrain along {pp.motion:.0f}°: "
                                 f"{np.degrees(app):.0f}° down"))  # fmt: skip
    if not drawn:
        ax.text(0.5, 0.5, "no motion direction", transform=ax.transAxes, ha="center")
        return
    ax.axhline(0.0, color="0.6", lw=0.8)
    ax.axvline(0.0, color="0.6", lw=0.8)
    if any(pp.downslope is not None for pp in pairs):
        ax.legend(loc="upper right", fontsize="small")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("along the motion [mm]" if many
                  else f"along the motion ({pairs[0].motion:.0f}°) [mm]")  # fmt: skip
    ax.set_ylabel("Up [mm]")
    ax.set_title("vertical section along the motion")


def polar_figure(
    pairs: Sequence[PairPlot],
    *,
    onset: dt.datetime,
    note: str,
    title_status: Any = None,
    status_color: Any = "k",
    interactive: bool = False,
    panels: Sequence[str] = DEFAULT_PANELS,
    dem: Any = None,
    half: float = MAP_HALF_M,
    path_half: float = MAP_HALF_M,
) -> Any:
    """The figure of one pair or several (``--together``), one panel per
    ``panels`` entry (:data:`PANELS`); with a map, it is large on the left
    and the other panels stacked on the right.

    ``dem`` (:class:`gps_plot.terrain.DemWindow`, metres about the map's
    centre, ``half`` its half width) for the map; each pair's ``pos`` is its
    rover in that frame.
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib import cm

    from gps_plot import timesmatplt as tplt

    tplt.init_plot_style()
    others = [p for p in panels if p != "map"]
    stacked = "map" in panels and bool(others)
    size = (21.0, 14.0) if stacked else (8.0 * len(panels) + 1.5, 9.0)
    if interactive:
        fig = plt.figure(figsize=size)
    else:  # no pyplot (and no GUI backend) for a saved figure
        from matplotlib.figure import Figure

        fig = Figure(figsize=size)
    right = 1.0 - 1.2 / size[0]
    top = 0.89 if stacked else 0.84
    fig.subplots_adjust(left=0.05, right=right, bottom=0.07, top=top,
                        wspace=0.18, hspace=0.3)  # fmt: skip
    vmax = max(
        (float(np.max(pp.since)) for pp in pairs if np.size(pp.since)), default=1.0
    )
    cmap = mpl.colormaps["viridis"]
    norm = mpl.colors.Normalize(vmin=0.0, vmax=max(vmax, 1e-9))
    style = {"cmap": cmap, "norm": norm}
    centred = 0
    if stacked:
        grid = fig.add_gridspec(len(others), 2, width_ratios=[MAP_WIDTH, 1.0])
        where = {"map": grid[:, 0], **{p: grid[k, 1] for k, p in enumerate(others)}}
    else:
        grid = fig.add_gridspec(1, len(panels))
        where = {p: grid[k] for k, p in enumerate(panels)}
    for panel in panels:
        if panel == "sector":
            ax = fig.add_subplot(where[panel], projection="polar")
            centred = _sector(ax, pairs, style)
        elif panel == "map":
            note += "; " + _map(fig.add_subplot(where[panel]), pairs, style, dem,
                                half, path_half)  # fmt: skip
        elif panel == "section":
            _section(fig.add_subplot(where[panel]), pairs, style)
        elif panel == "hodograph":
            _hodograph(fig.add_subplot(where[panel]), pairs, style)
        else:
            raise ValueError(f"unknown panel {panel!r}: one of {', '.join(PANELS)}")

    cax = fig.add_axes((right + 0.015, 0.14, 0.012, 0.62))
    bar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    bar.set_label(f"days since {onset:%Y-%m-%d %H:%M}")

    rovers = ", ".join(pp.rover for pp in pairs)
    refs = sorted({pp.name[5:] for pp in pairs})
    fig.suptitle(
        rf"\Huge {rovers} -- {', '.join(refs)} \ \Large where {rovers} went since "
        rf"{onset:%Y-%m-%d}",
        y=0.975 if stacked else 0.96,
    )
    if title_status is not None:
        fig.text(0.5, 0.94 if stacked else 0.905, title_status, ha="center",
                 fontsize="large", color=status_color)  # fmt: skip
    if centred:
        note += (
            # no "%": the figure text goes through TeX, where it starts a comment
            f"; {centred} point(s) under {100 * SECTOR_FRACTION:.0f} per cent of the "
            "largest displacement and outside the sector drawn at the centre"
        )
    fig.text(0.01, 0.015, note, fontsize="x-small", color="0.3")
    return fig


def _read_pair(name: str, args: Any, config: Any, onset: Any, end: Any,
               read_from: Any, window: float, directions: Any) -> tuple[PairPlot, str]:  # fmt: skip
    """One pair's data, direction and DEM slope, and its part of the note."""
    from gps_plot import direction as dr
    from gps_plot.axes import apriori

    if args.rtk:
        t, y, s = dr._rtk(name, read_from, end, config)
    else:
        t, y, s = dr._daily(name, read_from, end, config, args.frame)
    have = np.isfinite(np.asarray(y, dtype=float)).all(axis=0)
    if not have.any():
        raise ValueError("no data in the period")
    first = float(np.asarray(t)[have].min())
    t_on = max(first if onset is None else dr_days(onset), first)
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
    ok = np.isfinite(d[0]) & np.isfinite(d[1])
    pp = PairPlot(name, (np.asarray(td) - t_on)[ok], np.asarray(d)[:, ok], motion,
                  t_on=t_on)  # fmt: skip
    note = how if motion is None else f"motion {motion:.0f} deg ({how})"
    if not args.no_dem:
        from gps_plot import terrain
        from gps_plot.map import gis_dir

        radius = args.slope_radius or terrain.SLOPE_RADIUS_M
        try:
            pp.lonlat = terrain.station_lonlat(pp.rover)
            sl = terrain.slope_at(*pp.lonlat, gis_dir(config), radius=radius,
                                  res=args.dem_res)  # fmt: skip
        except (OSError, ValueError) as exc:
            print(f"{name}: no DEM ({exc})", file=sys.stderr)
        else:
            pp.downslope = (sl.azimuth, sl.dip)
            note += f", {sl.source} DEM slope over {radius:g} m"
    b0 = apriori(name)
    if b0 is not None:  # the reference, seen from the rover
        pp.ref = (name[5:], float(np.degrees(np.arctan2(-b0[1], -b0[0])) % 360.0),
                  float(np.hypot(b0[0], b0[1])))  # fmt: skip
    return pp, f"{name}: {note}"


def _place(pairs: Sequence[PairPlot], config: Any, panels: Sequence[str],
           map_half: float, res: str = "auto") -> tuple[Any, float]:  # fmt: skip
    """Put the rovers on one map: each pair's ``pos`` about their centre, and
    the DEM window holding them all (None without a DEM); its half width."""
    import pyproj

    from gps_plot import terrain
    from gps_plot.map import gis_dir

    located = [pp for pp in pairs if pp.lonlat is not None]
    if "map" not in panels or not located:
        return None, map_half
    to_isn = pyproj.Transformer.from_crs("EPSG:4326", terrain.ISN93, always_xy=True)
    xy = np.array([to_isn.transform(pp.lonlat[0], pp.lonlat[1])
                   for pp in located if pp.lonlat is not None])  # fmt: skip
    centre = xy.mean(axis=0)
    for pp, (x, y) in zip(located, xy, strict=True):
        pp.pos = (float(x - centre[0]), float(y - centre[1]))
    spread = float(np.abs(xy - centre).max())
    half = spread + map_half
    to_geo = pyproj.Transformer.from_crs(terrain.ISN93, "EPSG:4326", always_xy=True)
    lon, lat = to_geo.transform(*centre)
    try:
        return terrain.dem_window(lon, lat, gis_dir(config), half, res=res), half
    except OSError as exc:
        print(f"no DEM for the map ({exc})", file=sys.stderr)
        return None, half


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
    panels = [x.strip() for x in args.panels.split(",") if x.strip()]
    if not panels or set(panels) - set(PANELS):
        parser.error(f"--panels: from {', '.join(PANELS)}")
    if args.together and len(args.Baselines) > len(PAIR_STYLES):
        parser.error(f"--together: at most {len(PAIR_STYLES)} pairs")

    directions = dr.resolve_directions(
        args.Baselines, "motion", motion=args.direction,
        span=args.direction_from, config=config, frame=args.frame, rtk=args.rtk,
    )  # fmt: skip
    read_from = start or dt.datetime(1990, 1, 1)
    got: list[tuple[PairPlot, str]] = []
    failed = 0
    for name in args.Baselines:
        try:
            got.append(_read_pair(name, args, config, onset, end, read_from, window,
                                  directions))  # fmt: skip
        except Exception as exc:  # one bad pair must not sink the rest
            print(f"{name}: {exc}", file=sys.stderr)
            failed += 1
    if not got:
        return 1
    kind = "RTK 10-min medians" if args.rtk else "daily, cleaned plate series"
    span = f"{window:g} d" if window >= 1 else f"{24 * window:g} h"
    head = (
        f"rover - reference, {kind}; displacement from the median of the first "
        f"{span} after the onset"
    )
    groups = [[g for g in got]] if args.together else [[g] for g in got]
    for group in groups:
        pairs = [pp for pp, _ in group]
        for pp, (marker, line, slope) in zip(pairs, PAIR_STYLES, strict=False):
            pp.marker, pp.colour, pp.slope_colour = marker, line, slope
        dem, half = _place(pairs, config, panels, args.map_half, args.dem_res)
        note = f"{head}; " + "; ".join(n for _, n in group)
        # one onset for the figure (the earliest): every pair's days from it
        t0 = min(pp.t_on for pp in pairs)
        for pp in pairs:
            pp.since = pp.since + (pp.t_on - t0)
        when = dt.datetime(1970, 1, 1) + dt.timedelta(days=t0)
        last = when + dt.timedelta(
            days=max(float(np.max(pp.since)) for pp in pairs if np.size(pp.since))
        )
        colour: Any
        status: str
        if args.rtk:  # minutes old, not days: no daily currency colour
            status, colour = f"Last epoch: {last:%d %b %Y %H:%M} UTC", "k"
        else:
            title = tplt.make_title(pairs[0].name, last)
            status, colour = title.status, title.status_color
        fig = polar_figure(pairs, onset=when, note=note,
                           title_status=status, status_color=colour,
                           interactive=not args.save, panels=panels, dem=dem,
                           half=half, path_half=args.map_half)  # fmt: skip
        tplt.inpLogo(fig)  # type: ignore[no-untyped-call]
        if not args.save:
            import matplotlib.pyplot as plt

            plt.show()
            continue
        # A_B-polar-<period>; together: A_C_B (the rovers, then the references)
        refs = sorted({pp.name[5:] for pp in pairs})
        names = "_".join([*(pp.rover for pp in pairs), *refs])
        stem = f"{names}-polar-{period}" + ("-rtk" if args.rtk else "")
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
        print(
            f"plotted {', '.join(pp.name for pp in pairs)}: {os.path.join(fig_dir, stem)}"
        )
    return 1 if failed == len(args.Baselines) else 0


def dr_days(when: dt.datetime) -> float:
    """A naive UTC datetime as days since 1970."""
    return (when - dt.datetime(1970, 1, 1)).total_seconds() / 86400.0
