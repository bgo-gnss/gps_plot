"""``gpsplot polar``: where a pair's rover went, since an onset.

Panels (``--panels``, default sector,map,profile), one point per epoch,
coloured by the time since the onset:

* sector: the azimuth (compass, clockwise from north) and size of the
  horizontal displacement, with the motion direction (red), the across
  direction (blue) and the DEM downslope at the rover;
* map: the paths on the terrain (grey hillshade, ground steeper than 30°
  tinted orange, contours) around the rover, scaled up to show;
* profile: the ground along each rover's motion (DEM), the rover on it and
  its daily points scaled up, with weekly medians and the fitted plunge
  (:mod:`gps_plot.profile`);
* section: the displacement along the motion direction against Up [mm]
  (optional; the profile without the ground);
* hodograph: north against east (optional).

The colour bar sits inside the map (else the profile) when there is one.

The displacement is rover − reference, unsigned (where the rover went),
from its level at the onset (the median of the first ``--window``). The
onset is ``--onset``: a date, an events.yaml event id, ``auto`` (the
default: the newest events.yaml event whose region holds a rover, inside
the period) or ``start`` (the start of the period). The motion direction is the pair's (station_areas.yaml
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
from collections.abc import Mapping, Sequence
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

#: the epochs' colours, days since the onset (as the track maps)
CMAP = "turbo"
#: the colours span only the newest this many days (BGÓ 2026-11-08): older
#: points are open, a black outline without fill (``--colour-days``)
COLOUR_DAYS = 28.0
#: the gap between the map and the labels of the panels beside it [in]
#: (BGÓ 2026-11-08: narrower)
STACKED_GAP_IN = 0.08
#: the map: ground steeper than this [°] tinted in this colour (BGÓ
#: 2026-10-31, the 2 m ArcticDEM dots map), lit from the north-west
STEEP_DEG = 30.0
STEEP_CMAP = "Oranges"
SHADE_AZIMUTH, SHADE_ALTITUDE = 315.0, 45.0

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
        "(azimuth, horizontal size), the paths on the terrain and the "
        "ground profile along the motion, coloured by the time since the onset, "
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
        "--onset", default="auto",
        help="the onset: YYYYMMDD[-HH:MM], an events.yaml event id, auto "
        "(default: the newest events.yaml event whose region holds a rover, "
        "inside the period; else its start) or start (the start of the period)",
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
        "--colour-days",
        type=float,
        default=COLOUR_DAYS,
        metavar="DAYS",
        help=f"colour only the newest DAYS (default {COLOUR_DAYS:g}); older "
        "points are a black outline without fill; 0 = colour them all",
    )
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


def auto_onset(
    rovers: Sequence[str], start: dt.datetime | None, end: dt.datetime
) -> tuple[dt.datetime, str] | None:
    """(onset, event id) of the newest events.yaml event, of any kind, whose
    region (a station_areas.yaml area) holds one of ``rovers`` and whose onset
    is inside [``start``, ``end``); None when there is none."""
    from geo_dataread.events import read_events, read_station_areas, station_events
    from geo_dataread.events import KINDS

    areas = read_station_areas()
    found = {
        e.id: e
        for r in rovers
        for e in station_events(r, read_events(), areas, kinds=KINDS)
        if (start is None or e.onset >= start.date()) and e.onset < end.date()
    }
    if not found:
        return None
    event = max(found.values(), key=lambda e: (e.onset, e.id))
    return dt.datetime.combine(event.onset, dt.time()), event.id


def resolve_span(args: argparse.Namespace, now: dt.datetime) -> tuple[Any, Any, str]:
    """(start, end, the period's name for the file) of the plotted span."""
    end = args.end or (now if args.rtk else dt.datetime.combine(now.date(), dt.time()))
    if args.start is not None and args.special is None:
        # a fixed start running on to today keeps one name (since<start>),
        # so the daily product overwrites itself; an -e window names both ends
        name = (f"since{args.start:%Y%m%d}" if args.end is None
                else f"{args.start:%Y%m%d}-{end:%Y%m%d}")  # fmt: skip
        return args.start, end, name
    special = args.special or ("week" if args.rtk else "year")
    days = {"90d": 91, "year": 366, "week": 7, "month": 31}.get(special)
    start = None if days is None else end - dt.timedelta(days=days)
    return start, end, special


def displacement(
    t: Any, y: Any, sigma: Any, onset: float, window: float
) -> tuple[Any, Any, Any]:
    """(t, d, σ) from ``onset`` on: ``y`` less its robust level over the
    first ``window`` after the onset (:func:`gps_analysis.direction.end_level`),
    and the points' own σ.

    Raises:
        ValueError: no samples within ``window`` of the onset.
    """
    from gps_analysis.direction import end_level

    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    keep = t >= onset
    first = keep & (t <= onset + window)
    level, _, n = end_level(y[:, first], sigma[:, first])
    if n == 0 or not np.isfinite(level).all():
        raise ValueError("no data at the onset")
    return t[keep], y[:, keep] - level[:, None], sigma[:, keep]


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
PANELS = ("sector", "map", "profile", "section", "hodograph")
#: BGÓ 2026-11-01: the profile replaces the section in production
DEFAULT_PANELS = ("sector", "map", "profile")
#: half-width [m] of the map panel's DEM window around the rover
MAP_HALF_M = 120.0
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
    ``ref`` (name, azimuth from the rover [°], distance [m]) or None,
    ``sig`` (3, N) the points' σ [mm] or None.
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
    sig: Any = None

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


def _sector(ax: Any, pairs: Sequence[PairPlot], style: dict[str, Any],
            legend_inside: bool = False) -> int:  # fmt: skip
    """The polar sector; returns how many points were drawn at the centre.
    ``legend_inside``: the legend under the sector, not right of it (no room
    there when the colour bar is not on the side); inside the circle it
    covered the radius labels."""
    from matplotlib.ticker import FuncFormatter

    many = len(pairs) > 1
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    rs = [np.hypot(pp.d[0], pp.d[1]) for pp in pairs]
    azs = [np.degrees(np.arctan2(pp.d[1], pp.d[0])) % 360.0 for pp in pairs]
    rtop = max((float(r.max()) for r in rs if r.size), default=0.0)
    # the sector holds the points that carry a direction: past SECTOR_FRACTION
    # of the largest displacement (and MIN_RADIUS_MM); a smaller point outside
    # it is drawn at the centre, never hidden
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
                   linewidths=0.6, **style)  # fmt: skip

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
    if lines and legend_inside:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.04), ncols=2,
                  fontsize="x-small", framealpha=0.85)  # fmt: skip
    elif lines:
        ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize="small")
    return centred


def _hodograph(ax: Any, pairs: Sequence[PairPlot], style: dict[str, Any]) -> None:
    """North against east [mm]."""
    many = len(pairs) > 1
    rmax = max((float(np.hypot(pp.d[0], pp.d[1]).max()) for pp in pairs
                if pp.d.shape[1]), default=1.0) * 1.1  # fmt: skip
    for pp in pairs:
        ax.scatter(pp.d[1], pp.d[0], c=pp.since, s=16, marker=pp.marker, zorder=3,
                   edgecolors="k", linewidths=0.6, **style)  # fmt: skip
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


def read_segments(path: str) -> list[Any]:
    """The (lon, lat) segments of a GMT multi-segment file (``>`` between
    them); empty when it is missing."""
    segments: list[Any] = []
    points: list[tuple[float, float]] = []
    try:
        with open(path) as fh:
            for line in fh:
                if line.startswith(">"):
                    if points:
                        segments.append(np.array(points))
                    points = []
                elif line.strip() and not line.startswith("#"):
                    lon, lat = line.split()[:2]
                    points.append((float(lon), float(lat)))
    except OSError:
        return []
    if points:
        segments.append(np.array(points))
    return segments


def iceland_layers(gis: str) -> dict[str, list[Any]]:
    """The GMT maps' locator layers (:mod:`gps_plot.insets`): ``coast``,
    ``swarms``, ``glaciers``, ``volcanoes`` as (lon, lat) segments; a
    missing file gives an empty layer."""
    from gps_plot import insets

    files = {"coast": insets.LOCATOR_COAST, "swarms": insets.LOCATOR_SWARMS,
             "glaciers": insets.LOCATOR_GLACIERS,
             "volcanoes": insets.LOCATOR_VOLCANOES}  # fmt: skip
    return {k: read_segments(os.path.join(gis, f)) for k, f in files.items()}


def gmt_colour(value: str) -> Any:
    """A GMT colour (``119/136/153`` or a name) for matplotlib."""
    if "/" in value:
        return tuple(float(c) / 255.0 for c in value.split("/"))
    return value


def gmt_pen(value: str) -> tuple[float, Any]:
    """A GMT pen (``0.2p,black``): (width [pt], colour)."""
    width, _, colour = value.partition(",")
    return float(width.rstrip("p")), gmt_colour(colour or "black")


def _locator(ax: Any, layers: Mapping[str, Sequence[Any]],
             centre: tuple[float, float]) -> None:  # fmt: skip
    """The Iceland locator in ``ax``'s top right, drawn as the GMT maps'
    (:func:`gps_plot.insets.draw_locator`): no frame and no sea, white land
    with a black coast, the fissure swarms slate, the glaciers near-white,
    the central volcanoes outlined, and the map a red square."""
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes  # type: ignore[import-untyped]

    from gps_plot import insets

    box = inset_axes(ax, width=2.2, height=1.5, loc="upper right",
                     bbox_to_anchor=(0.0, 0.0, 1.0, 0.95),
                     bbox_transform=ax.transAxes, borderpad=0.4)  # fmt: skip
    box.set_zorder(20)
    box.axis("off")

    def draw(name: str, fill: str | None, pen: str) -> None:
        width, colour = gmt_pen(pen)
        for seg in layers.get(name, ()):
            box.fill(seg[:, 0], seg[:, 1], fc=gmt_colour(fill) if fill else "none",
                     ec=colour, lw=width)  # fmt: skip

    draw("coast", insets.LOCATOR_LAND, "0p,none")
    draw("swarms", insets.LOCATOR_SWARM_FILL, insets.LOCATOR_SWARM_PEN)
    draw("glaciers", insets.LOCATOR_GLACIER_FILL, insets.LOCATOR_GLACIER_PEN)
    draw("volcanoes", None, insets.LOCATOR_VOLCANO_PEN)
    draw("coast", None, insets.LOCATOR_COAST_PEN)  # over the swarms' sea edges
    box.plot(*centre, marker="s", ms=6, mfc="none", mec="red", mew=1.5)
    w, s_, e, n = insets.LOCATOR_CORNERS
    box.set_xlim(w, e)
    box.set_ylim(s_, n)
    box.set_aspect(1.0 / np.cos(np.radians(65.0)))


def degmin(value: float, pos: str, neg: str) -> str:
    """``64°00.62'N``: degrees and minutes, as the GMT maps write them."""
    deg, minutes = divmod(round(abs(value) * 60.0, 2), 60.0)
    return f"{deg:.0f}°{minutes:05.2f}'{pos if value >= 0 else neg}"


def _metres(pos: str, neg: str) -> Any:
    """Tick labels ``200 m E`` / ``200 m W`` (``0`` at the centre)."""
    from matplotlib.ticker import FuncFormatter

    return FuncFormatter(
        lambda v, _: "0" if abs(v) < 1e-9 else f"{abs(v):g} m {pos if v > 0 else neg}"
    )


def _gmt_look(ax: Any, half: float, centre: tuple[float, float] | None,
              locator: Mapping[str, Sequence[Any]]) -> None:  # fmt: skip
    """The map panel dressed as the GMT maps (BGÓ 2026-11-08): the tick
    labels inside the frame (``200 m E``, ``400 m N``), the centre marked
    with its coordinates, and an Iceland locator with the map on it."""
    import matplotlib.patheffects as pe
    from matplotlib.ticker import MaxNLocator

    halo = [pe.withStroke(linewidth=2.5, foreground="w")]
    for axis, pos, neg, side in ((ax.xaxis, "E", "W", "x"), (ax.yaxis, "N", "S", "y")):
        axis.set_major_locator(MaxNLocator(6, steps=[1, 2, 5, 10]))
        axis.set_major_formatter(_metres(pos, neg))
        ax.tick_params(axis=side, direction="in", pad=-5, labelsize="medium")
        ticks = [t for t in axis.get_major_locator()() if abs(t) < 0.95 * half]
        axis.set_ticks(ticks)
        for tick in axis.get_major_ticks():
            label = tick.label1
            label.set_path_effects(halo)
            if side == "x":
                label.set_verticalalignment("bottom")
            else:
                label.set_horizontalalignment("left")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.plot(0, 0, marker="+", ms=10, mew=1.2, color="k", zorder=2)
    if centre is not None:
        lon, lat = centre
        ax.annotate(f"0, 0: {degmin(lat, 'N', 'S')} {degmin(lon, 'E', 'W')}",
                    xy=(0.99, 0.99), xycoords="axes fraction", ha="right",
                    va="top", fontsize="small", zorder=21,
                    bbox={"boxstyle": "round", "fc": "w", "alpha": 0.85})  # fmt: skip
    if centre is not None and locator.get("coast"):
        _locator(ax, locator, centre)


def _map(ax: Any, pairs: Sequence[PairPlot], style: dict[str, Any], dem: Any,
         half: float, path_half: float, centre: tuple[float, float] | None = None,
         locator: Mapping[str, Sequence[Any]] | None = None) -> str:  # fmt: skip
    """The DEM with each rover's path since the onset scaled up, as velocity
    arrows on a map; returns the scale for the note. ``path_half``: the half
    width the longest path is scaled to fill about half of. ``centre``: the
    (lon, lat) of the map's 0, 0, ``locator`` the Iceland locator's
    layers (:func:`iceland_layers`)."""
    import matplotlib as mpl

    from gps_plot import terrain

    many = len(pairs) > 1
    if dem is not None:
        dx = float(abs(dem.x[1] - dem.x[0]))
        extent = (dem.x[0], dem.x[-1], dem.y[0], dem.y[-1])
        # grey relief, the steep ground tinted over it (scarps and gullies)
        shade = terrain.hillshade(dem.z, dx, azimuth=SHADE_AZIMUTH,
                                  altitude=SHADE_ALTITUDE)  # fmt: skip
        ax.imshow(shade, extent=extent, origin="lower", cmap="gray", vmin=0, vmax=1,
                  zorder=0)  # fmt: skip
        angle, _ = terrain.slope_field(dem.z, dx)
        ax.imshow(np.ma.masked_less(angle, STEEP_DEG), extent=extent, origin="lower",
                  cmap=STEEP_CMAP, vmin=20, vmax=60, alpha=0.35, zorder=0.5,
                  interpolation="nearest")  # fmt: skip
        ax.fill([np.nan] * 3, [np.nan] * 3, color=mpl.colormaps[STEEP_CMAP](0.55),
                alpha=0.6, label=f"steeper than {STEEP_DEG:g}°")  # fmt: skip
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
        order = np.argsort(pp.since)  # the newest on top
        ax.scatter(px[order], py[order], c=pp.since[order], s=60, marker=pp.marker,
                   zorder=3, edgecolors="k", linewidths=0.4, **style)  # fmt: skip
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
    # the dots' scale, lower right
    bar_mm = _nice(hmax / 2)
    xb, yb = 0.55 * half, -0.9 * half
    ax.plot([xb, xb + bar_mm * scale], [yb, yb], "o-", color="k", lw=2.5, ms=4,
            zorder=6)  # fmt: skip
    ax.annotate(f"{bar_mm:g} mm (dots)", (xb, yb), xytext=(0, 7),
                textcoords="offset points", fontsize="small", zorder=6,
                bbox={"boxstyle": "round", "fc": "w", "alpha": 0.85})  # fmt: skip
    ax.set_xlim(-half, half)
    ax.set_ylim(-half, half)
    ax.set_aspect("equal")
    _gmt_look(ax, half, centre, locator or {})
    # above the tick labels inside the frame
    ax.legend(loc="lower left", fontsize="small", borderaxespad=1.8)
    what = "paths" if many else "path"
    src = f"{dem.source} DEM, " if dem is not None else ""
    ax.set_title(f"terrain ({src}{what} {scale:g} m per mm)")
    return f"map paths scaled {scale:g} m per mm"


#: the 16 compass points, from north clockwise
COMPASS = ("N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
           "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW")  # fmt: skip


def compass(deg: float) -> str:
    """The compass point (16) of an azimuth [°]."""
    return COMPASS[int(round((deg % 360.0) / 22.5)) % 16]


def _usetex() -> bool:
    import matplotlib as mpl

    return bool(mpl.rcParams["text.usetex"])


def section_heading(motions: Sequence[float]) -> tuple[float, bool]:
    """The mean of the ``motions`` azimuths [°] and whether the section is
    drawn mirrored: a westward motion runs to the left, so the section reads
    as seen from the south (BGÓ 2026-10-31: drawn to the right, a slide
    going down to the west looked like one going down to the east)."""
    a = np.radians(np.asarray(motions, dtype=float))
    mean = float(np.degrees(np.arctan2(np.sin(a).sum(), np.cos(a).sum()))) % 360.0
    return mean, bool(np.sin(np.radians(mean)) < 0)


def _section(ax: Any, pairs: Sequence[PairPlot], style: dict[str, Any]) -> None:
    """Along each pair's motion against Up [mm], with the terrain's apparent
    dip along it; mirrored when the motion is westward
    (:func:`section_heading`), the compass points at the axis ends."""
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
                   zorder=3, edgecolors="k", linewidths=0.6, **style)  # fmt: skip
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
    mean, mirrored = section_heading([pp.motion for pp in pairs
                                      if pp.motion is not None])  # fmt: skip
    if mirrored:
        ax.invert_xaxis()
    if any(pp.downslope is not None for pp in pairs):
        ax.legend(loc="best", fontsize="small")
    ax.set_aspect("equal", adjustable="datalim")
    ahead, back = compass(mean), compass(mean + 180.0)
    left, right = (ahead, back) if mirrored else (back, ahead)
    what = ("along the motion [mm]" if many
            else f"along the motion ({pairs[0].motion:.0f}°) [mm]")  # fmt: skip
    ax.set_xlabel(rf"{left} $\leftarrow$ \quad {what} \quad $\rightarrow$ {right}"
                  if _usetex() else
                  rf"{left} $\leftarrow$   {what}   $\rightarrow$ {right}")  # fmt: skip
    ax.set_ylabel("Up [mm]")
    ax.set_title("vertical section along the motion")


#: the stacked figure with a profile: the profile's row height against the
#: sector's (the map's width follows from the height, _square_map_ratio)
PROFILE_ROW = 1.5
#: a profile-only figure: its width, the axes' width and the gap between
#: the sub-panels [in] (BGÓ 2026-11-01: closer together)
PROFILE_ALONE_W = 13.0
PROFILE_ALONE_AX_W = 11.6
PROFILE_GAP_IN = 0.85
#: its margins [in]: left, top (the title and the status line), bottom
PROFILE_ALONE_LEFT = 0.9
PROFILE_ALONE_TOP = 1.6
PROFILE_ALONE_BOTTOM = 1.05


#: the band over the map of a stacked figure (title, status line), inches
STACKED_TOP_IN = 1.05


def _square_map_ratio(size: tuple[float, float], pars: Any) -> float:
    """The map column's width ratio that makes its cell square, so the map
    fills it top to bottom and the column beside it can match its height."""
    high = (pars.top - pars.bottom) * size[1]
    total = (pars.right - pars.left) * size[0] / (1.0 + pars.wspace / 2.0)
    return float(np.clip(high / max(total - high, 1e-9), 1.2, 2.2))


def _close_up(fig: Any, mapax: Any, column: list[Any]) -> None:
    """The panels beside the map moved left, their labels
    :data:`STACKED_GAP_IN` right of it (their aspect is fixed: a wider cell
    only centres them), and the figure trimmed by the width freed on the
    right (BGÓ 2026-11-08)."""
    renderer = fig._get_renderer()
    to_fig = fig.transFigure.inverted()
    width, height = fig.get_size_inches()
    box = mapax.get_position()
    tight = [a.get_tightbbox(renderer).transformed(to_fig) for a in column]
    shift = min(t.x0 for t in tight) - (box.x1 + STACKED_GAP_IN / width)
    if shift <= 0:
        return
    for a in column:
        f = a.get_position(original=True)
        a.set_position((f.x0 - shift, f.y0, f.width, f.height))
        a.apply_aspect()
    tight = [a.get_tightbbox(renderer).transformed(to_fig) for a in column]
    spare = 1.0 - max(t.x1 for t in tight) - STACKED_GAP_IN / width
    if spare <= 0:
        return
    scale = 1.0 / (1.0 - spare)  # the same inches in the narrower figure
    fig.set_size_inches(width / scale, height)
    for a in [mapax, *column]:
        f = a.get_position(original=True)
        a.set_position((f.x0 * scale, f.y0, f.width * scale, f.height))
        a.apply_aspect()


def _frame_on_map(fig: Any, mapax: Any, column: list[Any], title: str,
                  status: str | None, status_color: str) -> None:  # fmt: skip
    """A stacked figure framed on its map (BGÓ 2026-11-03): no map title, the
    figure title and status line centred over the map, the logo in its top
    left corner, and the panels beside it fitted to it: the top title on
    its top edge, the lowest frame on its bottom edge (x labels in line)."""
    import matplotlib as mpl

    from gps_plot import timesmatplt as tplt

    mapax.set_title("")
    renderer = fig._get_renderer()
    mapax.apply_aspect()
    box = mapax.get_position()
    to_fig = fig.transFigure.inverted()
    for _ in range(3):  # the labels keep their size in inches: converge
        tight = [a.get_tightbbox(renderer).transformed(to_fig) for a in column]
        frames = [a.get_position(original=True) for a in column]
        lo, hi = min(f.y0 for f in frames), max(f.y1 for f in frames)
        above = max(t.y1 for t in tight) - hi
        scale = (box.y1 - above - box.y0) / (hi - lo)
        for a, f in zip(column, frames, strict=True):
            a.set_position((f.x0, box.y0 + (f.y0 - lo) * scale,
                            f.width, f.height * scale))  # fmt: skip
            a.apply_aspect()
    _close_up(fig, mapax, column)
    box = mapax.get_position()
    h = fig.get_size_inches()[1]
    middle = 0.5  # the title on the whole figure (BGÓ 2026-11-03)
    if status is not None:
        fig.text(middle, box.y1 + 0.08 / h, status, ha="center", va="bottom",
                 fontsize="large", color=status_color)  # fmt: skip
    fig.text(middle, box.y1 + 0.42 / h, title, ha="center", va="bottom",
             fontsize=mpl.rcParams["figure.titlesize"])  # fmt: skip
    wide = box.width * fig.get_size_inches()[0]
    tplt.inpLogo(fig, top_left=(box.x0 + 0.006, box.y1 - 0.008, 0.3 * wide * 0.186),  # type: ignore[no-untyped-call]
                 backdrop=True)  # fmt: skip


def older_than(days: float) -> str:
    """``older than 4 weeks`` (whole weeks), else ``older than 10 days``."""
    if days % 7 == 0:
        weeks = int(days // 7)
        return f"older than {weeks} week{'s' if weeks > 1 else ''}"
    return f"older than {days:g} days"


def _inner_colorbar(fig: Any, ax: Any, mappable: Any, label: str, loc: str,
                    older: str | None = None) -> None:  # fmt: skip
    """A horizontal colour bar inside ``ax`` at ``loc``, on a white box of
    a fixed size in inches so the label, the bar and its ticks always sit
    inside it; ``borderpad`` as a legend's, so it lines up with one.
    ``older``: the points below the colours are open (a black outline):
    an open triangle at the bar's start and a line under it say so."""
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    high = 0.62 if older is None else 0.8
    box = inset_axes(ax, width="36%", height=high, loc=loc, borderpad=0.5)
    box.set_facecolor((1.0, 1.0, 1.0, 0.85))
    box.set_zorder(20)
    box.set_xticks([])
    box.set_yticks([])
    for side in box.spines.values():
        side.set_edgecolor("0.5")
    lift = 0.02 if older is None else 0.14  # room for the note under the ticks
    cax = inset_axes(box, width="90%", height=0.13, loc="center",
                     bbox_to_anchor=(0.0, lift, 1.0, 1.0),
                     bbox_transform=box.transAxes, borderpad=0)  # fmt: skip
    cax.set_zorder(21)  # over its box
    bar = fig.colorbar(mappable, cax=cax, orientation="horizontal",
                       extend="min" if older else "neither",
                       extendfrac=0.04)  # fmt: skip
    if older:
        box.plot([0.08], [0.15], marker="o", ms=5, mfc="none", mec="k", mew=0.8,
                 transform=box.transAxes)  # fmt: skip
        box.text(0.11, 0.15, f"black outline, no fill: {older}", fontsize="x-small",
                 va="center", transform=box.transAxes)  # fmt: skip
    bar.ax.tick_params(labelsize="xx-small", length=2, pad=1)
    bar.ax.xaxis.set_label_position("top")
    bar.set_label(label, fontsize="x-small", labelpad=2)


def polar_figure(
    pairs: Sequence[PairPlot],
    *,
    onset: dt.datetime,
    title_status: Any = None,
    status_color: Any = "k",
    interactive: bool = False,
    panels: Sequence[str] = DEFAULT_PANELS,
    dem: Any = None,
    half: float = MAP_HALF_M,
    path_half: float = MAP_HALF_M,
    profile_dem: Any = None,
    profile_half: float | None = None,
    centre: tuple[float, float] | None = None,
    locator: Mapping[str, Sequence[Any]] | None = None,
    colour_days: float = COLOUR_DAYS,
) -> Any:
    """The figure of one pair or several (``--together``), one panel per
    ``panels`` entry (:data:`PANELS`); with a map, it is large on the left
    and the other panels stacked on the right.

    ``dem`` (:class:`gps_plot.terrain.DemWindow`, metres about the map's
    centre, ``half`` its half width) for the map; each pair's ``pos`` is its
    rover in that frame. ``profile_dem``: the same about the same centre,
    wide enough for the profiles (``profile_half`` [m] either side of each
    rover; default :data:`gps_plot.profile.PROFILE_HALF_M`). ``centre``:
    the map's 0, 0 (lon, lat); ``locator``: its Iceland layers.
    ``colour_days``: the colours span the newest this many days, older
    points are open (0: all coloured).
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib import cm

    from gps_plot import profile as prf
    from gps_plot import timesmatplt as tplt

    tplt.init_plot_style()
    p_half = prf.PROFILE_HALF_M if profile_half is None else profile_half
    others = [p for p in panels if p != "map"]
    stacked = "map" in panels and bool(others)
    alone = tuple(panels) == ("profile",)
    size = (21.0, 14.0) if stacked else (8.0 * len(panels) + 1.5, 9.0)
    heights: list[float] = []
    if alone:  # one width, the heights following the ground (equal axes)
        heights = [PROFILE_ALONE_AX_W * sp / (2 * p_half)
                   for sp in prf.spans(pairs, profile_dem, p_half)]  # fmt: skip
        size = (
            PROFILE_ALONE_W,
            sum(heights)
            + PROFILE_GAP_IN * (len(heights) - 1)
            + PROFILE_ALONE_TOP
            + PROFILE_ALONE_BOTTOM,
        )
    if interactive:
        fig = plt.figure(figsize=size)
    else:  # no pyplot (and no GUI backend) for a saved figure
        from matplotlib.figure import Figure

        fig = Figure(figsize=size)
    # the colour bar inside the map (else the profile), as on the other maps
    inside = "map" in panels or "profile" in panels
    right = 0.985 if inside else 1.0 - 1.2 / size[0]
    top = 1.0 - STACKED_TOP_IN / size[1] if stacked else 0.84
    if alone:
        left = PROFILE_ALONE_LEFT / size[0]
        fig.subplots_adjust(left=left, right=left + PROFILE_ALONE_AX_W / size[0],
                            bottom=PROFILE_ALONE_BOTTOM / size[1],
                            top=1.0 - PROFILE_ALONE_TOP / size[1])  # fmt: skip
    else:
        fig.subplots_adjust(left=0.05, right=right, bottom=0.07, top=top,
                            wspace=0.1 if inside else 0.18,
                            hspace=0.3)  # fmt: skip
    vmax = max(
        (float(np.max(pp.since)) for pp in pairs if np.size(pp.since)), default=1.0
    )
    # below the colour window: no fill, so the scatters' black edge is all
    vmin = max(0.0, vmax - colour_days) if colour_days > 0 else 0.0
    cmap = mpl.colormaps[CMAP].with_extremes(under="none")
    norm = mpl.colors.Normalize(vmin=vmin, vmax=max(vmax, vmin + 1e-9))
    style = {"cmap": cmap, "norm": norm}
    if stacked:
        widths = [_square_map_ratio(size, fig.subplotpars), 1.0]
        rows = [PROFILE_ROW if p == "profile" else 1.0 for p in others]
        grid = fig.add_gridspec(len(others), 2, width_ratios=widths, height_ratios=rows)
        where = {"map": grid[:, 0], **{p: grid[k, 1] for k, p in enumerate(others)}}
    else:
        grid = fig.add_gridspec(1, len(panels))
        where = {p: grid[k] for k, p in enumerate(panels)}
    holder = None  # the panel the colour bar goes in
    for panel in panels:
        if panel == "sector":
            ax = fig.add_subplot(where[panel], projection="polar")
            _sector(ax, pairs, style, legend_inside=inside)
        elif panel == "map":
            holder = fig.add_subplot(where[panel])
            _map(holder, pairs, style, dem, half, path_half, centre, locator)
        elif panel == "profile":
            gap = (PROFILE_GAP_IN / np.mean(heights)) if alone else 0.35
            axes, _ = prf.panel(fig, where[panel], pairs, style, profile_dem,
                                p_half, small=not alone, hspace=gap)  # fmt: skip
            if "map" not in panels:
                holder = axes[0]
        elif panel == "section":
            _section(fig.add_subplot(where[panel]), pairs, style)
        elif panel == "hodograph":
            _hodograph(fig.add_subplot(where[panel]), pairs, style)
        else:
            raise ValueError(f"unknown panel {panel!r}: one of {', '.join(PANELS)}")

    # the panels beside the map (not its insets: the locator)
    column = [a for a in fig.axes
              if a is not holder and a.get_axes_locator() is None]  # fmt: skip
    label = f"days since {onset:%Y-%m-%d %H:%M}"
    older = older_than(colour_days) if vmin > 0 else None
    if holder is not None:
        _inner_colorbar(fig, holder, cm.ScalarMappable(norm=norm, cmap=cmap), label,
                        "lower center" if "map" in panels else "upper left",
                        older)  # fmt: skip
    else:
        cax = fig.add_axes((right + 0.015, 0.14, 0.012, 0.62))
        bar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                           extend="min" if older else "neither")  # fmt: skip
        bar.set_label(label + (f"; black outline, no fill: {older}" if older else ""))

    rovers = ", ".join(pp.rover for pp in pairs)
    refs = sorted({pp.name[5:] for pp in pairs})
    title = rf"\Huge {rovers} -- {', '.join(refs)} movement since {onset:%Y-%m-%d}"
    if stacked and holder is not None:
        _frame_on_map(fig, holder, column, title, title_status, status_color)
        return fig
    # the logo in the band above the panels, left of the title (BGÓ 2026-11-03):
    # at most half the band high and a fifth of the figure wide (the logo is
    # 0.186 as high as wide), so a narrow profile-only figure keeps its title
    band = (1.0 - fig.subplotpars.top) * size[1]
    high = min(0.75, 0.5 * band, 0.18 * size[0] * 0.186)
    tplt.inpLogo(fig, top_left=(0.01, 1.0 - 0.1 * band / size[1], high))  # type: ignore[no-untyped-call]
    fig.suptitle(
        title,
        y=1.0 - 0.12 / size[1] if alone else 0.975 if stacked else 0.96,
    )
    if title_status is not None:
        fig.text(0.5, 1.0 - 1.0 / size[1] if alone else 0.94 if stacked else 0.905,
                 title_status, ha="center", fontsize="large",
                 color=status_color)  # fmt: skip
    # no footnote (BGÓ 2026-11-02): run() prints it to the log
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
    td, d, sd = displacement(t, y, s, t_on, window)
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
                  t_on=t_on, sig=np.asarray(sd)[:, ok])  # fmt: skip
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
           map_half: float, res: str = "auto",
           ) -> tuple[Any, float, tuple[float, float] | None]:  # fmt: skip
    """Put the rovers on one map: each pair's ``pos`` about their centre, and
    the DEM window holding them all (None without a DEM); its half width and
    the centre (lon, lat)."""
    import pyproj

    from gps_plot import terrain
    from gps_plot.map import gis_dir

    located = [pp for pp in pairs if pp.lonlat is not None]
    if "map" not in panels or not located:
        return None, map_half, None
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
    where = (float(lon), float(lat))
    try:
        return terrain.dem_window(lon, lat, gis_dir(config), half, res=res), half, where
    except OSError as exc:
        print(f"no DEM for the map ({exc})", file=sys.stderr)
        return None, half, where


def run(args: argparse.Namespace, parser: argparse.ArgumentParser, config: Any) -> int:
    import matplotlib as mpl

    from gps_plot import direction as dr
    from gps_plot import timesmatplt as tplt

    now = dt.datetime.now(dt.UTC).replace(tzinfo=None)
    start, end, period = resolve_span(args, now)
    if start is not None and start >= end:
        parser.error(f"start {start} is not before end {end}")
    onset_note = ""
    try:
        if args.onset == "auto":
            rovers = [b.split("-")[0].upper() for b in args.Baselines]
            hit = auto_onset(rovers, start, end)
            onset, onset_note = (
                (None, "")
                if hit is None
                else (hit[0], f"; onset: events.yaml {hit[1]}")
            )
        elif args.onset == "start":
            onset = None
        else:
            onset = resolve_onset(args.onset)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    # read from a day before the onset: a start on the onset day itself
    # (plots.yaml fixedstart:) would otherwise move the zero to the first
    # daily epoch (11:58), half a day after the events.yaml onset
    if onset is not None and start is not None and onset <= start:
        start = onset - dt.timedelta(days=1)
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
        f"{span} after the onset{onset_note}"
    )
    groups = [[g for g in got]] if args.together else [[g] for g in got]
    locator: dict[str, list[Any]] = {}
    if "map" in panels:  # the locator's Iceland
        from gps_plot.map import gis_dir

        locator = iceland_layers(gis_dir(config))
    for group in groups:
        pairs = [pp for pp, _ in group]
        for pp, (marker, line, slope) in zip(pairs, PAIR_STYLES, strict=False):
            pp.marker, pp.colour, pp.slope_colour = marker, line, slope
        dem, half, centre = _place(pairs, config, panels, args.map_half,
                                   args.dem_res)  # fmt: skip
        profile_dem = None
        if "profile" in panels:  # its own window: the profiles reach past the map
            from gps_plot.profile import PROFILE_HALF_M

            profile_dem, _, _ = _place(pairs, config, ["map"], PROFILE_HALF_M + 30.0,
                                    args.dem_res)  # fmt: skip
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
        print(note)
        fig = polar_figure(pairs, onset=when,
                           title_status=status, status_color=colour,
                           interactive=not args.save, panels=panels, dem=dem,
                           half=half, path_half=args.map_half,
                           profile_dem=profile_dem, centre=centre,
                           locator=locator,
                           colour_days=args.colour_days)  # fmt: skip
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
