"""``gpsplot map``: PyGMT maps over :mod:`gps_plot.maps`.

Four kinds, one subcommand each:

* ``station`` — station markers (:func:`maps.station_map`).
* ``velocity`` — horizontal (or ``--vertical``) velocity arrows with 1-σ
  ellipses (:func:`maps.velocity_map`). Per ``--window``/``--last`` a line
  fitted to each station's series over that window (the Reykjanes maps of
  gps_data_analyses, ``gpsvelo``), one colour per window; without a window
  the stored secular rates; or a gps_api ``--geojson`` product.
* ``deformation`` — displacement from -s to -e (:func:`maps.deformation_vectors`),
  optionally with Mogi model fields (``--mogi``).
* ``slip`` — a gps_api Okada slip product (:func:`maps.slip_map`).

Stations are stations.cfg patterns (``"SEY?"``, ``"*"``) and/or every
station inside ``--region``; ``--exclude`` drops some. PyGMT and the series
readers are imported only when a map is drawn (``gpsplot --help`` and
``gpsplot map --help`` need neither).
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import datetime as dt
import fnmatch
import math
import os
import sys
import textwrap
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from gps_plot import gmt_overrides
from gps_plot.cli import common

KINDS = ("station", "velocity", "deformation", "slip")
REFS = ("detrend", "plate")
#: one colour per window, in order (the Reykjanes maps: blue, then red)
WINDOW_COLORS = ("blue", "red", "darkgreen", "darkorange", "purple", "brown")
#: the 90th-percentile arrow is drawn this long [cm] by the automatic scale
TARGET_ARROW_CM = 2.0
#: the Veðurstofa name logo, top left (source checkout only: not packaged):
#: the two-line logo of the RTK plots, drawn 30 % transparent as they do
DEFAULT_LOGO = (
    Path(__file__).resolve().parents[2] / "logos" / "vi_logos" / "VI_Two_Line_Blue.png"
)
LOGO_WIDTH = "3.6c"
LOGO_TRANSPARENCY = 30
DATE_FORMAT = "%Y%m%d"
#: --dem land colours: one grey, so the hillshade alone shows the relief
#: (the Reykjanes maps' lightgrey.cpt, darkened from 225-235 to 200 so the
#: white glaciers stand out). The CPTs live in gps_plot/cpt (lightgrey,
#: slate, relief, sea); see :mod:`gps_plot.palettes` for the name lookup.
DEFAULT_DEM_CPT = "lightgrey"
#: the colour-map options, checked before a map is drawn
CMAP_OPTIONS = ("dem_cmap", "bathymetry_cmap", "glacier_cmap")
#: ``--dem-cmap list`` (any colour-map option) prints the names and exits
LIST = "list"
#: --dem iceland: the spliced Iceland DEM, at the level the map needs
ICELAND_DEM = "iceland"
#: the local GIS directory (dem/, vector/, cpt/): gis_dir in postprocess.cfg
DEFAULT_GIS_DIR = "~/gps-data/gis"
#: station maps label their stations by default only up to this many
LABEL_LIMIT = 60
#: a map with no stations and no --region (``gpsplot map base``): Iceland
DEFAULT_BASE_REGION = (-24.6, -13.2, 63.2, 66.7)
#: ``map base``: a station map without stations
BASE = "base"
#: the legend group of stations.cfg stations drawn beside --stations-file ones
STATIONS_CFG_GROUP = "GPS stations"
TITLE_FONT = "12p,Helvetica-Bold"
#: thin ellipses under coloured arrows (the Reykjanes maps)
ELLIPSE_PEN = "0.5p,gray30"

EXAMPLES = """\
examples:
  gpsplot map station "*" --region -22.8/-21.9/63.75/64.1 --save pdf
                        every station of stations.cfg on Reykjanes, as markers
  gpsplot map velocity --region -22.8/-21.9/63.75/64.1 \\
      --window 20231027:20231109 --window 20231123:20231205 --save pdf
                        two windows, plate removed, one colour each
  gpsplot map velocity --region -22.8/-21.9/63.75/64.1 --last 7 --vertical --save png
                        vertical rate over the last 7 days (north arrow = uplift)
  gpsplot map velocity --region -24.6/-13.2/63.2/66.7 --save pdf
                        stored secular rates, Iceland
  gpsplot map deformation --region -22.8/-21.9/63.75/64.1 -s 20231101 -e 20231115 \\
      --mogi -22.46,63.87,4.5,1.2e7 --save pdf
                        displacement 1-15 Nov 2023 next to a Mogi source
                        (lon, lat, 4.5 km deep, 1.2e7 m3)
  gpsplot map slip models/reykjanes_slip.json --view plane --save pdf
                        an Okada slip model, seen on the fault plane
  gpsplot map station --tos --dem iceland --projection M18c --save pdf
                        every GPS station in TOS, by status, Iceland DEM
  gpsplot map station --tos --dem iceland --dem-cmap slate --bathymetry-cmap none
                        blue-grey land, flat sea; --dem-cmap list: every CPT
  gpsplot map station --tos --dem iceland --dem-smooth 3 --save pdf
                        softer relief: the hillshade smoothed 3 km wide
  gpsplot map station --tos-search "--epos" --dem iceland --save pdf
                        the EPOS stations; any 'tos search' line selects
                        (GPS stations unless it names a subtype)
  gpsplot map station --tos-search "'subtype = [GPS stöð, SIL stöð]'" --save pdf
                        GPS and SIL stations, one symbol per subtype
  gpsplot map station --tos --stations-file planned.txt:Planned --save pdf
                        TOS GPS stations plus the sites in a file
  gpsplot map base --dem iceland --save pdf
                        the background alone (or --no-stations on a map)
  gpsplot map station --tos --dem iceland --gmt glaciers:pen=1p,blue \\
      --gmt coast:A=500/0/1 --gmt-config MAP_FRAME_TYPE=plain
                        blue glacier outlines, no islands under 500 km2 or lakes,
                        a plain map frame: any GMT argument for one drawing
                        step (--gmt STEP:KEY=VALUE) or GMT setting
  --dem iceland picks the level of the Iceland DEM the map needs, from the
  local GIS directory (gis_dir in postprocess.cfg [PATHS], default
  ~/gps-data/gis: dem/, vector/, cpt/); --dem @earth_relief_15s
  draws a hillshade from the GMT remote relief;
  --overlay lavas.geojson draws outlines (e.g. the LMÍ/NÍ lava flows)."""


# ---------------------------------------------------------------------------
# options
# ---------------------------------------------------------------------------


def _region(value: str) -> tuple[float, float, float, float]:
    try:
        w, e, s, n = (float(v) for v in value.split("/"))
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r}: use W/E/S/N in degrees") from None
    if not (w < e and s < n):
        raise argparse.ArgumentTypeError(f"{value!r}: need W < E and S < N")
    return (w, e, s, n)


def _date(value: str) -> dt.datetime:
    try:
        return dt.datetime.strptime(value, DATE_FORMAT)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r}: use YYYYMMDD") from None


def _window(value: str) -> tuple[dt.datetime, dt.datetime | None]:
    """``START:END`` (YYYYMMDD); an empty END means up to now."""
    start, sep, end = value.partition(":")
    if not sep:
        raise argparse.ArgumentTypeError(f"{value!r}: use START:END (YYYYMMDD)")
    s = _date(start)
    e = _date(end) if end else None
    if e is not None and e <= s:
        raise argparse.ArgumentTypeError(f"{value!r}: END is not after START")
    return (s, e)


def _mogi(value: str) -> tuple[float, float, float, float]:
    try:
        lon, lat, depth, dv = (float(v) for v in value.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"{value!r}: use LON,LAT,DEPTH_KM,DV_M3"
        ) from None
    if depth <= 0:
        raise argparse.ArgumentTypeError(f"{value!r}: depth must be > 0 km")
    return (lon, lat, depth, dv)


def _add_map_options(p: argparse.ArgumentParser, config: Any) -> None:
    g = p.add_argument_group("map")
    g.add_argument(
        "--region", type=_region, default=None, metavar="W/E/S/N",
        help="map bounds in degrees; with no station given, every station "
        "inside it is plotted (default: the stations' box + --margin)",
    )  # fmt: skip
    g.add_argument(
        "--margin", type=float, default=None, metavar="DEG",
        help="padding around the stations when --region is not given",
    )  # fmt: skip
    g.add_argument("--projection", default=None, help="GMT projection (default M12c)")
    g.add_argument(
        "--resolution", default=None,
        help="shoreline resolution: auto (default), full, high, intermediate, low",
    )  # fmt: skip
    g.add_argument(
        "--dem", default=None, metavar="GRID",
        help="hillshade background: 'iceland' (the Iceland topo-bathymetry "
        "DEM, the level the map needs, with the IS 50V coast), a grid file, "
        "or a GMT remote grid such as @earth_relief_15s",
    )  # fmt: skip
    g.add_argument(
        "--dem-cmap", default=None, metavar="CPT",
        help="land colours of the --dem background: lightgrey (default), "
        "slate (the Bárðarbunga maps' blue-grey), relief, a GMT CPT (geo, "
        "batlow, …), a CPT file, or a NAME.cpt in the GIS cpt/ directory "
        "(gis_dir in postprocess.cfg, default ~/gps-data/gis); "
        "a CPT with a sea half (geo, sealand) colours the sea floor too. "
        "'list' prints every name",
    )  # fmt: skip
    g.add_argument(
        "--bathymetry-cmap", default=None, metavar="CPT",
        help="sea-floor colours (z < 0), named as --dem-cmap (default: the "
        "--dem-cmap's sea half, else gpsplot's blues 'sea'); 'none' = flat "
        "water without sea-floor relief",
    )  # fmt: skip
    g.add_argument(
        "--glacier-cmap", default=None, metavar="CPT|COLOUR",
        help="glacier colours: a colour (default white) or a CPT, named as "
        "--dem-cmap, stretched over the ice heights",
    )  # fmt: skip
    g.add_argument(
        "--dem-smooth", type=float, default=None, metavar="KM",
        help="smooth the --dem hillshade with a Gaussian filter KM wide "
        "(e.g. 3 for an Iceland overview); colours and coasts stay sharp",
    )  # fmt: skip
    g.add_argument(
        "--glaciers", default=None, metavar="FILE",
        help="glacier polygons, drawn white with their own gentle hillshade "
        "(default with --dem iceland: IMO's 2025 outlines; 'none' = off)",
    )  # fmt: skip
    g.add_argument(
        "--lakes", default=None, metavar="FILE",
        help="lake polygons, filled as water (default with --dem iceland: "
        "IS 50V lakes and wide rivers; 'none' = off)",
    )  # fmt: skip
    g.add_argument(
        "--overlay", action="append", default=[], metavar="GEOJSON",
        help="outline layer (WGS84 GeoJSON, repeatable), e.g. lava flows; "
        "drawn over the background and under the symbols",
    )  # fmt: skip
    g.add_argument(
        "--overlay-pen", default=None, metavar="PEN",
        help="outline pen of the --overlay layers (default 0.6p,darkred)",
    )  # fmt: skip
    g.add_argument(
        "--overlay-fill", default=None, metavar="FILL",
        help="polygon fill of the --overlay layers (default orange@60, "
        "60%% transparent; 'none' = outline only)",
    )  # fmt: skip
    g.add_argument(
        "--gmt", action="append", default=[], metavar="STEP:KEY=VALUE",
        help="a GMT argument for one drawing step, replacing gpsplot's own: "
        "KEY is a PyGMT long name or one GMT letter, e.g. coast:A=500/0/1, "
        "glaciers:pen=0.3p,blue, dem:t=20 (repeatable; steps: "
        + ", ".join(gmt_overrides.LAYERS) + ")",
    )  # fmt: skip
    g.add_argument(
        "--gmt-config", action="append", default=[], metavar="KEY=VALUE",
        help="a GMT default for the whole map, e.g. MAP_FRAME_TYPE=plain, "
        "FONT_ANNOT_PRIMARY=8p (repeatable)",
    )  # fmt: skip
    g.add_argument("--title", default=None, help="map title")
    g.add_argument(
        "--labels", dest="labels", action="store_const", const=True, default=None,
        help=f"station names (default: on; station maps with more than "
        f"{LABEL_LIMIT} stations: off)",
    )  # fmt: skip
    g.add_argument(
        "--no-labels", dest="labels", action="store_const", const=False,
        help="no station names",
    )  # fmt: skip
    g.add_argument(
        "--logo", default=None, metavar="PNG",
        help="logo at the top left (default the Veðurstofa logo; 'none' = no logo)",
    )  # fmt: skip
    g.add_argument(
        "--no-scale-bar", dest="scale_bar", action="store_false",
        help="no distance scale bar",
    )  # fmt: skip
    out = p.add_argument_group("output")
    out.add_argument(
        "--save", nargs="?", default=None, const="pdf",
        help="save as these formats, comma separated (pdf, png, eps; default "
        "pdf); without --save the map opens in a viewer",
    )  # fmt: skip
    out.add_argument("--name", default=None, help="fixed output basename")
    out.add_argument(
        "-d", "--figDir", default=None,
        help="figure directory (default: fig_dir in postprocess.cfg [PATHS], "
        f"currently {common.config_dir(config, 'fig_dir', None) or 'unset'})",
    )  # fmt: skip


def _add_station_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "Stations", nargs="*",
        help='stations or stations.cfg patterns ("SEY?", "*"); none = every '
        "station inside --region",
    )  # fmt: skip
    p.add_argument(
        "--exclude", default=None, metavar="STA[,STA...]",
        help="stations (or patterns) left out",
    )  # fmt: skip
    p.add_argument(
        "--no-stations", dest="markers", action="store_false",
        help="no station symbols or names (the stations still set the "
        "map's extent when --region is not given)",
    )  # fmt: skip


def _add_series_args(p: argparse.ArgumentParser) -> None:
    common.add_series_options(
        p,
        refs=REFS,
        ref_help="series the vectors are measured on: detrend = plate "
        "velocity and the stored secular + seasonal model removed, so the "
        "vectors are relative to each station's background; plate = only "
        "the plate velocity removed. Default: plate for --window/--last "
        "velocity maps, detrend otherwise",
        views=["raw", "cleaned"],
        types=["POS", "TOT"],
        sessions=["24h", "8h"],
        session_help="24h daily solutions (default) or 8h sessions (itrf2020)",
    )


def build_parser(
    parser: argparse.ArgumentParser | None = None, config: Any = None
) -> argparse.ArgumentParser:
    if parser is None:
        parser = argparse.ArgumentParser(prog="gpsplot map")
    parser.description = textwrap.fill(
        "Maps (PyGMT): stations, velocities, displacements and slip.",
        width=78,
    )
    parser.formatter_class = argparse.RawDescriptionHelpFormatter
    parser.epilog = EXAMPLES
    sub = parser.add_subparsers(dest="kind", metavar="KIND", required=True)

    p = sub.add_parser(
        "station", aliases=[BASE],
        help="station markers ('base': the background alone, as --no-stations)",
    )  # fmt: skip
    _add_station_args(p)
    p.add_argument(
        "--tos", action="store_true",
        help="the GPS stations of TOS (the station register; with campaign, "
        "external and discontinued ones, one symbol each) instead of "
        "stations.cfg; patterns and --region select among them",
    )  # fmt: skip
    p.add_argument(
        "--tos-search", default=None, metavar="SEARCH",
        help="the TOS stations a 'tos search' selects (implies --tos): its "
        "expressions and flags in one quoted string, e.g. \"--epos\", "
        "\"--active-gps --receiver polarx5\", \"'subtype = SIL stöð'\"; "
        "'subtype = GPS stöð' unless the search names a subtype "
        "('subtype = all' lifts it). See 'tos search --help'",
    )  # fmt: skip
    p.add_argument(
        "--group-by", default=None, metavar="CODE",
        help="legend groups of TOS stations: status (default for GPS "
        "stations), subtype (default for a mix) or any TOS attribute code",
    )  # fmt: skip
    p.add_argument(
        "--refresh-tos", action="store_true",
        help="refetch the TOS station list (cached for a day)",
    )  # fmt: skip
    p.add_argument(
        "--stations-file", action="append", default=[], metavar="FILE[:LABEL]",
        help="stations from a file, one legend group each (repeatable): "
        "'MARKER LON LAT [NAME]' lines, a CSV with marker/lon/lat columns or "
        "GeoJSON points; LABEL names the group (default: the file name). "
        "Alone they are the whole map; with stations, --tos or --tos-search "
        "they are drawn beside them",
    )  # fmt: skip
    _add_map_options(p, config)

    p = sub.add_parser(
        "velocity",
        help="velocity arrows: windowed line fits, stored rates or a GeoJSON",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=EXAMPLES,
    )
    _add_station_args(p)
    p.add_argument(
        "--window", type=_window, action="append", default=None,
        metavar="START:END",
        help="fit a line to each station's series over START:END (YYYYMMDD; "
        "END empty = now); repeat for more windows, one colour each",
    )  # fmt: skip
    p.add_argument(
        "--last", type=int, action="append", default=None, metavar="DAYS",
        help="a window of the last DAYS days (repeatable)",
    )  # fmt: skip
    p.add_argument(
        "-e", "--end", type=_date, default=None,
        help="YYYYMMDD: the end of --last and of an open --window (default now)",
    )  # fmt: skip
    p.add_argument(
        "--geojson", default=None, metavar="FILE",
        help="a gps_api velocities product instead of the series",
    )  # fmt: skip
    p.add_argument(
        "--vertical", action="store_true",
        help="the vertical rate, drawn as an arrow pointing north for uplift",
    )  # fmt: skip
    p.add_argument(
        "--min-snr", type=float, default=0.0, metavar="R",
        help="drop vectors with |v| / |2σ| below R (the Reykjanes maps used "
        "0.75); default 0 = keep all",
    )  # fmt: skip
    p.add_argument(
        "--min-epochs", type=int, default=3,
        help="fewest epochs in a window for a fit (default 3)",
    )  # fmt: skip
    p.add_argument("--scale", type=float, default=None, metavar="CM_PER_UNIT",
                   help="arrow length per mm/yr [cm] (default: automatic)")  # fmt: skip
    p.add_argument("--scale-ref", type=float, default=None, metavar="MM_YR",
                   help="length of the legend arrow (default: automatic)")  # fmt: skip
    _add_series_args(p)
    _add_map_options(p, config)

    p = sub.add_parser("deformation", help="displacement from -s to -e (+ Mogi)")
    _add_station_args(p)
    p.add_argument("-s", "--start", type=_date, required=True, help="YYYYMMDD")
    p.add_argument("-e", "--end", type=_date, default=None,
                   help="YYYYMMDD (default now)")  # fmt: skip
    p.add_argument(
        "--mogi", type=_mogi, action="append", default=None,
        metavar="LON,LAT,DEPTH_KM,DV_M3",
        help="add a Mogi model field (repeatable)"
    )  # fmt: skip
    p.add_argument("--scale", type=float, default=None, metavar="CM_PER_MM",
                   help="arrow length per mm [cm] (default: automatic)")  # fmt: skip
    p.add_argument("--scale-ref", type=float, default=None, metavar="MM",
                   help="length of the legend arrow (default: automatic)")  # fmt: skip
    _add_series_args(p)
    _add_map_options(p, config)

    p = sub.add_parser("slip", help="a gps_api Okada slip product")
    p.add_argument(
        "product", help="SlipDistributionResult JSON (models/<region>_slip.json)"
    )
    p.add_argument(
        "--component", default=None,
        choices=["opening", "strike_slip", "dip_slip"],
        help="slip component (default: the largest potency)",
    )  # fmt: skip
    p.add_argument("--view", default="map", choices=["map", "plane"],
                   help="map view or the fault-plane section")  # fmt: skip
    p.add_argument("--stations", nargs="+", default=None, metavar="STA",
                   help="station markers on the map view (patterns allowed)")  # fmt: skip
    _add_map_options(p, config)
    return parser


# ---------------------------------------------------------------------------
# stations, windows, scale
# ---------------------------------------------------------------------------


def _patterns(specs: Sequence[str], stations: Sequence[str]) -> list[str]:
    out: list[str] = []
    for spec in specs:
        got = sorted(s for s in stations if fnmatch.fnmatchcase(s, spec.upper()))
        if not got:
            raise ValueError(f"{spec!r} matches no station")
        out.extend(g for g in got if g not in out)
    return out


def select_stations(
    specs: Sequence[str],
    stations: Sequence[str],
    coords: Any,
    *,
    region: tuple[float, float, float, float] | None,
    exclude: str | None,
) -> tuple[list[Any], list[str]]:
    """``(coordinates, dropped)`` of the stations the map shows.

    ``specs`` are patterns over ``stations`` (all of them when empty, then
    ``region`` is required); with ``region`` only stations inside it stay.
    ``coords(markers)`` resolves one marker list (``maps.station_coordinates``);
    a station it cannot place is dropped, not fatal.

    Raises:
        ValueError: no specs and no region, a pattern matching nothing, or
            nothing left.
    """
    if not specs and region is None:
        raise ValueError("give stations, or --region for every station inside it")
    names = _patterns(specs or ["*"], stations)
    if exclude:
        pats = [p.upper() for p in exclude.split(",") if p]
        names = [n for n in names if not any(fnmatch.fnmatchcase(n, p) for p in pats)]
    named = {s.upper() for s in specs if not any(ch in s for ch in "*?[")}
    kept, dropped = [], []
    for name in names:
        try:
            (c,) = coords([name])
        except (KeyError, ValueError) as exc:
            # a pattern sweeps in sections without coordinates: only a
            # station asked for by name is worth a word
            if name in named:
                dropped.append(f"{name}: {exc}")
            continue
        if region is not None:
            w, e, s, n = region
            if not (w <= c.lon <= e and s <= c.lat <= n):
                continue
        kept.append(c)
    if not kept:
        raise ValueError("no station to plot")
    return kept, dropped


def resolve_windows(
    windows: Sequence[tuple[dt.datetime, dt.datetime | None]] | None,
    last: Sequence[int] | None,
    now: dt.datetime,
) -> list[tuple[dt.datetime, dt.datetime]]:
    """``--window`` and ``--last`` as concrete (start, end) pairs, in order."""
    out = [(s, e or now) for s, e in windows or ()]
    out += [(now - dt.timedelta(days=d), now) for d in last or ()]
    return out


def window_label(start: dt.datetime, end: dt.datetime) -> str:
    return f"{start:%Y/%m/%d} - {end:%Y/%m/%d}"


def _nice(x: float) -> float:
    """The 1-2-5 number at or below ``x`` (> 0)."""
    k = math.floor(math.log10(x))
    for m in (5, 2, 1):
        if m * 10**k <= x:
            return float(m * 10**k)
    return float(10**k)


def auto_scale(lengths: Any, target_cm: float = TARGET_ARROW_CM) -> tuple[float, float]:
    """``(scale, scale_ref)``: cm per unit, and a round legend arrow.

    The 90th percentile of ``lengths`` is drawn ``target_cm`` long; the
    legend arrow is the 1-2-5 number at or below it.
    """
    v = np.asarray(lengths, dtype=float)
    v = v[np.isfinite(v) & (v > 0)]
    ref = float(np.percentile(v, 90)) if v.size else 1.0
    return target_cm / ref, _nice(ref)


def _yearf(t: dt.datetime) -> float:
    from gtimes.timefunc import TimetoYearf

    return float(TimetoYearf(*t.timetuple()[:6]))


def _request(args: argparse.Namespace, parser: Any, config: Any,
             start: dt.datetime, end: dt.datetime) -> Any:  # fmt: skip
    from gps_plot.sources import gps as gps_source

    kwargs: dict[str, Any] = {
        "frame": args.frame, "session": args.session, "figDir": "",
        "tType": args.tType,
        "Dir": args.Dir, "pos_id": None,
    }  # fmt: skip
    common.resolve_source(kwargs, args, parser, config)
    return gps_source.SeriesRequest(
        fstart=_yearf(start),
        fend=_yearf(end),
        ref=args.ref or "detrend",
        Dir=kwargs["Dir"],
        tType=kwargs["tType"],
        uncert=args.uncert,
        view=args.view,
        pos_id=kwargs.get("pos_id"),
    )


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


def _style(
    args: argparse.Namespace,
    region: Any,
    *,
    config: Any = None,
    points: Sequence[Any] = (),
) -> dict[str, Any]:
    """Map-function keywords from the map options.

    ``--dem iceland`` needs the map's extent to pick a level: ``region``,
    else the box around ``points`` (what the map function derives too),
    else all Iceland.
    """
    kw: dict[str, Any] = {"region": region, "title": args.title}
    for opt, key in (("margin", "margin"), ("projection", "projection"),
                     ("resolution", "resolution"), ("dem", "dem_grid"),
                     ("dem_cmap", "dem_cmap"),
                     ("bathymetry_cmap", "bathymetry_cmap"),
                     ("glacier_cmap", "glacier_cmap"),
                     ("dem_smooth", "dem_smooth")):  # fmt: skip
        if getattr(args, opt) is not None:
            kw[key] = getattr(args, opt)
    if str(kw.get("dem_grid", "")).lower() == ICELAND_DEM:
        from gps_plot import maps

        extent = region
        if extent is None and points:
            extent = _region_of(points, args.margin, maps)
        choice = maps.iceland_dem(
            gis_dir(config),
            extent or maps.ICELAND_DEM_LEVELS[-1][2],
            kw.get("projection", maps.DEFAULT_PROJECTION),
        )
        kw["dem_grid"] = choice.grid
        for key in ("coastline", "glaciers", "lakes", "glacier_contours"):
            if getattr(choice, key) is not None:
                kw[key] = getattr(choice, key)
    for key in ("glaciers", "lakes"):
        value = getattr(args, key, None)
        if value is not None:
            if value.lower() == "none":
                kw.pop(key, None)
                if key == "glaciers":
                    kw.pop("glacier_contours", None)
            else:
                kw[key] = value
    if "dem_grid" in kw:
        kw.setdefault("dem_cmap", DEFAULT_DEM_CPT)
    if args.overlay:
        from gps_plot.maps import DEFAULT_OVERLAY_FILL, DEFAULT_OVERLAY_PEN, MapOverlay

        fill = args.overlay_fill or DEFAULT_OVERLAY_FILL
        kw["overlays"] = tuple(
            MapOverlay(path, pen=args.overlay_pen or DEFAULT_OVERLAY_PEN,
                       fill=None if fill.lower() == "none" else fill)
            for path in args.overlay
        )  # fmt: skip
    return kw


def resolve_palettes(args: argparse.Namespace, pygmt_mod: Any, user_dir: str) -> None:
    """Resolve the colour-map options in place, before the map is drawn.

    A name found as a CPT file (``user_dir``, gpsplot's own) becomes its
    path, so the map functions need not know ``user_dir``. A name that is
    no CPT anywhere must be a colour GMT knows: GMT would otherwise fail
    late, mid-map, with a traceback.

    Raises:
        ValueError: naming the closest known colour maps.
    """
    from gps_plot import palettes

    for opt in CMAP_OPTIONS:
        name = getattr(args, opt, None)
        if name is None or (opt == "bathymetry_cmap" and name.lower() == palettes.NONE):
            continue
        palette = palettes.resolve(name, user_dir=user_dir)
        if palette.kind == palettes.FILE:
            setattr(args, opt, palette.source)
        if palette.kind != palettes.COLOUR:
            continue
        try:  # not a CPT anywhere: GMT must know it as a colour
            palettes.segment(pygmt_mod, palette, 0.0, 1.0)
        except Exception as exc:  # noqa: BLE001 - GMTCLibError, kept lazy
            close = palettes.suggest(name, user_dir=user_dir)
            hint = f"; did you mean {', '.join(close)}?" if close else ""
            flag = "--" + opt.replace("_", "-")
            raise ValueError(
                f"{flag} {name!r} is not a CPT (gpsplot's, yours in "
                f"{user_dir}, GMT's or a file) nor a GMT colour{hint} "
                f"({flag} list prints the names)"
            ) from exc


def print_palettes(user_dir: str) -> None:
    """Every colour-map name, by source (``--dem-cmap list``)."""
    from gps_plot import palettes

    for source, names in palettes.available(user_dir=user_dir).items():
        print(f"{source}:")
        print(textwrap.indent(textwrap.fill(" ".join(names) or "(none)", 76), "  "))


def cpt_dir(config: Any) -> str:
    """The user's CPT directory: ``cpt/`` of :func:`gis_dir`."""
    return os.path.join(gis_dir(config), "cpt")


def gis_dir(config: Any) -> str:
    """The local GIS directory (DEM grids, vector layers, CPTs): gis_dir in
    postprocess.cfg [PATHS], else :data:`DEFAULT_GIS_DIR`."""
    found = common.config_dir(config, "gis_dir", None) if config is not None else None
    return os.path.expanduser(found or DEFAULT_GIS_DIR)


def scale_bar_km(region: Sequence[float]) -> float:
    """A round scale-bar length: about a fifth of the map width [km]."""
    w, e, south, north = region
    lat = math.radians((south + north) / 2.0)
    width = (e - w) * 111.32 * max(math.cos(lat), 0.01)
    return _nice(max(width / 5.0, 0.1))


def _finish(fig: Any, args: argparse.Namespace) -> None:
    """Scale bar and logo on top of a finished map view."""
    if args.scale_bar:
        km = scale_bar_km(fig.region)
        # bottom right: top centre sat on the map, the SW corner holds the
        # vector legend, the logo is top left, the status legend top right
        from pygmt.params import Position

        fig.scalebar(
            length=f"{km:g}k", label=True, unit=True, box="+gwhite@20",
            position=Position("BR", cstype="inside", offset=("0.4c", "1.0c")),
        )  # fmt: skip
    logo = args.logo if args.logo is not None else str(DEFAULT_LOGO)
    if logo.lower() != "none" and os.path.isfile(logo):
        fig.image(
            **gmt_overrides.layer_kw(
                "logo",
                imagefile=logo,
                position=f"jTL+o0.35c/0.15c+w{LOGO_WIDTH}",
                transparency=LOGO_TRANSPARENCY,
            )
        )


def _save(fig: Any, args: argparse.Namespace, config: Any, stem: str) -> list[str]:
    if not args.save:
        fig.show(method="external")
        return []
    fig_dir = args.figDir if args.figDir is not None else (
        common.config_dir(config, "fig_dir", "") or ""
    )  # fmt: skip
    base = os.path.join(fig_dir, args.name or stem)
    paths = []
    for fmt in args.save.split(","):
        path = f"{base}.{fmt.strip()}"
        fig.savefig(path)
        paths.append(path)
    return paths


def _stem(kind: str, first: str, n: int, *parts: str) -> str:
    return "-".join([f"map_{kind}_{first}_{n}", *[p for p in parts if p]])


def _report(dropped: Sequence[str]) -> None:
    for line in dropped:
        print(f"dropped {line}", file=sys.stderr)


def run(args: argparse.Namespace, parser: argparse.ArgumentParser, config: Any) -> int:
    import warnings

    import pygmt

    from gps_plot import maps

    try:
        layers = gmt_overrides.parse(args.gmt)
        gmt_overrides.check(layers, pygmt)
        defaults = gmt_config(args.gmt_config)
    except ValueError as exc:
        parser.error(str(exc))
    with contextlib.ExitStack() as stack:
        stack.enter_context(
            pygmt.config(
                FONT_TITLE=TITLE_FONT,
                MAP_TITLE_OFFSET="6p",
                **maps.DEFAULT_GMT_DEFAULTS,
            )
        )
        try:  # the user's defaults win; GMT checks them on entry
            stack.enter_context(pygmt.config(**defaults))
        except pygmt.exceptions.GMTCLibError as exc:
            parser.error(f"--gmt-config: {str(exc).splitlines()[-1]}")
        stack.enter_context(gmt_overrides.overrides(layers))
        # a GMT letter in --gmt is the user's choice: no PyGMT nagging
        stack.enter_context(warnings.catch_warnings())
        warnings.filterwarnings("ignore", "Short-form parameter", SyntaxWarning)
        return _run(args, parser, config, maps)


def gmt_config(specs: Sequence[str]) -> dict[str, str]:
    """``KEY=VALUE`` specs (``--gmt-config``) as GMT defaults.

    Raises:
        ValueError: a spec without ``=``.
    """
    out = {}
    for spec in specs:
        key, eq, value = spec.partition("=")
        if not (eq and key.strip()):
            raise ValueError(f"--gmt-config {spec!r}: use KEY=VALUE")
        out[key.strip().upper()] = value.strip()
    return out


def _run(args: argparse.Namespace, parser: Any, config: Any, maps: Any) -> int:
    if any(str(getattr(args, o, None)).lower() == LIST for o in CMAP_OPTIONS):
        print_palettes(cpt_dir(config))
        return 0
    try:
        import pygmt

        resolve_palettes(args, pygmt, cpt_dir(config))
    except ValueError as exc:
        parser.error(str(exc))

    def coords(markers: Sequence[str]) -> Any:
        return maps.station_coordinates(markers, parser=config)

    if args.kind in ("station", BASE):
        return _run_station(args, parser, config, maps, coords)
    if args.labels is None:
        args.labels = True
    if args.kind == "slip":
        return _run_slip(args, parser, config, maps, coords)
    if args.kind == "velocity" and args.geojson:
        return _run_geojson(args, config, maps)
    try:
        stations, dropped = select_stations(
            args.Stations, config.getStationInfo(), coords,
            region=args.region, exclude=args.exclude,
        )  # fmt: skip
    except ValueError as exc:
        parser.error(str(exc))
    _report(dropped)
    if args.kind == "velocity":
        return _run_velocity(args, parser, config, maps, stations)
    return _run_deformation(args, parser, config, maps, stations)


def tos_coords(stations: Sequence[Any], maps: Any, group_by: str | None = None) -> Any:
    """A ``coords(markers)`` resolver over TOS stations (for
    :func:`select_stations`): markers to coordinates carrying their legend
    group (:func:`gps_plot.sources.tos.group_key`; default the status).
    A marker held by stations of several subtypes resolves to the first."""
    from gps_plot.sources import tos

    by_marker: dict[str, Any] = {}
    for s in stations:
        by_marker.setdefault(
            s.marker,
            maps.StationCoordinate(
                s.marker, s.lon, s.lat, s.name, tos.group_key(s, group_by)
            ),
        )

    def coords(markers: Sequence[str]) -> Any:
        try:
            return tuple(by_marker[m.upper()] for m in markers)
        except KeyError as exc:
            raise KeyError(f"not a TOS station: {exc.args[0]}") from None

    return coords


def _inside(c: Any, region: tuple[float, float, float, float] | None) -> bool:
    if region is None:
        return True
    w, e, s, n = region
    return bool(w <= c.lon <= e and s <= c.lat <= n)


def _file_stations(args: Any, parser: Any, maps: Any) -> list[Any]:
    from gps_plot.sources import station_files

    out = []
    for spec in args.stations_file:
        try:
            out += station_files.read_stations(*station_files.split_spec(spec), maps)
        except (OSError, ValueError) as exc:
            parser.error(f"--stations-file: {exc}")
    return [c for c in out if _inside(c, args.region)]


def _tos_fleet(args: Any, parser: Any) -> tuple[Any, str]:
    """The TOS stations of --tos / --tos-search and their legend grouping."""
    from gps_plot.sources import tos

    try:
        if args.tos_search is not None:
            fleet = tos.search_stations(args.tos_search, refresh=args.refresh_tos)
        else:  # GPS stations only: by status unless --group-by says otherwise
            fleet = tos.tos_gps_stations(refresh=args.refresh_tos)
            return fleet, args.group_by or tos.GROUP_STATUS
    except ValueError as exc:
        parser.error(f"--tos-search: {exc}")
    if not fleet:
        parser.error("the TOS search found no station with a position")
    return fleet, args.group_by or tos.default_group(fleet)


def _run_station(args: Any, parser: Any, config: Any, maps: Any, coords: Any) -> int:
    if args.kind == BASE:
        args.markers = False
    use_tos = args.tos or args.tos_search is not None
    files = _file_stations(args, parser, maps)
    stations: list[Any] = []
    if use_tos or args.Stations or not (files or not args.markers):
        names: Sequence[str]
        if use_tos:
            fleet, group = _tos_fleet(args, parser)
            names, coords = [s.marker for s in fleet], tos_coords(fleet, maps, group)
            specs = args.Stations or (["*"] if args.region is None else [])
        else:
            names, specs = config.getStationInfo(), args.Stations
        try:
            stations, dropped = select_stations(
                specs, names, coords, region=args.region, exclude=args.exclude
            )
        except ValueError as exc:
            parser.error(str(exc))
        _report(dropped)
        if files and not any(c.status for c in stations):
            stations = [
                dataclasses.replace(c, status=STATIONS_CFG_GROUP) for c in stations
            ]
    stations += files
    region = args.region
    if region is None and not stations:
        region = DEFAULT_BASE_REGION
    if args.markers:
        n = len(stations)
        labels = args.labels if args.labels is not None else n <= LABEL_LIMIT
    else:
        labels = False
    fig = maps.station_map(
        stations, labels=labels, markers=args.markers,
        **_style(args, region, config=config, points=stations),
    )  # fmt: skip
    _finish(fig, args)
    if stations:
        stem = _stem(
            "station", stations[0].marker, len(stations), "tos" if use_tos else ""
        )
    else:
        stem = _stem(BASE, "region", 0)
    if stations and not args.markers:
        stem = stem.replace("map_station_", f"map_{BASE}_", 1)
    for p in _save(fig, args, config, stem):
        print(p)
    return 0


def _run_velocity(args: Any, parser: Any, config: Any, maps: Any, stations: Any) -> int:
    from gps_plot.sources import velocity as vsource

    now = args.end or dt.datetime.now()
    windows = resolve_windows(args.window, args.last, now)
    if windows and args.ref is None:
        # over days to weeks the secular rate is negligible, and plate covers
        # every station (about half lack a stored detrend record)
        args.ref = "plate"
    comp = 2 if args.vertical else None
    vectors, dropped, notes = [], [], []
    groups: list[tuple[str | None, str]] = []
    if windows:
        for k, (start, end) in enumerate(windows):
            label = window_label(start, end)
            groups.append((label, WINDOW_COLORS[k % len(WINDOW_COLORS)]))
            req = _request(args, parser, config, start, end)
            for c in stations:
                try:
                    r = vsource.window_velocity(
                        c.marker, req, min_epochs=args.min_epochs
                    )
                except Exception as exc:  # one station must not sink the map
                    dropped.append(f"{c.marker} {label}: {exc}")
                    continue
                vectors.append(_vector(maps, c, r, comp, label))
    else:
        now_req = _request(args, parser, config, now - dt.timedelta(days=1), now)
        groups.append(("secular", "blue"))
        for c in stations:
            try:
                r = vsource.record_velocity(c.marker, now_req)
            except Exception as exc:
                dropped.append(f"{c.marker}: {exc}")
                continue
            vectors.append(_vector(maps, c, r, comp, "secular"))
    kept = [v for v in vectors if snr(v) >= args.min_snr]
    notes += [f"{v.marker} {v.method}: |v|/|2σ| = {snr(v):.2f} < {args.min_snr:g}"
              for v in vectors if snr(v) < args.min_snr]  # fmt: skip
    _report(dropped + notes)
    if not kept:
        print("nothing plotted", file=sys.stderr)
        return 1
    no_record = sum("no stored detrend record" in d for d in dropped)
    if no_record and args.ref == "detrend":
        print(f"{no_record} without a stored detrend record: --ref plate draws "
              "them (plate velocity removed only)", file=sys.stderr)  # fmt: skip
    scale, ref = auto_scale([math.hypot(v.east, v.north) for v in kept])
    style = _style(
        args, args.region or _region_of(stations, args.margin, maps), config=config
    )
    if args.title is None:
        style["title"] = _velocity_title(args, bool(windows))
    fig = maps.velocity_map(
        kept,
        scale=args.scale or scale,
        scale_ref=args.scale_ref or ref,
        method_colors=dict(g for g in groups if g[0] is not None),
        ellipse_pen=ELLIPSE_PEN,
        labels=args.labels,
        markers=args.markers,
        **style,
    )
    _finish(fig, args)
    span = "_".join(f"{s:%Y%m%d}-{e:%Y%m%d}" for s, e in windows) or "secular"
    stem = _stem("velocity", stations[0].marker, len(stations), span,
                 "vertical" if args.vertical else "horizontal")  # fmt: skip
    for p in _save(fig, args, config, stem):
        print(p)
    return 0


def _vector(maps: Any, c: Any, r: Any, comp: int | None, method: str) -> Any:
    if comp is None:  # rows N, E, U
        e, n, se, sn = r.neu[1], r.neu[0], r.sigma[1], r.sigma[0]
    else:  # vertical: up as a north-pointing arrow (the gpsvelo convention)
        e, n, se, sn = 0.0, r.neu[comp], 0.0, r.sigma[comp]
    return maps.VelocityVector(
        marker=c.marker, lon=c.lon, lat=c.lat, east=float(e), north=float(n),
        sigma_east=float(se), sigma_north=float(sn), method=method,
    )  # fmt: skip


def snr(v: Any) -> float:
    """``|v| / |2σ|`` of a vector (inf when σ is 0)."""
    s = 2.0 * math.hypot(v.sigma_east, v.sigma_north)
    m = math.hypot(v.east, v.north)
    return math.inf if s == 0 else m / s


def _velocity_title(args: Any, windowed: bool) -> str:
    what = "Vertical velocity" if args.vertical else "Horizontal velocity"
    if not windowed:
        return f"{what}, secular (plate removed)"
    ref = (
        "relative to background"
        if (args.ref or "detrend") == "detrend"
        else "plate removed"
    )
    return f"{what}, {ref}"


def _region_of(stations: Any, margin: float | None, maps: Any) -> Any:
    return maps.region_around(
        stations, margin=maps.DEFAULT_MARGIN_DEG if margin is None else margin
    )


def _run_deformation(
    args: Any, parser: Any, config: Any, maps: Any, stations: Any
) -> int:
    from gps_plot.sources import velocity as vsource

    end = args.end or dt.datetime.now()
    if end <= args.start:
        parser.error(f"-e {end:%Y%m%d} is not after -s {args.start:%Y%m%d}")
    req = _request(args, parser, config, args.start, end)
    got, dropped = [], []
    for c in stations:
        try:
            got.append((c, vsource.window_displacement(c.marker, req)))
        except Exception as exc:
            dropped.append(f"{c.marker}: {exc}")
    _report(dropped)
    if not got:
        print("nothing plotted", file=sys.stderr)
        return 1
    coords = [c for c, _ in got]
    east = np.array([r.neu[1] for _, r in got])
    north = np.array([r.neu[0] for _, r in got])
    lon = np.array([c.lon for c in coords])
    lat = np.array([c.lat for c in coords])
    models = [
        maps.mogi_model_field(
            lon,
            lat,
            source_lon=m[0],
            source_lat=m[1],
            depth=m[2] * 1e3,
            dv=m[3],
            name=f"Mogi {m[2]:g} km, {m[3]:.2g} m³",
            color=WINDOW_COLORS[(k + 1) % len(WINDOW_COLORS)],
        )  # fmt: skip
        for k, m in enumerate(args.mogi or ())
    ]
    scale, ref = auto_scale(np.hypot(east, north))
    style = _style(args, args.region, config=config, points=coords)
    if args.title is None:
        style["title"] = f"Displacement {window_label(args.start, end)}"
    fig = maps.deformation_vectors(
        coords, east, north, models,
        observed_label=window_label(args.start, end),
        scale=args.scale or scale, scale_ref=args.scale_ref or ref,
        labels=args.labels, markers=args.markers, **style,
    )  # fmt: skip
    _finish(fig, args)
    stem = _stem("deformation", coords[0].marker, len(coords),
                 f"{args.start:%Y%m%d}-{end:%Y%m%d}", "mogi" if models else "")  # fmt: skip
    for p in _save(fig, args, config, stem):
        print(p)
    return 0


def _run_geojson(args: Any, config: Any, maps: Any) -> int:
    import json

    with open(args.geojson) as f:
        vectors = maps.velocity_vectors_from_geojson(json.load(f))
    if args.Stations:
        keep = set(_patterns(args.Stations, [v.marker for v in vectors]))
        vectors = tuple(v for v in vectors if v.marker in keep)
    if args.region is not None:
        w, e, s, n = args.region
        vectors = tuple(v for v in vectors if w <= v.lon <= e and s <= v.lat <= n)
    if not vectors:
        print("nothing plotted", file=sys.stderr)
        return 1
    scale, ref = auto_scale([math.hypot(v.east, v.north) for v in vectors])
    fig = maps.velocity_map(
        vectors, scale=args.scale or scale, scale_ref=args.scale_ref or ref,
        labels=args.labels, markers=args.markers,
        **_style(args, args.region, config=config, points=vectors),
    )  # fmt: skip
    _finish(fig, args)
    stem = _stem("velocity", Path(args.geojson).stem, len(vectors), "geojson")
    for p in _save(fig, args, config, stem):
        print(p)
    return 0


def _run_slip(args: Any, parser: Any, config: Any, maps: Any, coords: Any) -> int:
    import json

    with open(args.product) as f:
        product = json.load(f)
    stations = None
    if args.stations:
        try:
            stations, dropped = select_stations(
                args.stations, config.getStationInfo(), coords,
                region=None, exclude=None,
            )  # fmt: skip
        except ValueError as exc:
            parser.error(str(exc))
        _report(dropped)
    style = _style(args, args.region, config=config, points=stations or ())
    fig = maps.slip_map(
        product, component=args.component, view=args.view,
        stations=stations, labels=args.labels, **style,
    )  # fmt: skip
    if args.view == "map":
        _finish(fig, args)
    region = str(product.get("region", Path(args.product).stem))
    for p in _save(fig, args, config, f"map_slip_{region}-{args.view}"):
        print(p)
    return 0
