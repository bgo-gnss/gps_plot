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
import re
import os
import sys
import textwrap
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from gps_plot import gmt_overrides
from gps_plot.cli import common

KINDS = ("station", "velocity", "deformation", "slip")
REFS = ("detrend", "plate")
#: velocity maps: the velocity product's refs (geo_dataread.velocity.REFS);
#: detrend is the old name of background
VELOCITY_REFS = ("plate-periodic", "background", "plate", "pre-onset", "detrend")
#: one colour per window, in order (the Reykjanes maps: blue, then red)
WINDOW_COLORS = ("blue", "red", "darkgreen", "darkorange", "purple", "brown")
#: the 90th-percentile arrow is drawn this long [cm] by the automatic scale
TARGET_ARROW_CM = 2.0
#: ... and in a zoom inset, whose arrows get their own scale
ZOOM_ARROW_CM = 1.0
#: the Veðurstofa name logo, top left (source checkout only: not packaged):
#: the two-line logo of the RTK plots, drawn 30 % transparent as they do
DEFAULT_LOGO = (
    Path(__file__).resolve().parents[2] / "logos" / "vi_logos" / "VI_Two_Line_Blue.png"
)
LOGO_WIDTH = "2.5c"
#: the scale bar's height above the bottom edge, clear of the annotations
SCALE_BAR_FOOT = "1.0c"
SCALE_BAR_FOOT_OBLIQUE = "1.7c"
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
#: --dem arcticdem: the Iceland DEM, and a 2 m ArcticDEM area grid (resampled
#: to the map, its sea spliced) where one covers most of the map
ARCTIC_DEM = "arcticdem"
#: the local GIS directory (dem/, vector/, cpt/): gis_dir in postprocess.cfg
DEFAULT_GIS_DIR = "~/gps-data/gis"
#: station maps label their stations by default only up to this many
LABEL_LIMIT = 60
#: a --baselines pair, A-B
BASELINE_PAIR = re.compile(r"[A-Z0-9]{4}-[A-Z0-9]{4}")
#: the padding around a map of baselines alone: the lines fill the map
BASELINE_MARGIN_DEG = 0.03
#: a map with no stations and no --region (``gpsplot map base``): Iceland
DEFAULT_BASE_REGION = (-24.6, -13.2, 63.2, 66.7)
#: --fissure-swarms: the FutureVolc zones under the GIS directory
#: (vector/build_07_volcanic.sh: closed from its GeoJSON by close_outlines.py)
FISSURE_SWARM_ZONES = (
    "vector/futurevolc_fissure_swarms_closed.gmt",
    "vector/futurevolc_central_volcanoes_closed.gmt",
)
FISSURE_SWARM_OUTLINES = (
    "vector/futurevolc_fissure_swarms.geojson",
    "vector/futurevolc_central_volcanoes.geojson",
)
#: a full-Iceland map (neither --region nor --projection given): the
#: iceland3 overview's oblique stereographic frame, longitudes along their
#: meridians, and with a DEM a light-blue sea under the sea-floor hillshade
OVERVIEW_PROJECTION = "S-19/65/12c"
#: (the iceland3 frame -25/63/-12.8/66.7, opened south-west to -25.2°/62.9°
#: for the small logo bottom left, east to -12.4° for the scale bar, north
#: to 66.9° for the legend top left)
OVERVIEW_CORNERS = (-25.2, 62.9, -12.4, 66.9)
#: the overview's logo: smaller, bottom left (the annotations are drawn
#: over it should they meet)
OVERVIEW_LOGO_WIDTH = "1.6c"
#: ... set between the 24°W and 22°W annotations
OVERVIEW_LOGO_X = "1.45c"
OVERVIEW_SEA = "lightblue"
#: an extent at least this wide and tall [deg lon, deg lat] is all Iceland
FULL_ICELAND_SPAN = (8.0, 2.5)
#: ``map base``: a station map without stations
BASE = "base"
#: the legend group of stations.cfg stations drawn beside --stations-file ones
STATIONS_CFG_GROUP = "GPS stations"
TITLE_FONT = "12p,Helvetica-Bold"
#: --scale-style
SCALE_STYLES = ("fancy", "plain")
#: boxes kept free of insets [cm]: the logo (width, height, top left) and
#: the scale bar (width, top, bottom right); the vector legend's width
LOGO_BOX_CM = (3.0, 1.2)
SCALE_BOX_CM = (3.6, 2.0)
LEGEND_WIDTH_CM = 3.5
#: the scale bar's fonts: its 'km' label at the annotation size
SCALE_BAR_FONTS = {"FONT_LABEL": "8p", "FONT_ANNOT_PRIMARY": "7p",
                   "MAP_LABEL_OFFSET": "3p"}  # fmt: skip
#: room kept round each symbol for its name when placing the locator [cm]
LOCATOR_NAME_CM = 0.7
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
  gpsplot map velocity --region -22.72/-22.17/63.76/63.945 --last 365 --last 91 \\
      --min-fill 0.6 --dem iceland --title= \\
      --layers roads,towns,lavas,graben,fissures,barriers --scale-ref 100 --save pdf
                        Svartsengi, past year and past 3 months on one map,
                        report style: no title, only rates with data on 60%
                        of their window's days
  gpsplot map velocity --region -22.72/-22.17/63.76/63.945 --last 182:91 \\
      --last 91 --min-fill 0.6 --vertical --save pdf
                        the last 3 months against the 3 months before them
                        (windows can also overlap: --last 365 --last 30)
  gpsplot map velocity --region -24.6/-13.2/63.2/66.7 --save pdf
                        stored secular rates, Iceland
  gpsplot map deformation --region -22.8/-21.9/63.75/64.1 -s 20231101 -e 20231115 \\
      --mogi -22.46,63.87,4.5,1.2e7 --save pdf
                        displacement 1-15 Nov 2023 next to a Mogi source
                        (lon, lat, 4.5 km deep, 1.2e7 m3)
  gpsplot map slip models/reykjanes_slip.json --view plane --save pdf
                        an Okada slip model, seen on the fault plane
  gpsplot map base --dem iceland --dem-cmap icel --bathymetry-cmap lightblue \\
      --projection S-19/65/12c --region -25/63/-12.8/66.7+r --save pdf
                        the iceland3 overview: oblique stereographic, tilted
                        graticule, its relief colours and flat blue sea
  gpsplot map base --dem iceland --dem-cmap icel --bathymetry-cmap none \\
      --projection S-19/65/12c --region -25/63/-12.8/66.7+r \\
      --dem-zone vector/iceland3/fisswarms_fil.xy \\
      --dem-zone vector/iceland3/centralvolc.xy \\
      --dem-zone vector/iceland3/centralvolc2.xy:none --save pdf
                        icel-overv.pdf again: fissure swarms and central
                        volcanoes slate-blue under the hillshade
  gpsplot map station --tos --fissure-swarms --save pdf
                        every GPS station in TOS over the FutureVolc fissure
                        swarms and central volcanoes, Iceland overview
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


class Corners(tuple[float, float, float, float]):
    """``--region LON/LAT/LON/LAT+r``: the map's lower-left and
    upper-right corners, made a ``maps.CornerRegion`` once the projection
    is known."""


def _region(value: str) -> tuple[float, float, float, float]:
    from gps_plot import areas

    if value.lower() in areas.AREAS:
        return areas.AREAS[value.lower()]
    corner = value.endswith(("+r", "r"))
    body = value.removesuffix("+r").removesuffix("r") if corner else value
    try:
        a, b, c, d = (float(v) for v in body.split("/"))
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"{value!r}: use W/E/S/N or LON/LAT/LON/LAT+r corners in degrees, "
            "or an area name (reykjanes, svartsengi, …: gps_plot.areas)"
        ) from None
    if corner:
        if not (b < d):
            raise argparse.ArgumentTypeError(f"{value!r}: the upper-right corner "
                                             "must be north of the lower-left")  # fmt: skip
        return Corners((a, b, c, d))
    if not (a < b and c < d):
        raise argparse.ArgumentTypeError(f"{value!r}: need W < E and S < N")
    return (a, b, c, d)


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


def _last(value: str) -> tuple[int, int]:
    """``DAYS`` or ``FROM:TO`` days before the end: ``--last 7`` = ``7:0``."""
    first, sep, second = value.partition(":")
    try:
        span = (int(first), int(second) if sep else 0)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"{value!r}: use DAYS or FROM:TO (days before the end)"
        ) from None
    if not span[0] > span[1] >= 0:
        raise argparse.ArgumentTypeError(f"{value!r}: need FROM > TO >= 0")
    return span


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
        help="map bounds in degrees, or LON/LAT/LON/LAT+r: the lower-left and "
        "upper-right corners of a rectangle in --projection (an oblique map), "
        "or an area name (reykjanes, svartsengi, hengill, …); "
        "with no station given, every station inside it is plotted "
        "(default: the stations' box + --margin)",
    )  # fmt: skip
    g.add_argument(
        "--margin", type=float, default=None, metavar="DEG",
        help="padding around the stations when --region is not given",
    )  # fmt: skip
    g.add_argument(
        "--projection", default=None,
        help="GMT projection (default M12c; a full-Iceland map with neither "
        "--region nor --projection: the overview S-19/65/12c, see --region +r)",
    )  # fmt: skip
    g.add_argument(
        "--resolution", default=None,
        help="shoreline resolution: auto (default), full, high, intermediate, low",
    )  # fmt: skip
    g.add_argument(
        "--dem", default=None, metavar="GRID",
        help="hillshade background: 'iceland' (the Iceland topo-bathymetry "
        "DEM, the level the map needs, with the IS 50V coast), 'arcticdem' "
        "(iceland, with a 2 m ArcticDEM area grid where one covers most of "
        "the map: resampled to the map, its sea from the Iceland DEM), a grid file, "
        "or a GMT remote grid such as @earth_relief_15s (default on a "
        "full-Iceland overview: iceland, when built; 'none' = flat land)",
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
        "--sea-smooth", type=float, default=None, metavar="KM",
        help="smooth the --dem sea floor alone (colours and hillshade) with a "
        "Gaussian filter KM wide, e.g. 0.5: off Reykjanes the bathymetry is "
        "EMODnet's ~115 m cells, which the hillshade shows as blocks; land "
        "and the coast stay sharp",
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
        "--dem-zone", action="append", default=[], metavar="POLYGONS[:CMAP]",
        help="polygons whose land the DEM recolours under its hillshade "
        "(repeatable; later ones on top), as the iceland3 overview's fissure "
        "swarms and central volcanoes: e.g. vector/iceland3/fisswarms_fil.xy "
        "(a path, or one under the GIS directory); CMAP a colour or CPT name "
        "(default lightslategrey 119/136/153), or none for the outline alone",
    )  # fmt: skip
    g.add_argument(
        "--fissure-swarms", action="store_true",
        help="the FutureVolc fissure swarms and central volcanoes, slate-blue "
        "under the hillshade as on the iceland3 overview (--dem-zone of the "
        "GIS vector/futurevolc_*_closed.gmt, built by build_07_volcanic.sh)",
    )  # fmt: skip
    g.add_argument(
        "--steep", nargs="?", const="", default=None, metavar="DEG[:COLOUR]",
        help="tint the land steeper than DEG degrees under the hillshade "
        "(default 30, orange #f0a050), on top of any --dem-zone; the slope "
        "over the DEM averaged 6 m wide, patches under 1000 m² left out. "
        "Meant for 2 m ArcticDEM zooms (landslides), e.g. --steep 30:orange",
    )  # fmt: skip
    g.add_argument(
        "--layers", action="append", default=[], metavar="NAME[,NAME…]",
        help="named GIS layers or presets of the layer catalogue "
        "(map_layers.yaml; 'list' prints them, marking files not on this "
        "host), each with its own look and --gmt step; repeatable, in drawing "
        "order. -NAME takes a layer or preset out again: --layers "
        "volcanic,roads --layers=-swarms (the = is needed for a leading -)",
    )  # fmt: skip
    g.add_argument(
        "--inset", type=_region, default=None, metavar="W/E/S/N|AREA",
        help="a zoom inset of this box (e.g. svartsengi), its stations named "
        "there and not on the map; the box is outlined on the map",
    )  # fmt: skip
    g.add_argument(
        "--inset-corner", choices=("BL", "BR", "TL", "TR"), default=None,
        help="put the zoom inset in this corner inside the map, whatever it "
        "covers; a bottom corner sits on the legend of a track map "
        "(default: a free spot, else beside the map)",
    )  # fmt: skip
    g.add_argument(
        "--inset-width", type=float, default=None, metavar="CM",
        help="the zoom inset's width [cm] (default 6)",
    )  # fmt: skip
    g.add_argument(
        "--locator", dest="locator", action="store_const", const=True,
        default=None,
        help="a small Iceland map with the map's box in red (default: on, "
        "except on a full-Iceland map)",
    )  # fmt: skip
    g.add_argument(
        "--no-locator", dest="locator", action="store_const", const=False,
        help="no Iceland locator map",
    )  # fmt: skip
    g.add_argument(
        "--locator-corner", choices=("BL", "BR", "TL", "TR"), default=None,
        help="put the Iceland locator in this corner of the map, whatever it "
        "covers (default: the free spot nearest the bottom left)",
    )  # fmt: skip
    g.add_argument(
        "--scale-style", choices=SCALE_STYLES, default=None,
        help="scale bar: fancy (checkered, 'km' above; default) or plain "
        "(default on a full-Iceland overview)",
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
        "glaciers:pen=0.3p,blue, dem:t=20, calderas:pen=2p,red (repeatable; "
        "'--gmt list' prints every step, its PyGMT call and its options; "
        "every --layers layer is a step of its own name)",
    )  # fmt: skip
    g.add_argument(
        "--baselines", action="append", default=[], metavar="SPEC[,SPEC…]",
        help="draw baselines as lines between their stations, on any map "
        "kind (repeatable): pairs A-B, an area of station_areas.yaml (cdn "
        "slug: all its baselines) or AREA/GROUP (one stacks: group, e.g. "
        "reykjanes/yfirlit). With no station given a station map shows just "
        "their stations, as does any map without --region (--margin then "
        "defaults to 0.03)",
    )  # fmt: skip
    g.add_argument(
        "--baseline-pen", default=None, metavar="PEN",
        help="pen of the --baselines lines (default 1.5p,gray25)",
    )  # fmt: skip
    g.add_argument(
        "--strict-layers", action="store_true",
        help="refuse a --layers layer whose file is not on this host "
        "(default: draw the map without it, with a warning)",
    )  # fmt: skip
    g.add_argument(
        "--layers-file", default=None, metavar="FILE",
        help="the layer catalogue (layers, their look and GMT arguments, "
        "presets) over the built-in one (default: map_layers.yaml in the "
        "gpsconfig directory, when there)",
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
    g.add_argument(
        "--scale-bar-at", default="BR", type=_corner_code, metavar="CODE",
        help="where the distance scale bar sits: a GMT justification inside "
        "the map, BR (default), BL, BC, TL, TC, TR, ML, MC, MR",
    )  # fmt: skip
    g.add_argument(
        "--legend-at", default=None, type=_fractions, metavar="X,Y",
        help="the vector legend's lower-left corner as fractions of the map "
        "from its SW corner (default 0.03,0.09: bottom left), e.g. 0.4,0.7 "
        "to put it on open sea (velocity and deformation maps)",
    )  # fmt: skip
    out = p.add_argument_group("output")
    out.add_argument(
        "--save", nargs="?", default=None, const="pdf",
        help="save as these formats, comma separated (pdf, png, eps; default "
        "pdf); without --save the map opens in a viewer",
    )  # fmt: skip
    out.add_argument("--name", default=None, help="fixed output basename")
    out.add_argument(
        "--manifest", default=None, metavar="FILE",
        help="append the files written to FILE, one absolute path per line "
        "(gpsplot run)",
    )  # fmt: skip
    out.add_argument(
        "-d", "--figDir", default=None,
        help="figure directory (default: fig_dir in postprocess.cfg [PATHS], "
        f"currently {common.config_dir(config, 'fig_dir', None) or 'unset'})",
    )  # fmt: skip


def _add_track_options(p: argparse.ArgumentParser) -> None:
    """``map station --track``: daily positions since an event (track.py)."""
    from gps_plot import track

    g = p.add_argument_group(
        "tracks",
        "each station's daily horizontal positions since an event as dots, "
        "its trend before the event removed, coloured by days since the "
        "onset (newest red) and joined by a dotted path; a station that moved "
        "gets its rate since the onset (blue) and of the last days (red) as "
        "arrows with 1-sigma ellipses and in the legend",
    )
    g.add_argument(
        "--track", default=None, metavar="EVENT|DATE|Nd",
        help="the onset: an events.yaml event (its id, or a unique part of "
        "it), YYYYMMDD or Nd (N days before today: a daily map whose tracks "
        "move with it); turns the tracks on",
    )  # fmt: skip
    g.add_argument(
        "--track-events", default=None, metavar="FILE",
        help="events.yaml to look EVENT up in (default: the gpsconfig one)",
    )  # fmt: skip
    g.add_argument(
        "--track-pre", type=_window, default=None, metavar="START:END",
        help=f"the trend window before the event (default: {track.PRE_DAYS} "
        f"days up to {track.PRE_GAP_DAYS} days before the onset)",
    )  # fmt: skip
    g.add_argument(
        "--track-before", type=int, default=track.BEFORE_DAYS, metavar="DAYS",
        help=f"dots from DAYS before the onset (default {track.BEFORE_DAYS})",
    )  # fmt: skip
    g.add_argument(
        "--track-recent", type=int, default=track.RECENT_DAYS, metavar="DAYS",
        help=f"the recent rate's window (default {track.RECENT_DAYS} days)",
    )  # fmt: skip
    g.add_argument(
        "--track-status", action="store_true",
        help="also write <figure>.status.json: per station the rate over the "
        "last --track-status-days days and since the onset with σ, the newest "
        "day's movement and the newest day (track_status.py; aflogun's Vöktun "
        "tab shows it)",
    )  # fmt: skip
    g.add_argument(
        "--track-status-days", type=int, default=None, metavar="DAYS",
        help="the status file's rate window (default: --track-recent)",
    )  # fmt: skip
    g.add_argument(
        "--track-scale", type=float, default=None, metavar="M_PER_MM",
        help="metres on the map per mm of displacement (default: the "
        f"farthest track {track.TRACK_CM} cm long)",
    )  # fmt: skip
    g.add_argument(
        "--track-arrow-scale", type=float, default=None, metavar="CM",
        help="arrow length per 100 mm/yr [cm] (default: the fastest station "
        f"{track.ARROW_CM} cm)",
    )  # fmt: skip
    g.add_argument(
        "--no-track-arrows", dest="track_arrows", action="store_false",
        help="dots only: no rate arrows (the numbers stay in the legend)",
    )  # fmt: skip
    g.add_argument(
        "--track-series", default=None, metavar="STA[,STA...]",
        help="also draw these stations' tracks as time series (North, East) "
        "with the rate since the onset and the recent rate on the data, the "
        "legend's numbers: <figDir>/ts/<name>-<STA>.<fmt>",
    )  # fmt: skip
    g.add_argument(
        "--track-cmap", default=track.DEFAULT_CMAP, metavar="CPT",
        help=f"colours of the days (default {track.DEFAULT_CMAP}: blue old, red new)",
    )  # fmt: skip
    g.add_argument(
        "--track-common-mode", choices=("subtract", "off"), default="subtract",
        help="the daily network common mode (default subtract)",
    )  # fmt: skip
    g.add_argument(
        "--track-frame", default="itrf14", metavar="FRAME",
        help="the .pos series read (default itrf14)",
    )  # fmt: skip
    g.add_argument(
        "--track-ref", default=None, metavar="STA[,STA...]",
        help="every track relative to this station (STA - REF on common "
        "days, a baseline): motion they share, the network's or a region's, "
        "drops out; REF itself is not drawn as a track. Several stations: "
        "relative to their weighted mean (each one's trend and seasonal "
        "removed first; days at least half of them have), so one station's "
        "own wander moves the reference by its share only; they are drawn",
    )  # fmt: skip
    g.add_argument(
        "--track-trend", choices=("remove", "keep"), default="remove",
        help="the trend before the onset: remove it (default; each station's "
        "own line, so only a change at the onset shows) or keep it (only the "
        "seasonal removed, on the reference stations too: motion before the "
        "onset, creep, shows, its rate in the legend)",
    )  # fmt: skip
    g.add_argument(
        "--track-up", action=argparse.BooleanOptionalAction, default=True,
        help="the vertical too (default; --no-track-up: horizontal only): "
        "each station's vertical as an arrow from it (up drawn north, down "
        "south; the colour, units and scale of its horizontal arrow: a rate, "
        "or with few days its displacement; 1-σ in black), in the legend and "
        "as a "
        "third panel of the --track-series figures",
    )  # fmt: skip
    g.add_argument(
        "--track-path-days", type=int, default=track.PATH_MEDIAN_DAYS,
        metavar="N",
        help="the dotted path follows the running median of N days "
        f"(default {track.PATH_MEDIAN_DAYS}; 1 = the daily positions)",
    )  # fmt: skip
    g.add_argument(
        "--track-uncert", type=float, default=track.TRACK_UNCERT, metavar="MM",
        help="drop days with a horizontal formal sigma above MM (default "
        f"{track.TRACK_UNCERT:g}; the Up sigma does not count)",
    )  # fmt: skip


def track_onset(
    spec: str, events_file: str | None = None, today: dt.date | None = None
) -> tuple[dt.date, str]:
    """``--track``: a date, ``Nd`` (N days before ``today``, so a daily
    product's tracks move with it), or an events.yaml event's onset; and
    its label."""
    from geo_dataread.products.period import find_event, parse_date

    if re.fullmatch(r"\d+d", spec):
        day = (today or dt.date.today()) - dt.timedelta(days=int(spec[:-1]))
        return day, f"the last {spec[:-1]} days"
    try:
        day = parse_date(spec)
    except ValueError:
        pass  # an event name
    else:
        return day, f"{day:%Y-%m-%d}"
    from geo_dataread.events import read_events

    ev = find_event(spec, read_events(events_file))
    return ev.onset, ev.id


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


def _add_series_args(p: argparse.ArgumentParser, *, velocity: bool = False) -> None:
    if velocity:
        refs: Sequence[str] = VELOCITY_REFS
        ref_help = (
            "what the windowed velocities are relative to (the velocity "
            "product, geo_dataread.velocity), the plate always removed: "
            "plate-periodic (default) = the station's stored seasonal "
            "removed too; background (or detrend) = its stored secular + "
            "seasonal removed, the transient alone; plate = nothing more; "
            "pre-onset = plate-periodic less the station's own rate over "
            "--pre-window. A station without a stored record falls back to "
            "plate, drawn pale and named in the legend"
        )
    else:
        refs = REFS
        ref_help = (
            "series the vectors are measured on: detrend = plate "
            "velocity and the stored secular + seasonal model removed, so the "
            "vectors are relative to each station's background; plate = only "
            "the plate velocity removed. Default: detrend"
        )
    common.add_series_options(
        p,
        refs=refs,
        ref_help=ref_help,
        # declared (the velocity product's default): raw less the declared
        # blunder epochs, no detection (which flags unrest signal)
        views=["declared", "raw", "cleaned"] if velocity else ["raw", "cleaned"],
        types=["POS", "TOT"],
        sessions=["24h", "8h"],
        session_help="24h daily solutions (default) or 8h sessions (itrf2020)",
        view_default="declared" if velocity else "raw",
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
    p.add_argument(
        "--stations-frame", default=None, metavar="FRAME",
        help="the frame of the --stations-file coordinates: wgs84 (lon/lat), "
        "isn93, isn2004, isn2016, … (geofunc.coords); default lon/lat, or the "
        "frame a table's header names ('X (ISN93) [m]'). Such tables are read "
        "by geofunc.points: tab, ; or space separated, decimal commas",
    )  # fmt: skip
    p.add_argument(
        "--nearest", action="store_true",
        help="a thin arrow from each --stations-file point to the nearest "
        "station (stations.cfg, --tos …; the stations chosen, on the map or "
        "just off it), its distance along it; the "
        "list printed and, with --save, written beside the map as "
        "<name>-nearest.csv (or --nearest-csv)",
    )  # fmt: skip
    p.add_argument(
        "--nearest-from", default=None, metavar="FILE",
        help="draw the links of a nearest list instead of finding them: a CSV "
        "with name, lon, lat and nearest columns (what 'gpscoord nearest' "
        "writes, e.g. with --data-before/--join); a moved station A+B links to "
        "its last mark B",
    )  # fmt: skip
    p.add_argument(
        "--nearest-ranks", default="1", metavar="R[,R…]",
        help="which ranks of the --nearest-from list to draw (default 1; 1,2 = "
        "the nearest solid black and the second nearest dashed grey)",
    )  # fmt: skip
    p.add_argument(
        "--nearest-csv", default=None, metavar="FILE",
        help="write the --nearest list here instead (e.g. under ~/gps-data/data/), "
        "with or without --save",
    )  # fmt: skip
    _add_track_options(p)
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
        "--pre-window", type=_window, default=None, metavar="START:END",
        help="--ref pre-onset: the reference days before the onset, "
        "YYYYMMDD:YYYYMMDD both included (e.g. 20230806:20231024)",
    )  # fmt: skip
    p.add_argument(
        "--last", type=_last, action="append", default=None,
        metavar="DAYS|FROM:TO",
        help="a window of the last DAYS days, or from FROM to TO days before "
        "the end (182:91 = the 3 months before the last 3); repeatable, "
        "windows may overlap or follow each other",
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
        "--vector-ref", default=None, metavar="STA",
        help="every arrow relative to STA's rate in the same window (σ in "
        "quadrature), e.g. SEYD for the SEY? slope stations: what they "
        "share with their reference drops out; STA itself draws no arrow",
    )  # fmt: skip
    p.add_argument(
        "--short-snr", type=float, default=None, metavar="R",
        help=f"a window of at most {SHORT_WINDOW_DAYS} days (e.g. --last 7) "
        "draws an arrow only when |v| / |2σ| >= R (2 = clearly above the "
        "noise); its arrows are noise most weeks and show only when the "
        "ground moves",
    )  # fmt: skip
    p.add_argument(
        "--short-step", action="store_true",
        help=f"a window of at most {SHORT_WINDOW_DAYS} days is drawn as the "
        "station's step against the 14 days before it (median of the "
        "window's days minus median of those before, over the time between "
        "them, gps_analysis.recent_step) instead of a line through its few "
        "days: a slip of a week stands out of it, a line would not show it. "
        "With --vector-ref, on the daily baseline STA-REF",
    )  # fmt: skip
    p.add_argument(
        "--short-downslope", type=float, nargs="?", const=45.0, default=None,
        metavar="DEG",
        help="with --short-snr: the short window's arrow must also point "
        "within DEG (default 45) of the DEM downslope azimuth at the "
        "station (gpsplot direction's slope, 2 m/5 m DEM over 50 m)",
    )  # fmt: skip
    p.add_argument(
        "--short-coherent", type=int, default=1, metavar="N",
        help="with --short-snr: draw the short window's arrows only when at "
        "least N stations pass (default 1; 2 or 3 = several stations move "
        "together, one noisy station alone does not show)",
    )  # fmt: skip
    p.add_argument(
        "--total", action="store_true",
        help="each station's horizontal displacement since its first days "
        f"(median of the first {TOTAL_END_DAYS} days to the median of the "
        "last, up to --end; with --vector-ref on the daily baseline "
        "STA-REF): a purple line the way it moved, ending in a T bar, its "
        "length by the displacement, and the number by its name",
    )  # fmt: skip
    p.add_argument(
        "--min-epochs", type=int, default=None,
        help="fewest epochs in a window for a rate (default 5, the velocity "
        "product's gate)",
    )  # fmt: skip
    p.add_argument(
        "--min-fill", type=float, default=0.0, metavar="F",
        help="drop a window's rate when fewer than F of its days have a "
        "solution (e.g. 0.6: SAFH's 13 days of a year, THNA stopping in "
        "May); per window, so a year and a 3-month window share one map. "
        "Default 0: the product's gates only (they measure the span the "
        "data cover, not the days in it)",
    )  # fmt: skip
    p.add_argument(
        "--no-short-tier", dest="short_tier", action="store_false",
        help="leave out the velocity product's short-tier rates (a series "
        "short of the window's cover but with a strong signal, flagged "
        "short:n=N), e.g. a station that stopped half way through a year map",
    )  # fmt: skip
    p.add_argument(
        "--velocities", default=None, metavar="FILE",
        help="draw a velocity product set (gps-velocities estimate: .jsonl, "
        ".csv or .gmt of one set) instead of estimating here",
    )  # fmt: skip
    p.add_argument(
        "--velocities-out", default=None, metavar="DIR",
        help="also write the velocities drawn, as a product set per window "
        "(.gmt/.csv/.jsonl), so the file and the map are the same numbers",
    )  # fmt: skip
    p.add_argument("--scale", type=float, default=None, metavar="CM_PER_UNIT",
                   help="arrow length per mm/yr [cm] (default: automatic)")  # fmt: skip
    p.add_argument("--scale-ref", type=float, default=None, metavar="MM_YR",
                   help="length of the legend arrow (default: automatic)")  # fmt: skip
    _add_series_args(p, velocity=True)
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
    p.add_argument(
        "--vertical", action="store_true",
        help="the vertical displacement, drawn as an arrow pointing north for "
        "uplift (as map velocity --vertical); no --mogi",
    )  # fmt: skip
    p.add_argument(
        "--with-vertical", action="store_true",
        help="the vertical too, on the same map: an arrow pointing north for "
        f"uplift (south for subsidence) in {VERTICAL_COLOR}, at the "
        "horizontal arrows' scale; --values writes it in that colour",
    )  # fmt: skip
    p.add_argument(
        "--label-stations", default=None, metavar="STA[,STA...]",
        help="name only these stations (the rest keep their symbol and "
        "arrows), e.g. SENG,SKSH on a crowded map",
    )  # fmt: skip
    p.add_argument(
        "--values", default=None, metavar="STA[,STA...]|all",
        help="write these stations' displacement at their arrow tips (2.87 m; "
        "with --vertical signed, +1.32 m); name a few on a crowded map",
    )  # fmt: skip
    p.add_argument(
        "--late-start", action="store_true",
        help="a station installed after -s is measured from its first days "
        "instead of left out; marked STA* and the legend says so",
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


def resolve_baselines(
    specs: Sequence[str], areas: Callable[[], Mapping[str, Mapping[str, Any]]]
) -> list[str]:
    """The ``--baselines`` pairs, each once in the order written (a reversed
    repeat B-A is the same baseline). A spec is a pair ``A-B``, an area's cdn
    slug (all its baselines and stacks: pairs) or ``AREA/GROUP`` (one stacks:
    group); ``areas()`` gives the area table, read only when one is named.

    Raises:
        ValueError: an unknown area or group, or no pair at all.
    """
    from gps_plot.products import area_pairs

    pairs: list[str] = []
    for spec in (x.strip() for text in specs for x in text.split(",")):
        if not spec:
            continue
        if BASELINE_PAIR.fullmatch(spec.upper()):
            pairs.append(spec.upper())
            continue
        table = areas()
        slug, _, group = spec.partition("/")
        if slug not in table:
            raise ValueError(
                f"--baselines {spec}: neither a pair A-B nor an area "
                f"({', '.join(sorted(table))})"
            )
        stacks = table[slug].get("stacks") or {}
        if group and group not in stacks:
            raise ValueError(
                f"--baselines {spec}: no such group in {slug} "
                f"({', '.join(sorted(stacks)) or 'it has none'})"
            )
        pairs += (
            stacks[group]
            if group
            else area_pairs({"": table[slug].get("baselines") or [], **stacks})
        )
    out = area_pairs({"": pairs})
    if not out:
        raise ValueError("--baselines: no baseline given")
    return out


def _baselines(args: Any, parser: Any, maps: Any, coords: Any) -> None:
    """Resolve ``--baselines`` into ``args.baseline_overlay`` (:func:`_style`
    draws it). Stations left unnamed become the baselines' own: always on a
    station map, on the others when no --region selects them either."""
    specs = getattr(args, "baselines", None)
    if not specs:
        return
    from gps_plot.adhoc import load_areas_and_root

    try:
        pairs = resolve_baselines(specs, lambda: load_areas_and_root()[0])
        members = list(dict.fromkeys(m for p in pairs for m in p.split("-")))
        coordinates = [c for m in members for c in coords([m])]
    except (KeyError, ValueError) as exc:
        parser.error(f"--baselines: {exc}")
    args.baseline_overlay = maps.baseline_overlay(
        pairs, coordinates, args.baseline_pen or maps.DEFAULT_BASELINE_PEN
    )
    other = any(getattr(args, o, None) for o in ("tos", "tos_search", "stations_file"))
    if (
        hasattr(args, "Stations") and not args.Stations and not other
        and (args.kind in ("station", BASE) or args.region is None)
    ):  # fmt: skip
        args.Stations = members
        if args.region is None and args.margin is None:
            args.margin = BASELINE_MARGIN_DEG


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
        kept.append(c)
    if region is not None:
        from gps_plot import maps

        kept = maps.within(region, kept)
    if not kept:
        raise ValueError("no station to plot")
    return kept, dropped


def resolve_windows(
    windows: Sequence[tuple[dt.datetime, dt.datetime | None]] | None,
    last: Sequence[int | tuple[int, int]] | None,
    now: dt.datetime,
) -> list[tuple[dt.datetime, dt.datetime]]:
    """``--window`` and ``--last`` as concrete (start, end) pairs, in order."""
    out = [(s, e or now) for s, e in windows or ()]
    # --last 7: seven whole days, today included (the product's windows
    # are whole days, both ends in); --last 14:7 = the seven days before
    # those, so --last 182:91 and --last 91 follow each other with no gap
    for d in last or ():
        first, to = (d, 0) if isinstance(d, int) else d
        out.append((now - dt.timedelta(days=first - 1), now - dt.timedelta(days=to)))
    return out


def window_label(start: dt.datetime, end: dt.datetime) -> str:
    return f"{start:%Y/%m/%d} - {end:%Y/%m/%d}"


def at_label(day: dt.datetime) -> str:
    """The legend of a rate ON a date (a transient set's ``at``)."""
    return f"rate on {day:%Y/%m/%d}"


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
        view="raw" if args.view == "declared" else args.view,
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
    overview = _overview(args, region, points)
    if overview is not None:
        region = args.region = overview
        args.projection = OVERVIEW_PROJECTION
        if args.dem is None and _iceland_dem_built(config):
            args.dem = ICELAND_DEM
        if args.dem not in (None, "none") and args.bathymetry_cmap is None:
            args.bathymetry_cmap = OVERVIEW_SEA
    kw: dict[str, Any] = {"region": region, "title": args.title}
    for opt, key in (("margin", "margin"), ("projection", "projection"),
                     ("resolution", "resolution"), ("dem", "dem_grid"),
                     ("dem_cmap", "dem_cmap"),
                     ("bathymetry_cmap", "bathymetry_cmap"),
                     ("glacier_cmap", "glacier_cmap"),
                     ("dem_smooth", "dem_smooth"),
                     ("sea_smooth", "sea_smooth")):  # fmt: skip
        if getattr(args, opt) is not None:
            kw[key] = getattr(args, opt)
    if str(kw.get("dem_grid", "")).lower() == "none":
        kw.pop("dem_grid")
    fine_dem = str(kw.get("dem_grid", "")).lower() == ARCTIC_DEM
    if fine_dem:  # the Iceland DEM, with a 2 m area grid where one covers
        kw["dem_grid"] = ICELAND_DEM
    if str(kw.get("dem_grid", "")).lower() == ICELAND_DEM:
        from gps_plot import maps

        extent = region
        if extent is None and points:
            extent = _region_of(points, args.margin, maps)
        extent = extent or maps.ICELAND_DEM_LEVELS[-1][2]
        projection = kw.get("projection", maps.DEFAULT_PROJECTION)
        try:
            choice = maps.iceland_dem(gis_dir(config), extent, projection,
                                      prefer_fine=fine_dem)  # fmt: skip
        except FileNotFoundError as exc:
            # not on this host (it is not in git): GMT's own relief and coast
            kw["dem_grid"] = maps.earth_relief_fallback(extent, projection)
            print(f"--dem iceland: {exc}; drawing {kw['dem_grid']} instead",
                  file=sys.stderr)  # fmt: skip
        else:
            kw["dem_grid"] = choice.grid
            for key in ("coastline", "glaciers", "lakes", "glacier_contours"):
                if getattr(choice, key) is not None:
                    kw[key] = getattr(choice, key)
            if choice.fine:
                kw.update(maps.FINE_DEM_SHADING)
    if map_zones(args):
        kw["zones"] = tuple(map_zones(args))
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
    if getattr(args, "fissure_swarms", False):
        from gps_plot.maps import DEFAULT_ZONE_PEN, MapOverlay

        kw["overlays"] = tuple(
            MapOverlay(
                os.path.join(gis_dir(config), path), pen=DEFAULT_ZONE_PEN, fill=None
            )
            for path in FISSURE_SWARM_OUTLINES
        )
    if getattr(args, "layers", None):
        from gps_plot import gis_layers

        cat = getattr(args, "layer_catalogue", None) or gis_layers.BUILTIN
        kw["overlays"] = kw.get("overlays", ()) + gis_layers.overlays(
            gis_layers.expand(args.layers, cat), gis_dir(config), cat,
            strict=args.strict_layers,
        )  # fmt: skip
    if getattr(args, "legend_at", None) is not None:
        kw["legend_anchor"] = args.legend_at
    if getattr(args, "baseline_overlay", None) is not None:
        kw["overlays"] = (*kw.get("overlays", ()), args.baseline_overlay)
    if args.overlay:
        from gps_plot.maps import DEFAULT_OVERLAY_FILL, DEFAULT_OVERLAY_PEN, MapOverlay

        fill = args.overlay_fill or DEFAULT_OVERLAY_FILL
        kw["overlays"] = kw.get("overlays", ()) + tuple(
            MapOverlay(path, pen=args.overlay_pen or DEFAULT_OVERLAY_PEN,
                       fill=None if fill.lower() == "none" else fill)
            for path in args.overlay
        )  # fmt: skip
    return kw


def _zoom_dem(
    args: Any, config: Any, box: Sequence[float], width_cm: float
) -> dict[str, Any] | None:
    """The zoom inset's own DEM keywords under ``--dem iceland``: a 2 m area
    grid when one covers ``box`` and the inset's pixels are finer than the
    finest Iceland level (:func:`gps_plot.maps.iceland_dem`), else None:
    the map's grid, resampled and smoothed (:func:`gps_plot.insets.zoom_dem`)."""
    from gps_plot import insets, maps

    if str(args.dem or "").lower() != ICELAND_DEM:
        return None
    try:
        choice = maps.iceland_dem(
            gis_dir(config), box, insets.zoom_projection(width_cm)
        )
    except FileNotFoundError:
        return None
    kw: dict[str, Any] = {}
    if choice.fine:
        kw = {"dem_grid": choice.grid, **maps.FINE_DEM_SHADING}
    if any(isinstance(z, maps.SteepZone) for z in args.dem_zone or ()):
        kw["zones"] = tuple(args.dem_zone)  # the steep tint: the inset's only
    return kw or None


def _iceland_dem_built(config: Any) -> bool:
    """Whether the overview's ``--dem iceland`` level is on this host."""
    from gps_plot import maps

    name = maps.ICELAND_DEM_LEVELS[-1][0]
    return os.path.isfile(os.path.join(gis_dir(config), maps.GIS_DEM, name))


def _overview(args: argparse.Namespace, region: Any, points: Sequence[Any]) -> Any:
    """The overview ``CornerRegion`` for a full-Iceland map: no --region
    nor --projection given, an extent of at least :data:`FULL_ICELAND_SPAN`
    (``region``, else the box around ``points``), and every point inside
    the overview frame; else None."""
    if args.region is not None or getattr(args, "projection", None) is not None:
        return None
    from gps_plot import maps

    extent = region
    if extent is None and points:
        extent = _region_of(points, args.margin, maps)
    if extent is None:
        return None
    w, e, s, n = extent
    if e - w < FULL_ICELAND_SPAN[0] or n - s < FULL_ICELAND_SPAN[1]:
        return None
    frame = maps.corner_region(OVERVIEW_CORNERS, OVERVIEW_PROJECTION)
    if len(maps.within(frame, points)) < len(points):
        return None
    return frame


def resolve_palettes(args: argparse.Namespace, pygmt_mod: Any, user_dir: str) -> None:
    """Resolve the colour-map options in place, before the map is drawn.

    A name found as a CPT file (``user_dir``, gpsplot's own) becomes its
    path, so the map functions need not know ``user_dir``. A name that is
    no CPT anywhere must be a colour GMT knows: GMT would otherwise fail
    late, mid-map, with a traceback.

    Raises:
        ValueError: naming the closest known colour maps.
    """
    from gps_plot import maps, palettes

    for opt in CMAP_OPTIONS:
        name = getattr(args, opt, None)
        if name is None or (opt == "bathymetry_cmap" and name.lower() == palettes.NONE):
            continue
        setattr(args, opt, _resolve_cmap(name, "--" + opt.replace("_", "-"),
                                         pygmt_mod, user_dir))  # fmt: skip
    zones = []
    for spec in getattr(args, "dem_zone", None) or ():
        if isinstance(spec, str):
            path, cmap = zone_spec(spec, os.path.dirname(user_dir))
            if cmap.lower() != maps.ZONE_OUTLINE_ONLY:
                cmap = _resolve_cmap(cmap, "--dem-zone", pygmt_mod, user_dir)
            spec = maps.DemZone(path, cmap)
        zones.append(spec)
    if zones:
        args.dem_zone = zones


#: the track legend's foot above the bottom annotations [cm] (BGÓ
#: 2026-10-31: on them it crowded the labels)
TRACK_LEGEND_GAP_CM = 0.5


def map_zones(args: Any) -> list[Any]:
    """The ``--dem-zone`` zones of the map itself: with a zoom inset the
    ``--steep`` tint is the inset's only (BGÓ 2026-10-31: on the overview it
    paints whole ice caps)."""
    from gps_plot import maps

    zones = list(getattr(args, "dem_zone", None) or ())
    if getattr(args, "inset", None):
        zones = [z for z in zones if not isinstance(z, maps.SteepZone)]
    return zones


def steep_key(args: Any) -> tuple[str, str] | None:
    """The legend line of ``--steep`` (text, colour), or None."""
    from gps_plot import maps

    zone = next((z for z in getattr(args, "dem_zone", None) or ()
                 if isinstance(z, maps.SteepZone)), None)  # fmt: skip
    if zone is None:
        return None
    colour = (maps.STEEP_LEGEND_COLOUR if zone.cmap == maps.DEFAULT_STEEP_COLOUR
              else zone.cmap)  # fmt: skip
    where = " (inset)" if getattr(args, "inset", None) else ""
    return f"tinted: ground steeper than {zone.min_slope:g}°{where}", colour


def steep_zone(spec: str) -> Any:
    """The :class:`~gps_plot.maps.SteepZone` of a ``--steep DEG[:COLOUR]``
    value ('' = the defaults).

    Raises:
        ValueError: DEG is not a number in (0, 90).
    """
    from gps_plot import maps

    deg, _, colour = spec.partition(":")
    try:
        slope = float(deg) if deg else maps.DEFAULT_STEEP_DEG
    except ValueError:
        raise ValueError(f"--steep {spec!r}: DEG must be a number") from None
    if not 0 < slope < 90:
        raise ValueError(f"--steep {spec!r}: DEG must be between 0 and 90")
    return maps.SteepZone(min_slope=slope, cmap=colour or maps.DEFAULT_STEEP_COLOUR)


def _resolve_cmap(name: str, flag: str, pygmt_mod: Any, user_dir: str) -> str:
    """A colour-map name as the map functions take it: a CPT file's path,
    else the name (a GMT CPT, or a colour GMT must know)."""
    from gps_plot import palettes

    palette = palettes.resolve(name, user_dir=user_dir)
    if palette.kind == palettes.FILE:
        return palette.source
    if palette.kind != palettes.COLOUR:
        return name
    try:  # not a CPT anywhere: GMT must know it as a colour
        palettes.segment(pygmt_mod, palette, 0.0, 1.0)
    except Exception as exc:  # noqa: BLE001 - GMTCLibError, kept lazy
        close = palettes.suggest(name, user_dir=user_dir)
        hint = f"; did you mean {', '.join(close)}?" if close else ""
        raise ValueError(
            f"{flag} {name!r} is not a CPT (gpsplot's, yours in "
            f"{user_dir}, GMT's or a file) nor a GMT colour{hint} "
            f"({flag.split()[0]} list prints the names)"
        ) from exc
    return name


def zone_spec(spec: str, gis: str) -> tuple[str, str]:
    """``POLYGONS[:CMAP]`` -> (an existing polygon path, the CMAP name).
    A path that does not exist is looked up under the GIS directory
    ``gis``; a colon that is part of an existing path stays.

    Raises:
        ValueError: no such polygon file.
    """
    from gps_plot import maps

    path, sep, cmap = spec.rpartition(":")
    if not sep or not path or Path(spec).expanduser().exists():
        path, cmap = spec, ""
    for candidate in (Path(path).expanduser(), Path(gis) / path):
        if candidate.is_file():
            return str(candidate), cmap or maps.DEFAULT_ZONE_CMAP
    raise ValueError(f"--dem-zone {spec!r}: no polygon file {path} (nor in {gis})")


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
    """A round scale-bar length: the 1-2-5 number nearest (in ratio) a fifth
    of the map width [km], so a 23 km map gets 5 km, not 2."""
    w, e, south, north = region
    lat = math.radians((south + north) / 2.0)
    width = (e - w) * 111.32 * max(math.cos(lat), 0.01)
    target = max(width / 5.0, 0.1)
    low = _nice(target)
    # the next 1-2-5 step up: 1 -> 2, 2 -> 5, 5 -> 10
    high = low * (2.5 if round(low / 10 ** math.floor(math.log10(low))) == 2 else 2)
    return high if high / target < target / low else low


def _unlabel_inset(style: dict[str, Any], args: Any) -> dict[str, Any]:
    """``style`` for the map itself: with --inset, the overlay names inside
    the inset's box are left to the inset."""
    if args.inset is None or not style.get("overlays"):
        return style
    box = tuple(args.inset[:4])
    return {
        **style,
        "overlays": tuple(
            dataclasses.replace(ov, unlabelled_box=box) for ov in style["overlays"]
        ),
    }


def _in_inset(args: Any, points: Sequence[Any], maps: Any) -> list[Any]:
    """The points inside the --inset box (none without one)."""
    if args.inset is None:
        return []
    return list(maps.within(args.inset, points))


def legend_box_cm(
    fig: Any, style: dict[str, Any], lines: float, width_cm: float = LEGEND_WIDTH_CM
) -> tuple[float, ...]:
    """The vector legend's box (x0, x1, y0, y1) [cm], bottom left: ``lines``
    rows at the map functions' anchor and spacing (region fractions),
    ``width_cm`` wide (:func:`vector_legend_width_cm`)."""
    from gps_plot import insets, maps

    region = style.get("region") or tuple(fig.region)
    width, height = insets.size_cm(
        region, style.get("projection", maps.DEFAULT_PROJECTION)
    )
    anchor = style.get("legend_anchor", maps.DEFAULT_LEGEND_ANCHOR)
    spacing = style.get("legend_line_spacing", maps.DEFAULT_LEGEND_LINE_SPACING)
    fx, fy = maps._legend_anchor(anchor, region)
    top = (fy + lines * spacing) * height + 0.3
    if "legend_anchor" in style and anchor != maps.DEFAULT_LEGEND_ANCHOR:
        # --legend-at: the box where it was put, not the bottom-left corner
        x0, y0 = max(0.0, fx * width - 0.3), max(0.0, fy * height - 0.3)
        return (x0, fx * width + width_cm, y0, top)
    return (0.0, fx * width + width_cm, 0.0, top)


def portrait_legend(style: dict[str, Any], maps: Any) -> dict[str, Any]:
    """On a map taller than wide the vector legend's anchor and line
    spacing (region fractions of its height) follow its width instead, so
    the legend is as tall in cm as on a square map (a tall Seyðisfjörður
    map, SEYD north of the fjord); nothing on any other map."""
    region = style.get("region")
    projection = str(style.get("projection", maps.DEFAULT_PROJECTION))
    if "legend_anchor" in style:
        return {}  # --legend-at: the user's place stands
    if region is None or isinstance(region, maps.CornerRegion):
        return {}
    if not projection.upper().startswith("M"):
        return {}  # the shape below is Mercator's
    w, e, s, n = (float(v) for v in region[:4])

    def merc(lat: float) -> float:
        return math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))

    # width over height of a Mercator map, no GMT needed
    k = math.radians(e - w) / (merc(n) - merc(s))
    if k >= 1.0:
        return {}
    fx, fy = maps.DEFAULT_LEGEND_ANCHOR
    return {
        "legend_anchor": (fx, fy * k),
        "legend_line_spacing": maps.DEFAULT_LEGEND_LINE_SPACING * k,
    }


def vector_legend_width_cm(labels: Sequence[str], arrow_cm: float) -> float:
    """The width of a vector legend block [cm]: its longest label or the
    reference arrow, with the box's padding."""
    from gps_plot import maps

    texts = [maps.text_width_cm(t, maps.DEFAULT_LEGEND_FONT) for t in labels]
    return max([arrow_cm + 0.3, *texts]) + 2 * maps.LEGEND_BOX_PAD_CM + 0.2


def arrow_points(
    vectors: Sequence[Any], scale: float, region: Sequence[float], projection: str
) -> tuple[list[float], list[float]]:
    """Points along each vector's arrow, shaft and tip (lon, lat), so the
    insets keep off the arrows as well as the stations."""
    from gps_plot import insets

    width, height = insets.size_cm(region, projection)
    w, e, s, n = region[:4]
    lons: list[float] = []
    lats: list[float] = []
    for v in vectors:
        for f in (0.25, 0.5, 0.75, 1.0):
            lons.append(v.lon + f * v.east * scale / width * (e - w))
            lats.append(v.lat + f * v.north * scale / height * (n - s))
    return lons, lats


def _wants_locator(args: Any, region: Sequence[float]) -> bool:
    """--locator, else on unless the map is all Iceland."""
    if args.locator is not None:
        return bool(args.locator)
    w, e, s, n = region[:4]
    return e - w < FULL_ICELAND_SPAN[0] or n - s < FULL_ICELAND_SPAN[1]


def _insets(
    fig: Any,
    args: Any,
    config: Any,
    style: dict[str, Any],
    points: Sequence[Any],
    zoom_symbols: Any,
    legend_lines: float = 0,
    legend_width_cm: float = LEGEND_WIDTH_CM,
    arrows: float | None = None,
    locator_floor: float | None = None,
    reserve: Sequence[tuple[float, float, float, float]] = (),
    block: tuple[Any, Any] | None = None,
) -> None:
    """The zoom inset (--inset) and the Iceland locator, where they fit
    (:func:`gps_plot.insets.place`): clear of the logo, the scale bar,
    a vector legend of ``legend_lines`` rows ``legend_width_cm`` wide, the
    station symbols and, with ``arrows`` (the vectors' scale [cm per
    unit]; ``points`` are then the vectors), the arrows. ``locator_floor``
    [cm]: a bottom ``--locator-corner`` locator sits on it (a legend's top).
    ``reserve``: more boxes (x0, x1, y0, y1) [cm] to keep free and
    ``block``: more points (lons, lats) to keep off (a track map's legend,
    colour bar and dots)."""
    from gps_plot import insets, layout, maps

    region = style.get("region") or tuple(fig.region)
    zoom = args.inset is not None and zoom_symbols is not None
    locator = _wants_locator(args, region)
    if not (zoom or locator):
        return
    projection = style.get("projection", maps.DEFAULT_PROJECTION)
    sea = getattr(fig, "gpsplot_layout", None)
    if not isinstance(sea, layout.SeaLayout):
        sea = layout.build_box(region, projection)
    width, height = sea.width, sea.height
    sea.reserve((0.0, LOGO_BOX_CM[0], height - LOGO_BOX_CM[1], height))
    sea.reserve((width - SCALE_BOX_CM[0], width, 0.0, SCALE_BOX_CM[1]))
    on_legend = False
    if legend_lines:
        box = legend_box_cm(fig, style, legend_lines, legend_width_cm)
        sea.reserve(box)  # type: ignore[arg-type]
        corner = getattr(args, "locator_corner", None)
        if locator_floor is None and corner in ("BL", "LB"):
            # a bottom-left locator sits on the legend
            locator_floor = box[3] - layout.MARGIN_CM
            on_legend = True
    for b in reserve:
        sea.reserve(b)
    if block is not None:
        sea.block_points(list(block[0]), list(block[1]))
    if args.markers:
        sea.block_points([p.lon for p in points], [p.lat for p in points])
    if arrows is not None and not isinstance(region, maps.CornerRegion):
        sea.block_points(*arrow_points(points, arrows, region, projection))
    zoom_spot = None
    if zoom:
        box = args.inset
        zoom_width = getattr(args, "inset_width", None) or insets.ZOOM_WIDTH_CM
        size = insets.size_cm(box, insets.zoom_projection(zoom_width))
        sea.block_points(*_box_edge(box))
        zoom_corner = getattr(args, "inset_corner", None)
        spot: insets.Spot | None
        if zoom_corner is not None:
            spot = insets.corner_spot(size, (width, height), zoom_corner,
                                      floor=locator_floor)  # fmt: skip
            sea.reserve((spot.x, spot.x + spot.width, spot.y, spot.y + spot.height))
        else:
            spot = insets.place(sea, size, insets.ZOOM_CORNERS_PREF, outside=True,
                                map_size=(width, height))  # fmt: skip
        assert spot is not None  # outside=True always finds a spot
        zoom_spot = spot
        insets.draw_zoom_box(fig, box)
        insets.draw_zoom(fig, box, spot, style, zoom_symbols,
                         dem=_zoom_dem(args, config, box, spot.width))  # fmt: skip
    if locator:
        if args.markers:  # and their names
            sea.block_points([p.lon for p in points], [p.lat for p in points],
                             radius=LOCATOR_NAME_CM)  # fmt: skip
        names = maps.overlay_label_points(
            _unlabel_inset(style, args).get("overlays", ())
        )
        sea.block_points([a[0] for a in names], [a[1] for a in names],
                         radius=LOCATOR_NAME_CM)  # fmt: skip
        size = insets.locator_size()
        corner = getattr(args, "locator_corner", None)
        if corner is not None:
            # on a box map clear of the latitude annotations inside the frame
            # on a legend (a vector or track map's): at the left edge, as
            # far left and down as it goes (BGÓ 2026-11-01)
            on_legend = on_legend or (
                corner in ("BL", "LB") and locator_floor is not None
            )
            left = (None if isinstance(region, maps.CornerRegion) or on_legend
                    else layout.BOX_LEFT_LABELS_CM)  # fmt: skip
            spot = insets.corner_spot(
                size, (width, height), corner, floor=locator_floor, left=left
            )
        else:
            spot = insets.place(sea, size, insets.LOCATOR_CORNERS_PREF)
        if spot is None:  # beside the map: under a zoom inset there, else top
            spot = insets.beside(size, (width, height), below=zoom_spot)
        insets.draw_locator(fig, region, spot, gis_dir(config))


def _projection_width_cm(rj: str) -> float:
    """The width [cm] in a GMT ``-R… -J…`` string (an inset's: ``-JM6c``)."""
    import re

    got = re.search(r"-J[A-Za-z]+(?:\S*/)?([\d.]+)c", rj)
    if got is None:
        raise ValueError(f"no width in {rj!r}")
    return float(got.group(1))


def _box_edge(box: Sequence[float], n: int = 12) -> tuple[list[float], list[float]]:
    """Points along a lon/lat box's edge, so no inset covers it."""
    w, e, s, n_ = box
    t = [k / n for k in range(n + 1)]
    lons = [w + (e - w) * f for f in t] * 2 + [w] * (n + 1) + [e] * (n + 1)
    lats = [s] * (n + 1) + [n_] * (n + 1) + [s + (n_ - s) * f for f in t] * 2
    return lons, lats


def _finish(fig: Any, args: argparse.Namespace) -> None:
    """Scale bar and logo on top of a finished map view."""
    if args.scale_bar:
        km = scale_bar_km(fig.region)
        from gps_plot.maps import CornerRegion

        # above the longitude annotations, upright on an oblique map
        oblique = isinstance(args.region, CornerRegion)
        foot = SCALE_BAR_FOOT_OBLIQUE if oblique else SCALE_BAR_FOOT
        # bottom right: top centre sat on the map, the SW corner holds the
        # vector legend, the logo is top left, the status legend top right
        from pygmt.params import Position

        corner = getattr(args, "scale_bar_at", "BR")
        position = Position(corner, cstype="inside",
                            offset=("0.4c", foot if corner[0] == "B" else "0.4c"))  # fmt: skip
        style = args.scale_style or ("plain" if oblique else "fancy")
        fancy = style == "fancy"
        import pygmt

        with pygmt.config(**SCALE_BAR_FONTS):
            fig.scalebar(
                **gmt_overrides.layer_kw(
                    "scalebar", length=f"{km:g}k", label=True, unit=not fancy,
                    fancy=fancy, box="+gwhite@20", position=position,
                )
            )  # fmt: skip
    logo = args.logo if args.logo is not None else str(DEFAULT_LOGO)
    if logo.lower() != "none" and os.path.isfile(logo):
        position = f"jTL+o0.35c/0.15c+w{LOGO_WIDTH}"
        from gps_plot.maps import CornerRegion

        overview = isinstance(args.region, CornerRegion)
        if overview:  # bottom left, the annotations drawn over it
            position = f"jBL+o{OVERVIEW_LOGO_X}/0.15c+w{OVERVIEW_LOGO_WIDTH}"
        fig.image(
            **gmt_overrides.layer_kw(
                "logo",
                imagefile=logo,
                position=position,
                transparency=LOGO_TRANSPARENCY,
            )
        )
        frame = getattr(fig, "gpsplot_frame", None)
        if overview and frame:
            fig.basemap(**gmt_overrides.layer_kw("frame", frame=frame))


def _save(fig: Any, args: argparse.Namespace, config: Any, stem: str) -> list[str]:
    if not args.save:
        fig.show(method="external")
        return []
    fig_dir = args.figDir if args.figDir is not None else (
        common.config_dir(config, "fig_dir", "") or ""
    )  # fmt: skip
    if fig_dir:
        os.makedirs(fig_dir, exist_ok=True)
    base = os.path.join(fig_dir, args.name or stem)
    paths = []
    for fmt in args.save.split(","):
        path = f"{base}.{fmt.strip()}"
        fig.savefig(path)
        paths.append(path)
    if getattr(args, "manifest", None):
        with open(args.manifest, "a") as fh:
            fh.writelines(os.path.abspath(f) + "\n" for f in paths)
    return paths


def areas_slug(text: str) -> str:
    from gps_plot import areas

    return areas.slug(text)


def _stem(args: Any, kind: str, points: Sequence[Any], *parts: str) -> str:
    """The file stem of a map: ``map_{kind}_{area}_{selection}_{parts}``
    (:mod:`gps_plot.areas`); the area from the map's region, else the box
    around ``points``, else all Iceland."""
    from gps_plot import areas, maps
    from gps_plot.sources import station_files

    region = args.region
    if region is None:
        region = (
            _region_of(points, args.margin, maps) if points else DEFAULT_BASE_REGION
        )
    chosen = areas.selection(
        tos=bool(getattr(args, "tos", False)),
        tos_search=getattr(args, "tos_search", None),
        patterns=getattr(args, "Stations", None) or (),
        files=[
            station_files.split_spec(f)[1]
            for f in getattr(args, "stations_file", None) or ()
        ],
    )
    return areas.stem(kind, region, chosen, *parts)


def _report(dropped: Sequence[str]) -> None:
    for line in dropped:
        print(f"dropped {line}", file=sys.stderr)


def run(args: argparse.Namespace, parser: argparse.ArgumentParser, config: Any) -> int:
    import warnings

    import pygmt

    from gps_plot import maps

    try:
        # the layer catalogue first: its layers are --gmt steps too
        args.layer_catalogue = layer_catalogue(args, config)
        if any(spec.strip().lower() == gmt_overrides.LIST for spec in args.gmt):
            print(gmt_overrides.describe(pygmt))  # the catalogue's layers too
            return 0
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
                **(maps.DEFAULT_GMT_DEFAULTS | geo_format(args.region)),
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


#: a map narrower than this [deg of longitude] has its frame in decimal
#: minutes: whole minutes repeat on a 2 m ArcticDEM zoom (16°49'W twice)
DECIMAL_MINUTES_SPAN = 0.1


def geo_format(region: Any) -> dict[str, str]:
    """``FORMAT_GEO_MAP`` in decimal minutes for a ``--region`` narrower
    than :data:`DECIMAL_MINUTES_SPAN`, else nothing (the defaults')."""
    try:  # Corners: lon/lat of two corners
        east = region[2] if isinstance(region, Corners) else region[1]
        span = abs(float(east) - float(region[0]))
    except (TypeError, IndexError, ValueError):
        return {}
    return {"FORMAT_GEO_MAP": "ddd:mm.xF"} if span < DECIMAL_MINUTES_SPAN else {}


def layer_catalogue(args: Any, config: Any) -> Any:
    """``--layers-file``, or map_layers.yaml of the gpsconfig directory, or
    the built-in catalogue; its layers made ``--gmt`` steps.

    Raises:
        ValueError: a missing --layers-file, or an invalid catalogue.
    """
    from gps_plot import gis_layers

    path = getattr(args, "layers_file", None)
    if path and not os.path.isfile(path):
        raise ValueError(f"--layers-file {path}: no such file")
    if not path and config is not None and getattr(config, "config_path", None):
        path = os.path.join(config.config_path, gis_layers.CATALOGUE_FILE)
    cat = gis_layers.load_catalogue(path)
    gis_layers.register_steps(cat)
    return cat


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
    if args.layers:
        from gps_plot import gis_layers

        cat = getattr(args, "layer_catalogue", None) or gis_layers.BUILTIN
        if any(spec.strip().lower() == gis_layers.LIST for spec in args.layers):
            print(gis_layers.describe(cat, gis_dir(config)))
            return 0
        try:  # names checked here; a missing file is skipped (--strict-layers)
            gis_layers.overlays(
                gis_layers.expand(args.layers, cat), gis_dir(config), cat,
                strict=args.strict_layers, quiet=True,
            )  # fmt: skip
        except (ValueError, FileNotFoundError) as exc:
            parser.error(str(exc))
    if getattr(args, "fissure_swarms", False):
        missing = [z for z in (*FISSURE_SWARM_ZONES, *FISSURE_SWARM_OUTLINES)
                   if not os.path.isfile(os.path.join(gis_dir(config), z))]  # fmt: skip
        if missing:
            parser.error(
                f"--fissure-swarms: no {', '.join(missing)} in {gis_dir(config)}; "
                "run vector/build_07_volcanic.sh there"
            )
        # filled from the closed polygons, under any --dem-zone of one's own
        # (later zones on top); outlined from the GeoJSON as served, so no
        # closing edge shows (_style adds them as overlays)
        gis = gis_dir(config)
        args.dem_zone = [
            maps.DemZone(os.path.join(gis, z), maps.DEFAULT_ZONE_CMAP, pen=None)
            for z in FISSURE_SWARM_ZONES
        ] + list(args.dem_zone)
    if args.steep is not None:
        try:
            args.dem_zone = list(args.dem_zone) + [steep_zone(args.steep)]
        except ValueError as exc:
            parser.error(str(exc))
    try:
        import pygmt

        resolve_palettes(args, pygmt, cpt_dir(config))
    except ValueError as exc:
        parser.error(str(exc))

    if isinstance(args.region, Corners):
        try:
            args.region = maps.corner_region(
                args.region, args.projection or maps.DEFAULT_PROJECTION
            )
        except pygmt.exceptions.GMTCLibError as exc:
            parser.error(f"--region {'/'.join(map(str, args.region))}+r: {exc}")

    def coords(markers: Sequence[str]) -> Any:
        return maps.station_coordinates(markers, parser=config)

    _baselines(args, parser, maps, coords)
    if args.inset is not None and args.kind not in ("station", "velocity"):
        parser.error("--inset draws on station and velocity maps only")
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
        ref = (getattr(args, "vector_ref", None) or "").upper()
        if ref and ref not in {c.marker for c in stations}:
            # the reference's rates are needed even off the map (a zoom on
            # the slope without SEYD); it draws no arrow (relative_to)
            stations = [*stations, *coords([ref])]
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


def _file_stations(args: Any, parser: Any, maps: Any) -> list[Any]:
    from gps_plot.sources import station_files

    out = []
    for spec in args.stations_file:
        try:
            out += station_files.read_stations(
                *station_files.split_spec(spec), maps,
                frame=getattr(args, "stations_frame", None),
            )  # fmt: skip
        except (OSError, ValueError, KeyError) as exc:
            parser.error(f"--stations-file: {exc}")
    kept: list[Any] = maps.within(args.region, out)
    return kept


def _nearest_links(
    args: Any, parser: Any, files: Sequence[Any], stations: Sequence[Any], maps: Any
) -> list[tuple[Any, Any]]:
    """``--nearest``: each file point's link to the nearest of ``stations``
    (the map's selection without its region: a station just off the map
    still counts), with that :class:`geofunc.points.Neighbour`."""
    if getattr(args, "nearest_from", None):
        return _links_from(args, parser, stations, maps)
    if not getattr(args, "nearest", False):
        return []
    from geofunc.points import nearest

    cands = [(c.marker, c.lon, c.lat) for c in stations]
    if not files or not cands:
        parser.error("--nearest needs --stations-file points and stations to link to")
    by_marker = {m: (lon, lat) for m, lon, lat in cands}
    out = []
    for c in files:
        (nb,) = nearest((c.lon, c.lat), cands)
        lon, lat = by_marker[nb.id]
        link = maps.MapLink(c.lon, c.lat, lon, lat, f"{nb.distance / 1000:.1f} km")
        out.append((link, (c.marker, nb)))
    return out


def _links_from(
    args: Any, parser: Any, stations: Sequence[Any], maps: Any
) -> list[tuple[Any, Any]]:
    """``--nearest-from``: the links of a nearest list (``gpscoord nearest``),
    the ranks of ``--nearest-ranks`` (default 1); labels from its
    ``distance_km`` column (else measured again, geodesic)."""
    import csv

    from geofunc.points import nearest

    path = os.path.expanduser(args.nearest_from)
    try:
        with open(path, encoding="utf-8") as fh:
            rows = list(csv.DictReader(ln for ln in fh if not ln.startswith("#")))
    except OSError as exc:
        parser.error(f"--nearest-from: {exc}")
    by_marker = {c.marker: (c.lon, c.lat) for c in stations}
    try:
        ranks = {
            int(x)
            for x in str(getattr(args, "nearest_ranks", "1")).split(",")
            if x.strip()
        }
    except ValueError:
        parser.error("--nearest-ranks: give ranks as numbers, e.g. 1,2")
    out = []
    for r in rows:
        rank = int(r.get("rank") or 1)
        if rank not in ranks:
            continue
        try:
            name, lon, lat = r["name"], float(r["lon"]), float(r["lat"])
            target = r["nearest"].strip().upper()
        except (KeyError, ValueError):
            parser.error(
                f"--nearest-from {path}: needs name, lon, lat and nearest columns"
            )
        mark = target.split("+")[-1]  # a moved station: its last mark
        if mark not in by_marker:
            parser.error(f"--nearest-from: {mark} is not among the stations chosen")
        (nb,) = nearest((lon, lat), [(target, *by_marker[mark])])
        slon, slat = by_marker[mark]
        try:  # where the list places the station (gpscoord nearest --since: then)
            slon, slat = float(r["station_lon"]), float(r["station_lat"])
        except (KeyError, TypeError, ValueError):
            pass
        # the list's own distance when it gives one (gpscoord nearest --since
        # measures it to the station where it was at that epoch, not to its
        # approximate stations.cfg place)
        try:
            km = float(r["distance_km"])
        except (KeyError, TypeError, ValueError):
            km = nb.distance / 1000
        link = maps.MapLink(lon, lat, slon, slat, f"{km:.1f} km", rank)
        out.append((link, (name, nb)))
    return out


def _write_nearest(
    links: Sequence[tuple[Any, Any]], saved: Sequence[str], path: str | None = None
) -> None:
    """The ``--nearest`` list: printed, and written to ``path`` (default
    beside the saved map as ``<name>-nearest.csv``)."""
    import csv

    rows = [
        (m, f"{k.lon0:.6f}", f"{k.lat0:.6f}", nb.id, f"{nb.distance / 1000:.2f}",
         f"{nb.azimuth:.0f}")
        for k, (m, nb) in links
    ]  # fmt: skip
    head = ("point", "lon", "lat", "nearest", "distance_km", "azimuth_deg")
    for r in rows:
        print(f"{r[0]:<10} -> {r[3]:<5} {r[4]:>6} km  az {r[5]:>3}°")
    if path is None and saved:
        path = os.path.splitext(saved[0])[0] + "-nearest.csv"
    if path:
        path = os.path.expanduser(path)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(head)
            w.writerows(rows)
        print(path)


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


def _draw_tracks(
    fig: Any, args: Any, parser: Any, config: Any, style: dict[str, Any],
    stations: Sequence[Any], maps: Any,
) -> tuple[Any, list[tuple[str, Any]], Any, dict[str, Any] | None]:  # fmt: skip
    """``--track``: every station's track on ``fig``; what was drawn
    (:class:`gps_plot.track.Drawn`: the locator sits on the legend, the
    insets keep off it), the ``--track-series`` figures by station, the
    zoom inset's drawing (``draw(fig, width_cm)``: the tracks inside it)
    and, with ``--track-status``, the status file's content
    (:func:`gps_plot.track_status.status_record`)."""
    from gps_plot import track

    series = {s.strip().upper() for s in (args.track_series or "").split(",")
              if s.strip()}  # fmt: skip

    try:
        onset, label = track_onset(args.track, args.track_events)
    except (ValueError, OSError) as exc:
        parser.error(f"--track: {exc}")
    if args.track_pre is not None:
        start, end = args.track_pre
        pre = (start.date(), (end or dt.datetime.combine(onset, dt.time())).date())
    else:
        pre = track.pre_window(onset)
    pos_dir = common.config_dir(config, "pos_dir", None) if config else None
    # the series show more days before the onset than the dots: load those,
    # the map draws its own days (trimmed: same rates and zero)
    before = max(args.track_before, track.SERIES_BEFORE_DAYS if series else 0)
    common_mode = args.track_common_mode == "subtract"
    keep_trend = args.track_trend == "keep"
    refs = [r.strip().upper() for r in (args.track_ref or "").split(",") if r.strip()]
    ref_name = (
        None if not refs else refs[0] if len(refs) == 1
        else "the mean of " + ", ".join(refs)
    )  # fmt: skip
    reference = up_reference = None
    status_days = (
        (args.track_status_days or args.track_recent) if args.track_status else None
    )
    ref_info: dict[str, Any] = {}
    if refs:
        ref_info = {"asked": refs, "kept": [], "dropped": []}
        loaded, used = [], []
        for r in refs:
            try:
                loaded.append(track.load_series(
                    r, pre, frame=args.track_frame, pos_dir=pos_dir,
                    uncert=args.track_uncert, common_mode=common_mode, up=True,
                ))  # fmt: skip
                used.append(r)
            except (ValueError, OSError) as exc:
                if len(refs) == 1:
                    parser.error(f"--track-ref {r}: {exc}")
                # one of several: the mean of the others
                print(f"--track-ref: {r} left out: {exc}", file=sys.stderr)
        if not used:
            parser.error("--track-ref: none of the stations has data")
        if len(used) >= 3:
            try:
                kept, moved = track.screen_references(
                    {u: (t, y[:2]) for u, (t, y) in zip(used, loaded, strict=True)},
                    pre, onset,
                )  # fmt: skip
            except ValueError as exc:
                parser.error(f"--track-ref: {exc}")
            for name, speed in moved:
                print(f"--track-ref: {name} left out: it moves {speed:.0f} mm/yr "
                      "against the others since the onset", file=sys.stderr)  # fmt: skip
                ref_info["dropped"].append({"station": name, "speed": round(speed, 1)})
            loaded = [loaded[used.index(k)] for k in kept]
            used = kept
        ref_info["kept"] = list(used)
        if len(refs) > 1:
            ref_name = "the mean of " + ", ".join(used)
        horizontal = [(t, y[:2]) for t, y in loaded]
        try:
            reference = (
                horizontal[0]
                if len(horizontal) == 1 and not keep_trend
                else track.reference_mean(horizontal, pre, keep_trend=keep_trend)
            )
        except ValueError as exc:
            parser.error(f"--track-ref: {exc}")
        # the vertical's reference alike, from the days with an Up
        ups = [track.up_series(t, y) for t, y in loaded]
        try:
            up_reference = (
                ups[0] if len(ups) == 1 and not keep_trend
                else track.reference_mean(ups, pre, keep_trend=keep_trend)
            )  # fmt: skip
        except ValueError as exc:
            print(f"--track-ref: no vertical reference ({exc}): no vertical",
                  file=sys.stderr)  # fmt: skip
            up_reference = None
    tracks, dropped = [], []
    for c in stations:
        if len(refs) == 1 and c.marker == refs[0]:
            continue  # zero by construction
        try:
            tracks.append(track.load_track(
                c.marker, c.lon, c.lat, onset, pre, frame=args.track_frame,
                pos_dir=pos_dir, uncert=args.track_uncert, common_mode=common_mode,
                before_days=before, recent_days=args.track_recent,
                reference=reference, keep_trend=keep_trend,
                up_reference=up_reference, status_days=status_days,
            ))  # fmt: skip
        except (ValueError, OSError) as exc:
            dropped.append(f"{c.marker}: {exc}")
    _report(dropped)
    tracks = track.with_colours(tracks)  # SVIE blue, SVIN red, …
    print(f"tracks since {onset:%Y-%m-%d} ({label}), trend {pre[0]:%Y-%m-%d} - "
          f"{pre[1]:%Y-%m-%d}: {len(tracks)} stations", file=sys.stderr)  # fmt: skip
    region = style.get("region") or tuple(fig.region)
    projection = style.get("projection", maps.DEFAULT_PROJECTION)
    if args.locator_corner is None:
        args.locator_corner = "BL"  # on the legend
    from gps_plot.layout import BOTTOM_LABELS_CM, BOX_BOTTOM_LABELS_CM

    # a plain W/E/S/N map (Mercator) has a narrow band of longitude
    # annotations at the bottom: the legend sits just above it (BGÓ
    # 2026-10-29), the inset and the locator on the legend
    floor = (BOTTOM_LABELS_CM if isinstance(region, maps.CornerRegion)
             else BOX_BOTTOM_LABELS_CM) + TRACK_LEGEND_GAP_CM  # fmt: skip
    steep = steep_key(args)
    shown = [track.trimmed(tr, args.track_before) for tr in tracks]
    drawn = track.draw_tracks(
        fig, shown, onset=onset, region=region, projection=projection,
        recent_days=args.track_recent, m_per_mm=args.track_scale,
        arrows=args.track_arrows,
        arrow_scale=(args.track_arrow_scale / 100 if args.track_arrow_scale else None),
        cmap=args.track_cmap, legend_floor_cm=floor,
        path_days=args.track_path_days,
        note=f"motion relative to {ref_name}" if ref_name else None,
        up=args.track_up, keys=[steep] if steep else (),
    )  # fmt: skip
    span = track.colour_range(shown)  # the map's colour scale

    def zoom(f: Any, width_cm: float) -> None:
        track.draw_zoom_tracks(f, shown, args.inset, width_cm,
                               arrows=args.track_arrows,
                               path_days=args.track_path_days,
                               cpt=drawn.cpt, up=args.track_up)  # fmt: skip

    status = None
    if status_days is not None:
        from gps_plot import track_status

        status = track_status.status_record(
            tracks, onset=onset, event=label, window_days=status_days,
            frame=args.track_frame, common_mode=common_mode,
            keep_trend=keep_trend, reference=ref_info or None,
        )  # fmt: skip
    picked = [tr for tr in tracks if tr.marker in series]
    return drawn, [
        (tr.marker, track.series_figure(
            tr, onset, recent_days=args.track_recent, before_days=args.track_before,
            days_range=span, cmap=args.track_cmap, common_mode=common_mode,
            reference=ref_name, keep_trend=keep_trend))
        for tr in picked
    ], zoom, status  # fmt: skip


def _run_station(args: Any, parser: Any, config: Any, maps: Any, coords: Any) -> int:
    if args.kind == BASE:
        args.markers = False
    use_tos = args.tos or args.tos_search is not None
    files = _file_stations(args, parser, maps)
    stations: list[Any] = []
    pool: list[Any] = []  # --nearest searches these, on the map or off it
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
        if getattr(args, "nearest", False) or getattr(args, "nearest_from", None):
            try:
                pool, _ = select_stations(
                    specs or ["*"], names, coords, region=None, exclude=args.exclude
                )
            except ValueError as exc:
                parser.error(str(exc))
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
    style = _style(args, region, config=config, points=stations)
    zoomed = _in_inset(args, stations, maps)
    zoom_style = _unlabel_inset(style, args)
    avoid = maps.overlay_label_points(style.get("overlays", ()))
    fig = maps.station_map(
        stations, labels=labels, markers=args.markers,
        unlabelled={c.marker for c in zoomed}, **zoom_style,
    )  # fmt: skip

    links = _nearest_links(args, parser, files, pool, maps)
    maps.draw_ranked_links(fig, [k for k, _ in links],
                           projection=style.get("projection") or maps.DEFAULT_PROJECTION)  # fmt: skip

    def zoom_symbols(f: Any, rj: str) -> None:
        maps.draw_station_symbols(f, zoomed, markers=args.markers,
                                  labels=args.markers, status_legend=None,
                                  rj=rj, avoid=avoid)  # fmt: skip

    drawn = status = None
    series: list[tuple[str, Any]] = []
    if getattr(args, "track", None):  # the dots over the stations, as approved
        drawn, series, zoom_tracks, status = _draw_tracks(
            fig, args, parser, config, style, stations, maps
        )

        def zoom_draw(f: Any, rj: str) -> None:  # the stations, then tracks
            zoom_symbols(f, rj)
            zoom_tracks(f, _projection_width_cm(rj))

    elif getattr(args, "track_series", None):
        parser.error("--track-series needs --track")
    elif getattr(args, "track_status", False):
        parser.error("--track-status needs --track")
    _finish(fig, args)
    _insets(
        fig, args, config, style, stations,
        zoom_draw if drawn is not None else zoom_symbols,
        locator_floor=drawn.legend_top if drawn else None,
        reserve=drawn.boxes if drawn else (),
        block=(drawn.lons, drawn.lats) if drawn else None,
    )  # fmt: skip
    stem = _stem(
        args,
        "station" if stations and args.markers else BASE,
        stations,
        *(["track"] if drawn is not None else []),
    )
    saved = _save(fig, args, config, stem)
    for p in saved:
        print(p)
    if links:
        _write_nearest(links, saved, args.nearest_csv)
    if status is not None and saved:
        from gps_plot import track_status

        path = track_status.status_path(saved[0])
        track_status.write_status(path, status)
        print(path)
        if getattr(args, "manifest", None):  # published with the map
            with open(args.manifest, "a") as fh:
                fh.write(os.path.abspath(path) + "\n")
    for marker, sfig in series:
        for p in _save_series(sfig, args, config, f"{args.name or stem}-{marker}"):
            print(p)
    return 0


def _save_series(fig: Any, args: Any, config: Any, base: str) -> list[str]:
    """A ``--track-series`` figure (matplotlib) to ``<figDir>/ts/<base>``."""
    if not args.save:
        return []
    fig_dir = args.figDir if args.figDir is not None else (
        common.config_dir(config, "fig_dir", "") or ""
    )  # fmt: skip
    out = os.path.join(fig_dir, "ts")
    os.makedirs(out, exist_ok=True)
    paths = []
    for fmt in args.save.split(","):
        path = os.path.join(out, f"{base}.{fmt.strip()}")
        fig.savefig(path, dpi=150)
        paths.append(path)
    if getattr(args, "manifest", None):
        with open(args.manifest, "a") as fh:
            fh.writelines(os.path.abspath(f) + "\n" for f in paths)
    return paths


@dataclasses.dataclass(frozen=True)
class _Rate:
    """N/E/U values and their 1-σ of one station, as :func:`_vector` draws."""

    neu: Any
    sigma: Any


def _run_velocity(args: Any, parser: Any, config: Any, maps: Any, stations: Any) -> int:
    now = args.end or dt.datetime.now()
    windows = resolve_windows(args.window, args.last, now)
    comp = 2 if args.vertical else None
    vectors: list[Any] = []
    dropped: list[str] = []
    notes: list[str] = []
    groups: list[tuple[str | None, str]] = []
    by_marker = {c.marker: c for c in stations}
    if args.velocities:
        # a product set: its windows are the groups
        from geo_dataread import velocity as vprod

        try:
            records = vprod.read_velocity_records(args.velocities)
        except (OSError, ValueError) as exc:
            parser.error(f"--velocities: {exc}")
        # a transient set (gpsprod transient) is one group per date the rate
        # is FOR (``at``), not per fit window
        spans = sorted({(r.window, getattr(r, "at", None)) for r in records},
                       key=lambda k: (k[0], k[1] or 0.0))  # fmt: skip
        requested = {r.ref_requested for r in records}
        if len(requested) == 1 and args.ref is None:
            (args.ref,) = requested  # the title says what the set is
        elif args.ref is not None and requested != {
            "background" if args.ref == "detrend" else args.ref
        }:
            parser.error(
                f"--ref {args.ref}: the set {args.velocities} is ref "
                f"{', '.join(sorted(requested))}"
            )
        if getattr(args, "pre_window", None) is not None:
            parser.error(
                "--pre-window: a --velocities set carries its own reference window"
            )
        pres = {tuple(r.pre_onset["window"]) for r in records if r.pre_onset}
        if len(pres) == 1:
            import geo_dataread.gps_read as gpsr

            ((p0, p1),) = pres  # the title names the set's reference days
            args.pre_window = tuple(gpsr.toDateTime([p0 + 1e-6, p1 - 1e-6]))
        in_file = {r.station for r in records}
        dropped += [
            f"{m}: not in {args.velocities}" for m in sorted(set(by_marker) - in_file)
        ]
        if in_file - set(by_marker):
            notes.append(
                f"{len(in_file - set(by_marker))} stations of the set not selected"
            )
        for k, (span, at) in enumerate(spans):
            label = (
                window_label(_date_of(span[0]), _last_day(span[1]))
                if at is None
                else at_label(_date_of(at))
            )
            chosen = [
                r for r in records
                if (r.window, getattr(r, "at", None)) == (span, at)
                and r.station in by_marker
            ]  # fmt: skip
            _add_records(maps, chosen, by_marker, label,
                         WINDOW_COLORS[k % len(WINDOW_COLORS)], comp, groups,
                         vectors, dropped, args.min_fill)  # fmt: skip
        windows = [(_date_of(span[0]), _last_day(span[1])) for span, _at in spans]
    elif windows:
        # the velocity product's own estimator: the numbers of gps-velocities
        from geo_dataread import velocity as vprod

        settings = _product_settings(args, parser, config)
        for k, (start, end) in enumerate(windows):
            label = window_label(start, end)
            d0, d1 = start.date(), end.date()
            records = vprod.estimate_velocities(list(by_marker), d0, d1, settings)
            if args.velocities_out:
                paths = vprod.write_velocity_set(
                    records, args.velocities_out, vprod.set_name(settings, d0, d1)
                )
                print(paths["jsonl"])
            _add_records(maps, records, by_marker, label,
                         WINDOW_COLORS[k % len(WINDOW_COLORS)], comp, groups,
                         vectors, dropped, args.min_fill)  # fmt: skip
    else:
        # the stored records' secular rates (geo_dataread.products.secular_rate)
        from geo_dataread.products import secular_rate

        from gps_plot.sources import gps as gps_source

        now_req = _request(args, parser, config, now - dt.timedelta(days=1), now)
        groups.append(("secular", "blue"))
        for c in stations:
            try:
                sr = secular_rate(
                    c.marker, gps_source.station_record(c.marker, now_req)
                )
            except Exception as exc:
                dropped.append(f"{c.marker}: {exc}")
                continue
            vectors.append(_vector(maps, c, _Rate(sr.rate, sr.sigma), comp, "secular"))
    index = window_index(groups)
    short = {
        label for label, k in index.items()
        if k < len(windows) and window_days(*windows[k]) <= SHORT_WINDOW_DAYS
    }  # fmt: skip
    ref_sta = args.vector_ref.upper() if args.vector_ref else None
    if args.short_step:
        if not short:
            parser.error(f"--short-step: no window of at most {SHORT_WINDOW_DAYS} days")
        vectors = [v for v in vectors if v.method not in short]
        for label in sorted(short, key=lambda s: index[s]):
            if label.endswith((FALLBACK_SUFFIX, NO_SEASONAL_SUFFIX)):
                continue
            start, end = windows[index[label]]
            vectors += step_vectors(
                args, parser, config, maps, stations, ref_sta, start, end,
                label, comp, notes,
            )  # fmt: skip
    if ref_sta:
        vectors = relative_to(
            vectors, ref_sta, index, notes,
            skip=short if args.short_step else set(),
        )  # fmt: skip
    if args.short_snr is not None:
        if not short:
            parser.error(f"--short-snr: no window of at most {SHORT_WINDOW_DAYS} days")
        down: dict[str, float] = {}
        if args.short_downslope is not None:
            down = downslope_azimuths(
                [c for c in stations if c.marker in {v.marker for v in vectors}],
                config, notes,
            )  # fmt: skip
        vectors = short_filter(
            vectors, short, index, args.short_snr, down,
            args.short_downslope, args.short_coherent, notes,
        )  # fmt: skip
    kept = [v for v in vectors if snr(v) >= args.min_snr]
    notes += [f"{v.marker} {v.method}: |v|/|2σ| = {snr(v):.2f} < {args.min_snr:g}"
              for v in vectors if snr(v) < args.min_snr]  # fmt: skip
    _report(dropped + notes)
    # the map's stations without an arrow: a dot and a name; with
    # --short-snr a week none of them moved in is a map too (a series)
    moved = {v.marker for v in kept}
    quiet = [c for c in stations if c.marker not in moved]
    if not kept and (args.short_snr is None or not quiet):
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
    colors = dict(g for g in groups if g[0] is not None)
    region = style.get("region")
    if args.vertical and len(windows) > 1 and region is not None:
        # collinear up/down arrows would hide each other: side by side
        kept = side_by_side(
            kept, window_index(groups), region,
            style.get("projection", maps.DEFAULT_PROJECTION),
        )  # fmt: skip
    style.update(portrait_legend(style, maps))
    zoomed = _in_inset(args, kept, maps)
    zoom_style = _unlabel_inset(style, args)
    avoid = maps.overlay_label_points(style.get("overlays", ()))
    totals: dict[str, tuple[float, float]] = {}
    newest: float | None = None
    if args.total:
        said: list[str] = []
        totals, newest = total_displacements(
            args, parser, config, stations, ref_sta, now, said
        )
        for line in said:
            print(line, file=sys.stderr)
    # more legend lines: a week nothing moved in says which days it looked
    # at; --total says how new its newest day is
    days = (
        f"{min(s for s, _ in windows):%d.%m.}-{max(e for _, e in windows):%d.%m.%Y}"
        if windows
        else ""
    )
    nothing = (
        [] if kept else [(f"no arrow: no motion above the noise, {days}", "black")]
    )
    corner: tuple[str, str] | None = None
    if newest is not None:
        # as the track maps: green when the newest day is the map's day or
        # the one before (BGÓ 2026-11-01)
        from gps_plot import track

        day = dt.date.fromordinal(int(track.day_key(np.array([newest]))[0]))
        corner = track.newest_text(day, today=now.date())
    fig = maps.velocity_map(
        kept,
        scale=args.scale or scale,
        # no arrow drawn (a quiet week of a series): no legend arrow either
        scale_ref=(args.scale_ref or ref) if kept else None,
        method_colors=colors,
        ellipse_pen=ELLIPSE_PEN,
        labels=args.labels,
        markers=args.markers,
        unlabelled={v.marker for v in zoomed},
        scale_vertical=bool(args.vertical),
        quiet=quiet,
        totals=totals,
        notes=nothing,
        corner_note=corner,
        **zoom_style,
    )

    # the inset's own scale: the map's arrows would leave its box
    zoom_scale, zoom_ref = auto_scale(
        [math.hypot(v.east, v.north) for v in zoomed], target_cm=ZOOM_ARROW_CM
    )
    zoom_scale = min(zoom_scale, args.scale or scale)

    def zoom_symbols(f: Any, rj: str) -> None:
        maps.draw_velocity_vectors(
            f, zoomed, colors=colors, scale=zoom_scale,
            ellipse_pen=ELLIPSE_PEN, markers=args.markers, labels=args.labels,
            rj=rj, avoid=avoid,
        )  # fmt: skip
        if zoomed and zoom_scale < (args.scale or scale):
            maps.draw_scale_arrow(f, args.inset, zoom_ref, zoom_scale, "mm/yr")

    _finish(fig, args)
    used_scale = args.scale or scale
    _insets(
        fig, args, config, style, kept, zoom_symbols,
        # the lines drawn: a window's only when it has an arrow
        legend_lines=len({v.method for v in kept}) + bool(totals) + len(nothing)
        + bool(corner and not kept)
        + (1.8 if kept else 0.0),
        legend_width_cm=vector_legend_width_cm(
            [*(str(k) for k in colors), f"{args.scale_ref or ref:g} mm/yr",
             *(t for t, _ in nothing),
             *([f"{args.scale_ref or ref:g} mm/yr        {corner[0]}"] if corner else []),
             *(["100 mm moved since installed"]
               if totals else [])],
            (args.scale_ref or ref) * used_scale,
        ) + (maps.TOTAL_MAX_CM + maps.T_LINE_GAP_CM if totals else 0.0),
        arrows=used_scale,
    )  # fmt: skip
    # dict.fromkeys: a transient set's dates share one fit window
    span = "_".join(dict.fromkeys(f"{s:%Y%m%d}-{e:%Y%m%d}" for s, e in windows))
    span = span or "secular"
    stem = _stem(args, "velocity", stations, span,
                 "vertical" if args.vertical else "horizontal")  # fmt: skip
    for p in _save(fig, args, config, stem):
        print(p)
    return 0


def _product_settings(args: Any, parser: Any, config: Any) -> Any:
    """The velocity product's settings from the map's series options."""
    from geo_dataread.velocity import VelocitySettings

    if args.tType not in (None, "POS"):
        parser.error(
            "windowed velocities come from the velocity product, which reads "
            "the .pos series: drop --tType (pick the frame with --frame)"
        )
    from geo_dataread.pos import frame_name

    ref = "background" if args.ref == "detrend" else (args.ref or "plate-periodic")
    kw: dict[str, Any] = {
        # the full frame name: the settings (and fingerprint) of gpsprod
        # velocity and ts --rate for the same window (gpsprod step 6)
        "frame": frame_name(args.frame or common.DEFAULT_PLOT_FRAME),
        "session": args.session or "24h",
        "ref": ref,
        "view": args.view or "declared",
        "uncert": float(args.uncert),
        "pos_dir": args.Dir or common.config_dir(config, "pos_dir", None),
    }
    if args.min_epochs is not None:
        kw["min_obs"] = args.min_epochs
    if not getattr(args, "short_tier", True):
        kw["short_tier"] = False
    pre = getattr(args, "pre_window", None)
    if pre is not None:
        from geo_dataread.velocity import window_bounds

        if pre[1] is None:
            parser.error("--pre-window: give its END")
        kw["pre_window"] = window_bounds(pre[0].date(), pre[1].date())
    try:
        return VelocitySettings(**kw)
    except ValueError as exc:
        parser.error(str(exc))


#: a station without a stored record drawn on plate only: its arrows pale
FALLBACK_SUFFIX = " (plate only)"
#: ... and under ref pre-onset, where the plate drops out with the
#: reference: no stored seasonal removed (the record's seasonal:none)
NO_SEASONAL_SUFFIX = " (no seasonal)"
FALLBACK_TRANSPARENCY = 55


def _displacement_settings(args: Any, parser: Any, config: Any) -> Any:
    """The displacement product's settings from the map's series options."""
    from geo_dataread.products import DisplacementSettings, Source

    if args.tType not in (None, "POS"):
        parser.error(
            "deformation maps come from the displacement product, which reads "
            "the .pos series: drop --tType (pick the frame with --frame)"
        )
    refs = {None: "background", "detrend": "background", "itrf2008": "itrf"}
    try:
        return DisplacementSettings(
            source=Source(
                frame=args.frame or common.DEFAULT_PLOT_FRAME,
                session=args.session or "24h",
                view=args.view or "declared",
                ref=refs.get(args.ref, args.ref),
                uncert=float(args.uncert),
                pos_dir=args.Dir or common.config_dir(config, "pos_dir", None),
            )
        )
    except ValueError as exc:
        parser.error(str(exc))
        raise  # pragma: no cover


def _add_records(
    maps: Any,
    records: Sequence[Any],
    by_marker: dict[str, Any],
    label: str,
    color: str,
    comp: int | None,
    groups: list[tuple[str | None, str]],
    vectors: list[Any],
    dropped: list[str],
    min_fill: float = 0.0,
) -> None:
    """Velocity product records as one window's vectors: refused ones
    dropped with their reason, plate-only fallbacks as a pale group, and
    records with fewer than ``min_fill`` of the window's days
    (:func:`window_fill`) dropped."""
    pre_onset = any(r.ref == "pre-onset" for r in records)
    pale = label + (NO_SEASONAL_SUFFIX if pre_onset else FALLBACK_SUFFIX)
    groups.append((label, color))
    any_pale = False
    for r in records:
        c = by_marker.get(r.station)
        if c is None:
            continue
        if not r.accepted:
            dropped.append(f"{r.station} {label}: {r.refused}")
            continue
        fill = window_fill(r) if min_fill > 0 else 1.0
        if fill < min_fill:
            dropped.append(f"{r.station} {label}: fill={fill:.2f}<{min_fill:g}")
            continue
        part = "horizontal" if comp is None else "up"
        idx = [0, 1] if comp is None else [comp]
        if not all(math.isfinite(r.rate[i]) for i in idx):
            # a short-tier record whose other part stood
            why = r.parts_refused.get(part, "not estimated")
            dropped.append(f"{r.station} {label}: {part} {why}")
            continue
        fallback = "fallback:plate" in r.flags
        any_pale |= fallback
        vectors.append(_record_vector(maps, c, r, comp, pale if fallback else label))
    if any_pale:
        groups.append((pale, f"{color}@{FALLBACK_TRANSPARENCY}"))


#: the sideways gap between the windows' arrows on a vertical map [cm]
VERTICAL_ARROW_GAP_CM = 0.18


def window_index(groups: Sequence[tuple[str | None, str]]) -> dict[str, int]:
    """Each legend group's window, in order: a window's pale fallback
    group (:data:`FALLBACK_SUFFIX`, :data:`NO_SEASONAL_SUFFIX`) shares it."""
    out: dict[str, int] = {}
    k = -1
    for label, _ in groups:
        if label is None:
            continue
        if not label.endswith((FALLBACK_SUFFIX, NO_SEASONAL_SUFFIX)):
            k += 1
        out[label] = max(k, 0)
    return out


def side_by_side(
    vectors: Sequence[Any],
    index: dict[str, int],
    region: Sequence[float],
    projection: str,
    gap_cm: float = VERTICAL_ARROW_GAP_CM,
) -> list[Any]:
    """The vertical arrows of several windows moved apart east-west,
    ``gap_cm`` each, centred on the station (the station symbol stays)."""
    from gps_plot import insets
    from gps_plot.maps import CornerRegion

    if isinstance(region, CornerRegion):
        return list(vectors)
    n = max(index.values(), default=0) + 1
    width, _ = insets.size_cm(region, projection)
    deg = (region[1] - region[0]) / width
    return [
        dataclasses.replace(
            v, arrow_dlon=((index.get(v.method, 0) - (n - 1) / 2) * gap_cm * deg)
        )
        for v in vectors
    ]


def window_fill(r: Any) -> float:
    """The share of a velocity record's window days with a solution: its
    epochs (all components) over the window length in days (8 h sessions
    count a third of a day each)."""
    days = (r.window[1] - r.window[0]) * 365.25
    per_day = 3.0 if getattr(r, "session", "24h") == "8h" else 1.0
    return min(r.n_epochs) / per_day / days if days > 0 else 0.0


def _record_vector(maps: Any, c: Any, r: Any, comp: int | None, method: str) -> Any:
    """A velocity product record as a map vector (rate N/E/U, honest σ)."""
    if comp is None:
        e, n, se, sn = r.rate[1], r.rate[0], r.sigma[1], r.sigma[0]
        corr = r.corr_en if math.isfinite(r.corr_en) else 0.0
    else:  # vertical: up as a north-pointing arrow (the gpsvelo convention)
        e, n, se, sn, corr = 0.0, r.rate[comp], 0.0, r.sigma[comp], 0.0
    return maps.VelocityVector(
        marker=c.marker, lon=c.lon, lat=c.lat, east=float(e), north=float(n),
        sigma_east=float(se), sigma_north=float(sn), correlation_en=float(corr),
        method=method,
    )  # fmt: skip


def _date_of(yf: float) -> dt.datetime:
    """A fractional year as a datetime (the inverse of TimetoYearf)."""
    year = int(yf)
    start = dt.datetime(year, 1, 1)
    days = (dt.datetime(year + 1, 1, 1) - start).days
    # to the second: a midnight's float must not land at 23:59:59.99 the
    # day before (a window's start label one day early)
    return start + dt.timedelta(seconds=round((yf - year) * days * 86400))


def _last_day(end_yf: float) -> dt.datetime:
    """The last day of a product window: its end is the midnight after it
    (geo_dataread.velocity.window_bounds)."""
    end = _date_of(end_yf)
    if (
        abs(
            (
                end - end.replace(hour=0, minute=0, second=0, microsecond=0)
            ).total_seconds()
        )
        < 60
    ):
        return end.replace(hour=0, minute=0, second=0, microsecond=0) - dt.timedelta(
            days=1
        )
    return end


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


#: a window of at most this many days is "short": --short-snr filters it
SHORT_WINDOW_DAYS = 31


def window_days(start: dt.datetime, end: dt.datetime) -> int:
    """A window's length in whole days, both ends included."""
    return (end.date() - start.date()).days + 1


def relative_to(
    vectors: Sequence[Any],
    ref: str,
    index: dict[str, int],
    notes: list[str],
    skip: set[str] = frozenset(),  # type: ignore[assignment]
) -> list[Any]:
    """Each vector minus ``ref``'s vector of the same window (its legend
    group's :func:`window_index`), σ in quadrature (the correlation of the
    two rates is not known, so the ellipse is if anything too big). A
    window without a ``ref`` rate keeps none of its vectors; ``ref``'s own
    vectors (zero by definition) are dropped. Vectors of the groups in
    ``skip`` are already relative (:func:`step_vectors` on baselines)."""
    refs = {index.get(v.method, 0): v for v in vectors if v.marker == ref}
    out = []
    for v in vectors:
        if v.method in skip:
            out.append(v)
            continue
        if v.marker == ref:
            continue
        r = refs.get(index.get(v.method, 0))
        if r is None:
            notes.append(f"{v.marker} {v.method}: no {ref} rate in the window")
            continue
        out.append(
            dataclasses.replace(
                v,
                east=v.east - r.east,
                north=v.north - r.north,
                sigma_east=math.hypot(v.sigma_east, r.sigma_east),
                sigma_north=math.hypot(v.sigma_north, r.sigma_north),
                correlation_en=0.0,
            )
        )
    return out


#: --short-step: days of series read before a short window (the noise)
STEP_NOISE_DAYS = 365


def step_vectors(
    args: Any,
    parser: Any,
    config: Any,
    maps: Any,
    stations: Sequence[Any],
    ref: str | None,
    start: dt.datetime,
    end: dt.datetime,
    label: str,
    comp: int | None,
    notes: list[str],
) -> list[Any]:
    """Each station's step over a short window as a vector
    (:func:`gps_analysis.recent_step.recent_step`: the window's days against
    the 14 before it, as a rate in mm/yr), on the daily baseline STA-``ref``
    when ``ref`` is given (``ref`` itself draws none). The day-to-day noise
    comes from the year of series before the window's end."""
    from gps_analysis.recent_step import recent_step

    from gps_plot.sources import gps as gps_source

    day_end = dt.datetime.combine(end.date(), dt.time()) + dt.timedelta(days=1)
    req = _request(
        args, parser, config, day_end - dt.timedelta(days=STEP_NOISE_DAYS), day_end
    )
    opts = gps_source.BaselineOptions(sign="raw")
    out = []
    for c in stations:
        if c.marker == ref:
            continue
        spec = f"{c.marker}-{ref}" if ref else c.marker
        try:
            s = gps_source.fetch(spec, req, opts)
        except Exception as exc:
            notes.append(f"{spec} {label}: {exc}")
            continue
        r = recent_step(
            s.t, np.asarray(s.y)[:3], _yearf(day_end) - 1e-9,
            window_days=window_days(start, end),
        )  # fmt: skip
        if r is None:
            notes.append(f"{spec} {label}: too few days for a step")
            continue
        out.append(_vector(maps, c, _Rate(r.rate, r.rate_sigma), comp, label))
    return out


#: --total: the median of this many days at either end of the series
TOTAL_END_DAYS = 14
#: --total: the series starts no earlier than this
TOTAL_FIRST = dt.datetime(1995, 1, 1)


def total_motion(
    t: Any, y: Any, n: int = TOTAL_END_DAYS
) -> tuple[float, float, float, float] | None:
    """The horizontal displacement from a series' first ``n`` epochs to its
    last ``n``: ``(east, north, first epoch, last epoch)``, median(last) − median(first)
    of rows N, E of ``y`` (its units; epochs [yr]); None with fewer than
    ``2 n`` epochs (NaN epochs left out)."""
    tv = np.asarray(t, dtype=np.float64)
    a = np.asarray(y, dtype=np.float64)[:2]
    order = np.argsort(tv)
    tv, a = tv[order], a[:, order]
    ok = np.isfinite(tv) & np.isfinite(a).all(axis=0)
    tv, a = tv[ok], a[:, ok]
    if tv.size < 2 * n:
        return None
    d = np.median(a[:, -n:], axis=1) - np.median(a[:, :n], axis=1)
    return float(d[1]), float(d[0]), float(tv[0]), float(tv[-1])


def total_displacements(
    args: Any,
    parser: Any,
    config: Any,
    stations: Sequence[Any],
    ref: str | None,
    end: dt.datetime,
    notes: list[str],
) -> tuple[dict[str, tuple[float, float]], float | None]:
    """``--total``: each station's :func:`total_motion` (east, north [mm]),
    and the newest epoch of them all [yr] (None: no series), up to
    ``end``, on the daily baseline STA-``ref`` when ``ref`` is given
    (``ref`` itself has none)."""
    from gps_plot.sources import gps as gps_source

    day_end = dt.datetime.combine(end.date(), dt.time()) + dt.timedelta(days=1)
    req = _request(args, parser, config, TOTAL_FIRST, day_end)
    opts = gps_source.BaselineOptions(sign="raw")
    out = {}
    newest: float | None = None
    for c in stations:
        if c.marker == ref:
            continue
        spec = f"{c.marker}-{ref}" if ref else c.marker
        try:
            s = gps_source.fetch(spec, req, opts)
        except Exception as exc:
            notes.append(f"{spec} total: {exc}")
            continue
        r = total_motion(s.t, s.y)
        if r is None:
            notes.append(f"{spec} total: too few days")
            continue
        out[c.marker] = (r[0], r[1])
        newest = r[3] if newest is None else max(newest, r[3])
        notes.append(
            f"{spec} total: {math.hypot(r[0], r[1]):.0f} mm toward "
            f"{math.degrees(math.atan2(r[0], r[1])) % 360:.0f}° since {r[2]:.2f}"
        )
    return out, newest


def angle_between(a: float, b: float) -> float:
    """The smaller angle between two azimuths [deg], 0..180."""
    return abs((a - b + 180.0) % 360.0 - 180.0)


def short_filter(
    vectors: Sequence[Any],
    short: set[str],
    index: dict[str, int],
    min_snr: float,
    downslope: dict[str, float],
    tolerance: float | None,
    min_count: int,
    notes: list[str],
) -> list[Any]:
    """Keep a short window's vector (its legend group in ``short``) only
    when it stands out of the noise: ``|v|/|2σ| >= min_snr``; with a
    ``tolerance`` also pointing within it of the station's ``downslope``
    azimuth (a station without one is not tested); and with at least
    ``min_count`` stations of that window passing — otherwise the window
    draws no arrows. Longer windows pass unchanged."""
    passed: dict[int, list[Any]] = {}
    out = []
    for v in vectors:
        if v.method not in short:
            out.append(v)
            continue
        r = snr(v)
        if r < min_snr:
            notes.append(f"{v.marker} {v.method}: noise, |v|/|2σ| = {r:.2f}")
            continue
        if tolerance is not None and v.marker in downslope:
            az = math.degrees(math.atan2(v.east, v.north)) % 360.0
            off = angle_between(az, downslope[v.marker])
            if off > tolerance:
                notes.append(
                    f"{v.marker} {v.method}: {az:.0f}° is {off:.0f}° off "
                    f"downslope ({downslope[v.marker]:.0f}°)"
                )
                continue
        passed.setdefault(index.get(v.method, 0), []).append(v)
    for k, vs in passed.items():
        if len(vs) >= min_count:
            out += vs
        else:
            notes.append(
                f"window {k + 1}: {len(vs)} station(s) moved, fewer than "
                f"{min_count}: no arrows"
            )
    return out


def downslope_azimuths(
    stations: Sequence[Any], config: Any, notes: list[str]
) -> dict[str, float]:
    """The DEM downslope azimuth at each station (:func:`terrain.slope_at`);
    a station off the DEM is left out (and so not tested)."""
    from gps_plot import terrain

    out = {}
    for c in stations:
        try:
            out[c.marker] = terrain.slope_at(c.lon, c.lat, gis_dir(config)).azimuth
        except (FileNotFoundError, ValueError) as exc:
            notes.append(f"{c.marker}: no downslope ({exc})")
    return out


#: what a windowed velocity map's vectors are relative to, by --ref
VELOCITY_REF_TITLES = {
    "plate-periodic": "plate and seasonal removed",
    "background": "relative to background",
    "detrend": "relative to background",
    "plate": "plate removed",
    "pre-onset": "relative to the pre-onset rate",
}


def _velocity_title(args: Any, windowed: bool) -> str:
    what = "Vertical velocity" if args.vertical else "Horizontal velocity"
    if not windowed:
        return f"{what}, secular (plate removed)"
    ref = args.ref or ("plate-periodic" if not args.velocities else None)
    title = f"{what}, {VELOCITY_REF_TITLES[ref]}" if ref else what
    pre = getattr(args, "pre_window", None)
    if ref == "pre-onset" and pre is not None and pre[1] is not None:
        title += f" ({pre[0]:%Y-%m-%d} to {pre[1]:%Y-%m-%d})"
    return title


def _region_of(stations: Any, margin: float | None, maps: Any) -> Any:
    return maps.region_around(
        stations, margin=maps.DEFAULT_MARGIN_DEG if margin is None else margin
    )


def _run_deformation(
    args: Any, parser: Any, config: Any, maps: Any, stations: Any
) -> int:
    from geo_dataread.products import estimate_displacements
    from geo_dataread.products.velocity_core import window_bounds

    end = args.end or dt.datetime.now()
    if end <= args.start:
        parser.error(f"-e {end:%Y%m%d} is not after -s {args.start:%Y%m%d}")
    # the displacement product (gpsprod displacement, method ends): the level
    # of the 5 days at each end, honest σ (gpsprod step 6)
    settings = _displacement_settings(args, parser, config)
    if args.late_start:
        settings = dataclasses.replace(settings, late_start=True)
    by_marker = {c.marker: c for c in stations}
    rs = estimate_displacements(
        list(by_marker), window_bounds(args.start.date(), end.date()), settings
    )
    got: list[tuple[Any, Any]] = []
    dropped: list[str] = []
    late: list[str] = []
    early: list[str] = []
    # ended = its data stops ENDED_DAYS before the newest station's (a series
    # that lags today alike, the gamit-p01 finals, marks nobody)
    lags = {r.station: _end_lag(getattr(r, "flags", ())) for r in rs if r.accepted}
    newest = min(lags.values(), default=0.0)
    for r in rs:
        if r.accepted:
            coord = by_marker[r.station]
            flags = getattr(r, "flags", ())
            starts = [f for f in flags if f.startswith("start_at:")]
            if starts:  # --late-start: from its own first days
                late.append(f"{r.station} from {starts[0].split(':', 1)[1]}")
                coord = dataclasses.replace(coord, marker=f"{coord.marker}*")
            if lags[r.station] - newest > ENDED_DAYS:  # stopped: to its last data
                early.append(r.station)
                coord = dataclasses.replace(coord, marker=f"{coord.marker}†")
            got.append((coord, _Rate(np.asarray(r.displacement), np.asarray(r.sigma))))
        else:
            dropped.append(f"{r.station}: {r.refused}")
    _report(dropped)
    if not got:
        print("nothing plotted", file=sys.stderr)
        return 1
    coords = [c for c, _ in got]
    if args.vertical and args.mogi:
        parser.error("--vertical: a Mogi field is horizontal; drop --mogi")
    if args.vertical and args.with_vertical:
        parser.error("--with-vertical adds the vertical to a horizontal map; "
                     "drop --vertical")  # fmt: skip
    if args.vertical:  # uplift drawn north, as map velocity --vertical
        east = np.zeros(len(got))
        north = np.array([r.neu[2] for _, r in got])
    else:
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
    up = np.array([r.neu[2] for _, r in got])
    field_values: dict[str, dict[str, str]] = {}
    if args.with_vertical:  # its own colour, the horizontal arrows' scale
        name = "vertical (north = up)"
        models.append(maps.VectorField(name, np.zeros(len(got)), up,
                                       color=VERTICAL_COLOR))  # fmt: skip
        field_values[name] = (
            _value_labels(args.values, coords, np.zeros(len(got)), up, True, maps) or {}
        )
    sizes = np.hypot(east, north)
    if args.with_vertical:
        sizes = np.concatenate([sizes, np.abs(up)])
    scale, ref = auto_scale(sizes)
    style = _style(args, args.region, config=config, points=coords)
    notes = (["* from its first data"] if late else []) + (
        ["† to its last data"] if early else []
    )
    if args.title is None:
        what = "Vertical displacement" if args.vertical else "Displacement"
        style["title"] = f"{what} {window_label(args.start, end)}" + (
            f" ({', '.join(notes)})" if notes else ""
        )
    if late:
        print("late start: " + "; ".join(late), file=sys.stderr)
    if early:
        print("ended before -e: " + ", ".join(early), file=sys.stderr)
    fig = maps.deformation_vectors(
        coords, east, north, models,
        observed_label=window_label(args.start, end),
        value_labels=_value_labels(args.values, coords, east, north, args.vertical,
                                   maps),
        field_values=field_values,
        label_only=(
            {x.strip().upper() for x in args.label_stations.split(",") if x.strip()}
            if args.label_stations
            else None
        ),
        scale=args.scale or scale, scale_ref=args.scale_ref or ref,
        scale_vertical=args.vertical,
        labels=args.labels, markers=args.markers, **style,
    )  # fmt: skip
    _finish(fig, args)
    _insets(fig, args, config, style, coords, None, legend_lines=len(models) + 3)
    tag = ("mogi" if args.mogi else "vertical" if args.vertical
           else "with-vertical" if args.with_vertical else "")  # fmt: skip
    stem = _stem(args, "deformation", coords,
                 f"{args.start:%Y%m%d}-{end:%Y%m%d}", tag)  # fmt: skip
    for p in _save(fig, args, config, stem):
        print(p)
    return 0


#: a deformation station is drawn STA† when its data ends this many days
#: before the newest station's
ENDED_DAYS = 30


def _end_lag(flags: Sequence[str]) -> float:
    """Days a displacement's data ends before the period (``end_early:Nd``)."""
    for f in flags:
        if f.startswith("end_early:"):
            return float(f.split(":", 1)[1].rstrip("d"))
    return 0.0


#: colour of the vertical arrows on a horizontal deformation map
VERTICAL_COLOR = "30/90/200"


def _corner_code(value: str) -> str:
    """``--scale-bar-at``: a GMT two-letter justification (BR, TC, …)."""
    v = value.strip().upper()
    if len(v) != 2 or v[0] not in "BMT" or v[1] not in "LCR":
        raise argparse.ArgumentTypeError(
            f"{value!r}: a GMT justification such as BR, BL, TC, ML"
        )
    return v


def _fractions(value: str) -> tuple[float, float]:
    """``--legend-at X,Y``: two map fractions in [0, 1]."""
    try:
        x, y = (float(p) for p in value.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"{value!r}: X,Y fractions, e.g. 0.4,0.7"
        ) from None
    if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
        raise argparse.ArgumentTypeError(f"{value!r}: fractions between 0 and 1")
    return (x, y)


def _value_labels(
    spec: str | None,
    coords: Sequence[Any],
    east: Any,
    north: Any,
    vertical: bool,
    maps: Any,
) -> dict[str, str] | None:
    """``--values``: the text at each named station's arrow tip, by marker —
    its horizontal total, or with ``--vertical`` the signed Up (``+1.32 m``).
    Names match without the ``*`` (late start) and ``†`` (ended) marks."""  # fmt: skip
    if not spec:
        return None
    wanted = {s.strip().upper() for s in spec.split(",") if s.strip()}
    out: dict[str, str] = {}
    for c, e, n in zip(coords, east, north, strict=True):
        if "ALL" not in wanted and c.marker.rstrip("*†").upper() not in wanted:
            continue
        if vertical:
            out[c.marker] = ("+" if n >= 0 else "−") + maps.total_text(abs(float(n)))
        else:
            out[c.marker] = maps.total_text(math.hypot(float(e), float(n)))
    return out


def _run_geojson(args: Any, config: Any, maps: Any) -> int:
    import json

    with open(args.geojson) as f:
        vectors = maps.velocity_vectors_from_geojson(json.load(f))
    if args.Stations:
        keep = set(_patterns(args.Stations, [v.marker for v in vectors]))
        vectors = tuple(v for v in vectors if v.marker in keep)
    vectors = tuple(maps.within(args.region, vectors))
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
    stem = _stem(args, "velocity", vectors, areas_slug(Path(args.geojson).stem))
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
