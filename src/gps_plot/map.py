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

from gps_plot.cli import common

KINDS = ("station", "velocity", "deformation", "slip")
REFS = ("detrend", "plate")
#: one colour per window, in order (the Reykjanes maps: blue, then red)
WINDOW_COLORS = ("blue", "red", "darkgreen", "darkorange", "purple", "brown")
#: the 90th-percentile arrow is drawn this long [cm] by the automatic scale
TARGET_ARROW_CM = 2.0
#: the Veðurstofa name logo, top left (source checkout only: not packaged)
DEFAULT_LOGO = (
    Path(__file__).resolve().parents[2]
    / "logos"
    / "vi_logos"
    / "VI_bottom_gradient_office_web_pos_rgb.png"
)
DATE_FORMAT = "%Y%m%d"
#: the --dem background: one light grey, so the hillshade alone shows the
#: relief (the Reykjanes maps' lightgrey.cpt)
LIGHTGREY_CPT = "-11000 225/225/225 9000 235/235/235\n"
TITLE_FONT = "12p,Helvetica-Bold"
#: thin ellipses under coloured arrows (the Reykjanes maps)
ELLIPSE_PEN = "0.5p,gray30"

EXAMPLES = """\
examples:
  gpsplot map station "*" --region -22.8/-21.9/63.75/64.1 --save pdf
  gpsplot map velocity --region -22.8/-21.9/63.75/64.1 \\
      --window 20231027:20231109 --window 20231123:20231205 --save pdf
                        two windows, plate removed, one colour each
  gpsplot map velocity --region -22.8/-21.9/63.75/64.1 --last 7 --vertical --save png
  gpsplot map velocity --region -24.6/-13.2/63.2/66.7 --save pdf
                        stored secular rates, Iceland
  gpsplot map deformation --region -22.8/-21.9/63.75/64.1 -s 20231101 -e 20231115 \\
      --mogi -22.46,63.87,4.5,1.2e7 --save pdf
  gpsplot map slip models/reykjanes_slip.json --view plane --save pdf
  --dem @earth_relief_15s draws a hillshade from the GMT remote relief."""


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
        help="hillshade background from this grid (a file, or a GMT remote "
        "grid such as @earth_relief_15s)",
    )  # fmt: skip
    g.add_argument(
        "--dem-cmap", default=None, metavar="CPT",
        help="colour map of the --dem background (default geo)",
    )  # fmt: skip
    g.add_argument("--title", default=None, help="map title")
    g.add_argument(
        "--no-labels", dest="labels", action="store_false",
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

    p = sub.add_parser("station", help="station markers")
    _add_station_args(p)
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


def _style(args: argparse.Namespace, region: Any) -> dict[str, Any]:
    kw: dict[str, Any] = {"region": region, "title": args.title}
    for opt, key in (("margin", "margin"), ("projection", "projection"),
                     ("resolution", "resolution"), ("dem", "dem_grid"),
                     ("dem_cmap", "dem_cmap")):  # fmt: skip
        if getattr(args, opt) is not None:
            kw[key] = getattr(args, opt)
    if "dem_grid" in kw and "dem_cmap" not in kw:
        kw["dem_cmap"] = lightgrey_cpt()
    return kw


def lightgrey_cpt() -> str:
    """Path of the light-grey DEM colour map (written once to the cache)."""
    base = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    path = Path(base) / "gps_plot" / "lightgrey.cpt"
    if not path.is_file() or path.read_text() != LIGHTGREY_CPT:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(LIGHTGREY_CPT)
    return str(path)


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
        fig.scalebar(length=f"{km:g}k", position="TC", label=True, unit=True)
    logo = args.logo if args.logo is not None else str(DEFAULT_LOGO)
    if logo.lower() != "none" and os.path.isfile(logo):
        fig.image(imagefile=logo, position="jTL+o0.15c+w2.2c", box="+gwhite@20")


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
    import pygmt

    from gps_plot import maps

    with pygmt.config(FONT_TITLE=TITLE_FONT, MAP_TITLE_OFFSET="6p"):
        return _run(args, parser, config, maps)


def _run(args: argparse.Namespace, parser: Any, config: Any, maps: Any) -> int:
    def coords(markers: Sequence[str]) -> Any:
        return maps.station_coordinates(markers, parser=config)

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
    if args.kind == "station":
        fig = maps.station_map(
            stations, labels=args.labels, **_style(args, args.region)
        )
        _finish(fig, args)
        for p in _save(
            fig, args, config, _stem("station", stations[0].marker, len(stations))
        ):
            print(p)
        return 0
    if args.kind == "velocity":
        return _run_velocity(args, parser, config, maps, stations)
    return _run_deformation(args, parser, config, maps, stations)


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
    style = _style(args, args.region or _region_of(stations, args.margin, maps))
    if args.title is None:
        style["title"] = _velocity_title(args, bool(windows))
    fig = maps.velocity_map(
        kept,
        scale=args.scale or scale,
        scale_ref=args.scale_ref or ref,
        method_colors=dict(g for g in groups if g[0] is not None),
        ellipse_pen=ELLIPSE_PEN,
        labels=args.labels,
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
    style = _style(args, args.region)
    if args.title is None:
        style["title"] = f"Displacement {window_label(args.start, end)}"
    fig = maps.deformation_vectors(
        coords, east, north, models,
        observed_label=window_label(args.start, end),
        scale=args.scale or scale, scale_ref=args.scale_ref or ref,
        labels=args.labels, **style,
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
        labels=args.labels, **_style(args, args.region),
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
    style = _style(args, args.region)
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
