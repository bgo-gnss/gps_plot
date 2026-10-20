"""Options shared by the time-series subcommands (``ts``, ``stack``).

Each ``add_*`` function adds one argument group; each ``resolve_*`` turns
the parsed values into what the plot functions take. ``ts`` (via
``plot_gps_timeseries.build_parser``/``run``) and ``stack`` both use them,
so a flag means the same thing in either. The help texts are the ones
``plot-gps-timeseries`` always had; what differs per subcommand (``all``
fan-outs, ``--session both``) is passed in.
"""

from __future__ import annotations

import argparse
import datetime as dt
from collections.abc import Callable, Sequence
from typing import Any

#: the reference frame plotted when neither --tType nor --frame is given
DEFAULT_PLOT_FRAME = "itrf2014"

#: ``--ref`` value for "nothing removed": the positions in the frame that
#: ``--frame`` picks.  Internally (plotTime, geo_dataread) this is still the
#: historical token ``"itrf2008"``, which the CLI also accepts as an alias.
REF_FRAME = "ref_frame"
REF_FRAME_INTERNAL = "itrf2008"

#: frame names a user may type for --ref by mistake (they are --frame values)
ITRF_REF_NAMES = ("itrf2014", "itrf2020", "itrf08", "itrf14", "itrf20")

DATE_FORMAT = "%Y%m%d"


def ref_choice(value: str) -> str:
    """argparse type for ``--ref``: map the legacy alias, explain frame names."""
    v = value.strip().lower()
    if v == REF_FRAME_INTERNAL:
        return REF_FRAME
    if v in ITRF_REF_NAMES:
        raise argparse.ArgumentTypeError(
            f"{value!r} is a reference frame, not a --ref choice: use "
            f"--ref ref_frame --frame {value} (--frame picks the frame)"
        )
    return v


def config_dir(config: Any, option: str, default: str | None = None) -> Any:
    """Resolve a directory from postprocess.cfg, or `default` if unavailable.

    Directories (figDir, totPath) live in [PATHS], with a legacy [Configs]
    fallback — that is `getPostProcessDir`. `getPostProcessConfig` is a
    different method for config *files* in [FILES]; calling it here is what
    broke the console script.

    This must not raise: it runs while argparse is being built, so a missing key
    (totPath is commented out in the deployed config) or no deployed config at
    all — a container image before gpsconfig lands — would otherwise kill even
    `--help`. Callers that genuinely need the value fail at use time instead.
    """
    try:
        return config.getPostProcessDir(option)
    except Exception:
        return default


# ---------------------------------------------------------------------------
# options
# ---------------------------------------------------------------------------


def add_period_options(parser: Any, *, specials: Sequence[str]) -> None:
    """-s/-e/-D/--fix/--special: which time span is plotted."""
    parser.add_argument(
        "-s", "--start", type=str, default=None, help="start of the plot, YYYYMMDD"
    )
    parser.add_argument(
        "-e", "--end", type=str, default=None, help="end of the plot, YYYYMMDD"
    )
    parser.add_argument(
        "-D",
        type=int,
        default=0,
        help="Number of days to shift the given day positive subracts negativ adds",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="plot strictly the period specified, regardless of the data",
    )
    parser.add_argument(
        "--special",
        type=str,
        default=None,
        choices=list(specials),
        help="for routine plots: whole time series (all), One year long (year), ninety days (90d)",
    )


def add_output_options(parser: Any, config: Any) -> None:
    """--save/--name/-d/--ylim: where and how the figure is written."""
    parser.add_argument(
        "--save",
        type=str,
        nargs="?",
        default=None,
        const="eps",
        help="save figure to file(s): a format suffix or comma-separated "
        "formats, each written natively by matplotlib "
        "(e.g. 'png', 'pdf' or 'eps,pdf,png'); defaults to eps",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="fixed output basename (no directory, no suffix): the figure is "
        "written as <figDir>/<name>.<fmt>, bypassing the usual "
        "ref/view/period filename. Handy for re-rendering to one stable path "
        "you keep open in a viewer (e.g. --name RHOF-test). Every station and "
        "--ref/--special variant writes the SAME file, so only the last "
        "render survives a multi-variant run",
    )
    parser.add_argument(
        "-d",
        "--figDir",
        type=str,
        nargs="?",
        default=None,
        const=None,
        help="figure save directory (default: fig_dir in postprocess.cfg "
        "[PATHS], deployed from gps-config-data; currently "
        f"{config_dir(config, 'fig_dir', None) or 'unset -> current directory'}). "
        "Use -d . for the current directory.",
    )
    parser.add_argument(
        "--ylim",
        type=int,
        default=[],
        nargs="+",
        help="set limit for y-axis: list of one or two numbers in mm. "
        + "One number x will expand the y axis by x mm in each direciton. "
        + "Two numbers will give absolute lower and upper boundary of the y-axis",
    )


def add_series_options(
    parser: Any,
    *,
    refs: Sequence[str],
    ref_help: str,
    views: Sequence[str],
    types: Sequence[str],
    sessions: Sequence[str],
    session_help: str,
    view_default: str = "raw",
) -> None:
    """--ref/--view/--tType/--frame/--session/-u/-i: which series is read."""
    parser.add_argument(
        "--ref",
        type=ref_choice,
        default=None,
        choices=list(refs),
        help=ref_help,
    )
    parser.add_argument(
        "--view",
        type=str,
        default=view_default,
        # "detrended" is still accepted (old scripts) but no longer offered:
        # detrending is --ref detrend; --ref detrend --remove-steps all is
        # the same full-trajectory removal
        choices=list(views) + ["detrended"],
        metavar="{" + ",".join(views) + "}",
        help=f"data view: {' or '.join(views)} (default {view_default}; outlier epochs masked "
        "from the main series; the grey overlay is OFF by default -- add "
        "--show-outliers to see what was set aside). Detrending is --ref "
        "detrend (add --remove-steps all to remove the steps as well)",
    )
    parser.add_argument(
        "--tType",
        type=str,
        default=None,
        choices=list(types),
        help='the series to read. Default (since 2026-10-13): "POS" in the '
        f"{DEFAULT_PLOT_FRAME} frame — the PBO .pos files of the daily SINEX "
        "archive (N/E/U from X/Y/Z; okada's junction offsets applied to the "
        "itrf2008/itrf2014 sets); --frame picks "
        'another frame. "TOT": the GLOBK mb_STA_TOT.dat[123] files (the '
        'previous default), "08h" their 8-hour series, "JOIN" both',
    )
    parser.add_argument(
        "--frame",
        default=None,
        help="reference frame of the .pos series: itrf2008 (okada, 1995 on), "
        "itrf2014 (IGb14: okada re-aligned, 1995 on; the DEFAULT) or itrf2020 "
        "(IGb20: gamit-p01, 2017 on; the only frame with 8 h sessions). Short "
        "names itrf08/itrf14/itrf20 and igb14/igb20 also work. "
        "Implies --tType POS; plate removal and the detrend parameters follow "
        "the frame (detrend_params[.<frame>].json)",
    )
    parser.add_argument(
        "--session",
        default="24h",
        choices=list(sessions),
        help=session_help,
    )
    parser.add_argument(
        "-u",
        "--uncert",
        type=int,
        default=10,
        help="formal-uncertainty screen at read time [mm] — epochs with σ above "
        "this threshold are excluded from the series. Default 10 (the "
        "same screen as the detrend workbench; the batch estimator and "
        "geo_dataread.gps_read.getData default is 15)",
    )
    parser.add_argument(
        "-i",
        "--Dir",
        type=str,
        nargs="?",
        default=None,
        help="time series input directory (default: postprocess pos_dir for "
        "the .pos series, totPath for --tType TOT/08h/JOIN)",
    )


# ---------------------------------------------------------------------------
# resolution
# ---------------------------------------------------------------------------


def resolve_dates(
    args: argparse.Namespace, now: Callable[..., dt.datetime]
) -> tuple[dt.datetime | None, dt.datetime | None]:
    """``(start, end)`` from -s/-e/-D/--fix; ``now`` is ``currDatetime``."""
    start = end = None
    if args.fix or args.D:  # set the end date as yesterday
        end = now(-1)
    # defining sub-periods to plot, 'full'
    if args.start:  # start the plot at
        start = dt.datetime.strptime(args.start, DATE_FORMAT)
    if args.end:  # end the plot at
        end = dt.datetime.strptime(args.end, DATE_FORMAT)
    if args.D:
        start = now(days=-args.D, refday=end)
    return start, end


def resolve_source(
    kwargs: dict[str, Any],
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    config: Any,
) -> str:
    """--frame/--session/--pos-id -> pos_id (+ overlay_pos_id), the dirs.

    Pops ``frame`` and ``session`` from ``kwargs`` and returns the session.
    With neither --tType nor a product, the default is the .pos series in
    :data:`DEFAULT_PLOT_FRAME`; ``--session both`` also sets
    ``overlay_pos_id`` (the 8 h series of the same frame).
    """
    frame, session = kwargs.pop("frame"), kwargs.pop("session")
    if args.tType is None:
        kwargs["tType"] = args.tType = "POS"
    if args.tType == "POS" and frame is None and kwargs.get("pos_id") is None:
        # the 8 h sessions exist only in itrf2020 (gamit-p01) so far
        frame = DEFAULT_PLOT_FRAME if session == "24h" else "itrf2020"
    if frame is not None or session != "24h":
        from geo_dataread.pos import pos_product

        if kwargs.get("pos_id") is not None:
            parser.error("give --frame/--session or --pos-id, not both")
        try:
            kwargs["pos_id"] = pos_product(
                frame, "24h" if session == "both" else session
            )
            if session == "both":
                kwargs["overlay_pos_id"] = pos_product(frame, "8h")
        except ValueError as exc:
            parser.error(str(exc))
        kwargs["tType"] = args.tType = "POS"

    if kwargs["figDir"] is None:
        kwargs["figDir"] = config_dir(config, "fig_dir", "")

    if kwargs["Dir"] is None:
        kwargs["Dir"] = config_dir(
            config, "pos_dir" if args.tType == "POS" else "totPath"
        )
    return str(session)


#: the Icelandic network box (W, E, S, N) that ``all`` means by default: the
#: same box the detrend inventory uses, so global IGS sites stay out
ICELAND_REGION = (-25.5, -12.5, 62.5, 67.5)


def region_arg(value: str) -> tuple[float, float, float, float]:
    """``W/E/S/N`` in degrees (lon in [-180, 180)) -> a region tuple."""
    try:
        w, e, s, n = (float(v) for v in value.split("/"))
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"--region {value!r}: give W/E/S/N in degrees, e.g. -25/-13/63/67"
        ) from None
    if not (w < e and s < n):
        raise argparse.ArgumentTypeError(f"--region {value!r}: need W < E and S < N")
    return w, e, s, n


def add_selection_options(parser: Any) -> None:
    """``--region`` / ``--with-record``: which stations ``all`` (or a list) means."""
    parser.add_argument(
        "--region",
        type=region_arg,
        default=None,
        metavar="W/E/S/N",
        help="keep only stations inside this box (degrees; write it with = since "
        "west is negative: --region=-25/-13/63/67). "
        "With 'all' it replaces the default Iceland box %s; baselines "
        "(AAAA-BBBB) are kept when both ends are inside"
        % "/".join(f"{v:g}" for v in ICELAND_REGION),
    )
    parser.add_argument(
        "--with-record",
        action="store_true",
        help="keep only stations with a detrend record in the frame's detrend "
        "document (baselines: both ends)",
    )


def resolve_stations(
    stations: Sequence[str],
    kwargs: dict[str, Any],
    config: Any,
    *,
    region: tuple[float, float, float, float] | None = None,
    with_record: bool = False,
) -> list[str]:
    """The station list to plot, after ``all``, ``--region`` and ``--with-record``.

    ``all`` (POS series) = every station with a ``.pos`` file of the selected
    product inside ``region`` -- by default :data:`ICELAND_REGION`; for TOT
    series it stays the stations.cfg list. Call after :func:`resolve_source`
    (it reads ``tType``, ``pos_id`` and ``Dir`` from ``kwargs``).
    """
    from geo_dataread.baseline import split_baseline

    pos = kwargs.get("tType") == "POS"
    coords: dict[str, tuple[float, float]] = {}
    if pos:
        from geo_dataread.pos import DEFAULT_PROD_ID, pos_stations

        listed = pos_stations(
            kwargs.get("pos_id") or DEFAULT_PROD_ID, kwargs.get("Dir")
        )
        coords = {sta: (lon, lat) for sta, lon, lat in listed}

    if "all" in stations:
        if pos:
            box = region if region is not None else ICELAND_REGION
            picked = [
                sta for sta, (lon, lat) in coords.items() if _inside(lon, lat, box)
            ]
        else:
            picked = [str(s) for s in config.getStationInfo() if str(s) != "DEFAULTS"]
        rest = [s for s in stations if s != "all" and s not in picked]
        stations = picked + rest
        if not pos and region is not None:
            stations = _filter_region(stations, region, coords)
    elif region is not None:
        stations = _filter_region(stations, region, coords)

    if with_record:
        from geo_dataread import gps_views as gv

        from gps_plot import timesmatplt as tplt

        _frame, params = tplt.record_context(
            str(kwargs.get("tType")), kwargs.get("pos_id"), None
        )
        have = set(gv.read_detrend_params(params).get("stations") or {})
        stations = [
            s
            for s in stations
            if all(e.upper() in have for e in (split_baseline(s) or (s,)))
        ]
    return list(stations)


def _inside(lon: float, lat: float, box: tuple[float, float, float, float]) -> bool:
    w, e, s, n = box
    return w <= lon <= e and s <= lat <= n


def _station_lonlat(
    sta: str, coords: dict[str, tuple[float, float]]
) -> tuple[float, float] | None:
    """A station's position: its .pos header, else stations.cfg."""
    if sta.upper() in coords:
        return coords[sta.upper()]
    try:
        from geo_dataread.velocity import _coords

        lon, lat = _coords(sta.upper())
        return float(lon), float(lat)
    except Exception:  # noqa: BLE001 - an unknown station has no position
        return None


def _filter_region(
    stations: Sequence[str],
    box: tuple[float, float, float, float],
    coords: dict[str, tuple[float, float]],
) -> list[str]:
    from geo_dataread.baseline import split_baseline

    out = []
    for s in stations:
        ends = split_baseline(s) or (s,)
        pts = [_station_lonlat(e, coords) for e in ends]
        if all(p is not None and _inside(p[0], p[1], box) for p in pts):
            out.append(s)
    return out
