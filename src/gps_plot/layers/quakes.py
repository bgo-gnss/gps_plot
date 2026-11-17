"""Earthquake marks on time-series panels (``gpsplot ts --quake``).

A vertical line at the origin time on every panel and, on the top panel, a
label ``BB M5.2, dist 34 km`` — the magnitude and the epicentral distance
to the plotted point(s).  The layer only needs a :class:`~matplotlib.figure.
Figure` with date x-axes and the positions to measure from, so it works for
a single station, a baseline (one distance per end) or any other plot.

Prototype: quakes come from the command line (:func:`parse_quake`); reading
the earthquake catalog is a later step and would only replace the parser.
"""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Mapping, Sequence
from typing import Any, cast
from dataclasses import dataclass

from matplotlib.figure import Figure

#: mean Earth radius (km) for the great-circle distance
EARTH_RADIUS_KM = 6371.0
QUAKE_COLOR = "blue"
#: gap between the line and its label (points)
LABEL_GAP_PT = 4.0


@dataclass(frozen=True)
class Quake:
    """One earthquake: origin time (UTC), epicentre, magnitude, short name."""

    time: dt.datetime
    lat: float
    lon: float
    mag: float
    name: str = ""


def parse_quake(spec: str) -> Quake:
    """``TIME,LAT,LON,MAG[,NAME]`` -> :class:`Quake`.

    TIME is ISO (``2026-10-19`` or ``2026-10-19T14:32``); LAT/LON in decimal
    degrees (west negative).
    """
    parts = [p.strip() for p in spec.split(",")]
    if len(parts) not in (4, 5):
        raise ValueError(f"--quake expects TIME,LAT,LON,MAG[,NAME], got {spec!r}")
    try:
        time = dt.datetime.fromisoformat(parts[0])
        lat, lon, mag = (float(p) for p in parts[1:4])
    except ValueError as err:
        raise ValueError(f"--quake {spec!r}: {err}") from None
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise ValueError(f"--quake {spec!r}: latitude/longitude out of range")
    return Quake(time, lat, lon, mag, parts[4] if len(parts) == 5 else "")


def distance_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Great-circle (haversine) distance in km."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def quake_label(quake: Quake, points: Mapping[str, tuple[float, float]]) -> str:
    """``BB M5.2, dist 34 km``; points at different distances (a long
    baseline) get one each: ``dist VFLN 34, VFLS 52 km``.

    ``points`` maps a name to ``(lon, lat)``; an empty mapping drops the
    distance.
    """
    head = f"{quake.name} M{quake.mag:.1f}".strip()
    dists = [
        (n, distance_km(lon, lat, quake.lon, quake.lat))
        for n, (lon, lat) in points.items()
    ]
    if not dists:
        return head
    km = [round(d) for _, d in dists]
    if max(km) - min(km) <= 1:  # one station, or ends at the same distance
        return f"{head}, dist {sum(d for _, d in dists) / len(dists):.0f} km"
    return f"{head}, dist " + ", ".join(f"{n} {d:.0f}" for n, d in dists) + " km"


def add_quake_marks(
    fig: Figure,
    quakes: Sequence[Quake],
    points: Mapping[str, tuple[float, float]],
    color: str = QUAKE_COLOR,
) -> Figure:
    """A dashed line per quake on every axis, its label on the top axis.

    The label runs up the right side of the line from the bottom of the
    panel, so it stays clear of the title and of the step labels (which are
    centred).  Quakes outside the plotted time range are skipped: a clipped
    line vanishes but its label would not.
    """
    if not quakes or not fig.axes:
        return fig
    import matplotlib.dates as mdates
    from matplotlib.transforms import blended_transform_factory, offset_copy

    top = fig.axes[0]
    lo, hi = top.get_xlim()
    for q in quakes:
        x = float(mdates.date2num(q.time))  # type: ignore[no-untyped-call]
        if not lo <= x <= hi:
            continue
        for ax in fig.axes:
            # above the day grid, which would otherwise hide a line at 00:00
            ax.axvline(x, color=color, lw=1.5, ls="--", zorder=5)
        top.text(
            x,
            0.03,
            quake_label(q, points),
            transform=offset_copy(
                blended_transform_factory(top.transData, top.transAxes),
                fig=fig,
                x=LABEL_GAP_PT,
                units="points",
            ),
            rotation=90,
            ha="left",
            va="bottom",
            fontsize=9,
            color=color,
            zorder=6,
        )
    return fig


def station_points(sta: str) -> dict[str, tuple[float, float]]:
    """``{STA: (lon, lat)}`` for a station, or one entry per baseline end.

    Positions come from stations.cfg; an end without a position is left out
    (the label then shows the distances it can).
    """
    from geo_dataread.baseline import split_baseline
    from geo_dataread.velocity import _coords

    out: dict[str, tuple[float, float]] = {}
    for end in split_baseline(sta) or (sta,):
        try:
            out[end.upper()] = _coords(end.upper())
        except Exception:  # noqa: BLE001 - an unknown station has no position
            continue
    return out


# --------------------------------------------------------------------------
# Catalogue quakes and cumulative moment (gpsplot ts --quakes, PLAN-ts-quakes Q-T6)
# --------------------------------------------------------------------------

#: magnitude classes (PLAN-ts-quakes §1 P5): grey dots below MID, blue MID to
#: STAR, a red star from STAR, the value written from LABEL
MAG_MID = 4.0
MAG_STAR = 4.5
MAG_LABEL = 5.0
#: fill of an automatic solution: it has a location, not a class (no
#: magnitude is claimed for it, see :func:`quake_classes`). Automatic rows have
#: NO outline; a reviewed event is outlined, which is what marks it as reviewed
#: (BGÓ 2026-11-16).
AUTO_COLOR = "0.72"
#: reviewed outlines, matplotlib and GMT
REVIEWED_EDGE = "k"
REVIEWED_EDGE_W = 0.5
MAP_CIRCLE_PEN = "0.3p,black"
MAP_STAR_PEN = "0.4p,black"

SEAM_COLOR = "tab:red"
MOMENT_FILL = "0.85"
#: the outer right axis (moment) sits this far right of the panel, in axes
#: fractions; the panel's right edge moves in to make room for it
MOMENT_SPINE_OFFSET = 1.08
QUAKES_FIG_RIGHT = 0.84
#: top of the legend in axes fractions (below the logo)
LEGEND_TOP = 0.86
#: the --quake-* namespace keys a QuakeRequest carries (gpsquake.add_quake_args)
QUAKE_ARG_KEYS = (
    "quake_source",
    "quake_mode",
    "quake_min_q",
    "quake_no_duplicates",
    "quake_min_mag",
    "quake_max_mag",
    "quake_min_depth",
    "quake_max_depth",
    "quake_types",
    "quake_polygon",
    "quake_regions",
    "quake_no_cache",
)


@dataclass(frozen=True)
class QuakeRequest:
    """What ``--quakes`` asked for; picklable (it crosses the worker pool).

    ``region`` None = the station area's default region in
    ``quake_regions.yaml``. ``options`` holds the ``--quake-*`` values
    (:data:`QUAKE_ARG_KEYS`), read back with :func:`geo_dataread.gpsquake.
    quake_selection` / ``quake_kwargs`` (prefix ``quake-``).
    """

    region: str | None
    options: Mapping[str, object]
    star: float = MAG_STAR
    label: float = MAG_LABEL
    moment: bool = True


@dataclass(frozen=True)
class QuakeLayerInfo:
    """What the layer drew, for the console line and tests."""

    region: str
    n: int
    n_automatic: int
    moment_nm: float
    moment_automatic_nm: float
    seam: dt.datetime | None
    summary: str


def request_from_args(args: dict[str, Any]) -> QuakeRequest | None:
    """Pop ``--quakes`` (dest ``quake_catalogue``: ``quakes`` is the
    ``--quake`` marks list) and the ``--quake-*`` options out of ``args``
    (``vars(namespace)``); None when ``--quakes`` was not given."""
    region = args.pop("quake_catalogue", None)
    options = {k: args.pop(k, None) for k in QUAKE_ARG_KEYS}
    star = args.pop("quake_star", MAG_STAR)
    label = args.pop("quake_label", MAG_LABEL)
    no_moment = args.pop("no_moment", False)
    if region is None:
        return None
    return QuakeRequest(
        region=str(region) or None,
        options=options,
        star=float(star),
        label=float(label),
        moment=not no_moment,
    )


def _namespace(req: QuakeRequest) -> Any:
    import argparse

    return argparse.Namespace(**dict(req.options))


def area_region(sta: str, regions_file: Any = None) -> str:
    """The name of the default quake region of ``sta``'s station area.

    Raises :class:`ValueError` when no area of the station has one, or when
    its areas name different ones (then the region must be given).
    """
    from geo_dataread.events import read_station_areas
    from geo_dataread.seismic.selection import read_quake_regions, region_for_area

    regions = read_quake_regions(regions_file)
    areas = sorted(a for a, st in read_station_areas().items() if sta.upper() in st)
    found = {r.name: a for a in areas if (r := region_for_area(a, regions)) is not None}
    if not found:
        raise ValueError(
            f"{sta}: no default quake region for its areas ({', '.join(areas) or 'none'}); "
            "name one: --quakes REGION"
        )
    if len(found) > 1:
        raise ValueError(
            f"{sta}: its areas have different quake regions ({', '.join(sorted(found))}); "
            "name one: --quakes REGION"
        )
    return str(next(iter(found)))


def load_quakes(
    sta: str,
    req: QuakeRequest,
    start: dt.datetime,
    end: dt.datetime,
    *,
    readers: Mapping[str, object] | None = None,
) -> tuple[Any, Any, str]:
    """``(frame, report, region name)``: the catalogue of the request over
    ``[start, end)`` (naive times are UTC), with the moment estimate when
    ``req.moment``."""
    from geo_dataread import gpsquake as gq
    from geo_dataread.seismic.moment import moment_estimate
    from geo_dataread.seismic.selection import catalogue_for

    ns = _namespace(req)
    region = req.region
    if region is None and req.options.get("quake_polygon") is None:
        region = area_region(sta, req.options.get("quake_regions"))
    sel = gq.quake_selection(ns, region, prefix="quake-")
    kw = gq.quake_kwargs(ns, prefix="quake-")
    if readers is not None:
        kw["readers"] = readers
    df, report = catalogue_for(sel, start, end, **kw)
    if req.moment:
        df = moment_estimate(df)
    return df, report, sel.name or region or "selection"


def _tex(text: str) -> str:
    """Plain text made safe for the usetex figures."""
    for a, b in (("\\", r"\textbackslash{}"), ("_", r"\_"), ("%", r"\%"),
                 ("&", r"\&"), ("#", r"\#"), ("$", r"\$")):  # fmt: skip
        text = text.replace(a, b)
    return text


def _seam(report: object) -> dt.datetime | None:
    """The seam when the catalogue has a SIL part and a part after it."""
    parts = list(getattr(report, "parts", ()))
    sil = [p for p in parts if str(p.source).startswith("sil")]
    other = [p for p in parts if not str(p.source).startswith("sil")]
    if not sil or not other:
        return None
    t = min(p.start for p in other)
    return t.tz_convert("UTC").tz_localize(None).to_pydatetime()  # type: ignore[no-any-return]


def _numbers(column: Any) -> Any:
    """A column as a float array, NaN where missing.

    ``pd.to_numeric`` is typed as Scalar | Series | ndarray (and its
    ``to_numpy`` as absent on most of those), so the two steps are cast once
    here instead of at every call site.
    """
    import numpy as np
    import pandas as pd  # type: ignore[import-untyped]

    got = cast(Any, pd.to_numeric(column, errors="coerce"))
    return got.to_numpy(dtype=float, na_value=np.nan)


def quake_classes(
    df: object, *, star: float = MAG_STAR, label: float = MAG_LABEL
) -> dict[str, Any]:
    """The magnitude classes and the label mask, from REVIEWED rows only.

    An automatic solution carries ``aut.mag``, which is not a measured
    magnitude: the lane refuses to make a moment from it
    (:func:`geo_dataread.seismic.moment.moment_estimate`), and in the
    Bárðarbunga box every automatic row claiming M>=4.5 is an artefact at the
    fixed default depth 5.2 km (four of them on 2022-12-27 within 75 minutes
    at four scattered sites), while all 23 reviewed M>=4.5 sit in the caldera
    (BGÓ 2026-11-16). So an automatic row gets no class, no star and no label;
    it is drawn as one pale dot with no size and no outline.

    Returns ``reviewed``, ``automatic``, ``grey``, ``blue``, ``star`` and
    ``label`` as boolean arrays over ``df``'s rows.
    """
    import numpy as np
    import pandas as pd

    assert isinstance(df, pd.DataFrame)
    mag = (
        _numbers(df["magnitude"])
        if len(df)
        else np.zeros(0)
    )
    auto = (
        (df["evaluation_mode"].astype(object) == "automatic").to_numpy(dtype=bool)
        if len(df)
        else np.zeros(0, dtype=bool)
    )
    ok = np.isfinite(mag) & (mag >= 0)
    reviewed = ok & ~auto
    return {
        "reviewed": reviewed,
        "automatic": auto,
        "grey": reviewed & (mag < MAG_MID),
        "blue": reviewed & (mag >= MAG_MID) & (mag < star),
        "star": reviewed & (mag >= star),
        "label": reviewed & (mag >= label),
    }


def add_quake_panel(
    ax: object,
    df: object,
    report: object,
    region: str,
    *,
    star: float = MAG_STAR,
    label: float = MAG_LABEL,
    moment: bool = True,
    legend_below: bool = False,
) -> QuakeLayerInfo:
    """Quakes and their cumulative moment on twin axes of ``ax``.

    Magnitudes on a right axis (stem + dot per REVIEWED event; grey < 4, blue
    4–``star``, a red star from ``star``, the value written from ``label``).
    No quake has an outline; an automatic row is a pale dot with no class.
    The cumulative moment on an outer right axis:
    grey fill behind the data, a black step line in front, solid on SIL,
    dashed from the seam (moment estimated from SeisComP magnitudes, PLAN-
    ts-quakes Q4). A red dashed line marks the seam. ``ax`` keeps its
    data and limits; its background moves to the lowest twin.
    """
    import matplotlib.dates as mdates
    import numpy as np
    import pandas as pd
    from matplotlib.axes import Axes

    assert isinstance(ax, Axes) and isinstance(df, pd.DataFrame)
    fig = ax.figure
    lo, hi = ax.get_xlim()
    t = (
        cast(Any, df.index).tz_convert("UTC").tz_localize(None)
        if len(df)
        else pd.DatetimeIndex([])
    )
    x = np.zeros(0)
    if len(t):
        x = np.array(mdates.date2num(t.to_pydatetime()), dtype=np.float64)  # type: ignore[no-untyped-call]
    mag = _numbers(df["magnitude"]) if len(df) else np.zeros(0)
    auto = (
        (df["evaluation_mode"].astype(object) == "automatic").to_numpy(dtype=bool)
        if len(df)
        else np.zeros(0, dtype=bool)
    )
    seam = _seam(report)
    seam_x = float(mdates.date2num(seam)) if seam is not None else None  # type: ignore[no-untyped-call]

    base_z = ax.get_zorder()
    # the moment fill is the lowest layer and carries the panel background
    axf = ax.twinx()
    axf.set_zorder(base_z - 2)
    axm = ax.twinx()
    axm.set_zorder(base_z - 1)
    axm.patch.set_visible(False)
    ax.patch.set_visible(False)
    axf.spines["right"].set_position(("axes", MOMENT_SPINE_OFFSET))

    handles: list[object] = []
    ok = np.isfinite(mag) & (mag >= 0)
    cls = quake_classes(df, star=star, label=label)
    axm.vlines(x[ok], 0, mag[ok], color="0.8", lw=0.25, zorder=1)
    # no outline on a quake (BGÓ 2026-11-16): the fill carries the class
    edge = dict(mec=REVIEWED_EDGE, mew=REVIEWED_EDGE_W)
    classes: tuple[tuple[Any, dict[str, Any], str], ...] = (
        (cls["grey"], dict(marker="o", color="0.55", ms=2.0, alpha=0.7, **edge),
         rf"M $<$ {MAG_MID:g}"),
        (cls["blue"], dict(marker="o", color="tab:blue", ms=3.5, **edge),
         rf"{MAG_MID:g} $\leq$ M $<$ {star:g}"),
        (cls["star"], dict(marker="*", color="tab:red", ms=11, **edge),
         rf"M $\geq$ {star:g}"),
    )  # fmt: skip
    for sel, style, text in classes:
        sel = np.asarray(sel, dtype=bool)
        if not sel.any():
            continue
        (h,) = axm.plot(x[sel], mag[sel], ls="none", zorder=3, **style)
        h.set_label(text)
        handles.append(h)
    if cls["automatic"].any():
        # one size, the palest fill: an automatic row claims no magnitude
        (h,) = axm.plot(x[cls["automatic"]], mag[cls["automatic"]], ls="none",
                        marker="o", color=AUTO_COLOR, ms=2.0, zorder=2,
                        mec="none", label="automatic (no class)")  # fmt: skip
        handles.append(h)
    for xi, mi in zip(x[cls["label"]], mag[cls["label"]]):
        axm.annotate(f"{mi:.1f}", (xi, mi), xytext=(4, 4), textcoords="offset points",
                     fontsize=8, fontweight="bold", color="tab:red")  # fmt: skip
    top = float(np.nanmax(mag[ok])) if ok.any() else 0.0
    axm.set_ylim(0, max(5.5, top + 0.5))
    axm.set_ylabel("Magnitude")

    m0 = np.zeros(0)
    if moment and len(df) and "moment_nm" in df:
        m0 = _numbers(df["moment_nm"])
    total = float(np.nansum(m0)) if len(m0) else 0.0
    total_auto = float(np.nansum(m0[auto])) if len(m0) else 0.0
    if total > 0:
        exp = int(np.floor(np.log10(total)))
        cum = np.cumsum(np.nan_to_num(m0)) / 10.0**exp
        # the curve starts at zero at the left edge and runs to the right one
        xs = np.concatenate([[lo], x, [hi]])
        ys = np.concatenate([[0.0], cum, [cum[-1]]])
        axf.fill_between(xs, ys, step="post", color=MOMENT_FILL, lw=0)
        axf.set_ylim(0, cum[-1] * 1.05)
        axf.set_ylabel(rf"Seismic moment [$10^{{{exp}}}$ Nm]")
        axs = ax.twinx()  # the step line in front of the data
        axs.set_zorder(base_z + 1)
        axs.patch.set_visible(False)
        axs.set_ylim(axf.get_ylim())
        axs.axis("off")
        if seam_x is None or not lo < seam_x < hi:
            (h,) = axs.step(xs, ys, where="post", color="k", lw=1.2,
                            label="cumulative moment")  # fmt: skip
            handles.append(h)
        else:
            before = xs < seam_x
            y_seam = float(ys[before][-1])
            (h,) = axs.step(np.append(xs[before], seam_x), np.append(ys[before], y_seam),
                            where="post", color="k", lw=1.2,
                            label="cumulative moment")  # fmt: skip
            handles.append(h)
            (h,) = axs.step(np.insert(xs[~before], 0, seam_x),
                            np.insert(ys[~before], 0, y_seam), where="post",
                            color="k", lw=1.2, ls="--",
                            label=r"\ldots estimated from SeisComP magnitudes")  # fmt: skip
            handles.append(h)
    else:
        axf.set_yticks([])
        axf.spines["right"].set_visible(False)
    if seam_x is not None and lo <= seam_x <= hi:
        ax.axvline(seam_x, color=SEAM_COLOR, ls="--", lw=1.0, zorder=6)
        ax.text(seam_x, 0.02, " SeisComP from " + seam.strftime("%Y-%m-%d"),  # type: ignore[union-attr]
                transform=ax.get_xaxis_transform(), color=SEAM_COLOR, fontsize=8,
                rotation=90, va="bottom", ha="right", zorder=6)  # fmt: skip
    ax.set_xlim(lo, hi)
    n_auto = int(auto.sum())
    title = f"{_tex(region)}: n = {len(df)}"
    if n_auto:
        title += f", {n_auto} automatic"
    if total > 0 and total_auto >= 0.005 * total:
        title += f" ({100 * total_auto / total:.0f}\\% of the moment)"
    # below the logo, which sits in the top left corner of the top panel;
    # under the date axis in the one-panel figure (--one-panel), frameless:
    # above it is the title and the status line
    ax.legend(handles=handles, loc="upper left",
              bbox_to_anchor=(0.0, -0.13) if legend_below else (0.0, LEGEND_TOP),
              frameon=not legend_below, ncol=6 if legend_below else 3,
              fontsize="small",
              title=title, title_fontsize="small", framealpha=0.8,
              numpoints=1)  # fmt: skip
    if isinstance(fig, Figure) and fig.subplotpars.right > QUAKES_FIG_RIGHT:
        fig.subplots_adjust(right=QUAKES_FIG_RIGHT)
    summary = getattr(report, "summary", lambda: "")()
    return QuakeLayerInfo(
        region=region,
        n=len(df),
        n_automatic=n_auto,
        moment_nm=total,
        moment_automatic_nm=total_auto,
        seam=seam,
        summary=summary,
    )


# --------------------------------------------------------------------------
# The same catalogue on a GMT map (gpsplot map --quakes, PLAN-ts-quakes T19)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class MapQuakeInfo:
    """What a map's quake layer drew, for the console line and tests."""

    n: int
    n_automatic: int
    n_labelled: int


#: a map's own sizes [cm]: circle = base + scale x (M - min_mag); a star is
#: bigger, so a strong event still reads inside a dense cluster; an automatic
#: row is one small pale dot
MAP_CIRCLE_BASE_CM = 0.05
MAP_CIRCLE_SCALE_CM = 0.05
MAP_STAR_BASE_CM = 0.20
MAP_STAR_SCALE_CM = 0.05
MAP_STAR_MAX_CM = 0.42
MAP_AUTO_CM = 0.07
#: the automatic fill as GMT wants it: a bare "0.72" is not a grey name and
#: falls back to black (matplotlib's AUTO_COLOR is "0.72", GMT's is "gray72")
MAP_AUTO_FILL = "gray72"
#: a label row's height [cm], for the greedy placement
MAP_LABEL_ROW_CM = 0.22
MAP_LEGEND_CM = 0.24  # swatch
MAP_LEGEND_ROW_CM = 0.32
MAP_LEGEND_PAD_CM = 0.16
MAP_LEGEND_FONT = "7p,Helvetica"
MAP_LABEL_FONT = "6p,Helvetica-Bold,black"
#: the written values sit at the map's margin, tied to the star by a leader
MAP_LABEL_MARGIN_CM = 0.55
MAP_LEADER_PEN = "0.12p,gray30"
MAP_LEADER_STOP = 1.0
#: a white underprint, four passes, so a black number stays readable on the
#: grey swarm (GMT's text pen draws a box, not a glyph outline)
MAP_LABEL_HALO = "6p,Helvetica-Bold,white"
MAP_HALO_CM = 0.045


def _map_cm_grid(region: Any, size_cm: tuple[float, float]) -> tuple[Any, Any, Any]:
    """``(lon_of, lat_of, cm_of)`` for a region: cm -> lon/lat and back."""
    from gps_plot import maps

    width, height = size_cm

    def lon_of(cm: float) -> float:
        return float(maps.region_point(region, cm / width, 0.0)[0])

    def lat_of(cm: float) -> float:
        return float(maps.region_point(region, 0.0, cm / height)[1])

    def solve(f: Any, target: float) -> float:
        """Bisection: a Mercator y axis is not linear in latitude."""
        lo, hi = 0.0, 1.0
        rising = f(hi) >= f(lo)
        for _ in range(40):
            mid = (lo + hi) / 2
            if (f(mid) < target) == rising:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    def cm_of(lon: float, lat: float) -> tuple[float, float]:
        return (
            solve(lambda fx: maps.region_point(region, fx, 0.0)[0], lon) * width,
            solve(lambda fy: maps.region_point(region, 0.0, fy)[1], lat) * height,
        )

    return lon_of, lat_of, cm_of


#: the three periods, NEWEST FIRST, in the arrow colours the map already uses;
#: the last year red, since 2023 blue, older grey (BGÓ 2026-11-16)
PERIOD_FILLS = ("red", "blue", "gray50")
PERIOD_LABELS = ("síðustu 12 mán", "2023–2025", "eldri en 2023")
#: a detection below the map's class floor: dust, no rim. Two tiers, because one
#: size lost the M1-2.5 events inside the M<1 haze (BGÓ 2026-11-16)
DUST_CM = 0.018
DUST_CM_BIG = 0.032
DUST_SPLIT = 1.0
DUST_COLOR = "gray70"


def periods_of(df: object, now: dt.datetime | None = None) -> Any:
    """0 = the last 12 months, 1 = since 2023-01-01, 2 = older than 2023.

    The clock is the catalogue's newest event when ``now`` is not given, so a
    re-run of an old catalogue reproduces the same picture.
    """
    import numpy as np
    import pandas as pd

    assert isinstance(df, pd.DataFrame)
    if not len(df):
        return np.zeros(0, dtype=int)
    t = pd.DatetimeIndex(pd.to_datetime(df.index, utc=True))
    if now is None:
        now = t.max().to_pydatetime().replace(tzinfo=None)
    year_ago = pd.Timestamp(now, tz="UTC") - pd.Timedelta(days=365)
    since = pd.Timestamp("2023-01-01", tz="UTC")
    return np.where(t >= year_ago, 0, np.where(t >= since, 1, 2))


def add_quake_map_points(
    fig: Any,
    df: object,
    *,
    region: Any,
    min_mag: float,
    size_cm: tuple[float, float] | None = None,
    projection: str | None = None,
    star: float = MAG_STAR,
    label: float = MAG_LABEL,
    colour_by: str = "period",
    labels_max: int = 5,
    legend: bool = True,
    legend_corner: str = "TL",
    legend_clear_cm: float = 2.55,
    legend_at: tuple[float, float] | None = None,
    font: str = MAP_LEGEND_FONT,
) -> MapQuakeInfo:
    """The catalogue on a GMT map: dust, then the circles, then the stars.

    ``colour_by="period"`` (the default) colours every event by WHEN it
    happened, in the map's own three arrow colours, and leaves the SIZE to the
    magnitude: dust below ``min_mag``, a circle that grows with M, a star from
    ``star``. A reviewed event is outlined and an automatic one is not, but an
    automatic row is sized as dust whatever ``aut.mag`` claims (eight of them
    in this box claim M>=4.5 at the fixed default depth 5.2 km).
    ``colour_by="mag"`` is the older look: grey < 4, blue 4-``star``, red star.

    The value of a reviewed M >= ``label`` is written beside its star with a
    short leader. ``size_cm`` is the map size (pass it to skip the GMT call).
    ``legend_from`` is the legend's top-left in cm from the map's west and
    north edges, under the logo.
    """
    import numpy as np
    import pandas as pd

    from gps_plot import maps

    assert isinstance(df, pd.DataFrame)
    if size_cm is None:
        size_cm = maps._map_size_cm(region, projection or maps.DEFAULT_PROJECTION)
    lon_of, lat_of, cm_of = _map_cm_grid(region, size_cm)
    cls = quake_classes(df, star=star, label=label)
    lon = np.asarray(df["longitude"], dtype=float)
    lat = np.asarray(df["latitude"], dtype=float)
    mag = np.asarray(_numbers(df["magnitude"]), dtype=float)
    reviewed = np.asarray(cls["reviewed"], dtype=bool)
    automatic = np.asarray(cls["automatic"], dtype=bool)
    period = periods_of(df)

    def circle_cm(m: Any) -> Any:
        return np.clip(
            MAP_CIRCLE_BASE_CM + MAP_CIRCLE_SCALE_CM * (m - min_mag), 0.04, 0.45
        )

    def star_cm(m: Any) -> Any:
        return np.clip(
            MAP_STAR_BASE_CM + MAP_STAR_SCALE_CM * (m - star),
            MAP_STAR_BASE_CM,
            MAP_STAR_MAX_CM,
        )

    def draw(
        mask: Any,
        colour: str,
        style: str,
        pen: str | None,
        *,
        dust: float | None = None,
        big: bool = False,
    ) -> None:
        """One GMT call for a subset: the size is computed FROM the subset, or
        the arrays disagree on length (a masked x with an unmasked size)."""
        mask = np.asarray(mask, dtype=bool)
        if not mask.any():
            return
        m = mag[mask]
        size = dust if dust is not None else (star_cm(m) if big else circle_cm(m))
        kw: dict[str, Any] = {"style": style, "size": size, "fill": colour}
        if pen:
            kw["pen"] = pen
        fig.plot(x=lon[mask], y=lat[mask], **kw)

    if colour_by == "period":
        # oldest first, so the newest period ends up on top
        for k in (2, 1, 0):
            here = period == k
            fill = PERIOD_FILLS[k]
            small = mag < DUST_SPLIT
            draw(here & small, fill, "c", None, dust=DUST_CM)
            draw(here & ~small & (mag < min_mag), fill, "c", None,
                 dust=DUST_CM_BIG)  # fmt: skip
            draw(here & automatic, MAP_AUTO_FILL, "c", None, dust=DUST_CM)
            body = here & reviewed & (mag >= min_mag) & (mag < star)
            draw(body, fill, "c", MAP_CIRCLE_PEN)
            draw(here & reviewed & (mag >= star), fill, "a", MAP_STAR_PEN,
                 big=True)  # fmt: skip
    else:
        draw(mag < min_mag, DUST_COLOR, "c", None, dust=DUST_CM)
        draw(automatic, MAP_AUTO_FILL, "c", None, dust=DUST_CM)
        for sel, colour, big in (
            (cls["grey"], "gray55", False),
            (cls["blue"], "blue", False),
            (cls["star"], "red", True),
        ):  # fmt: skip
            sel = np.asarray(sel, dtype=bool) & reviewed
            pen = MAP_STAR_PEN if big else MAP_CIRCLE_PEN
            draw(sel, colour, "a" if big else "c", pen, big=big)  # fmt: skip

    # The N largest, written at the map's margin with their date and joined to
    # the star by a leader: inside the caldera the swarm leaves no room, and
    # five values out at the edge read better than twelve squeezed into it
    # (BGÓ 2026-11-16).
    placed: list[tuple[float, float, float, float]] = []
    width_cm, height_cm = size_cm
    times = pd.to_datetime(df.index, utc=True)
    all_stars = np.flatnonzero(np.asarray(cls["star"], dtype=bool))
    labels_max = int(labels_max)
    order: Any = (
        all_stars[np.argsort(-lat[all_stars])] if labels_max else np.zeros(0, int)
    )
    # biggest first decides WHICH; top-to-bottom decides the ORDER of the rows,
    # so the leaders do not cross
    keep = set(order[np.argsort(-mag[order])][:labels_max]) if len(order) else set()
    for i in [j for j in order if j in keep]:
        text = f"{mag[i]:.1f}   {times[i]:%d.%m.%Y}"
        w = maps.text_width_cm(text, MAP_LABEL_FONT)
        x0, y0 = cm_of(float(lon[i]), float(lat[i]))
        # out to the nearer margin: the swarm is in the middle, the edges free
        if x0 > width_cm / 2:
            left = width_cm - MAP_LABEL_MARGIN_CM - w
        else:
            left = MAP_LABEL_MARGIN_CM
        top = min(
            max(y0 + MAP_LABEL_ROW_CM / 2, MAP_LABEL_MARGIN_CM + 0.2),
            height_cm - MAP_LABEL_MARGIN_CM - 0.2,
        )
        for pbox in placed:  # step down clear of the label placed above it
            if abs(top - pbox[1]) < MAP_LABEL_ROW_CM and pbox[0] < left + w and pbox[2] > left:
                top = pbox[1] - MAP_LABEL_ROW_CM
        placed.append((left, top, left + w, top))
        # the leader runs from the star out to the start of the text
        tip = left - 0.06 if x0 > width_cm / 2 else left + w + 0.06
        fig.plot(
            x=[lon_of(x0), lon_of(tip)], y=[lat_of(y0), lat_of(top)],
            pen=MAP_LEADER_PEN,
        )  # fmt: skip
        for hx, hy in (
            (-MAP_HALO_CM, 0.0), (MAP_HALO_CM, 0.0),
            (0.0, -MAP_HALO_CM), (0.0, MAP_HALO_CM),
        ):  # fmt: skip
            fig.text(x=lon_of(left + hx), y=lat_of(top + hy), text=text,
                     font=MAP_LABEL_HALO, justify="LM")  # fmt: skip
        fig.text(x=lon_of(left), y=lat_of(top), text=text,
                 font=MAP_LABEL_FONT, justify="LM")  # fmt: skip

    if legend:
        n_auto = int(automatic.sum())
        if colour_by == "period":
            title = f"Skjálftar eftir tímabili: stærð = M, M < {min_mag:g} sem duft"
            items: list[tuple[str, str, str, float, bool]] = [
                ("c", PERIOD_FILLS[0], PERIOD_LABELS[0], MAP_LEGEND_CM, True),
                ("c", PERIOD_FILLS[1], PERIOD_LABELS[1], MAP_LEGEND_CM, True),
                ("c", PERIOD_FILLS[2], PERIOD_LABELS[2], MAP_LEGEND_CM, True),
                ("a", "red", f"M ≥ {star:g} (tala við M ≥ {label:g})",
                 MAP_LEGEND_CM, True),
                ("c", DUST_COLOR, f"M < {min_mag:g} (duft)", 0.10, False),
                ("c", MAP_AUTO_FILL, f"sjálfvirk, stærð óyfirfarin ({n_auto})",
                 0.10, False),
            ]  # fmt: skip
        else:
            title = f"Skjálftar M≥{min_mag:g}"
            items = [
                ("c", "gray55", f"M < {MAG_MID:g}", MAP_LEGEND_CM, True),
                ("c", "blue", f"{MAG_MID:g} ≤ M < {star:g}", MAP_LEGEND_CM, True),
                ("a", "red", f"M ≥ {star:g}", MAP_LEGEND_CM, True),
                ("c", MAP_AUTO_FILL, f"sjálfvirk, stærð óyfirfarin ({n_auto})",
                 0.10, False),
            ]  # fmt: skip
        text_cm = max(
            [maps.text_width_cm(title, font)]
            + [maps.text_width_cm(t, font) + c for _s, _f, t, c, _o in items]
        )
        wide = text_cm + 2 * MAP_LEGEND_PAD_CM
        tall = (2 + len(items)) * MAP_LEGEND_ROW_CM + 2 * MAP_LEGEND_PAD_CM
        width, height = size_cm
        gap = MAP_LEGEND_PAD_CM + 0.19
        corner = legend_corner.upper()
        note_box = getattr(fig, "gpsplot_note_box", None)
        if legend_at is not None:
            # an explicit lower-left: no corner logic, no layout engine
            x0, y0 = float(legend_at[0]), float(legend_at[1])
            top = y0 + tall
        elif corner.startswith("BELOW") and note_box is not None:
            # under the missing-data box, which is itself under the logo
            x0 = float(note_box[0])
            top = float(note_box[2]) - gap
            y0 = top - tall
        elif corner == "BL":
            x0, y0 = gap, gap
            top = y0 + tall
        elif corner == "TR":
            x0 = width - wide - gap
            top = height - legend_clear_cm
            y0 = top - tall
        else:  # TL, below the logo
            x0 = gap
            top = height - legend_clear_cm
            y0 = top - tall
        x0 = max(0.0, min(x0, width - wide))  # never off the frame
        y0 = max(0.0, y0)
        top = y0 + tall
        edge = [(x0, y0), (x0 + wide, y0), (x0 + wide, top), (x0, top), (x0, y0)]
        fig.plot(x=[lon_of(cx) for cx, _ in edge], y=[lat_of(cy) for _, cy in edge],
                 fill="white", pen="0.4p,gray40", close=True)  # fmt: skip
        fig.text(x=lon_of(x0 + MAP_LEGEND_PAD_CM),
                 y=lat_of(top - MAP_LEGEND_PAD_CM - 0.5 * MAP_LEGEND_ROW_CM),
                 text=title, font=font, justify="LM")  # fmt: skip
        for k, (style, colour, text, size, outlined) in enumerate(items):
            y = top - MAP_LEGEND_PAD_CM - (k + 1.5) * MAP_LEGEND_ROW_CM
            swatch: dict[str, Any] = {"fill": colour}
            if outlined:
                swatch["pen"] = MAP_STAR_PEN if style == "a" else MAP_CIRCLE_PEN
            fig.plot(x=[lon_of(x0 + MAP_LEGEND_PAD_CM + 0.5 * size)],
                     y=[lat_of(y)], style=f"{style}{size}c", **swatch)  # fmt: skip
            fig.text(x=lon_of(x0 + MAP_LEGEND_PAD_CM + size + 0.15),
                     y=lat_of(y), text=text, font=font, justify="LM")  # fmt: skip
    return MapQuakeInfo(
        n=len(df), n_automatic=int(automatic.sum()), n_labelled=len(placed)
    )
