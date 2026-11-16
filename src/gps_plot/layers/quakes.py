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
from typing import Any
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
) -> tuple[object, object, str]:
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


def add_quake_panel(
    ax: object,
    df: object,
    report: object,
    region: str,
    *,
    star: float = MAG_STAR,
    label: float = MAG_LABEL,
    moment: bool = True,
) -> QuakeLayerInfo:
    """Quakes and their cumulative moment on twin axes of ``ax``.

    Magnitudes on a right axis (stem + dot per event; grey < 4, blue
    4–``star``, a red star from ``star``, the value written from ``label``;
    automatic events hollow). The cumulative moment on an outer right axis:
    grey fill behind the data, a black step line in front, solid on SIL,
    dashed from the seam (moment estimated from SeisComP magnitudes, PLAN-
    ts-quakes Q4). A red dashed line marks the seam. ``ax`` keeps its
    data and limits; its background moves to the lowest twin.
    """
    import matplotlib.dates as mdates
    import numpy as np
    import pandas as pd  # type: ignore[import-untyped]
    from matplotlib.axes import Axes

    assert isinstance(ax, Axes) and isinstance(df, pd.DataFrame)
    fig = ax.figure
    lo, hi = ax.get_xlim()
    t = (
        df.index.tz_convert("UTC").tz_localize(None)
        if len(df)
        else pd.DatetimeIndex([])
    )
    x = np.zeros(0)
    if len(t):
        x = np.array(mdates.date2num(t.to_pydatetime()), dtype=np.float64)  # type: ignore[no-untyped-call]
    mag = (
        pd.to_numeric(df["magnitude"]).to_numpy(dtype=float, na_value=np.nan)
        if len(df)
        else np.zeros(0)
    )
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
    axm.vlines(x[ok], 0, mag[ok], color="0.8", lw=0.25, zorder=1)
    classes: tuple[tuple[Any, dict[str, Any], str], ...] = (
        (ok & (mag < MAG_MID), dict(marker="o", color="0.55", ms=2.0, alpha=0.7),
         rf"M $<$ {MAG_MID:g}"),
        (ok & (mag >= MAG_MID) & (mag < star), dict(marker="o", color="tab:blue", ms=3.5),
         rf"{MAG_MID:g} $\leq$ M $<$ {star:g}"),
        (ok & (mag >= star), dict(marker="*", color="tab:red", mec="k", mew=0.5, ms=11),
         rf"M $\geq$ {star:g}"),
    )  # fmt: skip
    for sel, style, text in classes:
        if not sel.any():
            continue
        for hollow in (False, True):
            s = sel & (auto if hollow else ~auto)
            if not s.any():
                continue
            kw: dict[str, Any] = dict(style)
            if hollow:
                kw["mfc"] = "none"
                kw["mec"] = kw.get("color")
            (h,) = axm.plot(x[s], mag[s], ls="none", zorder=3, **kw)
            if not hollow:
                h.set_label(text)
                handles.append(h)
    if auto.any():
        (h,) = axm.plot([], [], ls="none", marker="o", mfc="none", mec="0.4", ms=4,
                        label="automatic (hollow)")  # fmt: skip
        handles.append(h)
    for xi, mi in zip(x[ok & (mag >= label)], mag[ok & (mag >= label)]):
        axm.annotate(f"{mi:.1f}", (xi, mi), xytext=(4, 4), textcoords="offset points",
                     fontsize=8, fontweight="bold", color="tab:red")  # fmt: skip
    top = float(np.nanmax(mag[ok])) if ok.any() else 0.0
    axm.set_ylim(0, max(5.5, top + 0.5))
    axm.set_ylabel("Magnitude")

    m0 = np.zeros(0)
    if moment and len(df) and "moment_nm" in df:
        m0 = pd.to_numeric(df["moment_nm"]).to_numpy(dtype=float, na_value=np.nan)
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
    # below the logo, which sits in the top left corner of the top panel
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, LEGEND_TOP),
              ncol=3, fontsize="small",
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
