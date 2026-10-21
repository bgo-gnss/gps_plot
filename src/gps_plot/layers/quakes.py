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
from dataclasses import dataclass

from matplotlib.figure import Figure

#: mean Earth radius (km) for the great-circle distance
EARTH_RADIUS_KM = 6371.0
QUAKE_COLOR = "darkorange"


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
    from matplotlib.transforms import blended_transform_factory

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
            " " + quake_label(q, points),
            transform=blended_transform_factory(top.transData, top.transAxes),
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
