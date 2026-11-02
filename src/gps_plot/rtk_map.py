"""``gpsplot rtk-map``: the last hours of RTK motion on a map, one frame.

Operational overview for an area (BGÓ 2026-11-02: Reykjanes, the last
12 h, every 5 min). Each RTK pair ``ROVER-BASE`` gives the change of
``rover - base`` over the span: the median of its last ``--edge`` minutes
minus the median of its first (10-min medians, float kept as the RTK plots
do). The pairs are solved as one network per component (weighted least
squares, :func:`solve_network`) so the arrows share one frame instead of
each being relative to its own, moving, base: by default the network's own
frame, no net motion of its stations. Far anchors (``--anchors VOGC,KEIC``)
were tried first (2026-11-02) and dropped as the default: their 10-20 km
float links trended 20-50 mm in 12 h, a false common motion larger than a
quiet day's signal. The legend says which stations set the frame;
stations no live pair reaches are left out and said so. Horizontal motion is drawn as arrows (1-2-5 scale
arrow, :data:`gps_plot.map.auto_scale`), the vertical is written beside
each station.
"""

from __future__ import annotations

import argparse
import datetime as dt
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

#: 10-min medians; the span's first and last EDGE minutes give its ends
INTERVAL = "10min"
DEFAULT_EDGE_MIN = 60.0
#: a pair's end difference counts with this floor on its sigma [mm]
MIN_SIGMA_MM = 2.0


def pair_change(
    time: Any, neu: Any, t0: np.datetime64, t1: np.datetime64, edge_min: float
) -> tuple[np.ndarray, np.ndarray] | None:
    """``(change, sigma)`` N/E/U [mm] of one pair over ``[t0, t1]``: the
    median of its last ``edge_min`` minutes minus that of its first, sigma
    from the scatter in the two edges (MAD); None when an edge is empty."""
    t = np.asarray(time).astype("datetime64[s]")
    y = np.asarray(neu, dtype=float) * 1000.0
    edge = np.timedelta64(int(edge_min * 60), "s")
    first = (t >= t0) & (t < t0 + edge) & np.isfinite(y).all(axis=0)
    last = (t > t1 - edge) & (t <= t1) & np.isfinite(y).all(axis=0)
    if not first.any() or not last.any():
        return None
    a, b = np.median(y[:, first], axis=1), np.median(y[:, last], axis=1)

    def spread(sel: np.ndarray) -> np.ndarray:
        v = y[:, sel]
        mad = 1.4826 * np.median(np.abs(v - np.median(v, axis=1)[:, None]), axis=1)
        return np.asarray(mad / math.sqrt(max(sel.sum(), 1)))

    sigma = np.maximum(np.hypot(spread(first), spread(last)), MIN_SIGMA_MM)
    return b - a, sigma


def solve_network(
    changes: Mapping[tuple[str, str], tuple[np.ndarray, np.ndarray]],
    anchors: Sequence[str] = (),
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """``{station: (displacement, sigma)}`` N/E/U [mm] from pair changes
    ``{(rover, base): (change, sigma)}`` (change = d_rover - d_base).

    With ``anchors``: those held at zero, only the stations tied to one.
    Without (the default): the largest connected group of stations, with
    no net motion (their mean is zero, each component) — the frame of the
    network itself, free of the drift long links to far anchors carry
    (BGÓ 2026-11-02). Weighted least squares per component; sigma from its
    covariance."""
    if not anchors:
        return _solve_mean(changes)
    reach = {a for a in anchors if any(a in p for p in changes)}
    grew = True
    while grew:
        grew = False
        for r, b in changes:
            if (r in reach) != (b in reach):
                reach |= {r, b}
                grew = True
    free = sorted(reach - set(anchors))
    if not free:
        return {a: (np.zeros(3), np.zeros(3)) for a in reach}
    idx = {s: i for i, s in enumerate(free)}
    pairs = [p for p in changes if p[0] in reach and p[1] in reach]
    disp = np.zeros((len(free), 3))
    sig = np.zeros((len(free), 3))
    for c in range(3):
        rows, rhs, w = [], [], []
        for r, b in pairs:
            row = np.zeros(len(free))
            if r in idx:
                row[idx[r]] += 1.0
            if b in idx:
                row[idx[b]] -= 1.0
            if not row.any():
                continue  # anchor to anchor: no unknown
            v, s = changes[(r, b)]
            rows.append(row)
            rhs.append(v[c])
            w.append(1.0 / s[c])
        A = np.array(rows) * np.array(w)[:, None]
        y = np.array(rhs) * np.array(w)
        x, *_ = np.linalg.lstsq(A, y, rcond=None)
        cov = np.linalg.pinv(A.T @ A)
        disp[:, c] = x
        sig[:, c] = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    out = {s: (disp[i], sig[i]) for s, i in idx.items()}
    out.update({a: (np.zeros(3), np.zeros(3)) for a in anchors if a in reach})
    return out


def _groups(pairs: Sequence[tuple[str, str]]) -> list[set[str]]:
    """The connected groups of stations, largest first."""
    groups: list[set[str]] = []
    for r, b in pairs:
        hit = [g for g in groups if r in g or b in g]
        merged = {r, b}.union(*hit) if hit else {r, b}
        groups = [g for g in groups if g not in hit] + [merged]
    return sorted(groups, key=len, reverse=True)


def _solve_mean(
    changes: Mapping[tuple[str, str], tuple[np.ndarray, np.ndarray]],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """:func:`solve_network` without anchors: the largest group, mean zero."""
    groups = _groups(list(changes))
    if not groups:
        return {}
    stations = sorted(groups[0])
    idx = {s: i for i, s in enumerate(stations)}
    pairs = [p for p in changes if p[0] in idx and p[1] in idx]
    disp = np.zeros((len(stations), 3))
    sig = np.zeros((len(stations), 3))
    for c in range(3):
        rows, rhs, w = [], [], []
        for r, b in pairs:
            row = np.zeros(len(stations))
            row[idx[r]] += 1.0
            row[idx[b]] -= 1.0
            v, s_ = changes[(r, b)]
            rows.append(row)
            rhs.append(v[c])
            w.append(1.0 / s_[c])
        A = np.array(rows) * np.array(w)[:, None]
        y = np.array(rhs) * np.array(w)
        # the datum: no net motion of the group (a heavy constraint row)
        big = 1e3 * float(np.max(np.abs(A))) if A.size else 1.0
        A = np.vstack([A, np.full(len(stations), big / len(stations))])
        y = np.append(y, 0.0)
        x, *_ = np.linalg.lstsq(A, y, rcond=None)
        cov = np.linalg.pinv(A.T @ A)
        disp[:, c] = x
        sig[:, c] = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    return {s: (disp[i], sig[i]) for s, i in idx.items()}


def build_parser(prog: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=prog,
        description="The last hours of RTK motion on a map: a network of pairs "
        "solved with far anchors held still, one frame for all stations.",
    )
    p.add_argument("--pairs", required=True,
                   help="the RTK pairs of the network, ROVER-BASE, comma separated")  # fmt: skip
    p.add_argument("--stations", required=True,
                   help="the stations drawn, comma separated (others only tie the network)")  # fmt: skip
    p.add_argument("--anchors", default="",
                   help="stations held at zero; default none: the network's own frame, "
                   "no net motion of its stations (far anchors' long links drift)")  # fmt: skip
    p.add_argument(
        "--hours", type=float, default=12.0, help="the span ending now (default 12)"
    )
    p.add_argument("--edge", type=float, default=DEFAULT_EDGE_MIN,
                   help="minutes at each end whose median is the end's level (default 60)")  # fmt: skip
    p.add_argument(
        "-e", "--end", default=None, help="end, YYYYMMDD-HH:MM UTC (default now)"
    )
    p.add_argument("--region", required=True, help="W/E/S/N")
    p.add_argument("--projection", default="M15.5c")
    p.add_argument(
        "--dem", default="arcticdem", help="arcticdem (default), iceland or none"
    )
    p.add_argument("--sea-smooth", type=float, default=2.0)
    p.add_argument(
        "--rtk-dir", default=None, help="the RTK pair directories (default: config)"
    )
    p.add_argument("-d", "--figdir", default=".", help="output directory")
    p.add_argument("--name", default="rtk-map", help="file name without .png")
    p.add_argument("--title", default=None)
    return p


def run(prog: str, argv: list[str]) -> int:
    from geo_dataread.rtk_read import read_rtk_baseline

    from gps_plot import maps
    from gps_plot.map import auto_scale, gis_dir
    from gps_plot.plot_gps_timeseries import _load_config

    args = build_parser(prog).parse_args(argv)
    end = (dt.datetime.strptime(args.end, "%Y%m%d-%H:%M") if args.end
           else dt.datetime.now(dt.UTC).replace(tzinfo=None))  # fmt: skip
    start = end - dt.timedelta(hours=args.hours)
    t0, t1 = np.datetime64(start, "s"), np.datetime64(end, "s")
    changes: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    for pair in [x.strip().upper() for x in args.pairs.split(",") if x.strip()]:
        try:
            b = read_rtk_baseline(
                pair, start, end, root=args.rtk_dir, interval=INTERVAL
            )
            ch = pair_change(b.time, b.neu, t0, t1, args.edge)
        except Exception as exc:  # noqa: BLE001 - one dead pair must not stop the map
            print(f"{pair}: {exc}", file=sys.stderr)
            continue
        if ch is None:
            print(f"{pair}: no data at an end of the span", file=sys.stderr)
            continue
        rover, base = pair.split("-")
        changes[(rover, base)] = ch
    anchors = [x.strip().upper() for x in args.anchors.split(",") if x.strip()]
    solved = solve_network(changes, anchors)
    wanted = [x.strip().upper() for x in args.stations.split(",") if x.strip()]
    shown = [s for s in wanted if s in solved]
    missing = [s for s in wanted if s not in solved]
    if missing:
        print(f"not in the network (no live pair at both ends of the span): "
              f"{' '.join(missing)}", file=sys.stderr)  # fmt: skip
    if not shown:
        print("nothing to draw", file=sys.stderr)
        return 1
    coords = maps.station_coordinates(shown)
    labelled = [maps.StationCoordinate(f"{c.marker} {solved[c.marker][0][2]:+.0f}", c.lon, c.lat)
                for c in coords]  # fmt: skip
    d = np.array([solved[s][0] for s in shown])
    scale, ref = auto_scale(np.hypot(d[:, 0], d[:, 1]))
    w_, e_, s_, n_ = (float(v) for v in args.region.split("/"))
    region = (w_, e_, s_, n_)
    kw: dict[str, Any] = {}
    if args.dem != "none":
        choice = maps.iceland_dem(gis_dir(_load_config()), region, args.projection,
                                  prefer_fine=args.dem == "arcticdem")  # fmt: skip
        kw.update(dem_grid=choice.grid, coastline=choice.coastline, lakes=choice.lakes,
                  dem_cmap="lightgrey", sea_smooth=args.sea_smooth)  # fmt: skip
        if choice.fine:
            kw.update(maps.FINE_DEM_SHADING)
    Path(args.figdir).mkdir(parents=True, exist_ok=True)
    out = Path(args.figdir) / f"{args.name}.png"
    frame = "+".join(anchors) if anchors else f"the mean of {len(solved)} stations"
    print(f"frame: {frame} ({' '.join(sorted(solved))})")
    title = args.title or f"RTK {args.hours:g} h to {end:%Y-%m-%d %H:%M} UTC"
    maps.deformation_vectors(
        labelled, d[:, 1], d[:, 0],
        observed_label=f"{args.hours:g} h, relative to {frame}; label: vertical mm",
        observed_color="black", region=region, projection=args.projection,
        scale=scale, scale_ref=ref, title=title, labels=True, outfile=str(out), **kw,
    )  # fmt: skip
    print(out)
    return 0
