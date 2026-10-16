"""The legend over the sea, off the stations, on the full-Iceland overview.

On the oblique overview (:class:`gps_plot.maps.CornerRegion`) the logo
(bottom left) and the scale bar (bottom right) keep their places: the
overview frame (``map.OVERVIEW_CORNERS``) is drawn with sea there. The
legend goes where it fits: a :class:`SeaLayout` is a
sea/land mask of the map in paper cm (:data:`CELL_CM` cells; land is the
drawn DEM's z >= 0, else GSHHG land and lakes), with the station symbols
blocked (:meth:`SeaLayout.block_points`). :meth:`SeaLayout.place` puts
a box of a given size in the first of its corners
(:data:`LEGEND_CORNERS`) with room: the free position nearest it,
inside :data:`SEARCH` of the map, all sea and clear of the boxes
already placed, the logo and scale bar and the degree annotations
inside the frame. None when there is none: the legend keeps its usual
position.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Any

import numpy as np

#: mask cell [cm]
CELL_CM = 0.1
#: gap kept from the frame and between boxes [cm]
MARGIN_CM = 0.15
#: how far from its corner a box may move, as a fraction of the map size
SEARCH = 0.45
#: the bottom band of the upright longitude annotations [cm]
BOTTOM_LABELS_CM = 1.0
#: a latitude annotation's box at the left edge [cm]: width, half height
LEFT_LABEL_CM = (1.2, 0.25)
#: a Mercator map's annotation bands inside the frame, bottom and left [cm]
BOX_BOTTOM_LABELS_CM = 0.45
BOX_LEFT_LABELS_CM = 1.25

#: where the legend goes, in order of preference (the logo stays bottom
#: left and the scale bar bottom right: the overview frame keeps them at sea)
LEGEND_CORNERS = ("TL", "TR", "BL", "BR")
#: the fixed logo's box, bottom left: width, bottom, top [cm]
LOGO_BOX_CM = (2.45, 0.0, 0.9)
#: the fixed scale bar's box: width from the right, bottom, top [cm]
SCALE_BAR_BOX_CM = (3.4, 1.55, 2.45)
#: half the size of a station symbol, kept free of the legend [cm]
POINT_CM = 0.25


@dataclasses.dataclass
class SeaLayout:
    """Sea cells of a map and the boxes placed on it (all in cm, origin
    at the lower-left corner)."""

    width: float
    height: float
    sea: np.ndarray  # [row (y), col (x)], True = sea
    taken: list[tuple[float, float, float, float]] = dataclasses.field(
        default_factory=list
    )
    #: GMT ``-R… -J…`` of the map, to place points (None: no projection)
    rj: str | None = None
    #: cells kept free of every box, sea or land: the map symbols
    blocked: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.blocked is None:
            self.blocked = np.zeros_like(self.sea, dtype=bool)
        self._index()

    def _index(self) -> None:
        assert self.blocked is not None
        land = (~self.sea | self.blocked).astype(np.int32)
        self._sum = np.pad(land.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
        blocked = self.blocked.astype(np.int32)
        self._blocked_sum = np.pad(blocked.cumsum(0).cumsum(1), ((1, 0), (1, 0)))

    def block_points(
        self, lons: Sequence[float], lats: Sequence[float], radius: float = POINT_CM
    ) -> None:
        """Keep boxes off the map symbols at ``lons``/``lats``: the cells
        within ``radius`` cm of each are blocked from now on, for boxes at
        sea and on land alike."""
        if not len(lons) or self.rj is None:
            return
        assert self.blocked is not None
        from gps_plot.maps import _mapproject

        xy = _mapproject(self.rj, list(lons), list(lats))[:, :2]
        ny, nx = self.sea.shape
        cx = (np.arange(nx) + 0.5) * CELL_CM
        cy = (np.arange(ny) + 0.5) * CELL_CM
        for x, y in xy:
            cols = np.abs(cx - x) <= radius
            rows = np.abs(cy - y) <= radius
            self.blocked[np.ix_(rows, cols)] = True
        self._index()

    def block(self, mask: np.ndarray) -> None:
        """Block the cells of ``mask`` (shaped as :attr:`sea`) for every box."""
        assert self.blocked is not None
        self.blocked |= mask
        self._index()

    def _all_sea(
        self, x0: float, x1: float, y0: float, y1: float, on_land: bool = False
    ) -> bool:
        """No land (``on_land``: no blocked cell) under the box."""
        ny, nx = self.sea.shape
        c0, c1 = max(int(x0 / CELL_CM), 0), min(int(np.ceil(x1 / CELL_CM)), nx)
        r0, r1 = max(int(y0 / CELL_CM), 0), min(int(np.ceil(y1 / CELL_CM)), ny)
        if c0 >= c1 or r0 >= r1:
            return False
        s = self._blocked_sum if on_land else self._sum
        return bool(s[r1, c1] - s[r0, c1] - s[r1, c0] + s[r0, c0] == 0)

    def _clear(
        self, box: tuple[float, float, float, float], on_land: bool = False
    ) -> bool:
        x0, x1, y0, y1 = box
        if x0 < MARGIN_CM or y0 < MARGIN_CM:
            return False
        if x1 > self.width - MARGIN_CM or y1 > self.height - MARGIN_CM:
            return False
        for t0, t1, u0, u1 in self.taken:
            if (
                x0 < t1 + MARGIN_CM
                and t0 < x1 + MARGIN_CM
                and y0 < u1 + MARGIN_CM
                and u0 < y1 + MARGIN_CM
            ):
                return False
        return self._all_sea(x0, x1, y0, y1, on_land)

    def reserve(self, box: tuple[float, float, float, float]) -> None:
        """Keep ``box`` (x0, x1, y0, y1) free of anything placed later."""
        self.taken.append(box)

    def place(
        self,
        size: tuple[float, float],
        corners: str | Sequence[str],
        reach: float | None = None,
        on_land: bool = False,
    ) -> tuple[float, float] | None:
        """The lower-left (x, y) [cm] of a ``size`` (w, h) box in the free
        sea position nearest the first of ``corners`` (TL/TR/BL/BR) that
        has one, at most ``reach`` cm from it (default :data:`SEARCH` of
        the map); None when none has. The box is then taken. ``on_land``
        lets the box cover land, but still no symbol."""
        for corner in [corners] if isinstance(corners, str) else corners:
            spot = self._place(size, corner, reach, on_land)
            if spot is not None:
                return spot
        return None

    def _place(
        self,
        size: tuple[float, float],
        corner: str,
        reach: float | None,
        on_land: bool = False,
    ) -> tuple[float, float] | None:
        w, h = size
        right, top = corner[1] == "R", corner[0] == "T"
        reach_x = SEARCH * self.width if reach is None else reach
        reach_y = SEARCH * self.height if reach is None else reach
        steps = np.arange(0.0, max(reach_x, reach_y) + 1e-9, CELL_CM)
        best = None
        for dx in steps:
            if dx > reach_x or (best is not None and dx >= best[0]):
                break
            for dy in steps:
                if dy > reach_y:
                    break
                if best is not None and dx + dy >= best[0]:
                    break
                x0 = self.width - MARGIN_CM - w - dx if right else MARGIN_CM + dx
                y0 = self.height - MARGIN_CM - h - dy if top else MARGIN_CM + dy
                if self._clear((x0, x0 + w, y0, y0 + h), on_land):
                    best = (dx + dy, x0, y0)
                    break
        if best is None:
            return None
        _, x0, y0 = best
        self.taken.append((x0, x0 + w, y0, y0 + h))
        return x0, y0


def build(
    region: Any, *, land: Any = None, resolution: str = "intermediate"
) -> SeaLayout:
    """The :class:`SeaLayout` of an oblique ``region`` (a ``CornerRegion``):
    ``land`` is the drawn DEM (z >= 0 land), None tests GSHHG. The degree
    annotations inside the frame are reserved."""
    from gps_plot.maps import _mapproject

    width, height = region.size
    rj = f"-R{region.gmt} -J{region.projection}"
    xs = np.arange(CELL_CM / 2, width, CELL_CM)
    ys = np.arange(CELL_CM / 2, height, CELL_CM)
    gx, gy = np.meshgrid(xs, ys)
    lonlat = _mapproject(f"{rj} -I", gx.ravel(), gy.ravel())[:, :2]
    on_land = _is_land(lonlat, land, region, resolution).reshape(gy.shape)
    layout = SeaLayout(width, height, ~on_land, rj=rj)
    layout.reserve((0.0, width, 0.0, BOTTOM_LABELS_CM))
    # the logo (top left) and the scale bar (bottom right) keep their places
    lw_, ly0, ly1 = LOGO_BOX_CM
    layout.reserve((0.0, lw_, ly0, ly1))
    sw_, sy0, sy1 = SCALE_BAR_BOX_CM
    layout.reserve((width - sw_, width, sy0, sy1))
    # the latitude annotations: whole degrees along the left edge
    edge_y = np.linspace(0.0, height, 200)
    lats = _mapproject(f"{rj} -I", np.zeros_like(edge_y), edge_y)[:, 1]
    for lat in range(int(np.ceil(lats.min())), int(np.floor(lats.max())) + 1):
        y = float(np.interp(lat, lats, edge_y))
        lw, lh = LEFT_LABEL_CM
        layout.reserve((0.0, lw, y - lh, y + lh))
    return layout


def build_box(
    region: Sequence[float],
    projection: str,
    *,
    land: Any = None,
    resolution: str = "high",
) -> SeaLayout:
    """The :class:`SeaLayout` of a plain W/E/S/N ``region`` (a Mercator
    map): ``land`` the drawn DEM (z >= 0 land), None tests GSHHG. The
    annotations inside the frame (W and S) are reserved as bands."""
    from gps_plot.maps import _mapproject

    w, e, s, n = region
    rj = f"-R{w}/{e}/{s}/{n} -J{projection}"
    ((width, height, *_),) = _mapproject(f"{rj} -W")
    xs = np.arange(CELL_CM / 2, width, CELL_CM)
    ys = np.arange(CELL_CM / 2, height, CELL_CM)
    gx, gy = np.meshgrid(xs, ys)
    lonlat = _mapproject(f"{rj} -I", gx.ravel(), gy.ravel())[:, :2]
    on_land = _is_land(lonlat, land, region, resolution).reshape(gy.shape)
    layout = SeaLayout(float(width), float(height), ~on_land, rj=rj)
    layout.reserve((0.0, float(width), 0.0, BOX_BOTTOM_LABELS_CM))
    layout.reserve((0.0, BOX_LEFT_LABELS_CM, 0.0, float(height)))
    return layout


def _is_land(
    lonlat: np.ndarray, land: Any, region: Sequence[float], resolution: str
) -> np.ndarray:
    import pandas as pd  # type: ignore[import-untyped]
    import pygmt

    pts = pd.DataFrame(
        {"x": lonlat[:, 0], "y": lonlat[:, 1], "i": np.arange(len(lonlat))}
    )
    out = np.zeros(len(lonlat), dtype=bool)
    if land is not None:
        z = pygmt.grdtrack(grid=land, points=pts, newcolname="z")
        hit = z.dropna(subset=["z"])
        out[hit["i"].astype(int).to_numpy()] = hit["z"].to_numpy() >= 0
        return out
    w, e, s, n = region
    kept = pygmt.select(
        data=pts, mask_values="s/k/k/k/k", resolution=resolution,
        region=[w - 1, e + 1, s - 1, n + 1],
    )  # fmt: skip
    if kept is not None and len(kept):
        out[kept.iloc[:, 2].astype(int).to_numpy()] = True
    return out


def text_box_cm(lines: Sequence[str], *, symbol: bool = True) -> tuple[float, float]:
    """The size of a GMT legend of ``lines`` (one symbol + text a row) at
    the map's 8p annotation font, measured on rendered legends [cm]."""
    chars = max((len(t) for t in lines), default=0)
    return 0.85 * symbol + 0.135 * chars, 0.30 * len(lines) + 0.3


def km_per_cm(region: Any) -> float:
    """Ground km per map cm across the middle of an oblique ``region``."""
    import math

    (lon0, lat0), (lon1, lat1) = region.at(0.0, 0.5), region.at(1.0, 0.5)
    p0, p1, dl = math.radians(lat0), math.radians(lat1), math.radians(lon1 - lon0)
    c = math.sin(p0) * math.sin(p1) + math.cos(p0) * math.cos(p1) * math.cos(dl)
    return float(6371.0 * math.acos(max(-1.0, min(1.0, c))) / region.size[0])
