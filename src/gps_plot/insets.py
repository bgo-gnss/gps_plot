"""Insets on a regional map: the Iceland locator and a zoom box.

Both follow BGÓ's ``gmt_maps/bb_grein/fig3b-overvlabel.pdf``:

* **Locator** (:func:`draw_locator`): a small oblique Iceland (``S-19/65``,
  as the overview), white land, the fissure swarms slate-blue, the map's
  box in red.
* **Zoom** (:func:`draw_zoom`): a box of the map drawn again, larger,
  so a dense cluster (Svartsengi: ~20 stations in ~6 km) can be named.
  Its outline is drawn on the map; the stations inside are named only in
  the inset.

Where they go (:func:`place`): a :class:`gps_plot.layout.SeaLayout` of
the map, with the logo, the scale bar and the vector legend reserved and
the station symbols blocked. Each inset takes the free spot nearest its
preferred corners, at sea first, then over land (never over a symbol);
a zoom inset with no room inside goes beside the map, right of it.
"""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Callable, Sequence
from typing import Any

from gps_plot.gmt_overrides import layer_kw

#: the locator's frame: the iceland3 overview's corners, a little trimmed
LOCATOR_CORNERS = (-24.6, 63.2, -13.0, 66.65)
LOCATOR_WIDTH_CM = 2.8
LOCATOR_CENTRE = "-19/65"
LOCATOR_LAND = "white"
LOCATOR_SEA = "white"
LOCATOR_COAST_PEN = "0.2p,black"
LOCATOR_SWARM_FILL = "119/136/153"  # the overview's lightslategrey
LOCATOR_BOX_PEN = "1p,red"
LOCATOR_FRAME_PEN = "0.5p,black"
#: the locator prefers the bottom-left corner, as on fig3b
LOCATOR_CORNERS_PREF = ("BL", "TR", "BR", "TL")
#: the zoom inset: width [cm] and its preferred corners
ZOOM_WIDTH_CM = 6.0
ZOOM_CORNERS_PREF = ("BR", "TR", "BL", "TL")
ZOOM_BOX_PEN = "0.9p,black"
ZOOM_FRAME_PEN = "1p,black"
#: the gap between the map and a zoom inset set beside it [cm]
BESIDE_GAP_CM = 0.4
#: GIS files the locator draws (under the GIS directory)
LOCATOR_COAST = "vector/is50v_coast_wgs84_overview.gmt"
LOCATOR_SWARMS = "vector/futurevolc_fissure_swarms_closed.gmt"


@dataclasses.dataclass(frozen=True)
class Spot:
    """Where an inset goes: lower-left (x, y) [cm] from the map's origin
    and its size [cm]; ``beside`` when outside the map frame."""

    x: float
    y: float
    width: float
    height: float
    beside: bool = False


def size_cm(region: Any, projection: str) -> tuple[float, float]:
    """Map size [cm] of ``region`` (W/E/S/N, or a ``CornerRegion``) in
    ``projection``."""
    from gps_plot.maps import CornerRegion, _mapproject

    if isinstance(region, CornerRegion):
        return region.size
    w, e, s, n = region
    ((width, height, *_),) = _mapproject(f"-R{w}/{e}/{s}/{n} -J{projection} -W")
    return float(width), float(height)


def locator_projection(width_cm: float = LOCATOR_WIDTH_CM) -> str:
    return f"S{LOCATOR_CENTRE}/{width_cm:g}c"


def locator_size(width_cm: float = LOCATOR_WIDTH_CM) -> tuple[float, float]:
    from gps_plot.maps import corner_region

    return corner_region(LOCATOR_CORNERS, locator_projection(width_cm)).size


def zoom_projection(width_cm: float = ZOOM_WIDTH_CM) -> str:
    return f"M{width_cm:g}c"


def place(
    layout: Any,
    size: tuple[float, float],
    corners: Sequence[str],
    *,
    outside: bool = False,
    map_size: tuple[float, float] | None = None,
) -> Spot | None:
    """The :class:`Spot` of a ``size`` box on ``layout``: at sea first,
    then over land off the symbols; failing both, right of the map when
    ``outside`` (top aligned), else None."""
    for on_land in (False, True):
        got = layout.place(size, corners, reach=max(layout.width, layout.height),
                           on_land=on_land)  # fmt: skip
        if got is not None:
            return Spot(got[0], got[1], *size)
    if outside and map_size is not None:
        return beside(size, map_size)
    return None


def beside(
    size: tuple[float, float],
    map_size: tuple[float, float],
    below: Spot | None = None,
) -> Spot:
    """A ``size`` box right of the map: under ``below`` when that inset is
    there (left aligned with it), else top aligned with the map."""
    if below is not None and below.beside:
        return Spot(below.x, below.y - BESIDE_GAP_CM - size[1], *size, True)
    return Spot(map_size[0] + BESIDE_GAP_CM, map_size[1] - size[1], *size, True)


def _position(spot: Spot) -> Any:
    from pygmt.params import Position

    return Position((f"{spot.x:.3f}c", f"{spot.y:.3f}c"), cstype="plotcoords",
                    anchor="BL")  # fmt: skip


def draw_locator(fig: Any, region: Sequence[float], spot: Spot, gis_dir: str) -> None:
    """The Iceland locator at ``spot``: white land, slate fissure swarms,
    the lon/lat box ``region`` in red (as fig3b)."""
    from gps_plot.maps import corner_region

    projection = locator_projection(spot.width)
    frame = corner_region(LOCATOR_CORNERS, projection)
    w, e, s, n = region[:4]
    with fig.inset(
        **layer_kw(
            "inset",
            position=_position(spot),
            box=f"+g{LOCATOR_SEA}+p{LOCATOR_FRAME_PEN}",
            region=frame.gmt,
            projection=projection,
        )
    ):
        coast = os.path.join(gis_dir, LOCATOR_COAST)
        swarms = os.path.join(gis_dir, LOCATOR_SWARMS)
        if os.path.isfile(coast):
            fig.plot(**layer_kw("locator", data=coast, fill=LOCATOR_LAND,
                                pen=LOCATOR_COAST_PEN))  # fmt: skip
        else:  # GSHHG when the GIS coast is not built
            fig.coast(land=LOCATOR_LAND, shorelines=LOCATOR_COAST_PEN,
                      resolution="low")  # fmt: skip
        if os.path.isfile(swarms):
            fig.plot(**layer_kw("locator", data=swarms, fill=LOCATOR_SWARM_FILL))
            if os.path.isfile(coast):  # the coast over the swarms' sea edges
                fig.plot(**layer_kw("locator", data=coast, pen=LOCATOR_COAST_PEN))
        fig.plot(
            **layer_kw(
                "locator", x=[w, e, e, w, w], y=[s, s, n, n, s], pen=LOCATOR_BOX_PEN
            )
        )


def draw_zoom_box(fig: Any, box: Sequence[float]) -> None:
    """The zoom inset's box, outlined on the map."""
    w, e, s, n = box
    fig.plot(**layer_kw("zoom-box", x=[w, e, e, w, w], y=[s, s, n, n, s],
                        pen=ZOOM_BOX_PEN))  # fmt: skip


def draw_zoom(
    fig: Any,
    box: Sequence[float],
    spot: Spot,
    base: dict[str, Any],
    draw: Callable[[Any, str], None],
) -> None:
    """The map's ``box`` drawn again at ``spot``: its background from the
    base-layer keywords ``base`` (:func:`gps_plot.maps.draw_base`), then
    ``draw(fig, rj)`` for the symbols (stations, vectors, names), ``rj``
    the inset's GMT ``-R… -J…``."""
    import pygmt

    from gps_plot import maps

    projection = zoom_projection(spot.width)
    w, e, s, n = box
    with fig.inset(
        **layer_kw(
            "inset",
            position=_position(spot),
            box=f"+gwhite+p{ZOOM_FRAME_PEN}",
            region=[w, e, s, n],
            projection=projection,
        )
    ):
        style = {k: v for k, v in base.items() if k in maps.BASE_KEYS}
        maps.draw_base(pygmt, fig, region=(w, e, s, n), projection=projection, **style)
        draw(fig, f"-R{w}/{e}/{s}/{n} -J{projection}")
        # the frame over the background (the inset box lies under it)
        fig.basemap(**layer_kw("frame", frame=["nswe"]))
