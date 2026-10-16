"""gps_plot.layout: logo, legend and scale bar over the sea on the overview."""

from __future__ import annotations

import numpy as np
import pytest

from gps_plot import layout


def _sea(width=12.0, height=8.0, land=()):
    """A layout of ``width`` x ``height`` cm, land in the (x0, x1, y0, y1) boxes."""
    ny, nx = round(height / layout.CELL_CM), round(width / layout.CELL_CM)
    sea = np.ones((ny, nx), dtype=bool)
    for x0, x1, y0, y1 in land:
        sea[round(y0 / layout.CELL_CM) : round(y1 / layout.CELL_CM),
            round(x0 / layout.CELL_CM) : round(x1 / layout.CELL_CM)] = False  # fmt: skip
    return layout.SeaLayout(width, height, sea)


def test_box_goes_in_its_corner_over_open_sea():
    lay = _sea()
    x, y = lay.place((2.0, 1.0), "TR")
    m = layout.MARGIN_CM
    assert x == pytest.approx(12.0 - m - 2.0) and y == pytest.approx(8.0 - m - 1.0)
    x, y = lay.place((2.0, 1.0), "BL")
    assert (x, y) == (pytest.approx(m), pytest.approx(m))


def test_box_moves_off_land_and_off_placed_boxes():
    # a peninsula in the top-right corner: the box moves to its nearest sea
    lay = _sea(land=[(10.5, 12.0, 6.5, 8.0)])
    x, y = lay.place((2.0, 1.0), "TR")
    assert not (x + 2.0 > 10.5 and y + 1.0 > 6.5)  # clear of the land
    # the next box in the same corner keeps clear of the first
    x2, y2 = lay.place((2.0, 1.0), "TR")
    assert x2 + 2.0 <= x or y2 + 1.0 <= y or x2 >= x + 2.0 or y2 >= y + 1.0
    # reserved: the box keeps out
    lay = _sea()
    lay.reserve((0.0, 12.0, 0.0, 1.0))
    assert lay.place((2.0, 0.5), "BR")[1] >= 1.0 + layout.MARGIN_CM


def test_corners_in_order_and_none_without_sea():
    lay = _sea(land=[(0.0, 4.0, 4.0, 8.0)])  # the top left is land
    x, y = lay.place((2.0, 1.0), ("TL", "BL"), reach=0.5)
    assert y < 4.0  # TL had no sea within reach: BL
    assert _sea(land=[(0.0, 12.0, 0.0, 8.0)]).place((1.0, 1.0), ("TL", "BR")) is None


def test_legend_size_estimate():
    w, h = layout.text_box_cm(["External (2)", "Continuous (61)"])
    assert 2.5 < w < 3.2 and h == pytest.approx(0.9)


def _pygmt_reason() -> str | None:
    try:
        import pygmt  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return f"pygmt/GMT unavailable: {exc}"
    return None


@pytest.mark.skipif(_pygmt_reason() is not None, reason=_pygmt_reason() or "")
def test_overview_mask_from_the_dem():
    import xarray as xr

    from gps_plot import maps

    region = maps.corner_region((-25.0, 63.0, -12.8, 66.7), "S-19/65/12c")
    w, e, s, n = region
    lon = np.round(np.arange(w - 0.1, e + 0.1, 0.05), 4)
    lat = np.round(np.arange(s - 0.1, n + 0.1, 0.05), 4)
    # land east of 20°W only
    z = np.where(lon[None, :] > -20.0, 50.0, -50.0) + 0 * lat[:, None]
    grid = xr.DataArray(z, coords={"lat": lat, "lon": lon}, dims=("lat", "lon"))
    lay = layout.build(region, land=grid)
    assert lay.sea[:, :5].all() and not lay.sea[:, -5:].any()
    # the bottom annotation band, the logo, the scale bar and whole-degree
    # latitudes are reserved
    assert lay.taken[0] == (0.0, region.size[0], 0.0, layout.BOTTOM_LABELS_CM)
    assert len(lay.taken) >= 5
    # a box at the top right has no sea: the top left has
    x, _ = lay.place((2.0, 0.8), ("TR", "TL"))
    assert x < 3.0
    assert layout.km_per_cm(region) == pytest.approx(48, rel=0.05)


def test_stations_block_the_legend():
    lay = _sea()
    lay.rj = "-R0/12/0/8 -JX12c/8c"  # a plain cm frame: lon = x, lat = y
    lay.block_points([1.5], [6.8])  # a station in the top-left corner
    x, y = lay.place((2.0, 1.0), "TL")
    assert x > 1.5 + layout.POINT_CM or y + 1.0 < 6.8 - layout.POINT_CM
