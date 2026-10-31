"""gps_plot.terrain: the DEM slope at a station."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from gps_plot import direction, terrain


@pytest.mark.parametrize(
    "dzdx, dzdy, az",
    [(0.0, 1.0, 180.0), (0.0, -1.0, 0.0), (1.0, 0.0, 270.0), (-1.0, 0.0, 90.0)],
)
def test_downslope_is_against_the_gradient(dzdx: float, dzdy: float, az: float) -> None:
    got, dip = terrain.slope_from_gradient(dzdx, dzdy)
    assert got == pytest.approx(az) and dip == pytest.approx(45.0)


def test_fit_plane_recovers_a_plane() -> None:
    rng = np.random.default_rng(1)
    x, y = rng.uniform(-50, 50, 400), rng.uniform(-50, 50, 400)
    z = 0.2 * x - 0.3 * y + 100.0
    a, b, c, rms = terrain.fit_plane(x, y, z)
    assert (a, b, c) == pytest.approx((0.2, -0.3, 100.0)) and rms < 1e-9


def test_slope_at_on_a_synthetic_dem(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """A plane dipping 20° toward 300° (ISN93 grid around the point)."""
    import pygmt

    dem = tmp_path / terrain.DEM_5M
    dem.parent.mkdir(parents=True)
    dem.write_bytes(b"")
    az, dip = np.radians(300.0), np.tan(np.radians(20.0))
    gx, gy = -dip * np.sin(az), -dip * np.cos(az)  # gradient = uphill

    def grdcut(grid: str, region: list[float]) -> xr.DataArray:
        xs = np.arange(region[0], region[1] + 1, 5.0)
        ys = np.arange(region[2], region[3] + 1, 5.0)
        x0, y0 = (region[0] + region[1]) / 2, (region[2] + region[3]) / 2
        z = 500.0 + gx * (xs[None, :] - x0) + gy * (ys[:, None] - y0)
        return xr.DataArray(z, coords={"y": ys, "x": xs}, dims=("y", "x"))

    monkeypatch.setattr(pygmt, "grdcut", grdcut)
    sl = terrain.slope_at(-16.8, 64.0, str(tmp_path))
    assert sl.azimuth == pytest.approx(300.0) and sl.dip == pytest.approx(20.0)
    assert sl.height == pytest.approx(500.0) and sl.rms < 1e-6
    with pytest.raises(FileNotFoundError):
        terrain.slope_at(-16.8, 64.0, str(tmp_path / "none"))


def test_direction_line_with_the_dem_slope() -> None:
    class _D:
        azimuth, sigma_azimuth, plunge, sigma_plunge = 334.2, 3.9, -5.9, 17.9
        horizontal, significant = 18.0, True

    sl = terrain.Slope(259.5, 21.3, 825.0, 50.0, 3.1)
    got = direction.line("SVIE-SKFC", _D(), "span", sl)
    assert got.startswith("SVIE-SKFC: {motion: 334.2, slope: [260, 21]}  # ±3.9°")
    assert "DEM slope over 50 m" in got


def test_pick_dem_prefers_a_2m_grid_that_covers(monkeypatch: pytest.MonkeyPatch,
                                                tmp_path) -> None:  # fmt: skip
    five = tmp_path / terrain.DEM_5M
    five.parent.mkdir(parents=True)
    five.write_bytes(b"")
    two = tmp_path / terrain.DEM_2M_DIR
    two.mkdir(parents=True)
    (two / "area.nc").write_bytes(b"")
    (two / "README.md").write_text("not a grid")  # skipped, never grdinfo'd

    def region(path: str) -> tuple[float, float, float, float]:
        assert path.endswith(".nc")
        return 0.0, 100.0, 0.0, 100.0

    monkeypatch.setattr(terrain, "_region", region)
    gis = str(tmp_path)
    assert terrain.pick_dem(gis, (10, 20, 10, 20))[1] == "2m"
    assert terrain.pick_dem(gis, (90, 120, 10, 20))[1] == "5m"  # not inside
    assert terrain.pick_dem(gis, (10, 20, 10, 20), "5m")[1] == "5m"
    with pytest.raises(FileNotFoundError):
        terrain.pick_dem(gis, (90, 120, 10, 20), "2m")
    with pytest.raises(ValueError):
        terrain.pick_dem(gis, (10, 20, 10, 20), "1m")
