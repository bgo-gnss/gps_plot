"""The terrain at a station: the slope of the DEM around it.

For the slope axes (``--axes slope``, ``directions: {PAIR: {slope: [AZ,
DIP]}}``) and the downslope line of ``gpsplot polar``: the plane fitted
(least squares) to the 5 m Iceland DEM (ISN93, ``iceland_land_5m_isn93.tif``
under the GIS directory) within ``radius`` of the station gives the
downslope azimuth (clockwise from north) and the dip. The DEM is read
through GMT (``pygmt.grdcut``), one small window per station.
"""

from __future__ import annotations

import dataclasses
import os
from typing import Any

import numpy as np

#: the 5 m land DEM, under the GIS directory (gis_dir in postprocess.cfg)
DEM_5M = os.path.join("dem", "layers", "iceland_land_5m_isn93.tif")
#: ISN93 / Lambert 1993, the DEM's projection
ISN93 = "EPSG:3057"
#: the default radius [m] of the plane fitted around a station
SLOPE_RADIUS_M = 50.0


@dataclasses.dataclass(frozen=True)
class Slope:
    """The terrain plane around a point.

    Attributes:
        azimuth: Downslope direction [°, 0…360, clockwise from north].
        dip: Slope angle below the horizontal [°].
        height: The plane's height at the point [m, as the DEM].
        radius: The radius it was fitted over [m].
        rms: The DEM's scatter about the plane [m] (rough terrain is large).
    """

    azimuth: float
    dip: float
    height: float
    radius: float
    rms: float


def station_lonlat(sta: str, coord_file: str | None = None) -> tuple[float, float]:
    """(longitude, latitude) [°] of ``sta``'s a-priori position.

    Raises:
        ValueError: no a-priori coordinate for ``sta``.
    """
    import pyproj
    from geo_dataread.station_catalog import station_xyz

    xyz = station_xyz(sta, coord_file)
    if xyz is None:
        raise ValueError(f"{sta}: no a-priori coordinate")
    to_geo = pyproj.Transformer.from_crs("EPSG:4978", "EPSG:4326", always_xy=True)
    lon, lat, _ = to_geo.transform(float(xyz[0]), float(xyz[1]), float(xyz[2]))
    return float(lon), float(lat)


def fit_plane(x: Any, y: Any, z: Any) -> tuple[float, float, float, float]:
    """(dz/dx, dz/dy, z at x = y = 0, rms) of the least-squares plane."""
    a = np.column_stack([x, y, np.ones_like(x)])
    coef, *_ = np.linalg.lstsq(a, z, rcond=None)
    rms = float(np.sqrt(np.mean((a @ coef - z) ** 2)))
    return float(coef[0]), float(coef[1]), float(coef[2]), rms


def slope_from_gradient(dzdx: float, dzdy: float) -> tuple[float, float]:
    """(downslope azimuth, dip) [°] of a plane with east/north gradients."""
    azimuth = float(np.degrees(np.arctan2(-dzdx, -dzdy)) % 360.0)
    dip = float(np.degrees(np.arctan(np.hypot(dzdx, dzdy))))
    return azimuth, dip


def slope_at(
    lon: float, lat: float, gis_dir: str, *, radius: float = SLOPE_RADIUS_M
) -> Slope:
    """The DEM slope within ``radius`` [m] of (``lon``, ``lat``).

    Raises:
        FileNotFoundError: no DEM under ``gis_dir``.
        ValueError: fewer than ten DEM cells in the circle (off the DEM).
    """
    import pygmt
    import pyproj

    dem = os.path.join(os.path.expanduser(gis_dir), DEM_5M)
    if not os.path.isfile(dem):
        raise FileNotFoundError(f"no DEM at {dem}")
    to_isn = pyproj.Transformer.from_crs("EPSG:4326", ISN93, always_xy=True)
    x0, y0 = to_isn.transform(lon, lat)
    pad = radius + 10.0
    grid = pygmt.grdcut(grid=dem, region=[x0 - pad, x0 + pad, y0 - pad, y0 + pad])
    xs = np.asarray(grid.coords[grid.dims[1]], dtype=float) - x0
    ys = np.asarray(grid.coords[grid.dims[0]], dtype=float) - y0
    gx, gy = np.meshgrid(xs, ys)
    z = np.asarray(grid.values, dtype=float)
    ok = np.isfinite(z) & (np.hypot(gx, gy) <= radius)
    if ok.sum() < 10:
        raise ValueError(f"only {int(ok.sum())} DEM cells within {radius:g} m")
    dzdx, dzdy, height, rms = fit_plane(gx[ok], gy[ok], z[ok])
    azimuth, dip = slope_from_gradient(dzdx, dzdy)
    return Slope(azimuth, dip, height, radius, rms)


@dataclasses.dataclass(frozen=True)
class DemWindow:
    """A square of the DEM around a point: ``z`` (ny, nx) [m] at ``x``/``y``
    (ISN93 metres east/north of the point)."""

    x: Any
    y: Any
    z: Any


def dem_window(lon: float, lat: float, gis_dir: str, half: float) -> DemWindow:
    """The DEM within ``half`` [m] of (``lon``, ``lat``), in metres from it.

    Raises:
        FileNotFoundError: no DEM under ``gis_dir``.
    """
    import pygmt
    import pyproj

    dem = os.path.join(os.path.expanduser(gis_dir), DEM_5M)
    if not os.path.isfile(dem):
        raise FileNotFoundError(f"no DEM at {dem}")
    to_isn = pyproj.Transformer.from_crs("EPSG:4326", ISN93, always_xy=True)
    x0, y0 = to_isn.transform(lon, lat)
    grid = pygmt.grdcut(grid=dem, region=[x0 - half, x0 + half, y0 - half, y0 + half])
    xs = np.asarray(grid.coords[grid.dims[1]], dtype=float) - x0
    ys = np.asarray(grid.coords[grid.dims[0]], dtype=float) - y0
    return DemWindow(xs, ys, np.asarray(grid.values, dtype=float))


def hillshade(
    z: Any, dx: float, *, azimuth: float = 315.0, altitude: float = 45.0
) -> Any:
    """Shaded relief 0…1 of a north-up grid (rows south to north) of spacing ``dx``."""
    gy, gx = np.gradient(np.asarray(z, dtype=float), dx)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, -gy)  # downslope azimuth, clockwise from north
    az, alt = np.radians(azimuth), np.radians(altitude)
    shade = np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(
        az - aspect
    )
    return np.clip(shade, 0.0, 1.0)


def slope_field(z: Any, dx: float) -> tuple[Any, Any]:
    """(slope angle [°], downslope azimuth [°]) of every cell of a north-up
    grid (rows south to north) of spacing ``dx``."""
    gy, gx = np.gradient(np.asarray(z, dtype=float), dx)
    angle = np.degrees(np.arctan(np.hypot(gx, gy)))
    azimuth = np.degrees(np.arctan2(-gx, -gy)) % 360.0
    return angle, azimuth


def station_slope(sta: str, gis_dir: str, *, radius: float = SLOPE_RADIUS_M) -> Slope:
    """The DEM slope around ``sta``'s a-priori position (:func:`slope_at`)."""
    lon, lat = station_lonlat(sta)
    return slope_at(lon, lat, gis_dir, radius=radius)
