"""The terrain at a station: the slope of the DEM around it.

For the slope axes (``--axes slope``, ``directions: {PAIR: {slope: [AZ,
DIP]}}``) and the downslope line of ``gpsplot polar``: the plane fitted
(least squares) to the 5 m Iceland DEM (ISN93, ``iceland_land_5m_isn93.tif``
under the GIS directory, or a 2 m ArcticDEM area grid where one covers:
:func:`pick_dem`) within ``radius`` of the station gives the
downslope azimuth (clockwise from north) and the dip. The DEM is read
through GMT (``pygmt.grdcut``), one small window per station.
"""

from __future__ import annotations

import dataclasses
import functools
import os
from typing import Any

import numpy as np

#: the 5 m land DEM, under the GIS directory (gis_dir in postprocess.cfg)
DEM_5M = os.path.join("dem", "layers", "iceland_land_5m_isn93.tif")
#: 2 m grids of the monitoring areas (ArcticDEM v-1, LMÍ prerelease 2019,
#: geoid removed; ISN93), under the GIS directory: used where one holds the
#: whole window (``res="auto"``/``"2m"``)
DEM_2M_DIR = os.path.join("dem", "arcticdem_2m")
#: DEM choices: the finest that covers, or one
DEM_RES = ("auto", "2m", "5m")
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
        source: The DEM it was fitted on (``"2m"`` or ``"5m"``).
    """

    azimuth: float
    dip: float
    height: float
    radius: float
    rms: float
    source: str = "5m"


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


@functools.lru_cache(maxsize=64)
def _region(path: str) -> tuple[float, float, float, float]:
    import pygmt

    w, e, s, n = (float(v) for v in pygmt.grdinfo(path, per_column=True).split()[:4])
    return w, e, s, n


def pick_dem(
    gis_dir: str, region: tuple[float, float, float, float], res: str = "auto"
) -> tuple[str, str]:
    """(path, ``"2m"``/``"5m"``) of the DEM for an ISN93 ``region``.

    ``auto``: a 2 m area grid holding the whole region, else the 5 m DEM;
    ``2m`` insists on one; ``5m`` is the 5 m DEM.

    Raises:
        FileNotFoundError: no such DEM.
    """
    root = os.path.expanduser(gis_dir)
    if res not in DEM_RES:
        raise ValueError(f"DEM resolution {res!r}: one of {', '.join(DEM_RES)}")
    if res in ("auto", "2m"):
        folder = os.path.join(root, DEM_2M_DIR)
        for name in sorted(os.listdir(folder)) if os.path.isdir(folder) else ():
            path = os.path.join(folder, name)
            w, e, s, n = _region(path)
            if w <= region[0] and region[1] <= e and s <= region[2] and region[3] <= n:
                return path, "2m"
        if res == "2m":
            raise FileNotFoundError(f"no 2 m DEM under {folder} covers {region}")
    path = os.path.join(root, DEM_5M)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"no DEM at {path}")
    return path, "5m"


def slope_at(
    lon: float,
    lat: float,
    gis_dir: str,
    *,
    radius: float = SLOPE_RADIUS_M,
    res: str = "auto",
) -> Slope:
    """The DEM slope within ``radius`` [m] of (``lon``, ``lat``).

    Raises:
        FileNotFoundError: no DEM under ``gis_dir``.
        ValueError: fewer than ten DEM cells in the circle (off the DEM).
    """
    import pygmt
    import pyproj

    to_isn = pyproj.Transformer.from_crs("EPSG:4326", ISN93, always_xy=True)
    x0, y0 = to_isn.transform(lon, lat)
    pad = radius + 10.0
    region = (x0 - pad, x0 + pad, y0 - pad, y0 + pad)
    dem, source = pick_dem(gis_dir, region, res)
    grid = pygmt.grdcut(grid=dem, region=list(region))
    xs = np.asarray(grid.coords[grid.dims[1]], dtype=float) - x0
    ys = np.asarray(grid.coords[grid.dims[0]], dtype=float) - y0
    gx, gy = np.meshgrid(xs, ys)
    z = np.asarray(grid.values, dtype=float)
    ok = np.isfinite(z) & (np.hypot(gx, gy) <= radius)
    if ok.sum() < 10:
        raise ValueError(f"only {int(ok.sum())} DEM cells within {radius:g} m")
    dzdx, dzdy, height, rms = fit_plane(gx[ok], gy[ok], z[ok])
    azimuth, dip = slope_from_gradient(dzdx, dzdy)
    return Slope(azimuth, dip, height, radius, rms, source)


@dataclasses.dataclass(frozen=True)
class DemWindow:
    """A square of the DEM around a point: ``z`` (ny, nx) [m] at ``x``/``y``
    (ISN93 metres east/north of the point)."""

    x: Any
    y: Any
    z: Any
    source: str = "5m"


def dem_window(
    lon: float, lat: float, gis_dir: str, half: float, *, res: str = "auto"
) -> DemWindow:
    """The DEM within ``half`` [m] of (``lon``, ``lat``), in metres from it.

    Raises:
        FileNotFoundError: no DEM under ``gis_dir``.
    """
    import pygmt
    import pyproj

    to_isn = pyproj.Transformer.from_crs("EPSG:4326", ISN93, always_xy=True)
    x0, y0 = to_isn.transform(lon, lat)
    region = (x0 - half, x0 + half, y0 - half, y0 + half)
    dem, source = pick_dem(gis_dir, region, res)
    grid = pygmt.grdcut(grid=dem, region=list(region))
    xs = np.asarray(grid.coords[grid.dims[1]], dtype=float) - x0
    ys = np.asarray(grid.coords[grid.dims[0]], dtype=float) - y0
    return DemWindow(xs, ys, np.asarray(grid.values, dtype=float), source)


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


def station_slope(
    sta: str, gis_dir: str, *, radius: float = SLOPE_RADIUS_M, res: str = "auto"
) -> Slope:
    """The DEM slope around ``sta``'s a-priori position (:func:`slope_at`)."""
    lon, lat = station_lonlat(sta)
    return slope_at(lon, lat, gis_dir, radius=radius, res=res)
