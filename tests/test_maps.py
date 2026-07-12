"""Tests for the PyGMT map lane (gps_plot.maps).

Three layers, matching the optional-dependency design:

1. Config->coordinate plumbing (no GMT needed): a tmp gpsconfig fixture is
   fed through the real ``gps_parser.ConfigParser`` and the coordinate /
   label table :func:`gps_plot.maps.station_coordinates` builds from it is
   checked, plus region derivation and parameter-default hygiene (no
   hardcoded output path).
2. The lazy-import guard: importing ``gps_plot`` / ``gps_plot.maps`` works
   without pygmt; calling the map functions without it raises a helpful
   ImportError (simulated via monkeypatch so the test passes whether or
   not pygmt is actually installed).
3. Deformation-lane plumbing (no GMT needed): velocity-vector assembly
   from arrays / the gps_api velocities GeoJSON, model vector fields via
   gps_analysis forwards, and slip-patch corner geometry from a
   SlipDistributionResult-shaped product.
4. Env-gated integration tests that ACTUALLY render each map to a tmp
   PNG — skipped with a clear reason when pygmt or the GMT C library is
   unavailable (mirrors the opt-in gating style of gps_analysis' full
   TS14 verification test).
"""

from __future__ import annotations

import builtins
import importlib
import inspect
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from gps_plot.maps import (
    DEFAULT_MARGIN_DEG,
    SlipPatch,
    StationCoordinate,
    VectorField,
    VelocityVector,
    deformation_vectors,
    mogi_model_field,
    region_around,
    slip_map,
    slip_patches_from_product,
    station_coordinates,
    station_map,
    velocity_map,
    velocity_vectors,
    velocity_vectors_from_geojson,
)

# ---------------------------------------------------------------------------
# Fixture: a minimal deployed gpsconfig (stations.cfg + postprocess.cfg)
# ---------------------------------------------------------------------------

STATIONS_CFG = """\
[RHOF]
station_id = RHOF
station_name = Raufarhöfn
latitude = 66.461123
longitude = -15.946707 # inline comment, must be stripped
height = 78.764

[AKUR]
station_id = AKUR
station_name = Akureyri
latitude = 65.685309
longitude = -18.122529

[NOCOORD]
station_id = NOCOORD
station_name = Vantar hnit
"""


@pytest.fixture
def gpsconfig(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Point gps_parser at a tmp config dir with two known stations."""
    cfg_dir = tmp_path / "gpsconfig"
    cfg_dir.mkdir()
    (cfg_dir / "stations.cfg").write_text(STATIONS_CFG, encoding="utf-8")
    (cfg_dir / "postprocess.cfg").write_text("[Configs]\n", encoding="utf-8")
    monkeypatch.setenv("GPS_CONFIG_PATH", str(cfg_dir))
    gps_parser = pytest.importorskip(
        "gps_parser", reason="gps_parser (dev group) not installed"
    )
    return gps_parser.ConfigParser()


# ---------------------------------------------------------------------------
# 1. Config -> coordinates plumbing (no GMT needed)
# ---------------------------------------------------------------------------


def test_station_coordinates_from_config(gpsconfig) -> None:
    coords = station_coordinates(["RHOF", "AKUR"], parser=gpsconfig)
    assert [c.marker for c in coords] == ["RHOF", "AKUR"]  # order preserved
    rhof, akur = coords
    assert rhof.lat == pytest.approx(66.461123)
    # Inline "# ..." comment stripped by gps_parser before float().
    assert rhof.lon == pytest.approx(-15.946707)
    assert rhof.name == "Raufarhöfn"
    assert akur.name == "Akureyri"


def test_station_coordinates_unknown_marker(gpsconfig) -> None:
    with pytest.raises(KeyError, match="FAKE.*stations.cfg"):
        station_coordinates(["FAKE"], parser=gpsconfig)


def test_station_coordinates_missing_latlon(gpsconfig) -> None:
    with pytest.raises(ValueError, match="NOCOORD.*latitude/longitude"):
        station_coordinates(["NOCOORD"], parser=gpsconfig)


def test_region_around_bounding_box() -> None:
    coords = (
        StationCoordinate("A", lon=-20.0, lat=64.0),
        StationCoordinate("B", lon=-18.0, lat=66.0),
    )
    assert region_around(coords, margin=0.5) == (-20.5, -17.5, 63.5, 66.5)
    with pytest.raises(ValueError, match="zero stations"):
        region_around(())


def test_station_map_defaults_have_no_hardcoded_output() -> None:
    """Style/paths are parameters; no fixed output filename anywhere."""
    for func in (station_map, velocity_map, deformation_vectors, slip_map):
        sig = inspect.signature(func)
        assert sig.parameters["outfile"].default is None, func.__name__
        assert sig.parameters["region"].default is None, func.__name__
    sig = inspect.signature(station_map)
    assert sig.parameters["margin"].default == DEFAULT_MARGIN_DEG
    # The dormant gmtplot.velomap hardcoded "volcano_test.pdf" — the
    # absorbed module must not.
    source = inspect.getsource(importlib.import_module("gps_plot.maps"))
    assert "volcano_test" not in source
    assert "config.cfg" not in source


# ---------------------------------------------------------------------------
# 2. Lazy-import guard
# ---------------------------------------------------------------------------


def test_importing_gps_plot_needs_no_pygmt() -> None:
    """The package and the maps module import fine without pygmt."""
    import gps_plot
    import gps_plot.maps  # noqa: F401

    assert hasattr(gps_plot.maps, "station_map")


@pytest.fixture
def no_pygmt(monkeypatch: pytest.MonkeyPatch) -> None:
    """Simulate absent pygmt regardless of the real environment."""
    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object):
        if name == "pygmt" or name.startswith("pygmt."):
            raise ImportError("No module named 'pygmt'")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "pygmt", raising=False)
    monkeypatch.setattr(builtins, "__import__", fake_import)


_GUARD_COORDS = (StationCoordinate("RHOF", lon=-15.95, lat=66.46),)


def test_station_map_without_pygmt_raises_helpful_error(no_pygmt: None) -> None:
    """The guard maps a missing pygmt to an actionable ImportError."""
    with pytest.raises(ImportError, match="uv sync --extra maps"):
        station_map(_GUARD_COORDS)


def test_velocity_map_without_pygmt_raises_helpful_error(no_pygmt: None) -> None:
    vectors = (VelocityVector("RHOF", lon=-15.95, lat=66.46, east=5.0, north=2.0),)
    with pytest.raises(ImportError, match="uv sync --extra maps"):
        velocity_map(vectors)


def test_deformation_vectors_without_pygmt_raises_helpful_error(
    no_pygmt: None,
) -> None:
    with pytest.raises(ImportError, match="uv sync --extra maps"):
        deformation_vectors(_GUARD_COORDS, [5.0], [2.0])


def test_slip_map_without_pygmt_raises_helpful_error(no_pygmt: None) -> None:
    patch = SlipPatch(
        xs=(-22.4, -22.39, -22.39, -22.4), ys=(63.88, 63.88, 63.89, 63.89), value=0.5
    )
    with pytest.raises(ImportError, match="uv sync --extra maps"):
        slip_map([patch])


# ---------------------------------------------------------------------------
# 3. Deformation-lane plumbing (no GMT needed)
# ---------------------------------------------------------------------------

#: A gps_api GET /velocities product (VelocityCollection), two estimators.
VELOCITIES_GEOJSON = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [-15.946707, 66.461123, 78.764],
            },
            "properties": {
                "marker": "RHOF",
                "east": 2.5,
                "north": -1.0,
                "up": 0.5,
                "sigma_east": 0.4,
                "sigma_north": 0.3,
                "sigma_up": 1.0,
                "magnitude": 2.69,
                "azimuth": 111.8,
                "method": "wls",
                "window_start": "2020-01-01T00:00:00Z",
                "window_end": "2025-01-01T00:00:00Z",
            },
        },
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [-18.122529, 65.685309]},
            "properties": {
                "marker": "AKUR",
                "east": -3.0,
                "north": 4.0,
                "up": -0.2,
                "sigma_east": 1.6,  # honest colored-noise sigma: bigger ellipse
                "sigma_north": 1.2,
                "sigma_up": 3.0,
                "magnitude": 5.0,
                "azimuth": 323.1,
                "method": "mle",
                "window_start": "2020-01-01T00:00:00Z",
                "window_end": "2025-01-01T00:00:00Z",
            },
        },
    ],
}


def test_velocity_vectors_from_arrays(gpsconfig) -> None:
    """Arrays + stations.cfg positions -> velo-schema records."""
    vectors = velocity_vectors(
        ["RHOF", "AKUR"],
        east=[2.5, -3.0],
        north=[-1.0, 4.0],
        sigma_east=[0.4, 1.6],
        sigma_north=[0.3, 1.2],
        method="wls",
        parser=gpsconfig,
    )
    assert [v.marker for v in vectors] == ["RHOF", "AKUR"]
    rhof = vectors[0]
    assert rhof.lon == pytest.approx(-15.946707)  # from stations.cfg
    assert rhof.lat == pytest.approx(66.461123)
    assert rhof.east == pytest.approx(2.5)
    assert rhof.sigma_north == pytest.approx(0.3)
    assert rhof.correlation_en == 0.0  # per-component estimation default
    assert rhof.method == "wls"


def test_velocity_vectors_length_mismatch(gpsconfig) -> None:
    with pytest.raises(ValueError, match="north must have shape"):
        velocity_vectors(
            ["RHOF", "AKUR"], east=[1.0, 2.0], north=[1.0], parser=gpsconfig
        )


def test_velocity_vectors_from_geojson() -> None:
    """The gps_api velocities GeoJSON parses losslessly, order preserved."""
    vectors = velocity_vectors_from_geojson(VELOCITIES_GEOJSON)
    assert [v.marker for v in vectors] == ["RHOF", "AKUR"]
    rhof, akur = vectors
    assert rhof.lon == pytest.approx(-15.946707)  # [lon, lat(, height)]
    assert rhof.lat == pytest.approx(66.461123)
    assert rhof.method == "wls"
    assert akur.method == "mle"
    assert akur.sigma_east == pytest.approx(1.6)  # honest MLE sigma intact
    assert akur.correlation_en == 0.0  # not in VelocityProperties


def test_velocity_vectors_from_geojson_rejects_non_collection() -> None:
    with pytest.raises(ValueError, match="features"):
        velocity_vectors_from_geojson({"type": "Feature"})


def test_mogi_model_field_matches_gps_analysis_forward() -> None:
    """The field builder is a thin wrapper over gps_analysis.mogi_forward."""
    ga = pytest.importorskip(
        "gps_analysis", reason="gps_analysis (dev group) not installed"
    )
    lon = np.array([-22.43, -22.55])
    lat = np.array([63.88, 63.95])
    src_lon, src_lat, depth, dv = -22.47, 63.90, 3.7e3, 5.0e6
    field = mogi_model_field(
        lon,
        lat,
        source_lon=src_lon,
        source_lat=src_lat,
        depth=depth,
        dv=dv,
        name="ours",
        color="red",
    )
    e, n = ga.local_coordinates(lon, lat, src_lon, src_lat)
    enu = ga.mogi_forward(e, n, ga.MogiSource(x=0.0, y=0.0, depth=depth, dv=dv))
    np.testing.assert_allclose(np.asarray(field.east), enu[0] * 1e3, rtol=1e-12)
    np.testing.assert_allclose(np.asarray(field.north), enu[1] * 1e3, rtol=1e-12)
    assert field.source_lon == pytest.approx(src_lon)  # star marker position
    assert field.source_lat == pytest.approx(src_lat)
    assert field.name == "ours"


#: A SlipDistributionResult-shaped product (gps_api models/<region>_slip.json)
#: — 2 x 1 patches on a 60-degree-dipping plane, opening component.
def _slip_product() -> dict:
    origin_lon, origin_lat = -22.43, 63.88
    return {
        "region": "svartsengi",
        "source_type": "okada",
        "origin_lon": origin_lon,
        "origin_lat": origin_lat,
        "strike": 40.0,
        "dip": 60.0,
        "length_km": 4.0,
        "width_km": 2.0,
        "top_depth_km": 1.0,
        "n_strike": 2,
        "n_dip": 1,
        "components": ["opening"],
        "potency_m3": {"opening": 5.0e6},
        "patches": [
            {
                "index": 0,
                "row": 0,
                "col": 0,
                "lon": origin_lon - 0.012,
                "lat": origin_lat - 0.007,
                "east_m": -643.0,
                "north_m": -766.0,
                "depth_km": 1.87,
                "slip_m": {"opening": 0.8},
                "sigma_m": {"opening": 0.1},
            },
            {
                "index": 1,
                "row": 0,
                "col": 1,
                "lon": origin_lon + 0.012,
                "lat": origin_lat + 0.007,
                "east_m": 643.0,
                "north_m": 766.0,
                "depth_km": 1.87,
                "slip_m": {"opening": 0.3},
                "sigma_m": {"opening": 0.1},
            },
        ],
    }


def test_slip_patches_from_product_geometry() -> None:
    """Patch polygons: centroid preserved, spans match the plane tiling."""
    ga = pytest.importorskip(
        "gps_analysis", reason="gps_analysis (dev group) not installed"
    )
    product = _slip_product()
    patches, comp = slip_patches_from_product(product)
    assert comp == "opening"  # largest |potency| auto-selection
    assert len(patches) == 2
    assert [p.value for p in patches] == [0.8, 0.3]

    patch = patches[0]
    centroid = product["patches"][0]
    # Polygon centroid == the product's patch centroid (offsets cancel).
    assert sum(patch.xs) / 4 == pytest.approx(centroid["lon"])
    assert sum(patch.ys) / 4 == pytest.approx(centroid["lat"])
    # Corner spans, measured back in metres in the product-origin frame
    # (the frame whose metric the deg<->m conversion uses): along-strike
    # edge = L/n_strike, dip-direction edge = (W/n_dip) * cos(dip)
    # (surface projection).
    e, n = ga.local_coordinates(
        np.array(patch.xs),
        np.array(patch.ys),
        product["origin_lon"],
        product["origin_lat"],
    )
    along = math.hypot(e[1] - e[0], n[1] - n[0])
    across = math.hypot(e[2] - e[1], n[2] - n[1])
    assert along == pytest.approx(4.0e3 / 2, rel=1e-9)
    assert across == pytest.approx(2.0e3 * math.cos(math.radians(60.0)), rel=1e-9)


def test_slip_patches_unknown_component() -> None:
    with pytest.raises(ValueError, match="strike_slip.*not in"):
        slip_patches_from_product(_slip_product(), component="strike_slip")


def test_slip_map_rejects_bad_view() -> None:
    with pytest.raises(ValueError, match="view must be"):
        slip_map(_slip_product(), view="sideways")


def test_plane_patches_grid() -> None:
    """Plane view tiles the (along-strike, down-dip) km grid exactly."""
    from gps_plot.maps import _plane_patches

    patches = _plane_patches(_slip_product(), "opening")
    assert len(patches) == 2
    first, second = patches
    assert first.xs == (0.0, 2.0, 2.0, 0.0)  # col 0 of a 4 km / 2 plane
    assert first.ys == (0.0, 0.0, 2.0, 2.0)  # full 2 km down-dip extent
    assert second.xs == (2.0, 4.0, 4.0, 2.0)
    assert second.value == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# 4. Env-gated render integration tests
# ---------------------------------------------------------------------------


def _pygmt_unavailable_reason() -> str | None:
    """Import pygmt for real; return the skip reason when that fails.

    A missing GMT C library raises GMTCLibNotFoundError (not ImportError),
    so any exception counts as unavailable.
    """
    try:
        import pygmt  # noqa: F401
    except Exception as exc:  # noqa: BLE001 - see docstring
        return (
            f"pygmt/GMT unavailable ({type(exc).__name__}: {exc}); install "
            "the 'maps' extra and a system GMT >= 6 (set GMT_LIBRARY_PATH "
            "for a from-source build) to run the render test"
        )
    return None


_SKIP_REASON = _pygmt_unavailable_reason()


@pytest.mark.skipif(_SKIP_REASON is not None, reason=_SKIP_REASON or "")
def test_station_map_renders_png(gpsconfig, tmp_path: Path) -> None:
    """End-to-end: stations.cfg -> gps_parser -> PyGMT -> non-empty PNG."""
    out = tmp_path / "north_iceland_stations.png"
    fig = station_map(
        ["RHOF", "AKUR"],
        parser=gpsconfig,
        resolution="crude",  # keep the GSHHG load light for CI
        title="North Iceland test",
        outfile=out,
    )
    assert fig is not None
    assert out.is_file()
    assert os.path.getsize(out) > 1024  # a real raster, not an empty stub


@pytest.mark.skipif(_SKIP_REASON is not None, reason=_SKIP_REASON or "")
def test_velocity_map_renders_png(tmp_path: Path) -> None:
    """WLS + MLE vectors (GeoJSON product path) -> color-coded velo layers."""
    vectors = velocity_vectors_from_geojson(VELOCITIES_GEOJSON)
    out = tmp_path / "velocities.png"
    fig = velocity_map(
        vectors,
        resolution="crude",
        title="Velocity test",
        scale_ref=5.0,
        outfile=out,
    )
    assert fig is not None
    assert out.is_file()
    assert os.path.getsize(out) > 1024


@pytest.mark.skipif(_SKIP_REASON is not None, reason=_SKIP_REASON or "")
def test_deformation_vectors_renders_png(tmp_path: Path) -> None:
    """Observed + one model field (pre-computed, NaN row) -> overlay map."""
    stations = (
        StationCoordinate("STA1", lon=-22.55, lat=63.85),
        StationCoordinate("STA2", lon=-22.30, lat=63.95),
        StationCoordinate("STA3", lon=-22.45, lat=64.02),
    )
    observed_e = np.array([30.0, -12.0, np.nan])  # NaN = missing observation
    observed_n = np.array([-18.0, 25.0, np.nan])
    model = VectorField(
        name="Mogi test",
        east=np.array([27.0, -10.0, 4.0]),
        north=np.array([-16.0, 22.0, 8.0]),
        color="red",
        source_lon=-22.45,
        source_lat=63.92,
    )
    out = tmp_path / "deformation.png"
    fig = deformation_vectors(
        stations,
        observed_e,
        observed_n,
        [model],
        resolution="crude",
        title="Deformation test",
        labels=True,
        outfile=out,
    )
    assert fig is not None
    assert out.is_file()
    assert os.path.getsize(out) > 1024


@pytest.mark.skipif(_SKIP_REASON is not None, reason=_SKIP_REASON or "")
def test_slip_map_renders_png_map_view(tmp_path: Path) -> None:
    """SlipDistributionResult product -> colored patch polygons + colorbar."""
    pytest.importorskip("gps_analysis", reason="gps_analysis (dev group) needed")
    out = tmp_path / "slip_map.png"
    fig = slip_map(
        _slip_product(),
        resolution="crude",
        margin=0.2,
        title="Slip test",
        outfile=out,
    )
    assert fig is not None
    assert out.is_file()
    assert os.path.getsize(out) > 1024


@pytest.mark.skipif(_SKIP_REASON is not None, reason=_SKIP_REASON or "")
def test_slip_map_renders_png_plane_view(tmp_path: Path) -> None:
    """Fault-plane cross-section view (no coastline, km axes)."""
    out = tmp_path / "slip_plane.png"
    fig = slip_map(
        _slip_product(),
        view="plane",
        title="Slip plane test",
        outfile=out,
    )
    assert fig is not None
    assert out.is_file()
    assert os.path.getsize(out) > 1024
