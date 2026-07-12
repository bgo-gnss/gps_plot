"""Tests for the PyGMT map lane (gps_plot.maps).

Three layers, matching the optional-dependency design:

1. Config->coordinate plumbing (no GMT needed): a tmp gpsconfig fixture is
   fed through the real ``gps_parser.ConfigParser`` and the coordinate /
   label table :func:`gps_plot.maps.station_coordinates` builds from it is
   checked, plus region derivation and parameter-default hygiene (no
   hardcoded output path).
2. The lazy-import guard: importing ``gps_plot`` / ``gps_plot.maps`` works
   without pygmt; calling :func:`station_map` without it raises a helpful
   ImportError (simulated via monkeypatch so the test passes whether or
   not pygmt is actually installed).
3. One env-gated integration test that ACTUALLY renders a small station
   map to a tmp PNG — skipped with a clear reason when pygmt or the GMT C
   library is unavailable (mirrors the opt-in gating style of
   gps_analysis' full TS14 verification test).
"""

from __future__ import annotations

import builtins
import importlib
import inspect
import os
import sys
from pathlib import Path

import pytest

from gps_plot.maps import (
    DEFAULT_MARGIN_DEG,
    StationCoordinate,
    region_around,
    station_coordinates,
    station_map,
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
    sig = inspect.signature(station_map)
    assert sig.parameters["outfile"].default is None
    assert sig.parameters["region"].default is None  # derived, not baked in
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


def test_station_map_without_pygmt_raises_helpful_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Simulate absent pygmt: the guard maps it to an actionable ImportError."""
    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object):
        if name == "pygmt" or name.startswith("pygmt."):
            raise ImportError("No module named 'pygmt'")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "pygmt", raising=False)
    monkeypatch.setattr(builtins, "__import__", fake_import)
    coords = (StationCoordinate("RHOF", lon=-15.95, lat=66.46),)
    with pytest.raises(ImportError, match="uv sync --extra maps"):
        station_map(coords)


# ---------------------------------------------------------------------------
# 3. Env-gated render integration test
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
