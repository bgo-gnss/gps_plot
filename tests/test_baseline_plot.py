"""Baseline plots: AAAA-BBBB through plotTime (both ends, one pipeline)."""

from __future__ import annotations

import datetime
import sys
import types

import numpy as np
import pytest
from geo_dataread.gps_views import baseline_arrays

import gps_plot.timesmatplt as tplt


def _stub(monkeypatch, series: dict[str, np.ndarray], n: int = 30):
    """geo_dataread stub: getData returns per-station N/E/U, gps_views has
    the REAL baseline_arrays (the function under test lives there)."""
    end = datetime.datetime.now() - datetime.timedelta(days=1)
    yearf = np.linspace(2025.0, 2025.0 + n / 365.25, n)
    x = [end - datetime.timedelta(days=n - 1 - i) for i in range(n)]
    gps_read = types.ModuleType("geo_dataread.gps_read")
    gps_read.getData = lambda sta, **kw: (
        yearf,
        series[sta].copy(),
        np.full((3, n), 2.0),
        0.0,
    )
    gps_read.toDateTime = lambda yf: x[: len(yf)]
    gps_views = types.ModuleType("geo_dataread.gps_views")
    gps_views.baseline_arrays = baseline_arrays
    package = types.ModuleType("geo_dataread")
    package.gps_read, package.gps_views = gps_read, gps_views
    monkeypatch.setitem(sys.modules, "geo_dataread", package)
    monkeypatch.setitem(sys.modules, "geo_dataread.gps_read", gps_read)
    monkeypatch.setitem(sys.modules, "geo_dataread.gps_views", gps_views)
    return n


def _capture(monkeypatch):
    seen: dict[str, object] = {}
    real = tplt.stdTimesPlot

    def spy(x, data, Ddata, **kw):
        seen["data"], seen["Ddata"], seen["title"] = data, Ddata, kw["Title"]
        return real(x, data, Ddata, **kw)

    monkeypatch.setattr(tplt, "stdTimesPlot", spy)
    saved: list[str] = []
    monkeypatch.setattr(tplt, "saveFig", lambda fn, ft, fig, **kw: saved.append(fn))
    return seen, saved


def test_split_baseline_and_file_stem() -> None:
    assert tplt.split_baseline("VFLN-VFLS") == ("VFLN", "VFLS")
    assert tplt.split_baseline("vfln-vfls") == ("VFLN", "VFLS")
    assert tplt.split_baseline("VFLN") is None
    assert tplt.split_baseline("VFLN-VFLS-X") is None
    # a hyphenated stem would be read as station VFLN, frame VFLS
    assert tplt.baseline_file_stem("VFLN-VFLS") == "VFLN_VFLS-baseline"
    assert tplt.baseline_file_stem("SENG") == "SENG"


def test_baseline_plots_a_minus_b_with_combined_sigma(monkeypatch, tmp_path) -> None:
    n = 30
    ramp = np.arange(n, dtype=float)
    a = np.vstack([3 * ramp, ramp, 5.0 + 0 * ramp])
    b = np.vstack([ramp, ramp, 2.0 + 0 * ramp])
    _stub(monkeypatch, {"AAAA": a, "BBBB": b}, n)
    seen, saved = _capture(monkeypatch)
    tplt.plotTime(
        "AAAA-BBBB",
        ref="itrf2008",
        save="png",
        figDir=str(tmp_path),
        logo=False,
        special="90d",
    )
    np.testing.assert_allclose(seen["data"][0], 2 * ramp)  # (3-1)·t
    np.testing.assert_allclose(seen["data"][1], 0.0)
    np.testing.assert_allclose(seen["data"][2], 0.0)  # constant offset re-zeroed
    np.testing.assert_allclose(seen["Ddata"], np.sqrt(8.0))
    assert "Baseline" in seen["title"].main and "BBBB" in seen["title"].main
    assert saved == [str(tmp_path / "AAAA_BBBB-baseline-itrf2008-90d")]


def test_baseline_defaults_to_plate_and_names_two_plates(monkeypatch, tmp_path) -> None:
    n = _stub(monkeypatch, {"AAAA": np.zeros((3, 30)), "BBBB": np.zeros((3, 30))})
    import geofunc.geofunc as gf

    monkeypatch.setattr(gf, "plateDict", lambda: {"AAAA": "NA", "BBBB": "EU"})
    monkeypatch.setattr(
        gf, "plateFullname", lambda p: {"NA": "North American", "EU": "Eurasian"}[p]
    )
    seen, saved = _capture(monkeypatch)
    fig = tplt.plotTime(
        "AAAA-BBBB",
        ref=None,
        save="png",
        figDir=str(tmp_path),
        logo=False,
        special="90d",
    )
    assert fig._gps_ref == "plate"
    assert "AAAA: North American / BBBB: Eurasian" in seen["title"].main
    assert saved[0].endswith("AAAA_BBBB-baseline-plate-90d")
    assert n == 30


def test_single_station_default_is_unchanged(monkeypatch, tmp_path) -> None:
    _stub(monkeypatch, {"AAAA": np.zeros((3, 30))})
    _, saved = _capture(monkeypatch)
    fig = tplt.plotTime(
        "AAAA", ref=None, save="png", figDir=str(tmp_path), logo=False, special="90d"
    )
    assert fig._gps_ref == "itrf2008"
    assert saved == [str(tmp_path / "AAAA-itrf2008-90d")]


def test_detrended_baseline_refuses_an_end_without_a_record(monkeypatch) -> None:
    def fake(sta, **kw):
        z = np.zeros((3, 5))
        return tplt.StationSeries(
            np.linspace(2025, 2025.01, 5),
            z,
            z + 1,
            None,
            None,
            None,
            detrend_applied=(sta == "AAAA"),
        )

    _stub(monkeypatch, {})
    monkeypatch.setattr(tplt, "station_series", fake)
    with pytest.raises(ValueError, match="BBBB has none"):
        tplt.plotTime(
            "AAAA-BBBB", ref="itrf2008", view="detrended", save="png", logo=False
        )


def test_detrended_view_is_accepted_and_tagged(monkeypatch, tmp_path) -> None:
    """--view detrended was offered by the CLI but refused by plotTime."""

    def fake(sta, **kw):
        z = np.zeros((3, 30))
        return tplt.StationSeries(
            np.linspace(2025, 2025.08, 30), z, z + 1, None, None, None, True
        )

    _stub(monkeypatch, {})
    monkeypatch.setattr(tplt, "station_series", fake)
    seen, saved = _capture(monkeypatch)
    tplt.plotTime(
        "AAAA",
        ref="itrf2008",
        view="detrended",
        save="png",
        figDir=str(tmp_path),
        logo=False,
        special="90d",
    )
    assert saved == [str(tmp_path / "AAAA-itrf2008-detrended-90d")]
    assert "detrended" in seen["title"].main
