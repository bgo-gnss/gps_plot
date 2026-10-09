"""Baseline plots: AAAA-BBBB through plotTime (both ends, one pipeline)."""

from __future__ import annotations

import datetime
import sys
import types

import matplotlib.colors
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


def test_save_creates_a_missing_output_directory(tmp_path, capsys) -> None:
    """-d to a directory that does not exist used to die in matplotlib."""
    import matplotlib.figure

    fig = matplotlib.figure.Figure()
    fig.add_subplot().plot([0, 1])
    target = tmp_path / "not" / "there" / "plot"
    tplt.saveFig(str(target), "png", fig)
    assert (tmp_path / "not" / "there" / "plot.png").is_file()
    assert "created output directory" in capsys.readouterr().out


def test_show_ends_draws_both_stations_behind_the_baseline(
    monkeypatch, tmp_path
) -> None:
    n = 30
    ramp = np.arange(n, dtype=float)
    a = np.vstack([3 * ramp + 7, ramp, ramp])
    b = np.vstack([ramp + 2, ramp, ramp])
    _stub(monkeypatch, {"AAAA": a, "BBBB": b}, n)
    _, saved = _capture(monkeypatch)
    fig = tplt.plotTime(
        "AAAA-BBBB",
        ref="itrf2008",
        show_ends=True,
        save="png",
        figDir=str(tmp_path),
        logo=False,
        special="90d",
    )
    greys = {
        c: [
            ln
            for ln in fig.axes[0].get_lines()
            if ln.get_markerfacecolor() == c and ln.get_marker() == "o"
        ]
        for c in tplt.BASELINE_END_COLORS
    }
    a_line, b_line = (
        greys[tplt.BASELINE_END_COLORS[0]][0],
        greys[tplt.BASELINE_END_COLORS[1]][0],
    )
    # both ends start at 0 at the baseline's first epoch, so A − B is the baseline
    np.testing.assert_allclose(a_line.get_ydata(), 3 * ramp)
    np.testing.assert_allclose(b_line.get_ydata(), ramp)
    assert a_line.get_zorder() < 2  # behind the red baseline points
    labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert labels[1:] == ["AAAA", "BBBB"]
    assert saved == [str(tmp_path / "AAAA_BBBB-baseline-itrf2008-ends-90d")]


def test_show_ends_on_a_single_station_warns_and_is_ignored(
    monkeypatch, tmp_path
) -> None:
    _stub(monkeypatch, {"AAAA": np.zeros((3, 30))})
    _, saved = _capture(monkeypatch)
    with pytest.warns(UserWarning, match="baselines"):
        tplt.plotTime(
            "AAAA",
            ref="itrf2008",
            show_ends=True,
            save="png",
            figDir=str(tmp_path),
            logo=False,
            special="90d",
        )
    assert saved == [str(tmp_path / "AAAA-itrf2008-90d")]


def test_empirical_sigma_on_the_baseline_formal_sigma_on_the_ends(
    monkeypatch, tmp_path
) -> None:
    """ρ/k reach the baseline σ; the grey ends keep their formal error bars."""
    n = 30
    ramp = np.arange(n, dtype=float)
    _stub(monkeypatch, {"AAAA": np.vstack([ramp] * 3), "BBBB": np.vstack([ramp] * 3)}, n)
    seen, _ = _capture(monkeypatch)
    fig = tplt.plotTime(
        "AAAA-BBBB",
        ref="itrf2008",
        show_ends=True,
        save="png",
        figDir=str(tmp_path),
        logo=False,
        special="90d",
        baseline_rho=[0.75, 0.0, 1.0],
        baseline_sigma_scale=([1.0, 1.0, 1.0], [1.0, 1.0, 1.0]),
    )
    np.testing.assert_allclose(seen["Ddata"][0], 2.0 * np.sqrt(0.5))  # √(4+4−2·.75·4)
    np.testing.assert_allclose(seen["Ddata"][1], np.sqrt(8.0))  # ρ = 0: quadrature
    np.testing.assert_allclose(seen["Ddata"][2], 0.0)
    end_bars = [
        c
        for c in fig.axes[0].collections
        if tuple(np.round(c.get_colors()[0][:3], 3))
        in {tuple(np.round(matplotlib.colors.to_rgb(g), 3)) for g in tplt.BASELINE_END_COLORS}
    ]
    assert len(end_bars) == 2  # one formal error-bar set per end
    seg = end_bars[0].get_segments()[0]
    assert abs(seg[1][1] - seg[0][1]) == pytest.approx(4.0)  # ±2 formal σ
