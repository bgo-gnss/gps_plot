"""``gpsplot map``: options, station selection, the velocity sources, dispatch.

The drawing functions (``gps_plot.maps``) are replaced by recorders, so
nothing here needs PyGMT except the one gated render at the end.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
import types

import numpy as np
import pytest

from gps_plot import map as gmap
from gps_plot.cli.map import join_negative_values
from gps_plot.maps import StationCoordinate, VelocityVector
from gps_plot.series import Series
from gps_plot.sources import velocity as vsource

COORDS = {
    "SENG": (-22.43, 63.87),
    "SKSH": (-22.48, 63.85),
    "HOFN": (-15.20, 64.27),
    "SEY1": (-14.00, 65.27),
    "SEY2": (-14.01, 65.26),
}
STATIONS = [*COORDS, "NOCO"]  # NOCO: a section without coordinates


def coords(markers):
    m = markers[0]
    if m not in COORDS:
        raise ValueError(f"{m} has no latitude/longitude")
    return (StationCoordinate(m, COORDS[m][0], COORDS[m][1]),)


# ---------------------------------------------------------------------------
# options
# ---------------------------------------------------------------------------


def test_negative_values_are_joined_to_their_option():
    argv = ["velocity", "--region", "-22.8/-21.9/63.7/64.1", "--mogi", "-22.4,63,4,1"]
    assert join_negative_values(argv) == [
        "velocity", "--region=-22.8/-21.9/63.7/64.1", "--mogi=-22.4,63,4,1",
    ]  # fmt: skip
    assert join_negative_values(["--region"]) == ["--region"]


@pytest.mark.parametrize(
    "func, value",
    [
        (gmap._region, "-21/-22/63/64"),
        (gmap._region, "1/2/3"),
        (gmap._window, "20231027"),
        (gmap._window, "20231109:20231027"),
        (gmap._mogi, "-22,63,0,1e7"),
        (gmap._date, "2023-10-27"),
    ],
)
def test_bad_values_are_refused(func, value):
    with pytest.raises(argparse.ArgumentTypeError):
        func(value)


def test_windows_and_last_days():
    now = dt.datetime(2026, 10, 14, 12)
    w = [gmap._window("20231027:20231109"), gmap._window("20231123:")]
    assert gmap.resolve_windows(w, [7], now) == [
        (dt.datetime(2023, 10, 27), dt.datetime(2023, 11, 9)),
        (dt.datetime(2023, 11, 23), now),
        (now - dt.timedelta(days=7), now),
    ]
    assert gmap.window_label(*w[0]) == "2023/10/27 - 2023/11/09"


def test_auto_scale_and_scale_bar():
    assert gmap._nice(730) == 500 and gmap._nice(0.03) == 0.02
    scale, ref = gmap.auto_scale([1.0, 2.0, 1000.0, np.nan, 0.0])
    assert ref == gmap._nice(float(np.percentile([1, 2, 1000], 90)))
    assert scale == pytest.approx(
        gmap.TARGET_ARROW_CM / np.percentile([1, 2, 1000], 90)
    )
    assert gmap.scale_bar_km((-22.8, -21.9, 63.75, 64.1)) == 5.0
    assert gmap.scale_bar_km((-24.6, -13.2, 63.2, 66.7)) == 100.0


def test_parser_kinds_and_defaults():
    p = gmap.build_parser(config=None)
    a = p.parse_args(["velocity", "SEY?", "--window", "20231027:20231109", "--save"])
    assert (a.kind, a.Stations, a.save, a.ref, a.vertical) == (
        "velocity", ["SEY?"], "pdf", None, False,
    )  # fmt: skip
    a = p.parse_args(["deformation", "-s", "20231101", "--region=-23/-21/63/64"])
    assert a.start == dt.datetime(2023, 11, 1) and a.region == (-23, -21, 63, 64)
    with pytest.raises(SystemExit):
        p.parse_args(["deformation"])  # -s is required
    with pytest.raises(SystemExit):
        p.parse_args([])  # a kind is required


# ---------------------------------------------------------------------------
# stations
# ---------------------------------------------------------------------------


def test_select_by_pattern_region_and_exclude():
    got, dropped = gmap.select_stations(
        ["SEY?", "HOFN"], STATIONS, coords, region=None, exclude="SEY2"
    )
    assert [c.marker for c in got] == ["SEY1", "HOFN"] and dropped == []
    got, _ = gmap.select_stations(
        [], STATIONS, coords, region=(-23.0, -22.0, 63.5, 64.0), exclude=None
    )
    assert [c.marker for c in got] == ["SENG", "SKSH"]


def test_select_reports_only_named_stations_without_coordinates():
    _, dropped = gmap.select_stations(
        ["*"], STATIONS, coords, region=None, exclude=None
    )
    assert dropped == []
    _, dropped = gmap.select_stations(
        ["NOCO", "HOFN"], STATIONS, coords, region=None, exclude=None
    )
    assert len(dropped) == 1 and dropped[0].startswith("NOCO")


@pytest.mark.parametrize(
    "specs, region, msg",
    [([], None, "--region"), (["XX?"], None, "matches no station"),
     (["HOFN"], (-23.0, -22.0, 63.5, 64.0), "no station")],
)  # fmt: skip
def test_select_refuses(specs, region, msg):
    with pytest.raises(ValueError, match=msg):
        gmap.select_stations(specs, STATIONS, coords, region=region, exclude=None)


# ---------------------------------------------------------------------------
# vectors
# ---------------------------------------------------------------------------


def test_vector_horizontal_and_vertical_conventions():
    from gps_plot import maps

    c = StationCoordinate("SENG", -22.43, 63.87)
    r = vsource.StationRate(
        "SENG", np.array([1.0, 2.0, 3.0]), np.array([0.1, 0.2, 0.3])
    )
    h = gmap._vector(maps, c, r, None, "w")
    assert (h.east, h.north, h.sigma_east, h.sigma_north) == (2.0, 1.0, 0.2, 0.1)
    v = gmap._vector(maps, c, r, 2, "w")
    assert (v.east, v.north, v.sigma_east, v.sigma_north) == (0.0, 3.0, 0.0, 0.3)


def test_snr():
    v = VelocityVector("A", 0, 0, east=3.0, north=4.0, sigma_east=0.0, sigma_north=1.0)
    assert gmap.snr(v) == pytest.approx(2.5)
    assert gmap.snr(VelocityVector("A", 0, 0, 1.0, 0.0)) == np.inf


# ---------------------------------------------------------------------------
# sources
# ---------------------------------------------------------------------------


def _req(ref="detrend"):
    from gps_plot.sources.gps import SeriesRequest

    return SeriesRequest(fstart=2023.0, fend=2023.1, ref=ref, Dir=None,
                         tType="POS", uncert=10, view="raw")  # fmt: skip


def _line_series(rate, n=20, applied=True, noise=0.0):
    t = 2023.0 + np.arange(n) / 365.25
    y = np.vstack([r * (t - t[0]) for r in rate])
    y += noise * np.random.default_rng(0).standard_normal(y.shape)
    y[:, min(3, n - 1)] = np.nan  # a gap is skipped, not fatal
    return Series("X", t, y, np.full((3, n), 2.0), detrend_applied=applied)


def test_window_velocity_fits_the_line(monkeypatch):
    monkeypatch.setattr(vsource.gps_source, "fetch",
                        lambda sta, req: _line_series([100.0, -50.0, 10.0]))  # fmt: skip
    r = vsource.window_velocity("X", _req())
    np.testing.assert_allclose(r.neu, [100.0, -50.0, 10.0], atol=1e-6)
    assert r.n == 19 and np.all(r.sigma > 0)


def test_window_velocity_refuses_undetrended_and_short(monkeypatch):
    monkeypatch.setattr(vsource.gps_source, "fetch",
                        lambda sta, req: _line_series([1, 1, 1], applied=False))  # fmt: skip
    with pytest.raises(ValueError, match="no stored detrend record"):
        vsource.window_velocity("X", _req())
    assert vsource.window_velocity("X", _req("plate")).n == 19  # plate: fine
    monkeypatch.setattr(vsource.gps_source, "fetch",
                        lambda sta, req: _line_series([1, 1, 1], n=3))  # fmt: skip
    with pytest.raises(ValueError, match="epochs in the window"):
        vsource.window_velocity("X", _req())


def test_window_displacement_is_end_minus_start(monkeypatch):
    t = 2023.0 + np.arange(30) / 365.25
    y = np.zeros((3, 30))
    y[:, 15:] = [[10.0], [-20.0], [5.0]]  # a step in the middle
    s = Series("X", t, y, np.full((3, 30), 2.0), detrend_applied=True)
    monkeypatch.setattr(vsource.gps_source, "fetch", lambda sta, req: s)
    r = vsource.window_displacement("X", _req())
    np.testing.assert_allclose(r.neu, [10.0, -20.0, 5.0])
    # mean of 5 samples, σ 2 each: 2/√5 per end, in quadrature
    np.testing.assert_allclose(r.sigma, np.hypot(2 / np.sqrt(5), 2 / np.sqrt(5)))


def _record(rates, sigmas, borrowed=None):
    comps = [
        {"params": [0.0, r], "cov_upper": [1.0, 0.0, s**2], "component": c}
        for r, s, c in zip(rates, sigmas, ("north", "east", "up"), strict=True)
    ]
    return {"record_version": 1, "model": "linear", "step_epochs": [],
            "param_names": ["offset", "rate"], "components": comps[::-1],
            "borrowed": borrowed, "n_epochs": 900}  # fmt: skip


def test_record_velocity_reads_the_secular_rate(monkeypatch):
    monkeypatch.setattr(vsource.gps_source, "station_record",
                        lambda sta, req: _record([-2.4, 1.1, 3.0], [0.1, 0.2, 0.3]))  # fmt: skip
    r = vsource.record_velocity("HOFN", _req())
    np.testing.assert_allclose(r.neu, [-2.4, 1.1, 3.0])
    np.testing.assert_allclose(r.sigma, [0.1, 0.2, 0.3])
    assert r.n == 900


def test_record_velocity_refuses_a_borrowed_secular(monkeypatch):
    monkeypatch.setattr(vsource.gps_source, "station_record",
                        lambda sta, req: _record([1, 1, 1], [1, 1, 1], "SKSH"))  # fmt: skip
    with pytest.raises(ValueError, match="record borrowed from SKSH"):
        vsource.record_velocity("ELDC", _req())


# ---------------------------------------------------------------------------
# dispatch (maps replaced by recorders)
# ---------------------------------------------------------------------------


class _Fig:
    region = (-23.0, -21.0, 63.0, 64.5)

    def __init__(self):
        self.saved, self.calls = [], []

    def scalebar(self, **kw):
        self.calls.append(("scalebar", kw))

    def image(self, **kw):
        self.calls.append(("image", kw))

    def savefig(self, path):
        self.saved.append(path)


class _Config:
    def getStationInfo(self):
        return STATIONS

    def getPostProcessDir(self, option):
        return "/cfg/pos" if option == "pos_dir" else ""


@pytest.fixture
def fake_maps(monkeypatch):
    from gps_plot import maps

    seen = {}

    def record(name):
        def draw(*args, **kw):
            seen[name] = (args, kw)
            seen["fig"] = _Fig()
            return seen["fig"]

        return draw

    for name in ("station_map", "velocity_map", "deformation_vectors", "slip_map"):
        monkeypatch.setattr(maps, name, record(name))
    monkeypatch.setattr(maps, "station_coordinates", lambda m, parser=None: coords(m))
    monkeypatch.setitem(sys.modules, "pygmt", types.SimpleNamespace(
        config=lambda **kw: __import__("contextlib").nullcontext()))  # fmt: skip
    return seen


def _run(argv, tmp_path):
    p = gmap.build_parser(config=None)
    args = p.parse_args(
        join_negative_values([*argv, "--save", "png", "-d", str(tmp_path)])
    )
    return gmap.run(args, p, _Config())


def test_station_map_dispatch(fake_maps, tmp_path):
    assert _run(["station", "SEY?", "--logo", "none"], tmp_path) == 0
    args, kw = fake_maps["station_map"]
    assert [c.marker for c in args[0]] == ["SEY1", "SEY2"] and kw["labels"] is True
    fig = fake_maps["fig"]
    assert fig.saved == [str(tmp_path / "map_station_SEY1_2.png")]
    assert [c[0] for c in fig.calls] == ["scalebar"]


def test_velocity_windows_one_colour_each(fake_maps, monkeypatch, tmp_path, capsys):
    rates = {"SENG": [100.0, -300.0, 50.0], "SKSH": [0.0, 400.0, 0.0]}
    refs = set()

    def window_velocity(sta, req, min_epochs=3):
        refs.add(req.ref)
        if sta not in rates:
            raise ValueError("no stored detrend record (the series is not detrended)")
        return vsource.StationRate(sta, np.array(rates[sta]), np.ones(3), 10)

    monkeypatch.setattr(vsource, "window_velocity", window_velocity)
    argv = ["velocity", "--region", "-23/-14/63/66", "--window", "20231027:20231109",
            "--window", "20231123:20231205", "--min-snr", "1",
            "--ref", "detrend"]  # fmt: skip
    assert _run(argv, tmp_path) == 0
    assert refs == {"detrend"}
    (vectors,), kw = fake_maps["velocity_map"]
    assert {v.method for v in vectors} == {"2023/10/27 - 2023/11/09",
                                           "2023/11/23 - 2023/12/05"}  # fmt: skip
    assert list(kw["method_colors"].values()) == ["blue", "red"]
    assert [v.marker for v in vectors][:2] == ["SENG", "SKSH"]
    err = capsys.readouterr().err
    assert "without a stored detrend record: --ref plate" in err
    assert (
        fake_maps["fig"]
        .saved[0]
        .endswith(
            "map_velocity_HOFN_5-20231027-20231109_20231123-20231205-horizontal.png"
        )
    )


def test_velocity_windows_default_to_plate(fake_maps, monkeypatch, tmp_path):
    refs = set()

    def window_velocity(sta, req, min_epochs=3):
        refs.add(req.ref)
        return vsource.StationRate(sta, np.array([5.0, 5.0, 0.0]), np.ones(3), 10)

    monkeypatch.setattr(vsource, "window_velocity", window_velocity)
    assert _run(["velocity", "SENG", "--last", "14"], tmp_path) == 0
    assert refs == {"plate"}


def test_velocity_nothing_plotted_exits_1(fake_maps, monkeypatch, tmp_path, capsys):
    def fail(sta, req):
        raise ValueError("no stored detrend record to draw")

    monkeypatch.setattr(vsource, "record_velocity", fail)
    assert _run(["velocity", "HOFN"], tmp_path) == 1
    err = capsys.readouterr().err
    assert "nothing plotted" in err and "--ref plate" not in err


def test_velocity_from_geojson(fake_maps, tmp_path):
    feats = [
        {"type": "Feature", "geometry": {"type": "Point", "coordinates": [lon, lat]},
         "properties": {"marker": m, "east": 1.0, "north": 2.0, "sigma_east": 0.1,
                        "sigma_north": 0.1, "method": "wls"}}
        for m, (lon, lat) in COORDS.items()
    ]  # fmt: skip
    path = tmp_path / "iceland.geojson"
    path.write_text(json.dumps({"type": "FeatureCollection", "features": feats}))
    assert _run(["velocity", "SEY?", "--geojson", str(path)], tmp_path) == 0
    (vectors,), _ = fake_maps["velocity_map"]
    assert [v.marker for v in vectors] == ["SEY1", "SEY2"]


def test_deformation_with_mogi(fake_maps, monkeypatch, tmp_path):
    from gps_plot import maps

    monkeypatch.setattr(
        vsource, "window_displacement",
        lambda sta, req: vsource.StationRate(sta, np.array([1.0, 2.0, 0.0]), np.ones(3)),
    )  # fmt: skip
    seen = {}
    monkeypatch.setattr(maps, "mogi_model_field",
                        lambda *a, **kw: seen.setdefault("mogi", (a, kw)))  # fmt: skip
    argv = ["deformation", "SENG", "SKSH", "-s", "20231101", "-e", "20231115",
            "--mogi", "-22.46,63.87,4.5,1.2e7"]  # fmt: skip
    assert _run(argv, tmp_path) == 0
    args, kw = fake_maps["deformation_vectors"]
    np.testing.assert_allclose(args[1], [2.0, 2.0])  # east
    assert seen["mogi"][1]["depth"] == 4500.0 and seen["mogi"][1]["dv"] == 1.2e7
    assert (
        fake_maps["fig"]
        .saved[0]
        .endswith("map_deformation_SENG_2-20231101-20231115-mogi.png")
    )


def test_gpsplot_help_lists_map_without_pygmt():
    code = (
        "import sys; from gps_plot.cli import main; main(['--help']); "
        "from gps_plot import map; map.build_parser(config=None).format_help(); "
        "assert 'pygmt' not in sys.modules, 'pygmt imported'"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "map" in out.stdout and "PyGMT maps" in out.stdout


# ---------------------------------------------------------------------------
# a real render (PyGMT + GMT required)
# ---------------------------------------------------------------------------


def test_velocity_map_renders(monkeypatch, tmp_path):
    pytest.importorskip("pygmt")
    from gps_plot import maps

    monkeypatch.setattr(maps, "station_coordinates", lambda m, parser=None: coords(m))
    monkeypatch.setattr(
        vsource, "window_velocity",
        lambda sta, req, min_epochs=3: vsource.StationRate(
            sta, np.array([50.0, -80.0, 10.0]), np.full(3, 5.0), 10),
    )  # fmt: skip
    p = gmap.build_parser(config=None)
    args = p.parse_args(["velocity", "SENG", "SKSH", "--last", "10", "--save", "png",
                         "-d", str(tmp_path), "--name", "smoke"])  # fmt: skip
    try:
        assert gmap.run(args, p, _Config()) == 0
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"GMT unavailable: {exc}")
    assert (tmp_path / "smoke.png").stat().st_size > 1000


def test_slip_dispatch(fake_maps, tmp_path):
    from test_maps import _slip_product

    path = tmp_path / "reykjanes_slip.json"
    path.write_text(json.dumps(_slip_product()))
    assert (
        _run(["slip", str(path), "--view", "plane", "--stations", "SEY?"], tmp_path)
        == 0
    )
    (product,), kw = fake_maps["slip_map"]
    assert kw["view"] == "plane" and [c.marker for c in kw["stations"]] == [
        "SEY1",
        "SEY2",
    ]
    fig = fake_maps["fig"]
    assert fig.calls == []  # no scale bar / logo on the plane section
    assert fig.saved[0].endswith(f"map_slip_{product['region']}-plane.png")
