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
        (gmap._region, "-25/66.7/-12.8/63+r"),
        (gmap._window, "20231027"),
        (gmap._window, "20231109:20231027"),
        (gmap._mogi, "-22,63,0,1e7"),
        (gmap._date, "2023-10-27"),
    ],
)
def test_bad_values_are_refused(func, value):
    with pytest.raises(argparse.ArgumentTypeError):
        func(value)


def test_corner_region_is_parsed_as_corners():
    """``LON/LAT/LON/LAT+r`` (or GMT4's trailing ``r``) gives corners,
    lower-left first; W/E/S/N stays a plain box."""
    for value in ("-25/63/-12.8/66.7+r", "-25/63/-12.8/66.7r"):
        got = gmap._region(value)
        assert isinstance(got, gmap.Corners) and got == (-25, 63, -12.8, 66.7)
    plain = gmap._region("-24.6/-13.2/63.2/66.7")
    assert not isinstance(plain, gmap.Corners)


def test_dem_zone_spec_finds_the_file_and_its_cmap(tmp_path):
    (tmp_path / "vector").mkdir()
    poly = tmp_path / "vector" / "swarms.xy"
    poly.write_text("0 0\n1 0\n1 1\n0 0\n")
    gis = str(tmp_path)
    assert gmap.zone_spec("vector/swarms.xy", gis) == (str(poly), "119/136/153")
    assert gmap.zone_spec(f"{poly}:lightslategrey", gis) == (
        str(poly),
        "lightslategrey",
    )
    with pytest.raises(ValueError, match="no polygon file"):
        gmap.zone_spec("vector/none.xy:red", gis)


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
    # the insets project with real GMT: recorded here, drawn in test_maps
    monkeypatch.setattr(
        gmap, "_insets", lambda *a, **kw: seen.setdefault("insets", []).append(a)
    )
    monkeypatch.setitem(sys.modules, "pygmt", types.SimpleNamespace(
        config=lambda **kw: __import__("contextlib").nullcontext()))  # fmt: skip
    monkeypatch.setitem(sys.modules, "pygmt.params", types.SimpleNamespace(
        Position=lambda *a, **kw: (a, kw)))  # fmt: skip
    return seen


def _run(argv, tmp_path):
    p = gmap.build_parser(config=None)
    args = p.parse_args(
        join_negative_values([*argv, "--save", "png", "-d", str(tmp_path)])
    )
    return gmap.run(args, p, _Config())


def test_default_logo_is_the_two_line_logo_30_percent_transparent(fake_maps, tmp_path):
    assert _run(["station", "SEY?"], tmp_path) == 0
    kw = dict(fake_maps["fig"].calls)["image"]
    assert kw["imagefile"].endswith("VI_Two_Line_Blue.png")
    assert kw["transparency"] == gmap.LOGO_TRANSPARENCY
    assert "box" not in kw


def test_station_map_dispatch(fake_maps, tmp_path):
    assert _run(["station", "SEY?", "--logo", "none"], tmp_path) == 0
    args, kw = fake_maps["station_map"]
    assert [c.marker for c in args[0]] == ["SEY1", "SEY2"] and kw["labels"] is True
    fig = fake_maps["fig"]
    assert fig.saved == [
        str(tmp_path / "map_station_14.51W-13.5W_64.76N-65.77N_SEY.png")
    ]
    assert [c[0] for c in fig.calls] == ["scalebar"]


def _product_record(sta, rate, *, refused=None, flags=(), ref="plate-periodic"):
    """A velocity product record as gps-velocities would return it."""
    from geo_dataread.velocity import VelocityRecord

    return VelocityRecord(
        station=sta, lon=0.0, lat=0.0, frame="itrf2014", plate="NOAM",
        ref="plate" if "fallback:plate" in flags else ref, ref_requested=ref,
        series="imo.24hr_itrf14", view="cleaned", uncert=10.0,
        window=(2023.8, 2023.9), span_used=(2023.8, 2023.9), n_epochs=(30,) * 3,
        largest_gap=0.003, method="wls", model="linear",
        rate=tuple(rate) if refused is None else (np.nan,) * 3,
        sigma=(1.0, 1.0, 1.0), sigma_formal=(0.2, 0.2, 0.2),
        sigma_method="empirical_k", k=(5.0, 5.0, 5.0), k_source={}, corr_en=0.0,
        record=None, steps_in_window=(), outlier_abort=False, n_flagged=0,
        flags=tuple(flags), refused=refused, software={}, input_fingerprint="x",
    )  # fmt: skip


@pytest.fixture
def fake_product(monkeypatch):
    """geo_dataread.velocity.estimate_velocities replaced: SENG and SKSH
    have rates, SKSH only on plate (no record), every other station is
    refused. Records each call's (stations, start, end, settings)."""
    from geo_dataread import velocity as vprod

    rates = {"SENG": [100.0, -300.0, 50.0], "SKSH": [0.0, 400.0, 0.0]}
    calls = []

    def estimate(stations, start, end, settings, progress=None):
        calls.append((list(stations), start, end, settings))
        out = []
        for sta in sorted(stations):
            if sta not in rates:
                out.append(_product_record(sta, [0, 0, 0], refused="cover=0.10<0.8"))
            else:
                flags = ("fallback:plate", "background:none") if sta == "SKSH" else ()
                out.append(_product_record(sta, rates[sta], flags=flags,
                                           ref=settings.ref))  # fmt: skip
        return out

    monkeypatch.setattr(vprod, "estimate_velocities", estimate)
    return calls


def test_velocity_windows_one_colour_each(fake_maps, fake_product, tmp_path, capsys):
    argv = ["velocity", "--region", "-23/-14/63/66", "--window", "20231027:20231109",
            "--window", "20231123:20231205", "--min-snr", "1",
            "--ref", "detrend"]  # fmt: skip
    assert _run(argv, tmp_path) == 0
    assert {c[3].ref for c in fake_product} == {"background"}  # detrend = background
    assert fake_product[0][1:3] == (dt.date(2023, 10, 27), dt.date(2023, 11, 9))
    (vectors,), kw = fake_maps["velocity_map"]
    assert {v.method for v in vectors} == {
        "2023/10/27 - 2023/11/09", "2023/10/27 - 2023/11/09 (plate only)",
        "2023/11/23 - 2023/12/05", "2023/11/23 - 2023/12/05 (plate only)",
    }  # fmt: skip
    # each window its colour; its plate-only fallbacks the same colour, pale
    assert list(kw["method_colors"].values()) == ["blue", "blue@55", "red", "red@55"]
    seng = next(v for v in vectors if v.marker == "SENG")
    assert (seng.north, seng.east, seng.sigma_east) == (100.0, -300.0, 1.0)
    err = capsys.readouterr().err
    assert "cover=0.10<0.8" in err  # refusals are named
    assert (
        fake_maps["fig"]
        .saved[0]
        .endswith(
            "map_velocity_iceland_20231027-20231109_20231123-20231205_horizontal.png"
        )
    )


def test_velocity_windows_default_to_plate_periodic_cleaned(
    fake_maps, fake_product, tmp_path
):
    assert _run(["velocity", "SENG", "--last", "14"], tmp_path) == 0
    (settings,) = {c[3] for c in fake_product}
    assert (settings.ref, settings.view, settings.frame) == (
        "plate-periodic", "cleaned", "itrf2014",
    )  # fmt: skip
    assert settings.min_obs == 5


def test_velocity_windows_refuse_tot_and_8h(fake_maps, fake_product, tmp_path):
    with pytest.raises(SystemExit):
        _run(["velocity", "SENG", "--last", "14", "--tType", "TOT"], tmp_path)
    with pytest.raises(SystemExit):
        _run(["velocity", "SENG", "--last", "14", "--session", "8h"], tmp_path)


def test_velocity_set_from_file_draws_the_same_vectors(fake_maps, tmp_path):
    from geo_dataread import velocity as vprod

    recs = [_product_record("SENG", [1.0, 2.0, 3.0]),
            _product_record("SKSH", [0, 0, 0], refused="no_record")]  # fmt: skip
    paths = vprod.write_velocity_set(recs, tmp_path / "set", "s")
    assert _run(["velocity", "SENG", "SKSH", "--velocities", str(paths["csv"])],
                tmp_path) == 0  # fmt: skip
    (vectors,), _kw = fake_maps["velocity_map"]
    assert [(v.marker, v.north, v.east) for v in vectors] == [("SENG", 1.0, 2.0)]


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
        .endswith(
            "map_deformation_22.98W-21.93W_63.35N-64.37N_SENG-SKSH_20231101-20231115_mogi.png"
        )
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
    from geo_dataread import velocity as vprod

    monkeypatch.setattr(
        vprod, "estimate_velocities",
        lambda stations, start, end, settings, progress=None: [
            _product_record(s, [50.0, -80.0, 10.0]) for s in sorted(stations)],
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


def test_overlay_segments_polygons_holes_lines_and_crs():
    from gps_plot.maps import overlay_segments

    square = [[0, 0], [1, 0], [1, 1], [0, 0]]
    hole = [[0.2, 0.2], [0.4, 0.2], [0.4, 0.4], [0.2, 0.2]]
    fc = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [square, hole]}},
        {"type": "Feature", "geometry": {"type": "MultiPolygon", "coordinates": [[square]]}},
        {"type": "Feature", "geometry": {"type": "LineString", "coordinates": square}},
        {"type": "Feature", "geometry": {"type": "Point", "coordinates": [0, 0]}},
        {"type": "Feature", "geometry": None},
    ]}  # fmt: skip
    polys, lines = overlay_segments(fc)
    headers = [ln for ln in polys.splitlines() if ln.startswith(">")]
    assert headers == [">", "> -Ph", ">"]
    assert lines.splitlines()[0] == ">" and len(lines.splitlines()) == 5
    assert polys.splitlines()[1] == "0.0000000 0.0000000"
    ok = {**fc, "crs": {"type": "name",
                        "properties": {"name": "urn:ogc:def:crs:EPSG::4326"}}}  # fmt: skip
    assert overlay_segments(ok)[0] == polys
    isn93 = {**fc, "crs": {"type": "name", "properties": {"name": "EPSG:3057"}}}
    with pytest.raises(ValueError, match="WGS84"):
        overlay_segments(isn93)


def test_overlay_options_reach_the_map(fake_maps, tmp_path):
    from gps_plot.maps import MapOverlay

    argv = ["station", "SEY?", "--logo", "none", "--overlay", "a.geojson",
            "--overlay", "b.geojson", "--overlay-fill", "none"]  # fmt: skip
    assert _run(argv, tmp_path) == 0
    _args, kw = fake_maps["station_map"]
    assert kw["overlays"] == (
        MapOverlay("a.geojson", fill=None),
        MapOverlay("b.geojson", fill=None),
    )
    assert _run(["station", "SEY?", "--logo", "none"], tmp_path) == 0
    assert "overlays" not in fake_maps["station_map"][1]


def _tos_fleet():
    from gps_plot.sources.tos import TosStation

    return (
        TosStation("SENG", -22.43, 63.87, "Svartsengi", "active"),
        TosStation("S001", -22.70, 63.83, "Reykjanes", "campaign"),
        TosStation("FAFC", -22.30, 63.90, "Fagradalsfjall", "ended", "2023-07-12"),
        TosStation("HOFN", -15.20, 64.27, "Höfn", "active"),
    )


def test_station_tos_carries_status_and_reaches_stations_cfg_gaps(
    fake_maps, monkeypatch, tmp_path
):
    from gps_plot.sources import tos

    seen = {}

    def fleet(*, refresh=False):
        seen["refresh"] = refresh
        return _tos_fleet()

    monkeypatch.setattr(tos, "tos_gps_stations", fleet)
    argv = ["station", "--tos", "--region", "-23/-22/63.5/64", "--logo", "none"]
    assert _run([*argv, "--refresh-tos"], tmp_path) == 0
    (stations,), kw = fake_maps["station_map"]
    # S001 is not in stations.cfg; the TOS status rides on the coordinate
    assert [(c.marker, c.status) for c in stations] == [
        ("FAFC", "ended"),
        ("S001", "campaign"),
        ("SENG", "active"),
    ]
    assert seen["refresh"] is True and kw["labels"] is True
    assert fake_maps["fig"].saved == [
        str(tmp_path / "map_station_23W-22W_63.5N-64N_tos.png")
    ]
    # no region and no pattern: the whole fleet
    assert _run(["station", "--tos", "--logo", "none"], tmp_path) == 0
    assert len(fake_maps["station_map"][0][0]) == 4


def test_station_labels_off_above_the_limit(fake_maps, monkeypatch, tmp_path):
    from gps_plot.sources import tos
    from gps_plot.sources.tos import TosStation

    many = tuple(
        TosStation(f"S{i:03d}", -20 + i * 0.01, 64.0, None, "active")
        for i in range(gmap.LABEL_LIMIT + 1)
    )
    monkeypatch.setattr(tos, "tos_gps_stations", lambda *, refresh=False: many)
    assert _run(["station", "--tos", "--logo", "none"], tmp_path) == 0
    assert fake_maps["station_map"][1]["labels"] is False
    assert _run(["station", "--tos", "--labels", "--logo", "none"], tmp_path) == 0
    assert fake_maps["station_map"][1]["labels"] is True


def test_dem_iceland_picks_a_level_and_the_coast(fake_maps, monkeypatch, tmp_path):
    from gps_plot import maps

    gis = tmp_path / "gis"
    (gis / maps.GIS_DEM).mkdir(parents=True)
    (gis / maps.GIS_VECTOR).mkdir()
    for name, _cell, _box in maps.ICELAND_DEM_LEVELS:
        (gis / maps.GIS_DEM / name).write_text("")
    (gis / maps.ICELAND_COAST).write_text("")
    (gis / maps.ICELAND_COAST_OVERVIEW).write_text("")
    monkeypatch.setattr(gmap, "DEFAULT_GIS_DIR", str(gis))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    argv = ["station", "SEY?", "--logo", "none", "--dem", "iceland"]
    assert _run([*argv, "--region", "-24.8/-13.2/63.2/66.7"], tmp_path) == 0
    kw = fake_maps["station_map"][1]
    assert kw["dem_grid"] == str(gis / "dem" / "iceland_topobathy_15s.nc")
    assert kw["coastline"] == str(gis / maps.ICELAND_COAST_OVERVIEW)
    assert kw["dem_cmap"] == "lightgrey"
    assert _run([*argv, "--region", "-14.2/-13.9/65.2/65.3", "--dem-cmap", "geo"],
                tmp_path) == 0  # fmt: skip
    kw = fake_maps["station_map"][1]
    assert kw["dem_grid"] == str(gis / "dem" / "iceland_topobathy_1s.nc")
    assert kw["coastline"] == str(gis / maps.ICELAND_COAST)
    assert kw["dem_cmap"] == "geo"
    assert "glaciers" not in kw  # not built in this DEM directory
    (gis / maps.ICELAND_GLACIERS).write_text("")
    assert _run([*argv, "--region", "-14.2/-13.9/65.2/65.3"], tmp_path) == 0
    assert fake_maps["station_map"][1]["glaciers"] == str(gis / maps.ICELAND_GLACIERS)
    assert _run([*argv, "--region", "-14.2/-13.9/65.2/65.3", "--glaciers", "none",
                 "--lakes", "my_lakes.gmt"], tmp_path) == 0  # fmt: skip
    kw = fake_maps["station_map"][1]
    assert "glaciers" not in kw and kw["lakes"] == "my_lakes.gmt"


def test_colour_map_options_reach_the_map(fake_maps, tmp_path):
    argv = ["station", "SEY?", "--logo", "none", "--dem", "grid.nc"]
    assert _run([*argv, "--bathymetry-cmap", "none", "--glacier-cmap", "batlow"],
                tmp_path) == 0  # fmt: skip
    kw = fake_maps["station_map"][1]
    assert kw["dem_cmap"] == "lightgrey"
    assert (kw["bathymetry_cmap"], kw["glacier_cmap"]) == ("none", "batlow")


def test_resolve_palettes_finds_yours_and_suggests_the_name_meant(tmp_path):
    pygmt = pytest.importorskip("pygmt")
    (tmp_path / "mine.cpt").write_text("0 red 3000 white\n")
    ok = argparse.Namespace(dem_cmap="mine", bathymetry_cmap="none",
                            glacier_cmap="white")  # fmt: skip
    gmap.resolve_palettes(ok, pygmt, str(tmp_path))
    assert ok.dem_cmap == str(tmp_path / "mine.cpt")  # a path for the map
    assert (ok.bathymetry_cmap, ok.glacier_cmap) == ("none", "white")
    bad = argparse.Namespace(dem_cmap="slade", bathymetry_cmap=None, glacier_cmap=None)
    with pytest.raises(ValueError, match="did you mean slate"):
        gmap.resolve_palettes(bad, pygmt, str(tmp_path))


def test_gmt_overrides_hold_while_the_map_is_drawn(fake_maps, monkeypatch, tmp_path):
    from gps_plot import gmt_overrides, maps

    seen = {}

    def station_map(*args, **kw):
        seen["glaciers"] = gmt_overrides.layer_kw("glaciers", pen="a", fill="b")
        return types.SimpleNamespace(savefig=lambda path: None, region=(0, 1, 0, 1))

    monkeypatch.setattr(maps, "station_map", station_map)
    monkeypatch.setattr(gmt_overrides, "check", lambda layers, pygmt: None)
    argv = [
        "station",
        "SEY?",
        "--logo",
        "none",
        "--no-scale-bar",
        "--gmt",
        "glaciers:W=0.3p,blue",
        "--gmt-config",
        "map_frame_type=plain",
    ]
    assert _run(argv, tmp_path) == 0  # fmt: skip
    assert seen["glaciers"] == {"fill": "b", "W": "0.3p,blue"}
    assert gmt_overrides.layer_kw("glaciers", pen="a") == {"pen": "a"}  # reset
    assert gmap.gmt_config(["map_frame_type=plain"]) == {"MAP_FRAME_TYPE": "plain"}
    with pytest.raises(ValueError, match="KEY=VALUE"):
        gmap.gmt_config(["MAP_FRAME_TYPE"])


def test_base_map_and_no_stations(fake_maps, monkeypatch, tmp_path):
    from gps_plot import maps

    # no stations, no region: all Iceland, in the iceland3 overview frame
    assert _run(["base", "--logo", "none"], tmp_path) == 0
    (stations,), kw = fake_maps["station_map"]
    assert stations == [] and kw["markers"] is False and kw["labels"] is False
    assert isinstance(kw["region"], maps.CornerRegion)
    assert kw["region"].corners == gmap.OVERVIEW_CORNERS
    assert kw["projection"] == gmap.OVERVIEW_PROJECTION
    assert fake_maps["fig"].saved == [str(tmp_path / "map_base_iceland.png")]
    # a projection of one's own keeps the plain Iceland box
    assert _run(["base", "--projection", "M12c", "--logo", "none"], tmp_path) == 0
    (_,), kw = fake_maps["station_map"]
    assert kw["region"] == gmap.DEFAULT_BASE_REGION and kw["projection"] == "M12c"
    # stations still set the extent, but are not drawn
    assert _run(["station", "SEY?", "--no-stations", "--logo", "none"], tmp_path) == 0
    (stations,), kw = fake_maps["station_map"]
    assert len(stations) == 2 and kw["markers"] is False and kw["region"] is None
    assert fake_maps["fig"].saved == [
        str(tmp_path / "map_base_14.51W-13.5W_64.76N-65.77N_SEY.png")
    ]


def test_stations_files_alone_or_beside_tos(fake_maps, monkeypatch, tmp_path):
    from gps_plot.sources import tos

    monkeypatch.setattr(tos, "tos_gps_stations", lambda *, refresh=False: _tos_fleet())
    planned = tmp_path / "planned.txt"
    planned.write_text("PLA1 -22.5 63.8\nFAR1 -15.0 66.0\n")
    spec = f"{planned}:Planned"
    assert _run(["station", "--stations-file", spec, "--logo", "none"], tmp_path) == 0
    (stations,), _ = fake_maps["station_map"]
    assert [(c.marker, c.status) for c in stations] == [
        ("PLA1", "Planned"), ("FAR1", "Planned"),
    ]  # fmt: skip
    argv = ["station", "--tos", "--region", "-23/-22/63.5/64", "--stations-file", spec]
    assert _run([*argv, "--logo", "none"], tmp_path) == 0
    (stations,), _ = fake_maps["station_map"]
    # TOS by status, the file's stations inside the region beside them
    assert [(c.marker, c.status) for c in stations] == [
        ("FAFC", "ended"), ("S001", "campaign"), ("SENG", "active"),
        ("PLA1", "Planned"),
    ]  # fmt: skip
    # stations.cfg stations beside a file get a group of their own
    assert _run(["station", "SEY?", "--stations-file", spec, "--logo", "none"],
                tmp_path) == 0  # fmt: skip
    (stations,), _ = fake_maps["station_map"]
    assert {c.status for c in stations} == {gmap.STATIONS_CFG_GROUP, "Planned"}


def test_tos_search_selects_and_groups(fake_maps, monkeypatch, tmp_path):
    from gps_plot.sources import tos
    from gps_plot.sources.tos import TosStation

    seen = {}
    fleet = (
        TosStation("SENG", -22.43, 63.87, "Svartsengi", "active", None, "GPS stöð",
                   {"continuity": "continuous"}),
        TosStation("SIL1", -22.40, 63.90, "Sil", "active", None, "SIL stöð", {}),
    )  # fmt: skip

    def search(text, *, refresh=False):
        seen.update(text=text, refresh=refresh)
        if text == "bad":
            raise ValueError("tos search: bad")
        return fleet

    monkeypatch.setattr(tos, "search_stations", search)
    assert _run(["station", "--tos-search", "--epos 'name ~ x'", "--logo", "none"],
                tmp_path) == 0  # fmt: skip
    assert seen == {"text": "--epos 'name ~ x'", "refresh": False}
    (stations,), _ = fake_maps["station_map"]
    assert [(c.marker, c.status) for c in stations] == [
        ("SENG", "GPS stöð"), ("SIL1", "SIL stöð"),
    ]  # a mix: by subtype  # fmt: skip
    argv = ["station", "--tos-search", "--epos", "--group-by", "continuity"]
    assert _run([*argv, "--logo", "none"], tmp_path) == 0
    (stations,), _ = fake_maps["station_map"]
    assert [c.status for c in stations] == ["continuous", "continuity: none"]
    with pytest.raises(SystemExit):
        _run(["station", "--tos-search", "bad"], tmp_path)


def test_no_stations_reaches_the_vector_maps(fake_maps, fake_product, tmp_path):
    argv = ["velocity", "SEY?", "--window", "20231027:20231109", "--no-stations"]
    assert _run([*argv, "--logo", "none"], tmp_path) == 1  # every SEY? refused
    argv = ["velocity", "SENG", "--window", "20231027:20231109", "--no-stations"]
    assert _run([*argv, "--logo", "none"], tmp_path) == 0
    assert fake_maps["velocity_map"][1]["markers"] is False


def test_full_iceland_overview_only_when_everything_fits():
    """All Iceland with no --region/--projection: the overview frame; a
    small extent, a station outside the frame (Kolbeinsey), or a
    projection of one's own keep the plain box."""
    from types import SimpleNamespace

    from gps_plot import maps

    def args(**kw):
        return SimpleNamespace(
            **{"region": None, "projection": None, "margin": None, **kw}
        )

    spread = [
        maps.StationCoordinate("W", -23.0, 64.0),
        maps.StationCoordinate("E", -14.5, 65.9),
    ]
    frame = gmap._overview(args(), None, spread)
    assert isinstance(frame, maps.CornerRegion)
    assert gmap._overview(args(), (-22.8, -21.9, 63.75, 64.1), []) is None
    kolbeinsey = [*spread, maps.StationCoordinate("KOSK", -18.7, 67.1)]
    assert gmap._overview(args(), None, kolbeinsey) is None
    assert gmap._overview(args(projection="M12c"), None, spread) is None


def test_overview_defaults_to_the_iceland_dem(fake_maps, monkeypatch, tmp_path):
    """A full-Iceland map takes --dem iceland when it is built; --dem none
    keeps flat land; without the DEM on the host, no DEM."""
    from gps_plot import maps

    def choice(gis, region, proj):
        return maps.DemChoice("dem.nc", None)

    monkeypatch.setattr(maps, "iceland_dem", choice)
    monkeypatch.setattr(gmap, "_iceland_dem_built", lambda config: True)
    assert _run(["base", "--logo", "none"], tmp_path) == 0
    (_,), kw = fake_maps["station_map"]
    assert kw["dem_grid"] == "dem.nc" and kw["bathymetry_cmap"] == gmap.OVERVIEW_SEA
    assert _run(["base", "--dem", "none", "--logo", "none"], tmp_path) == 0
    (_,), kw = fake_maps["station_map"]
    assert "dem_grid" not in kw
    monkeypatch.setattr(gmap, "_iceland_dem_built", lambda config: False)
    assert _run(["base", "--logo", "none"], tmp_path) == 0
    (_,), kw = fake_maps["station_map"]
    assert "dem_grid" not in kw


def test_fissure_swarms_flag(fake_maps, monkeypatch, tmp_path):
    """--fissure-swarms: the closed FutureVolc polygons as zones (no
    outline), the GeoJSON as outline overlays; a missing file stops."""
    from gps_plot import maps

    gis = tmp_path / "gis"
    (gis / "vector").mkdir(parents=True)
    monkeypatch.setattr(gmap, "gis_dir", lambda config: str(gis))
    monkeypatch.setattr(gmap, "_iceland_dem_built", lambda config: False)
    argv = ["base", "--region", "-22.8/-21.9/63.75/64.1", "--fissure-swarms",
            "--logo", "none"]  # fmt: skip
    with pytest.raises(SystemExit):
        _run(argv, tmp_path)
    for f in (*gmap.FISSURE_SWARM_ZONES, *gmap.FISSURE_SWARM_OUTLINES):
        (gis / f).write_text("")
    assert _run(argv, tmp_path) == 0
    (_,), kw = fake_maps["station_map"]
    assert [z.polygons for z in kw["zones"]] == [
        str(gis / z) for z in gmap.FISSURE_SWARM_ZONES
    ]
    assert all(z.pen is None and z.cmap == maps.DEFAULT_ZONE_CMAP for z in kw["zones"])
    assert [o.path for o in kw["overlays"]] == [
        str(gis / o) for o in gmap.FISSURE_SWARM_OUTLINES
    ]
    assert all(o.fill is None for o in kw["overlays"])
