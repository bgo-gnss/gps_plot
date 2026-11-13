"""gpsplot polar: where a pair's rover went since an onset."""

from __future__ import annotations

import argparse
import datetime as dt
import pathlib

import numpy as np
import pytest

from gps_plot import polar
from gps_plot.cli import find_command


def test_registered() -> None:
    cmd = find_command("polar")
    assert cmd is not None and cmd.target == "gps_plot.cli.polar:run"


@pytest.mark.parametrize(
    "az, lo, hi",
    [
        ([300.0, 330.0], 285.0, 345.0),
        ([350.0, 20.0], -25.0, 35.0),  # across north: starts below 0
        ([10.0, 100.0, 200.0, 300.0], 0.0, 360.0),  # all round
        ([], 0.0, 360.0),
    ],
)
def test_sector_limits(az: list[float], lo: float, hi: float) -> None:
    got = polar.sector_limits(az)
    assert got == pytest.approx((lo, hi))


def test_displacement_from_the_onset_level() -> None:
    t = np.arange(20.0)
    y = np.vstack([t, -t, np.zeros(20)]) + 5.0
    y[0, 3] = 999.0  # an outlier in the onset window: the median ignores it
    td, d, sd = polar.displacement(t, y, np.ones_like(y), onset=2.0, window=4.0)
    assert sd.shape == d.shape
    assert td[0] == 2.0 and len(td) == 18
    # the onset level: the median over t = 2..6 (N: 7, 999, 9, 10, 11 → 10)
    np.testing.assert_allclose(d[:, -1], [14.0, -15.0, 0.0])
    with pytest.raises(ValueError, match="onset"):
        polar.displacement(t, y * np.nan, np.ones_like(y), onset=2.0, window=4.0)


def test_auto_onset_is_the_newest_event_of_a_rovers_area(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import geo_dataread.events as ev

    events = [
        ev.Event("old", "landslide", dt.date(2025, 5, 1), region="slope"),
        ev.Event("new", "landslide", dt.date(2026, 9, 23), region="slope"),
        ev.Event("later", "landslide", dt.date(2026, 11, 2), region="slope"),
        ev.Event("other", "dike", dt.date(2026, 10, 1), region="reykjanes"),
    ]
    monkeypatch.setattr(ev, "read_events", lambda path=None: events)
    monkeypatch.setattr(
        ev, "read_station_areas",
        lambda path=None: {"slope": frozenset({"SVIE"}),
                           "reykjanes": frozenset({"ELDC"})},
    )  # fmt: skip
    end = dt.datetime(2026, 10, 31)
    got = polar.auto_onset(["SVIE"], dt.datetime(2025, 10, 31), end)
    assert got == (dt.datetime(2026, 9, 23), "new")  # not "later" (after the end)
    # an event before the period does not count; nothing in the rover's area
    assert polar.auto_onset(["SVIE"], dt.datetime(2026, 10, 1), end) is None
    assert polar.auto_onset(["SKFC"], None, end) is None


def test_resolve_onset(monkeypatch: pytest.MonkeyPatch) -> None:
    assert polar.resolve_onset(None) is None
    assert polar.resolve_onset("20260920") == dt.datetime(2026, 9, 20)
    import geo_dataread.events as ev

    monkeypatch.setattr(
        ev, "read_events",
        lambda: [ev.Event("rn-2026-x", "dike", dt.date(2026, 8, 1))],
    )  # fmt: skip
    assert polar.resolve_onset("rn-2026-x") == dt.datetime(2026, 8, 1)
    with pytest.raises(ValueError, match="no such"):
        polar.resolve_onset("nonsense")


def test_resolve_span() -> None:
    parser = polar.build_parser()
    now = dt.datetime(2026, 10, 30, 12)
    start, end, name = polar.resolve_span(parser.parse_args(["A-B"]), now)
    assert (end - start).days == 366 and name == "year" and end.hour == 0
    # --rtk: the daily figure's year, to now (the RTK runs on to the hour)
    start, end, name = polar.resolve_span(parser.parse_args(["A-B", "--rtk"]), now)
    assert (end - start).days == 366 and name == "year" and end == now
    args = parser.parse_args(["A-B", "-s", "20260101", "-e", "20260301"])
    assert polar.resolve_span(args, now)[2] == "20260101-20260301"
    # a fixed start running on to today keeps one name day after day
    args = parser.parse_args(["A-B", "-s", "20260923"])
    start, end, name = polar.resolve_span(args, now)
    assert name == "since20260923" and end == dt.datetime(2026, 10, 30)


def test_run_saves_one_figure_per_pair(
    monkeypatch: pytest.MonkeyPatch, tmp_path, capsys: pytest.CaptureFixture[str]
) -> None:
    from gps_plot import direction as dr
    from gps_plot import timesmatplt as tplt

    t = 20300.0 + np.arange(60.0)  # days since 1970
    a = np.radians(335.0)
    y = np.array([np.cos(a), np.sin(a), 0.0])[:, None] * np.linspace(0, 20, 60)
    monkeypatch.setattr(dr, "_daily", lambda name, s, e, c, f: (t, y, np.ones_like(y)))
    monkeypatch.setattr(dr, "config_directions", lambda config: {})
    saved: list[str] = []
    monkeypatch.setattr(tplt, "saveFig", lambda fn, ft, fig: saved.append(fn))
    monkeypatch.setattr(tplt, "inpLogo", lambda fig, **kw: None)
    parser = polar.build_parser()
    args = parser.parse_args(
        ["SVIE-SKFC", "-s", "20250801", "-e", "20251001", "--save", "png",
         "-d", str(tmp_path), "--no-dem"]
    )  # fmt: skip
    assert polar.run(args, parser, object()) == 0
    assert saved == [str(tmp_path / "SVIE_SKFC-polar-20250801-20251001")]
    assert "plotted SVIE-SKFC" in capsys.readouterr().out


def test_bad_onset_is_a_usage_error() -> None:
    parser = polar.build_parser()
    args = parser.parse_args(["A-B", "--onset", "202"])
    with pytest.raises((SystemExit, argparse.ArgumentTypeError)):
        polar.run(args, parser, object())


def _pair(name: str, motion: float, **kw: object) -> polar.PairPlot:
    d = np.vstack(
        [np.linspace(0, 10, 30), np.linspace(0, -5, 30), -np.linspace(0, 3, 30)]
    )
    return polar.PairPlot(name, np.arange(30.0), d, motion, **kw)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "panels",
    [
        ("sector", "map", "section"),
        ("sector", "map", "profile"),
        ("sector", "hodograph"),
        ("map",),
        ("profile",),
    ],
)
@pytest.mark.parametrize("together", [False, True])
def test_figure_panels(panels: tuple[str, ...], together: bool) -> None:
    from gps_plot import terrain

    xs = np.arange(-300.0, 301.0, 5.0)
    z = 800.0 - 0.4 * xs[None, :] + 0.0 * xs[:, None]  # falls toward east
    dem = terrain.DemWindow(xs, xs.copy(), z)
    pairs = [_pair("SVIE-SKFC", 333.0, downslope=(90.0, 21.8),
                   ref=("SKFC", 260.0, 9300.0), pos=(150.0, 0.0))]  # fmt: skip
    if together:
        pairs.append(_pair("SVIN-SKFC", 292.0, marker="^", colour="magenta",
                           pos=(-150.0, 0.0)))  # fmt: skip
    fig = polar.polar_figure(pairs, onset=dt.datetime(2025, 7, 30),
                             panels=panels, dem=dem, half=300.0)  # fmt: skip
    titles = [ax.get_title() for ax in fig.axes]
    if panels == ("map",):
        assert any(t.startswith("terrain (5m DEM, path") for t in titles)
    elif "map" in panels:  # stacked: no map title, the figure's over the map
        assert not any(t.startswith("terrain") for t in titles)
        assert not fig.get_suptitle()
    if "section" in panels:
        assert "vertical section along the motion" in titles
    with pytest.raises(ValueError, match="unknown panel"):
        polar.polar_figure(pairs, onset=dt.datetime(2025, 7, 30),
                           panels=("pie",))  # fmt: skip


def test_together_needs_few_pairs() -> None:
    parser = polar.build_parser()
    args = parser.parse_args(["A-B", "C-D", "E-F", "G-H", "I-J", "--together"])
    with pytest.raises(SystemExit):
        polar.run(args, parser, object())


def test_slope_field_and_nice() -> None:
    from gps_plot import terrain

    x = np.arange(0.0, 50.0, 5.0)
    z = 100.0 - np.tan(np.radians(30.0)) * x[None, :] + 0.0 * x[:, None]
    angle, fall = terrain.slope_field(z, 5.0)
    assert np.allclose(angle, 30.0) and np.allclose(fall, 90.0)
    assert (
        polar._nice(4.3) == 5.0 and polar._nice(1.4) == 1.0 and polar._nice(17) == 20.0
    )


def test_a_westward_section_is_mirrored_with_compass_ends():
    from gps_plot.polar import compass, section_heading

    assert [compass(d) for d in (0, 292, 313, 334, 359, 90)] == [
        "N", "WNW", "NW", "NNW", "N", "E"
    ]  # fmt: skip
    mean, mirrored = section_heading([292.0, 334.0])
    assert mean == pytest.approx(313.0) and mirrored
    assert section_heading([110.0]) == (pytest.approx(110.0), False)
    assert section_heading([350.0, 20.0])[0] == pytest.approx(5.0)


def test_degmin_as_the_gmt_maps() -> None:
    assert polar.degmin(64.00917, "N", "S") == "64°00.55'N"
    assert polar.degmin(-16.80933, "E", "W") == "16°48.56'W"
    assert polar.degmin(-16.99999, "E", "W") == "17°00.00'W"


def test_iceland_layers_read_gmt_segments(tmp_path: pathlib.Path) -> None:
    from gps_plot import insets

    coast = tmp_path / insets.LOCATOR_COAST
    coast.parent.mkdir(parents=True)
    coast.write_text(
        "# @VGMT1.0\n>\n-20 64\n-19 64.5\n>\n-15 65\n-14 65.2\n-14.5 65.4\n"
    )
    layers = polar.iceland_layers(str(tmp_path))
    assert [s.shape for s in layers["coast"]] == [(2, 2), (3, 2)]
    assert layers["swarms"] == [] and layers["glaciers"] == []  # missing files


def test_gmt_colours_and_pens() -> None:
    assert polar.gmt_colour("white") == "white"
    assert polar.gmt_colour("255/0/51") == pytest.approx((1.0, 0.0, 0.2))
    assert polar.gmt_pen("0.2p,black") == (0.2, "black")


@pytest.mark.parametrize(("days", "vmin"), [(10.0, 19.0), (0.0, 0.0), (100.0, 0.0)])
def test_colours_only_the_newest_days(days: float, vmin: float) -> None:
    """Points older than the colour window are open: no fill (BGÓ 2026-11-08)."""
    from matplotlib.collections import PathCollection

    fig = polar.polar_figure([_pair("SVIE-SKFC", 333.0)],
                             onset=dt.datetime(2025, 7, 30), panels=("hodograph",),
                             colour_days=days)  # fmt: skip
    dots = [c for ax in fig.axes for c in ax.collections
            if isinstance(c, PathCollection)]  # fmt: skip
    assert dots and all(c.norm.vmin == pytest.approx(vmin) for c in dots)
    rgba = dots[0].to_rgba(dots[0].get_array())
    assert (rgba[:, 3] == 0).sum() == round(vmin)  # days 0 .. vmin-1 open


def test_panels_close_up_to_the_map() -> None:
    """The panels beside the map start STACKED_GAP_IN right of it (BGÓ 2026-11-08)."""
    from gps_plot import terrain

    xs = np.arange(-300.0, 301.0, 5.0)
    dem = terrain.DemWindow(xs, xs.copy(), 800.0 - 0.4 * xs[None, :] + 0 * xs[:, None])
    pairs = [_pair("SVIE-SKFC", 333.0, downslope=(90.0, 21.8), pos=(150.0, 0.0))]
    fig = polar.polar_figure(pairs, onset=dt.datetime(2025, 7, 30),
                             panels=("sector", "map", "profile"), dem=dem,
                             half=300.0, profile_dem=dem,
                             locator={"coast": [np.array([[-20.0, 64.0], [-19.0, 65.0],
                                                          [-18.0, 64.0]])]})  # fmt: skip
    renderer = fig._get_renderer()
    to_fig = fig.transFigure.inverted()
    free = [a for a in fig.axes if a.get_axes_locator() is None]
    mapax = max(free, key=lambda a: a.get_position().width)
    edge = mapax.get_position().x1
    beside = [a for a in free if a.get_position().x0 > edge - 0.05]  # not the logo
    left = min(a.get_tightbbox(renderer).transformed(to_fig).x0 for a in beside)
    gap_in = (left - mapax.get_position().x1) * fig.get_size_inches()[0]
    assert gap_in == pytest.approx(polar.STACKED_GAP_IN, abs=0.02)
    assert fig.get_size_inches()[0] < 21.0  # the freed width trimmed


def test_older_note() -> None:
    assert (polar.older_note(dt.datetime(2026, 9, 23), 28.0)
            == "since 23.09.2026 until 28 days ago")  # fmt: skip


def test_total_direction_is_the_movement_since_the_onset() -> None:
    """The direction is the end of the movement since the onset (the map's
    grey line), only when it stands out of its noise (BGÓ 2026-11-11)."""
    import numpy as np

    from gps_plot import polar

    since = np.arange(30.0)
    d = np.vstack([since * 1.0, -since * 1.0, np.zeros(30)])  # NW, 315°
    sig = np.full((3, 30), 1.0)
    assert polar.total_direction(since, d, sig) == pytest.approx(315.0)
    flat = np.zeros((3, 30))
    assert polar.total_direction(since, flat, sig) is None
    assert polar.total_direction(np.zeros(0), np.zeros((3, 0)), None) is None


def test_room_to_edge_toward_an_azimuth() -> None:
    from gps_plot import polar

    assert polar.room_to_edge(0.0, 0.0, 0.0, 100.0) == pytest.approx(100.0)
    assert polar.room_to_edge(-90.0, 0.0, 270.0, 100.0) == pytest.approx(10.0)
    # north-west from near the west edge: the west edge comes first
    assert polar.room_to_edge(-80.0, 0.0, 315.0, 100.0) == pytest.approx(20 * 2**0.5)
