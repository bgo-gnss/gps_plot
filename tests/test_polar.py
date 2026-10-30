"""gpsplot polar: where a pair's rover went since an onset."""

from __future__ import annotations

import argparse
import datetime as dt

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
    td, d = polar.displacement(t, y, np.ones_like(y), onset=2.0, window=4.0)
    assert td[0] == 2.0 and len(td) == 18
    # the onset level: the median over t = 2..6 (N: 7, 999, 9, 10, 11 → 10)
    np.testing.assert_allclose(d[:, -1], [14.0, -15.0, 0.0])
    with pytest.raises(ValueError, match="onset"):
        polar.displacement(t, y * np.nan, np.ones_like(y), onset=2.0, window=4.0)


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
    start, end, name = polar.resolve_span(parser.parse_args(["A-B", "--rtk"]), now)
    assert (end - start).days == 7 and name == "week" and end == now
    args = parser.parse_args(["A-B", "-s", "20260101", "-e", "20260301"])
    assert polar.resolve_span(args, now)[2] == "20260101-20260301"


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
    monkeypatch.setattr(tplt, "inpLogo", lambda fig: None)
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
    "panels", [("sector", "map", "section"), ("sector", "hodograph"), ("map",)]
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
    fig = polar.polar_figure(pairs, onset=dt.datetime(2025, 7, 30), note="n",
                             panels=panels, dem=dem, half=300.0)  # fmt: skip
    titles = [ax.get_title() for ax in fig.axes]
    if "map" in panels:
        assert any(t.startswith("terrain, path") for t in titles)
    if "section" in panels:
        assert "vertical section along the motion" in titles
    with pytest.raises(ValueError, match="unknown panel"):
        polar.polar_figure(pairs, onset=dt.datetime(2025, 7, 30), note="n",
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
