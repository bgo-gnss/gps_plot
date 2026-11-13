"""The profile panel of gpsplot polar: ground along the motion, plunge fit."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from gps_plot import polar
from gps_plot import profile as prf


def test_plunge_fit_recovers_a_known_line() -> None:
    rng = np.random.default_rng(1)
    along = np.linspace(1.0, 20.0, 40)
    up = -np.tan(np.radians(30.0)) * along + rng.normal(0, 0.5, along.size)
    pl = prf.plunge_fit(along, up, np.full(40, 0.5), np.full(40, 0.5))
    assert pl is not None and pl.n == 40
    assert pl.deg == pytest.approx(30.0, abs=1.0)
    assert 0 < pl.sigma < 1.0
    # σ much larger than the scatter: the wedge follows σ, not the misfit
    wide = prf.plunge_fit(along, up, np.full(40, 5.0), np.full(40, 5.0))
    assert wide is not None and wide.chi2 < 1 and wide.sigma > 3 * pl.sigma
    # rising motion is a negative plunge
    rise = prf.plunge_fit(along, -up, np.full(40, 0.5), np.full(40, 0.5))
    assert rise is not None and rise.deg == pytest.approx(-30.0, abs=1.0)


def test_plunge_fit_needs_points_and_motion() -> None:
    one = np.array([1.0])
    assert prf.plunge_fit(one, one, one, one) is None
    zero = np.zeros(5)
    assert prf.plunge_fit(zero, np.ones(5), np.ones(5), np.ones(5)) is None
    nan = np.full(5, np.nan)  # no σ (e.g. a pair read without it)
    assert prf.plunge_fit(np.arange(5.0), np.arange(5.0), nan, nan) is None


def test_weekly_medians_need_three_days() -> None:
    t = np.array([0.0, 1.0, 2.0, 8.0, 9.0, 15.0, 16.0, 17.0, 18.0])
    v = t * 2.0
    wt, wa, wu = prf.weekly_medians(t, v, -v)
    np.testing.assert_allclose(wt, [1.0, 16.5])  # week 1 has only 2 days
    np.testing.assert_allclose(wa, [2.0, 33.0])
    np.testing.assert_allclose(wu, [-2.0, -33.0])
    assert prf.weekly_medians([], [], [])[0].size == 0


def test_daily_medians_reduce_rtk_epochs_to_days() -> None:
    t = np.array([0.1, 0.5, 0.9, 1.2, 1.3])
    t2, v = prf.daily_medians(t, np.array([1.0, 2.0, 30.0, 4.0, 6.0]))
    np.testing.assert_allclose(t2, [0.5, 1.25])
    np.testing.assert_allclose(v, [2.0, 5.0])
    # daily data passes through (sorted)
    t3, v3 = prf.daily_medians([2.0, 0.0, 1.0], [3.0, 1.0, 2.0])
    np.testing.assert_allclose(t3, [0.0, 1.0, 2.0])
    np.testing.assert_allclose(v3, [1.0, 2.0, 3.0])


def test_apparent_dip() -> None:
    assert prf.apparent_dip(90.0, (90.0, 20.0)) == pytest.approx(20.0)
    assert prf.apparent_dip(0.0, (90.0, 20.0)) == pytest.approx(0.0, abs=1e-9)
    assert prf.apparent_dip(270.0, (90.0, 20.0)) == pytest.approx(-20.0)


def _dem() -> object:
    from gps_plot import terrain

    xs = np.arange(-400.0, 401.0, 5.0)
    z = 800.0 - 0.4 * xs[None, :] + 0.0 * xs[:, None]  # falls toward east
    return terrain.DemWindow(xs, xs.copy(), z)


def test_sample_is_bilinear_and_nan_outside() -> None:
    dem = _dem()
    got = prf.sample(dem, np.array([0.0, 2.5, 1000.0]), np.array([0.0, 7.0, 0.0]))
    np.testing.assert_allclose(got[:2], [800.0, 799.0])
    assert np.isnan(got[2])


def _pair(motion: float, pos: tuple[float, float] = (0.0, 0.0)) -> polar.PairPlot:
    n = 30
    a = np.radians(motion)
    h = np.linspace(0, 10, n)
    d = np.vstack([h * np.cos(a), h * np.sin(a), -0.5 * h])
    return polar.PairPlot("SVIE-SKFC", np.arange(float(n)), d, motion, pos=pos,
                          downslope=(90.0, 21.8), sig=np.ones_like(d))  # fmt: skip


@pytest.mark.parametrize("motion, west", [(90.0, False), (292.0, True)])
def test_prepare_puts_the_rover_on_the_ground(motion: float, west: bool) -> None:
    pp = _pair(motion, pos=(100.0, 0.0))
    prof = prf.prepare(pp, _dem(), scale=5.0)
    assert prof.west is west
    assert prof.z0 == pytest.approx(760.0)  # 800 - 0.4 * 100
    assert prof.plunge is not None
    assert prof.plunge.deg == pytest.approx(np.degrees(np.arctan(0.5)), abs=0.1)
    # the range holds the ground within the half width and the scaled points
    near = prof.z[np.abs(prof.s) <= prf.PROFILE_HALF_M]
    assert prof.lo < min(np.nanmin(near), prof.z0 - 5 * 5.0)
    assert prof.hi > np.nanmax(near)
    # the box: the ground and the rover, padded
    assert prof.box == pytest.approx(np.nanmax(near) - np.nanmin(near) + 15 + 25, abs=1)
    # past the DEM edge (the profile reaches 250 m from a rover at 300 m)
    edge = prf.prepare(_pair(90.0, pos=(300.0, 0.0)), _dem(), scale=5.0)
    assert np.isnan(edge.z).any() and np.isfinite(edge.z).any()


def test_prepare_without_a_dem() -> None:
    prof = prf.prepare(_pair(90.0), None, scale=5.0)
    assert prof.z0 == 0.0 and np.isnan(prof.z).all()
    assert prof.hi > prof.lo


@pytest.mark.parametrize("dem", [True, False])
def test_profile_figures(dem: bool) -> None:
    pairs = [_pair(90.0, pos=(100.0, 0.0)), _pair(292.0, pos=(-100.0, 0.0))]
    pairs[1].name, pairs[1].marker = "SVIN-SKFC", "^"
    pairs.append(polar.PairPlot("XXXX-SKFC", np.arange(3.0), np.zeros((3, 3)), None))
    for panels in (("profile",), ("sector", "map", "profile")):
        fig = polar.polar_figure(pairs, onset=dt.datetime(2026, 9, 23),
                                 panels=panels, dem=_dem() if dem else None,
                                 half=300.0, profile_dem=_dem() if dem else None)  # fmt: skip
        titles = [ax.get_title() for ax in fig.axes]
        # one sub-panel per pair with a motion; the one without is left out
        assert sum(t.endswith("ground along its motion (90°)") for t in titles) == 1
        assert sum("(292°)" in t for t in titles) == 1
        assert not any("XXXX" in t for t in titles)
        # the colour bar is inside a panel: no side axes past the panels
        assert max(ax.get_position().x1 for ax in fig.axes) < 1.0
        # no footnote (BGÓ 2026-11-02)
        assert not any("plunge" in t.get_text() for t in fig.texts)
        assert "movement since 2026-09-23" in fig.texts[0].get_text()


def test_no_motion_at_all() -> None:
    pairs = [polar.PairPlot("XXXX-SKFC", np.arange(3.0), np.zeros((3, 3)), None)]
    fig = polar.polar_figure(pairs, onset=dt.datetime(2026, 9, 23),
                             panels=("sector", "profile"))  # fmt: skip
    assert any(t.get_text() == "no motion direction" for ax in fig.axes
               for t in ax.texts)  # fmt: skip


def test_scale_counts_rtk_points_but_not_one_wild_hour() -> None:
    from types import SimpleNamespace

    daily = np.array([[0.0, -10.0], [0.0, 0.0], [0.0, -10.0]])  # 14 mm
    pp = SimpleNamespace(d=daily, rtk=None, rtk_since=np.array([]))
    alone = prf.scale_for([pp])
    up = np.r_[np.linspace(-40.0, 0.0, 200), 500.0]  # one wild hour
    rtk = np.vstack([np.zeros_like(up), np.zeros_like(up), up])
    pp_rtk = SimpleNamespace(d=daily, rtk=SimpleNamespace(d=rtk, path=rtk[:, :1]),
                             rtk_since=np.arange(up.size))  # fmt: skip
    with_rtk = prf.scale_for([pp_rtk])
    assert with_rtk < alone  # the RTK points fit in too
    assert with_rtk >= polar._nice(0.5 * prf.PROFILE_HALF_M / 60.0)  # not the 500


def test_end_along_up_is_the_newest_level() -> None:
    t = np.arange(10.0)
    along, up = -2.0 * t, -t
    prof = prf.Profile(None, None, None, 0.0, t, along, up, np.ones(10),
                       np.ones(10), None, 0.0, 1.0)  # fmt: skip
    got = prf.end_along_up(prof)
    assert got is not None
    assert len(got) == 4
    assert got[0] == pytest.approx(-18.0, abs=2.1)
    assert got[1] == pytest.approx(-9.0, abs=1.1)
