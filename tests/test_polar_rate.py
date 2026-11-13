"""polar --rate: one rate source per figure, and the arrows' windows for
the time series (gps_plot.polar)."""

from __future__ import annotations

import datetime as dt
import json

import matplotlib.pyplot as plt
import numpy as np
import pytest

from gps_plot import polar, polar_rtk

HOUR = 1.0 / 24.0


def _rate(north: float, east: float, sigma: float) -> polar_rtk.RtkRate:
    return polar_rtk.RtkRate(north, east, 0.0, sigma, sigma, sigma, 1.0, 24,
                             (sigma,) * 3, (1.0,) * 3, (sigma,) * 3)  # fmt: skip


def _pair(significant: bool) -> polar.PairPlot:
    """30 daily points from day 20400, an RTK tail of 3 days after them."""
    d = np.vstack(
        [np.linspace(0, 10, 30), np.linspace(0, -5, 30), -np.linspace(0, 3, 30)]
    )
    pp = polar.PairPlot("SVIN-SKFC", np.arange(30.0), d, 300.0, t_on=20400.0)
    t = 20430.0 + np.arange(0.0, 3.0, HOUR)
    rd = d[:, -1:] + np.vstack([3.0 * (t - t[0]), -2.0 * (t - t[0]), 0 * t])
    rate = _rate(3.0, -2.0, 0.5 if significant else 5.0)
    al = polar_rtk.Alignment(np.zeros(3), 10, np.ones(3))
    pp.rtk = polar_rtk.RtkOverlay(t, rd, al, rate, "1h", 3.0, 1.0, rd)
    pp.rtk_since = t - 20400.0
    return pp


@pytest.mark.parametrize("significant", [False, True])
def test_rate_step_follows_the_rate_source(significant: bool) -> None:
    pp = _pair(significant)
    pp.rate_from = "rtk"
    step = polar.rate_step(pp)
    if significant:
        k = polar.ARROW_DAYS
        assert step == pytest.approx((3.0 * k, -2.0 * k, 0.0))
        assert "RTK 24 h" in polar.rate_step_label(pp)
    else:
        assert step is None
    pp.rate_from = "daily"  # no track: no daily step either
    assert polar.rate_step(pp) is None


@pytest.mark.parametrize("source", ["daily", "rtk"])
def test_polar_figure_draws_one_rate_source(source: str) -> None:
    pp = _pair(True)
    pp.rate_from = source
    fig = polar.polar_figure([pp], onset=dt.datetime(2025, 11, 9),
                             panels=("sector", "map"))  # fmt: skip
    labels = [t.get_text() for ax in fig.axes if ax.get_legend()
              for t in ax.get_legend().get_texts()]  # fmt: skip
    assert any("RTK 24 h" in x for x in labels) is (source == "rtk")
    plt.close(fig)


def test_rates_file_has_each_arrows_window(tmp_path) -> None:
    pp = _pair(True)
    path = tmp_path / "x-rtk-rates.json"
    polar.write_rates(str(path), [pp], dt.datetime(2026, 11, 12, 22, 0))
    got = json.loads(path.read_text())["rovers"]["SVIN"]
    assert got["pair"] == "SVIN-SKFC"
    assert "daily" not in got  # no track
    assert got["rtk"]["window_days"] == 1.0 and got["rtk"]["shown"] is True
    newest = dt.datetime(1970, 1, 1) + dt.timedelta(days=float(pp.rtk.t.max()))
    assert got["rtk"]["end"] == newest.isoformat(timespec="minutes")


def test_rate_option() -> None:
    parser = polar.build_parser()
    assert parser.parse_args(["SVIN-SKFC", "--rtk", "--rate", "rtk"]).rate == "rtk"
    assert parser.parse_args(["SVIN-SKFC"]).rate == "both"
    with pytest.raises(SystemExit):
        parser.parse_args(["SVIN-SKFC", "--rate", "fast"])


def test_via_pair_prefers_the_station_first() -> None:
    via = ["SVIN-SVIE", "SVIE-SKFC"]
    assert polar.via_pair("SVIN", via) == "SVIN-SVIE"
    assert polar.via_pair("SVIE", via) == "SVIE-SKFC"  # not SVIN-SVIE
    assert polar.via_pair("SKFC", via) == "SVIE-SKFC"
    assert polar.via_pair("FAGC", via) is None


def test_track_days_are_noon_days_since_1970() -> None:
    from gps_plot import track

    t = np.array(
        [track.yearf(dt.date(2026, 11, 11)), track.yearf(dt.date(2024, 2, 29))]
    )
    got = polar.track_days(t)
    want = [(dt.date(2026, 11, 11) - dt.date(1970, 1, 1)).days + 0.5,
            (dt.date(2024, 2, 29) - dt.date(1970, 1, 1)).days + 0.5]  # fmt: skip
    assert got == pytest.approx(want)


def test_anchor_interpolates_then_goes_on_at_its_rate() -> None:
    a = polar_rtk.Anchor("SVIE", np.array([0.5, 1.5, 2.5]),
                         np.array([[0.0, 1.0, 2.0], [0.0, 0.0, 0.0], [np.nan, 5.0, 5.0]]),
                         np.array([1.0, 0.0, -2.0]), np.array([0.2, 0.2, 1.0]))  # fmt: skip
    at = a.at(np.array([1.0, 2.5, 3.5]))
    assert at[0] == pytest.approx([0.5, 2.0, 3.0])  # linear, then 1 mm/d on
    assert at[2] == pytest.approx([5.0, 5.0, 3.0])  # its own days only
    r = polar_rtk.with_anchor_sigma(_rate(1.0, 0.0, 0.3), a)
    assert r is not None
    assert r.sigma_north == pytest.approx(np.hypot(0.3, 0.2))
    assert r.sigma_up == pytest.approx(np.hypot(0.3, 1.0))
    assert polar_rtk.with_anchor_sigma(None, a) is None


def test_frame_option_parses() -> None:
    parser = polar.build_parser()
    args = parser.parse_args(["SVIN", "SVIE", "--refs", "SKFC,SLEC", "--rtk",
                              "--via", "SVIN-SVIE", "--via", "SVIE-SKFC"])  # fmt: skip
    assert args.refs == "SKFC,SLEC" and args.via == ["SVIN-SVIE", "SVIE-SKFC"]


def test_short_window_needs_its_own_significance() -> None:
    """BGÓ 2026-11-13: a 6 h rate of 48.9 ± 24.5 mm/d passed the change
    test; a short window is drawn only above SHORT_SIGMA σ as well."""
    import dataclasses

    r = dataclasses.replace(_rate(48.9, 0.0, 24.5), trigger=True)
    assert not r.significant  # 2.0 σ
    assert dataclasses.replace(_rate(48.9, 0.0, 10.0), trigger=True).significant
    assert not dataclasses.replace(_rate(48.9, 0.0, 1.0), trigger=False).significant


def test_to_circle_cuts_a_step_at_the_radius() -> None:
    f = polar.to_circle(0.0, 0.0, 10.0, 0.0, 4.0)
    assert f == pytest.approx(0.4)
    assert polar.to_circle(1.0, 0.0, 1.0, 0.0, 10.0) == 1.0  # never reaches it
    assert polar.to_circle(3.0, 4.0, 10.0, 0.0, 6.0) == pytest.approx(
        (-3.0 + np.sqrt(9.0 + 11.0)) / 10.0
    )


def test_a_huge_rtk_arrow_leaves_the_sector_size() -> None:
    """BGÓ 2026-11-13: the sector is sized by the dots; an RTK arrow
    longer than that stops at its edge."""

    def rmax(north: float) -> float:
        pp = _pair(True)
        pp.rate_from = "rtk"
        pp.rtk.rate = _rate(north, 0.0, 0.5)
        fig = polar.polar_figure([pp], onset=dt.datetime(2025, 11, 9),
                                 panels=("sector",))  # fmt: skip
        got = next(ax.get_rmax() for ax in fig.axes if ax.name == "polar")
        plt.close(fig)
        return float(got)

    assert rmax(300.0) == pytest.approx(rmax(3.0))
