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
