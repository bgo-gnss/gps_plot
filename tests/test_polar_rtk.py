"""polar --rtk: the RTK tail on the daily polar figure (gps_plot.polar_rtk)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from gps_plot import polar, polar_rtk

HOUR = 1.0 / 24.0


def _series(days: float, rate: tuple[float, float, float], noise: float = 0.0,
            diurnal: float = 0.0, seed: int = 1) -> tuple[np.ndarray, np.ndarray]:  # fmt: skip
    """Hourly bins [days since 1970] of a line at ``rate`` mm/d, white noise
    and a diurnal sinusoid (the sidereal multipath's stand-in) of amplitude
    ``diurnal`` [mm]."""
    rng = np.random.default_rng(seed)
    t = 20400.0 + np.arange(0.0, days, HOUR) + HOUR / 2
    tt = t - t[0]
    y = np.array([r * tt for r in rate])
    y += noise * rng.standard_normal(y.shape)
    y += diurnal * np.sin(2 * np.pi * tt * 1.0027 + np.array([[0.3], [1.1], [2.0]]))
    return t, y


def test_line_fit_recovers_a_line() -> None:
    t = np.linspace(0, 1, 25)
    slope, sigma, res = polar_rtk.line_fit(t, 3.0 * t + 2.0)
    assert slope == pytest.approx(3.0)
    assert sigma == pytest.approx(0.0, abs=1e-9)
    np.testing.assert_allclose(res, 0.0, atol=1e-9)


def test_rate_recovered_from_white_noise() -> None:
    t, y = _series(10.0, (2.0, -1.0, 0.0), noise=1.0)
    r = polar_rtk.estimate_rate(t, y, window=1.0)
    assert r is not None
    assert r.north == pytest.approx(2.0, abs=3 * r.sigma_north)
    assert r.east == pytest.approx(-1.0, abs=3 * r.sigma_east)
    # white noise: σ about the white-noise σ of a 24-point line (1/√2 mm/d)
    assert 0.4 < r.sigma_north < 1.4
    assert r.window == 1.0 and r.n == 24


def test_diurnal_noise_inflates_sigma() -> None:
    """A 24 h line cannot see a diurnal cycle in its own residuals; the
    repeatability of window rates does."""
    t, y = _series(10.0, (0.0, 0.0, 0.0), noise=1.0, seed=2)
    white = polar_rtk.estimate_rate(t, y, window=1.0)
    t, y = _series(10.0, (0.0, 0.0, 0.0), noise=1.0, diurnal=8.0, seed=2)
    cyc = polar_rtk.estimate_rate(t, y, window=1.0)
    assert white is not None and cyc is not None
    assert cyc.sigma_north > 3 * white.sigma_north
    # the repeatability, not white × k, sets it
    assert cyc.repeat[0] > cyc.white[0] * cyc.k[0]


def test_repeat_sigma_ignores_a_steady_rate_and_acceleration() -> None:
    t = 20400.0 + np.arange(0.0, 10.0, HOUR)
    tt = t - t[0]
    steady = polar_rtk.repeat_sigma(t, 5.0 * tt, 1.0, HOUR)
    accel = polar_rtk.repeat_sigma(t, 0.5 * tt**2, 1.0, HOUR)
    assert steady == pytest.approx(0.0, abs=1e-6)
    assert accel == pytest.approx(0.0, abs=1e-6)


def test_repeat_sigma_needs_history() -> None:
    t, y = _series(2.5, (1.0, 0.0, 0.0), noise=1.0)  # rates over 1.5 d only
    assert np.isnan(polar_rtk.repeat_sigma(t, y[0], 1.0, HOUR))


def test_too_few_bins_gives_no_rate() -> None:
    t, y = _series(10.0, (1.0, 0.0, 0.0), noise=1.0)
    keep = (t < t.max() - 1.0) | (np.arange(t.size) % 4 == 0)  # 6 of 24 in the window
    assert polar_rtk.estimate_rate(t[keep], y[:, keep], window=1.0) is None


def _rate(north: float, east: float, sigma: float) -> polar_rtk.RtkRate:
    return polar_rtk.RtkRate(north, east, 0.0, sigma, sigma, sigma, 1.0, 24,
                             (sigma,) * 3, (1.0,) * 3, (sigma,) * 3)  # fmt: skip


@pytest.mark.parametrize("snr, shown", [(1.5, False), (2.0, False), (3.0, True)])
def test_significance_gate(snr: float, shown: bool) -> None:
    r = _rate(0.6 * snr, -0.8 * snr, 1.0)  # speed = snr, σ along it = 1
    assert r.speed == pytest.approx(snr)
    assert r.sigma_speed == pytest.approx(1.0)
    assert r.significant is shown


def test_as_track_rate_in_mm_per_year() -> None:
    from gps_plot.track import YEAR

    tr = _rate(1.0, 2.0, 0.5).as_track_rate()
    assert tr.north == pytest.approx(YEAR) and tr.east == pytest.approx(2 * YEAR)
    assert tr.sigma_north == pytest.approx(0.5 * YEAR)


def test_day_medians_need_half_a_day() -> None:
    t = 20400.0 + np.r_[np.arange(0.0, 1.0, HOUR), 1.0 + np.arange(0.0, 0.25, HOUR)]
    t = t + HOUR / 2
    y = np.vstack([np.ones_like(t), 2 * np.ones_like(t), np.zeros_like(t)])
    days, med = polar_rtk.day_medians(t, y, HOUR)
    np.testing.assert_array_equal(days, [20400.0])  # the 6-hour day is out
    np.testing.assert_allclose(med[:, 0], [1.0, 2.0, 0.0])


def test_align_shift_recovers_an_offset() -> None:
    t, y = _series(12.0, (1.0, -2.0, 0.5), noise=3.0)
    offset = np.array([1856000.0, -8414500.0, 399600.0])  # a full baseline
    t_day = 20400.0 + np.arange(12.0) + 0.499  # the daily epochs (11:58)
    d_day = np.array([r * (t_day - t[0]) for r in (1.0, -2.0, 0.5)])
    al = polar_rtk.align_shift(t, y + offset[:, None], t_day, d_day, HOUR)
    np.testing.assert_allclose(al.shift, -offset, atol=1.5)
    assert al.days == 12
    assert (al.scatter < 2.0).all()


def test_align_shift_needs_a_common_day() -> None:
    t, y = _series(3.0, (0.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="no day common"):
        polar_rtk.align_shift(t, y, np.array([20500.5]), np.zeros((3, 1)), HOUR)


def test_overlay_keeps_the_tail_on_the_daily_path() -> None:
    t, y = _series(14.0, (1.0, 0.0, 0.0), noise=0.5)
    t_day = 20400.0 + np.arange(14.0) + 0.5
    d_day = np.array([(t_day - t[0]), 0 * t_day, 0 * t_day])
    ov = polar_rtk.overlay(t, y + 1000.0, t_day, d_day, days=3.0, window=1.0)
    assert ov.t.min() > t.max() - 3.0
    # on the daily path: the newest bins near the daily line's end
    assert ov.d[0, -1] == pytest.approx(t.max() - t[0], abs=2.5)
    assert ov.path.shape == ov.d.shape and ov.window == 1.0
    assert ov.rate is not None


def test_bins_centred_and_thin_ones_dropped() -> None:
    t = np.array([0.0, HOUR, 2 * HOUR])
    y = np.ones((3, 3))
    tb, yb = polar_rtk.bins(t, y, np.array([60, 5, 58]), HOUR)
    np.testing.assert_allclose(tb, [HOUR / 2, 2.5 * HOUR])
    assert yb.shape == (3, 2)


def test_labels_are_usetex_safe() -> None:
    r = _rate(3.0, 0.0, 1.0)
    for text in (polar_rtk.rate_label(r), polar_rtk.quiet_label(r, 1.0),
                 polar_rtk.quiet_label(None, 3.0)):  # fmt: skip
        assert "σ" not in text and "±" not in text
    assert "24 h" in polar_rtk.rate_label(r)
    assert polar_rtk.span_text(3.0) == "72 h"
    assert polar_rtk.span_text(7.0) == "7 d"
    assert polar_rtk.span_text(10 / 1440) == "10 min"


@pytest.mark.parametrize("significant", [False, True])
def test_figure_with_the_rtk_tail(significant: bool) -> None:
    """The figure draws with an RTK overlay, arrow or note."""
    d = np.vstack(
        [np.linspace(0, 10, 30), np.linspace(0, -5, 30), -np.linspace(0, 3, 30)]
    )
    pp = polar.PairPlot("SVIN-SKFC", np.arange(30.0), d, 300.0)
    t = 30.0 + np.arange(0.0, 3.0, HOUR)
    rd = d[:, -1:] + np.vstack([0.3 * (t - 30), -0.2 * (t - 30), 0 * t])
    rate = _rate(3.0, -2.0, 0.5 if significant else 5.0)
    al = polar_rtk.Alignment(np.zeros(3), 10, np.ones(3))
    pp.rtk = polar_rtk.RtkOverlay(t, rd, al, rate, "1h", 3.0, 1.0, rd)
    pp.rtk_since = t
    import datetime as dt

    fig = polar.polar_figure([pp], onset=dt.datetime(2026, 9, 23),
                             panels=("sector", "map", "profile"))  # fmt: skip
    labels = [t.get_text() for ax in fig.axes if ax.get_legend()
              for t in ax.get_legend().get_texts()]  # fmt: skip
    assert any("RTK 24 h" in x for x in labels)
    assert any("below" in x for x in labels) is (not significant)


def test_pick_rate_takes_the_shortest_significant_window() -> None:
    quiet, slow, fast = _rate(0.5, 0.0, 1.0), _rate(3.0, 0.0, 1.0), _rate(9.0, 0.0, 1.0)
    assert polar_rtk.pick_rate([None, quiet, slow]) is slow
    assert polar_rtk.pick_rate([fast, quiet, slow]) is fast
    assert polar_rtk.pick_rate([quiet, None, quiet]) is quiet  # the longest
    assert polar_rtk.pick_rate([None, None]) is None


def _minutes(days: float, rate: float, noise: float, seed: int = 2):
    rng = np.random.default_rng(seed)
    t = 20400.0 + np.arange(0.0, days, 1 / 1440) + 1 / 2880
    tt = t - t[0]
    y = np.vstack([rate * tt, 0 * tt, 0 * tt]) + noise * rng.standard_normal(
        (3, t.size)
    )
    return t, y


def _hourly(tm, ym):
    """Hourly medians of minute values (as the reader's 1h bins)."""
    h = np.floor((tm - tm[0]) * 24)
    keys = np.unique(h)
    th = np.array([tm[h == k].mean() for k in keys])
    yh = np.array([[np.median(ym[c, h == k]) for k in keys] for c in range(3)])
    return th, yh


def test_cascade_quiet_draws_no_change_window() -> None:
    """Steady 1 mm/d and noise: no shorter window fires; the 72 h line wins
    or nothing (never a minute window on noise)."""
    tm, ym = _minutes(12.0, 1.0, 3.0)
    th, yh = _hourly(tm, ym)
    ov = polar_rtk.overlay(th, yh, th[::24], yh[:, ::24], days=3.0, minutes=(tm, ym))
    assert ov.windows == polar_rtk.RATE_WINDOWS and len(ov.rates) == 6
    short = [r for r in ov.rates[:-1] if r is not None]
    assert short and all(math.isfinite(r.threshold) for r in short)
    assert all(r.threshold >= polar_rtk.change_limit(r.window)[0] for r in short)
    assert all(r.threshold >= 4.0 for r in short if r.window < 1 / 24)
    assert ov.rate is not None and ov.rate.window == 3.0


def test_cascade_sudden_speed_up_takes_over() -> None:
    """Steady 1 mm/d for days, then 2 m/d in the newest half hour: a
    minute window takes over (the 10-minute one is near its limit: 2 m/d
    is ~4 σ of a 10-minute slope through 3 mm noise)."""
    tm, ym = _minutes(12.0, 1.0, 3.0)
    late = tm > tm.max() - 30 / 1440
    ym[0, late] += 2000.0 * (tm[late] - (tm.max() - 30 / 1440))
    th, yh = _hourly(tm, ym)
    ov = polar_rtk.overlay(th, yh, th[::24], yh[:, ::24], days=3.0, minutes=(tm, ym))
    assert ov.rate is not None and ov.rate.significant
    assert ov.rate.window < 1 / 24
    assert ov.rate.z > ov.rate.threshold
    assert "speed-up" in polar_rtk.rate_label(ov.rate)


def test_window_rates_match_the_loop() -> None:
    t, y = _series(3.0, (2.0, 0.0, 0.0), noise=1.0)
    ends, rates = polar_rtk.window_rates(t, y[0], 0.5, HOUR)
    for e, r in list(zip(ends, rates, strict=True))[::7]:
        m = (t > e - 0.5 + 1e-9) & (t <= e + 1e-9)
        assert r == pytest.approx(polar_rtk.line_fit(t[m], y[0, m])[0], rel=1e-6)


def test_quiet_label_lists_the_windows() -> None:
    r = _rate(0.2, 0.0, 3.0)
    text = polar_rtk.quiet_label(r, 3.0, polar_rtk.RATE_WINDOWS)
    assert text.startswith("RTK 10 min–72 h: no speed-up, 24 h ")
    assert "σ" not in text and "±" not in text


def test_clean_minutes_drops_spikes_keeps_a_ramp() -> None:
    tm, ym = _minutes(2.0, 0.0, 1.0)
    ym[0, 1000] += 80.0  # an isolated spike
    late = tm > tm.max() - 30 / 1440
    ym[1, late] += 12000.0 * (tm[late] - (tm.max() - 30 / 1440))  # 12 m/d ramp
    _, yc = polar_rtk.clean_minutes(tm, ym)
    assert np.isnan(yc[0, 1000])
    assert np.isfinite(yc[1, late][2:-2]).all()  # the ramp survives
    assert yc[1, -3] - yc[1, np.flatnonzero(late)[0]] > 200.0


def test_clean_minutes_takes_off_the_sidereal_multipath() -> None:
    tm, ym = _minutes(10.0, 0.0, 0.5)
    sid = polar_rtk.SIDEREAL_S / 86400.0
    path = 5.0 * np.sin(2 * np.pi * (tm - tm[0]) / sid * 3)  # repeats each sidereal day
    ym = ym + path
    _, yc = polar_rtk.clean_minutes(tm, ym)
    last = tm > tm[0] + 8.0  # seven sidereal days back exist
    assert np.nanstd(yc[0, last]) < 0.4 * np.nanstd(ym[0, last])


def test_change_limits_per_window() -> None:
    assert polar_rtk.change_limit(10 / 1440) == (4.0, 0.999)
    assert polar_rtk.change_limit(0.25) == (3.0, 0.99)
    assert polar_rtk.change_limit(0.5)[0] == 3.5
    assert polar_rtk.change_limit(1.0)[0] == 4.0
