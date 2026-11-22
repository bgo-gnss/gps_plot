"""timesmatplt.mark_high_sigma: the days with a σ above 10 mm stand out."""

from __future__ import annotations

import datetime as dt

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402

from gps_plot import timesmatplt as tplt  # noqa: E402


def test_high_sigma_days_get_orange_bars_and_a_note() -> None:
    """BGÓ 2026-11-22: --uncert 15 by default, the days above 10 mm marked."""
    x = [dt.datetime(2026, 11, 1) + dt.timedelta(days=i) for i in range(10)]
    y = np.zeros((3, 10))
    dy = np.ones((3, 10))
    dy[2, 5] = 12.0  # the Up of one day
    fig = tplt.stdTimesPlot(x, y, dy)
    notes = [t.get_text() for t in fig.axes[1].texts]
    assert notes == [r"$\sigma$ above 10 mm: 1 day (orange bars)"]
    # all three panels mark that day
    for ax in fig.axes[:3]:
        assert any(
            c.get_color() is not None
            and np.allclose(np.atleast_2d(c.get_color())[0][:3],
                            matplotlib.colors.to_rgb(tplt.HIGH_SIGMA_COLOR))
            for c in ax.collections
        )  # fmt: skip
    quiet = tplt.stdTimesPlot(x, y, np.ones((3, 10)))
    assert not quiet.axes[1].texts


def test_old_high_sigma_days_are_dropped_recent_kept() -> None:
    """BGÓ 2026-11-22: the noisy days of the last N days are shown, older
    ones removed as the 10 mm screen did."""
    t = 2026.0 + np.arange(100) / 365.25
    y = np.ones((3, 100))
    dy = np.ones((3, 100))
    dy[2, [10, 95]] = 12.0  # one old, one recent noisy day
    got = tplt.drop_old_high_sigma(t, y, dy, 30.0)
    assert np.isnan(got[:, 10]).all() and np.isfinite(got[:, 95]).all()
    assert np.isfinite(got[:, :10]).all()
    assert np.isfinite(tplt.drop_old_high_sigma(t, y, dy, -1.0)).all()
    assert np.isnan(tplt.drop_old_high_sigma(t, y, dy, 0.0)[:, 95]).all()


def test_rate_lines_keep_the_products_sigma_screen() -> None:
    """BGÓ 2026-11-22: the plot's --uncert (15) is not the rates' (10)."""
    from geo_dataread.products.velocity_core import DEFAULT_UNCERT

    assert tplt._rate_uncert(None) == DEFAULT_UNCERT == 10.0
    assert tplt._rate_uncert(15) == 15.0
