"""rtk_rate: the RTK rate warning, the speed over 24 h / 72 h."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gps_plot import rtk_rate


def _hourly(days: float, rate_from: float | None, rate: float, noise: float = 0.5,
            seed: int = 3) -> pd.DataFrame:  # fmt: skip
    """Hourly N/E/U [mm]: white noise, and from ``rate_from`` days on a
    northward motion of ``rate`` mm/d."""
    rng = np.random.default_rng(seed)
    t = pd.date_range("2026-11-01", periods=int(days * 24), freq="1h")
    x = np.arange(t.size) / 24.0
    y = noise * rng.standard_normal((t.size, 3))
    if rate_from is not None:
        y[:, 0] += rate * np.clip(x - rate_from, 0, None)
    return pd.DataFrame(y, index=t, columns=["N", "E", "U"])


def test_spec_defaults_and_checks() -> None:
    spec = rtk_rate.spec_of({})
    assert spec is not None
    assert spec["windows_h"] == [24.0, 72.0]
    assert spec["horizontal_mm_per_day"] == [1.0, 3.0, 5.0]
    assert spec["components"] == ["horizontal", "up"]
    assert spec["significance_sigma"] is None
    assert rtk_rate.spec_of(None) is None and rtk_rate.spec_of(False) is None
    with pytest.raises(ValueError):
        rtk_rate.spec_of({"horizontal_mm_per_day": [3, 1]})
    with pytest.raises(ValueError):
        rtk_rate.spec_of({"components": ["sideways"]})


@pytest.mark.parametrize(
    "value, sigma, level",
    [(0.5, 0.1, 0), (1.5, 0.1, 1), (3.5, 0.1, 2), (6.0, 0.1, 3), (6.0, 4.0, 0),
     (2.0, float("nan"), 0)],
)  # fmt: skip
def test_levels_need_the_edge_and_the_sigma(
    value: float, sigma: float, level: int
) -> None:
    assert rtk_rate.levels_of(value, sigma, (1.0, 3.0, 5.0), 2.0) == level


def test_window_rate_recovers_a_line() -> None:
    h = _hourly(4.0, 0.0, 2.0, noise=0.0)
    got = rtk_rate.window_rates(h, 24.0, pd.DatetimeIndex([h.index[-1]]))
    assert got[0, 0, 0] == pytest.approx(2.0)
    assert got[0, 0, 1] == pytest.approx(0.0, abs=1e-9)


def test_quiet_is_green_and_a_new_pace_is_not() -> None:
    spec = rtk_rate.spec_of({"components": ["horizontal"]})
    assert spec is not None
    quiet = _hourly(20.0, None, 0.0)
    rows = rtk_rate.pair_rates(quiet, spec, quiet.index[-1], 3.0)
    assert max(r["level"] for r in rows) == 0
    # 3.5 mm/d north for the last four days
    moving = _hourly(20.0, 16.0, 3.5)
    rows = rtk_rate.pair_rates(moving, spec, moving.index[-1], 3.0)
    by = {r["window_h"]: r for r in rows}
    assert by[24.0]["level"] == 2  # 3.5 mm/d: orange
    assert by[24.0]["horizontal_mm_per_day"] == pytest.approx(3.5, abs=0.3)
    assert by[72.0]["level"] == 2
    assert by[72.0]["azimuth"] in (0.0, 360.0)


def test_up_left_out_when_not_asked() -> None:
    h = _hourly(20.0, None, 0.0)
    h["U"] += -8.0 * np.clip((np.arange(len(h)) / 24.0) - 16.0, 0, None)
    both = rtk_rate.spec_of({})
    horiz = rtk_rate.spec_of({"components": ["horizontal"]})
    assert both is not None and horiz is not None
    assert max(r["level"] for r in rtk_rate.pair_rates(h, both, h.index[-1], 3.0)) == 3
    assert max(r["level"] for r in rtk_rate.pair_rates(h, horiz, h.index[-1], 3.0)) == 0


def test_up_has_its_own_sigma_factor() -> None:
    """BGÓ 2026-11-13: Up on, with a higher σ factor, fires only when far
    out of its noise."""
    h = _hourly(20.0, None, 0.0)
    h["U"] += -8.0 * np.clip((np.arange(len(h)) / 24.0) - 16.0, 0, None)
    spec = rtk_rate.spec_of({"up_significance_sigma": 1e6})
    assert spec is not None and spec["up_significance_sigma"] == 1e6
    assert max(r["level"] for r in rtk_rate.pair_rates(h, spec, h.index[-1], 3.0)) == 0
    assert rtk_rate.spec_of({})["up_significance_sigma"] is None  # type: ignore[index]
