"""rtk_accel: the RTK acceleration warning (PLAN-rtk-trigger T10)."""

from __future__ import annotations

import datetime as dt
from typing import Any

import numpy as np
import pandas as pd
import pytest

from gps_plot import rtk_accel as ra
from gps_plot import rtk_status as rs

NOW = dt.datetime(2026, 11, 20, 12, 0)
J24 = ra.SCALES.index(24.0)
J6 = ra.SCALES.index(6.0)


def _minutes(
    start: dt.datetime, end: dt.datetime, v_mm_d: float = 0.0,
    dv_mm_d: float = 0.0, onset: dt.datetime | None = None, noise: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:  # fmt: skip
    """One-minute epochs, east motion v (+ dv after ``onset``), N(0, noise)."""
    t = pd.date_range(start, end, freq="1min")
    days = ((t - t[0]) / pd.Timedelta("1D")).to_numpy()
    rng = np.random.default_rng(1)
    y = rng.normal(0.0, noise, (3, t.size))
    y[1] += v_mm_d * days
    if onset is not None:
        after = ((t - pd.Timestamp(onset)) / pd.Timedelta("1D")).to_numpy()
        y[1] += dv_mm_d * np.clip(after, 0, None)
    return t.to_numpy(), y


def _reader(**kw: Any) -> ra.Reader:
    def read(pair: str, start: dt.datetime, end: dt.datetime) -> Any:
        t, y = _minutes(NOW - dt.timedelta(days=30), NOW, **kw)
        keep = (t >= np.datetime64(start)) & (t <= np.datetime64(end))
        return t[keep], y[:, keep]

    return read


def _values(**kw: Any) -> np.ndarray:
    grid = pd.date_range(NOW - dt.timedelta(hours=6), NOW, freq=ra.GRID)
    v = ra.pair_values(_reader(**kw), "A-B", NOW - ra.HISTORY, NOW, grid)
    assert v is not None
    return v


def test_steady_motion_is_not_acceleration() -> None:
    v = _values(v_mm_d=20.0)
    assert np.nanmax(np.abs(v[-1, :, 1])) < 1.0  # mm: the background explains it


def test_velocity_change_shows_as_dv_times_t() -> None:
    # 50 h after the onset the 24 h scale (48 h window) sees ~24 mm/d x 1 d
    v = _values(dv_mm_d=24.0, onset=NOW - dt.timedelta(hours=50))
    assert v[-1, J24, 1] == pytest.approx(24.0, rel=0.25)
    # 20 h after it the 6 h scale (18 h window) sees ~6 mm; by 60 h the
    # change is its background (L + 48 h): a scale forgets after that
    v = _values(dv_mm_d=24.0, onset=NOW - dt.timedelta(hours=20))
    assert v[-1, J6, 1] == pytest.approx(6.0, rel=0.25)
    v = _values(dv_mm_d=24.0, onset=NOW - dt.timedelta(hours=80))
    assert abs(v[-1, J6, 1]) < 1.0


def test_no_look_ahead() -> None:
    """A value at t must not change when later data arrive."""
    grid = pd.date_range(
        NOW - dt.timedelta(hours=3), NOW - dt.timedelta(hours=2), freq=ra.GRID
    )
    read = _reader(dv_mm_d=50.0, onset=NOW - dt.timedelta(hours=1))
    early = ra.pair_values(
        read, "A-B", NOW - ra.HISTORY, NOW - dt.timedelta(hours=2), grid
    )
    late = ra.pair_values(read, "A-B", NOW - ra.HISTORY, NOW, grid)
    assert early is not None and late is not None
    np.testing.assert_allclose(early, late, equal_nan=True)


def test_levels_hold_for_half_the_scale() -> None:
    z = np.zeros((20, len(ra.SCALES)))
    z[10, 0] = 5.0  # one step over every k at 0.25 h: persistence 1 step
    z[10:12, J6] = 5.0  # 2 steps at 6 h: needs 12 (0.5 T) → not enough
    lv = ra.scale_levels(z, (1.0, 2.0, 3.0))
    assert lv[10, 0] == 3 and np.nanmax(lv[:, J6]) == 0


def test_calibrate_spreads_thin_multipliers_and_loads() -> None:
    read = _reader()
    got = ra.calibrate(read, ["A-B"], [("2026-11-15", "2026-11-18")])
    k = got["multipliers"]
    assert k[1] >= 1.5 * k[0] - 1e-3 and k[2] >= 2.0 * k[0] - 1e-3
    assert got["episodes"] == [0, 0, 0]
    fit = ra.load({"x": {"accel": got}}, "x")
    assert fit is not None and fit.scales["A-B"].shape == (len(ra.SCALES), 2)
    assert ra.load({"x": {}}, "x") is None
    with pytest.raises(ValueError, match="other scales"):
        ra.load({"x": {"accel": {**got, "scales_h": [1.0]}}}, "x")


def _fit() -> ra.Accel:
    return ra.Accel(scales={"A-B": np.ones((len(ra.SCALES), 2))}, ks=(2.0, 4.0, 8.0))


def test_live_reports_the_scale_and_velocity_change() -> None:
    read = _reader(dv_mm_d=24.0, onset=NOW - dt.timedelta(hours=60))
    got = ra.live(read, ["A-B", "NOFIT-B"], _fit(), NOW)
    assert got["level"] == 3
    by = got["set_by"]
    assert by["pair"] == "A-B" and by["component"] == "horizontal"
    assert by["dv_mm_per_day"] == pytest.approx(24.0, rel=0.3)
    assert [r["pair"] for r in got["pairs"]] == ["A-B"]  # no fit, no row


def test_live_without_data_has_no_level() -> None:
    def empty(pair: str, start: dt.datetime, end: dt.datetime) -> Any:
        return np.array([], "datetime64[ns]"), np.zeros((3, 0))

    got = ra.live(empty, ["A-B"], _fit(), NOW)
    assert got["level"] is None and got["pairs"][0]["stale"]


def test_area_status_takes_the_higher_rule() -> None:
    entry = {"monitoring": {"rtk": {"figure": "f.png", "baselines": ["A-B"],
             "horizontal_mm": [100, 200, 400], "up_mm": [100, 200, 400],
             "statistic": "accel"}}}  # fmt: skip
    spec = rs.rtk_spec(entry)
    assert spec is not None and spec["statistic"] == "accel"
    read = _reader(dv_mm_d=24.0, onset=NOW - dt.timedelta(hours=60))
    rec = rs.area_status("a", spec, ["A-B"], read, NOW)
    assert rec["rule"].endswith("(accel: not calibrated)") and rec["level"] == 0
    spec["accel_fit"] = _fit()
    rec = rs.area_status("a", spec, ["A-B"], read, NOW)
    assert rec["rule"] == "accel" and rec["level"] == 3
    assert rec["accel"]["set_by"]["scale_h"] in ra.SCALES


def test_thresholds_per_scale_in_mm_and_mm_per_day() -> None:
    q = np.ones((len(ra.SCALES), 2))
    q[:, 1] = 2.0
    fit = ra.Accel(scales={"A-B": q}, ks=(1.5, 2.0, 3.0))
    got = ra.thresholds(fit, ["A-B", "NONE-B"])
    assert list(got) == ["A-B"]
    row24 = got["A-B"][ra.SCALES.index(24.0)]
    assert row24["horizontal_mm"] == [1.5, 2.0, 3.0] and row24["up_mm"] == [
        3.0,
        4.0,
        6.0,
    ]
    row6 = got["A-B"][ra.SCALES.index(6.0)]
    assert row6["horizontal_mm_per_day"] == [6.0, 8.0, 12.0]
