"""rtk_tie: RTK within the day, tied to the daily positions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gps_plot import rtk_tie as rt


def _daily(days: pd.DatetimeIndex, v: float) -> pd.DataFrame:
    x = np.arange(len(days), dtype=float) * v
    return pd.DataFrame({"N": x, "E": -x, "U": 0.0 * x}, index=days + rt.NOON)


def _hourly(
    start: str, hours: int, v_day: float, offset: float = 100.0
) -> pd.DataFrame:
    t = pd.date_range(start, periods=hours, freq="1h") + pd.Timedelta("1h")
    x = offset + v_day * ((t - pd.Timestamp(start)) / pd.Timedelta("1D")).to_numpy()
    return pd.DataFrame({"N": x, "E": -x, "U": 0.0 * x}, index=t.as_unit("ns"))


def test_day_medians_at_noon_and_need_hours() -> None:
    b = _hourly("2026-11-01", 24 * 3 + 5, 0.0)
    m = rt.day_medians(b)
    assert list(m.index) == [pd.Timestamp(f"2026-11-0{d} 12:00") for d in (1, 2, 3)]


def test_tie_follows_daily_and_adds_motion_after_the_last_day() -> None:
    days = pd.date_range("2026-11-01", periods=4, freq="1D")
    daily = _daily(days, 2.0)  # 2 mm/d in the frame
    b = _hourly("2026-11-01", 24 * 6, 2.0)  # the pair: same motion, other zero
    got = rt.tie(daily, b)
    assert got is not None
    s, newest = got
    assert newest == days[-1] + rt.NOON
    # between noons the tied series is the daily line (the RTK offset gone)
    np.testing.assert_allclose(s.loc["2026-11-02 12:00":"2026-11-04 12:00", "N"],
                               2.0 * ((s.loc["2026-11-02 12:00":"2026-11-04 12:00"].index
                                       - (days[0] + rt.NOON)) / pd.Timedelta("1D")), atol=0.2)  # fmt: skip
    # after the newest daily day: that day + the RTK change since (2 mm/d on)
    # (± the half hour by which end-labelled 1-h bins centre after noon)
    assert abs(s.loc["2026-11-06 12:00", "N"] - (6.0 + 2.0 * 2)) < 0.1


def test_rotate_along_motion() -> None:
    df = pd.DataFrame({"N": [0.0], "E": [1.0], "U": [2.0]})
    r = rt.rotate(df, 90.0)
    assert r["along"].iloc[0] == 1.0 and abs(r["across"].iloc[0]) < 1e-12
    assert r["U"].iloc[0] == 2.0


def test_zeroed_needs_three_solutions() -> None:
    days = pd.DatetimeIndex(["2026-09-01", "2026-09-20", "2026-09-21", "2026-09-22"])
    df = pd.DataFrame({"N": [10.0, 0.0, 1.0, 2.0], "E": 0.0, "U": 0.0}, index=days)
    z = rt._zeroed(df, pd.Timestamp("2026-09-01"))
    assert z["N"].iloc[0] == 9.0  # zero = median(10, 0, 1), not the lone 10


def test_clip_to_main_range() -> None:
    main = pd.DataFrame({"N": [0.0, 10.0], "E": 0.0, "U": 0.0})
    other = pd.DataFrame({"N": [5.0, 100.0], "E": 0.0, "U": 0.0})
    got = rt._clip_to(main, other)
    assert got["N"].iloc[0] == 5.0 and np.isnan(got["N"].iloc[1])


def test_rtk_series_one_minute_and_hourly() -> None:
    t = pd.date_range("2026-11-01", "2026-11-12", freq="1min")
    rng = np.random.default_rng(0)
    y = rng.normal(0.0, 1.0, (3, t.size)) / 1000.0  # m, as the reader gives mm below
    y_mm = y * 1000.0

    def read(pair: str, a: object, b: object) -> object:
        keep = (t >= pd.Timestamp(a)) & (t <= pd.Timestamp(b))
        return t[keep].to_numpy(), y_mm[:, keep]

    start = pd.Timestamp("2026-11-11").to_pydatetime()
    end = pd.Timestamp("2026-11-12").to_pydatetime()
    one, hour = rt.rtk_series(read, "A-B", start, end)
    assert len(one) > 1000 and pd.Series(one.index).diff().median() == pd.Timedelta(
        "1min"
    )
    assert pd.Series(hour.index).diff().median() == pd.Timedelta("1h")
    assert one.index[0] >= pd.Timestamp(start) and list(one.columns) == list(rt.COMP)


def test_route_station_anchor_and_sign() -> None:
    assert rt._route("SVIN-SVIE", ["SVIN", "SVIE"]) == ("SVIN", "SVIE", 1.0)
    assert rt._route("svie-skfc", ["SVIN", "SVIE"]) == ("SVIE", "SKFC", 1.0)
    assert rt._route("SKFC-SVIE", ["SVIE"]) == ("SVIE", "SKFC", -1.0)
    with pytest.raises(ValueError, match="neither end"):
        rt._route("AAAA-BBBB", ["SVIN"])


def test_empirical_sigma_from_day_to_day_scatter() -> None:
    rng = np.random.default_rng(3)
    n = 400
    days = np.arange(738000, 738000 + n)
    v = pd.DataFrame({c: 0.05 * np.arange(n) + rng.normal(0, 2.0, n) for c in rt.COMP},
                     index=days)  # fmt: skip
    formal = pd.DataFrame({c: np.full(n, 0.5) for c in rt.COMP}, index=days)
    got = rt.empirical_sigma(v, formal)
    # 2 mm white noise behind a slow trend; the formal 0.5 mm scaled up to it
    assert all(abs(float(got[c].median()) - 2.0) < 0.3 for c in rt.COMP)
    few = rt.empirical_sigma(v.iloc[:5], formal.iloc[:5])
    assert (few == 0.5).all().all()  # too few days: formal kept
