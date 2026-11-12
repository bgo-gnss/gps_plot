"""rtk_tie: RTK within the day, tied to the daily positions."""

from __future__ import annotations

import datetime as dt

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


def test_onset_direction_of_the_movement_since() -> None:
    days = pd.date_range("2026-09-23", periods=40, freq="1D") + rt.NOON
    a = np.radians(305.0)
    x = np.linspace(0.0, 30.0, 40)
    v = pd.DataFrame({"N": x * np.cos(a), "E": x * np.sin(a), "U": 0.0}, index=days)
    sig = pd.DataFrame({"N": 1.0, "E": 1.0, "U": 3.0}, index=days)
    got = rt.onset_direction(v, sig, dt.date(2026, 9, 23))
    assert got is not None and abs(got - 305.0) < 0.5
    still = v * 0.0
    assert rt.onset_direction(still, sig, dt.date(2026, 9, 23)) is None


def test_line_sigma_ignores_slow_motion() -> None:
    t = pd.date_range("2026-11-01", periods=24 * 60 * 2, freq="1min")
    rng = np.random.default_rng(5)
    slow = np.linspace(0.0, 50.0, t.size)  # 25 mm/d
    df = pd.DataFrame({c: slow + rng.normal(0, 3.0, t.size) for c in rt.COMP}, index=t)
    got = rt.line_sigma(df, None, sub_day=True)
    assert np.all(np.abs(got - 3.0) < 0.4)


def test_figure_drops_an_empty_series() -> None:
    """A 6 h span after noon has no daily point: that series drops out
    instead of failing the figure."""
    import datetime as dt

    import matplotlib

    matplotlib.use("Agg")
    idx = pd.date_range("2026-11-12 12:00", periods=60, freq="1min")
    full = pd.DataFrame(np.ones((60, 3)), index=idx, columns=["n", "e", "u"])
    empty = full.iloc[:0]
    ser = [rt._series("RTK", full, None, None, "rtk"),
           rt._series("daily", empty, None, None, "daily")]  # fmt: skip
    fig = rt.figure(ser, ["line", "marked"], ["red", "red"], main="SVIN",
                         sub="test", start=pd.Timestamp(idx[0]), end=pd.Timestamp(idx[-1]),
                         last=idx[-1].to_pydatetime(), now=dt.datetime(2026, 11, 12, 13, 0),
                         rtk=True)  # fmt: skip
    assert fig is not None


def test_fit_line_recovers_the_slope_and_level() -> None:
    t = pd.date_range("2026-11-01", periods=48, freq="1h")
    x = np.arange(48) / 24.0
    df = pd.DataFrame({"N": 2.0 * x + 5.0, "E": -x, "U": np.nan}, index=t)
    got = rt.fit_line(df, t[0], t[-1])
    assert got is not None
    slope, mid, tc = got
    assert slope[:2] == pytest.approx([2.0, -1.0])
    assert np.isnan(slope[2])  # no Up values: no Up line
    at = (tc - t[0]) / pd.Timedelta(days=1)
    assert mid[0] == pytest.approx(2.0 * at + 5.0)
    assert rt.fit_line(df, t[-1], t[-1]) is None  # under three values


def test_read_rates(tmp_path) -> None:
    import json

    path = tmp_path / "r.json"
    path.write_text(json.dumps({"rovers": {"SVIN": {
        "pair": "SVIN-SKFC",
        "daily": {"window_days": 14, "end": "2026-11-11T11:59", "shown": True,
                  "speed": 1.0},
        "rtk": {"window_days": 0.25, "end": "2026-11-12T22:30", "shown": False,
                "speed": 2.0},
    }}}))  # fmt: skip
    got = rt.read_rates(str(path))
    assert got["SVIN"]["daily"] == rt.RateWindow(
        14.0, pd.Timestamp("2026-11-11 11:59"), True
    )
    assert got["SVIN"]["rtk"].days == 0.25 and not got["SVIN"]["rtk"].shown
    assert rt.read_rates(None) == {}


@pytest.mark.parametrize("shown", [True, False])
def test_rate_lines_draw_over_the_window(shown: bool) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1)
    for ax in axes:
        ax.set_ylim(-10, 10)
    t = pd.date_range("2026-11-10", periods=72, freq="1h")
    df = pd.DataFrame({"N": np.arange(72) / 24.0, "E": 0.0, "U": 0.0}, index=t)
    win = rt.RateWindow(1.0, t[-1], shown)
    rt.rate_lines(fig, [(df, win, "red", "SVIN")], t[0])
    lines = [ln for ln in axes[0].get_lines() if ln.get_color() == "red"]
    assert len(lines) == 1
    assert lines[0].get_linestyle() == ("-" if shown else "--")
    texts = [x.get_text() for x in axes[0].get_legend().get_texts()]
    assert any("SVIN 24 h: +1.00 mm/d" in x for x in texts)
    assert ("below its test" in texts[0]) is (not shown)
    assert axes[0].get_ylim() == (-10, 10)
    plt.close(fig)


def test_tick_labels_past_the_panel_edge_are_hidden() -> None:
    from matplotlib.figure import Figure

    fig = Figure(figsize=(6, 6))
    axes = [fig.add_subplot(3, 1, i + 1) for i in range(3)]
    for ax in axes:
        ax.set_xlim(0, 10)
        ax.set_xticks(
            [0.2, 5, 10], ["a long left label", "middle", "a long right label"]
        )
    rt.trim_tick_labels(fig)
    shown = [lab.get_text() for lab in axes[0].get_xticklabels() if lab.get_visible()]
    assert shown == ["middle"]


def test_every_figure_saves_at_one_width() -> None:
    from matplotlib.figure import Figure

    fig = Figure(figsize=(13, 20))
    fig.add_subplot(1, 1, 1).text(
        1.2, 0.5, "far right", transform=fig.axes[0].transAxes
    )
    bb = rt.save_bbox(fig)
    assert (bb.x0, bb.x1) == rt.SAVE_X
