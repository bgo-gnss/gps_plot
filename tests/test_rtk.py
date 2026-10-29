"""``gpsplot rtk``: patterns, directions, zeroing, the joins, spans, failures."""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt

import numpy as np
import pytest

from gps_plot import rtk
from gps_plot.series import Series
from gps_plot.sources import rtk as source

DIRS = ["SEY1-SEYD", "SEY2-SEYD", "SEY3-SEYD", "AUSV-ELDC", "ELDC-AUSV", "SKFC-SVIN"]


def test_expand_matches_existing_directories_only():
    assert source.expand_rtk(["SEY[1-2]-SEYD"], DIRS) == ["SEY1-SEYD", "SEY2-SEYD"]
    assert source.expand_rtk(["*-AUSV", "AUSV-*"], DIRS) == ["ELDC-AUSV", "AUSV-ELDC"]
    assert source.expand_rtk(["sey1-seyd", "SEY?-SEYD"], DIRS) == DIRS[:3]


def test_a_plain_pair_processed_the_other_way_is_kept_as_written():
    assert source.expand_rtk(["SVIN-SKFC"], DIRS) == ["SVIN-SKFC"]
    assert source.rtk_direction("SVIN-SKFC", DIRS) == ("SKFC-SVIN", True)
    assert source.rtk_direction("SEY1-SEYD", DIRS) == ("SEY1-SEYD", False)
    with pytest.raises(ValueError, match="no RTK baseline"):
        source.rtk_direction("SEY4-SEYD", DIRS)


def test_a_flipped_baseline_is_negated(monkeypatch):
    import geo_dataread.rtk_read as rr

    seen = {}

    def read(name, start, end, **kw):
        seen["name"] = name
        t = _minutes(3)
        return rr.RtkBaseline(name, t, np.ones((3, 3)) * [[0.001], [0.002], [0.0]],
                              np.full((3, 3), 0.001), np.ones(3, int), "60s")  # fmt: skip

    monkeypatch.setattr(rr, "read_rtk_baseline", read)
    s = source.fetch_rtk("SVIN-SKFC", None, None, directory="SKFC-SVIN", flipped=True)
    assert seen["name"] == "SKFC-SVIN" and s.label == "SVIN-SKFC"
    np.testing.assert_allclose(s.meta["level"], [-1.0, -2.0, 0.0])


@pytest.mark.parametrize(
    "spec, msg", [("SEY4-SEYD", "matches no RTK"), ("SEY1", "ROVER-BASE")]
)
def test_expand_refuses(spec, msg):
    with pytest.raises(ValueError, match=msg):
        source.expand_rtk([spec], DIRS)


def _minutes(n: int, start: str = "2026-10-01T00:00") -> np.ndarray:
    return np.datetime64(start, "ns") + np.arange(n) * np.timedelta64(1, "m")


def test_zero_level_is_the_weighted_mean_of_the_first_hour():
    t = _minutes(180)
    y = np.vstack([np.full(180, 5.0), np.arange(180.0), np.full(180, -2.0)])
    sig = np.ones((3, 180))
    level = source.zero_level(t, y, sig)
    np.testing.assert_allclose(level, [5.0, np.mean(np.arange(61.0)), -2.0])
    assert np.all(source.zero_level(t[:0], y[:, :0], sig[:, :0]) == 0)


def test_align_daily_is_the_median_over_the_overlap():
    t_rtk = _minutes(5 * 1440)
    rtk_line = Series("A-B", t_rtk, np.vstack([np.full(t_rtk.size, 10.0)] * 3),
                      None, meta={"source": "rtk"})  # fmt: skip
    t_day = np.datetime64("2026-09-29T12:00", "ns") + np.arange(8) * np.timedelta64(
        1, "D"
    )
    y_day = np.vstack([np.arange(8.0), np.zeros(8), np.full(8, 3.0)])
    daily = Series("A-B daily", t_day, y_day, np.ones((3, 8)))
    got = source.align_daily(daily, rtk_line)
    inside = (t_day >= t_rtk[0]) & (t_day <= t_rtk[-1])
    assert got.meta["aligned"]["days"] == int(inside.sum()) == 5
    np.testing.assert_allclose(
        got.meta["aligned"]["shift"], [10 - np.median(np.arange(8.0)[inside]), 10, 7]
    )
    np.testing.assert_allclose(got.y, y_day + got.meta["aligned"]["shift"][:, None])


def test_align_daily_without_overlap_is_an_error():
    rtk_line = Series("A-B", _minutes(60), np.zeros((3, 60)), None)
    daily = Series("d", np.array(["2020-01-01T12:00"], dtype="datetime64[ns]"),
                   np.zeros((3, 1)), np.ones((3, 1)))  # fmt: skip
    with pytest.raises(ValueError, match="no daily solution"):
        source.align_daily(daily, rtk_line)


def test_align_moves_an_rtk_tail_onto_daily_points():
    t_day = np.datetime64("2026-09-01T12:00", "ns") + np.arange(40) * np.timedelta64(
        1, "D"
    )
    daily = Series("A-B daily", t_day, np.vstack([np.full(40, 4.0)] * 3),
                   np.ones((3, 40)))  # fmt: skip
    t_rtk = _minutes(3 * 1440, "2026-10-08T00:00")  # last 3 days of the daily
    tail = Series("A-B", t_rtk, np.vstack([np.full(t_rtk.size, -1.0)] * 3), None)
    got = source.align(tail, daily)
    assert got.meta["aligned"]["days"] == 3  # the noons of Oct 8, 9 and 10
    np.testing.assert_allclose(got.meta["aligned"]["shift"], [5.0, 5.0, 5.0])
    with pytest.raises(ValueError, match="do not overlap"):
        source.align(
            dataclasses.replace(tail, t=t_rtk + np.timedelta64(30, "D")), daily
        )


def _args(**kw):
    base = dict(start=None, end=None, special=None, keep_single=False,
                fixed_only=False, longterm=None, rtk_span=None,
                rtk_overlap=None)  # fmt: skip
    return argparse.Namespace(**{**base, **kw})


def test_span_defaults_to_two_days_before_now():
    now = dt.datetime(2026, 10, 14, 11, 0)
    assert rtk.resolve_span(_args(), now) == (
        now - dt.timedelta(days=2),
        now,
        "twodays",
    )
    end = dt.datetime(2026, 10, 10, 12)
    assert rtk.resolve_span(_args(end=end, special="6h"), now)[:2] == (
        end - dt.timedelta(hours=6),
        end,
    )
    s = dt.datetime(2026, 10, 1)
    assert rtk.resolve_span(_args(start=s, end=end), now) == (s, end, None)


def test_longterm_spans():
    now = dt.datetime(2026, 10, 29, 12, 0)
    assert rtk.resolve_span(_args(longterm="year"), now) == (
        now - dt.timedelta(days=7), now, "week",
    )  # fmt: skip
    assert rtk.resolve_span(_args(longterm="90d", rtk_span="day"), now)[0] == (
        now - dt.timedelta(days=1)
    )
    assert rtk.longterm_start("year", now) == now - dt.timedelta(days=365)
    assert rtk.longterm_start("full", now) == rtk.FULL_START


def test_rtk_tail_just_overlaps_the_daily():
    start, plot_start = dt.datetime(2026, 10, 22), dt.datetime(2026, 10, 15)
    daily = Series("d", np.array(["2026-10-27T12:00"], dtype="datetime64[ns]"),
                   np.zeros((3, 1)), np.ones((3, 1)))  # fmt: skip
    assert rtk.rtk_start(start, plot_start, daily, 2) == dt.datetime(2026, 10, 25, 12)
    assert rtk.rtk_start(start, plot_start, daily, 30) == plot_start
    assert rtk.rtk_start(start, plot_start, daily, None) == start
    assert rtk.rtk_start(start, plot_start, None, 2) == start


def test_quality_flags():
    assert rtk.drop_q(_args()) == (5,)  # float too, as rtk_gps (BGÓ 2026-10-29)
    assert rtk.drop_q(_args(fixed_only=True)) == (2, 3, 4, 5, 6)
    assert rtk.drop_q(_args(keep_single=True)) == ()


def test_time_arguments():
    assert rtk._when("20261013-06:30") == dt.datetime(2026, 10, 13, 6, 30)
    assert rtk._when("20261013") == dt.datetime(2026, 10, 13)
    with pytest.raises(argparse.ArgumentTypeError):
        rtk._when("2026-10-13")


def test_baseline_index_groups_daily_with_its_rtk_line():
    s = [Series(n, None, None, None, meta={"source": src})
         for n, src in [("a", "rtk"), ("a d", None), ("b", "rtk"), ("b d", None)]]  # fmt: skip
    assert rtk.baseline_index(s) == [0, 0, 1, 1]
    # --longterm: daily first, grouped by pair
    s = [Series(n, None, None, None, meta={"pair": p})
         for n, p in [("a d", "a"), ("a", "a"), ("b d", "b"), ("b", "b")]]  # fmt: skip
    assert rtk.baseline_index(s) == [0, 0, 1, 1]


def _longterm(monkeypatch, daily_t, rtk_t):
    def daily(name, start, end, args, config):
        return Series(f"{name} daily", daily_t, np.zeros((3, daily_t.size)),
                      np.ones((3, daily_t.size)))  # fmt: skip

    def fetch(name, start, end, **kw):
        return Series(name, rtk_t, np.ones((3, rtk_t.size)), None,
                      meta={"source": "rtk"})  # fmt: skip

    monkeypatch.setattr(rtk, "_daily", daily)
    monkeypatch.setattr(source, "fetch_rtk", fetch)
    args = _args(max_sigma=0.1)
    return rtk._longterm_pair("SVIN-SKFC", None, None, None, args, None, "/r",
                              DIRS, "10min")  # fmt: skip


def test_longterm_pair_aligns_the_rtk_tail_on_the_daily(monkeypatch):
    days = np.datetime64("2026-10-20T12:00", "ns") + np.arange(5) * np.timedelta64(
        1, "D"
    )
    got, out = _longterm(monkeypatch, days, _minutes(2 * 1440, "2026-10-22T00:00"))
    assert [style for _, style in got] == ["marked", "line"]
    assert all(s.meta["pair"] == "SVIN-SKFC" for s, _ in got)
    np.testing.assert_allclose(got[1][0].y, 0.0)
    assert out["notes"] == ["SVIN-SKFC -1/-1/-1 (2 d)"] and not out["dropped"]


def test_longterm_pair_without_overlap_keeps_its_own_zero(monkeypatch):
    days = np.array(["2026-10-20T12:00"], dtype="datetime64[ns]")
    got, out = _longterm(monkeypatch, days, _minutes(60, "2026-10-29T00:00"))
    np.testing.assert_allclose(got[1][0].y, 1.0)  # not moved
    assert "NOT aligned" in out["notes"][0]


class _Config:
    def getPostProcessDir(self, option):
        return {"rtk_dir": "/cfg/rtk"}.get(option, "/tmp")


def test_run_drops_failures_and_exits_1_when_none_plot(monkeypatch, tmp_path, capsys):
    import geo_dataread.rtk_read as rr

    monkeypatch.setattr(rr, "list_baselines", lambda root=None: DIRS)

    def fail(*a, **k):
        raise FileNotFoundError("no files")

    monkeypatch.setattr(source, "fetch_rtk", fail)
    parser = rtk.build_parser(config=_Config())
    args = parser.parse_args(["SEY?-SEYD", "--save", "png", "-d", str(tmp_path)])
    assert rtk.run(args, parser, _Config()) == 1
    err = capsys.readouterr().err
    assert err.count("dropped") == 3 and "nothing plotted" in err


def test_rtk_dir_comes_from_the_config(monkeypatch, tmp_path):
    import geo_dataread.rtk_read as rr

    seen = {}

    def listing(root=None):
        seen["root"] = str(root)
        return DIRS

    monkeypatch.setattr(rr, "list_baselines", listing)
    monkeypatch.setattr(
        source, "fetch_rtk", lambda *a, **k: (_ for _ in ()).throw(OSError())
    )
    parser = rtk.build_parser(config=_Config())
    rtk.run(parser.parse_args(["SEY1-SEYD", "--save", "png"]), parser, _Config())
    assert seen["root"] == "/cfg/rtk"
    rtk.run(parser.parse_args(["SEY1-SEYD", "--rtk-dir", "/x", "--save", "png"]),
            parser, _Config())  # fmt: skip
    assert seen["root"] == "/x"


def test_a_line_breaks_at_gaps():
    import datetime as _dt

    from gps_plot.compose import break_gaps

    t0 = _dt.datetime(2026, 10, 27)
    x = [t0 + _dt.timedelta(minutes=10 * k) for k in (0, 1, 2, 3, 300, 301)]
    y = np.vstack([np.arange(6.0)] * 3)
    xs, ys = break_gaps(x, y)
    assert len(xs) == 7 and xs[4] == x[3]
    assert np.isnan(ys[:, 4]).all() and ys[0, 5] == 4.0
    assert break_gaps(x[:4], y[:, :4])[0] == x[:4]  # no gap, unchanged
