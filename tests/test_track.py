"""map station --track: the numbers of a station's track (no GMT needed)."""

from __future__ import annotations

import dataclasses
import datetime as dt
import math
import re

import numpy as np
import pytest

from gps_plot import map as gmap
from gps_plot import track

ONSET = dt.date(2026, 9, 23)
PRE = (dt.date(2024, 1, 1), dt.date(2026, 9, 16))


def _series(
    move: tuple[float, float] = (0.0, 0.0),
    noise: float = 1.0,
    days_after: int = 35,
    seed: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Daily N/E: a trend, a seasonal and noise; after the onset a steady
    motion of ``move`` [mm/yr] on top."""
    rng = np.random.default_rng(seed)
    first = track.yearf(PRE[0])
    n = int((track.yearf(ONSET) - first) * 365.25) + days_after
    t = first + np.arange(n) / 365.25
    y = np.vstack([
        -8.0 * (t - 2025) + 1.5 * np.sin(2 * np.pi * t),
        3.0 * (t - 2025) + 1.0 * np.cos(2 * np.pi * t),
    ])  # fmt: skip
    after = np.clip(t - track.yearf(ONSET), 0, None)
    y += np.array(move)[:, None] * after
    y += rng.normal(0.0, noise, y.shape)
    return t, y


def test_trend_is_removed_and_rate_recovered() -> None:
    t, y = _series(move=(200.0, -120.0))
    tr = track.station_track("SVIE", -16.8, 64.0, t, y, ONSET, PRE)
    assert tr.days[0] == pytest.approx(-track.BEFORE_DAYS, abs=1)
    before = tr.days < 0
    assert abs(np.mean(tr.north[before])) < 1.0  # dots on the station
    assert tr.since is not None and tr.moved
    assert tr.since.north == pytest.approx(200, abs=4 * tr.since.sigma_north)
    assert tr.since.east == pytest.approx(-120, abs=4 * tr.since.sigma_east)
    assert tr.recent is not None


def test_a_still_station_has_not_moved() -> None:
    t, y = _series(seed=3)
    tr = track.station_track("SKFC", -17.0, 64.0, t, y, ONSET, PRE)
    assert not tr.moved
    assert track.legend_entries([tr], ONSET, 14) == []


def test_recent_sigma_is_floored_by_the_daily_scatter() -> None:
    """Six days can fit a line too well: the σ never drops below what the
    pre-onset scatter allows."""
    t, y = _series(seed=5)
    tr = track.station_track("SKFC", -17.0, 64.0, t, y, ONSET, PRE, recent_days=6)
    assert tr.recent is not None
    k = t >= t[-1] - 6 / 365.25
    tm = t[k] - t[k].mean()
    floor = 0.8 / math.sqrt(float(np.sum(tm**2)))  # noise 1 mm, a bit less
    assert tr.recent.sigma_north > floor


def test_too_few_days_gives_a_displacement() -> None:
    t, y = _series(days_after=40)
    keep = t < track.yearf(ONSET)
    keep[[-12, -5]] = True  # two days after the onset
    y = y.copy()
    y[:, -12:] += np.array([[7.0], [-17.0]])
    tr = track.station_track("SVIN", -16.8, 64.0, t[keep], y[:, keep], ONSET, PRE)
    assert tr.n_after == 2 and tr.since is None and tr.moved
    ((text, colour),) = track.legend_entries([tr], ONSET, 14)
    assert colour == track.SINCE_COLOR and "(2 days)" in text
    assert re.search(r"SVIN moved 1[789] ± \d mm toward 29\d°", text)


def test_legend_lines_per_station() -> None:
    t, y = _series(move=(200.0, -120.0))
    tr = track.station_track("SVIE", -16.8, 64.0, t, y, ONSET, PRE)
    lines = track.legend_entries([tr], ONSET, 14)
    assert [c for _, c in lines] == [track.SINCE_COLOR]
    assert "since 23.09.2026" in lines[0][0]
    # 200 N, -120 E mm/yr: toward 329° (north-west)
    assert re.search(r"toward 3[23]\d° since", lines[0][0])
    assert re.search(r", moved \d+ ± \d mm$", lines[0][0])  # the total so far


def test_no_data_after_the_onset_is_refused() -> None:
    t, y = _series()
    keep = t < track.yearf(ONSET)
    with pytest.raises(ValueError, match="no data since"):
        track.station_track("ROTH", -16.6, 64.0, t[keep], y[:, keep], ONSET, PRE)


def test_short_trend_window_is_refused() -> None:
    t, y = _series()
    with pytest.raises(ValueError, match="trend window"):
        track.station_track("X", 0, 64, t, y, ONSET, (dt.date(2026, 8, 1), PRE[1]))


def test_dot_scale_and_offsets() -> None:
    t, y = _series(move=(200.0, -120.0))
    tr = track.station_track("SVIE", -16.8, 64.0, t, y, ONSET, PRE)
    m_per_mm, ref = track.dot_scale([tr], km_per_cm=1.5)
    far = float(np.max(np.hypot(tr.north, tr.east)))
    assert far * m_per_mm == pytest.approx(track.TRACK_CM * 1500.0)
    assert ref <= far
    lon, lat = track.offset_lonlat(-16.8, 64.0, [10.0], [0.0], 100.0)
    assert lat[0] - 64.0 == pytest.approx(1.0 / 111.195)  # 10 mm -> 1 km north
    assert lon[0] == pytest.approx(-16.8)


def test_pre_window_default() -> None:
    a, b = track.pre_window(ONSET)
    assert (ONSET - b).days == track.PRE_GAP_DAYS
    assert (b - a).days == track.PRE_DAYS


def test_cli_track_options_and_onset(tmp_path) -> None:
    p = gmap.build_parser(config=None)
    args = p.parse_args(["station", "SVIE", "--track", "20260923",
                         "--track-pre", "20240101:20260916", "--no-track-arrows"])  # fmt: skip
    assert args.track == "20260923" and args.track_arrows is False
    assert args.track_recent == track.RECENT_DAYS
    assert gmap.track_onset("20260923") == (ONSET, "2026-09-23")
    f = tmp_path / "events.yaml"
    f.write_text("events:\n- {id: svinafellsheidi-2026-09-landslide, kind: landslide,"
                 " onset: 2026-09-23}\n")  # fmt: skip
    assert gmap.track_onset("svinafellsheidi", str(f)) == (
        ONSET,
        "svinafellsheidi-2026-09-landslide",
    )
    with pytest.raises(ValueError, match="no event"):
        gmap.track_onset("askja", str(f))


def test_locator_sits_on_a_floor() -> None:
    from gps_plot import insets

    spot = insets.corner_spot((4.0, 3.0), (15.5, 10.0), "BL", floor=3.2)
    assert spot.y > 3.2
    top = insets.corner_spot((4.0, 3.0), (15.5, 10.0), "TL", floor=3.2)
    assert top.y == pytest.approx(10.0 - 3.0 - 0.15)


def test_trimmed_keeps_the_rates_and_the_series_lines_match_them() -> None:
    t, y = _series(move=(200.0, -120.0))
    tr = track.station_track("SVIE", -16.8, 64.0, t, y, ONSET, PRE, before_days=90)
    short = track.trimmed(tr, 30)
    assert (
        short.days[0] >= -30 and short.since == tr.since and short.n_after == tr.n_after
    )
    assert tr.since is not None
    days, mm = track.rate_line(tr, 0, tr.since, 0.0)
    slope = (mm[1] - mm[0]) / (days[1] - days[0]) * track.YEAR
    assert slope == pytest.approx(tr.since.north)
    fig = track.series_figure(tr, ONSET, before_days=30)
    texts = [t.get_text() for ax in fig.axes[:2] for t in ax.texts]
    assert any(f"v = {tr.since.north:.0f} ±" in x for x in texts)


def test_running_median_smooths_a_spike_and_keeps_the_ends() -> None:
    y = np.array([0.0, 1.0, 2.0, 30.0, 4.0, 5.0, 6.0])
    m = track.running_median(y, 5)
    assert m[3] == 4.0 and m.size == y.size
    assert track.running_median(y, 1) is y


def test_relative_to_takes_the_shared_motion_out_on_common_days() -> None:
    t, y = _series(move=(200.0, -120.0), seed=7)
    shared = np.vstack([5.0 * np.sin(40 * t), 3.0 * np.cos(30 * t)])
    t_ref, y_ref = t[::2], shared[:, ::2]  # the reference has every 2nd day
    t2, d = track.relative_to(t, y + shared, t_ref, y_ref)
    assert t2.size == t_ref.size
    assert np.allclose(d, y[:, ::2])  # the shared signal gone
    a = track.station_track("SVIE", -16.8, 64.0, t2, d, ONSET, PRE)
    assert a.since is not None
    assert a.since.north == pytest.approx(200, abs=4 * a.since.sigma_north)


def test_reference_mean_shares_out_one_stations_wander() -> None:
    pre = (dt.date(2024, 1, 1), dt.date(2025, 1, 1))
    days = np.arange(0, 600)
    t = 2024.0 + (days + 0.5) / track.YEAR
    rng = np.random.default_rng(0)
    after = t >= 2025.0
    quiet = [(t, rng.normal(0, 1.0, (2, t.size)) + 5.0 * (t - 2024)) for _ in range(3)]
    moving = (t, quiet[0][1] + np.where(after, 30.0 * (t - 2025.0), 0.0))
    tm, ym = track.reference_mean([*quiet[1:], moving], pre)
    assert tm.size == t.size
    # the trend before is gone; a third of the mover's 30 mm/yr is left
    late = tm >= 2025.0
    slope = np.polyfit(tm[late], ym[0, late], 1)[0]
    assert slope == pytest.approx(10.0, abs=1.5)
    # a day only one of three has is left out (fewer than half)
    lone = (np.array([2030.0]), np.zeros((2, 1)))
    tm2, _ = track.reference_mean([*quiet[1:], (np.concatenate([t, lone[0]]),
                                                np.hstack([moving[1], lone[1]]))], pre)  # fmt: skip
    assert tm2.size == t.size
    with pytest.raises(ValueError):
        track.reference_mean([], pre)


def test_day_key_is_one_per_calendar_day_in_leap_years() -> None:
    days = [dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(800)]
    t = np.array([track.yearf(d) for d in days])
    key = track.day_key(t)
    assert np.array_equal(key, [d.toordinal() for d in days])
    np.testing.assert_allclose(track.from_day_key(key), t)


def test_screen_references_leaves_out_the_one_that_moves() -> None:
    pre = (dt.date(2024, 1, 1), dt.date(2025, 1, 1))
    onset = dt.date(2025, 1, 10)
    days = [dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(560)]
    t = np.array([track.yearf(d) for d in days])
    rng = np.random.default_rng(2)
    after = t >= track.yearf(onset)
    series = {
        f"Q{i}": (t, rng.normal(0, 1.0, (2, t.size)) + 4.0 * (t - 2024))
        for i in range(3)
    }
    mover = rng.normal(0, 1.0, (2, t.size))
    mover[1] += np.where(after, 200.0 * (t - track.yearf(onset)), 0.0)
    series["MOVE"] = (t, mover)
    kept, out = track.screen_references(series, pre, onset)
    assert kept == ["Q0", "Q1", "Q2"]
    assert [name for name, _ in out] == ["MOVE"]
    assert out[0][1] == pytest.approx(200.0, rel=0.15)
    # quiet ones all stay; two cannot be screened
    assert (
        track.screen_references({k: series[k] for k in ("Q0", "Q1", "Q2")}, pre, onset)[
            1
        ]
        == []
    )
    two = {k: series[k] for k in ("Q0", "MOVE")}
    assert track.screen_references(two, pre, onset) == (["Q0", "MOVE"], [])


def test_keep_trend_keeps_and_measures_the_motion_before() -> None:
    # creep of (-8, 3) mm/yr before the onset: removed by default, kept and
    # measured with keep_trend (the seasonal removed either way)
    t, y = _series(move=(200.0, -120.0), seed=4)
    tr = track.station_track("SVIE", -16.8, 64.0, t, y, ONSET, PRE, keep_trend=True)
    assert tr.before is not None
    assert tr.before.north == pytest.approx(-8.0, abs=4 * tr.before.sigma_north)
    assert tr.before.east == pytest.approx(3.0, abs=4 * tr.before.sigma_east)
    old = tr.days < -2
    slope = np.polyfit(tr.days[old] / 365.25, tr.north[old], 1)[0]
    assert slope == pytest.approx(-8.0, abs=8.0)  # walks, not flat
    assert any(text.startswith("SVIE before: N -8")
               for text, _ in track.legend_entries([tr], ONSET, 14))  # fmt: skip
    assert track.station_track("SVIE", -16.8, 64.0, t, y, ONSET, PRE).before is None


def test_detrend_keep_line_removes_only_the_seasonal() -> None:
    t, y = _series(noise=0.0)
    pre = t < track.yearf(PRE[1])
    r, rms = track.detrend(t, y, pre, keep_line=True)
    assert np.allclose(np.diff(r[0]) * 365.25, -8.0, atol=1e-6)
    assert np.allclose(rms, 0.0, atol=1e-9)


def test_reference_mean_keep_trend_carries_the_common_motion() -> None:
    a, b = _series(seed=5), _series(seed=6)
    t, mean = track.reference_mean([a, b], PRE, keep_trend=True)
    assert np.polyfit(t, mean[0], 1)[0] == pytest.approx(-8.0, abs=0.5)
    t, mean = track.reference_mean([a, b], PRE)
    assert abs(np.polyfit(t, mean[0], 1)[0]) < 0.5


def test_wander_scale_white_noise_near_one_random_walk_above() -> None:
    rng = np.random.default_rng(7)
    t = 2024 + np.arange(900) / 365.25
    pre = np.ones(t.size, dtype=bool)
    white = rng.normal(0, 1, (2, t.size))
    walk = np.cumsum(rng.normal(0, 0.3, (2, t.size)), axis=1)
    walk -= np.polyval(np.polyfit(t, walk[0], 1), t)
    assert np.all(track.wander_scale(t, white, pre, 35) < 1.5)
    assert track.wander_scale(t, walk + white, pre, 35)[0] > 1.5


def test_scaled_rate_and_none() -> None:
    r = track.Rate(10.0, -5.0, 1.0, 2.0)
    s = track.scaled(r, np.array([2.0, 3.0]))
    assert s is not None and (s.sigma_north, s.sigma_east) == (2.0, 6.0)
    assert (s.north, s.east) == (10.0, -5.0)
    assert track.scaled(None, np.ones(2)) is None


def test_inset_helpers() -> None:
    assert gmap._projection_width_cm("-R-16.8/-16.7/64/64.1 -JM6c") == 6.0
    assert gmap._projection_width_cm("-R0/1/0/1 -JM5.5c") == 5.5
    t, y = _series(move=(200.0, -120.0))
    tr = track.station_track("SVIE", -16.8, 64.0, t, y, ONSET, PRE)
    lo, hi = track.colour_range([tr])
    assert lo == pytest.approx(-track.BEFORE_DAYS, abs=1) and hi > 30
    scale, ref = track.arrow_scale_of([tr], 1.5)
    assert tr.since is not None
    assert scale * tr.since.speed == pytest.approx(1.5)
    assert ref == track.nice(tr.since.speed / 2)
    still = track.station_track("SKFC", -17.0, 64.0, *_series(seed=3), ONSET, PRE)
    assert track.arrow_scale_of([still]) == (0.01, 100.0)


def test_few_days_zero_extrapolated_gets_a_wider_sigma() -> None:
    """No days just before the onset: the zero is the trend carried across
    the gap, and its σ (fit prediction + 14-day wander) widens the
    displacement's; the newest position is not a mean of days a week apart."""
    t, y = _series(days_after=40)
    keep = t < track.yearf(ONSET) - 120 / 365.25  # a 4-month gap before
    keep[[-12, -5]] = True
    y = y.copy()
    y[:, -12:] += np.array([[7.0], [-17.0]])
    tr = track.station_track("SVIN", -16.8, 64.0, t[keep], y[:, keep], ONSET, PRE)
    assert tr.zero_extrapolated and tr.end_count == 1
    assert all(z > 0 for z in tr.zero_sigma)
    assert tr.end_sigmas[0] > tr.scatter_ne[0]
    ((text, _),) = track.legend_entries([tr], ONSET, 14)
    assert ", none just before)" in text


def test_extrapolation_sigma_grows_with_the_gap() -> None:
    t, y = _series()
    pre = (t >= track.yearf(PRE[0])) & (t < track.yearf(PRE[1]))
    near = track.extrapolation_sigma(t, y, pre, track.yearf(PRE[1]))
    far = track.extrapolation_sigma(t, y, pre, track.yearf(PRE[1]) + 1.0)
    assert np.all(far > near) and np.all(near > 0)


def test_a_few_days_station_is_listed_even_if_not_beyond_three_sigma() -> None:
    t, y = _series(days_after=40)
    keep = t < track.yearf(ONSET)
    keep[[-5]] = True  # one day after, on the station
    tr = track.station_track("SVIN", -16.8, 64.0, t[keep], y[:, keep], ONSET, PRE)
    assert not tr.moved
    ((text, colour),) = track.legend_entries([tr], ONSET, 14)
    assert colour == track.SINCE_COLOR and "(1 day)" in text


def test_newest_day_text_green_through_yesterday_else_red() -> None:
    today = dt.date(2026, 10, 29)
    assert track.newest_text(dt.date(2026, 10, 28), today) == (
        "last datapoint: 28.10.2026",
        track.FRESH_COLOR,
    )
    assert track.newest_text(dt.date(2026, 10, 27), today)[1] == track.STALE_COLOR
    t, y = _series()
    tr = track.station_track("SVIE", -16.8, 64.0, t, y, ONSET, PRE)
    assert track.newest_day([tr]) == dt.date.fromordinal(
        int(track.day_key(np.array([t[-1]]))[0])
    )
    assert track.newest_day([]) is None


def test_the_recent_rate_has_no_legend_line() -> None:
    t, y = _series(move=(200.0, -120.0))
    tr = track.station_track("SVIE", -16.8, 64.0, t, y, ONSET, PRE)
    assert tr.recent is not None
    assert not any(
        "last 14 days" in t for t, _ in track.legend_entries([tr], ONSET, 14)
    )


def test_zoom_dem_resamples_finer_and_smooths() -> None:
    from gps_plot import insets

    kw = insets.zoom_dem((-16.84, -16.785, 63.998, 64.022), "M6c", 0.0)
    assert 0 < kw["dem_resample"] < 1 / 3600
    assert kw["dem_smooth"] == insets.ZOOM_DEM_SMOOTH_KM


def test_vertical_track_measures_a_drop_since_the_onset() -> None:
    t, y = _series(days_after=40)
    rng = np.random.default_rng(7)
    u = (2.0 * np.sin(2 * np.pi * t) + rng.normal(0.0, 3.0, t.size))[None, :]
    u = u + np.where(t >= track.yearf(ONSET), -20.0, 0.0)  # a 20 mm drop
    vt = track.vertical_track(t, u, ONSET, PRE)
    assert vt.n_after > 30 and vt.since is not None and vt.moved
    assert vt.end == pytest.approx(-20.0, abs=4 * vt.end_sigma)
    assert track.up_text("SVIN", vt, ONSET).startswith("SVIN vertical -")


def test_legend_lists_a_station_that_moved_only_vertically() -> None:
    t, y = _series(seed=3)
    tr = track.station_track("SKFC", -17.0, 64.0, t, y, ONSET, PRE)
    rng = np.random.default_rng(2)
    u = rng.normal(0.0, 2.0, t.size)[None, :] + np.where(
        t >= track.yearf(ONSET), -30.0, 0.0
    )
    tr = dataclasses.replace(tr, vertical=track.vertical_track(t, u, ONSET, PRE))
    assert not tr.moved
    ((text, colour),) = track.legend_entries([tr], ONSET, 14)
    assert colour == track.SINCE_COLOR and "vertical" in text
    assert track.legend_entries([tr], ONSET, 14, up=False) == []


def test_up_series_keeps_the_days_with_an_up() -> None:
    t = np.arange(5.0)
    y = np.array([[1.0] * 5, [2.0] * 5, [3.0, np.nan, 4.0, np.nan, 5.0]])
    tu, u = track.up_series(t, y)
    assert tu.tolist() == [0.0, 2.0, 4.0] and u.shape == (1, 3)


def test_with_colours_one_per_listed_station_in_order() -> None:
    t, y = _series(move=(200.0, -120.0))
    a = track.station_track("SVIE", -16.8, 64.0, t, y, ONSET, PRE)
    t2, y2 = _series(seed=3)
    still = track.station_track("SKFC", -17.0, 64.0, t2, y2, ONSET, PRE)
    b = dataclasses.replace(a, marker="SVIN")
    got = track.with_colours([a, still, b])
    assert [tr.colour for tr in got] == ["blue", track.SINCE_COLOR, "red"]
    lines = track.legend_entries(got, ONSET, 14)
    assert [c for _, c in lines] == ["blue", "red"]
