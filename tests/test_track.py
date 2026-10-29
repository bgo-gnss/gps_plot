"""map station --track: the numbers of a station's track (no GMT needed)."""

from __future__ import annotations

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
    assert colour == "black" and "2 days of data" in text
    assert re.search(r"SVIN moved 1[789] ± \d mm toward 29\d°", text)


def test_legend_lines_per_station() -> None:
    t, y = _series(move=(200.0, -120.0))
    tr = track.station_track("SVIE", -16.8, 64.0, t, y, ONSET, PRE)
    lines = track.legend_entries([tr], ONSET, 14)
    assert [c for _, c in lines] == [track.SINCE_COLOR, track.RECENT_COLOR]
    assert "since 23.09.2026" in lines[0][0] and "last 14 days" in lines[1][0]
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
