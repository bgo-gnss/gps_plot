"""gps_plot.sources.gps: a station or a baseline as a Series."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from gps_plot import timesmatplt as tplt
from gps_plot.series import NEU, Series
from gps_plot.sources import gps

REQ = gps.SeriesRequest(
    fstart=None, fend=None, ref="plate", Dir=None, tType="POS", uncert=15, view="raw"
)


def _fake_station_series(monkeypatch: pytest.MonkeyPatch, **per_station: Any) -> list:
    calls: list[tuple[str, dict[str, Any]]] = []

    def fake(sta: str, **kw: Any) -> tplt.StationSeries:
        calls.append((sta, kw))
        t = np.linspace(2025.0, 2025.1, 10)
        y = np.vstack([np.arange(10.0) * (i + 1) for i in range(3)])
        y = y + per_station.get(sta, {}).get("shift", 0.0)
        return tplt.StationSeries(
            t,
            y,
            np.ones((3, 10)),
            None,
            None,
            per_station.get(sta, {}).get("aborted"),
            per_station.get(sta, {}).get("detrended", False),
        )

    monkeypatch.setattr(tplt, "station_series", fake)
    return calls


def test_station_passes_the_arrays_through(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _fake_station_series(monkeypatch)
    s = gps.fetch("HOFN", REQ)
    assert isinstance(s, Series)
    assert (s.label, s.components, s.ends) == ("HOFN", NEU, None)
    assert s.y.shape == (3, 10)
    assert calls[0][0] == "HOFN"
    assert calls[0][1]["pos_id"] is None and calls[0][1]["ref"] == "plate"


def test_request_replace_keeps_the_rest(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _fake_station_series(monkeypatch)
    gps.fetch("HOFN", REQ.replace(pos_id="imo.08hr_itrf20"))
    assert calls[0][1]["pos_id"] == "imo.08hr_itrf20"
    assert calls[0][1]["uncert"] == 15


def test_baseline_differences_the_ends(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_station_series(
        monkeypatch,
        AAAA={"shift": 5.0, "aborted": [True, False, False]},
        BBBB={"aborted": [False, False, True]},
    )
    s = gps.fetch("AAAA-BBBB", REQ)
    assert s.label == "AAAA-BBBB"
    np.testing.assert_allclose(s.y, 0.0, atol=1e-12)  # same ramp, re-zeroed
    np.testing.assert_allclose(s.sigma, np.sqrt(2.0))  # formal quadrature
    assert list(s.aborted) == [True, False, True]
    assert s.outliers is None and s.provisional is None
    assert s.ends is not None and [e.label for e in s.ends] == ["AAAA", "BBBB"]
    assert s.meta["rho"] is None and s.meta["sigma_note"] is None


def test_detrended_baseline_needs_both_records(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_station_series(monkeypatch, AAAA={"detrended": True})
    req = REQ.replace(ref="detrend")
    with pytest.raises(ValueError, match="BBBB has none"):
        gps.fetch("AAAA-BBBB", req)
    # the 8 h overlay does not repeat the check (the main series did)
    s = gps.fetch("AAAA-BBBB", req, gps.BaselineOptions(check_view=False))
    assert s.detrend_applied is False


def test_join_baseline_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_station_series(monkeypatch)
    with pytest.raises(ValueError, match="JOIN"):
        gps.fetch("AAAA-BBBB", REQ.replace(tType="JOIN"))


def test_outlier_params_reach_station_series_as_the_same_object(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gps_analysis.outliers import OutlierParams

    calls = _fake_station_series(monkeypatch)
    params = OutlierParams(window_n_sigma=3.0)
    gps.fetch("AAAA-BBBB", REQ.replace(view="cleaned", outlier_params=params))
    assert [kw["outlier_params"] for _, kw in calls] == [params, params]
    assert all(kw["outlier_params"] is params for _, kw in calls)


def test_baseline_carries_the_ends_outliers_as_its_own(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A day set aside at either end is drawn on the baseline (grey), at
    the baseline's value with the flagged value put back -- also when the
    flagged day is the first common one (the zero moves)."""
    t = np.linspace(2025.0, 2025.1, 10)
    ramp = np.vstack([np.arange(10.0)] * 3)

    def fake(sta: str, **kw: Any) -> tplt.StationSeries:
        y = ramp * (2.0 if sta == "AAAA" else 1.0)  # baseline = the ramp
        flagged = np.full((3, 10), np.nan)
        day = 4 if sta == "AAAA" else 0  # BBBB: its first day is an outlier
        y = y.copy()
        flagged[2, day] = y[2, day] + (30.0 if sta == "AAAA" else -5.0)
        y[2, day] = np.nan
        sig = np.ones((3, 10))
        return tplt.StationSeries(
            t,
            y,
            sig,
            (flagged, np.where(np.isfinite(flagged), 2.0, np.nan)),
            None,
            None,
            False,
        )

    monkeypatch.setattr(tplt, "station_series", fake)
    s = gps.fetch("AAAA-BBBB", REQ)
    assert s.outliers is not None
    oy, ody = s.outliers
    assert np.isnan(oy[:2]).all()  # N and E had nothing set aside
    got = np.flatnonzero(np.isfinite(oy[2]))
    assert list(got) == [0, 4]
    # the cleaned Up starts at day 1 (day 0 is NaN): its zero is ramp[1]=1
    clean_up = s.y[2]
    assert np.isnan(clean_up[0]) and clean_up[1] == pytest.approx(0.0)
    # day 4: AAAA +30 -> baseline 4 + 30, on the cleaned zero (-1)
    assert oy[2, 4] == pytest.approx(4.0 + 30.0 - 1.0)
    # day 0: BBBB -5 -> A - B = 0 - (0 - 5) = +5, on the cleaned zero
    assert oy[2, 0] == pytest.approx(5.0 - 1.0)
    assert ody[2, 4] == pytest.approx(np.hypot(2.0, 1.0))  # flagged σ + the other end


def test_baseline_without_flags_has_no_outliers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_station_series(monkeypatch)
    assert gps.fetch("AAAA-BBBB", REQ).outliers is None


def test_start_zero_leaves_an_outlier_among_the_first_samples_out():
    import numpy as np

    from gps_plot.sources import gps as source

    y = np.array([190.0, -140.0, -139.0, -141.0, -140.5, -139.5, -140.2, -138.0])
    sigma = np.full_like(y, 3.0)
    idx = source.first_clean(y, sigma, np.isfinite(y))
    assert 0 not in idx and len(idx) == source.ZERO_SAMPLES
    # no outlier: the plain first samples
    y[0] = -140.0
    assert list(source.first_clean(y, sigma, np.isfinite(y))) == [0, 1, 2, 3, 4]
