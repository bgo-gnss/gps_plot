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
