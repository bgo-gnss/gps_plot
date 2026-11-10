"""rtk_status: an area's RTK warning (status file) and the bench replay."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from gps_plot import rtk_status as rs

NOW = dt.datetime(2026, 11, 10, 12, 0)
LIMITS = {"horizontal_mm": [10, 20, 40], "up_mm": [20, 40, 80]}


def _entry(**rtk: Any) -> dict[str, Any]:
    return {"monitoring": {"status": "x", "rtk": {"figure": "f.png", **rtk}}}


def _series(step_mm: float, at: dt.datetime, newest: dt.datetime = NOW) -> Any:
    """One-minute epochs over 8 h up to ``newest``, ±1 mm noise, a horizontal
    (north) step at ``at``."""
    t = np.arange(
        np.datetime64(newest - dt.timedelta(hours=8), "m"),
        np.datetime64(newest, "m") + 1,
    ).astype("datetime64[ns]")
    rng = np.random.default_rng(0)
    y = rng.normal(0.0, 1.0, (3, t.size))
    y[0, t >= np.datetime64(at, "ns")] += step_mm
    return t, y


def test_spec_needs_limits_and_checks_them() -> None:
    assert rs.rtk_spec({}) is None
    assert rs.rtk_spec(_entry()) is None  # a figure alone: no limits, no status
    spec = rs.rtk_spec(_entry(baselines=["A-B"], **LIMITS))
    assert spec is not None and spec["window_minutes"] == 60
    assert spec["horizontal_mm"] == [10, 20, 40] and spec["baselines"] == ["A-B"]
    with pytest.raises(ValueError):
        rs.rtk_spec(_entry(horizontal_mm=[20, 10]))
    with pytest.raises(ValueError):
        rs.rtk_spec(_entry(horizontal_mm=[10], window_minutes=0))


def test_area_status_levels_and_set_by() -> None:
    spec = rs.rtk_spec(_entry(**LIMITS))
    assert spec is not None
    data = {
        "QUIE-BASE": _series(0.0, NOW),
        "MOVE-BASE": _series(25.0, NOW - dt.timedelta(minutes=30)),
    }

    def read(pair: str, start: dt.datetime, end: dt.datetime) -> Any:
        return data[pair]

    rec = rs.area_status("a", spec, ["QUIE-BASE", "MOVE-BASE"], read, NOW)
    assert rec["schema"] == rs.SCHEMA and rec["provisional"] is True
    assert rec["level"] == 2 and rec["level_name"] == "orange"  # 25 mm ≥ 20
    assert rec["set_by"]["pair"] == "MOVE-BASE"
    assert rec["set_by"]["component"] == "horizontal"
    quiet = next(r for r in rec["baselines"] if r["pair"] == "QUIE-BASE")
    assert quiet["level"] == 0 and quiet["n_window"] == 60


def test_old_or_missing_data_is_grey_never_green() -> None:
    spec = rs.rtk_spec(_entry(**LIMITS))
    assert spec is not None
    old = _series(0.0, NOW, newest=NOW - dt.timedelta(hours=1))

    def read(pair: str, start: dt.datetime, end: dt.datetime) -> Any:
        if pair == "GONE-BASE":
            raise ValueError("no RTK baseline")
        return old

    rec = rs.area_status("a", spec, ["OLDD-BASE", "GONE-BASE"], read, NOW)
    assert rec["level"] is None and rec["level_name"] == "grey"
    rows = {r["pair"]: r for r in rec["baselines"]}
    assert rows["OLDD-BASE"]["stale"] is True and "error" in rows["GONE-BASE"]


def test_write_status_is_atomic_json(tmp_path: Path) -> None:
    path = tmp_path / "a" / "graphs" / "rtk" / rs.STATUS_NAME
    rs.write_status(path, {"schema": 1, "level": 0})
    assert json.loads(path.read_text()) == {"schema": 1, "level": 0}
    assert not list(path.parent.glob(".*tmp"))


def test_bench_replay_finds_the_first_time_of_each_level() -> None:
    spec = rs.rtk_spec(_entry(**LIMITS))
    assert spec is not None
    onset = NOW - dt.timedelta(hours=2)
    t, y = _series(50.0, onset)
    grid = [
        onset - dt.timedelta(minutes=30) + dt.timedelta(minutes=5 * k)
        for k in range(20)
    ]
    first = rs.first_times(rs.replay(rs.minutes(t), y, grid, spec))
    # the window's MEDIAN flips once more than half of its 60 min is past the
    # step: a sudden step shows ~W/2 after it (T3 adds the short windows)
    assert all(when > onset for when in first.values())
    assert first[1] <= first[2] <= first[3]
    assert dt.timedelta(minutes=25) <= first[3] - onset <= dt.timedelta(minutes=35)


def test_onset_of_reads_onset_and_time() -> None:
    ev = [
        {"id": "e", "onset": "2023-11-10", "onset_time": "15:00"},
        {"id": "f", "onset": "2024-01-01"},
    ]
    assert rs.onset_of(ev, "e") == dt.datetime(2023, 11, 10, 15, 0)
    with pytest.raises(ValueError):
        rs.onset_of(ev, "f")
    with pytest.raises(ValueError):
        rs.onset_of(ev, "nope")
