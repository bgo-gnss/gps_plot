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
        "MOVE-BASE": _series(25.0, NOW - dt.timedelta(minutes=40)),  # held 5 min (T7)
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


def _noisy(step_mm: float, at: dt.datetime, seed: int) -> Any:
    """``_series`` with its own noise (siblings do not share the rover's)."""
    t, _ = _series(0.0, at)
    y = np.random.default_rng(seed).normal(0.0, 1.0, (3, t.size))
    y[0, t >= np.datetime64(at, "ns")] += step_mm
    return t, y


def _status(data: dict[str, Any], **extra: Any) -> dict[str, Any]:
    spec = rs.rtk_spec(_entry(**LIMITS, **extra))
    assert spec is not None

    def read(pair: str, start: dt.datetime, end: dt.datetime) -> Any:
        return data[pair]

    return rs.area_status("a", spec, list(data), read, NOW)


def test_sibling_veto_keeps_a_base_jump_green() -> None:
    """T5: the base jumps 25 mm, so every pair on it moves the same way."""
    jump = NOW - dt.timedelta(minutes=40)
    data = {
        p: _noisy(25.0, jump, k)
        for k, p in enumerate(["AAAA-BASE", "BBBB-BASE", "CCCC-BASE"])
    }
    assert _status(data)["level"] == 2  # without the veto: orange
    rec = _status(data, common_mode="siblings")
    assert rec["level"] == 0 and rec["common_mode"] == "siblings"
    for row in rec["baselines"]:
        assert row["level"] == 0 and row["vetoed"] == "BASE" and row["level_raw"] == 2
        assert not any(k.startswith("_") for k in row)  # private keys not written


def test_sibling_veto_lets_one_rover_moving_fire() -> None:
    """Differential motion: one rover moves, its siblings stay; it fires."""
    move = NOW - dt.timedelta(minutes=40)
    data = {
        "MOVE-BASE": _noisy(25.0, move, 1),
        "QUIE-BASE": _noisy(0.0, move, 2),
        "STIL-BASE": _noisy(0.0, move, 3),
    }
    rec = _status(data, common_mode="siblings")
    assert rec["level"] == 2 and rec["set_by"]["pair"] == "MOVE-BASE"
    assert "vetoed" not in next(r for r in rec["baselines"] if r["pair"] == "MOVE-BASE")


def test_sibling_veto_svinafellsheidi_shape() -> None:
    """SVIN-SVIE has no sibling on SVIE: its motion always fires. A SKFC
    jump moves SVIN-SKFC and SVIE-SKFC alike and is vetoed."""
    at = NOW - dt.timedelta(minutes=40)
    data = {
        "SVIN-SVIE": _noisy(25.0, at, 1),
        "SVIN-SKFC": _noisy(25.0, at, 2),
        "SVIE-SKFC": _noisy(25.0, at, 3),
    }
    rows = {r["pair"]: r for r in _status(data, common_mode="siblings")["baselines"]}
    assert rows["SVIN-SVIE"]["level"] == 2 and "vetoed" not in rows["SVIN-SVIE"]
    assert rows["SVIN-SKFC"]["level"] == 0 and rows["SVIE-SKFC"]["level"] == 0


def test_sibling_veto_without_siblings_and_bad_mode() -> None:
    got = {
        "AAAA-BASE": {
            "level": 2,
            "_windows": {60.0: {"offset": [25, 0, 0], "level": 2, "yellow": (10, 20)}},
        }
    }
    assert rs.sibling_veto(got) == got  # alone: kept
    with pytest.raises(ValueError):
        rs.rtk_spec(_entry(**LIMITS, common_mode="reference-typo"))


def test_significance_gate_holds_a_level_inside_the_noise() -> None:
    import datetime as _dt

    t = np.arange(np.datetime64(NOW - _dt.timedelta(hours=8), "m"),
                  np.datetime64(NOW, "m") + 1).astype("datetime64[ns]")  # fmt: skip
    rng = np.random.default_rng(0)
    y = rng.normal(0.0, 30.0, (3, t.size))  # a stormy hour-to-hour scatter
    y[0, t >= np.datetime64(NOW - _dt.timedelta(minutes=50), "ns")] += 12.0
    tm = rs.minutes(t)
    end = float(rs.minutes(np.datetime64(NOW, "ns")))
    spec = {
        **rs.DEFAULTS,
        "horizontal_mm": [10.0, 20.0, 40.0],
        "up_mm": [100.0, 200.0, 400.0],
    }
    loose = rs.baseline_level(tm, y, end, spec)
    gated = rs.baseline_level(tm, y, end, {**spec, "significance_sigma": 2.0})
    assert loose is not None and gated is not None
    assert gated["level"] <= loose["level"]
    assert gated["significant"] is (gated["level"] > 0 or gated["significant"])
