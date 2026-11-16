"""rtk_bench: the RTK warning bench over events and quiet periods (T2)."""

from __future__ import annotations

import datetime as dt
from typing import Any

import numpy as np
import pytest

from gps_plot import rtk_bench as rb

DOC = {
    "test": {
        "pairs": ["MOVE-BASE", "QUIE-BASE", "GONE-BASE"],
        "horizontal_mm": [10, 20, 40],
        "up_mm": [20, 40, 80],
        "before_hours": 2,
        "after_hours": 2,
        "events": [
            {"id": "e1", "swarm_onset": "2024-01-01 12:00", "set": "tune"},
            {"id": "e2", "swarm_onset": "2024-02-01 12:00", "set": "test"},
        ],
        "quiet": {"calibration": [["2024-03-01", "2024-03-03"]]},
    }
}


def _minutes(start: dt.datetime, end: dt.datetime) -> np.ndarray:
    return np.arange(np.datetime64(start, "m"), np.datetime64(end, "m") + 1).astype(
        "datetime64[ns]"
    )


def _reader(step_at: dt.datetime | None, spikes: list[dt.datetime] = ()) -> Any:  # type: ignore[assignment]
    """MOVE-BASE: a 50 mm north step at ``step_at`` (and 30 mm hour-long
    bumps at ``spikes``); QUIE-BASE: noise; GONE-BASE: no data."""

    def read(pair: str, start: dt.datetime, end: dt.datetime) -> Any:
        if pair == "GONE-BASE":
            raise ValueError("no RTK baseline")
        t = _minutes(start, end)
        y = np.random.default_rng(0).normal(0.0, 1.0, (3, t.size))
        if pair == "MOVE-BASE":
            if step_at is not None:
                y[0, t >= np.datetime64(step_at, "ns")] += 50.0
            for s in spikes:
                on = (t >= np.datetime64(s, "ns")) & (
                    t < np.datetime64(s + dt.timedelta(hours=1), "ns")
                )
                y[0, on] += 30.0
        return t, y

    return read


def test_parse_bench_reads_events_quiet_and_edges() -> None:
    b = rb.parse_bench(DOC, "test")
    assert b.pairs == ("MOVE-BASE", "QUIE-BASE", "GONE-BASE")
    assert [e.set for e in b.events] == ["tune", "test"]
    assert b.events[0].onset == dt.datetime(2024, 1, 1, 12, 0)
    assert b.quiet["calibration"] == (
        (dt.datetime(2024, 3, 1), dt.datetime(2024, 3, 3)),
    )
    assert b.before == dt.timedelta(hours=2)
    with pytest.raises(ValueError):
        rb.parse_bench(DOC, "nope")
    bad = {"a": {**DOC["test"], "events": [{"id": "x", "set": "tune"}]}}
    with pytest.raises(ValueError):
        rb.parse_bench(bad, "a")


def test_episodes_count_rises_not_steps() -> None:
    assert rb.episodes([0, 1, 1, 0, None, 1, 3, 3, 1, 0]) == {1: 2, 2: 1, 3: 1}
    assert rb.episodes([None, None]) == {1: 0, 2: 0, 3: 0}


def test_area_level_is_the_highest_verdict() -> None:
    assert rb.area_levels({"A-B": [0, None, 2], "C-D": [1, None, None]}) == [1, None, 2]


def test_red_needs_two_independent_pairs() -> None:
    from gps_plot import rtk_status as rs

    # one red pair alone: orange
    assert rs.combine({"SVIN-SKFC": 3, "SVIE-SKFC": 1}) == 2
    # two reds sharing only their base: a base glitch, orange
    assert rs.combine({"SVIN-SKFC": 3, "SVIE-SKFC": 3}) == 2
    # sharing a rover, or nothing: red
    assert rs.combine({"SVIN-SVIE": 3, "SVIN-SKFC": 3}) == 3
    assert rs.combine({"THOB-ELDC": 3, "SENG-SUDV": 3}) == 3
    assert rs.combine({"A-B": None}) is None


def test_event_delay_per_pair_and_area_and_missing_pairs() -> None:
    b = rb.parse_bench(DOC, "test")
    onset = b.events[0].onset
    r = rb.event_result(b, b.events[0], _reader(onset + dt.timedelta(minutes=10)))
    assert r["missing"] == ["GONE-BASE"]
    move = r["pairs"]["MOVE-BASE"]
    # the 60 min median flips ~30 min after the step, which is 10 min in
    assert dt.timedelta(minutes=30) <= move[3] - onset <= dt.timedelta(minutes=50)
    # the quiet pair never raises the area; one red pair alone is orange (T7)
    assert r["area"] == {1: move[1], 2: move[2]}
    assert r["pairs"]["QUIE-BASE"] == {}
    text = rb.event_report(b, [r])
    assert "| e1 | 2024-01-01 12:00 |" in text and "no data" in text


def test_quiet_alarms_per_month() -> None:
    b = rb.parse_bench(DOC, "test")
    spikes = [dt.datetime(2024, 3, 1, 10), dt.datetime(2024, 3, 2, 10)]
    r = rb.quiet_result(b, b.quiet["calibration"], _reader(None, spikes))
    counts, months = r["pairs"]["MOVE-BASE"]
    assert counts[1] == 2 and counts[3] == 0  # two hour-long 30 mm bumps
    assert months == pytest.approx(2 / 30.44, rel=0.01)  # two days, all with a verdict
    assert r["pairs"]["GONE-BASE"] == ({1: 0, 2: 0, 3: 0}, 0.0)
    assert r["area"][0][1] == 2
    assert "| **area** |" in rb.quiet_report(b, "calibration", r)


def test_report_is_reproducible() -> None:
    a = rb.run_config(DOC, "test", _reader(dt.datetime(2024, 1, 1, 12)), ["tune"], [])
    b = rb.run_config(DOC, "test", _reader(dt.datetime(2024, 1, 1, 12)), ["tune"], [])
    assert a == b and "e2" not in a
    with pytest.raises(ValueError):
        rb.run_config(DOC, "test", _reader(None), ["tune"], ["evaluation"])


def test_persistent_drops_one_step_blips_keeps_runs() -> None:
    assert rb.persistent([0, 2, 0, 1, 1, 3, 3, None, 3], 1) == [
        0,
        0,
        0,
        0,
        1,
        1,
        3,
        None,
        3,
    ]
    assert rb.persistent([1, 1], 0) == [1, 1]


def test_pair_levels_applies_the_sibling_veto_before_persistence() -> None:
    """T5 on the bench as live: a base jump on both pairs stays green."""
    from gps_plot import rtk_status as rs

    spec = rs.rtk_spec({"monitoring": {"rtk": {"horizontal_mm": [10, 20, 40],
                                               "up_mm": [20, 40, 80]}}})  # fmt: skip
    assert spec is not None
    jump = dt.datetime(2024, 1, 1, 12, 0)
    t = _minutes(jump - dt.timedelta(hours=8), jump + dt.timedelta(hours=2))
    times = rb.grid(jump - dt.timedelta(hours=1), jump + dt.timedelta(hours=2))
    steps = {}
    for k, pair in enumerate(["AAAA-BASE", "BBBB-BASE"]):
        y = np.random.default_rng(k).normal(0.0, 1.0, (3, t.size))
        y[0, t >= np.datetime64(jump, "ns")] += 30.0
        steps[pair] = rs.replay(rs.minutes(t), y, times, spec)
    assert max(v or 0 for v in rb.pair_levels(steps, spec)["AAAA-BASE"]) >= 2
    vetoed = rb.pair_levels(steps, {**spec, "common_mode": "siblings"})
    assert all(not v for lv in vetoed.values() for v in lv)


def test_rate_counting_matches_the_bench_functions() -> None:
    """T4d: the vectorised levels, persistence, two-red area and episodes
    agree with persistent / area_levels / episodes, gaps included."""
    rng = np.random.default_rng(3)
    names = ["AAAA-BASE", "BBBB-BASE", "CCCC-OTHR"]
    tops = {}
    for n in names:
        t = rng.gamma(2.0, 1.0, 3000)
        t[rng.random(3000) < 0.1] = np.nan
        tops[n] = t
    ks = [3.0, 5.0, 7.0]
    ref = {}
    for n in names:
        raw = [None if np.isnan(v) else int(sum(k <= v for k in ks)) for v in tops[n]]
        ref[n] = rb.persistent(raw, 1)
        got = rb._levels(tops[n], ks)
        assert [None if np.isnan(v) else int(v) for v in got] == ref[n]
    area_ref = rb.area_levels(ref)
    area = rb._area({n: rb._levels(tops[n], ks) for n in names})
    assert [None if np.isnan(v) else int(v) for v in area] == area_ref
    assert rb._episodes(area) == rb.episodes(area_ref)
    counts, months = rb.area_episodes([tops], ks)
    assert counts == rb.episodes(area_ref) and months > 0


def test_rate_ks_meet_the_targets_and_increase() -> None:
    rng = np.random.default_rng(4)
    tops = [{n: rng.gamma(2.0, 1.0, 20000) for n in ("AAAA-BASE", "BBBB-OTHR")}]
    targets = {"yellow": 2.0, "orange": 0.5, "red": 0.1}
    ks, found = rb.rate_ks(tops, targets)
    assert ks[0] < ks[1] < ks[2]
    counts, months = rb.area_episodes(tops, ks)
    for lvl, name in enumerate(("yellow", "orange", "red"), start=1):
        assert counts[lvl] <= targets[name] * months
        assert found[name]["k"] == pytest.approx(ks[lvl - 1], rel=1e-3)


def test_agree_two_needs_two_independent_pairs_at_every_level() -> None:
    from gps_plot import rtk_status as rs

    assert rs.combine({"SENG-ELDC": 2, "THOB-ELDC": 0}, agree=2) == 0  # alone
    assert rs.combine({"SENG-ELDC": 2, "THOB-SUDV": 1}, agree=2) == 1
    assert rs.combine({"SENG-ELDC": 2, "THOB-ELDC": 1}, agree=2) == 0  # base only
    assert rs.combine({"SENG-ELDC": 3, "THOB-SUDV": 3}, agree=2) == 3
    assert rs.combine({"SENG-ELDC": 3, "THOB-ELDC": 3}, agree=2) == 0  # base only
    rng = np.random.default_rng(5)
    names = ["SENG-ELDC", "THOB-ELDC", "THOB-SUDV", "GEVK-SUDV"]
    tops = {n: rng.gamma(2.0, 1.0, 2000) for n in names}
    ks = [3.0, 5.0, 7.0]
    lv = {n: rb._levels(tops[n], ks) for n in names}
    ref = rb.area_levels({n: [int(v) for v in lv[n]] for n in names}, agree=2)
    assert [int(v) for v in rb._area(lv, agree=2)] == ref
