"""map station --track --track-status: the status file (no GMT needed)."""

from __future__ import annotations

import dataclasses
import datetime as dt
import json

import numpy as np
import pytest
from test_track import ONSET, PRE, _series

from gps_plot import map as gmap
from gps_plot import track, track_status


def test_status_window_is_its_own_rate() -> None:
    t, y = _series(move=(365.25 * 2, 0.0))  # 2 mm/d north
    tr = track.station_track("SVIE", -16.8, 64.0, t, y, ONSET, PRE, status_days=7)
    assert tr.status is not None and tr.status_window == 7
    assert tr.status.north == pytest.approx(730, abs=4 * tr.status.sigma_north)
    # without status_days nothing changes
    plain = track.station_track("SVIE", -16.8, 64.0, t, y, ONSET, PRE)
    assert plain.status is None
    assert plain.recent == tr.recent


def test_status_window_widens_for_sparse_days() -> None:
    t, y = _series(move=(100.0, 0.0), days_after=60)
    after = t >= track.yearf(ONSET)
    keep = ~after | (np.arange(t.size) % 4 == 0)  # every 4th day since the onset
    tr = track.station_track("SVIN", -16.8, 64.0, t[keep], y[:, keep], ONSET, PRE,
                             status_days=7)  # fmt: skip
    assert tr.status_window > 7


def test_last_step_is_significant_only_out_of_the_noise() -> None:
    t, y = _series(seed=4)
    tr = track.station_track("SKFC", -17.0, 64.0, t, y, ONSET, PRE, status_days=7)
    step = track_status.last_step(tr)
    assert step is not None and not step["significant"]
    jumped = dataclasses.replace(tr, north=tr.north.copy())
    jumped.north[-1] += 20.0  # 20 mm in a day, noise 1 mm
    step = track_status.last_step(jumped)
    assert step is not None and step["significant"]
    assert step["N"][0] == pytest.approx(20, abs=5)
    assert step["to"] == track_status.day_of(float(tr.t[-1])).isoformat()


def test_status_record_roles_and_units(tmp_path) -> None:
    t, y = _series(move=(200.0, 0.0))
    moving = track.station_track("SVIE", -16.8, 64.0, t, y, ONSET, PRE, status_days=7)
    t2, y2 = _series(seed=7)
    ref = track.station_track("SKFC", -17.0, 64.0, t2, y2, ONSET, PRE, status_days=7)
    rec = track_status.status_record(
        [moving, ref], onset=ONSET, event="ev", window_days=7, frame="itrf14",
        common_mode=True, keep_trend=False,
        reference={"asked": ["SKFC", "KVSK"], "kept": ["SKFC"],
                   "dropped": [{"station": "KVSK", "speed": 9.0}]},
        made=dt.datetime(2026, 11, 7, 12, 0),
    )  # fmt: skip
    assert rec["schema"] == 1 and rec["rate_unit"] == "mm/yr"
    rows = {r["station"]: r for r in rec["stations"]}
    assert rows["SKFC"]["role"] == "reference"
    assert rows["SVIE"]["role"] == "station" and rows["SVIE"]["moved"]
    win = rows["SVIE"]["window"]
    assert win["window_days"] == 7 and len(win["speed"]) == 2
    assert rec["newest_day"] == rows["SVIE"]["last_day"]
    path = track_status.status_path(str(tmp_path / "x.png"))
    assert path.endswith("x.status.json")
    track_status.write_status(path, rec)
    assert json.loads(open(path).read())["event"] == "ev"


def test_cli_track_status_options() -> None:
    p = gmap.build_parser(config=None)
    a = p.parse_args(["station", "SVIE", "--track", "20260923", "--track-status",
                      "--track-status-days", "7"])  # fmt: skip
    assert a.track_status and a.track_status_days == 7
    a = p.parse_args(["station", "SVIE", "--track", "20260923"])
    assert not a.track_status and a.track_status_days is None
