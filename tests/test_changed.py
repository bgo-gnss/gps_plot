"""gpsplot run --changed: fingerprints and which jobs to redraw."""

from __future__ import annotations

from pathlib import Path

from gps_plot import changed
from gps_plot.products import Job


def _job(tool: str, *stations: str) -> Job:
    return Job("p", None, "d", stations, (), False, tool=tool)


def test_station_prints_follow_each_station_s_own_files(tmp_path: Path) -> None:
    for name in ("SVIN.imo.24hr_itrf14.pos", "SVIN.autcln.24hr.csv",
                 "SKFC.imo.24hr_itrf14.pos", "network_cm.itrf14.csv",
                 "TSSUM.warning"):  # fmt: skip
        (tmp_path / name).write_text(name)
    cache: dict = {}
    first = changed.station_prints(tmp_path, cache)
    assert sorted(first) == ["SKFC", "SVIN"]  # no network lists, no logs
    (tmp_path / "network_cm.itrf14.csv").write_text("new day")
    assert changed.station_prints(tmp_path, cache) == first
    (tmp_path / "SVIN.autcln.24hr.csv").write_text("icing")
    again = changed.station_prints(tmp_path, cache)
    assert changed.changed_stations(first, again) == {"SVIN"}
    # a station gone or new counts as changed
    assert changed.changed_stations({"AAAA": "x"}, {"BBBB": "y"}) == {"AAAA", "BBBB"}


def test_setup_print_follows_config_and_code(tmp_path: Path) -> None:
    cfg, code = tmp_path / "cfg", tmp_path / "code"
    cfg.mkdir()
    code.mkdir()
    (cfg / "plots.yaml").write_text("a")
    (cfg / "database.cfg").write_text("secret")  # never part of it
    (code / "m.py").write_text("x = 1")
    cache: dict = {}
    one = changed.setup_print(cfg, code, cache)
    (cfg / "database.cfg").write_text("other")
    assert changed.setup_print(cfg, code, cache) == one
    (code / "m.py").write_text("x = 2")
    two = changed.setup_print(cfg, code, cache)
    assert two != one
    (cfg / "plots.yaml").write_text("b")
    assert changed.setup_print(cfg, code, cache) != two


def test_narrow_keeps_what_touches_a_changed_station() -> None:
    jobs = [
        _job("ts", "active"),
        _job("ts", "SVIN-SKFC", "SVIE-SKFC", "HOFN-SKFC"),
        _job("stack", "SEY?-SEYD"),
        _job("polar", "SVIN-SKFC", "SVIE-SKFC"),
        _job("map"),
        _job("ts", "OLKE"),
        _job("map", "SEY1", "SEYD"),
    ]
    out = changed.narrow(
        jobs, {"SVIE", "SEY3"}, {"active": ["SVIE", "HOFN"], "all": None}
    )
    assert out[0].stations == ("SVIE",)  # through active
    assert out[1].stations == ("SVIE-SKFC",)  # a pair by either end
    assert out[2] is jobs[2]  # a stack pattern, whole
    assert out[3] is jobs[3]  # polar --together: one figure, whole
    assert out[4] is jobs[4]  # a map of a region runs on any change
    assert len(out) == 5  # OLKE did not change, nor the SEY map's stations
    assert changed.narrow(jobs, set(), {"active": None}) == []


def test_state_round_trip_and_a_bad_file(tmp_path: Path) -> None:
    path = tmp_path / changed.STATE_FILE
    assert changed.load_state(path) == {}
    changed.save_state(path, {"setup": "s", "stations": {"SVIN": "x"}})
    assert changed.load_state(path)["stations"] == {"SVIN": "x"}
    path.write_text("not json")
    assert changed.load_state(path) == {}
