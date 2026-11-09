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


def test_a_named_state_has_its_own_file() -> None:
    assert changed.state_file() == changed.STATE_FILE
    named = changed.state_file("areas")
    assert named == Path("_runs") / ".fingerprints.areas.json"
    assert named != changed.state_file("standard")


def test_active_of_a_changed_run_stays_inside_the_box(monkeypatch) -> None:
    """--changed turns ``active`` into a station list, so the box ``ts active``
    applies has to be applied here: ZIMM has no plate and would fail the run."""
    import configparser
    from types import SimpleNamespace

    import geo_dataread.pos as pos

    from gps_plot.cli import run

    cfg = configparser.ConfigParser()
    cfg.read_string("[VONC]\nx = 1\n[ZIMM]\nx = 1\n[KARV]\nstation_status = inactive\n")
    seen = {}

    def inside_only(prod, directory, region):
        seen["prod"], seen["region"] = prod, region
        return [("VONC", -18.5, 64.0), ("KARV", -15.0, 65.0)]

    monkeypatch.setattr(pos, "pos_stations", inside_only)
    monkeypatch.setattr("gps_plot.cli.common.config_dir", lambda *a: None)
    assert run._active(SimpleNamespace(config=cfg)) == ["VONC"]
    assert seen["prod"] == "imo.24hr_itrf14"


def test_check_files_names_a_map_layer_not_on_this_host(tmp_path, monkeypatch) -> None:
    from argparse import Namespace
    from types import SimpleNamespace

    from gps_plot import gis_layers
    from gps_plot import map as gmap
    from gps_plot.cli import run

    parser = SimpleNamespace(
        parse_known_args=lambda probe: (Namespace(layers=["volcanoes"]), [])
    )
    monkeypatch.setattr(run, "_tool", lambda *a, **k: (None, parser))
    monkeypatch.setattr(gmap, "layer_catalogue", lambda *a: gis_layers.BUILTIN)
    monkeypatch.setattr(gmap, "gis_dir", lambda c: str(tmp_path))
    jobs = [Job("mapped", None, "d", (), (), False, tool="map", kind="station")]
    jobs.append(_job("ts", "VONC"))

    missing = run._check_files(jobs, None)
    assert len(missing) == 1 and missing[0].startswith("mapped: ")

    layer = gis_layers.BUILTIN.layers["volcanoes"]
    made = Path(gis_layers.layer_file(layer, str(tmp_path)))
    made.parent.mkdir(parents=True)
    made.write_text("{}")
    assert run._check_files(jobs, None) == []


def test_plots_root_prefers_the_flag_then_the_host(monkeypatch) -> None:
    from gps_plot.cli import common

    cfg = {"plots_dir": "/from/yaml"}
    host: list[str | None] = ["/from/host"]
    monkeypatch.setattr(common, "config_dir", lambda *a: host[0])
    assert common.plots_root("/from/flag", None, cfg) == Path("/from/flag")
    assert common.plots_root(None, None, cfg) == Path("/from/host")
    host[0] = None
    assert common.plots_root(None, None, cfg) == Path("/from/yaml")

