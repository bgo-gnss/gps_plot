"""plots.yaml expansion: areas, events, runs, overrides, publish gate."""

from __future__ import annotations

import datetime as dt

import pytest

from gps_plot import products

AREAS_DOC = {
    "category_labels": {"volcanic": "x"},
    "volcanic_areas": {
        "reykjanesskagi": {"cdn": "reykjanes", "stations": ["SENG", "ELDC  # x"]},
        "hengill": {"cdn": "hengill", "stations": ["OLKE"]},
        "nocdn": {"stations": ["AAAA"]},
    },
    "regional_areas": {"reykjanes": {"cdn": "reykjanes", "stations": ["SENG", "KRIV"]}},
}
EVENTS = [
    {"region": "reykjanes", "kind": "eruption", "onset": dt.date(2021, 3, 19)},
    {"region": "reykjanes", "kind": "dike", "onset": "2021-02-24"},
    {"region": "reykjanes", "kind": "unknown", "onset": "2021-01-01"},
    {"region": "hekla", "kind": "eruption", "onset": "2000-02-26"},
]
CFG = {
    "defaults": {"frame": "itrf14", "save": "png", "legacy_names": True},
    "products": {
        "standard": {
            "stations": "active",
            "dir": "timeseries/graphs/standard_24h",
            "runs": [{"ref": "plate", "special": ["90d", "year"], "publish": True}],
        },
        "areas": {
            "stations": "area",
            "dir": "locations/volcanos/{cdn}/graphs/24h",
            "events": "catalog",
            "runs": [
                {"ref": "plate", "special": ["90d"], "ylim": 20, "publish": True},
                {"session": "8h", "special": ["90d"], "fixedstart": False},
            ],
            "overrides": {"reykjanes": {"ylim": 15, "fixedstart": ["2023-11-12"]}},
        },
    },
}


def test_area_table_merges_members_by_cdn_slug() -> None:
    table = products.area_table(AREAS_DOC)
    assert sorted(table) == ["hengill", "reykjanes"]
    assert table["reykjanes"]["stations"] == ["SENG", "ELDC", "KRIV"]


def test_event_specs_by_region_and_kind() -> None:
    assert products.event_specs(EVENTS, ["reykjanes"]) == ["20210224,b", "20210319,r"]


def test_expand_jobs_dirs_publish_and_overrides() -> None:
    jobs = products.expand(CFG, areas=products.area_table(AREAS_DOC), events=EVENTS)
    std = [j for j in jobs if j.product == "standard"]
    assert len(std) == 1 and std[0].stations == ("active",) and std[0].publish
    assert "90d,year" in std[0].argv
    rey = [j for j in jobs if j.area == "reykjanes"]
    # plate 90d + plate fixedstart (override) + 8h (opted out of fixedstart)
    assert len(rey) == 3
    assert rey[0].out_dir == "locations/volcanos/reykjanes/graphs/24h"
    assert rey[0].argv[rey[0].argv.index("--ylim") + 1] == "15"
    assert "-s" in rey[1].argv and "20231112" in rey[1].argv
    assert rey[2].publish is False and "-s" not in rey[2].argv
    assert "20210319,r" in rey[0].argv
    hen = [j for j in jobs if j.area == "hengill"]
    assert "--events" not in hen[0].argv
    assert hen[0].argv[hen[0].argv.index("--ylim") + 1] == "20"


def test_expand_filters_product_and_area() -> None:
    areas = products.area_table(AREAS_DOC)
    only = products.expand(CFG, areas=areas, area="hengill")
    assert {j.area for j in only} == {"hengill"}
    assert all(
        j.product == "standard"
        for j in products.expand(CFG, areas=areas, product="standard")
    )


def test_run_argv_refuses_unknown_keys_and_empty_runs() -> None:
    with pytest.raises(ValueError, match="unknown"):
        products.run_argv({"special": ["90d"], "colour": "red"}, {})
    with pytest.raises(ValueError, match="special"):
        products.run_argv({"ref": "plate"}, {})


def test_run_refuses_an_unknown_prepare_step(tmp_path, monkeypatch, capsys) -> None:
    import yaml

    from gps_plot.cli import run as run_cli

    plots = tmp_path / "plots.yaml"
    plots.write_text(yaml.safe_dump({**CFG, "prepare": ["network_days", "bogus"]}))
    for name in ("station_areas.yaml", "events.yaml"):
        (tmp_path / name).write_text(
            yaml.safe_dump(AREAS_DOC if "areas" in name else {"events": []})
        )
    monkeypatch.setattr(
        run_cli.driver,
        "_load_config",
        lambda: type("C", (), {"config_path": str(tmp_path)})(),
    )
    assert run_cli.run("gpsplot run", [str(plots), "--dry-run"]) == 2
    assert "bogus" in capsys.readouterr().out


def test_dry_run_lists_the_prepare_steps_first(tmp_path, monkeypatch, capsys) -> None:
    import yaml

    from gps_plot.cli import run as run_cli

    plots = tmp_path / "plots.yaml"
    plots.write_text(yaml.safe_dump({**CFG, "prepare": ["network_days"]}))
    for name in ("station_areas.yaml", "events.yaml"):
        (tmp_path / name).write_text(
            yaml.safe_dump(AREAS_DOC if "areas" in name else {"events": []})
        )
    monkeypatch.setattr(
        run_cli.driver,
        "_load_config",
        lambda: type("C", (), {"config_path": str(tmp_path)})(),
    )
    called = []
    monkeypatch.setitem(
        run_cli.PREPARE_STEPS, "network_days", lambda w: called.append(w) or 0
    )
    assert run_cli.run("gpsplot run", [str(plots), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert (
        out.startswith("prepare: network_days") and not called
    )  # dry run: listed, not run
    assert run_cli.run("gpsplot run", [str(plots), "--dry-run", "--no-prepare"]) == 0
    assert "prepare:" not in capsys.readouterr().out


def test_args_are_appended_general_to_specific() -> None:
    cfg = {
        "defaults": {"args": ["--traceback"]},
        "products": {
            "areas": {
                "stations": "area",
                "dir": "a/{cdn}",
                "args": "--show-icing --outlier-sigma 4",
                "runs": [
                    {
                        "ref": "detrend",
                        "special": ["90d"],
                        "args": ["--outlier-sigma 3", "--show-outliers"],
                    }
                ],
                "overrides": {"reykjanes": {"args": ["--outlier-sigma", "2.5"]}},
            }
        },
    }
    jobs = products.expand(cfg, areas=products.area_table(AREAS_DOC))
    rey = next(j for j in jobs if j.area == "reykjanes").argv
    tail = rey[rey.index("--traceback") : rey.index("--special")]
    assert tail == (
        "--traceback",
        "--show-icing",
        "--outlier-sigma",
        "4",
        "--outlier-sigma",
        "3",
        "--show-outliers",
        "--outlier-sigma",
        "2.5",
    )
    hen = next(j for j in jobs if j.area == "hengill").argv
    assert hen[hen.index("--show-outliers") - 1] == "3"  # no override there


@pytest.mark.parametrize("bad", [["-d", "/tmp"], "--figDir=/tmp", ["--special 90d"]])
def test_args_refuse_runner_owned_options(bad: object) -> None:
    with pytest.raises(ValueError, match="not allowed"):
        products.run_argv({"special": ["90d"], "args": bad}, {})


def test_args_must_be_a_string_or_list() -> None:
    with pytest.raises(ValueError, match="string or a list"):
        products.extra_args({"outlier_sigma": 3.5})


def _run_cli(tmp_path, monkeypatch, cfg):
    import yaml

    from gps_plot.cli import run as run_cli

    plots = tmp_path / "plots.yaml"
    plots.write_text(yaml.safe_dump(cfg))
    for name in ("station_areas.yaml", "events.yaml"):
        (tmp_path / name).write_text(
            yaml.safe_dump(AREAS_DOC if "areas" in name else {"events": []})
        )
    monkeypatch.setattr(
        run_cli.driver,
        "_load_config",
        lambda: type("C", (), {"config_path": str(tmp_path)})(),
    )
    return run_cli.run("gpsplot run", [str(plots), "--dry-run"])


def test_run_refuses_an_unknown_ts_option_before_anything(
    tmp_path, monkeypatch, capsys
) -> None:
    cfg = {**CFG, "defaults": {**CFG["defaults"], "args": ["--no-such-flag"]}}
    assert _run_cli(tmp_path, monkeypatch, cfg) == 2
    out = capsys.readouterr().out
    assert "nothing run" in out and "--no-such-flag" in out


def test_run_reports_a_refused_args_key(tmp_path, monkeypatch, capsys) -> None:
    cfg = {**CFG, "defaults": {**CFG["defaults"], "args": ["-j", "4"]}}
    assert _run_cli(tmp_path, monkeypatch, cfg) == 2
    assert "gpsplot run -j" in capsys.readouterr().out


def test_select_stations_narrows_each_job_and_reports_the_unused() -> None:
    jobs = products.expand(CFG, areas=products.area_table(AREAS_DOC))
    kept, unused = products.select_stations(
        jobs, ["kriv", "OLKE", "ZZZZ"], {"active": ["OLKE"], "all": None}
    )
    by = {(j.product, j.area): j.stations for j in kept}
    assert by[("standard", None)] == ("OLKE",)  # active matched via keywords
    assert by[("areas", "reykjanes")] == ("KRIV",)
    assert by[("areas", "hengill")] == ("OLKE",)
    assert unused == ["ZZZZ"]
    # every run of an area keeps its jobs (fixedstart included)
    assert sum(j.area == "reykjanes" for j in kept) == sum(
        j.area == "reykjanes" for j in jobs
    )


def test_select_stations_drops_jobs_without_a_wanted_station() -> None:
    jobs = products.expand(CFG, areas=products.area_table(AREAS_DOC))
    kept, unused = products.select_stations(jobs, ["SENG"], {"active": []})
    assert {j.area for j in kept} == {"reykjanes"} and not unused


def test_run_takes_stations_and_a_plots_file_in_any_order(
    tmp_path, monkeypatch, capsys
) -> None:
    import yaml

    from gps_plot.cli import run as run_cli

    plots = tmp_path / "plots.yaml"
    plots.write_text(yaml.safe_dump(CFG))
    for name in ("station_areas.yaml", "events.yaml"):
        (tmp_path / name).write_text(
            yaml.safe_dump(AREAS_DOC if "areas" in name else {"events": []})
        )
    monkeypatch.setattr(
        run_cli.driver,
        "_load_config",
        lambda: type("C", (), {"config_path": str(tmp_path)})(),
    )
    monkeypatch.setattr(run_cli, "_active", lambda config: ["SENG"])
    argv = ["seng,kriv", str(plots), "--area", "reykjanes", "--dry-run"]
    assert run_cli.run("gpsplot run", argv) == 0
    out = capsys.readouterr().out
    assert "2 stations: SENG KRIV" in out and "ELDC" not in out
    assert run_cli.run("gpsplot run", [str(plots), "QQQQ", "--dry-run"]) == 1
    out = capsys.readouterr().out
    assert "left out: QQQQ" in out and "no jobs" in out
