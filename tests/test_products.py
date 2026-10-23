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


def test_expand_takes_several_products_and_refuses_unknown_ones() -> None:
    areas = products.area_table(AREAS_DOC)
    both = products.expand(CFG, areas=areas, product=["standard", "areas"])
    assert {j.product for j in both} == {"standard", "areas"}
    with pytest.raises(ValueError, match="no product stacks"):
        products.expand(CFG, areas=areas, product=["standard", "stacks"])


def test_product_option_splits_a_comma_list() -> None:
    from gps_plot.cli import run as run_cli

    assert run_cli._names(None) is None
    assert run_cli._names("baselines, stacks,") == ["baselines", "stacks"]


def test_expand_takes_several_areas_and_refuses_unknown_ones() -> None:
    areas = products.area_table(AREAS_DOC)
    both = products.expand(CFG, areas=areas, area=["hengill", "reykjanes"])
    assert {j.area for j in both} == {"hengill", "reykjanes"}
    with pytest.raises(ValueError, match="no area askja"):
        products.expand(CFG, areas=areas, area=["hengill", "askja"])


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


BL_AREAS_DOC = {
    "category_roots": {"infrastructure": "voktun/landsvirkjun"},
    "volcanic_areas": {
        "reykjanesskagi": {
            "cdn": "reykjanes",
            "stations": ["SENG", "ELDC"],
            "baselines": {
                "eldc": ["SENG-ELDC", "skSH-ELDC  # x", "AUSV-ELDC"],
                "ausv": ["ELDC-AUSV", "SENG-AUSV"],
            },
        },
    },
    "regional_areas": {"reykjanes": {"cdn": "reykjanes", "stations": ["KRIV"]}},
    "infrastructure_areas": {
        "vatnsfell": {
            "cdn": "vatnsfell",
            "stations": ["VFLN", "VFLS"],
            "baselines": {
                "vatnsfell": ["VFLN-VFLS"],
                "budh": ["VFLS-BUDH", "VFLN-BUDH"],
            },
        },
    },
}
BL_CFG = {
    "defaults": {"frame": "itrf14", "save": "png", "legacy_names": True},
    "products": {
        "baselines": {
            "stations": "area_baselines",
            "dir": "locations/{root}/{cdn}/graphs/baselines",
            "events": "catalog",
            "runs": [{"ref": "plate", "special": ["90d", "year"], "publish": True}],
            "overrides": {"reykjanes": {"fixedstart": ["2023-11-12"]}},
        },
        "stacks": {
            "stations": "area_stacks",
            "dir": "locations/{root}/{cdn}/graphs/baselines",
            "events": "catalog",
            "runs": [{"ref": "plate", "special": ["90d", "year"], "publish": True}],
        },
    },
}


def test_area_table_reads_baseline_groups_and_the_category() -> None:
    table = products.area_table(BL_AREAS_DOC)
    rey = table["reykjanes"]
    assert rey["category"] == "volcanic"  # the first area of the slug
    assert rey["stations"] == ["SENG", "ELDC", "KRIV"]
    assert rey["baselines"]["eldc"] == ["SENG-ELDC", "SKSH-ELDC", "AUSV-ELDC"]
    assert rey["root"] == "volcanos"  # no category_roots entry
    assert table["vatnsfell"]["category"] == "infrastructure"
    assert table["vatnsfell"]["root"] == "voktun/landsvirkjun"


def test_area_pairs_keep_one_direction_of_a_reversed_repeat() -> None:
    groups = products.area_table(BL_AREAS_DOC)["reykjanes"]["baselines"]
    # ELDC-AUSV (the ausv fan) is AUSV-ELDC reversed: plotted once
    assert products.area_pairs(groups) == [
        "SENG-ELDC",
        "SKSH-ELDC",
        "AUSV-ELDC",
        "SENG-AUSV",
    ]


@pytest.mark.parametrize(
    ("value", "match"),
    [
        ({"g": ["SENG-ELD"]}, "not AAAA-BBBB"),
        ({"g": ["SENG-SENG"]}, "itself"),
        ({"g": ["SENG-ELDC", "ELDC-SENG"]}, "twice"),
        ({"g": "SENG-ELDC"}, "a list"),
        ("SENG-ELDC", "mapping of groups or a list"),
    ],
)
def test_area_baselines_refuse_malformed_lists(value: object, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        products.area_baselines(value)


def test_area_baselines_plain_list_is_one_unnamed_group() -> None:
    assert products.area_baselines(["vfln-vfls"]) == {"": ["VFLN-VFLS"]}


def test_area_table_refuses_a_group_that_differs_between_same_slug_areas() -> None:
    doc = {
        "volcanic_areas": {
            "a": {"cdn": "x", "stations": [], "baselines": {"g": ["AAAA-BBBB"]}}
        },
        "regional_areas": {
            "b": {"cdn": "x", "stations": [], "baselines": {"g": ["CCCC-BBBB"]}}
        },
    }
    with pytest.raises(ValueError, match="differs"):
        products.area_table(doc)


def test_baseline_products_expand_to_ts_and_stack_jobs() -> None:
    jobs = products.expand(BL_CFG, areas=products.area_table(BL_AREAS_DOC))
    single = [j for j in jobs if j.product == "baselines"]
    assert {j.tool for j in single} == {"ts"}
    vfl = next(j for j in single if j.area == "vatnsfell")
    assert vfl.stations == ("VFLN-VFLS", "VFLS-BUDH", "VFLN-BUDH")
    assert vfl.out_dir == "locations/voktun/landsvirkjun/vatnsfell/graphs/baselines"
    assert "--skip-missing" in vfl.argv and "--special" in vfl.argv
    rey = [j for j in single if j.area == "reykjanes"]
    assert rey[0].out_dir == "locations/volcanos/reykjanes/graphs/baselines"
    assert len(rey) == 2  # the 90d,year job + the fixedstart override

    stacks = [j for j in jobs if j.product == "stacks"]
    assert {j.tool for j in stacks} == {"stack"}
    # one figure per group of 2+ pairs and period; vatnsfell's 1-pair group none
    assert sorted({(j.area, j.group) for j in stacks}) == [
        ("reykjanes", "ausv"),
        ("reykjanes", "eldc"),
        ("vatnsfell", "budh"),
    ]
    budh = [j for j in stacks if j.group == "budh"]
    assert [j.argv[j.argv.index("--special") + 1] for j in budh] == ["90d", "year"]
    argv = budh[0].argv
    assert argv[-2:] == ("--label", "budh")
    assert "--skip-missing" not in argv and "--legacy-names" not in argv
    assert not any("--events" in j.argv for j in stacks)  # stack draws none
    assert budh[0].name == "stacks/vatnsfell/budh"


def test_stack_runs_refuse_the_ts_operational_preset() -> None:
    with pytest.raises(ValueError, match="operational"):
        products.run_argv({"special": ["90d"], "operational": True}, {}, tool="stack")


def test_select_stations_matches_baselines_by_an_end_and_keeps_stacks_whole() -> None:
    jobs = products.expand(BL_CFG, areas=products.area_table(BL_AREAS_DOC))
    kept, unused = products.select_stations(jobs, ["BUDH"])
    assert not unused
    single = next(j for j in kept if j.product == "baselines")
    assert single.stations == ("VFLS-BUDH", "VFLN-BUDH")
    stack = next(j for j in kept if j.product == "stacks")
    assert stack.group == "budh" and len(stack.stations) == 2
    kept, _ = products.select_stations(jobs, ["vfln-vfls"])
    assert [j.stations for j in kept] == [("VFLN-VFLS",)]  # in no stack


def test_dry_run_checks_stack_options_with_the_stack_parser(
    tmp_path, monkeypatch, capsys
) -> None:
    import yaml

    from gps_plot.cli import run as run_cli

    cfg = {**BL_CFG, "products": {"stacks": BL_CFG["products"]["stacks"]}}
    plots = tmp_path / "plots.yaml"
    plots.write_text(yaml.safe_dump(cfg))
    (tmp_path / "station_areas.yaml").write_text(yaml.safe_dump(BL_AREAS_DOC))
    (tmp_path / "events.yaml").write_text(yaml.safe_dump({"events": []}))
    monkeypatch.setattr(
        run_cli.driver,
        "_load_config",
        lambda: type("C", (), {"config_path": str(tmp_path)})(),
    )
    monkeypatch.setattr(run_cli, "_tool", _fake_tool(run_cli))
    assert run_cli.run("gpsplot run", [str(plots), "--dry-run"]) == 0
    assert "    stack --save png" in capsys.readouterr().out
    cfg["products"]["stacks"]["args"] = ["--skip-missing"]  # a ts-only flag
    plots.write_text(yaml.safe_dump(cfg))
    assert run_cli.run("gpsplot run", [str(plots), "--dry-run"]) == 2
    assert "--skip-missing" in capsys.readouterr().out


def _fake_tool(run_cli):
    """The real stack parser without the station-config lookup."""
    import argparse

    from gps_plot import stack

    def tool(name, config, **kw):
        assert name == "stack"
        parser = stack.build_parser(
            argparse.ArgumentParser(prog="gpsplot stack", **kw),
            config=type("C", (), {"getStationInfo": lambda self: []})(),
        )
        return stack.run, parser

    return tool


def test_stack_file_stem_uses_the_label() -> None:
    from gps_plot import stack

    assert stack.file_stem(["SEY1-SEYD", "SEY2-SEYD"]) == "stack_SEY1_SEYD_2"
    assert stack.file_stem(["SEY1-SEYD"], "seyd") == "stack_seyd"


def test_stack_groups_take_their_own_overrides_after_the_area() -> None:
    cfg = {
        **BL_CFG,
        "products": {
            "stacks": {
                **BL_CFG["products"]["stacks"],
                "args": ["--offset 20"],
                "overrides": {
                    "reykjanes": {"args": ["--offset 30"]},
                    "reykjanes/eldc": {"args": ["--offset 10"], "special": ["90d"]},
                },
            }
        },
    }
    jobs = products.expand(cfg, areas=products.area_table(BL_AREAS_DOC))

    def offsets(group: str) -> list[list[str]]:
        return [
            [j.argv[i + 1] for i, a in enumerate(j.argv) if a == "--offset"]
            for j in jobs
            if j.group == group
        ]

    assert offsets("eldc") == [["20", "30", "10"]]  # 90d only, the group's last
    assert offsets("ausv") == [["20", "30"]] * 2
    assert offsets("budh") == [["20"]] * 2


@pytest.mark.parametrize(
    ("series", "label", "offset", "pos_id", "stem"),
    [
        # baselines only: the frame cancels -> no frame tag
        (
            ["SEY1-SEYD", "SEY2-SEYD"],
            "seyd",
            20,
            "imo.24hr_itrf14",
            "stack_seyd-plate-90d",
        ),
        # ... but an 8 h series is still another product
        (["SEY1-SEYD"], "seyd", 0, "imo.08hr_itrf20", "stack_seyd-plate-8h-90d"),
        # a station in the stack keeps the frame
        (
            ["SEY1-SEYD", "SEYD"],
            None,
            0,
            "imo.24hr_itrf14",
            "stack_SEY1_SEYD_2-plate-itrf2014-90d",
        ),
        # unlabelled stacks still name their offset
        (
            ["HOFN", "RHOF"],
            None,
            20,
            "imo.24hr_itrf14",
            "stack_HOFN_2-plate-offset20-itrf2014-90d",
        ),
    ],
)
def test_stack_output_stem(series, label, offset, pos_id, stem) -> None:
    import datetime as _dt

    from gps_plot import stack

    day = _dt.datetime(2026, 1, 1)
    got = stack.output_stem(
        series,
        label=label,
        ref="plate",
        detrend_terms=None,
        view="raw",
        offset=offset,
        tType="POS",
        pos_id=pos_id,
        special="90d",
        first=day,
        last=day,
    )
    assert got == stem
