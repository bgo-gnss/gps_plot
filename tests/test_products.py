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
