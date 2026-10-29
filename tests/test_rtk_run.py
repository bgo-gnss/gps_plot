"""``gpsplot rtk-run``: which plots, their files, stale ones, the lock."""

from __future__ import annotations

import fcntl
from pathlib import Path

import pytest

from gps_plot import rtk_run

DIRS = ["SEY1-SEYD", "SEY2-SEYD", "SKFC-SVIN", "SVIE-SKFC"]
AREAS = {
    "seydisfjordur": {
        "root": "skridurogflod",
        "baselines": ["SEY1-SEYD"],
        "stacks": {"seyd": ["SEY1-SEYD", "SEY2-SEYD", "SEY3-SEYD"]},
    },
    "svinafellsheidi": {
        "root": "skridurogflod",
        "baselines": ["SVIN-SKFC", "SVIE-SKFC", "SVIN-SVIE"],
        "stacks": {},
    },
    "hekla": {"baselines": ["HEKR-ISAK"], "stacks": {}},
}


def test_plan_takes_the_rtk_pairs_of_baselines_and_stacks_once():
    jobs = [j for j in rtk_run.plan({}, AREAS, DIRS) if j.run == "14d"]
    assert [(j.area, j.pair) for j in jobs] == [
        ("seydisfjordur", "SEY1-SEYD"),
        ("seydisfjordur", "SEY2-SEYD"),  # SEY3-SEYD is no RTK baseline
        ("svinafellsheidi", "SVIN-SKFC"),  # processed as SKFC-SVIN
        ("svinafellsheidi", "SVIE-SKFC"),
    ]
    j = jobs[2]
    assert j.out_dir == "locations/skridurogflod/svinafellsheidi/graphs/rtk"
    assert j.stem == "SVIN_SKFC-rtk-14d"
    assert j.argv == ("--longterm", "14d", "--rtk-overlap", "2")


def test_plan_runs_overrides_and_area_filter():
    cfg = {
        "rtk": {
            "dir": "x/{cdn}",
            "runs": [{"name": "14d", "args": "--longterm 14d"}, {"name": "2d"}],
            "overrides": {"svinafellsheidi": {"args": ["--fixed-only"]}},
        }
    }
    jobs = rtk_run.plan(cfg, AREAS, DIRS, area=["svinafellsheidi"])
    assert [j.stem for j in jobs] == [
        "SVIN_SKFC-rtk-14d", "SVIE_SKFC-rtk-14d", "SVIN_SKFC-rtk-2d", "SVIE_SKFC-rtk-2d",
    ]  # fmt: skip
    assert jobs[0].argv == ("--longterm", "14d", "--fixed-only")
    assert jobs[2].argv == ("--fixed-only",)
    assert jobs[0].out_dir == "x/svinafellsheidi"


@pytest.mark.parametrize(
    "cfg, area, msg",
    [
        ({"rtk": {"runs": [{"args": "--longterm 14d"}]}}, None, "distinct names"),
        ({"rtk": {"runs": [{"name": "a"}, {"name": "a"}]}}, None, "distinct names"),
        ({"rtk": {"runs": [{"name": "a", "args": "-d /tmp"}]}}, None, "set by"),
        ({}, ["nowhere"], "unknown area"),
    ],
)
def test_plan_refuses(cfg, area, msg):
    with pytest.raises(ValueError, match=msg):
        rtk_run.plan(cfg, AREAS, DIRS, area=area)


def test_stale_only_touches_rtk_plots_no_job_writes(tmp_path):
    jobs = rtk_run.plan({}, AREAS, DIRS, area=["seydisfjordur"])
    assert sorted({j.run for j in jobs}) == ["14d", "3d", "7d"]
    d = tmp_path / jobs[0].out_dir
    d.mkdir(parents=True)
    for name in ["SEY1_SEYD-rtk-14d.png", "SEY4_SEYD-rtk-14d.png",
                 "SEY1_SEYD-rtk-old.png", "SEY1_SEYD-baseline-plate-90d.png"]:  # fmt: skip
        (d / name).touch()
    assert [f.name for f in rtk_run.stale(jobs, tmp_path)] == [
        "SEY1_SEYD-rtk-old.png", "SEY4_SEYD-rtk-14d.png",
    ]  # fmt: skip


def test_plot_renames_into_place_or_reports_nothing(monkeypatch, tmp_path):
    from gps_plot import rtk

    job = rtk_run.RtkJob("a", "SEY1-SEYD", "14d", "out", ("--longterm", "14d"))
    seen = {}

    def fake_run(args, parser, config):
        seen["args"] = args
        Path(args.figDir, f"{args.name}.png").write_bytes(b"png")
        return 0

    monkeypatch.setattr(rtk, "run", fake_run)
    assert rtk_run.plot(job, tmp_path, None) is True
    assert (tmp_path / "out" / "SEY1_SEYD-rtk-14d.png").read_bytes() == b"png"
    assert seen["args"].longterm == "14d" and seen["args"].save == "png"
    assert [p.name for p in (tmp_path / "out").iterdir()] == ["SEY1_SEYD-rtk-14d.png"]

    monkeypatch.setattr(rtk, "run", lambda args, parser, config: 1)
    assert rtk_run.plot(job, tmp_path, None) is False
    assert (tmp_path / "out" / "SEY1_SEYD-rtk-14d.png").read_bytes() == b"png"


def test_a_run_still_going_makes_the_next_skip(monkeypatch, tmp_path, capsys):
    import geo_dataread.rtk_read as rr

    from gps_plot import plot_gps_timeseries, products

    class Config:
        config_path = str(tmp_path)

        def getPostProcessDir(self, option):
            return str(tmp_path)

    (tmp_path / "plots.yaml").write_text(f"plots_dir: {tmp_path / 'plots'}\n")
    monkeypatch.setattr(plot_gps_timeseries, "_load_config", lambda: Config())
    monkeypatch.setattr(products, "area_table", lambda doc: AREAS)
    monkeypatch.setattr(
        products, "load_yaml", lambda p: {"plots_dir": str(tmp_path / "plots")}
    )
    monkeypatch.setattr(rr, "list_baselines", lambda root=None: DIRS)
    monkeypatch.setattr(rtk_run, "plot", lambda *a: pytest.fail("plotted while locked"))
    (tmp_path / "plots").mkdir()
    with open(tmp_path / "plots" / rtk_run.LOCK_NAME, "w") as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        assert rtk_run.run("rtk-run", []) == 0
    assert "still going" in capsys.readouterr().out
