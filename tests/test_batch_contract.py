"""gpsplot ts batch contract: outcomes, --strict, summary, exit status, manifest."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

import gps_plot.plot_gps_timeseries as driver
import gps_plot.timesmatplt as tplt

KW = {"ref": "plate", "special": "90d", "save": "png"}


def _fake_plot(tmp_path: Path, warn: str | None = None) -> Any:
    def plot(sta: str, **_: Any) -> Any:
        if warn:
            warnings.warn(warn, UserWarning, stacklevel=2)
        fig = plt.figure()
        tplt.saveFig(str(tmp_path / f"{sta}-plate-90d"), "png", fig)
        return fig

    return plot


def test_ok_outcome_lists_the_written_files(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(tplt, "plotTime", _fake_plot(tmp_path))
    out = driver.tryTimes("VONC", **KW)
    assert out.status == "ok"
    assert out.files == [str(tmp_path / "VONC-plate-90d.png")]
    assert Path(out.files[0]).exists()


def test_no_data_is_a_skip_not_a_failure(monkeypatch) -> None:
    def plot(sta: str, **_: Any) -> Any:
        raise tplt.NoDataError(f"no data for station {sta}")

    monkeypatch.setattr(tplt, "plotTime", plot)
    out = driver.tryTimes("ASVE", **KW)
    assert (out.status, out.reason) == ("skipped", "no data for station ASVE")


def test_an_exception_is_one_failed_outcome(monkeypatch, capsys) -> None:
    def plot(sta: str, **_: Any) -> Any:
        raise ValueError("no plate assigned")

    monkeypatch.setattr(tplt, "plotTime", plot)
    out = driver.tryTimes("LAMB", **KW)
    assert out.status == "failed"
    assert out.reason == "ValueError: no plate assigned"
    err = capsys.readouterr().err
    assert "FAILED LAMB plate/90d" in err and "Traceback" not in err


@pytest.mark.parametrize(
    "message",
    [
        "VONC: no detrend record for this station; serving raw series",
        "outlier detection failed (x); serving unflagged data",
        "Cannot determine plate velocity at station LAMB",
    ],
)
def test_strict_refuses_degraded_output_before_writing(
    tmp_path, monkeypatch, message
) -> None:
    monkeypatch.setattr(tplt, "plotTime", _fake_plot(tmp_path, warn=message))
    lax = driver.tryTimes("VONC", **KW)
    assert lax.status == "ok"
    (tmp_path / "VONC-plate-90d.png").unlink()
    strict = driver.tryTimes("VONC", strict=True, **KW)
    assert strict.status == "failed" and "UserWarning" in strict.reason
    assert not (tmp_path / "VONC-plate-90d.png").exists()


def test_strict_lets_other_warnings_through(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        tplt, "plotTime", _fake_plot(tmp_path, warn="VONC: no model line (x)")
    )
    assert driver.tryTimes("VONC", strict=True, **KW).status == "ok"


def test_summary_exit_status_and_manifest(tmp_path, capsys) -> None:
    P = driver.PlotOutcome
    outcomes = [
        P("B", "plate/90d", "ok", files=[str(tmp_path / "b.png")]),
        P("A", "plate/90d", "ok", files=[str(tmp_path / "a.png")]),
        P("C", "plate/90d", "skipped", "no data for station C"),
    ]
    manifest = tmp_path / "manifest.txt"
    assert driver.summarize(outcomes, str(manifest)) == 0
    assert manifest.read_text().splitlines() == [
        str(tmp_path / "a.png"),
        str(tmp_path / "b.png"),
    ]
    outcomes.append(P("D", "plate/90d", "failed", "ValueError: x"))
    assert driver.summarize(outcomes) == 1
    err = capsys.readouterr().err
    assert "2 plotted (2 files), 1 skipped (no data in period), 1 failed" in err
    assert "failed D plate/90d: ValueError: x" in err


def test_nothing_usable_in_the_period_is_a_skip(monkeypatch) -> None:
    def plot(sta: str, **_: Any) -> Any:
        raise ValueError(
            "uncertainty screening left no nonzero data and no offset was given"
        )

    monkeypatch.setattr(tplt, "plotTime", plot)
    assert driver.tryTimes("ROTH", **KW).status == "skipped"


def test_active_stations_keep_passive_drop_retired() -> None:
    import configparser
    from types import SimpleNamespace

    from gps_plot.cli.common import active_stations

    cfg = configparser.ConfigParser()
    cfg.read_string(
        """
[DEFAULTS]
x = 1
[VONC]
station_status =
[ZIMM]
station_role = passive   # IGS
[KARV]
station_status = discontinued
[OLDS]
station_status = inactive
[SENG]
station_role = active
[BJAC]
station_role = passive
station_type = campaign  # benchmark
"""
    )
    assert active_stations(SimpleNamespace(config=cfg)) == ["VONC", "ZIMM", "SENG"]


def test_special_list_validates_each_item() -> None:
    import argparse

    from gps_plot.cli.common import special_list

    check = special_list(["all", "90d", "year", "full"])
    assert check("90d, year,full") == "90d,year,full"
    for bad in ("90d,decade", ",", "week"):
        with pytest.raises(argparse.ArgumentTypeError):
            check(bad)


def test_special_list_expands_to_one_variant_each() -> None:
    specials = ["90d", "year", "full", "fixedstart"]
    got = driver._expand_variants(
        {"ref": "plate", "special": "90d,year"}, ["plate"], specials
    )
    assert [v["special"] for v in got] == ["90d", "year"]
    got = driver._expand_variants({"ref": "plate", "special": "full,all"}, [], specials)
    assert [v["special"] for v in got] == ["full", "90d", "year", "fixedstart"]


def test_operational_fills_defaults_but_an_explicit_flag_wins() -> None:
    parser = driver.build_parser(config=None)
    args = parser.parse_args(["VONC", "--operational", "--outlier-sigma", "5"])
    driver.apply_operational(args, parser)
    assert args.view == "cleaned" and args.strict and args.legacy_names
    assert (args.show_outliers, args.show_obstruction) == (10, 30)
    assert args.outlier_sigma == 5


def test_is_no_data() -> None:
    assert tplt.is_no_data(tplt.NoDataError("no data for station X"))
    assert tplt.is_no_data(ValueError("uncertainty screening left no nonzero data"))
    assert not tplt.is_no_data(ValueError("no plate assigned"))
