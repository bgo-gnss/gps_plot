"""What ``gpsplot ts`` hands to ``plotTime``, pinned per command line.

The pixel gate checks rendered plots; it never sees argument parsing or the
resolution in ``run`` (dates, frame/session -> product, directories, the
``--ref``/``--special`` fan-out). A dest added, renamed or dropped there
reaches ``plotTime(**kwargs)`` as a TypeError, which ``tryTimes`` catches
per station, so it would show only as a missing file. This pins the exact
variant kwargs for a table of argv against ``data/ts_kwargs_snapshot.json``.

Regenerate (only for an intended change): ``GPS_PLOT_SNAPSHOT=write pytest
tests/test_ts_kwargs_snapshot.py``.
"""

from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path

import pytest

SNAPSHOT = Path(__file__).parent / "data" / "ts_kwargs_snapshot.json"

ARGV: dict[str, list[str]] = {
    "plain": ["HOFN"],
    "dates": ["HOFN", "-s", "20240101", "-e", "20250101", "--fix"],
    "shiftD": ["HOFN", "-D", "30"],
    "fix": ["HOFN", "--fix"],
    "special": ["RHOF", "--special", "90d"],
    "all": ["RHOF", "SENG", "--ref", "all", "--special", "all"],
    "frame08": ["RHOF", "--ref", "ref_frame", "--frame", "itrf2008"],
    "session8h": ["SENG", "--session", "8h", "-u", "20"],
    "sessionboth": ["SENG", "--session", "both", "-s", "20250101"],
    "tot": ["REYK", "--tType", "TOT"],
    "dirs": ["HOFN", "-d", "/tmp/figs", "-i", "/tmp/pos"],
    "detrend": ["SENG", "--ref", "detrend", "--detrend-terms", "periodic"],
    "model": ["HOFN", "--model", "both", "--special", "year"],
    "modelterms": [
        "HOFN", "--model", "residual", "--detrend-terms", "periodic",
        "--model-terms", "linear,periodic",
    ],
    "cleaned": [
        "SENG", "--view", "cleaned", "--outlier-param", "window_n_sigma=3.0",
        "--show-outliers", "5", "--stages", "S0,S4",
    ],
    "steps": ["SELF", "--ref", "detrend", "--remove-steps", "earthquake",
              "--annotate-steps"],
    "baseline": ["VFLN-VFLS", "--show-ends", "--baseline-sigma", "empirical"],
    "name": ["HOFN", "--name", "HOFN-test", "--ylim", "5"],
    "rate": ["SENG", "--ref", "plate", "--rate"],
    "ratewindow": ["SENG", "--ref", "detrend", "--rate-window", "20260601:20261015",
                   "--rate-window", "20250901:20251231"],
}  # fmt: skip


def _fixed_now(days: int = 0, refday: dt.datetime | None = None) -> dt.datetime:
    base = refday or dt.datetime(2026, 10, 14)
    return base + dt.timedelta(days=days)


class _Config:
    """A stand-in postprocess config: the same paths on every host."""

    def getPostProcessDir(self, option: str) -> str:
        return f"/cfg/{option}"

    def getStationInfo(self) -> list[str]:
        return ["HOFN", "RHOF"]


def _capture(argv: list[str], monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    from gps_plot import plot_gps_timeseries as driver

    got: list[dict] = []
    monkeypatch.setattr(driver, "currDatetime", _fixed_now)
    monkeypatch.setattr(
        driver, "plotStation", lambda sta, variants, **_: got.extend(
            {"sta": sta, **v} for v in variants
        ) or [],
    )  # fmt: skip
    config = _Config()
    parser = driver.build_parser(config=config)
    driver.run(parser.parse_args(argv + ["--save", "png", "-j", "1"]), parser, config)
    return [{k: repr(v) for k, v in sorted(d.items())} for d in got]


def test_ts_kwargs_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    got = {name: _capture(argv, monkeypatch) for name, argv in ARGV.items()}
    if os.environ.get("GPS_PLOT_SNAPSHOT") == "write":
        SNAPSHOT.write_text(json.dumps(got, indent=1, sort_keys=True) + "\n")
        pytest.skip("snapshot written")
    want = json.loads(SNAPSHOT.read_text())
    assert sorted(got) == sorted(want)
    for name in ARGV:
        assert got[name] == want[name], name


@pytest.mark.parametrize(
    "argv, message",
    [
        (["HOFN", "--detrend-terms", "periodic"], "--detrend-terms needs --ref detrend"),
        (["HOFN", "--frame", "itrf2020", "--pos-id", "imo.24hr_itrf20"],
         "give --frame/--session or --pos-id, not both"),
        (["HOFN", "--frame", "itrf1997"], "itrf1997"),
        (["HOFN", "--ref", "detrend", "--detrend-terms", "seasonal"], "seasonal"),
    ],
)  # fmt: skip
def test_ts_errors_unchanged(argv, message, capsys, monkeypatch) -> None:
    from gps_plot import plot_gps_timeseries as driver

    monkeypatch.setattr(driver, "plotStation", lambda sta, variants: None)
    config = _Config()
    parser = driver.build_parser(config=config)
    with pytest.raises(SystemExit):
        driver.run(parser.parse_args(argv + ["--save", "png"]), parser, config)
    assert message in capsys.readouterr().err
