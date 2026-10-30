"""gpsplot direction: the motion direction of a series, as a config line."""

from __future__ import annotations

import argparse
import datetime as dt

import numpy as np
import pytest

from gps_plot import direction
from gps_plot.cli import find_command


def test_window_units() -> None:
    assert direction.window_days("7d") == 7.0
    assert direction.window_days("6h") == 0.25
    assert direction.window_days("30min") == pytest.approx(1 / 48)
    with pytest.raises(argparse.ArgumentTypeError):
        direction.window_days("week")


def test_registered() -> None:
    cmd = find_command("direction")
    assert cmd is not None and cmd.target == "gps_plot.cli.direction:run"


def _ramp(az: float, n: int = 60, size: float = 20.0):
    t = 20000.0 + np.arange(n, dtype=float)  # days since 1970, daily
    a = np.radians(az)
    y = np.array([np.cos(a), np.sin(a), 0.0])[:, None] * np.linspace(0, size, n)
    return t, y, np.ones_like(y)


def test_run_prints_one_line_per_series(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    data = {"AAAA-BBBB": _ramp(205.0), "CCCC-DDDD": _ramp(0.0, size=0.5)}

    def daily(spec, start, end, args, config):
        if spec not in data:
            raise ValueError("no data for station")
        return data[spec]

    monkeypatch.setattr(direction, "_daily", daily)
    parser = direction.build_parser()
    args = parser.parse_args(
        ["AAAA-BBBB", "CCCC-DDDD", "EEEE-FFFF", "-s", "20240101", "-e", "20240401"]
    )
    assert direction.run(args, parser, None) == 0
    out, err = capsys.readouterr()
    lines = out.splitlines()
    assert lines[0].startswith("AAAA-BBBB: {motion: 205.0}  # ±")
    assert "NOT significant" not in lines[0]
    assert lines[1].startswith("CCCC-DDDD:") and "NOT significant" in lines[1]
    assert "EEEE-FFFF: no data" in err
    # the span printed is the data's own
    t0 = dt.datetime(1970, 1, 1) + dt.timedelta(days=20000.0)
    assert f"{t0:%Y-%m-%d}" in lines[0]


def test_start_after_end_is_an_error() -> None:
    parser = direction.build_parser()
    args = parser.parse_args(["A-B", "-s", "20240401", "-e", "20240101"])
    with pytest.raises(SystemExit):
        direction.run(args, parser, None)


def test_parse_span_and_slope() -> None:
    a, b = direction.parse_span("20260920:20261029")
    assert (a, b) == (dt.datetime(2026, 9, 20), dt.datetime(2026, 10, 29))
    a, b = direction.parse_span("20261029-12:00:20261029-18:30")
    assert a.hour == 12 and b.minute == 30
    for bad in ("20261029", "20261029:20260920", "2026-10-29:2026-10-30"):
        with pytest.raises(argparse.ArgumentTypeError):
            direction.parse_span(bad)
    assert direction.parse_slope("210,28") == (210.0, 28.0)
    with pytest.raises(argparse.ArgumentTypeError):
        direction.parse_slope("210,95")


def test_resolve_directions_precedence(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    table = {"AAAA-BBBB": {"motion": 10.0, "slope": (200.0, 20.0)}}
    monkeypatch.setattr(direction, "config_directions", lambda config: table)

    class _D:
        azimuth, sigma_azimuth, significant = 99.0, 2.0, True

    monkeypatch.setattr(direction, "estimate", lambda *a, **k: (_D(), 20000.0, 20030.0))
    pairs = ["AAAA-BBBB", "CCCC-DDDD"]
    cfg = object()
    assert direction.resolve_directions(pairs, "along", config=cfg) == {}
    got = direction.resolve_directions(pairs, "motion", config=cfg)
    assert got == {"AAAA-BBBB": table["AAAA-BBBB"]}  # CCCC-DDDD: none → left out
    span = (dt.datetime(2026, 1, 1), dt.datetime(2026, 2, 1))
    got = direction.resolve_directions(pairs, "motion", span=span, config=cfg)
    assert got["AAAA-BBBB"]["motion"] == 99.0 and got["CCCC-DDDD"]["motion"] == 99.0
    assert "motion 99.0°" in capsys.readouterr().err
    got = direction.resolve_directions(pairs, "motion", motion=5.0, span=span,
                                       config=cfg)  # fmt: skip
    assert {p: g["motion"] for p, g in got.items()} == {p: 5.0 for p in pairs}
    got = direction.resolve_directions(pairs, "slope", config=cfg)
    assert list(got) == ["AAAA-BBBB"]
