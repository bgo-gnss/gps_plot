"""--along: one station along a direction (PLAN-ts-quakes Q-T5)."""

from __future__ import annotations

import argparse
import dataclasses
import math

import numpy as np
import pytest

from gps_plot.axes import station_frame
from gps_plot.direction import along_azimuth, parse_along
from gps_plot.layers import rate as lr


def test_parse_along():
    assert parse_along("opt") == "opt"
    assert parse_along("motion") == "motion"
    assert parse_along("284.8") == "284.8"
    for bad in ("north", "360", "-1"):
        with pytest.raises(argparse.ArgumentTypeError):
            parse_along(bad)


def test_station_frame_projects_toward_the_azimuth():
    ax = station_frame(90.0)  # east
    v = ax.matrix @ np.array([0.0, 10.0, 3.0])  # N, E, U
    np.testing.assert_allclose(v, [10.0, 0.0, 3.0], atol=1e-12)
    assert ax.keys == ("motion", "across", "up")
    # across is 90° clockwise of the motion: for motion north, across = east
    v = station_frame(0.0).matrix @ np.array([0.0, 4.0, 0.0])
    np.testing.assert_allclose(v, [0.0, 4.0, 0.0], atol=1e-12)


def test_along_azimuth_number_and_missing_motion():
    t = np.linspace(2020.0, 2021.0, 50)
    y = np.zeros((3, 50))
    s = np.ones((3, 50))
    assert along_azimuth("XXXX", "284.8", t, y, s)[0] == pytest.approx(284.8)
    with pytest.raises(ValueError, match="directions"):
        along_azimuth("XXXX", "motion", t, y, s, None)


def test_along_azimuth_opt_recovers_the_motion():
    t = np.arange(2020.0, 2022.0, 1 / 365.25)
    az = math.radians(285.0)
    rng = np.random.default_rng(0)
    d = 20.0 * (t - t[0])
    y = np.vstack([d * math.cos(az), d * math.sin(az), 0 * d])
    y = y + rng.normal(0.0, 1.0, y.shape)
    got, note = along_azimuth("XXXX", "opt", t, y, np.ones_like(y))
    assert got == pytest.approx(285.0, abs=1.0)
    assert "besta stefna" in note


@dataclasses.dataclass(frozen=True)
class _Rec:
    rate: tuple[float, float, float]
    sigma: tuple[float, float, float]
    corr_en: float = 0.0
    accepted: bool = True


def test_rotate_record_rate_and_sigma():
    rec = _Rec((3.0, 4.0, 7.0), (1.0, 2.0, 5.0))
    out = lr.rotate_record(rec, station_frame(90.0))
    np.testing.assert_allclose(out.rate, (4.0, -3.0, 7.0), atol=1e-12)
    np.testing.assert_allclose(out.sigma, (2.0, 1.0, 5.0), atol=1e-12)
    # 45°: σ_motion² = (σn² + σe² + 2ρσnσe)/2
    out = lr.rotate_record(_Rec((0, 0, 0), (1.0, 1.0, 1.0), corr_en=0.5),
                           station_frame(45.0))  # fmt: skip
    assert out.sigma[0] == pytest.approx(math.sqrt(1.5))
    assert out.sigma[1] == pytest.approx(math.sqrt(0.5))
    assert lr.rotate_record(rec, None) is rec


def test_the_ts_parser_refuses_along_with_model():
    from gps_plot.plot_gps_timeseries import build_parser, run

    parser = build_parser()
    bad = parser.parse_args(["VONC", "--along", "opt", "--model", "curve"])
    with pytest.raises(SystemExit):
        run(bad, parser)
