"""--rate-ref: the series and the rates relative to a window's rate."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from gps_plot.layers import rate as lr


@dataclasses.dataclass(frozen=True)
class _Rec:
    rate: tuple[float, float, float]
    sigma: tuple[float, float, float]
    window: tuple[float, float]
    accepted: bool = True
    refused: str | None = None


def test_subtract_rate_keeps_zero_at_t0_and_input():
    t = np.array([2014.0, 2014.5, 2015.0])
    y = np.array([[0.0, 1.0, 2.0], [0.0, 5.0, 10.0], [1.0, 1.0, 1.0]])
    before = y.copy()
    out = lr.subtract_rate(t, y, (2.0, 10.0, 0.0), 2014.0)
    np.testing.assert_allclose(out, [[0, 0, 0], [0, 0, 0], [1, 1, 1]])
    np.testing.assert_array_equal(y, before)


def test_relative_record_subtracts_and_adds_sigma_in_quadrature():
    ref = _Rec((5.0, 10.0, 30.0), (1.0, 1.0, 4.0), (2013.67, 2014.08))
    rec = _Rec((5.0, -16.0, 30.0), (3.0, 1.0, 3.0), (2014.08, 2014.62))
    out = lr.relative_record(rec, ref)
    assert out.rate == (0.0, -26.0, 0.0)
    assert out.sigma == pytest.approx((math.sqrt(10), math.sqrt(2), 5.0))


def test_relative_record_same_window_keeps_its_sigma():
    ref = _Rec((5.0, 10.0, 30.0), (1.0, 1.0, 4.0), (2013.67, 2014.08))
    out = lr.relative_record(ref, ref)
    assert out.rate == (0.0, 0.0, 0.0)
    assert out.sigma == ref.sigma


def test_relative_record_passes_refused_and_none():
    rec = _Rec((math.nan,) * 3, (math.nan,) * 3, (0.0, 1.0), accepted=False)
    ref = _Rec((1.0, 1.0, 1.0), (1.0, 1.0, 1.0), (2.0, 3.0))
    assert lr.relative_record(rec, ref) is rec
    assert lr.relative_record(ref, None) is ref


def test_the_ts_parser_takes_rate_ref_and_refuses_it_with_model():
    from gps_plot.plot_gps_timeseries import build_parser, run

    parser = build_parser()
    args = parser.parse_args(["VONC", "--rate-ref", "20130901:20140131"])
    assert args.rate_ref[0] < args.rate_ref[1]
    bad = parser.parse_args(
        ["VONC", "--rate-ref", "20130901:20140131", "--model", "curve"]
    )
    with pytest.raises(SystemExit):
        run(bad, parser)
    with pytest.raises(SystemExit):
        parser.parse_args(["VONC", "--rate-ref", "20140131:20130901"])


@pytest.mark.parametrize(
    ("given", "ref", "terms"),
    [
        ({"rate_ref": (2013.67, 2014.08)}, "detrend", ("periodic",)),
        ({"along": "opt"}, "detrend", ("periodic",)),
        ({"rate_ref": (2013.67, 2014.08), "ref": "plate"}, "plate", None),
        ({"along": "opt", "detrend_terms": ("linear",)}, None, ("linear",)),
        ({"rate_ref": (2013.67, 2014.08), "model": "curve"}, None, None),
        ({}, None, None),
    ],
)
def test_seasonal_free_default(given, ref, terms):
    from gps_plot.plot_gps_timeseries import seasonal_free_default

    kw = {"ref": None, "model": None, "detrend_terms": None, "rate_ref": None,
          "along": None, **given}  # fmt: skip
    seasonal_free_default(kw)
    assert kw["ref"] == ref and kw["detrend_terms"] == terms


def test_seasonal_kept():
    assert lr.seasonal_kept("plate", None)
    assert lr.seasonal_kept("itrf2008", None)
    assert not lr.seasonal_kept("detrend", None)  # default terms remove it
    assert not lr.seasonal_kept("detrend", ("periodic",))
    assert lr.seasonal_kept("detrend", ("linear",))
