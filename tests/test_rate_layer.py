"""gpsplot ts --rate: the velocity product's rate on the panels."""

from __future__ import annotations

import argparse

import pytest

from gps_plot import plot_gps_timeseries as pgt
from gps_plot.layers import rate as layer

BASE = {"view": "raw", "pos_id": "imo.24hr_itrf14", "uncert": 10, "Dir": "/x"}


@pytest.mark.parametrize(
    ("ref", "terms", "vref"),
    [
        ("plate", None, "plate"),
        ("detrend", None, "background"),
        ("detrend", ("linear", "periodic"), "background"),
        ("detrend", ("linear", "periodic", "steps"), "background"),
        ("detrend", ("periodic",), "plate-periodic"),
    ],
)
def test_the_plot_ref_maps_to_the_product_ref(ref, terms, vref):
    s = layer.rate_settings(ref=ref, detrend_terms=terms, **BASE)
    assert (s.ref, s.frame, s.view, s.uncert) == (vref, "itrf14", "raw", 10.0)


def test_refs_without_a_product_counterpart_are_refused():
    with pytest.raises(ValueError, match="--detrend-terms linear"):
        layer.rate_settings(ref="detrend", detrend_terms=("linear",), **BASE)
    with pytest.raises(ValueError, match="--ref itrf2008"):
        layer.rate_settings(ref="itrf2008", detrend_terms=None, **BASE)
    with pytest.raises(ValueError, match="--view detrended"):
        layer.rate_settings(ref="plate", detrend_terms=None,
                            **{**BASE, "view": "detrended"})  # fmt: skip


def test_tex_escapes_refusal_reasons():
    assert layer._tex("n_obs=4<5; cover=0.37<0.8") == (
        "n\\_obs=4$<$5; cover=0.37$<$0.8"
    )


def test_rate_window_takes_whole_days():
    a, b = pgt._rate_window("20260101:20260101")
    assert b - a == pytest.approx(1 / 365, rel=1e-6)
    with pytest.raises(argparse.ArgumentTypeError):
        pgt._rate_window("20260102:20260101")
    with pytest.raises(argparse.ArgumentTypeError):
        pgt._rate_window("2026-01-01")


def test_default_window_is_whole_days_like_the_map():
    """ts --rate without --rate-window: the map's window over the same dates,
    so the same T, k(T) and σ."""
    import datetime as dt

    import numpy as np
    from geo_dataread.velocity import window_bounds, yearf

    from gps_plot.layers.rate import default_window

    noon = np.array([yearf(dt.datetime(2026, 8, 15, 12)),
                     yearf(dt.datetime(2026, 10, 15, 12))])  # fmt: skip
    want = window_bounds(dt.date(2026, 8, 15), dt.date(2026, 10, 15))
    assert default_window(noon, None, None) == want
    # the read bounds (-s/-e, midnights) win over the epochs
    s, e = yearf(dt.date(2026, 8, 15)), yearf(dt.date(2026, 10, 15))
    assert default_window(noon[:1] + 0.01, s, e) == want
