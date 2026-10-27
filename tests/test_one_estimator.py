"""gpsprod step 6: one estimator — gpsprod velocity, gpsplot map velocity and
gpsplot ts --rate build the same velocity-product settings (and so the same
fingerprint and numbers) for the same station and window."""

from __future__ import annotations

import argparse

import pytest

from gps_plot import map as gmap
from gps_plot.layers.rate import rate_settings


def _gpsprod(*extra: str):
    from geo_dataread.products import cli

    args = cli.build_parser().parse_args(
        ["velocity", "SENG", "--start", "20240901", "--end", "20241031",
         "--pos-dir", "/pos", *extra]
    )  # fmt: skip
    return cli.velocity_settings(args)


def _map(ref: str | None):
    args = argparse.Namespace(
        tType=None, session="24h", ref=ref, frame=None, view=None, uncert=10.0,
        Dir="/pos", min_epochs=None, pre_window=None,
    )  # fmt: skip
    return gmap._product_settings(args, argparse.ArgumentParser(), None)


@pytest.mark.parametrize(
    ("prod_ref", "map_ref", "ts_ref", "terms"),
    [
        ("plate-periodic", None, "detrend", ["periodic"]),
        ("background", "detrend", "detrend", ["all"]),
        ("plate", "plate", "plate", None),
    ],
)
def test_three_entry_points_one_settings(prod_ref, map_ref, ts_ref, terms) -> None:
    from geo_dataread.products.velocity_core import _fingerprint_settings

    prod = _gpsprod("--ref", prod_ref)
    mp = _map(map_ref)
    ts = rate_settings(
        ref=ts_ref, detrend_terms=terms, view="raw", pos_id="imo.24hr_itrf14",
        uncert=10.0, Dir="/pos",
    )  # fmt: skip
    assert prod == mp == ts
    assert (
        _fingerprint_settings(prod)
        == _fingerprint_settings(mp)
        == _fingerprint_settings(ts)
    )
    assert prod.frame == "itrf2014" and prod.session == "24h"


def test_eight_hour_sessions_agree_too() -> None:
    prod = _gpsprod("--session", "8h", "--frame", "itrf20")
    args = argparse.Namespace(
        tType=None, session="8h", ref=None, frame="itrf20", view=None, uncert=10.0,
        Dir="/pos", min_epochs=None, pre_window=None,
    )  # fmt: skip
    mp = gmap._product_settings(args, argparse.ArgumentParser(), None)
    ts = rate_settings(ref="detrend", detrend_terms=["periodic"], view="raw",
                       pos_id="imo.08hr_itrf20", uncert=10.0, Dir="/pos")  # fmt: skip
    assert prod == mp == ts and prod.session == "8h"
