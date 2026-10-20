"""Station selection for `gpsplot ts`: 'all', --region, --with-record."""

from __future__ import annotations

import argparse

import pytest

from gps_plot.cli import common


def _pos(path, sta, lat, lon_east):
    path.write_text(
        "PBO Station Position Time Series. Reference Frame : IGb14\n"
        f"4-character ID: {sta}\n"
        f"NEU Reference position :    {lat}  {lon_east} 100.0 (IGb14/WGS84)\n"
        "*YYYYMMDD HHMMSS\n"
    )


@pytest.fixture
def pos_dir(tmp_path):
    _pos(tmp_path / "GRAN.imo.24hr_itrf14.pos", "GRAN", 65.9186, 342.4214)
    _pos(tmp_path / "REYK.imo.24hr_itrf14.pos", "REYK", 64.1388, 338.0447)
    _pos(tmp_path / "SEYD.imo.24hr_itrf14.pos", "SEYD", 65.2600, 346.0000)
    _pos(tmp_path / "ONSA.imo.24hr_itrf14.pos", "ONSA", 57.3953, 11.9255)
    return tmp_path


def _kw(pos_dir):
    return {"tType": "POS", "pos_id": "imo.24hr_itrf14", "Dir": str(pos_dir)}


def test_all_is_the_pos_stations_in_the_iceland_box(pos_dir):
    got = common.resolve_stations(["all"], _kw(pos_dir), config=None)
    assert got == ["GRAN", "REYK", "SEYD"]  # ONSA (Sweden) left out


def test_region_replaces_the_box_and_filters_lists(pos_dir):
    north = (-19.0, -13.0, 65.0, 67.0)
    assert common.resolve_stations(["all"], _kw(pos_dir), None, region=north) == [
        "GRAN",
        "SEYD",
    ]
    listed = common.resolve_stations(
        ["REYK", "GRAN", "GRAN-SEYD", "REYK-SEYD"], _kw(pos_dir), None, region=north
    )
    assert listed == ["GRAN", "GRAN-SEYD"]  # a baseline needs both ends inside


def test_with_record_keeps_stations_and_baselines_with_records(pos_dir, monkeypatch):
    from geo_dataread import gps_views

    from gps_plot import timesmatplt

    monkeypatch.setattr(timesmatplt, "record_context", lambda *a: ("f", "doc"))
    monkeypatch.setattr(
        gps_views,
        "read_detrend_params",
        lambda p: {"stations": {"GRAN": {}, "SEYD": {}}},
    )
    got = common.resolve_stations(
        ["all", "GRAN-SEYD", "REYK-SEYD"], _kw(pos_dir), None, with_record=True
    )
    assert got == ["GRAN", "SEYD", "GRAN-SEYD"]


def test_region_arg_parses_and_refuses():
    assert common.region_arg("-25/-13/63/67") == (-25.0, -13.0, 63.0, 67.0)
    with pytest.raises(argparse.ArgumentTypeError):
        common.region_arg("-13/-25/63/67")
    with pytest.raises(argparse.ArgumentTypeError):
        common.region_arg("iceland")
