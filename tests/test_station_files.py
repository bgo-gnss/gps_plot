"""gps_plot.sources.station_files: --stations-file in three layouts."""

from __future__ import annotations

import json

import pytest

from gps_plot import maps
from gps_plot.sources import station_files as sf


def read(path, label="L"):
    return [(c.marker, c.lon, c.lat, c.name, c.status)
            for c in sf.read_stations(path, label, maps)]  # fmt: skip


def test_text_csv_and_geojson(tmp_path):
    txt = tmp_path / "planned.txt"
    txt.write_text("# sites\npla1 -19.0 64.5 Sprengi sandur\n\nPLA2 -16.5 65.2  # ok\n")
    assert read(txt) == [
        ("PLA1", -19.0, 64.5, "Sprengi sandur", "L"),
        ("PLA2", -16.5, 65.2, None, "L"),
    ]
    csv = tmp_path / "s.csv"
    csv.write_text("Station,Longitude,Latitude,Name\nAAAA,-20,64,A\nBBBB,-21,65,\n")
    assert read(csv) == [
        ("AAAA", -20.0, 64.0, "A", "L"),
        ("BBBB", -21.0, 65.0, None, "L"),
    ]
    gj = tmp_path / "s.geojson"
    gj.write_text(json.dumps({"type": "FeatureCollection", "features": [
        {"type": "Feature", "geometry": {"type": "Point", "coordinates": [-20, 64]},
         "properties": {"marker": "cccc", "name": "C"}},
        {"type": "Feature", "geometry": {"type": "LineString", "coordinates": []},
         "properties": {}},
    ]}))  # fmt: skip
    assert read(gj) == [("CCCC", -20.0, 64.0, "C", "L")]


def test_bad_files_name_the_line(tmp_path):
    txt = tmp_path / "bad.txt"
    txt.write_text("AAAA -20 64\nBBBB -20\n")
    with pytest.raises(ValueError, match="line 2: use MARKER LON LAT"):
        sf.read_stations(txt, "L", maps)
    txt.write_text("AAAA west 64\n")
    with pytest.raises(ValueError, match="not numbers"):
        sf.read_stations(txt, "L", maps)
    csv = tmp_path / "bad.csv"
    csv.write_text("name,x\nA,1\n")
    with pytest.raises(ValueError, match="no marker, lat column"):
        sf.read_stations(csv, "L", maps)
    empty = tmp_path / "empty.txt"
    empty.write_text("# nothing\n")
    with pytest.raises(ValueError, match="no stations"):
        sf.read_stations(empty, "L", maps)


def test_split_spec(tmp_path):
    assert sf.split_spec("a/planned.txt") == ("a/planned.txt", "planned")
    assert sf.split_spec("a/planned.txt:Planned sites") == (
        "a/planned.txt",
        "Planned sites",
    )
    odd = tmp_path / "x:y.txt"
    odd.write_text("")
    assert sf.split_spec(str(odd)) == (str(odd), "x:y")
