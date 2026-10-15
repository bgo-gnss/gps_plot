"""TOS GPS stations for maps: current attribute values, status, the cache."""

from __future__ import annotations

import datetime as dt
import json

import pytest

from gps_plot.sources import tos

TODAY = dt.date(2026, 10, 15)


def attr(code, value, start="2000-01-01T00:00:00", stop=None):
    return {"code": code, "value": value, "date_from": start, "date_to": stop}


def entity(marker, *extra, subtype="GPS stöð", lat="64.0", lon="-21.0"):
    attrs = [attr("marker", marker.lower()), attr("lat", lat), attr("lon", lon)]
    if subtype is not None:
        attrs.append(attr("subtype", subtype))
    return {"attributes": [*attrs, *extra]}


def by_marker(entities):
    return {s.marker: s for s in tos.gps_stations(entities, today=TODAY)}


def test_only_gps_stations_by_their_open_subtype():
    got = by_marker([
        entity("AAAA"),
        entity("SILX", subtype="SIL stöð"),
        # TKJ2: a GPS station with a closed second subtype
        entity("TKJ2", attr("subtype", "Tungnakvislarjokulsskrida lower ",
                            "2020-06-29T00:00:00", "2020-06-29T00:00:00")),
        entity("NOPOS", lat=None),
    ])  # fmt: skip
    assert sorted(got) == ["AAAA", "TKJ2"]


def test_a_moved_station_is_placed_at_its_open_position():
    moved = {"attributes": [
        attr("marker", "movd"), attr("subtype", "GPS stöð"),
        attr("lat", "63.0", stop="2010-01-01T00:00:00"), attr("lon", "-20.0", stop="2010-01-01T00:00:00"),
        attr("lat", "64.5", start="2010-01-01T00:00:00"), attr("lon", "-19.5", start="2010-01-01T00:00:00"),
    ]}  # fmt: skip
    s = by_marker([moved])["MOVD"]
    assert (s.lat, s.lon) == (64.5, -19.5)


@pytest.mark.parametrize(
    ("extra", "status"),
    [
        ([], "active"),
        ([attr("continuity", "campaign")], "campaign"),
        ([attr("continuity", "campaign", stop="2020-10-02T00:00:00")], "active"),
        ([attr("operational_status", "external")], "external"),
        # an open end, and the zero-length period TOS also records an end as
        ([attr("date_end", "2023-07-12 00:00", "2023-07-12T00:00:00")], "ended"),
        ([attr("date_end", "2006-01-01 00:00\t", "2006-01-01T00:00:00",
               "2006-01-01T00:00:00")], "ended"),
        # an end revoked later: the station runs again (GRVV after Grindavík)
        ([attr("date_end", "2025-05-06 00:00", "2025-05-06T00:00:00",
               "2025-11-04T12:00:00")], "active"),
        # an end still to come
        ([attr("date_end", "2027-01-01 00:00", "2027-01-01T00:00:00")], "active"),
    ],
)  # fmt: skip
def test_status(extra, status):
    assert by_marker([entity("STAT", *extra)])["STAT"].status == status


def test_one_marker_two_entities_the_running_one_wins():
    old = entity("SOHO", attr("date_end", "2021-11-19 00:00", "2021-11-19T00:00:00"),
                 lat="63.5")  # fmt: skip
    new = entity("SOHO", attr("date_start", "2023-07-24"), lat="63.6")
    for order in ([old, new], [new, old]):
        s = by_marker(order)["SOHO"]
        assert (s.status, s.lat) == ("active", 63.6)


def test_cache_is_used_refreshed_and_never_empty(tmp_path):
    path = tmp_path / "fleet.json"
    calls = []

    def fetch():
        calls.append(1)
        return [entity("AAAA")]

    assert tos.load_entities(path=path, fetch=fetch) == [entity("AAAA")]
    assert tos.load_entities(path=path, fetch=fetch) == [entity("AAAA")]
    assert len(calls) == 1
    tos.load_entities(path=path, fetch=fetch, refresh=True)
    assert len(calls) == 2
    assert json.loads(path.read_text()) == [entity("AAAA")]
    with pytest.raises(RuntimeError, match="no stations"):
        tos.load_entities(path=path, fetch=lambda: [], refresh=True)
