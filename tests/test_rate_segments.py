"""gpsplot ts --rate-segments: the spans between events."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from gps_plot.layers import rate as lr

D = dt.date


def test_spans_between_gaps_take_out_the_gap_days():
    spans = lr.segment_spans(
        D(2024, 1, 1), D(2024, 12, 31),
        [(D(2024, 6, 1), D(2024, 6, 1)), (D(2024, 3, 1), D(2024, 3, 20))],
        min_days=1,
    )  # fmt: skip
    assert spans == [
        (D(2024, 1, 1), D(2024, 2, 29)),
        (D(2024, 3, 21), D(2024, 5, 31)),
        (D(2024, 6, 2), D(2024, 12, 31)),
    ]


def test_spans_drop_short_ones_and_handle_overlap_and_edges():
    gaps = [
        (D(2023, 12, 1), D(2024, 1, 10)),  # starts before the window
        (D(2024, 2, 1), D(2024, 2, 20)),
        (D(2024, 2, 10), D(2024, 3, 1)),  # overlaps the previous gap
        (D(2024, 12, 20), D(2025, 1, 31)),  # runs past the end
    ]
    spans = lr.segment_spans(D(2024, 1, 1), D(2024, 12, 31), gaps, min_days=30)
    # 11 Jan - 31 Jan is 21 days: dropped
    assert spans == [(D(2024, 3, 2), D(2024, 12, 19))]


def test_no_gaps_is_one_span():
    assert lr.segment_spans(D(2024, 1, 1), D(2024, 1, 31), []) == [
        (D(2024, 1, 1), D(2024, 1, 31))
    ]


def test_event_gaps_span_or_onset():
    from geo_dataread.events import Event

    e = Event("x", "eruption", D(2024, 3, 16), D(2024, 5, 9))
    assert lr.event_gaps([e]) == [(D(2024, 3, 16), D(2024, 5, 9))]
    assert lr.event_gaps([e], cut="onset") == [(D(2024, 3, 16), D(2024, 3, 16))]


def test_step_gaps_are_the_step_days():
    # 2023-11-10 at noon
    from geo_dataread.velocity import yearf

    epoch = yearf(dt.datetime(2023, 11, 10, 12))
    assert lr.step_gaps([epoch]) == [(D(2023, 11, 10), D(2023, 11, 10))]
    assert lr.step_gaps([]) == []


def test_segment_windows_are_whole_days_that_exclude_the_step():
    from geo_dataread.velocity import window_bounds, yearf

    t = np.array([yearf(D(2024, 1, 1)), yearf(D(2024, 12, 31))])
    windows = lr.segment_windows(
        t, None, None, [(D(2024, 6, 1), D(2024, 6, 1))], min_days=1
    )
    assert windows == [
        window_bounds(D(2024, 1, 1), D(2024, 5, 31)),
        window_bounds(D(2024, 6, 2), D(2024, 12, 31)),
    ]
    step = yearf(dt.datetime(2024, 6, 1, 12))
    assert all(not (a < step < b) for a, b in windows)


def test_parse_segment_sources():
    assert lr.parse_segment_sources("steps") == ("steps",)
    assert lr.parse_segment_sources("events, steps,events") == ("events", "steps")
    for bad in ("", "dikes", "steps,quakes"):
        with pytest.raises(ValueError, match="one of"):
            lr.parse_segment_sources(bad)


def test_segment_gaps_refuse_a_missing_catalog(tmp_path):
    with pytest.raises(ValueError, match="events"):
        lr.segment_gaps("NYLA", ("events",), events=str(tmp_path / "no.yaml"))
    with pytest.raises(ValueError, match="segment-cut"):
        lr.segment_gaps("NYLA", ("events",), cut="middle")


def test_segment_table_lists_rates_and_refusals():
    class R:
        def __init__(self, refused=None):
            self.window = (2024.0, 2024.5)
            self.accepted = refused is None
            self.refused = refused
            self.rate = (1.0, -2.0, 3.0)
            self.sigma = (0.1, 0.2, 0.3)
            self.flags = ("seasonal:regional",)

    text = lr.segment_table("GRVM", [R(), R("cover=0.5<0.8")])
    assert "-2.0 ±  0.2" in text and "[seasonal:regional]" in text
    assert "refused: cover=0.5<0.8" in text


def test_the_ts_parser_refuses_segments_with_windows():
    from gps_plot.plot_gps_timeseries import build_parser, run

    parser = build_parser()
    args = parser.parse_args(
        ["NYLA", "--ref", "plate", "--rate-segments", "steps",
         "--rate-window", "20200101:20210101"]
    )  # fmt: skip
    with pytest.raises(SystemExit):
        run(args, parser)
