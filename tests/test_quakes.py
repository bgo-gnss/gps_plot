"""Earthquake marks: parsing, distance, label and drawing on any figure."""

from __future__ import annotations

import datetime as dt

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from gps_plot.layers.quakes import (
    Quake,
    add_quake_marks,
    distance_km,
    parse_quake,
    quake_label,
)

BB = Quake(dt.datetime(2026, 10, 19), 64.63, -17.53, 5.2, "BB")


def test_parse_quake_with_and_without_name() -> None:
    q = parse_quake("2026-10-19T14:32,64.63,-17.53,5.2,BB")
    assert q == Quake(dt.datetime(2026, 10, 19, 14, 32), 64.63, -17.53, 5.2, "BB")
    assert parse_quake("2026-10-19,64.63,-17.53,5.2").name == ""


@pytest.mark.parametrize(
    "spec", ["2026-10-19,64.6,-17.5", "19/10/2026,64.6,-17.5,5.2", "2026-10-19,95,0,5"]
)
def test_parse_quake_refuses_bad_specs(spec: str) -> None:
    with pytest.raises(ValueError):
        parse_quake(spec)


def test_distance_km_one_degree_of_latitude() -> None:
    assert distance_km(-18.0, 64.0, -18.0, 65.0) == pytest.approx(111.19, abs=0.01)


def test_label_single_point_and_no_point() -> None:
    assert quake_label(BB, {"VONC": (-17.53, 64.63)}) == "BB M5.2, dist 0 km"
    assert quake_label(BB, {}) == "BB M5.2"


def test_label_baseline_ends_same_and_different_distance() -> None:
    near = {"A": (-17.0, 64.63), "B": (-17.001, 64.63)}
    assert quake_label(BB, near).count("km") == 1
    assert "A" not in quake_label(BB, near)
    far = {"A": (-17.0, 64.63), "B": (-16.0, 64.63)}
    assert quake_label(BB, far) == "BB M5.2, dist A 25, B 73 km"


def test_marks_on_every_axis_label_on_top_only() -> None:
    fig, axes = plt.subplots(3, 1, sharex=True)
    for ax in axes:
        ax.plot([dt.datetime(2026, 10, 1), dt.datetime(2026, 10, 21)], [0, 1])
    add_quake_marks(fig, [BB], {"VFLN": (-16.6, 64.9)})
    assert all(len(ax.lines) == 2 for ax in axes)
    assert [len(ax.texts) for ax in axes] == [1, 0, 0]
    assert axes[0].texts[0].get_text().strip().startswith("BB M5.2, dist ")
    plt.close(fig)


def test_quake_outside_the_plotted_range_is_skipped() -> None:
    fig, ax = plt.subplots()
    ax.plot([dt.datetime(2020, 1, 1), dt.datetime(2020, 6, 1)], [0, 1])
    add_quake_marks(fig, [BB], {})
    assert len(ax.lines) == 1 and not ax.texts
    plt.close(fig)
