"""--rate-segments auto (Q-T3) and --one-panel (Q-T7)."""

from __future__ import annotations


import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from gps_plot.axes import station_frame
from gps_plot.layers import rate as lr

#: gps_analysis.breaks is on gps_analysis feat/rate-breaks until it merges
needs_breaks = pytest.mark.skipif(
    __import__("importlib").util.find_spec("gps_analysis.breaks") is None,
    reason="gps_analysis.breaks not installed",
)


def series(breaks=(2021.0, 2024.0), rates=(10.0, 20.0, 40.0)):
    rng = np.random.default_rng(1)
    t = np.arange(2018.0, 2026.5, 1 / 365.25)
    y = np.zeros_like(t)
    edges = [t[0], *breaks]
    for i, r in enumerate(rates):
        y += r * np.clip(
            t - edges[i], 0, (edges[i + 1] if i + 1 < len(edges) else 1e9) - edges[i]
        )
    return t, y + rng.normal(0, 1.5, t.size)


def test_parse_auto_combines():
    assert lr.parse_segment_sources("auto") == ("auto",)
    assert lr.parse_segment_sources("events,auto") == ("events", "auto")


@needs_breaks
def test_auto_breaks_on_the_along_component():
    t, y = series()
    yy = np.vstack([y, np.zeros_like(y), np.zeros_like(y)])
    ss = np.ones_like(yy)
    found = lr.auto_breaks("TEST", t, yy, ss, station_frame(90.0))
    assert found.series == "motion"
    assert len(found.breaks) == 2
    assert found.breaks[0] == pytest.approx(2021.0, abs=0.1)
    assert found.breaks[1] == pytest.approx(2024.0, abs=0.1)
    assert all(g >= found.min_bic_gain for g in found.bic_gains)
    gaps = lr.break_gaps(found)
    assert len(gaps) == 2 and gaps[0][0] == gaps[0][1]
    assert "BIC gains" in lr.break_note(found)


@needs_breaks
def test_auto_breaks_projects_horizontals_without_along():
    t, y = series()
    a = np.radians(30.0)  # motion toward N30E: N = cos, E = sin
    yy = np.vstack([y * np.cos(a), y * np.sin(a), np.zeros_like(y)])
    found = lr.auto_breaks("TEST", t, yy, np.ones_like(yy))
    assert found.series.startswith("motion 30.")
    assert len(found.breaks) == 2


@needs_breaks
def test_no_break_and_gain():
    t, y = series(breaks=(), rates=(10.0,))
    yy = np.vstack([y, y * 0, y * 0])
    found = lr.auto_breaks("TEST", t, yy, np.ones_like(yy), station_frame(0.0))
    assert found.breaks == () and "none" in lr.break_note(found)
    big = lr.auto_breaks(
        "TEST", *series()[:1], np.vstack([series()[1]] + [np.zeros(series()[0].size)] * 2),
        np.ones((3, series()[0].size)), station_frame(0.0), min_bic_gain=1e9,
    )  # fmt: skip
    assert big.breaks == () and big.min_bic_gain == 1e9


def test_keep_one_panel():
    from gps_plot.timesmatplt import ONE_PANEL_SIZE, keep_one_panel

    fig, axes = plt.subplots(3, 1)
    fig.suptitle("x")
    twin = axes[0].twinx()
    keep_one_panel(fig, list(axes), quakes=True)
    assert fig.axes == [axes[0], twin]
    assert tuple(fig.get_size_inches()) == pytest.approx(ONE_PANEL_SIZE)
    assert twin.get_position().bounds == pytest.approx(axes[0].get_position().bounds)
    plt.close(fig)


def test_cli_options():
    from gps_plot.plot_gps_timeseries import build_parser

    a = build_parser().parse_args(
        ["VONC", "--rate-segments", "auto", "--break-gain", "40", "--one-panel"]
    )
    assert a.break_gain == 40.0 and a.one_panel and a.rate_segments == "auto"
