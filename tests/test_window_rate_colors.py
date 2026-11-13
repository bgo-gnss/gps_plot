"""Display-only window styles: no change to rate estimates or spans."""

from types import SimpleNamespace
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from gps_plot.layers.rate import add_rate_lines
from gps_plot.timesmatplt import addEvent


def test_rate_lines_use_requested_colors_only_inside_their_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from geo_dataread import velocity

    calls: list[tuple[str, float, float, object]] = []

    def estimate(sta: str, start: float, end: float, settings: Any) -> SimpleNamespace:
        calls.append((sta, start, end, settings))
        return SimpleNamespace(
            accepted=True, rate=(2.0, 3.0, 4.0), sigma=(0.1, 0.2, 0.3)
        )

    monkeypatch.setattr(velocity, "estimate_station_velocity", estimate)
    fig, axes = plt.subplots(3)
    t = np.linspace(2015.0, 2026.0, 500)
    y = np.array([2 * t, 3 * t, 4 * t])
    windows = [(2018.2, 2023.2), (2025.0, 2026.0), (2025.75, 2026.0)]
    settings = object()
    records = add_rate_lines(
        fig,
        "KISA",
        t,
        y,
        np.ones_like(y),
        windows,
        settings,
        colors=("black", "blue", "red"),
    )
    assert len(records) == 3
    assert calls == [("KISA", a, b, settings) for a, b in windows]
    for ax in axes:
        assert [line.get_color() for line in ax.lines] == ["black", "blue", "red"]
        for line, (start, end) in zip(ax.lines, windows, strict=True):
            dates = line.get_xdata()
            assert (
                start
                <= dates[0].year + (dates[0].timetuple().tm_yday - 1) / 365.25
                < end
            )
            assert dates[0] <= dates[1]
    plt.close(fig)


def test_labelled_display_markers_do_not_change_limits_or_repeat_the_label() -> None:
    fig, axes = plt.subplots(3)
    for ax in axes:
        ax.set_ylim(-10, 20)
    addEvent(
        {
            "20180301": ["gray", "KISA approx. March 2018"],
            "20230301": ["gray", "KISA approx. March 2023"],
        },
        fig,
    )
    assert [len(ax.lines) for ax in axes] == [2, 2, 2]
    assert [len(ax.texts) for ax in axes] == [2, 0, 0]
    assert all(ax.get_ylim() == pytest.approx((-10, 20)) for ax in axes)
    plt.close(fig)
