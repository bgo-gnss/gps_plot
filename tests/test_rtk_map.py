"""gpsplot rtk-map: a pair's change over the span and the network solve."""

from __future__ import annotations

import numpy as np
import pytest

from gps_plot import rtk_map


def test_pair_change_is_last_edge_minus_first_edge():
    t = np.datetime64("2026-11-02T00:00") + np.arange(73) * np.timedelta64(10, "m")
    neu = np.zeros((3, 73))
    neu[:, 36:] = [[0.010], [0.020], [-0.005]]  # a 10/20/-5 mm step mid-span
    ch, sig = rtk_map.pair_change(t, neu, t[0], t[-1], 60.0)
    assert ch == pytest.approx([10.0, 20.0, -5.0])
    assert (sig >= rtk_map.MIN_SIGMA_MM).all()
    assert rtk_map.pair_change(t[:3], neu[:, :3], t[0], t[-1] + 600, 5.0) is None


def _changes(truth, pairs, noise=0.0):
    return {(r, b): (np.array(truth[r]) - np.array(truth[b]) + noise, np.ones(3))
            for r, b in pairs}  # fmt: skip


def test_network_mean_frame_recovers_the_relative_field():
    truth = {"A": [5.0, 0.0, 1.0], "B": [-5.0, 2.0, 0.0], "C": [0.0, -2.0, -1.0],
             "D": [9.0, 9.0, 9.0]}  # fmt: skip
    pairs = [("A", "B"), ("A", "C"), ("C", "B"), ("D", "E")]  # D-E: a separate group
    truth["E"] = [0.0, 0.0, 0.0]
    sol = rtk_map.solve_network(_changes(truth, pairs))
    assert set(sol) == {"A", "B", "C"}  # the largest connected group only
    mean = np.mean([truth[s] for s in "ABC"], axis=0)
    for s in "ABC":
        assert sol[s][0] == pytest.approx(np.array(truth[s]) - mean, abs=1e-6)
    assert np.mean([sol[s][0] for s in "ABC"], axis=0) == pytest.approx(0.0, abs=1e-6)


def test_anchors_hold_their_stations_at_zero():
    truth = {"A": [3.0, 1.0, 0.0], "V": [0.0, 0.0, 0.0], "K": [0.0, 0.0, 0.0]}
    sol = rtk_map.solve_network(_changes(truth, [("A", "V"), ("A", "K")]), ["V", "K"])
    assert sol["A"][0] == pytest.approx([3.0, 1.0, 0.0])
    assert sol["V"][0] == pytest.approx([0.0, 0.0, 0.0])
