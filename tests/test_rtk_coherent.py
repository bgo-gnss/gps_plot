"""The coherent RTK statistic (T9, gps_plot.rtk_coherent): synthetic only."""

from __future__ import annotations

import datetime as dt
from typing import Any

import numpy as np
import pytest

from gps_plot import rtk_coherent as rc
from gps_plot import rtk_status as rs

W = (15.0, 30.0, 60.0)
WEST = np.array([0.0, -1.0, 0.0])
SW = np.array([-0.6, -0.8, 0.0])


def _fit(**over: Any) -> rc.Coherent:
    """Three template pairs, two on ELDC, one on SUDV; scales 10 mm."""
    base = {
        "windows": W,
        "min_pairs": 2,
        "confirm_z": 1.0,
        "template": {"SENG-ELDC": WEST, "THOB-ELDC": WEST, "THOB-SUDV": SW},
        "pair_scales": {p: np.full(3, 10.0) for p in ("SENG-ELDC", "THOB-ELDC",
                                                       "THOB-SUDV")},
        "area_scales": np.ones(3),
        "ks": (2.0, 3.0, 4.0),
    }  # fmt: skip
    base.update(over)
    return rc.Coherent(**base)


def _offs(vec: Any, steps: int = 1) -> np.ndarray:
    """(steps, windows, 3): the same offset in every window."""
    return np.tile(np.asarray(vec, float), (steps, len(W), 1))


def test_params_defaults_and_checks() -> None:
    got = rc.params(None)
    assert got["windows_minutes"] == [15.0, 30.0, 60.0] and got["min_pairs"] == 2
    assert rc.params({"windows_minutes": [60, 15]})["windows_minutes"] == [15.0, 60.0]
    with pytest.raises(ValueError):
        rc.params({"min_pairs": 0})
    with pytest.raises(ValueError):
        rc.params({"windows_minutes": [15, 30], "template_window": 60})
    with pytest.raises(ValueError):
        rc.params({"template_span": [150, 90]})


def test_template_keeps_agreeing_pairs_only() -> None:
    windows = [5.0, 15.0, 30.0, 60.0]
    n = 100

    def ev(vecs: dict[str, Any]) -> dict[str, np.ndarray]:
        return {p: np.tile(np.asarray(v, float), (n, 4, 1)) for p, v in vecs.items()}

    tune = [
        ("a", 10, ev({"KEEP-BASE": [0, -30, 5], "FLIP-BASE": [0, 20, 0],
                      "ONCE-BASE": [10, 0, 0]})),
        ("b", 10, ev({"KEEP-BASE": [3, -60, 0], "FLIP-BASE": [0, -20, 0]})),
    ]  # fmt: skip
    tpl, rep = rc.learn_template(tune, windows, 60.0, (90.0, 150.0))
    assert set(tpl) == {"KEEP-BASE"}
    assert tpl["KEEP-BASE"] @ np.array([0, -1, 0]) > 0.99
    assert np.isclose(np.linalg.norm(tpl["KEEP-BASE"]), 1.0)
    assert rep["FLIP-BASE"]["kept"] is False and rep["ONCE-BASE"]["events"] == ["a"]


def test_projection_is_one_sided() -> None:
    c = _fit()
    z = rc.pair_z({"SENG-ELDC": _offs([0, 30, 0]), "THOB-ELDC": _offs([0, -30, 0])}, c)
    assert np.all(z["SENG-ELDC"] == 0.0)  # moved against the pattern: nothing
    assert np.allclose(z["THOB-ELDC"], 3.0)


def test_statistic_needs_min_pairs_and_ignores_others() -> None:
    c = _fit()
    one = {"SENG-ELDC": _offs([0, -50, 0])}
    assert np.isnan(rc.statistic(rc.pair_z(one, c), c)).all()
    two = {**one, "THOB-SUDV": _offs([np.nan] * 3), "XXXX-YYYY": _offs([0, -99, 0])}
    assert np.isnan(rc.statistic(rc.pair_z(two, c), c)).all()  # one has a verdict
    three = {**one, "THOB-ELDC": _offs([0, -10, 0])}
    # equal weights: Σ z / √n
    assert np.allclose(rc.statistic(rc.pair_z(three, c), c), (5.0 + 1.0) / np.sqrt(2))


def test_weights_put_the_strong_pair_first() -> None:
    """Matched filter: Σ a·z / √Σ a²; a pair expected to move 4× more (in
    noise units) dominates, a weak pair's noise counts little."""
    c = _fit(weights={"SENG-ELDC": 4.0, "THOB-ELDC": 1.0})
    offs = {"SENG-ELDC": _offs([0, -30, 0]), "THOB-ELDC": _offs([0, 0, 0])}
    got = rc.statistic(rc.pair_z(offs, c), c)
    assert np.allclose(got, 4.0 * 3.0 / np.sqrt(17.0))
    flat = rc.statistic(rc.pair_z(offs, _fit()), _fit())
    assert got[0] > flat[0]


def test_a_base_jump_never_reaches_red() -> None:
    """T7 analogue: ELDC moving east moves every ELDC pair west by the same
    vector; the two ELDC pairs share only their base, so they cannot confirm
    red however large the jump."""
    c = _fit()
    for mm in (50.0, 200.0):
        offs = {p: _offs([0, -mm, 0], 2) for p in ("SENG-ELDC", "THOB-ELDC")}
        lv = rc.area_levels(offs, c)
        assert lv[-1] == 2
        rec = rc.live({p: [None, None] for p in offs}, c)  # no verdicts
        assert rec["level"] is None
    # an intrusion: the SUDV pair moves too, independent of ELDC -> red
    offs = {
        "SENG-ELDC": _offs([0, -50, 0], 2),
        "THOB-ELDC": _offs([0, -50, 0], 2),
        "THOB-SUDV": _offs([-30, -40, 0], 2),
    }
    assert rc.area_levels(offs, c)[-1] == 3


def test_persistence_one_step() -> None:
    c = _fit()
    offs = {p: _offs([0, -50, 0], 3) for p in ("SENG-ELDC", "THOB-ELDC")}
    for p in offs:
        offs[p][0] = 0.0  # quiet, then two steps of motion
    lv = rc.area_levels(offs, c)
    assert list(lv) == [0.0, 0.0, 2.0]


def _rec(vec: Any) -> dict[str, Any]:
    return {"level": 0, "_windows": {w: {"offset": list(vec)} for w in W}}


def test_live_reports_value_pairs_and_the_held_red() -> None:
    c = _fit()
    rec = rc.live(
        {
            "SENG-ELDC": [_rec([0, -50, 0]), _rec([0, -50, 0])],
            "THOB-ELDC": [_rec([0, -50, 0]), _rec([0, -50, 0])],
            "OTHR-BASE": [_rec([0, -50, 0]), _rec([0, -50, 0])],
        },
        c,
    )
    assert rec["level"] == 2 and rec["value"] == round(10.0 / np.sqrt(2), 2)
    assert rec["pairs"] == ["SENG-ELDC", "THOB-ELDC"]
    assert rec["red_unconfirmed"] is True and rec["confirmed"] is False


def test_offsets_of_records() -> None:
    got = rc.offsets_of([None, {"_windows": {15.0: {"offset": [1, 2, 3]}}}], W)
    assert np.isnan(got[0]).all() and list(got[1, 0]) == [1, 2, 3]
    assert np.isnan(got[1, 1:]).all()


def test_load_round_trip() -> None:
    doc = {
        "a": {
            "coherent": {
                "windows_minutes": [15.0, 30.0, 60.0],
                "min_pairs": 2,
                "confirm_z": 1.0,
                "multipliers": [2.0, 3.0, 4.0],
                "area_scales": [1.0, 1.0, 1.0],
                "pairs": {"SENG-ELDC": {"direction": [0, -1, 0],
                                        "scales_mm": [10, 10, 10]}},
            }
        }
    }  # fmt: skip
    c = rc.load(doc, "a")
    assert c is not None and c.ks == (2.0, 3.0, 4.0)
    assert list(c.template["SENG-ELDC"]) == [0, -1, 0]
    assert c.weights == {"SENG-ELDC": 1.0}  # an older file: equal weights
    assert rc.load(doc, "b") is None and rc.load({"a": {}}, "a") is None


def test_fit_ks_meets_the_targets() -> None:
    rng = np.random.default_rng(1)
    series = [np.abs(rng.normal(0, 1, 8640)) for _ in range(3)]  # 3 months
    targets = {"yellow": 1.0, "orange": 1 / 3, "red": 1 / 12}
    ks, found = rc.fit_ks(series, targets)
    assert ks[0] < ks[1] < ks[2]
    for name in ("yellow", "orange", "red"):
        f = found[name]
        assert f["episodes"] <= targets[name] * f["months"]


# --- the live status and the thresholds file ------------------------------

NOW = dt.datetime(2026, 11, 10, 12, 0)


def _series(vec: Any, at: dt.datetime, seed: int) -> Any:
    t = np.arange(
        np.datetime64(NOW - dt.timedelta(hours=8), "m"),
        np.datetime64(NOW, "m") + np.timedelta64(1, "m"),
    ).astype("datetime64[ns]")
    y = np.random.default_rng(seed).normal(0.0, 1.0, (3, t.size))
    y[:, t >= np.datetime64(at, "ns")] += np.asarray(vec, float)[:, None]
    return t, y


def _status(data: dict[str, Any], fit: rc.Coherent | None) -> dict[str, Any]:
    spec = rs.rtk_spec(
        {
            "monitoring": {
                "rtk": {
                    "horizontal_mm": [10, 20, 40],
                    "up_mm": [20, 40, 80],
                    "windows_minutes": [5, 15, 30, 60],
                    "statistic": "coherent",
                }
            }
        }
    )
    assert spec is not None
    if fit is not None:
        spec["coherent_fit"] = fit

    def read(pair: str, start: dt.datetime, end: dt.datetime) -> Any:
        return data[pair]

    return rs.area_status("a", spec, list(data), read, NOW)


def test_area_status_uses_the_coherent_level_and_says_so() -> None:
    at = NOW - dt.timedelta(minutes=40)
    data = {
        "SENG-ELDC": _series([0, -15, 0], at, 1),  # under the pair edges' orange
        "THOB-ELDC": _series([0, -15, 0], at, 2),
        "THOB-SUDV": _series([-9, -12, 0], at, 3),
    }
    pairs = _status(data, None)
    assert pairs["rule"] == "pairs, agree 1 (coherent: not calibrated)"
    assert pairs["level"] == 1  # 15 mm: each pair yellow
    # each pair z ≈ 1.5 along its direction, S ≈ 1.5; two independent pairs
    # above confirm_z, so red stands
    rec = _status(data, _fit(ks=(1.0, 1.2, 1.4)))
    assert rec["rule"] == "coherent" and rec["coherent"]["pairs"] == sorted(data)
    assert rec["level"] == 3 and rec["coherent"]["confirmed"] is True
    assert "red_unconfirmed" not in rec
    # one SUDV pair alone left: too few pairs -> the pairs rule, said so
    alone = _status({"THOB-SUDV": data["THOB-SUDV"]}, _fit(ks=(1.0, 1.2, 1.4)))
    assert alone["rule"] == "pairs, agree 1 (coherent: too few pairs)"


def test_rtk_spec_statistic_and_merge_keeps_the_other_calibration() -> None:
    spec = rs.rtk_spec({"monitoring": {"rtk": {"horizontal_mm": [1, 2, 3]}}})
    assert spec is not None and spec["statistic"] == "pairs"
    with pytest.raises(ValueError):
        rs.rtk_spec({"monitoring": {"rtk": {"horizontal_mm": [1], "statistic": "x"}}})
    old = {"a": {"pairs": {"P-B": {}}, "coherent": {"k": 1}}, "b": {"x": 1}}
    got = rs.merge_area(old, "a", {"pairs": {"Q-B": {}}, "agree": 2})
    assert got["a"]["coherent"] == {"k": 1} and got["a"]["pairs"] == {"Q-B": {}}
    assert got["b"] == {"x": 1}
