"""gps_plot.axes / labels: baseline series re-signed (+ = lengthening) and labelled."""

from __future__ import annotations

import numpy as np
import pytest
from geofunc.axes import along_axes, sign_axes

from gps_plot import axes, labels, rtk
from gps_plot.series import Series

NAN = np.nan


def _series(**kw: object) -> Series:
    y = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    base: dict[str, object] = {
        "label": "SEY1-SEYD",
        "t": np.array([0.0, 1.0]),
        "y": y,
        "sigma": y / 10.0,
    }
    base.update(kw)
    return Series(**base)  # type: ignore[arg-type]  # (test kwargs)


def test_orient_flips_only_the_components_that_shorten() -> None:
    ax = sign_axes([-2582.9, 351.0, 32.9])  # SEY1 south, east and above SEYD
    s = axes.orient(_series(meta={"level": np.array([1.0, 2.0, 3.0])}), ax)
    assert np.allclose(s.y, [[-1.0, -2.0], [3.0, 4.0], [5.0, 6.0]])
    assert np.allclose(s.sigma, [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    assert np.allclose(s.meta["level"], [-1.0, 2.0, 3.0])
    assert s.components == ("north", "east", "up")
    assert s.meta["apart"] and s.meta["raw"] == (False, False, False)


def test_orient_carries_overlays_ends_and_aborted() -> None:
    ax = sign_axes([-10.0, -10.0, 10.0])
    flagged = np.array([[NAN, 7.0], [NAN, 8.0], [9.0, NAN]])
    end = _series(label="SEY1")
    s = axes.orient(
        _series(
            outliers=(flagged, np.ones((3, 2))),
            buried=(np.array([[NAN] * 2, [NAN] * 2, [4.0, NAN]]), np.ones((3, 2))),
            aborted=[False, True, False],
            ends=(end, _series(label="SEYD", y=np.zeros((3, 2)))),
        ),
        ax,
    )
    assert s.outliers is not None and s.buried is not None and s.ends is not None
    assert np.allclose(s.outliers[0], -flagged * [[1], [1], [-1]], equal_nan=True)
    # a re-signing keeps the Up-only snow days out of N and E
    assert np.isnan(s.buried[0][:2]).all() and s.buried[0][2, 0] == 4.0
    assert s.aborted == [False, True, False]
    # the ends are re-signed alike: A′ − B′ is still the plotted series
    assert np.allclose(s.ends[0].y - s.ends[1].y, axes.orient(end, ax).y)


def test_orient_marks_mixed_components_aborted_on_rotation() -> None:
    s = axes.orient(_series(aborted=[True, False, False]), along_axes(0.5))
    assert s.aborted == [True, True, False]
    assert s.components == ("along", "across", "up") and not s.meta["apart"]


def test_orient_refuses_a_series_already_oriented() -> None:
    with pytest.raises(ValueError):
        axes.orient(_series(components=("along", "across", "up")), along_axes(0.0))


def test_pair_axes_without_coordinates_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    import geo_dataread.station_catalog as sc

    monkeypatch.setattr(sc, "apriori_baseline", lambda a, b, path=None: None)
    assert axes.pair_axes("AAAA-BBBB") is None
    monkeypatch.setattr(
        sc, "apriori_baseline", lambda a, b, path=None: np.array([-5.0, 5.0, 0.2])
    )
    ax = axes.pair_axes("AAAA-BBBB")
    assert ax is not None and ax.raw == (False, False, True)


def test_axis_labels_in_both_languages() -> None:
    neu = ("north", "east", "up")
    assert labels.axis_labels(neu, "en", "mm") == ("North [mm]", "East [mm]", "Up [mm]")
    assert labels.axis_labels(
        neu, "is", "cm", apart=True, raw=(False, False, True)
    ) == (
        "Norður [cm] (+ fjær)",
        "Austur [cm] (+ fjær)",
        "Upp [cm] (A − B)",
    )
    assert labels.component("along", "is") == "Lengd"
    assert labels.component("across", "en") == "Across"
    assert labels.component("xyz", "en") == "xyz"


def test_orienter_signs_each_pair_and_reports_the_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    found = {
        "SEY1-SEYD": sign_axes([-2582.9, 351.0, 32.9]),
        "SEY2-SEYD": sign_axes([-2000.0, 300.0, 0.4]),
    }
    monkeypatch.setattr(axes, "pair_axes", lambda pair: found.get(pair))
    o = rtk.Orienter("apart")
    s1 = o(_series(meta={"pair": "SEY1-SEYD"}))
    assert np.allclose(s1.y[0], [-1.0, -2.0])
    none = _series(label="AAAA-BBBB")
    assert o(none) is none  # no coordinates: left as A − B
    o(_series(label="SEY2-SEYD"))
    apart, raw = o.summary()
    assert apart and raw == (False, False, False)  # only one pair has Up raw
    keep = rtk.Orienter("raw")
    plain = _series()
    assert keep(plain) is plain and keep.summary() == (False, (False, False, False))


def test_sign_option_defaults_to_apart() -> None:
    class _Config:
        def get(self, *a: object, **k: object) -> None:
            return None

    parser = rtk.build_parser(config=_Config())
    assert parser.parse_args(["SEY1-SEYD"]).sign == "apart"
    assert parser.parse_args(["SEY1-SEYD", "--sign", "raw"]).sign == "raw"
