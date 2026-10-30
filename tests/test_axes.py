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
        "SEY1-SEYD": np.array([-2582.9, 351.0, 32.9]),
        "SEY2-SEYD": np.array([-2000.0, 300.0, 0.4]),
    }
    monkeypatch.setattr(axes, "apriori", lambda pair: found.get(pair))
    o = axes.Orienter("apart")
    s1 = o(_series(meta={"pair": "SEY1-SEYD"}))
    assert np.allclose(s1.y[0], [-1.0, -2.0])
    none = _series(label="AAAA-BBBB")
    assert o(none) is none  # no coordinates: left as A − B
    assert not o.summary()[0]  # a pair left A - B: no "+ apart" on the axes
    o = axes.Orienter("apart")
    o(_series(meta={"pair": "SEY1-SEYD"}))
    o(_series(label="SEY2-SEYD"))
    apart, raw = o.summary()
    assert apart and raw == (False, False, False)  # only one pair has Up raw
    station = _series(label="HOFN")
    assert o(station) is station  # a single station is never signed
    keep = axes.Orienter("raw")
    plain = _series()
    assert keep(plain) is plain and keep.summary() == (False, (False, False, False))


def test_orient_carries_the_correlations() -> None:
    """Re-signing flips the sign of ρ for each pair it splits; σ unchanged."""
    corr = np.array([[0.2, 0.2], [0.4, 0.4], [-0.1, -0.1]])  # NE, NU, EU
    s = axes.orient(_series(corr=corr), sign_axes([-2582.9, 351.0, 32.9]))
    np.testing.assert_allclose(s.corr[:, 0], [-0.2, -0.4, -0.1])
    np.testing.assert_allclose(s.sigma, _series().sigma)
    # a rotation mixes: σ_along from the full covariance, not quadrature
    ax = along_axes(np.pi / 4)  # radians
    s2 = axes.orient(_series(corr=np.array([[0.9], [0.0], [0.0]]) * [1, 1]), ax)
    sn, se = 0.1, 0.3  # first epoch's σ_N, σ_E
    want = np.sqrt((sn**2 + se**2 + 2 * 0.9 * sn * se) / 2)
    assert s2.sigma[0, 0] == pytest.approx(want)
    assert axes.orient(_series(), ax).corr is None  # unknown stays unknown


def test_orient_refuses_to_sign_twice() -> None:
    ax = sign_axes([-2582.9, 351.0, 32.9])
    with pytest.raises(ValueError):
        axes.orient(axes.orient(_series(), ax), ax)


def test_sign_pair_and_label_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    found = {"SEY1-SEYD": sign_axes([-2582.9, 351.0, 0.3])}
    monkeypatch.setattr(axes, "pair_axes", lambda pair: found.get(pair))
    s = axes.sign_pair("SEY1-SEYD", _series())
    assert np.allclose(s.y[0], [-1.0, -2.0]) and s.meta["raw"] == (False, False, True)
    assert axes.sign_pair("SEY1-SEYD", _series(), "raw").meta == {}
    station = _series(label="HOFN")
    assert axes.sign_pair("HOFN", station) is station
    with pytest.warns(UserWarning, match="no a-priori"):
        axes.sign_pair("AAAA-BBBB", _series())
    assert axes.label_flags([s]) == (True, (False, False, True))
    assert axes.label_flags([s, station]) == (False, (False, False, False))


def test_fetch_signs_a_baseline_once(monkeypatch: pytest.MonkeyPatch) -> None:
    from gps_plot.sources import gps as source

    def station(sta: str, req: object) -> Series:
        y = np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]) * (1 if sta == "SEY1" else 0)
        return _series(label=sta, y=y, sigma=np.ones((3, 2)))

    monkeypatch.setattr(source, "fetch_station", station)
    monkeypatch.setattr(
        axes, "pair_axes", lambda pair: sign_axes([-2582.9, 351.0, 32.9])
    )
    req = source.SeriesRequest(
        fstart=None, fend=None, ref="plate", Dir=None, tType="POS", uncert=10,
        view="raw",
    )  # fmt: skip
    s = source.fetch("SEY1-SEYD", req)
    assert np.allclose(s.y[:, -1], [-1.0, 1.0, 1.0]) and s.meta["apart"]
    assert s.ends is not None and np.allclose(s.ends[0].y[:, -1], [-1.0, 1.0, 1.0])
    raw = source.fetch("SEY1-SEYD", req, source.BaselineOptions(sign="raw"))
    assert np.allclose(raw.y[:, -1], [1.0, 1.0, 1.0]) and "apart" not in raw.meta


def test_sign_option_defaults_to_apart() -> None:
    class _Config:
        def get(self, *a: object, **k: object) -> None:
            return None

    parser = rtk.build_parser(config=_Config())
    assert parser.parse_args(["SEY1-SEYD"]).sign == "apart"
    assert parser.parse_args(["SEY1-SEYD", "--sign", "raw"]).sign == "raw"


def test_daily_sign_options() -> None:
    from gps_plot import plot_gps_timeseries as driver
    from gps_plot import stack

    class _Config:
        def getPostProcessDir(self, option: str) -> str:
            return f"/cfg/{option}"

        def getStationInfo(self) -> list[str]:
            return ["HOFN"]

        def get(self, *a: object, **k: object) -> None:
            return None

    ts = driver.build_parser(config=_Config())
    assert ts.parse_args(["SEY1-SEYD"]).baseline_sign == "apart"
    assert ts.parse_args(["SEY1-SEYD", "--sign", "raw"]).baseline_sign == "raw"
    st = stack.build_parser(config=_Config())
    assert st.parse_args(["SEY1-SEYD"]).sign == "apart"


B0 = np.array([-2582.9, 351.0, 32.9])  # SEY1-SEYD: south, east and above


def _rtk(level_m: object, **kw: object) -> Series:
    return _series(meta={"pair": "SEY1-SEYD", "source": "rtk",
                         "level": np.asarray(level_m) * 1000.0}, **kw)  # fmt: skip


def test_pair_frame_along_keeps_up_signed() -> None:
    ax = axes.pair_frame(B0, "along")
    h = B0[:2] / np.hypot(*B0[:2])
    np.testing.assert_allclose(ax.matrix[0], [h[0], h[1], 0.0])
    np.testing.assert_allclose(ax.matrix[1], [-h[1], h[0], 0.0])  # clockwise
    np.testing.assert_allclose(ax.matrix[2], [0.0, 0.0, 1.0])
    down = axes.pair_frame(B0 * [1, 1, -1], "along")
    np.testing.assert_allclose(down.matrix[2], [0.0, 0.0, -1.0])  # + = apart
    assert axes.pair_frame(B0 * [1, 1, 0.01], "along").raw == (False, False, True)
    three = axes.pair_frame(B0, "3d")
    np.testing.assert_allclose(three.matrix[0], B0 / np.linalg.norm(B0))
    assert three.keys == ("along", "across", "normal")
    with pytest.raises(ValueError):
        axes.pair_frame(B0, "sideways")


def test_along_lengthening_is_positive() -> None:
    """The rover moving straight away from the reference plots + along."""
    ax = axes.pair_frame(B0, "along")
    step = 10.0 * B0 / np.linalg.norm(B0)  # mm, along the 3-D line
    y = np.stack([np.zeros(3), step], axis=1)
    s = axes.orient(_series(y=y, sigma=np.ones((3, 2))), ax)
    assert s.y[0, 1] > 9.9 and abs(s.y[1, 1]) < 1e-9


def test_orienter_along_checks_the_direction(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(axes, "apriori", lambda pair: B0)
    near = axes.Orienter(axes="along")
    turned = np.array([[np.cos(0.2e-2), -np.sin(0.2e-2)],
                       [np.sin(0.2e-2), np.cos(0.2e-2)]]) @ B0[:2]  # 0.11°  # fmt: skip
    near(_rtk([*turned, B0[2]]))
    np.testing.assert_allclose(
        near.found["SEY1-SEYD"].matrix, axes.pair_frame(B0, "along").matrix
    )
    far = axes.Orienter(axes="along")
    data = np.array([-2582.9, 400.0, 32.9])  # ~1.1° off
    far(_rtk(data))
    np.testing.assert_allclose(
        far.found["SEY1-SEYD"].matrix, axes.pair_frame(data, "along").matrix
    )
    assert "off the RTK vector" in capsys.readouterr().err
    # the daily series that follows gets the same axes
    d = far(_series(meta={"pair": "SEY1-SEYD"}))
    assert d.components == ("along", "across", "up")


def test_orienter_along_without_a_direction(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(axes, "apriori", lambda pair: None)
    with pytest.raises(ValueError, match="no direction"):
        axes.Orienter(axes="along")(_series(meta={"pair": "SEY1-SEYD"}))
    s = _series(meta={"pair": "SEY1-SEYD"})
    assert axes.Orienter()(s) is s  # neu: left as A - B
    # an RTK series is its own direction
    o = axes.Orienter(axes="along")
    assert o(_rtk(B0)).components == ("along", "across", "up")


def test_orienter_labels(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(axes, "apriori", lambda pair: B0)
    o = axes.Orienter(axes="along")
    o(_series(meta={"pair": "SEY1-SEYD"}))
    assert o.labels("en", "mm") == ("Along [mm] (+ apart)", "Across [mm]",
                                    "Up [mm] (+ apart)")  # fmt: skip
    assert o.labels("is", "cm")[:2] == ("Lengd [cm] (+ fjær)", "Þvert [cm]")
    n = axes.Orienter()
    n(_series(meta={"pair": "SEY1-SEYD"}))
    assert n.labels("en", "mm")[0] == "North [mm] (+ apart)"
    assert n.keys == ("north", "east", "up")


def test_axes_option() -> None:
    class _Config:
        def get(self, *a: object, **k: object) -> None:
            return None

    parser = rtk.build_parser(config=_Config())
    assert parser.parse_args(["SEY1-SEYD"]).axes == "neu"
    assert parser.parse_args(["SEY1-SEYD", "--axes", "3d"]).axes == "3d"
