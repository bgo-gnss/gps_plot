"""gpsplot seasonal: the fold, the harmonics, the regional prediction."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from gps_plot import seasonal as sv

NAMES = [
    "offset",
    "rate",
    "cos_annual",
    "sin_annual",
    "cos_semiannual",
    "sin_semiannual",
]


def _record(coeffs: np.ndarray, **extra: object) -> dict:
    n = len(NAMES)
    iu = np.triu_indices(n)
    cov = np.diag(np.full(n, 0.01))
    comps = [
        {"params": [0.0, 0.0, *map(float, coeffs[k])], "cov_upper": cov[iu].tolist()}
        for k in range(3)
    ]
    rec = {"model": "lineperiodic", "param_names": NAMES, "components": comps,
           "window": [2010.0, 2016.0]}  # fmt: skip
    rec.update(extra)
    return rec


def test_fold_is_the_phase_of_the_basis():
    t = np.array([2010.25, 2011.25, 2017.999])
    assert np.allclose(sv.fold(t), [0.25, 0.25, 0.999])
    c = np.array([[1.0, 2.0, 0.5, -0.3]] * 3)
    assert np.allclose(sv.seasonal_curve(c, t), sv.seasonal_curve(c, sv.fold(t)))


def test_harmonics_recover_amplitude_and_peak():
    # 3 mm annual peaking at 0.6 of the year, 1 mm semiannual
    a, b = 3 * np.cos(2 * np.pi * 0.6), 3 * np.sin(2 * np.pi * 0.6)
    h = sv.harmonics([a, b, 1.0, 0.0])
    assert h.annual == pytest.approx(3.0)
    assert h.peak == pytest.approx(0.6)
    assert h.semiannual == pytest.approx(1.0)
    # the annual term alone is largest at its peak
    x = np.linspace(0, 1, 10001)
    annual = sv.seasonal_curve(np.array([[a, b, 0, 0]] * 3), x)[0]
    assert x[np.argmax(annual)] == pytest.approx(0.6, abs=1e-3)


def test_peak_label():
    assert sv.peak_label(0.0) == "1 Jan"
    assert sv.peak_label(31 / 365.25) == "1 Feb"


def test_binned_profile_recovers_a_seasonal_in_noise():
    rng = np.random.default_rng(1)
    t = 2010 + np.arange(0, 8 * 365) / 365.25
    c = np.array([2.0, -1.0, 0.5, 0.2])
    y = sv.seasonal_curve(np.array([c] * 3), t)[0] + rng.normal(0, 2, t.size)
    cx, med, err = sv.binned_profile(sv.fold(t), y, bin_days=15)
    truth = sv.seasonal_curve(np.array([c] * 3), cx)[0]
    assert np.all(np.isfinite(med))
    assert np.max(np.abs(med - truth)) < 5 * np.max(err)


def test_binned_profile_leaves_thin_bins_empty():
    cx, med, err = sv.binned_profile([0.01, 0.02], [1.0, 2.0], bin_days=30)
    assert np.isnan(med).all() and np.isnan(err).all()


def test_rms_difference_matches_the_curves():
    rng = np.random.default_rng(2)
    c1, c2 = rng.normal(size=(3, 4)), rng.normal(size=(3, 4))
    x = np.linspace(0, 1, 20000, endpoint=False)
    d = sv.seasonal_curve(c1, x) - sv.seasonal_curve(c2, x)
    assert np.allclose(
        sv.rms_difference(c1, c2), np.sqrt(np.mean(d**2, axis=1)), rtol=1e-6
    )


def test_curve_sigma_of_a_diagonal_covariance():
    # σ² on all four terms: b·b = cos² + sin² + cos² + sin² = 2, so σ(x) = σ√2
    cov = np.array([np.eye(4) * 0.25] * 3)
    s = sv.curve_sigma(cov, np.linspace(0, 1, 7))
    assert np.allclose(s, 0.5 * np.sqrt(2))


def test_periodic_cov_is_taken_by_name():
    rec = _record(np.ones((3, 4)))
    cov = sv.periodic_cov(rec)
    assert cov is not None and cov.shape == (3, 4, 4)
    assert np.allclose(cov[0], np.eye(4) * 0.01)
    rec["components"][1].pop("cov_upper")
    assert sv.periodic_cov(rec) is None


def test_align_datum_puts_the_residual_on_the_curve():
    t = 2010 + np.arange(0, 3 * 365) / 365.25
    c = np.array([[1.0, 0.0, 0.0, 0.0]] * 3)
    y = sv.seasonal_curve(c, t) + np.array([[5.0], [-2.0], [10.0]])
    y[0, 3] = np.nan
    out = sv.align_datum(t, y, c)
    want = sv.seasonal_curve(c, t)
    want[0, 3] = np.nan
    assert np.allclose(out, want, equal_nan=True)


POS = {"AAAA": (-22.0, 64.0), "BBBB": (-22.2, 64.0), "CCCC": (-22.0, 64.2),
       "DDDD": (-22.1, 63.9), "TGT": (-22.1, 64.05)}  # fmt: skip


def _coords(sta: str) -> tuple[float, float]:
    return POS[sta]


def _doc() -> dict:
    rng = np.random.default_rng(3)
    stations = {
        s: _record(rng.normal(size=(3, 4))) for s in ("AAAA", "BBBB", "CCCC", "DDDD")
    }
    return {"stations": stations, "regional_seasonal": {"sigma_pred": [0.5, 0.5, 1.5]}}


def test_regional_prediction_never_uses_the_station_itself():
    doc = _doc()
    got, donors = sv.regional_prediction("AAAA", doc["stations"], _coords)
    assert "AAAA" in donors  # it is a donor for others...
    assert got.refused is None
    assert "AAAA" not in got.weights  # ...but not for itself
    own = donors["AAAA"][2]
    assert not np.allclose(got.coeffs, own)


def test_a_regional_station_shows_its_stored_curve_and_no_data(monkeypatch):
    doc = _doc()
    stored = np.full((3, 4), 0.7)
    doc["stations"]["TGT"] = {
        "model": "periodic",
        "param_names": ["cos_annual", "sin_annual", "cos_semiannual", "sin_semiannual"],
        "seasonal_only": True,
        "components": [{"params": stored[k].tolist()} for k in range(3)],
    }

    def no_read(*a, **k):  # a seasonal-only record has no secular: never read
        raise AssertionError("read the series of a seasonal-only station")

    monkeypatch.setattr(sv, "_read_residual", no_read)
    view = sv.station_view(
        "TGT", doc, frame="itrf2014", pos_id="imo.24hr_itrf14", pos_dir=".",
        start=None, end=None, clean=True, uncert=10, coords=_coords,
    )  # fmt: skip
    assert view.seasonal_only and view.data is None and view.own is None
    assert np.allclose(view.regional, stored)
    assert np.allclose(view.regional_sigma, [0.5, 0.5, 1.5])
    fig = sv.seasonal_figure(view, n_donors=2)
    assert len(fig.axes) == 3


def test_an_own_station_folds_its_residual(monkeypatch):
    doc = _doc()
    t = 2010 + np.arange(0, 4 * 365) / 365.25
    from geo_dataread.regional_seasonal import periodic_coeffs

    c = periodic_coeffs(doc["stations"]["BBBB"])
    y = sv.seasonal_curve(c, t) + 3.0
    seen = {}

    def fake_read(sta, start, end, **kw):
        seen.update(start=start, end=end)
        return t, y.copy()

    monkeypatch.setattr(sv, "_read_residual", fake_read)
    view = sv.station_view(
        "BBBB", doc, frame="itrf2014", pos_id="imo.24hr_itrf14", pos_dir=".",
        start=None, end=None, clean=True, uncert=10, coords=_coords,
    )  # fmt: skip
    assert seen == {"start": 2010.0, "end": 2016.0}  # the record's window
    assert view.data is not None
    assert np.allclose(view.data[1], sv.seasonal_curve(c, t))  # datum aligned
    assert view.regional is not None and "BBBB" not in view.weights
    fig = sv.seasonal_figure(view)
    assert "secular + steps removed" in fig._suptitle.get_text()


@pytest.mark.parametrize(
    ("change", "match"),
    [(lambda d: d["stations"].pop("AAAA"), "no detrend record"),
     (lambda d: d["stations"]["AAAA"].update(param_names=["offset", "rate"]),
      "no periodic terms")],
)  # fmt: skip
def test_station_view_refuses(change, match):
    doc = _doc()
    change(doc)
    with pytest.raises(ValueError, match=match):
        sv.station_view(
            "AAAA", doc, frame="itrf2014", pos_id="imo.24hr_itrf14", pos_dir=".",
            start=None, end=None, clean=True, uncert=10, coords=_coords,
        )  # fmt: skip


def test_seasonal_is_a_gpsplot_command():
    from gps_plot.cli import find_command

    cmd = find_command("seasonal")
    assert cmd is not None and cmd.target == "gps_plot.cli.seasonal:run"
