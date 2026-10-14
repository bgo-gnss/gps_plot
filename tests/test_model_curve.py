"""``gpsplot ts --model``: the stored-model line, its source and its layer.

The acceptance identity (design §6 step 3): at the data epochs,
``plotted data − line`` equals ``full plate series − stored model`` in every
view — full or windowed read, raw or cleaned, curve (plate) or residual
(detrend), with or without ``--remove-steps``. The line is built through the
same ``trajectory_views`` that removes terms from the data, and shifted by
``window_datum_shift`` into the window's datum, so this holds by
construction; the test pins it on the deployed SELF record, whose itrf14
record carries a step (2008.41).
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from matplotlib.figure import Figure

from gps_plot.layers.model import (
    add_model_curve,
    fit_window_mask,
    model_note,
    off_scale_note,
)
from gps_plot.series import Series


# ---------------------------------------------------------------------------
# layer (no data)
# ---------------------------------------------------------------------------


def test_fit_window_mask():
    t = np.array([2000.0, 2001.0, 2002.0, 2003.0])
    np.testing.assert_array_equal(
        fit_window_mask(t, [2001.0, 2002.0]), [False, True, True, False]
    )
    assert fit_window_mask(t, None).all()


def test_off_scale_note_only_when_the_whole_line_is_outside():
    assert off_scale_note([5.0, 6.0], (0.0, 10.0)) is None
    assert off_scale_note([5.0, 60.0], (0.0, 10.0)) is None
    assert off_scale_note([-500.0, -480.0], (0.0, 10.0)) == (
        "model line off scale, at -490 mm"
    )
    assert off_scale_note([np.nan], (0.0, 10.0)) is None


def test_model_note_names_the_fit_window():
    s = Series("X model", np.array([2000.0]), np.zeros((3, 1)), None)
    assert model_note(s) == "line: stored model"
    s2 = Series("X model", s.t, s.y, None, meta={"window": [2015.4836, 2019.9651]})
    assert "2015.48 to 2019.97" in model_note(s2)


def test_add_model_curve_keeps_the_data_limits():
    fig = Figure()
    axes = [fig.add_subplot(3, 1, i + 1) for i in range(3)]
    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(-1, 1)
    t = np.linspace(2020.0, 2021.0, 20)
    model = Series(
        "X model",
        t,
        np.vstack([t * 0, t * 0 + 50.0, t * 0]),
        None,
        meta={"window": [2020.0, 2020.5]},
    )
    add_model_curve(fig, model)
    for ax in axes:
        assert ax.get_ylim() == (-1.0, 1.0)
        assert len(ax.lines) == 2  # dashed (all) + solid (fit window)
    # the East line sits at 50 mm, off the ±1 scale: said, not hidden
    assert any("off scale" in t.get_text() for t in axes[1].texts)
    assert not axes[0].texts


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------


def test_model_variants_both_is_curve_then_residual():
    from gps_plot.plot_gps_timeseries import _model_variants

    out = _model_variants([{"special": "90d"}, {"special": "year"}], "both")
    assert [(v["special"], v["ref"], v["model"]) for v in out] == [
        ("90d", "plate", "curve"),
        ("90d", "detrend", "residual"),
        ("year", "plate", "curve"),
        ("year", "detrend", "residual"),
    ]
    assert _model_variants([{}], "curve") == [{"ref": "plate", "model": "curve"}]


@pytest.mark.parametrize(
    "argv, message",
    [
        (["HOFN", "--model", "curve", "--ref", "ref_frame"], "leave --ref out"),
        (["HOFN", "--model", "both", "--ref", "plate"], "leave --ref out"),
        (["HOFN", "--model", "curve", "--ref", "all"], "leave --ref out"),
        (["SENG-SKSH", "--model", "curve"], "baselines"),
        (["HOFN", "--model", "curve", "--session", "both"], "daily series only"),
        (["HOFN", "--model", "residual", "--session", "8h"], "daily series only"),
        (
            ["HOFN", "--model", "curve", "--detrend-terms", "periodic"],
            "--detrend-terms needs --ref detrend",
        ),
    ],
)
def test_model_refusals(argv, message, capsys):
    from gps_plot import plot_gps_timeseries as driver

    parser = driver.build_parser(config=None)
    with pytest.raises(SystemExit):
        driver.run(parser.parse_args(argv + ["--save", "png"]), parser, None)
    assert message in capsys.readouterr().err


def test_plot_time_refuses_a_model_over_the_reference_frame():
    from gps_plot.timesmatplt import plotTime

    with pytest.raises(ValueError, match="--ref plate or detrend"):
        plotTime("HOFN", ref="itrf2008", model="curve", save="png")
    with pytest.raises(ValueError, match="baseline"):
        plotTime("SENG-SKSH", ref="plate", model="curve", save="png")
    with pytest.raises(ValueError, match="daily series only"):
        plotTime(
            "HOFN",
            ref="plate",
            model="curve",
            save="png",
            tType="POS",
            pos_id="imo.24hr_itrf20",
            overlay_pos_id="imo.08hr_itrf20",
        )
    with pytest.raises(ValueError, match="daily series only"):
        plotTime(
            "HOFN",
            ref="plate",
            model="curve",
            save="png",
            tType="POS",
            pos_id="imo.08hr_itrf20",
        )


# ---------------------------------------------------------------------------
# the acceptance identity, on the deployed SELF record
# ---------------------------------------------------------------------------


def _deployed():
    try:
        from geo_dataread import gps_views as gv
        from geo_dataread.pos import pos_product

        from gps_plot import plot_gps_timeseries as driver

        pos_dir = driver._config_dir(driver._load_config(), "pos_dir")
        pid = pos_product("itrf2014", "24h")
        doc = gv.read_detrend_params(gv.default_params_path(pid))
        record, _ = gv.station_detrend_record(doc, "SELF")
    except Exception:
        return None
    if record is None or not record.get("step_epochs"):
        return None
    return pos_dir, pid, record


DEPLOYED = _deployed()
requires_deployed = pytest.mark.skipif(
    DEPLOYED is None, reason="needs the deployed SELF itrf14 .pos + record"
)


@requires_deployed
@pytest.mark.parametrize("window", [(None, None), (2007.0, 2010.0), (2024.0, None)])
@pytest.mark.parametrize(
    "ref, view, remove",
    [
        ("plate", "raw", None),
        ("detrend", "raw", None),
        ("detrend", "cleaned", None),
        ("detrend", "raw", "all"),
        ("plate", "cleaned", "all"),
    ],
)
def test_data_minus_line_is_the_model_residual(window, ref, view, remove):
    import geo_dataread.gps_read as gpsr
    from gps_analysis import evaluate_record
    from gps_parser.outlier_catalogs import STEP_KINDS

    from gps_plot.sources import gps as source

    pos_dir, pid, record = DEPLOYED
    req = source.SeriesRequest(
        fstart=window[0],
        fend=window[1],
        ref=ref,
        Dir=pos_dir,
        tType="POS",
        uncert=15,
        view=view,
        remove_kinds=list(STEP_KINDS) if remove else None,
        pos_id=pid,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plotted = source.fetch_station("SELF", req)
        line = source.model_values("SELF", req, plotted.t, record)
    t_full, d_full, _, _ = gpsr.getData(
        "SELF", ref="plate", Dir=pos_dir, tType="POS", uncert=15, pos_id=pid
    )
    _, i, j = np.intersect1d(t_full, plotted.t, return_indices=True)
    want = d_full[:, i] - np.asarray(evaluate_record(record, t_full[i]))
    got = (plotted.y - line)[:, j]
    ok = np.isfinite(got)
    assert ok.sum() > 100
    np.testing.assert_allclose(got[ok], want[ok], atol=1e-6)


# ---------------------------------------------------------------------------
# --model-terms: the line's terms, independent of the data's
# ---------------------------------------------------------------------------


def test_model_terms_parse_in_canonical_order():
    from gps_plot.plot_gps_timeseries import _model_terms

    assert _model_terms(None) is None
    assert _model_terms("periodic, linear") == ("linear", "periodic")
    assert _model_terms("steps,transient") == ("steps", "transient")
    with pytest.raises(ValueError, match="seasonal"):
        _model_terms("seasonal")


def test_model_terms_needs_model(capsys):
    from gps_plot import plot_gps_timeseries as driver

    parser = driver.build_parser(config=None)
    with pytest.raises(SystemExit):
        driver.run(
            parser.parse_args(["HOFN", "--model-terms", "periodic", "--save", "png"]),
            parser,
            None,
        )
    assert "--model-terms needs --model" in capsys.readouterr().err


def test_model_note_names_the_terms():
    s = Series("X model", np.array([2000.0]), np.zeros((3, 1)), None,
               meta={"terms": ("linear", "periodic")})  # fmt: skip
    assert model_note(s) == "line: stored model (linear + periodic)"


@requires_deployed
@pytest.mark.parametrize("window", [(None, None), (2007.0, 2010.0)])
def test_model_terms_line_is_those_groups_whatever_the_data(window):
    """All four groups = the full model; one group = that group alone, in
    the window's datum — regardless of what the view removed from the data."""
    from geo_dataread.gps_views import window_datum_shift
    from gps_analysis import evaluate_record

    from gps_plot.sources import gps as source

    pos_dir, pid, record = DEPLOYED
    req = source.SeriesRequest(
        fstart=window[0], fend=window[1], ref="detrend", Dir=pos_dir,
        tType="POS", uncert=15, view="raw", detrend_terms=("periodic",),
        pos_id=pid,
    )  # fmt: skip
    t = np.linspace(2006.0, 2011.0, 400)
    shift = window_datum_shift(
        "SELF", fstart=window[0], fend=window[1], Dir=pos_dir, tType="POS",
        pos_id=pid,
    )  # fmt: skip
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        every = source.model_values(
            "SELF", req, t, record, ("linear", "periodic", "steps", "transient")
        )
        periodic = source.model_values("SELF", req, t, record, ("periodic",))
    np.testing.assert_allclose(
        every, np.asarray(evaluate_record(record, t)) - shift[:, None], atol=1e-9
    )
    np.testing.assert_allclose(
        periodic,
        np.asarray(evaluate_record(record, t, groups=("periodic",))) - shift[:, None],
        atol=1e-9,
    )


# ---------------------------------------------------------------------------
# detrended stations re-zeroed at their start (ts --ref detrend, stack)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "spec, changes, applied",
    [
        ("SENG-SKSH", {}, True),
        ("SELF", {"ref_position": "first"}, True),
        ("SELF", {}, False),
    ],
)
def test_rezero_does_not_apply(spec, changes, applied):
    from gps_plot.sources import gps as source

    req = source.SeriesRequest(
        fstart=2024.0, fend=None, ref="detrend", Dir=None, tType="POS",
        uncert=10, view="raw", **changes,
    )  # fmt: skip
    s = Series(spec, np.zeros(1), np.ones((3, 1)), np.ones((3, 1)),
               detrend_applied=applied)  # fmt: skip
    assert source.detrended_shift(spec, req, s) is None
    assert source.rezero_detrended(spec, req, s) is s


def test_rezero_uses_the_first_five_finite_samples_per_component():
    from gps_analysis.baseline import estimate_offset

    from gps_plot.sources import gps as source

    t = np.linspace(2024.0, 2024.1, 12)
    y = np.vstack([np.arange(12.0) + 7, np.arange(12.0) * 2 - 3, np.full(12, 4.0)])
    y[0, :2] = np.nan  # a masked (cleaned) start
    sig = np.ones((3, 12)) + np.arange(12) / 10
    req = source.SeriesRequest(
        fstart=2024.0, fend=None, ref="detrend", Dir=None, tType="POS",
        uncert=10, view="cleaned",
    )  # fmt: skip
    s = Series("SELF", t, y, sig, outliers=(y.copy(), sig), detrend_applied=True)
    z = source.rezero_detrended("SELF", req, s)
    want0 = float(estimate_offset(t[2:7], y[0, 2:7], sig[0, 2:7]))
    np.testing.assert_allclose(z.y[0], y[0] - want0)
    np.testing.assert_allclose(z.y[2], 0.0)
    np.testing.assert_allclose(z.outliers[0], z.y)  # moved with the data
    np.testing.assert_allclose(z.meta["zero_shift"][0], -want0)


@requires_deployed
@pytest.mark.parametrize("window", [(None, None), (2007.0, 2010.0), (2024.0, None)])
@pytest.mark.parametrize("view", ["raw", "cleaned"])
def test_rezeroed_detrended_is_the_residual_from_zero(window, view):
    """Re-zeroed, a --ref detrend station is D_full(plate) − model[linear +
    periodic] up to one constant per component, and starts at zero; its
    outliers sit on the same residual."""
    import geo_dataread.gps_read as gpsr
    from gps_analysis import evaluate_record

    from gps_plot.sources import gps as source

    pos_dir, pid, record = DEPLOYED
    req = source.SeriesRequest(
        fstart=window[0], fend=window[1], ref="detrend", Dir=pos_dir,
        tType="POS", uncert=15, view=view, pos_id=pid,
    )  # fmt: skip
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = source.rezero_detrended("SELF", req, source.fetch_station("SELF", req))
    t_full, d_full, _, _ = gpsr.getData(
        "SELF", ref="plate", Dir=pos_dir, tType="POS", uncert=15, pos_id=pid
    )
    _, i, j = np.intersect1d(t_full, s.t, return_indices=True)
    want = d_full[:, i] - np.asarray(
        evaluate_record(record, t_full[i], groups=("secular", "periodic"))
    )
    got = np.asarray(s.y)[:, j]
    for c in range(3):
        ok = np.isfinite(got[c])
        diff = got[c][ok] - want[c][ok]
        np.testing.assert_allclose(diff, diff[0], atol=1e-6)
        first = np.flatnonzero(np.isfinite(np.asarray(s.y)[c]))[:5]
        assert abs(np.average(np.asarray(s.y)[c][first],
                              weights=1 / np.asarray(s.sigma)[c][first])) < 1e-9  # fmt: skip
        if s.outliers is not None:
            o = np.asarray(s.outliers[0])[c, j]
            f = np.isfinite(o)
            np.testing.assert_allclose(o[f] - want[c][f], diff[0], atol=1e-6)
