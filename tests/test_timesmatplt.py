"""Tests for the modernized static export path (gps_plot.timesmatplt).

Covers the static-plot-modernize deliverables: native single-``savefig``
export to EPS/PDF/PNG, the semantic red/green "Last datapoint" header
logic (and its visibility in the rasterized PNG), and the reusable-figure
path.  Runs headless on Agg with synthetic data; the export tests need a
working TeX toolchain (Path B keeps ``text.usetex``) and are skipped
where TeX is unavailable.
"""

import matplotlib

matplotlib.use("Agg")  # before any pyplot import — headless CI/dev

import datetime
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pytest
from gtimes.timefunc import currDate

from gps_plot import timesmatplt as tplt

HAVE_TEX = all(shutil.which(tool) for tool in ("latex", "dvipng", "dvips", "gs"))

needs_tex = pytest.mark.skipif(
    not HAVE_TEX, reason="TeX toolchain (latex/dvipng/dvips/gs) not available"
)


def _synthetic_series(n_days: int, end: datetime.datetime):
    """Daily 3-component random-walk series ending at ``end`` (noon epochs)."""
    rng = np.random.default_rng(42)
    x = [end - datetime.timedelta(days=int(d)) for d in range(n_days - 1, -1, -1)]
    y = np.cumsum(rng.normal(0.0, 1.5, size=(3, n_days)), axis=1)
    dy = np.full((3, n_days), 2.0)
    return x, y, dy


def _yesterday_noon() -> datetime.datetime:
    d = currDate(-1)
    return datetime.datetime(d.year, d.month, d.day, 12, 0, 0)


# ---------------------------------------------------------------------------
# Semantic red/green status logic
# ---------------------------------------------------------------------------


def test_data_is_current_yesterday_is_green():
    assert tplt.data_is_current(_yesterday_noon()) is True


def test_data_is_current_stale_is_red():
    assert tplt.data_is_current(datetime.datetime(2025, 4, 21, 12, 0)) is False
    assert tplt.data_is_current(np.datetime64("2025-04-21T12:00")) is False


def test_make_title_status_color_green():
    title = tplt.make_title("TEST", _yesterday_noon(), ref="PLATE")
    assert title.status_color == tplt.STATUS_CURRENT_COLOR
    assert "Last datapoint:" in title.status
    assert "TEST (TEST)" in title.main


def test_make_title_status_color_red():
    title = tplt.make_title("TEST", datetime.datetime(2025, 4, 21, 12), ref="PLATE")
    assert title.status_color == tplt.STATUS_STALE_COLOR


def test_makelatexTitle_legacy_strings_keep_textcolor():
    green = tplt.makelatexTitle("TEST", _yesterday_noon())
    stale = tplt.makelatexTitle("TEST", datetime.datetime(2025, 4, 21, 12))
    assert r"\textcolor{green}{Last datapoint:" in green[1]
    assert r"\textcolor{red}{Last datapoint:" in stale[1]
    assert r"\Huge\textcolor{black}{TEST (TEST)}" in green[0]


def test_save_formats_normalization():
    assert tplt._save_formats("png") == ["png"]
    assert tplt._save_formats("eps,pdf, png") == ["eps", "pdf", "png"]
    assert tplt._save_formats([".pdf", "png"]) == ["pdf", "png"]
    with pytest.raises(ValueError):
        tplt._save_formats("")


# ---------------------------------------------------------------------------
# Native three-format export + header color in the rendered output
# ---------------------------------------------------------------------------


@needs_tex
def test_three_format_export_green_header(tmp_path):
    """One figure -> native EPS+PDF+PNG; green status text visible in PNG."""
    x, y, dy = _synthetic_series(90, _yesterday_noon())
    title = tplt.make_title("GRN1", x[-1], ref="PLATE")
    assert title.status_color == tplt.STATUS_CURRENT_COLOR

    fig = tplt.stdTimesPlot(x, y, dy, Title=title, fig=tplt._reusable_figure())
    base = tmp_path / "GRN1-plate-90d"
    tplt.saveFig(str(base), ("eps", "pdf", "png"), fig)

    eps, pdf, png = (base.with_suffix(s) for s in (".eps", ".pdf", ".png"))
    for f in (eps, pdf, png):
        assert f.exists() and f.stat().st_size > 0, f
    assert pdf.read_bytes()[:5] == b"%PDF-"
    assert b"%!PS-Adobe" in eps.read_bytes()[:64]
    # pure LaTeX-green (0,1,0) status text must be present in the EPS source
    assert b"1.000 setgreen" in eps.read_bytes() or b"0 1 0" in eps.read_bytes()

    # ... and in the rasterized PNG: green glyph pixels in the title band.
    img = plt.imread(png)
    green = (img[..., 1] > 0.8) & (img[..., 0] < 0.3) & (img[..., 2] < 0.3)
    assert int(green.sum()) > 50

    # ... and the last-point highlight (revived BGÓ 2026-07-27) reaches the
    # raster too: this series ends yesterday, so the green title fragment and
    # the lightgreen dot must BOTH be present — they share one decision.
    light = matplotlib.colors.to_rgb("lightgreen")
    lg = (
        (np.abs(img[..., 0] - light[0]) < 0.06)
        & (np.abs(img[..., 1] - light[1]) < 0.06)
        & (np.abs(img[..., 2] - light[2]) < 0.06)
    )
    assert int(lg.sum()) > 0


@needs_tex
def test_three_format_export_red_header_and_figure_reuse(tmp_path):
    """Stale data -> red status; reusing the module Figure stays valid."""
    x, y, dy = _synthetic_series(90, datetime.datetime(2025, 4, 21, 12))
    title = tplt.make_title("RED1", x[-1], ref="PLATE")
    assert title.status_color == tplt.STATUS_STALE_COLOR

    fig1 = tplt._reusable_figure()
    fig = tplt.stdTimesPlot(x, y, dy, Title=title, fig=fig1)
    assert fig is fig1  # reuse, not reallocation

    base = tmp_path / "RED1-plate-90d"
    tplt.saveFig(str(base), "eps,pdf,png", fig)
    for suffix in (".eps", ".pdf", ".png"):
        f = base.with_suffix(suffix)
        assert f.exists() and f.stat().st_size > 0, f

    img = plt.imread(base.with_suffix(".png"))
    # no green status pixels for stale data (data markers are red anyway)
    green = (img[..., 1] > 0.8) & (img[..., 0] < 0.3) & (img[..., 2] < 0.3)
    assert int(green.sum()) == 0
    # red pixels exist (status text and/or markers share pure red)
    red = (img[..., 0] > 0.8) & (img[..., 1] < 0.3) & (img[..., 2] < 0.3)
    assert int(red.sum()) > 50

    # second build on the SAME figure (clear + rebuild) still exports
    x2, y2, dy2 = _synthetic_series(60, _yesterday_noon())
    title2 = tplt.make_title("GRN2", x2[-1], ref="PLATE")
    fig2 = tplt.stdTimesPlot(x2, y2, dy2, Title=title2, fig=fig1)
    assert fig2 is fig1
    base2 = tmp_path / "GRN2-plate-60d"
    tplt.saveFig(str(base2), "png", fig2)
    img2 = plt.imread(base2.with_suffix(".png"))
    green2 = (img2[..., 1] > 0.8) & (img2[..., 0] < 0.3) & (img2[..., 2] < 0.3)
    assert int(green2.sum()) > 50


# ---------------------------------------------------------------------------
# view toggle (raw | cleaned | detrended) — internal-delivery slice
# ---------------------------------------------------------------------------


class _StubResolved:
    """Stand-in for gps_views.ResolvedOutlierConfig."""

    def __init__(self, params, min_outlier=None):
        self.params = params
        self.min_outlier = min_outlier
        self.overrides_applied = {}
        self.overrides_source = None
        self.min_outlier_source = None


def _install_gps_views_stub(
    monkeypatch,
    flags,
    *,
    steps=(),
    windows=(),
    catalog_params="CATALOG",
    catalog_floor=None,
    provisional=None,
):
    """Stub the whole gps_views cleaning chain and record what it received.

    Mirrors the real module surface ``_mask_outliers`` depends on: the two
    catalog resolvers, the params/floor resolver and the detector.  The
    returned dict captures every call so a test can assert the plumbing
    rather than just the flags.
    """
    import sys
    import types

    calls = {}
    gps_views = types.ModuleType("geo_dataread.gps_views")

    declared_steps = np.asarray(steps, dtype=float)

    def station_step_epochs(sta, *, steps=None):
        calls["step_sta"] = sta
        return declared_steps, None

    def resolve_protect_windows(sta, protect_windows=None):
        calls["window_sta"] = sta
        return tuple(windows), None

    def resolve_outlier_detection(
        sta, *, outlier_params=None, min_outlier=None, outlier_overrides=None
    ):
        calls["resolve_sta"] = sta
        calls["explicit_params"] = outlier_params
        calls["overrides_path"] = outlier_overrides
        # real precedence: explicit arg beats the catalog row
        return _StubResolved(
            catalog_params if outlier_params is None else outlier_params,
            catalog_floor,
        )

    def detect_view_outliers(yearf, data, Ddata=None, **kwargs):
        calls["detect_kwargs"] = kwargs
        return flags, {
            "outlier_abort": False,
            "degraded": False,
            "degrade_reason": None,
            "n_flagged": int(np.count_nonzero(flags)),
            "provisional": np.asarray(provisional, dtype=bool)
            if provisional is not None
            else np.zeros_like(np.asarray(flags), dtype=bool),
            "n_provisional": 0 if provisional is None else int(np.sum(provisional)),
        }

    gps_views.station_step_epochs = station_step_epochs
    gps_views.resolve_protect_windows = resolve_protect_windows
    gps_views.resolve_outlier_detection = resolve_outlier_detection
    gps_views.detect_view_outliers = detect_view_outliers
    package = types.ModuleType("geo_dataread")
    package.gps_views = gps_views
    monkeypatch.setitem(sys.modules, "geo_dataread", package)
    monkeypatch.setitem(sys.modules, "geo_dataread.gps_views", gps_views)
    return calls


def test_plot_time_rejects_unknown_view():
    with pytest.raises(ValueError, match="view must be"):
        tplt.plotTime("SENG", view="bogus", save="png")


def test_mask_outliers_masks_and_overlays(monkeypatch):
    n = 10
    yearf = np.linspace(2020.0, 2020.1, n)
    data = np.arange(3.0 * n).reshape(3, n)
    ddata = np.full((3, n), 0.5)
    flags = np.zeros((3, n), dtype=bool)
    flags[0, 2] = flags[2, 7] = True
    _install_gps_views_stub(monkeypatch, flags)

    cleaned, overlay, _prov = tplt._mask_outliers("RHOF", yearf, data.copy(), ddata)
    assert overlay is not None
    out_data, out_ddata = overlay
    # mask only: flagged epochs NaN in the main series, present in overlay
    assert np.isnan(cleaned[0, 2]) and np.isnan(cleaned[2, 7])
    assert cleaned[1, 2] == data[1, 2]  # per-component mask
    assert out_data[0, 2] == data[0, 2] and out_data[2, 7] == data[2, 7]
    assert np.isnan(out_data[0, 3])
    assert out_ddata[0, 2] == 0.5 and np.isnan(out_ddata[1, 2])
    # unflagged values byte-identical
    keep = ~flags
    np.testing.assert_array_equal(cleaned[keep], data[keep])


def test_mask_outliers_no_flags_returns_input_unchanged(monkeypatch):
    n = 6
    yearf = np.linspace(2021.0, 2021.05, n)
    data = np.arange(3.0 * n).reshape(3, n)
    ddata = np.full((3, n), 0.5)
    _install_gps_views_stub(monkeypatch, np.zeros((3, n), dtype=bool))

    cleaned, overlay, _prov = tplt._mask_outliers("RHOF", yearf, data, ddata)
    assert overlay is None
    assert cleaned is data  # raw path: same object, no copy, no mask


# ---------------------------------------------------------------------------
# Station-aware resolution chain (mirrors gps_views.read_gps_view)
# ---------------------------------------------------------------------------


def test_mask_outliers_resolves_every_station_catalog():
    """All three catalogs are keyed on the station and reach the detector."""
    n = 8
    yearf = np.linspace(2020.0, 2020.1, n)
    data = np.zeros((3, n))
    ddata = np.full((3, n), 0.5)
    steps = (2020.03, 2020.07)
    windows = ((2020.01, 2020.02),)

    with pytest.MonkeyPatch.context() as mp:
        calls = _install_gps_views_stub(
            mp,
            np.zeros((3, n), dtype=bool),
            steps=steps,
            windows=windows,
            catalog_floor=(1.0, 2.0, 3.0),
        )
        tplt._mask_outliers("HOFN", yearf, data, ddata)

    assert calls["step_sta"] == calls["window_sta"] == calls["resolve_sta"] == "HOFN"
    dk = calls["detect_kwargs"]
    np.testing.assert_allclose(dk["step_epochs"], steps)
    assert dk["protect_windows"] == windows
    assert dk["min_outlier"] == (1.0, 2.0, 3.0)


def test_mask_outliers_defers_to_catalog_params_when_none_passed():
    """No explicit params => the station's catalog row reaches the detector."""
    n = 6
    yearf = np.linspace(2021.0, 2021.05, n)
    data = np.zeros((3, n))

    with pytest.MonkeyPatch.context() as mp:
        calls = _install_gps_views_stub(
            mp, np.zeros((3, n), dtype=bool), catalog_params="CATALOG"
        )
        tplt._mask_outliers("RHOF", yearf, data, None)

    assert calls["explicit_params"] is None
    assert calls["detect_kwargs"]["outlier_params"] == "CATALOG"


def test_mask_outliers_explicit_params_and_overrides_path_win():
    """Explicit params beat the catalog; the overrides path is forwarded."""
    n = 6
    yearf = np.linspace(2021.0, 2021.05, n)
    data = np.zeros((3, n))

    with pytest.MonkeyPatch.context() as mp:
        calls = _install_gps_views_stub(mp, np.zeros((3, n), dtype=bool))
        tplt._mask_outliers(
            "RHOF",
            yearf,
            data,
            None,
            outlier_params="EXPLICIT",
            outlier_overrides="/tmp/ov.csv",
        )

    assert calls["explicit_params"] == "EXPLICIT"
    assert calls["overrides_path"] == "/tmp/ov.csv"
    assert calls["detect_kwargs"]["outlier_params"] == "EXPLICIT"


def test_mask_outliers_passes_no_steps_as_none_not_empty_array():
    """An empty step catalog must send None, the detector's 'no steps' value."""
    n = 5
    yearf = np.linspace(2021.0, 2021.05, n)
    data = np.zeros((3, n))

    with pytest.MonkeyPatch.context() as mp:
        calls = _install_gps_views_stub(mp, np.zeros((3, n), dtype=bool), steps=())
        tplt._mask_outliers("RHOF", yearf, data, None)

    assert calls["detect_kwargs"]["step_epochs"] is None


def test_std_times_plot_handles_nan_masked_series():
    """The cleaned view feeds NaN-masked arrays; ylim math must survive."""
    x, y, dy = _synthetic_series(60, _yesterday_noon())
    y = y.copy()
    y[0, 5] = y[1, 10] = np.nan  # masked outlier epochs
    fig = tplt.stdTimesPlot(
        x, y, dy, Title="NANV", ylim=[5], fig=tplt._reusable_figure()
    )
    for ax in fig.axes[:3]:
        lo, hi = ax.get_ylim()
        assert np.isfinite(lo) and np.isfinite(hi)


# ---------------------------------------------------------------------------
# Last-point highlight — the green dot paired with the green title fragment
# ---------------------------------------------------------------------------


def _highlight_markers(ax):
    """Lines on ``ax`` drawn with the last-point highlight face color."""
    return [
        ln
        for ln in ax.get_lines()
        if ln.get_markerfacecolor() == "lightgreen" and len(ln.get_xdata()) == 1
    ]


def test_last_point_highlight_drawn_when_data_is_current():
    """Series ending yesterday => one green dot per component axis."""
    x, y, dy = _synthetic_series(40, _yesterday_noon())
    fig = tplt.stdTimesPlot(x, y, dy, Title="GRN", fig=tplt._reusable_figure())
    for i, ax in enumerate(fig.axes[:3]):
        markers = _highlight_markers(ax)
        assert len(markers) == 1, f"component {i}: expected one highlight marker"
        assert markers[0].get_markeredgecolor() == "black"
        assert markers[0].get_ydata()[0] == pytest.approx(y[i][-1])


def test_last_point_highlight_absent_when_data_is_stale():
    """Stale series (the local TOT snapshot case) => no marker anywhere."""
    stale_end = _yesterday_noon() - datetime.timedelta(days=10)
    x, y, dy = _synthetic_series(40, stale_end)
    fig = tplt.stdTimesPlot(x, y, dy, Title="RED", fig=tplt._reusable_figure())
    for ax in fig.axes[:3]:
        assert _highlight_markers(ax) == []


def test_highlight_last_override_wins_over_local_derivation():
    """plotTime passes the title's decision explicitly; it must be honoured."""
    stale_end = _yesterday_noon() - datetime.timedelta(days=10)
    x, y, dy = _synthetic_series(20, stale_end)
    fig = tplt.stdTimesPlot(
        x, y, dy, Title="OVR", fig=tplt._reusable_figure(), highlight_last=True
    )
    assert all(len(_highlight_markers(ax)) == 1 for ax in fig.axes[:3])

    x, y, dy = _synthetic_series(20, _yesterday_noon())
    fig = tplt.stdTimesPlot(
        x, y, dy, Title="OVR", fig=tplt._reusable_figure(), highlight_last=False
    )
    assert all(_highlight_markers(ax) == [] for ax in fig.axes[:3])


def test_highlight_skips_component_whose_last_epoch_was_masked():
    """Cleaned view: flagged final epoch draws no dot on THAT axis only."""
    x, y, dy = _synthetic_series(30, _yesterday_noon())
    y = y.copy()
    y[1, -1] = np.nan  # east flagged as an outlier at the last epoch
    fig = tplt.stdTimesPlot(x, y, dy, Title="MSK", fig=tplt._reusable_figure())
    assert len(_highlight_markers(fig.axes[0])) == 1
    assert len(_highlight_markers(fig.axes[2])) == 1
    drawn = _highlight_markers(fig.axes[1])
    assert drawn == [] or np.isnan(drawn[0].get_ydata()[0])


# ---------------------------------------------------------------------------
# Outlier overlay styling (cleaned view) — grey, not red
# ---------------------------------------------------------------------------


def test_outlier_overlay_colors_are_grey():
    assert tplt.OUTLIER_ERRORBAR_COLOR == "grey"
    assert tplt.OUTLIER_EDGE_COLOR == "dimgrey"
    assert tplt.OUTLIER_FACE_COLOR == "lightgrey"


def test_add_data_applies_outlier_colors():
    """addData keeps its red defaults; the grey comes from the call site."""
    x, y, dy = _synthetic_series(12, _yesterday_noon())
    fig = tplt.stdFrame(fig=tplt._reusable_figure())[0]
    tplt.addData(
        x,
        y,
        dy,
        fig,
        ecolor=tplt.OUTLIER_ERRORBAR_COLOR,
        markerfacecolor=tplt.OUTLIER_FACE_COLOR,
        markeredgecolor=tplt.OUTLIER_EDGE_COLOR,
    )
    for ax in fig.axes[:3]:
        # errorbar() also leaves a marker-less Line2D of the same length
        overlay = [
            ln
            for ln in ax.get_lines()
            if len(ln.get_xdata()) == len(x) and ln.get_marker() == "o"
        ]
        assert len(overlay) == 1, "expected exactly one overlay marker series"
        assert overlay[0].get_markerfacecolor() == "lightgrey"
        assert overlay[0].get_markeredgecolor() == "dimgrey"


# ---------------------------------------------------------------------------
# --name: fixed output basename
# ---------------------------------------------------------------------------


def _stub_gps_read(monkeypatch, end: datetime.datetime, n_days: int = 30):
    """Stub geo_dataread.gps_read so plotTime runs without station data."""
    import sys
    import types

    x, y, dy = _synthetic_series(n_days, end)
    yearf = np.linspace(2025.0, 2025.0 + n_days / 365.25, n_days)

    gps_read = types.ModuleType("geo_dataread.gps_read")
    gps_read.getData = lambda sta, **kw: (yearf, y, dy, 0.0)
    gps_read.toDateTime = lambda _yearf: x
    package = types.ModuleType("geo_dataread")
    package.gps_read = gps_read
    monkeypatch.setitem(sys.modules, "geo_dataread", package)
    monkeypatch.setitem(sys.modules, "geo_dataread.gps_read", gps_read)


def test_name_gives_a_fixed_basename(monkeypatch, tmp_path):
    _stub_gps_read(monkeypatch, _yesterday_noon())
    saved = []
    monkeypatch.setattr(tplt, "saveFig", lambda fn, ft, fig, **kw: saved.append(fn))

    tplt.plotTime(
        "RHOF", save="png", figDir=str(tmp_path), logo=False, name="RHOF-test"
    )
    assert saved == [str(tmp_path / "RHOF-test")]


def test_name_is_stable_across_ref_and_special(monkeypatch, tmp_path):
    """Same --name must overwrite one path, whatever the variant."""
    _stub_gps_read(monkeypatch, _yesterday_noon())
    saved = []
    monkeypatch.setattr(tplt, "saveFig", lambda fn, ft, fig, **kw: saved.append(fn))

    for special in ("90d", "year"):
        tplt.plotTime(
            "RHOF",
            save="png",
            figDir=str(tmp_path),
            logo=False,
            name="RHOF-test",
            special=special,
        )
    assert saved == [str(tmp_path / "RHOF-test")] * 2


def test_without_name_the_legacy_filename_is_unchanged(monkeypatch, tmp_path):
    _stub_gps_read(monkeypatch, _yesterday_noon())
    saved = []
    monkeypatch.setattr(tplt, "saveFig", lambda fn, ft, fig, **kw: saved.append(fn))

    tplt.plotTime("RHOF", save="png", figDir=str(tmp_path), logo=False, special="90d")
    assert saved == [str(tmp_path / "RHOF-itrf2008-90d")]


def _grey_overlay_lines(fig):
    """Overlay marker series drawn with the outlier face colour."""
    return [
        ln
        for ax in fig.axes[:3]
        for ln in ax.get_lines()
        if ln.get_markerfacecolor() == tplt.OUTLIER_FACE_COLOR
    ]


def _cleaned_fig(monkeypatch, tmp_path, *, hide):
    """Render the cleaned view with one flagged epoch per component."""
    _stub_gps_read(monkeypatch, _yesterday_noon())
    monkeypatch.setattr(tplt, "saveFig", lambda fn, ft, fig, **kw: None)
    flags = np.zeros((3, 30), dtype=bool)
    flags[:, 7] = True
    _install_gps_views_stub(monkeypatch, flags)
    return tplt.plotTime(
        "RHOF",
        save="png",
        figDir=str(tmp_path),
        logo=False,
        view="cleaned",
        hide_outliers=hide,
    )


def test_outlier_overlay_drawn_by_default(monkeypatch, tmp_path):
    fig = _cleaned_fig(monkeypatch, tmp_path, hide=False)
    assert len(_grey_overlay_lines(fig)) == 3  # one per component axis


def test_hide_outliers_suppresses_the_overlay(monkeypatch, tmp_path):
    fig = _cleaned_fig(monkeypatch, tmp_path, hide=True)
    assert _grey_overlay_lines(fig) == []


def test_hide_outliers_does_not_restore_the_masked_epochs(monkeypatch, tmp_path):
    """Display-only: the flagged epoch stays NaN in the plotted series.

    The whole point of the flag is that it changes what is SHOWN, never
    what is plotted as data -- a hidden overlay must not smuggle the
    outlier back into the main series.
    """
    fig = _cleaned_fig(monkeypatch, tmp_path, hide=True)
    for ax in fig.axes[:3]:
        main = [ln for ln in ax.get_lines() if len(ln.get_ydata()) == 30]
        assert main, "main series not drawn"
        assert np.isnan(np.asarray(main[0].get_ydata(), dtype=float)[7])


def test_hide_outliers_is_inert_on_the_raw_view(monkeypatch, tmp_path):
    _stub_gps_read(monkeypatch, _yesterday_noon())
    monkeypatch.setattr(tplt, "saveFig", lambda fn, ft, fig, **kw: None)
    fig = tplt.plotTime(
        "RHOF", save="png", figDir=str(tmp_path), logo=False, hide_outliers=True
    )
    for ax in fig.axes[:3]:
        main = [ln for ln in ax.get_lines() if len(ln.get_ydata()) == 30]
        assert main and np.all(np.isfinite(np.asarray(main[0].get_ydata(), float)))


def _provisional_lines(fig):
    return [
        ln
        for ax in fig.axes[:3]
        for ln in ax.get_lines()
        if ln.get_markerfacecolor() == tplt.PROVISIONAL_FACE_COLOR
    ]


def _fig_with_provisional(monkeypatch, tmp_path, *, hide_outliers=False):
    """Cleaned view: epoch 7 flagged, epoch 25 provisional (recent)."""
    _stub_gps_read(monkeypatch, _yesterday_noon())
    monkeypatch.setattr(tplt, "saveFig", lambda fn, ft, fig, **kw: None)
    flags = np.zeros((3, 30), dtype=bool)
    flags[:, 7] = True
    prov = np.zeros((3, 30), dtype=bool)
    prov[:, 25] = True
    _install_gps_views_stub(monkeypatch, flags, provisional=prov)
    return tplt.plotTime(
        "RHOF",
        save="png",
        figDir=str(tmp_path),
        logo=False,
        view="cleaned",
        hide_outliers=hide_outliers,
    )


def test_provisional_epochs_get_their_own_marker(monkeypatch, tmp_path):
    fig = _fig_with_provisional(monkeypatch, tmp_path)
    assert len(_provisional_lines(fig)) == 3  # one per component axis


def test_provisional_epochs_stay_in_the_main_series(monkeypatch, tmp_path):
    """The core invariant: provisional ANNOTATES, it never removes.

    A flagged epoch is NaN in the plotted series; a provisional one must
    still be finite there -- otherwise the marker would silently do the
    masking it exists to avoid.
    """
    fig = _fig_with_provisional(monkeypatch, tmp_path)
    for ax in fig.axes[:3]:
        main = [ln for ln in ax.get_lines() if len(ln.get_ydata()) == 30]
        y = np.asarray(main[0].get_ydata(), dtype=float)
        assert np.isnan(y[7]), "flagged epoch must be masked"
        assert np.isfinite(y[25]), "provisional epoch must be kept"


def test_hide_outliers_does_not_hide_provisional(monkeypatch, tmp_path):
    """Decluttering removes DECIDED outliers, never undecided ones."""
    fig = _fig_with_provisional(monkeypatch, tmp_path, hide_outliers=True)
    assert _grey_overlay_lines(fig) == []
    assert len(_provisional_lines(fig)) == 3


def test_no_provisional_marker_when_none_reported(monkeypatch, tmp_path):
    fig = _cleaned_fig(monkeypatch, tmp_path, hide=False)
    assert _provisional_lines(fig) == []


def test_missing_provisional_key_degrades_quietly(monkeypatch, tmp_path):
    """An older geo_dataread has no 'provisional' key -- must not break."""
    _stub_gps_read(monkeypatch, _yesterday_noon())
    monkeypatch.setattr(tplt, "saveFig", lambda fn, ft, fig, **kw: None)
    flags = np.zeros((3, 30), dtype=bool)
    flags[:, 7] = True
    _install_gps_views_stub(monkeypatch, flags)

    import geo_dataread.gps_views as stub

    inner = stub.detect_view_outliers

    def legacy(yearf, data, Ddata=None, **kwargs):
        kwargs.pop("provisional_days", None)
        f, prov = inner(yearf, data, Ddata, **kwargs)
        prov.pop("provisional", None)
        prov.pop("n_provisional", None)
        return f, prov

    stub.detect_view_outliers = legacy
    fig = tplt.plotTime(
        "RHOF", save="png", figDir=str(tmp_path), logo=False, view="cleaned"
    )
    assert _provisional_lines(fig) == []
    assert len(_grey_overlay_lines(fig)) == 3  # cleaning itself still works


def test_plot_time_forwards_outlier_levers_to_the_masker(monkeypatch, tmp_path):
    """plotTime -> _mask_outliers: sta first, both levers as keywords."""
    _stub_gps_read(monkeypatch, _yesterday_noon())
    monkeypatch.setattr(tplt, "saveFig", lambda fn, ft, fig, **kw: None)
    seen = {}

    def spy(sta, yearf, data, Ddata, **kwargs):
        seen["sta"] = sta
        seen.update(kwargs)
        return data, None, None

    monkeypatch.setattr(tplt, "_mask_outliers", spy)
    tplt.plotTime(
        "RHOF",
        save="png",
        figDir=str(tmp_path),
        logo=False,
        view="cleaned",
        outlier_params="PARAMS",
        outlier_overrides="/tmp/ov.csv",
        provisional_days=7.0,
    )
    assert seen == {
        "sta": "RHOF",
        "outlier_params": "PARAMS",
        "outlier_overrides": "/tmp/ov.csv",
        "provisional_days": 7.0,
    }


def test_raw_view_never_touches_the_outlier_chain(monkeypatch, tmp_path):
    """The levers are cleaned-view only — raw must not resolve any catalog."""
    _stub_gps_read(monkeypatch, _yesterday_noon())
    monkeypatch.setattr(tplt, "saveFig", lambda fn, ft, fig, **kw: None)

    def boom(*a, **k):
        raise AssertionError("_mask_outliers called for view='raw'")

    monkeypatch.setattr(tplt, "_mask_outliers", boom)
    tplt.plotTime(
        "RHOF", save="png", figDir=str(tmp_path), logo=False, outlier_params="PARAMS"
    )
