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

    # the last-point highlight is suppressed (BGÓ 2026-07-11) — no lightgreen marker
    light = matplotlib.colors.to_rgb("lightgreen")
    lg = (
        (np.abs(img[..., 0] - light[0]) < 0.06)
        & (np.abs(img[..., 1] - light[1]) < 0.06)
        & (np.abs(img[..., 2] - light[2]) < 0.06)
    )
    assert int(lg.sum()) == 0


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


def _install_gps_views_stub(monkeypatch, flags):
    """Stub geo_dataread.gps_views.detect_view_outliers with fixed flags."""
    import sys
    import types

    gps_views = types.ModuleType("geo_dataread.gps_views")

    def detect_view_outliers(yearf, data, Ddata=None, **kwargs):
        return flags, {
            "outlier_abort": False,
            "degraded": False,
            "degrade_reason": None,
            "n_flagged": int(np.count_nonzero(flags)),
        }

    gps_views.detect_view_outliers = detect_view_outliers
    package = types.ModuleType("geo_dataread")
    package.gps_views = gps_views
    monkeypatch.setitem(sys.modules, "geo_dataread", package)
    monkeypatch.setitem(sys.modules, "geo_dataread.gps_views", gps_views)


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

    cleaned, overlay = tplt._mask_outliers(yearf, data.copy(), ddata)
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

    cleaned, overlay = tplt._mask_outliers(yearf, data, ddata)
    assert overlay is None
    assert cleaned is data  # raw path: same object, no copy, no mask


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
