"""Tests for the batch driver (gps_plot.plot_gps_timeseries).

Covers the perf-plot-parallel pass: the --ref/--special "all" variant
expansion (station-outer inversion), the per-process raw-read cache
(bit-identical, copy-on-hit semantics), and the per-station worker's
fault tolerance.  ``geo_dataread`` is stubbed via ``sys.modules`` so the
suite runs in the repo's own env (which does not ship the production
readers).
"""

import sys
import types

import numpy as np
import pytest

from gps_plot import plot_gps_timeseries as driver


# ---------------------------------------------------------------------------
# geo_dataread stub
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_gps_read(monkeypatch):
    """Install a stub ``geo_dataread.gps_read`` exposing openGlobkTimes."""
    gps_read = types.ModuleType("geo_dataread.gps_read")
    calls = []

    def openGlobkTimes(sta, Dir=None, tType="TOT"):
        calls.append((sta, Dir, tType))
        n = 5
        yearf = np.linspace(2020.0, 2020.1, n)
        data = np.arange(3.0 * n).reshape(3, n)
        ddata = np.full((3, n), 0.5)
        return yearf, data, ddata

    gps_read.openGlobkTimes = openGlobkTimes
    gps_read.calls = calls

    package = types.ModuleType("geo_dataread")
    package.gps_read = gps_read
    monkeypatch.setitem(sys.modules, "geo_dataread", package)
    monkeypatch.setitem(sys.modules, "geo_dataread.gps_read", gps_read)
    return gps_read


# ---------------------------------------------------------------------------
# _expand_variants
# ---------------------------------------------------------------------------

REFS = ["plate", "detrend", "itrf2008"]
SPECIALS = ["90d", "year", "full", "fixedstart"]


def test_expand_variants_passthrough():
    kwargs = {"ref": "plate", "special": "90d", "save": "png"}
    variants = driver._expand_variants(kwargs, REFS, SPECIALS)
    assert variants == [kwargs]
    assert variants[0] is not kwargs  # fresh dict, safe to ship to workers


def test_expand_variants_special_all():
    kwargs = {"ref": "plate", "special": "all", "save": "png"}
    variants = driver._expand_variants(kwargs, REFS, SPECIALS)
    assert [v["special"] for v in variants] == SPECIALS
    assert all(v["ref"] == "plate" for v in variants)
    assert all(v["save"] == "png" for v in variants)


def test_expand_variants_ref_all():
    kwargs = {"ref": "all", "special": "year"}
    variants = driver._expand_variants(kwargs, REFS, SPECIALS)
    assert [v["ref"] for v in variants] == REFS
    assert all(v["special"] == "year" for v in variants)


def test_expand_variants_both_all_legacy_order():
    # legacy loop order: ref outer, special inner
    variants = driver._expand_variants({"ref": "all", "special": "all"}, REFS, SPECIALS)
    assert [(v["ref"], v["special"]) for v in variants] == [
        (r, s) for r in REFS for s in SPECIALS
    ]


# ---------------------------------------------------------------------------
# _install_raw_read_cache
# ---------------------------------------------------------------------------


def test_raw_read_cache_reads_once_per_key(fake_gps_read):
    driver._install_raw_read_cache()
    fake_gps_read.openGlobkTimes("SENG", Dir="/data/", tType="TOT")
    fake_gps_read.openGlobkTimes("SENG", Dir="/data/", tType="TOT")
    assert fake_gps_read.calls == [("SENG", "/data/", "TOT")]  # one disk read

    fake_gps_read.openGlobkTimes("SENG", Dir="/data/", tType="08h")  # JOIN path
    assert len(fake_gps_read.calls) == 2  # distinct key -> new read


def test_raw_read_cache_returns_pristine_copies(fake_gps_read):
    driver._install_raw_read_cache()
    yearf1, data1, ddata1 = fake_gps_read.openGlobkTimes("SKSH", Dir="/data/")
    # getData mutates in place (iprep m->mm, plate removal): simulate it
    data1 *= 1000
    ddata1 *= 1000
    yearf2, data2, ddata2 = fake_gps_read.openGlobkTimes("SKSH", Dir="/data/")
    assert data2.max() < 20  # unscaled -- the cache handed out a fresh copy
    np.testing.assert_array_equal(yearf1, yearf2)
    assert data1 is not data2 and ddata1 is not ddata2


def test_raw_read_cache_install_is_idempotent(fake_gps_read):
    driver._install_raw_read_cache()
    wrapped = fake_gps_read.openGlobkTimes
    driver._install_raw_read_cache()
    assert fake_gps_read.openGlobkTimes is wrapped  # not double-wrapped


# ---------------------------------------------------------------------------
# plotStation
# ---------------------------------------------------------------------------


def test_plot_station_runs_every_variant(fake_gps_read, monkeypatch):
    seen = []
    monkeypatch.setattr(driver, "tryTimes", lambda sta, **kw: seen.append((sta, kw)))
    variants = [{"ref": "plate", "special": s} for s in ("90d", "year")]
    driver.plotStation("SENG", variants)
    assert seen == [
        ("SENG", {"ref": "plate", "special": "90d"}),
        ("SENG", {"ref": "plate", "special": "year"}),
    ]


def test_plot_station_survives_a_bad_variant(fake_gps_read, monkeypatch, capsys):
    # tryTimes' own fault tolerance: a failing plot must not stop the rest
    def boom(sta, **kwargs):
        print("%s Plotting" % sta)
        if kwargs["special"] == "90d":
            raise ValueError("no data for station %s" % sta)
        print("plotted %s using: %s, %s" % (sta, kwargs["ref"], kwargs["special"]))

    monkeypatch.setattr(driver.tplt, "plotTime", boom)
    variants = [{"ref": "plate", "special": s} for s in ("90d", "year")]
    driver.plotStation("SENG", variants)  # must not raise
    out = capsys.readouterr().out
    assert "plotted SENG using: plate, year" in out
