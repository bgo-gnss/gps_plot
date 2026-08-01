"""Tests for the batch driver (gps_plot.plot_gps_timeseries).

Covers the perf-plot-parallel pass: the --ref/--special "all" variant
expansion (station-outer inversion), the per-process raw-read cache
(bit-identical, copy-on-hit semantics), and the per-station worker's
fault tolerance.  ``geo_dataread`` is stubbed via ``sys.modules`` so the
suite runs in the repo's own env (which does not ship the production
readers).
"""

import importlib.util
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


# ---------------------------------------------------------------------------
# --outlier-param: NAME=VALUE -> OutlierParams (thresholds of the cleaned view)
# ---------------------------------------------------------------------------

# A MARK, not a module-level importorskip: the latter raises out of the module
# body, so a missing gps_analysis would drop this file's pre-existing variant /
# cache / worker coverage too, not just the outlier lane.
requires_gps_analysis = pytest.mark.skipif(
    importlib.util.find_spec("gps_analysis") is None,
    reason="gps_analysis (dev-group sibling) not installed",
)


@requires_gps_analysis
def test_build_outlier_params_none_when_nothing_assigned():
    """None (not a default-valued object) is what defers to the catalog."""
    assert driver._build_outlier_params(None) is None
    assert driver._build_outlier_params([]) is None


@requires_gps_analysis
def test_build_outlier_params_coerces_field_types():
    params = driver._build_outlier_params(
        ["window_n_sigma=3.5", "window_min_count=7", "scale_estimator=qn"]
    )
    assert params.window_n_sigma == 3.5
    assert isinstance(params.window_min_count, int)
    assert params.window_min_count == 7
    assert params.scale_estimator == "qn"


@requires_gps_analysis
def test_build_outlier_params_leaves_unassigned_fields_at_spec_defaults():
    """No default is restated in the CLI — untouched fields must match."""
    from gps_analysis import OutlierParams

    spec = OutlierParams()
    params = driver._build_outlier_params(["window_n_sigma=3.5"])
    assert params.window_n_sigma == 3.5
    assert params.global_n_sigma == spec.global_n_sigma
    assert params.max_flag_fraction == spec.max_flag_fraction
    assert params.despike is spec.despike


@requires_gps_analysis
def test_help_lists_every_outlier_param_name():
    """``--help`` must document the flag the leaf actually accepts.

    The listing is generated, so the failure this guards is the one a
    hand-written list would produce silently: the leaf gains a threshold,
    ``--outlier-param`` accepts it, and the help never mentions it. The
    grouping map is allowed to be incomplete — an unknown field falls into
    "other" — and this asserts that the fallback really catches it.
    """
    import dataclasses

    from gps_analysis import OutlierParams

    text = driver.outlier_param_help()
    for field in dataclasses.fields(OutlierParams):
        assert "%s=" % field.name in text, "%s is undocumented" % field.name


@requires_gps_analysis
def test_help_shows_the_spec_defaults_in_a_spelling_the_flag_accepts():
    """A line must be copyable onto the command line, booleans included.

    ``bool("False")`` is True, so ``despike=False`` printed Python-style
    would parse back as the OPPOSITE of what the help says.
    """
    text = driver.outlier_param_help()
    assert "despike=false" in text and "despike=False" not in text
    assert "max_flag_fraction=0.05" in text
    assert driver._build_outlier_params(["despike=false"]).despike is False


def test_help_survives_a_missing_gps_analysis(monkeypatch):
    """``--help`` is the last thing allowed to fail on a bare install."""
    import builtins

    real_import = builtins.__import__

    def _no_gps_analysis(name, *args, **kwargs):
        if name == "gps_analysis":
            raise ImportError("simulated bare install")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_gps_analysis)
    assert driver.outlier_param_help() == ""


@requires_gps_analysis
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("true", True),
        ("True", True),
        ("yes", True),
        ("1", True),
        ("false", False),
        ("FALSE", False),
        ("no", False),
        ("0", False),
    ],
)
def test_build_outlier_params_parses_booleans(raw, expected):
    """bool('False') is True — the coercion must not cast strings blindly."""
    params = driver._build_outlier_params(["despike=%s" % raw])
    assert params.despike is expected


@requires_gps_analysis
def test_build_outlier_params_rejects_a_non_boolean_for_a_bool_field():
    with pytest.raises(SystemExit, match="expected a boolean"):
        driver._build_outlier_params(["despike=maybe"])


@requires_gps_analysis
def test_build_outlier_params_rejects_a_missing_equals():
    with pytest.raises(SystemExit, match="NAME=VALUE"):
        driver._build_outlier_params(["window_n_sigma"])


@requires_gps_analysis
def test_build_outlier_params_lists_valid_fields_on_an_unknown_name():
    with pytest.raises(SystemExit, match="unknown field") as excinfo:
        driver._build_outlier_params(["n_sigma=3"])
    # the error doubles as the discovery mechanism, so it must enumerate
    assert "window_n_sigma" in str(excinfo.value)
    assert "global_n_sigma" in str(excinfo.value)


@requires_gps_analysis
def test_build_outlier_params_rejects_a_non_numeric_value():
    with pytest.raises(SystemExit, match="expected float"):
        driver._build_outlier_params(["window_n_sigma=wide"])


@requires_gps_analysis
def test_build_outlier_params_surfaces_post_init_validation():
    """OutlierParams.__post_init__ errors must arrive as clean CLI errors."""
    with pytest.raises(SystemExit, match="global_n_sigma must be > 0"):
        driver._build_outlier_params(["global_n_sigma=-1"])
    with pytest.raises(SystemExit, match="scale_estimator must be"):
        driver._build_outlier_params(["scale_estimator=bogus"])


@requires_gps_analysis
def test_build_outlier_params_survives_the_worker_process_boundary():
    """main() ships the params through ProcessPoolExecutor -- must pickle.

    plotStation is submitted to a pool when --save is given for more than
    one station, so an OutlierParams that could not pickle would break
    only the parallel path, which the serial single-station runs never hit.
    """
    import pickle

    params = driver._build_outlier_params(
        ["window_n_sigma=3.0", "despike=true", "scale_estimator=qn"]
    )
    restored = pickle.loads(pickle.dumps(params))
    assert restored == params
    assert restored.window_n_sigma == 3.0
    assert restored.despike is True
