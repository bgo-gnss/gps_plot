"""Round-trip proof for the detrend workbench (PLAN DoD 5).

The workbench is only worth having if a record it commits is actually
consumed by the path that reads records in production. Without this test it
is a figure generator: the PDFs would look right whether or not anything
downstream ever used the result.

Runs the FULL loop — estimate, commit, read back through
``plot-gps-timeseries``'s data path — against a throwaway ``GPS_CONFIG_PATH``
so the operator's own parameter document is never touched.

Needs the local joined TOT dataset and the 33-station document; skipped
without them, so a checkout on another machine stays green.
"""

from __future__ import annotations

import shutil
import warnings
from pathlib import Path

import numpy as np
import pytest

TOT = Path.home() / "gps-data" / "TOT"
TESTCFG = Path.home() / "gps-data" / "testcfg"
STA = "RHOF"

#: Thresholds for "the trajectory really was removed", CALIBRATED rather than
#: guessed. Measured on RHOF: a CORRECT record leaves rate residuals
#: [0.003, 0.075, 0.034] mm/yr and seasonal [0.123, 0.031, 0.305] mm; a
#: north/east swap leaves rate [0.483, 0.406, 0.034]. These sit between —
#: ~4x above the correct case, comfortably below the corruption.
#:
#: Honest limit: a component swap is only detectable to the extent the
#: components differ. RHOF's N/E rates are -0.706 / -1.186 mm/yr, so the swap
#: leaves 0.48. On a station whose N and E rates happened to match, this test
#: could not see it — no residual-based check can.
MAX_RESIDUAL_RATE = 0.3
MAX_RESIDUAL_SEASONAL = 1.0

pytestmark = pytest.mark.skipif(
    not (TOT.is_dir() and (TESTCFG / "detrend_params.json").is_file()),
    reason="needs ~/gps-data/TOT and ~/gps-data/testcfg/detrend_params.json",
)


@pytest.fixture
def gpsconfig(tmp_path, monkeypatch):
    """A disposable copy of the deployed config tree."""
    dst = tmp_path / "gpsconfig"
    shutil.copytree(TESTCFG, dst, symlinks=False)
    monkeypatch.setenv("GPS_CONFIG_PATH", str(dst))
    return dst


def _detrended() -> tuple[np.ndarray, np.ndarray, bool]:
    """The detrended series, through the PRODUCTION read path."""
    import geo_dataread.gps_read as gpsr

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        y, d, _s, _o = gpsr.getData(
            STA, ref="detrend", Dir=str(TOT), tType="TOT", uncert=10
        )
    absent = any("absent from the detrend parameter" in str(w.message) for w in caught)
    return y, d, absent


def _residual_trend(yearf: np.ndarray, series: np.ndarray) -> tuple[float, float]:
    """Refit the DETRENDED series: |rate| and seasonal amplitude should be ~0.

    This is the assertion the first version of this test lacked, and its
    absence was not cosmetic — an adversarial review committed records with
    flipped seasonal signs, swapped north/east components, a wrong frame tag
    and a stale window, and EVERY ONE passed the old "warning gone + std
    dropped" pair. Only a doubled rate was caught. A std drop proves
    something was subtracted; it does not prove the right thing was.
    """
    from gps_analysis.fitting import fit_components
    from gps_analysis.models import lineperiodic

    fit = fit_components(lineperiodic, yearf, series)[0]
    rate = abs(float(fit.params[1]))
    seasonal = float(np.hypot(fit.params[2], fit.params[3]))
    return rate, seasonal


def test_commit_then_detrended_view_consumes_the_record(gpsconfig, tmp_path):
    """The whole point: commit a record, and production reads it back."""
    import json

    from gps_plot.detrend_workbench import build_record, commit_record

    doc = gpsconfig / "detrend_params.json"
    before_doc = json.loads(doc.read_text())
    assert STA not in before_doc["stations"], "fixture assumes RHOF is absent"

    y0, d0, absent_before = _detrended()
    assert absent_before, "expected the 'absent record' degrade before commit"
    std_before = float(np.nanstd(d0[0]))

    record, *_ = build_record(STA, tot_dir=str(TOT), max_gap_years=2.0)
    path, n_before, n_after = commit_record(STA, record, params_path=doc)
    assert (path, n_after) == (doc, n_before + 1)

    y1, d1, absent_after = _detrended()
    assert not absent_after, "record was committed but production did not read it"
    # (a) consumption: something was subtracted
    assert float(np.nanstd(d1[0])) < std_before * 0.75

    # (b) CORRECTNESS, on ALL THREE components: refitting the detrended
    # series must find no trend and no seasonal left. This is what catches a
    # sign flip, a component swap, or a stale window -- none of which (a)
    # can see.
    for c, name in enumerate(("north", "east", "up")):
        rate, seasonal = _residual_trend(y1, d1[c])
        assert rate < MAX_RESIDUAL_RATE, f"{name}: {rate:.3f} mm/yr of trend left in"
        assert seasonal < MAX_RESIDUAL_SEASONAL, f"{name}: {seasonal:.3f} mm seasonal"

    after_doc = json.loads(doc.read_text())
    assert after_doc["schema_version"] == before_doc["schema_version"]
    assert after_doc["frame"] == before_doc["frame"]
    for sta, rec in before_doc["stations"].items():
        assert after_doc["stations"][sta] == rec, f"{sta} was disturbed"


def test_commit_refuses_to_create_the_document(tmp_path, monkeypatch):
    """The footgun: a new one-station file becomes what everything reads."""
    from gps_plot.detrend_workbench import commit_record

    missing = tmp_path / "nowhere" / "detrend_params.json"
    missing.parent.mkdir()
    with pytest.raises(RuntimeError, match="Refusing to create"):
        commit_record(STA, {"model": "lineperiodic"}, params_path=missing)
    assert not missing.exists()


def test_commit_refuses_to_clobber_without_force(gpsconfig):
    from gps_plot.detrend_workbench import build_record, commit_record

    doc = gpsconfig / "detrend_params.json"
    # A REAL record: the earlier version of this test committed the stub
    # {"model": "lineperiodic"}, which the validator now (correctly) refuses.
    # That stub passing was itself the bug -- it would have degraded the
    # station to raw at every read.
    rec, *_ = build_record(STA, tot_dir=str(TOT), max_gap_years=2.0)
    commit_record(STA, rec, params_path=doc)
    with pytest.raises(RuntimeError, match="already has a record"):
        commit_record(STA, rec, params_path=doc)
    commit_record(STA, rec, params_path=doc, force=True)  # explicit is fine


def test_commit_rejects_an_unusable_record(gpsconfig):
    """Validation at commit time, where the operator is looking."""
    from gps_plot.detrend_workbench import commit_record

    doc = gpsconfig / "detrend_params.json"
    with pytest.raises(RuntimeError, match="not usable by the apply path"):
        commit_record(STA, {"model": "lineperiodic"}, params_path=doc)


def test_commit_rejects_a_frame_mismatch(gpsconfig):
    """Design §2.5: applying across frames is refused, not fudged.

    Caught at COMMIT rather than at read, because the production read path
    now passes frame= and would otherwise raise for every consumer.
    """
    from gps_plot.detrend_workbench import build_record, commit_record

    doc = gpsconfig / "detrend_params.json"
    rec, *_ = build_record(STA, tot_dir=str(TOT), max_gap_years=2.0)
    rec["frame"] = "itrf2008"
    with pytest.raises(RuntimeError, match="!= document frame"):
        commit_record(STA, rec, params_path=doc, force=True)


@pytest.mark.parametrize(
    "corrupt,label",
    [
        (lambda r: _flip_seasonal(r), "seasonal signs flipped"),
        (lambda r: _swap_components(r), "north/east swapped"),
    ],
)
def test_correctness_assertions_catch_wrong_records(gpsconfig, corrupt, label):
    """The wrong records that passed the FIRST version of this test.

    Each was constructed by an adversarial review and slipped through the
    old "warning gone + std dropped" pair. They must not slip through now.
    """
    from gps_plot.detrend_workbench import build_record, commit_record

    doc = gpsconfig / "detrend_params.json"
    record, *_ = build_record(STA, tot_dir=str(TOT), max_gap_years=2.0)
    commit_record(STA, corrupt(record), params_path=doc, force=True)

    y, d, absent = _detrended()
    assert not absent, "the corrupt record should still be CONSUMED"
    caught = any(
        _residual_trend(y, d[c])[0] >= MAX_RESIDUAL_RATE
        or _residual_trend(y, d[c])[1] >= MAX_RESIDUAL_SEASONAL
        for c in range(3)
    )
    assert caught, f"{label}: corruption slipped through the assertions"


def _flip_seasonal(record):
    for comp in record["components"]:
        p = list(comp["params"])
        for i in (2, 3, 4, 5):
            if i < len(p):
                p[i] = -p[i]
        comp["params"] = p
    return record


def _swap_components(record):
    comps = record["components"]
    comps[0]["params"], comps[1]["params"] = comps[1]["params"], comps[0]["params"]
    return record
