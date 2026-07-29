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


def _detrended_north_std() -> float:
    """std of the detrended north series, through the PRODUCTION read path."""
    import geo_dataread.gps_read as gpsr

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _y, d, _s, _o = gpsr.getData(
            STA, ref="detrend", Dir=str(TOT), tType="TOT", uncert=10
        )
    absent = any("absent from the detrend parameter" in str(w.message) for w in caught)
    return float(np.nanstd(d[0])), absent


def test_commit_then_detrended_view_consumes_the_record(gpsconfig, tmp_path):
    """The whole point: commit a record, and production reads it back."""
    import json

    from gps_plot.detrend_workbench import build_record, commit_record

    doc = gpsconfig / "detrend_params.json"
    before_doc = json.loads(doc.read_text())
    assert STA not in before_doc["stations"], "fixture assumes RHOF is absent"

    std_before, absent_before = _detrended_north_std()
    assert absent_before, "expected the 'absent record' degrade before commit"

    record, *_ = build_record(STA, tot_dir=str(TOT), max_gap_years=2.0)
    path, n_before, n_after = commit_record(STA, record, params_path=doc)
    assert (path, n_after) == (doc, n_before + 1)

    std_after, absent_after = _detrended_north_std()
    assert not absent_after, "record was committed but production did not read it"
    # the numeric half: a real trajectory removal, not a no-op
    assert std_after < std_before * 0.75, (std_before, std_after)

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
    from gps_plot.detrend_workbench import commit_record

    doc = gpsconfig / "detrend_params.json"
    rec = {"model": "lineperiodic", "fitted_at": None}
    commit_record(STA, rec, params_path=doc)
    with pytest.raises(RuntimeError, match="already has a record"):
        commit_record(STA, rec, params_path=doc)
    commit_record(STA, rec, params_path=doc, force=True)  # explicit is fine
