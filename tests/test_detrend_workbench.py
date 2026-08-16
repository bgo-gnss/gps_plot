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

import json
import shutil
import warnings
from pathlib import Path

import numpy as np
import pytest

TOT = Path.home() / "gps-data" / "TOT"
TESTCFG = Path.home() / "gps-data" / "testcfg"
STA = "RHOF"
STA_SELF = "SELF"

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
    """A disposable copy of the deployed config tree, with ``STA`` REMOVED.

    Removing it ESTABLISHES the precondition these tests need (a station with
    no stored record) instead of assuming it. The earlier version only
    asserted ``STA not in stations``, which meant the suite broke the moment
    an operator legitimately curated RHOF into the live document — a test
    failing because someone did their job is a bad test, not a bad commit.

    The copy is disposable, so the deletion never touches the real document.
    """
    dst = tmp_path / "gpsconfig"
    shutil.copytree(TESTCFG, dst, symlinks=False)

    doc = dst / "detrend_params.json"
    payload = json.loads(doc.read_text())
    payload["stations"].pop(STA, None)
    doc.write_text(json.dumps(payload, indent=1))

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
    # guaranteed by the gpsconfig fixture, which removes STA from its copy
    assert STA not in before_doc["stations"]

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


# ---------------------------------------------------------------------------
# T5 — TOS equipment lines (tier A)
# ---------------------------------------------------------------------------

TOS_FIXTURE = Path(__file__).parent / "data" / "tos_SELF.json"


def _dev(subtype, model=None, serial=None):
    from gps_plot.detrend_workbench import DeviceInfo

    return DeviceInfo(subtype, model, serial)


#: SELF's devices, as TOS returns them (verified live 2026-08-01). Kept in
#: the TEST rather than folded into ``tos_SELF.json`` so the fixture stays a
#: faithful mirror of one station payload — which is what
#: ``test_tos_live_matches_the_fixture`` compares against.
#:
#: Antenna 4528 is the case that earns the serial column: TRM29659.00 /
#: 263955 runs to 2010-06-03 and re-joins the SAME day, so without an
#: identity check that day claims an antenna change it never had.
SELF_DEVICES = {
    5803: _dev("sim_card"),
    4520: _dev("antenna", "TRM29659.00", "190269"),
    4783: _dev("gnss_receiver", "TRIMBLE 4000SSI", "15404"),
    5247: _dev("monument"),
    4458: _dev("antenna", "TRM33429.20+GP", "0220171774"),
    4782: _dev("gnss_receiver", "TRIMBLE 4000SSI", "15269"),
    5248: _dev("monument"),
    4454: _dev("antenna", "TRM33429.00+GP", "0220156111"),
    4820: _dev("gnss_receiver", "TRIMBLE 4000SSI", "24908"),
    5249: _dev("monument"),
    5250: _dev("monument"),
    5320: _dev("radome"),
    4827: _dev("gnss_receiver", "TRIMBLE 5700", "0220268934"),
    4528: _dev("antenna", "TRM29659.00", "263955"),
    4885: _dev("gnss_receiver", "TRIMBLE NETRS", "4808145652"),
    20290: _dev("modem_gsm"),
}


def test_tos_epochs_keep_only_antenna_and_receiver_installs():
    """The merge gate is the FIXTURE, not the network.

    A network-dependent assertion cannot gate a merge on a machine off the
    VPN. The live-TOS check below is marked and skipped by default.

    SELF's payload is 17 device joins over 7 distinct ``time_from`` values,
    and every filter has to fire to get from those to the two lines an
    operator expects. The ``1000-01-01`` "since forever" sentinel is a
    registration artefact. Only ``antenna`` / ``gnss_receiver`` can move a
    phase centre, so the monuments, the radome, the SIM card and the 2024
    GSM modem earn nothing. Three joins are 3-4 day CAMPAIGN deployments
    from before the station's real life, registered exactly like permanent
    equipment — only their duration tells them apart. And one visit that
    swaps antenna AND receiver is two rows sharing an epoch, so what
    survives still coalesces per day.

    17 rows -> 2 lines. Each number below is a different filter's evidence,
    which is why they are asserted separately rather than as one count.
    """
    import json

    from gps_plot.detrend_workbench import tos_equipment_epochs

    payload = json.loads(TOS_FIXTURE.read_text())
    assert len(payload["children_connections"]) == 17, "fixture drifted"

    events = tos_equipment_epochs("SELF", payload=payload, devices=SELF_DEVICES)
    labels = [lbl for _, lbl in events]
    assert labels == [
        "2002-02-05 (rx TRIMBLE 5700, ant TRM29659.00)",
        # receiver TRIMBLE 5700 -> NETRS, but antenna TRM29659.00 / 263955
        # CONTINUES (re-registered the day it is closed). A receiver change,
        # and it must not claim an antenna change it never had.
        "2010-06-03 (rx TRIMBLE NETRS)",
    ]
    assert not any("2024-04-26" in lbl for lbl in labels), "GSM modem drew a line"
    assert not any(lbl.startswith("1000") for lbl in labels), "sentinel leaked"
    assert not any(lbl.startswith("2001-") for lbl in labels), "campaign gear drew"
    assert events == sorted(events), "epochs must be ordered"
    # the 2008 Ölfus coseismic is NOT here: TOS knows equipment, not seismicity
    assert not any(2008.0 < e < 2009.0 for e, _ in events)


def test_short_campaign_joins_are_not_station_equipment():
    """3-4 day joins are campaign gear; an OPEN join counts however young.

    The distinction has to be duration, because TOS registers a campaign
    deployment exactly like a permanent install — same subtype, same
    attributes. SELF's three (2001-07-01, -07-16, -09-14, all 3-4 d) sit
    against installs of 3040 days and open, so the 30-day floor separates
    them with two orders of magnitude to spare.

    The open case is the one worth pinning: treating "no end date" as zero
    days would hide the newest equipment on every station in the network.
    """
    from gps_plot.detrend_workbench import tos_equipment_epochs

    ant = _dev("antenna", "TRM29659.00", "263955")
    payload = {
        "children_connections": [
            {"id_entity_child": 1, "time_from": "2001-07-01", "time_to": "2001-07-04"},
            {"id_entity_child": 1, "time_from": "2015-01-01", "time_to": "2015-03-01"},
            {"id_entity_child": 1, "time_from": "2026-07-30"},  # open, days old
        ]
    }
    labels = [
        lbl for _, lbl in tos_equipment_epochs("X", payload=payload, devices={1: ant})
    ]
    assert labels == ["2015-01-01 (ant TRM29659.00)", "2026-07-30 (ant TRM29659.00)"]


def test_removals_never_draw_a_line():
    """``time_to`` is read by nothing: a line is an INSTALL, per operator rule.

    Constructed rather than fixture-driven, because neither RHOF nor SELF
    has a removal day without an install on it — the case is real but not
    present in the working set, and a filter nobody exercises is a filter
    nobody can trust.
    """
    from gps_plot.detrend_workbench import tos_equipment_epochs

    payload = {
        "children_connections": [
            # installed 2010, pulled 2015 and NOT replaced
            {"id_entity_child": 1, "time_from": "2010-05-04", "time_to": "2015-09-01"},
        ]
    }
    events = tos_equipment_epochs(
        "X", payload=payload, devices={1: _dev("antenna", "TRM29659.00", "263955")}
    )
    assert [lbl for _, lbl in events] == ["2010-05-04 (ant TRM29659.00)"]


def test_a_telecoms_visit_earns_no_line():
    """RHOF 2023-08-16: a GSM modem + a SIM card, two rows, one visit.

    This drew a full-height line across all three components before the
    subtype filter. It cannot displace the antenna by a micron.
    """
    from gps_plot.detrend_workbench import tos_equipment_epochs

    payload = {
        "children_connections": [
            {"id_entity_child": 18662, "time_from": "2023-08-16T13:20:00"},
            {"id_entity_child": 18659, "time_from": "2023-08-16T15:30:00"},
        ]
    }
    devices = {18662: _dev("modem_gsm"), 18659: _dev("sim_card")}
    assert tos_equipment_epochs("RHOF", payload=payload, devices=devices) == []


def test_total_lookup_failure_raises_instead_of_reporting_no_changes():
    """An unresolvable fleet must NOT read as "this station never changed".

    Empty output and total failure render identically — a figure with no
    green lines — so the blanket case has to reach the operator through the
    caller's existing warning. A PARTIAL failure is survivable: unresolved
    means not whitelisted, and the count is printed.
    """
    from gps_plot.detrend_workbench import tos_equipment_epochs

    payload = {
        "children_connections": [
            {"id_entity_child": 7, "time_from": "2010-05-04"},
            {"id_entity_child": 8, "time_from": "2012-06-06"},
        ]
    }
    with pytest.raises(RuntimeError, match="resolved nothing"):
        tos_equipment_epochs("X", payload=payload, devices={})

    # one of two resolves -> a line for it, no exception
    events = tos_equipment_epochs(
        "X", payload=payload, devices={7: _dev("antenna", "TRM29659.00", "263955")}
    )
    assert [lbl for _, lbl in events] == ["2010-05-04 (ant TRM29659.00)"]


def test_tos_failure_degrades_to_a_warning(monkeypatch):
    """Off-VPN the workbench must still work, minus the lines."""
    from gps_plot import detrend_workbench as wb

    with pytest.raises(RuntimeError, match="TOS lookup failed"):
        wb.tos_equipment_epochs("SELF", url="http://127.0.0.1:9/nope")


@pytest.mark.network
def test_tos_live_matches_the_fixture():
    """Live TOS, skipped by default — run with -m network."""
    import json

    from gps_plot.detrend_workbench import tos_equipment_epochs

    live = tos_equipment_epochs("SELF")
    cached = tos_equipment_epochs(
        "SELF", payload=json.loads(TOS_FIXTURE.read_text()), devices=SELF_DEVICES
    )
    assert live == cached, (
        "TOS changed; refresh tests/data/tos_SELF.json AND SELF_DEVICES — the "
        "device records are a second mirror and drift independently of the payload"
    )


# ---------------------------------------------------------------------------
# T6 — seismic / declared event lines (tier A, second half)
# ---------------------------------------------------------------------------


def test_declared_steps_split_seismic_from_equipment(tmp_path):
    """``kind`` is the discriminator, and it is why steps.csv is read directly.

    ``gps_views.station_step_epochs`` flattens to bare epochs and DROPS
    kind/source/comment — so it cannot tell an earthquake from an antenna
    swap, which is the entire point of drawing them in different colours.
    """
    from gps_plot.detrend_workbench import declared_event_epochs

    csv = tmp_path / "steps.csv"
    csv.write_text(
        "sta,epoch_yearf,component,kind,source,comment\n"
        "SELF,2008.4085,ALL,earthquake,manual,Olfus M6.3\n"
        "SELF,2010.4205,ALL,equipment,tos,receiver swap\n"
        # same epoch declared per-component must coalesce to ONE event
        "SELF,2024.3183,N,equipment,tos,antenna\n"
        "SELF,2024.3183,E,equipment,tos,antenna\n"
        "SELF,2024.3183,U,equipment,tos,antenna\n"
    )
    seismic, other = declared_event_epochs("SELF", steps_catalog=csv)
    assert [round(e, 4) for e, _ in seismic] == [2008.4085]
    assert "Olfus" in seismic[0][1]
    assert [round(e, 4) for e, _ in other] == [2010.4205, 2024.3183]
    assert len(other) == 2, "per-component rows must coalesce to one event"


def test_declared_steps_missing_catalog_is_not_an_error(tmp_path):
    """Catalogs are enhancements — a missing one yields no lines, not a crash."""
    from gps_plot.detrend_workbench import declared_event_epochs

    assert declared_event_epochs("SELF", steps_catalog=tmp_path / "nope.csv") == (
        [],
        [],
    )


def test_parse_events_and_rejects_bad_input():
    from gtimes.timefunc import TimetoYearf

    from gps_plot.detrend_workbench import parse_events

    got = parse_events(["20080529,Olfus M6.3"])
    assert len(got) == 1
    assert got[0][0] == pytest.approx(float(TimetoYearf(2008, 5, 29)), abs=1e-6)
    assert got[0][1] == "Olfus M6.3"
    assert parse_events(["20240426"])[0][1] == "2024-04-26"
    for bad in ("2008-05-29", "200805", "abcdefgh"):
        with pytest.raises(SystemExit, match="YYYYMMDD"):
            parse_events([bad])


def test_events_outside_the_plotted_span_are_split_off():
    """An install that predates the solution must not reach the figure.

    ``axvline`` clips to the axes but its label does not, so an
    out-of-span event loses its line and keeps its caption — stranded in
    the margin beside an axis it does not mark. Measured on BJTV:
    installed 2021-08-09, series starts 2024.09.
    """
    from gps_plot.detrend_workbench import clip_events_to_span

    yearf = np.linspace(2024.1, 2026.5, 100)
    events = [
        (2021.6041, "2021-08-09 (rx TRIMBLE NETR5)"),
        (2024.7363, "2024-09-26 (rx TRIMBLE NETR9)"),
        (2030.0, "future"),
    ]
    inside, outside = clip_events_to_span(events, yearf)
    assert [e[0] for e in inside] == [2024.7363]
    assert [e[0] for e in outside] == [2021.6041, 2030.0]
    # the bounds themselves are INSIDE — an event on the first or last
    # epoch marks a real column of data
    edges = [(float(yearf[0]), "first"), (float(yearf[-1]), "last")]
    assert clip_events_to_span(edges, yearf)[1] == []


def test_clip_keeps_everything_when_there_is_no_span():
    """No span means nothing can be outside it — never empty the annotation."""
    from gps_plot.detrend_workbench import clip_events_to_span

    events = [(2021.6, "a"), (2030.0, "b")]
    for empty in (np.array([]), np.array([np.nan, np.nan])):
        inside, outside = clip_events_to_span(events, empty)
        assert inside == events and outside == []


def test_render_clips_events_itself(tmp_path, monkeypatch):
    """The clip lives in ``render``, so no caller can forget it.

    Both PDF pages draw from these lists; a caller-side clip would leave
    a direct ``render()`` call (tests, a REPL) drawing into the margin.
    """
    import gps_plot.detrend_workbench as wb

    record, yearf, data, sigma, est = _windowed_estimate()
    lo = float(np.nanmin(yearf))
    drawn: list[tuple[float, str]] = []
    monkeypatch.setattr(
        wb,
        "add_event_lines",
        lambda fig, events, color: drawn.extend(events) or fig,
    )
    wb.render(
        STA,
        record,
        yearf,
        data,
        sigma,
        tmp_path / "clipped.pdf",
        outliers=est.outliers,
        tos_events=[(lo - 3.0, "long before"), (lo + 0.5, "in span")],
        seismic_events=[(lo - 1.0, "also before")],
    )
    assert drawn, "the in-span event must still be drawn"
    assert [label for _e, label in drawn] == ["in span", "in span"]  # two pages


def test_declared_step_coincides_with_the_seismic_event_line(tmp_path):
    """The plan's acceptance criterion: the declared epoch IS the earthquake.

    Both halves go through the real code paths — the catalog epoch via
    ``declared_event_epochs`` (darkred line), the calendar date via
    ``parse_events`` (``--event``) — and they must land on the same epoch.

    On the constants: ``TimetoYearf`` returns the **noon** daily-solution
    epoch, so May 29 2008 is ``149.5/366 = 2008.40847``, not the naive
    midnight ``149/366 = 2008.4071``. The plan's ticket hardcoded the
    midnight value; the declared steps.csv row and ``TimetoYearf`` agree with
    each other to 3e-5, which is the coincidence the criterion was after.

    The fixture supplies the row rather than the deployed catalog: the
    analysis-lane catalogs are not deployed on this host, so
    ``declared_event_epochs("SELF")`` reads an absent file and correctly
    returns nothing.
    """
    from gtimes.timefunc import TimetoYearf

    from gps_plot.detrend_workbench import declared_event_epochs, parse_events

    csv = tmp_path / "steps.csv"
    csv.write_text(
        "sta,epoch_yearf,component,kind,source,comment\n"
        "SELF,2008.4085,ALL,earthquake,manual,Olfus M6.3\n"
    )
    seismic, _ = declared_event_epochs(STA_SELF, steps_catalog=csv)
    declared = seismic[0][0]
    supplied = parse_events(["20080529,Olfus M6.3"])[0][0]

    assert abs(declared - supplied) < 1e-4
    assert abs(supplied - float(TimetoYearf(2008, 5, 29))) < 1e-9
    assert abs(supplied - 2008.4071) > 1e-3, "the midnight constant is NOT this epoch"


def test_a_half_day_epoch_shift_does_not_move_the_fit():
    """Robustness, not tolerance-shopping: on daily data the midnight and
    noon epochs fall between the same pair of observations, so the step term
    is identical either way. Worth pinning — it means an operator who types
    a midnight fractional year still gets the right fit, only a line drawn
    half a day early."""
    from gtimes.timefunc import TimetoYearf

    from gps_plot.detrend_workbench import build_record

    noon = float(TimetoYearf(2008, 5, 29))
    midnight = 2008.4071
    assert 0 < noon - midnight < 2.0 / 365.25, "the two must straddle no epoch"
    # RHOF, not SELF: SELF's steps.csv DECLARES 2008.4085, so passing either
    # spelling there merges into two steps for one event -- which is now a
    # hard error (see the next test). RHOF declares none, so each run here
    # fits exactly one step and the comparison is of epochs, not of counts.
    a, *_ = build_record(STA, tot_dir=str(TOT), max_gap_years=2.0, steps=[midnight])
    b, *_ = build_record(STA, tot_dir=str(TOT), max_gap_years=2.0, steps=[noon])
    assert [round(float(v), 4) for v in a["rms"]] == [
        round(float(v), 4) for v in b["rms"]
    ]


def test_redeclaring_a_catalog_step_is_refused_not_silently_degenerate():
    """The footgun the half-day test used to walk into unnoticed.

    SELF's ``steps.csv`` declares the Ölfus coseismic at 2008.4085. An
    operator who types ``--step 2008.4071`` for the SAME event gets both,
    because ``_declared_step_epochs`` treats the catalog as a floor and
    merges. Half a day apart on daily data, no epoch separates them, so the
    two Heaviside columns are IDENTICAL and the design matrix is rank
    deficient: the individual amplitudes are meaningless (only their sum is
    determined) and the covariance comes back infinite.

    That used to happen silently -- the fit still predicted well, so an rms
    comparison could not see it. It is now refused, naming both epochs.
    """
    from gps_plot.detrend_workbench import build_record

    with pytest.raises(RuntimeError, match="not separable"):
        build_record(STA_SELF, tot_dir=str(TOT), max_gap_years=2.0, steps=[2008.4071])


# ---------------------------------------------------------------------------
# Outlier overlay: the figure must agree with the record printed beside it
# ---------------------------------------------------------------------------


def test_grey_overlay_count_matches_the_printed_n_rejected():
    """The one check that rules out the wrong mask.

    Two masks are available here and they look identical on a figure: the
    FIT's inlier verdict, and a fresh ``detect_view_outliers`` run of the
    kind ``plot-gps-timeseries --view cleaned`` does. They disagree by
    construction — the view detector sees neither the fit window nor a
    ``--step`` declared on the command line — so a workbench page drawn
    from the second one would silently contradict the ``n_rejected`` the
    same run prints. Count parity is what distinguishes them.
    """
    from gps_plot.detrend_workbench import build_record, split_outliers

    record, _yearf, data, sigma, est = build_record(
        STA_SELF, tot_dir=str(TOT), max_gap_years=1.0
    )
    n_rejected = [int(v) for v in record["n_rejected"]]
    assert n_rejected == [int(v) for v in est.outliers.sum(axis=1)]
    assert any(n_rejected), "SELF should reject something; a no-op proves nothing"

    kept, overlay = split_outliers(data, sigma, est.outliers)
    assert overlay is not None
    for c, n in enumerate(n_rejected):
        # MASK, not filter: the arrays keep their length so they stay
        # index-aligned with the plotted x, and the two are disjoint.
        assert kept.shape == data.shape
        newly_nan = int(np.count_nonzero(np.isnan(kept[c]) & ~np.isnan(data[c])))
        assert newly_nan == n
        assert int(np.count_nonzero(~np.isnan(overlay[0][c]))) == n


def test_split_outliers_is_a_noop_when_nothing_was_rejected():
    """No rejections must leave the ORIGINAL array, not a copy full of data.

    ``render`` relies on this to skip the overlay entirely; returning a
    fresh array would work but hides that nothing happened.
    """
    from gps_plot.detrend_workbench import split_outliers

    data = np.arange(12, dtype=float).reshape(3, 4)
    sigma = np.ones_like(data)
    kept, overlay = split_outliers(data, sigma, np.zeros_like(data, dtype=bool))
    assert overlay is None
    assert kept is data


def test_hide_outliers_changes_display_only(tmp_path):
    """``--hide-outliers`` must not touch the record or the masked series.

    Same contract as ``plotTime``: the epochs are already out of the fit,
    so this decides only whether the figure still shows them.
    """
    from gps_plot.detrend_workbench import build_record, render

    record, yearf, data, sigma, est = build_record(
        STA_SELF, tot_dir=str(TOT), max_gap_years=1.0
    )
    shown = render(
        STA_SELF,
        record,
        yearf,
        data,
        sigma,
        tmp_path / "shown.pdf",
        outliers=est.outliers,
    )
    hidden = render(
        STA_SELF,
        record,
        yearf,
        data,
        sigma,
        tmp_path / "hidden.pdf",
        outliers=est.outliers,
        hide_outliers=True,
    )
    assert shown.is_file() and hidden.is_file()
    # the overlay is real artist output, so dropping it must shrink the file
    assert hidden.stat().st_size < shown.stat().st_size


# ---------------------------------------------------------------------------
# Out-of-window screen: a SECOND lane, on epochs the fit never judged
# ---------------------------------------------------------------------------

#: A window that ends well before the series does, so most of the station's
#: history is out of it.  A record fitted on an open window judges every
#: epoch and leaves this lane with nothing to prove.
WINDOW_END = 2014.5


def _windowed_estimate(sta=STA):
    from gps_plot.detrend_workbench import build_record

    return build_record(
        sta, tot_dir=str(TOT), segments=((None, WINDOW_END),), max_gap_years=2.0
    )


def test_screen_flags_only_epochs_the_fit_left_unjudged():
    """The two lanes must be disjoint, and the fit's must be untouched.

    This is the invariant that lets the figure carry two greys at all: a
    reader counts the solid grey and gets the record's ``n_rejected``,
    counts the hollow grey and gets epochs the record has no opinion
    about.  If the screen leaked one flag inside the window, the printed
    ``n_rejected`` would stop matching the figure — the exact failure
    ``test_grey_overlay_count_matches_the_printed_n_rejected`` guards
    against on the other side.
    """
    from gps_plot.detrend_workbench import screen_outside_window

    record, yearf, data, sigma, est = _windowed_estimate()
    in_window = np.asarray(est.in_window, dtype=bool)
    assert not in_window.all(), "the window must actually clip the series"

    flags, prov = screen_outside_window(STA, yearf, data, sigma, est)
    assert flags is not None
    assert flags.shape == data.shape
    assert not flags[:, in_window].any(), "the screen must not judge inside"
    assert not np.asarray(prov)[:, in_window].any()
    assert flags.any(), "RHOF has blunders after 2014; a no-op proves nothing"

    # and the fit's own verdict is unchanged by the screen having run
    assert [int(v) for v in record["n_rejected"]] == [
        int(v) for v in est.outliers.sum(axis=1)
    ]
    assert not (flags & np.asarray(est.outliers, dtype=bool)).any()


def test_restrict_narrows_the_verdict_not_the_detection():
    """``restrict`` must not change WHAT the detector sees, only what it says.

    Detection fits its own trajectory across the series, so screening a
    post-2014 fragment on its own would be a different — and worse —
    estimate than screening the full series and reporting a slice of it.
    Equality with the unrestricted run masked down is what proves the
    detector still got the whole span; a truncating implementation passes
    every shape assertion and fails this one.
    """
    from gps_plot.timesmatplt import view_flags

    _record, yearf, data, sigma, est = _windowed_estimate()
    outside = ~np.asarray(est.in_window, dtype=bool)

    full, _fp, _fa = view_flags(STA, yearf, data, sigma)
    narrowed, _np_, _na = view_flags(STA, yearf, data, sigma, restrict=outside)
    assert np.array_equal(narrowed, full & outside)
    assert narrowed.any() and not np.array_equal(narrowed, full)


def test_screen_is_fed_the_fits_own_declared_steps():
    """A ``--step`` typed at the CLI has to reach the screen too.

    And the record cannot deliver it: ``record["step_epochs"]`` keeps only
    the epochs INSIDE the fit window, so a step declared in the screened
    stretch — the only place this lane looks — is absent from it by
    construction.  A detector run without a declared step over-flags
    around it at any threshold, the same hazard that keeps
    ``split_outliers`` from re-running the view detector inside the
    window.
    """
    from gps_plot import detrend_workbench as wb

    step = 2016.5
    _record, yearf, data, sigma, est = wb.build_record(
        STA,
        tot_dir=str(TOT),
        segments=((None, WINDOW_END),),
        max_gap_years=2.0,
        steps=[step],
    )
    seen = {}

    def _spy(*args, **kwargs):
        seen.update(kwargs)
        return (
            np.zeros(np.shape(data), dtype=bool),
            np.zeros(np.shape(data), dtype=bool),
            None,
        )

    import gps_plot.timesmatplt as tplt

    assert step not in [float(v) for v in (_record["step_epochs"] or ())], (
        "the record must NOT carry an out-of-window step; that is why the "
        "screen cannot use it as its source"
    )

    original = tplt.view_flags
    tplt.view_flags = _spy
    try:
        wb.screen_outside_window(STA, yearf, data, sigma, est, steps=[step])
    finally:
        tplt.view_flags = original

    assert step in [float(v) for v in np.atleast_1d(seen["step_epochs"])]


def test_provisional_days_reaches_the_detector_from_the_cli(tmp_path, monkeypatch):
    """The gold lane's only bound must be reachable, not just accepted.

    ``provisional_days`` decides which indeterminate epochs are recent
    enough to mark. Out-of-window is where a pre-unrest window leaves a
    decade of epochs, and indeterminate clusters also sit at old
    mid-series gaps — so an unreachable bound would let those dominate a
    lane documented as being about RECENT undecided epochs.
    """
    from gps_plot import detrend_workbench as wb
    import gps_plot.timesmatplt as tplt

    seen = {}
    original = tplt.view_flags

    def _spy(*args, **kwargs):
        seen.update(kwargs)
        return original(*args, **kwargs)

    tplt.view_flags = _spy
    try:
        rc = wb.main(
            [
                STA,
                "--tot-dir",
                str(TOT),
                "--window-end",
                str(WINDOW_END),
                "--max-gap-years",
                "2.0",
                "--no-tos",
                "--provisional-days",
                "45",
                "--out",
                str(tmp_path / "prov.pdf"),
            ]
        )
    finally:
        tplt.view_flags = original
    assert rc == 0
    assert seen["provisional_days"] == 45.0


def test_screen_failure_leaves_the_epochs_unjudged_and_the_figure_intact():
    """The screen is a reading aid; losing it must never lose the figure.

    Same graceful-degrade rule as every catalog resolver on this path: a
    warning and no lane, rather than an operator with no PDF.
    """
    from gps_plot import detrend_workbench as wb
    import gps_plot.timesmatplt as tplt

    record, yearf, data, sigma, est = _windowed_estimate()

    def _boom(*args, **kwargs):
        raise RuntimeError("detector exploded")

    original = tplt.view_flags
    tplt.view_flags = _boom
    try:
        flags, prov = wb.screen_outside_window(STA, yearf, data, sigma, est)
    finally:
        tplt.view_flags = original
    assert flags is None and prov is None


def test_hide_outliers_drops_the_out_of_window_lane_too(tmp_path):
    """``--hide-outliers`` means both greys: they are both decided verdicts.

    Only the gold provisional lane survives it, and for the reason it
    survives in ``plotTime`` — hiding a decided outlier declutters, hiding
    an undecided one hides the thing most worth looking at.
    """
    from gps_plot.detrend_workbench import render, screen_outside_window

    record, yearf, data, sigma, est = _windowed_estimate()
    outside, prov = screen_outside_window(STA, yearf, data, sigma, est)
    assert outside is not None and outside.any()

    def _render(name, **kw):
        return render(
            STA,
            record,
            yearf,
            data,
            sigma,
            tmp_path / name,
            outliers=est.outliers,
            **kw,
        )

    plain = _render("plain.pdf")
    screened = _render(
        "screened.pdf", outside_outliers=outside, outside_provisional=prov
    )
    hidden = _render(
        "hidden.pdf",
        outside_outliers=outside,
        outside_provisional=prov,
        hide_outliers=True,
    )
    assert screened.stat().st_size > plain.stat().st_size, "the lane must draw"
    assert hidden.stat().st_size < screened.stat().st_size


# ---------------------------------------------------------------------------
# --show-outliers: the SAME verdicts, drawn the other way round
# ---------------------------------------------------------------------------


def _capture_lanes(monkeypatch, tmp_path, name, **kw):
    """Render once, returning every ``addData`` layer keyed by marker fill.

    Colour is the whole of what ``--show-outliers`` changes, so the test
    has to see the artists rather than the file: a size comparison would
    pass just as happily on a figure that painted the inversion onto the
    wrong epochs.
    """
    import gps_plot.timesmatplt as tplt
    from gps_plot.detrend_workbench import render

    record, yearf, data, sigma, est = _windowed_estimate()
    outside, prov = screen_outside_window_or_skip(yearf, data, sigma, est)
    calls: list[dict] = []
    real = tplt.addData

    def spy(x, y, Dy, fig, **kwargs):
        calls.append({"y": np.asarray(y, dtype=float), **kwargs})
        return real(x, y, Dy, fig, **kwargs)

    monkeypatch.setattr(tplt, "addData", spy)
    render(
        STA,
        record,
        yearf,
        data,
        sigma,
        tmp_path / name,
        outliers=est.outliers,
        outside_outliers=outside,
        outside_provisional=prov,
        **kw,
    )
    # First occurrence per colour is page 1; `calls` is kept so a test can
    # check the SECOND page got the same treatment -- the detrended page
    # builds its own lanes, so an inversion could land on one page only.
    by_face: dict[str, np.ndarray] = {}
    for call in calls:
        face = call.get("markerfacecolor")
        if face is not None and face not in by_face:
            by_face[face] = call["y"]
    return np.asarray(data, dtype=float), est, outside, by_face, calls


def screen_outside_window_or_skip(yearf, data, sigma, est):
    from gps_plot.detrend_workbench import screen_outside_window

    outside, prov = screen_outside_window(STA, yearf, data, sigma, est)
    if outside is None or not outside.any():
        pytest.skip("no out-of-window flags on this station/dataset")
    return outside, prov


def test_show_outliers_draws_every_epoch_exactly_once(monkeypatch, tmp_path):
    """The inversion is a repaint, not a re-selection.

    Grey now means "no verdict against it" and red means "flagged", so
    the three marker layers must PARTITION the finite series: an epoch
    drawn twice would read as flagged on top of clean, and one drawn
    nowhere would vanish from a figure whose whole job is to show what
    the detector did.
    """
    import gps_plot.timesmatplt as tplt

    data, est, outside, by_face, calls = _capture_lanes(
        monkeypatch, tmp_path, "inverted.pdf", show_outliers=True
    )
    from gps_plot.detrend_workbench import INVERTED_FACE_COLOR, OUTSIDE_FACE_COLOR

    n_grey = sum(
        1 for c in calls if c.get("markerfacecolor") == tplt.OUTLIER_FACE_COLOR
    )
    assert n_grey == 2, "both pages must invert, not just the plate-frame one"

    grey = np.isfinite(by_face[tplt.OUTLIER_FACE_COLOR])
    red = np.isfinite(by_face[INVERTED_FACE_COLOR])
    hollow = np.isfinite(by_face[OUTSIDE_FACE_COLOR])

    finite = np.isfinite(data)
    assert not (grey & red).any(), "an epoch cannot be both kept and rejected"
    assert not (grey & hollow).any()
    assert not (red & hollow).any(), "the two red lanes must stay countable"
    assert ((grey | red | hollow) == finite).all(), "the layers must partition"
    # and the red lanes are exactly the masks, unchanged by the inversion
    assert (red == (np.asarray(est.outliers) & finite)).all()
    assert (hollow == (np.asarray(outside) & finite)).all()


def test_show_outliers_leaves_the_default_view_alone(monkeypatch, tmp_path):
    """Without the flag nothing is repainted — no grey layer is drawn.

    The kept series stays ``stdTimesPlot``'s own red, which is what makes
    the flag a pure addition: every existing figure is bit-identical.
    """
    import gps_plot.timesmatplt as tplt

    _data, _est, _outside, by_face, _calls = _capture_lanes(
        monkeypatch, tmp_path, "plain.pdf"
    )
    grey = by_face.get(tplt.OUTLIER_FACE_COLOR)
    assert grey is not None, "the fit's own rejections are still grey"
    # ... and that grey is the REJECTIONS, not the kept series
    assert np.isfinite(grey).sum() == int(np.asarray(_est.outliers).sum())


def test_show_outliers_is_exclusive_with_hide_outliers():
    """Contradictory by definition: drop the verdicts vs. show only them."""
    from gps_plot.detrend_workbench import _build_parser

    with pytest.raises(SystemExit):
        _build_parser().parse_args([STA, "--show-outliers", "--hide-outliers"])


def test_screening_masks_the_out_of_window_blunders_from_the_series():
    """Masked, not merely marked — that is what lets the y-axis tighten.

    The operational complaint this lane answers is a single post-window
    blunder owning the axis: RHOF north spans 73 mm out-of-window and 27
    once screened.  Marking without masking would leave the range intact
    and the figure just as unreadable.
    """
    from gps_plot.detrend_workbench import screen_outside_window, split_outliers

    _record, yearf, data, sigma, est = _windowed_estimate()
    outside, _prov = screen_outside_window(STA, yearf, data, sigma, est)
    kept, overlay = split_outliers(data, sigma, outside)
    assert overlay is not None

    unjudged = ~np.asarray(est.in_window, dtype=bool)
    before = np.nanmax(data[0][unjudged]) - np.nanmin(data[0][unjudged])
    after = np.nanmax(kept[0][unjudged]) - np.nanmin(kept[0][unjudged])
    assert after < before / 2


# ---------------------------------------------------------------------------
# --out routing: scratch figdir shared with tools/local-plot/figview.sh
# ---------------------------------------------------------------------------


def test_bare_out_lands_in_figdir_explicit_path_does_not(tmp_path, monkeypatch):
    """A bare filename is a convenience; a path is an instruction.

    ``$FIGDIR`` must never relocate an explicit path, or figview.sh (which
    exports FIGDIR) would silently hijack a caller's ``--out /some/where.pdf``.
    """
    from gps_plot.detrend_workbench import default_figdir, resolve_out

    monkeypatch.setenv("FIGDIR", str(tmp_path))
    assert default_figdir() == tmp_path
    assert resolve_out(None, "SELF") == tmp_path / "SELF-detrend-workbench.pdf"
    assert resolve_out("SELF-iter1.pdf", "SELF") == tmp_path / "SELF-iter1.pdf"
    # explicit paths survive an exported FIGDIR untouched
    assert resolve_out("/abs/x.pdf", "SELF") == Path("/abs/x.pdf")
    assert resolve_out("~/y.pdf", "SELF") == Path.home() / "y.pdf"
    assert resolve_out("sub/z.pdf", "SELF") == Path("sub/z.pdf")


def test_figdir_falls_back_to_the_checkout_then_cwd(monkeypatch):
    """Unset FIGDIR resolves to the source checkout's gitignored tmp-figdir.

    Guarded on ``pyproject.toml`` beside the package root: a wheel install
    resolves into site-packages, which is the wrong place to write a figure and
    may not even be writable — so that case must degrade to CWD.
    """
    import gps_plot.detrend_workbench as wb

    monkeypatch.delenv("FIGDIR", raising=False)
    got = wb.default_figdir()
    root = Path(wb.__file__).resolve().parents[2]
    if (root / "pyproject.toml").is_file():
        assert got == root / wb.SCRATCH_FIGDIR
        assert got.name == "tmp-figdir", "must match figview.sh's default"
    else:
        assert got == Path.cwd()


# ---------------------------------------------------------------------------
# --segment: a union fit domain from the command line
# ---------------------------------------------------------------------------


def test_segments_estimate_the_offset_a_single_window_cannot(tmp_path):
    """The whole point of the feature, on the station that motivated it.

    SELF carries the 2008-05-29 Ölfus M6.3: a coseismic offset with
    transients either side. A contiguous window either includes the
    transients (biasing the rate) or starts after the quake (never
    estimating the offset at all). Excising the transient leaves flanks on
    both sides, which is exactly what makes the step estimable -- the step
    epoch itself lies INSIDE the excision.
    """
    from gps_plot.detrend_workbench import build_record

    record, *_ = build_record(
        STA_SELF,
        tot_dir=str(TOT),
        segments=((2002.1, 2008.35), (2008.7, 2019.5)),
        max_gap_years=1.5,
    )
    assert record["segments"] == [[2002.1, 2008.35], [2008.7, 2019.5]]
    assert record["window"] == [2002.1, 2019.5], "window is the hull"
    assert len(record["segment_gaps"]) == 1

    step = record["step_epochs"][0]
    assert 2008.35 < step < 2008.7, "the step is inside the excised gap"
    amps = [
        c["params"][record["param_names"].index("step_amp_1")]
        for c in record["components"]
    ]
    # steps.csv annotates this event as -150.8 mm north; the fit must find it
    assert amps[0] == pytest.approx(-150.8, abs=1.0)


def test_cli_segment_flag_reaches_the_fit(tmp_path, monkeypatch):
    from gps_plot import detrend_workbench as wb

    seen = {}
    original = wb.build_record

    def _spy(*args, **kwargs):
        seen.update(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(wb, "build_record", _spy)
    rc = wb.main(
        [
            STA_SELF,
            "--tot-dir",
            str(TOT),
            "--no-tos",
            "--segment",
            "2002.1:2008.35",
            "--segment",
            "2008.7:2019.5",
            "--max-gap-years",
            "1.5",
            "--out",
            str(tmp_path / "seg.pdf"),
        ]
    )
    assert rc == 0
    assert seen["segments"] == ((2002.1, 2008.35), (2008.7, 2019.5))


def test_cli_refuses_segment_mixed_with_window_flags(tmp_path):
    """Two ways to say which epochs are fitted; letting one win silently
    would change stored science without saying so."""
    from gps_plot.detrend_workbench import main

    with pytest.raises(SystemExit, match="cannot be combined"):
        main([STA_SELF, "--segment", "2002.1:2008.35", "--window-end", "2019.5"])


@pytest.mark.parametrize("spec", ["2002.1-2008.35", "2002.1:2008.35:x", "a:b"])
def test_cli_rejects_a_malformed_segment(spec):
    from gps_plot.detrend_workbench import main

    with pytest.raises(SystemExit):
        main([STA_SELF, "--segment", spec])


def test_open_bounds_are_expressible_from_the_cli(tmp_path, monkeypatch):
    """An empty side means open, matching --window-start/--window-end."""
    from gps_plot import detrend_workbench as wb

    seen = {}
    original = wb.build_record
    monkeypatch.setattr(
        wb, "build_record", lambda *a, **k: (seen.update(k), original(*a, **k))[1]
    )
    rc = wb.main(
        [
            STA_SELF,
            "--tot-dir",
            str(TOT),
            "--no-tos",
            "--segment",
            ":2008.35",
            "--segment",
            "2008.7:",
            "--max-gap-years",
            "1.5",
            "--out",
            str(tmp_path / "open.pdf"),
        ]
    )
    assert rc == 0
    assert seen["segments"] == ((None, 2008.35), (2008.7, None))


def test_commit_stores_model_and_terms_where_the_batch_will_find_them(tmp_path):
    """`--commit` must store the FIT-time decisions, not just the record.

    The batch RECOMPUTES the record, so a model or a transient living only
    inside it is invisible to `gps-estimate-detrend` — exactly the hole the
    stage plans were in until e6dd887. This is the write half of
    geo_dataread's `detrend.estimation.models` block.
    """
    import yaml

    from geo_dataread.analysis_yaml import (
        StationModel,
        read_station_models,
        write_station_model,
    )

    path = tmp_path / "analysis.yaml"
    # what `--commit` assembles from the parsed flags
    entry = StationModel(model="periodic", terms=("log@2008.4085,tau=1.0",))
    write_station_model(path, "SELF", entry)
    assert read_station_models(path) == {"SELF": entry}

    # and it must coexist with a stage plan for the same station in one file
    from geo_dataread.stage_plan import (
        build_stage_plan,
        read_stage_plans,
        write_stage_plan,
    )

    plan = build_stage_plan(["fit:periodic,step"], [])
    write_stage_plan(path, "SELF", plan)
    doc = yaml.safe_load(path.read_text())["detrend"]["estimation"]
    assert set(doc) == {"models", "stage_plans"}
    assert read_station_models(path) == {"SELF": entry}
    assert read_stage_plans(path) == {"SELF": plan}


def test_commit_writes_model_terms_only_when_the_operator_set_them(tmp_path):
    """No flags, no entry: the 37 deployed stations must gain nothing."""
    import inspect

    from gps_plot import detrend_workbench as wb

    src = inspect.getsource(wb.main)
    assert "if args.model is not None or args.term:" in src
    assert "write_station_model(yaml_path, sta, entry)" in src
