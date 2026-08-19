"""The picker's contract: it writes the grammar the CLI parses.

The round trip is the whole point — anything the picker emits must parse back
into the plan it came from, or "one grammar, two producers" is a slogan.
"""

from __future__ import annotations

import shlex

import pytest

from geo_dataread.stage_plan import build_stage_plan
from gps_plot.detrend_picker import render_command


@pytest.fixture(autouse=True)
def _isolated_session_store(tmp_path, monkeypatch):
    """Keep the suite out of the operator's real session store.

    ``PickerWindow.__init__`` auto-loads a session and ``save_session``
    writes one, both under ``$XDG_STATE_HOME``.  Unisolated, this suite
    OVERWROTE and then unlinked a real ``sessions/SELF.json`` in its
    teardown -- destroying live curation state -- and a real session for a
    tested station would silently change what the assertions here run
    against (restored picks are indistinguishable from defaults).
    """
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))


def _roundtrip(stages: list[str], holds: list[str]):
    plan = build_stage_plan(stages, holds)
    cmd = shlex.split(render_command("SELF", plan))
    again_stages = [cmd[i + 1] for i, a in enumerate(cmd) if a == "--stage"]
    again_holds = [cmd[i + 1] for i, a in enumerate(cmd) if a == "--hold"]
    return plan, build_stage_plan(again_stages, again_holds), cmd


@pytest.mark.parametrize(
    "stages, holds",
    [
        (
            ["clean:secular,periodic@2001.6:2019.5", "long:secular"],
            ["long:periodic=stage:clean"],
        ),
        (
            ["clean:secular,periodic@2001.6:2008.0", "long:secular,step"],
            ["long:periodic=stage:clean"],
        ),
        (["fit:periodic"], ["secular=donor:OLAC"]),
        (["a:secular@:2008.35;2008.7:"], []),
        (["a:secular@:"], []),
    ],
)
def test_emitted_command_parses_back_to_the_same_plan(stages, holds) -> None:
    plan, again, _ = _roundtrip(stages, holds)
    assert again == plan


def test_command_targets_the_workbench(self=None) -> None:
    plan = build_stage_plan(["clean:secular"], [])
    cmd = shlex.split(render_command("SELF", plan))
    assert cmd[0] == "gps-detrend-workbench"
    assert cmd[1] == "SELF"


def test_stage_prefix_only_when_ambiguous() -> None:
    # One stage: the borrow one-liner stays short.
    _, _, cmd = _roundtrip(["fit:periodic"], ["secular=donor:OLAC"])
    assert "secular=donor:OLAC" in cmd
    # Two stages: the prefix is emitted, because without it the CLI refuses.
    _, _, cmd2 = _roundtrip(
        ["clean:secular,periodic", "long:secular"], ["long:periodic=stage:clean"]
    )
    assert "long:periodic=stage:clean" in cmd2


def test_inherit_is_emitted_as_no_at_sign() -> None:
    # segments=None must not become '@:', which would silently change
    # "inherit the caller's domain" into "the full span".
    _, _, cmd = _roundtrip(["long:secular"], [])
    assert "long:secular" in cmd
    assert not any(a.startswith("long:secular@") for a in cmd)


def test_extra_flags_are_appended() -> None:
    plan = build_stage_plan(["a:secular"], [])
    cmd = render_command("SELF", plan, ["--max-gap-years", "1.5"])
    assert cmd.endswith("--max-gap-years 1.5")


def test_headless_backend_refuses_rather_than_hanging() -> None:
    import matplotlib

    from gps_plot.detrend_picker import pick_stage_plan

    old = matplotlib.get_backend()
    matplotlib.use("Agg", force=True)
    try:
        with pytest.raises(RuntimeError, match="cannot show a window"):
            pick_stage_plan("SELF", [2000.0], [[0.0]], None, {"model": "linear"})
    finally:
        matplotlib.use(old, force=True)


class TestMarimoNotebook:
    """The notebook is a real artifact, so it gets real checks.

    marimo enforces that each name is defined in exactly ONE cell — that
    strictness is what makes its dataflow analysable, and it caught a genuine
    duplicate-import bug while this was being written. ``marimo export`` runs
    that analysis, so exporting IS the structural test.
    """

    NOTEBOOK = "notebooks/detrend_picker.py"

    def test_notebook_exists_and_is_a_marimo_app(self) -> None:
        from pathlib import Path

        src = Path(self.NOTEBOOK)
        if not src.is_file():
            pytest.skip("notebook not present in this checkout")
        text = src.read_text()
        assert "marimo.App(" in text
        assert "app.run()" in text

    def test_dataflow_analyses_cleanly(self) -> None:
        # Catches duplicate definitions and cycles across cells.
        import shutil
        import subprocess
        from pathlib import Path

        if not Path(self.NOTEBOOK).is_file():
            pytest.skip("notebook not present in this checkout")
        if shutil.which("marimo") is None:
            pytest.skip("marimo not installed (dev group)")
        r = subprocess.run(
            ["marimo", "export", "script", self.NOTEBOOK, "-o", "/dev/null"],
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert r.returncode == 0, r.stderr

    def test_notebook_never_commits(self) -> None:
        # The contract: it emits the command, the workbench stores. A --commit
        # in here would be a second path to stored science.
        from pathlib import Path

        if not Path(self.NOTEBOOK).is_file():
            pytest.skip("notebook not present in this checkout")
        text = Path(self.NOTEBOOK).read_text()
        assert "commit_record" not in text
        assert "write_stage_plan" not in text


class TestQtPicker:
    """The Qt picker is a GUI, so test what can be tested without a screen."""

    def test_import_guard_explains_itself(self) -> None:
        from gps_plot.detrend_picker_qt import _require_qt

        pg, qtw = _require_qt()  # installed in the dev group
        assert hasattr(pg, "LinearRegionItem")
        assert hasattr(qtw, "QApplication")

    def test_picks_reach_the_fit_and_the_command(self) -> None:
        # Drives the real widgets offscreen: move a pick, assert the fit and
        # the emitted command both follow. This is the whole contract.
        import os

        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        import numpy as np

        import geo_dataread.gps_read as gpsr
        from geo_dataread.detrend_estimate import FitDefaults, resolve_fit_settings
        from gps_plot.detrend_picker_qt import PickerWindow, _require_qt

        _pg, qtw = _require_qt()
        _app = qtw.QApplication.instance() or qtw.QApplication([])

        try:
            yearf, data, sigma, _ = gpsr.getData(
                "NYLA", ref="plate", Dir=None, tType="TOT", uncert=10
            )
        except Exception:  # pragma: no cover - station data not present
            pytest.skip("NYLA data not available in this environment")
        if yearf is None or len(yearf) == 0:  # pragma: no cover
            pytest.skip("NYLA data not available")

        settings = resolve_fit_settings("NYLA", None, FitDefaults(max_gap_years=3.0))
        w = PickerWindow(
            "NYLA",
            np.asarray(yearf, float),
            np.atleast_2d(np.asarray(data, float)),
            np.atleast_2d(np.asarray(sigma, float)),
            settings,
            max_gap_years=3.0,
            uncert=10.0,
        )

        # Full domain on NYLA aborts, so there is no record and the command
        # carries no --segment.
        assert "--segment" not in w.command.text()

        # Drag the domain: the fit must follow and the flag must appear.
        for r in w.domain_regions:
            r.blockSignals(True)
            r.setRegion((w.span[0], 2020.0))
            r.blockSignals(False)
        w.refit()
        assert w.record is not None
        assert max(w.record["rms"]) < 10.0  # was 65 mm across the unrest
        assert "--segment" in w.command.text()

        # Double-click equivalent: declaring the 2023-11-10 dike as a step.
        for r in w.domain_regions:
            r.blockSignals(True)
            r.setRegion(w.span)
            r.blockSignals(False)
        w._add_step(2023.862)
        assert "--step 2023.862" in w.command.text()
        assert w.record is not None
        assert w.record["step_epochs"] == [2023.862]

        # A transient reaches the model and the record's terms.
        w._clear_steps()
        w.cb_term.setChecked(True)
        for ln in w.onset_lines:
            ln.blockSignals(True)
            ln.setValue(2018.0)
            ln.blockSignals(False)
        w.refit()
        assert "--term log@2018.0,tau=2.0" in w.command.text()
        assert w.record is not None
        assert w.record["record_version"] == 2

    def test_emitted_command_parses_back(self) -> None:
        # Same contract as the other producers: whatever it prints must be
        # re-parseable by the grammar, or "one grammar" is a slogan.
        import shlex

        from geo_dataread.stage_plan import build_stage_plan
        from geo_dataread.term_spec import parse_term_spec

        cmd = (
            "gps-detrend-workbench NYLA --term log@2018.0,tau=2.0 "
            "--stage clean:secular,periodic@2006.5767:2018.0 "
            "--stage long:secular,transient --hold long:periodic=stage:clean "
            "--segment 2006.5767:2020.0 --max-gap-years 3.0"
        )
        c = shlex.split(cmd)
        parse_term_spec(c[c.index("--term") + 1])
        build_stage_plan(
            [c[i + 1] for i, a in enumerate(c) if a == "--stage"],
            [c[i + 1] for i, a in enumerate(c) if a == "--hold"],
        )


class TestQtPickerBorrowedFeatures:
    """Three features taken from the published pickers, each attributed.

    SARI (Santamaria-Gomez 2019, doi:10.1007/s10291-019-0846-y) — metadata
    fusion: flag candidate discontinuities from equipment history.
    TSAnalyzer (Wu et al. 2017, doi:10.1007/s10291-017-0637-2) — a reloadable
    pick file. Both carry a Lomb-Scargle residual periodogram.
    """

    @staticmethod
    def _window(sta="SELF", gap=1.5, catalog_window=None):
        import os

        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        import numpy as np

        import geo_dataread.gps_read as gpsr
        from geo_dataread.detrend_estimate import FitDefaults, resolve_fit_settings
        from gps_plot.detrend_picker_qt import PickerWindow, _require_qt

        _pg, qtw = _require_qt()
        qtw.QApplication.instance() or qtw.QApplication([])
        try:
            yearf, data, sigma, _ = gpsr.getData(
                sta, ref="plate", Dir=None, tType="TOT", uncert=10
            )
        except Exception:  # pragma: no cover
            pytest.skip(f"{sta} data not available")
        if yearf is None or len(yearf) == 0:  # pragma: no cover
            pytest.skip(f"{sta} data not available")
        catalog = None
        if catalog_window is not None:
            from geo_dataread.detrend_estimate import FitCatalogRow

            catalog = {
                sta: FitCatalogRow(
                    window_start=catalog_window[0], window_end=catalog_window[1]
                )
            }
        settings = resolve_fit_settings(
            sta, catalog, FitDefaults(max_gap_years=gap), catalog_source="test.csv"
        )
        return PickerWindow(
            sta,
            np.asarray(yearf, float),
            np.atleast_2d(np.asarray(data, float)),
            np.atleast_2d(np.asarray(sigma, float)),
            settings,
            max_gap_years=gap,
            uncert=10.0,
        )

    def test_declared_events_are_drawn(self) -> None:
        # SELF's 2008 Olfus M6.3 is in steps.csv, so it must be marked BEFORE
        # the operator decides whether to declare a step there.
        w = self._window()
        assert w.declared_events
        assert any(abs(e - 2008.4085) < 0.01 for e, _lbl in w.declared_events)

    def test_residual_spectrum_is_computed(self) -> None:
        import numpy as np

        w = self._window()
        f, p = w.spec_curves[0].getData()
        assert f is not None and len(f) > 100
        # SELF's seasonal is well modelled, so annual power must be small.
        annual = float(p[int(np.argmin(np.abs(f - 1.0)))])
        assert annual < 0.2

    def test_session_round_trips(self) -> None:
        w = self._window()
        w._add_step(2008.4085)
        for r in w.domain_regions:
            r.blockSignals(True)
            r.setRegion((w.span[0], 2019.5))
            r.blockSignals(False)
        w.cb_term.setChecked(True)
        w.tau.setValue(1.5)
        w.refit()
        w.save_session()
        path = w._session_path()
        try:
            assert path.exists()
            w2 = self._window()
            assert [round(g[0].value(), 4) for g in w2.step_lines] == [2008.4085]
            assert round(w2.domain_regions[0].getRegion()[1], 4) == 2019.5
            assert w2.cb_term.isChecked()
            assert w2.tau.value() == pytest.approx(1.5)
        finally:
            path.unlink(missing_ok=True)

    def test_session_is_not_provenance(self) -> None:
        # The emitted COMMAND is the record; the session is convenience. If
        # this ever stopped being true there would be two paths to stored
        # science, which is exactly what the CLI-first design avoids.
        import json

        w = self._window()
        w.save_session()
        path = w._session_path()
        try:
            assert "convenience" in json.loads(path.read_text())["note"]
        finally:
            path.unlink(missing_ok=True)

    def test_a_picked_step_does_not_erase_the_declared_ones(self) -> None:
        """The picker's fit and the command it emits must be the SAME fit.

        Regression (2026-08-09): ``_current`` did
        ``dataclasses.replace(settings, steps=picked)``, which REPLACED the
        station's declared steps, while ``--step`` on the workbench merges
        with ``steps.csv`` and the fit catalog. So the figure the operator
        judged and the fit their copied command reproduces diverged.

        SELF is the sharp case -- dropping its declared 2008 Olfus
        coseismic makes the excess-candidate rule abort, so the picker
        showed a stale curve with NO record while the emitted command
        fitted cleanly. On a milder station the divergence is silent:
        different step set, different inliers, different rate.
        """
        w = self._window()
        declared = [e for e, _lbl in w.declared_events]
        assert any(abs(e - 2008.4085) < 0.01 for e in declared)

        w._add_step(2015.5)
        assert w.record is not None, "the declared step is missing -> abort"
        # The picked step is emitted; the declared one is the workbench's
        # floor and must NOT be, or it would be declared twice.
        assert "--step 2015.5" in w.command.text()
        assert "2008.4085" not in w.command.text()
        # ...but it must be in the FIT.
        fitted = [round(float(v), 4) for v in w.record["step_epochs"]]
        assert 2008.4085 in fitted and 2015.5 in fitted

        # And the equivalence itself: the settings the picker fits with are
        # the settings the emitted command resolves to.
        from gps_plot.detrend_workbench import _override_settings, estimate_record

        cli = _override_settings(
            w.base_settings,
            "SELF",
            segments=tuple(
                tuple(round(v, 4) for v in r.getRegion()) for r in w.domain_regions[:1]
            ),
            steps=(2015.5,),
        )
        est = estimate_record("SELF", w.yearf, w.data, w.sigma, settings=cli)
        assert est is not None
        assert [round(float(v), 4) for v in est.record["step_epochs"]] == fitted
        assert est.record["n_rejected"] == w.record["n_rejected"]

    def test_the_header_names_the_fitted_steps_not_the_clicked_ones(self) -> None:
        # The merge is invisible on the plot -- a declared step draws no
        # pick line -- so the header is where it has to be legible.
        w = self._window()
        w._add_step(2015.5)
        assert "picked [2015.5]" in w.header.text()
        assert "fitting [2008.4085, 2015.5]" in w.header.text()

    def test_the_domain_region_opens_on_the_catalog_window(self) -> None:
        """The picker must not assert a fit domain the catalog contradicts.

        Regression (2026-08-16), the picked-step bug one lever over: the
        region opened on the DATA SPAN, and ``--segment`` was emitted only
        when the region differed from that span. So on a station whose
        ``fit_windows.csv`` row declares a narrow pre-unrest window, an
        operator who touched nothing got a figure fitted over everything and
        a command that emitted no ``--segment`` at all — reproducing the
        catalog window instead. Two different fits, no indication.

        Latent when found: the deployed catalog has no window rows yet.
        """
        w = self._window(catalog_window=(2009.0, 2019.5))
        lo, hi = (round(v, 4) for v in w.domain_regions[0].getRegion())
        assert (lo, hi) == (2009.0, 2019.5)
        assert lo > w.span[0] and hi < w.span[1], "the window must actually bite"

        # untouched -> defer to the catalog entirely: no flag, and the fit is
        # the one `gps-detrend-workbench SELF` would produce with no flags
        assert "--segment" not in w.command.text()
        settings, _extra, _terms, _stages = w._current()
        assert settings.segments == w.base_settings.segments

        from gps_plot.detrend_workbench import estimate_record

        cli = estimate_record(
            "SELF", w.yearf, w.data, w.sigma, settings=w.base_settings
        )
        assert cli is not None and w.record is not None
        assert cli.record["window"] == w.record["window"]
        assert cli.record["n_epochs"] == w.record["n_epochs"]

    def test_moving_the_region_emits_the_segment_it_fits(self) -> None:
        w = self._window(catalog_window=(2009.0, 2019.5))
        for r in w.domain_regions:
            r.blockSignals(True)
            r.setRegion((2012.0, 2018.0))
            r.blockSignals(False)
        w.refit()
        assert "--segment 2012.0:2018.0" in w.command.text()
        assert w.record is not None
        assert w.record["window"] == [2012.0, 2018.0]

    def test_reset_returns_to_the_declared_domain_not_the_data_span(self) -> None:
        w = self._window(catalog_window=(2009.0, 2019.5))
        for r in w.domain_regions:
            r.blockSignals(True)
            r.setRegion((2012.0, 2018.0))
            r.blockSignals(False)
        w.refit()
        w._reset_domain()
        assert tuple(round(v, 4) for v in w.domain_regions[0].getRegion()) == (
            2009.0,
            2019.5,
        )
        assert "--segment" not in w.command.text()

    def test_no_catalog_window_still_opens_on_the_full_span(self) -> None:
        """The 37 deployed stations have no window row — nothing moves for them."""
        w = self._window()
        assert tuple(round(v, 4) for v in w.domain_regions[0].getRegion()) == tuple(
            round(v, 4) for v in w.span
        )
        assert "--segment" not in w.command.text()

    def test_the_emitted_command_carries_every_run_parameter(self) -> None:
        """A flag that changes the data but not the command is a divergence.

        Three, all in the same family as the picked-step bug:
        ``--tot-dir`` was never emitted, so a picker run against a
        non-default TOT directory emitted a command reading the DEFAULT one
        — a different series entirely. ``--uncert`` was a float here and an
        int in the workbench, so ``str(12.0)`` produced a command
        ``gps-detrend-workbench`` REFUSES to parse. And ``--provisional-days``
        was never emitted at all.
        """
        w = self._window()
        w.tot_dir = "/data/alt-tot"
        w.uncert = 12
        w.provisional_days = 30.0
        w.refit()
        cmd = w.command.text()
        assert "--tot-dir /data/alt-tot" in cmd
        assert "--uncert 12" in cmd and "--uncert 12.0" not in cmd
        assert "--provisional-days 30.0" in cmd

        # and it must actually PARSE on the workbench side
        import shlex

        from gps_plot.detrend_workbench import _build_parser

        argv = shlex.split(cmd)
        assert argv[0] == "gps-detrend-workbench"
        ns = _build_parser().parse_args(argv[1:])
        assert ns.uncert == 12 and ns.tot_dir == "/data/alt-tot"
        assert ns.provisional_days == 30.0

    def test_provisional_days_reaches_the_window(self) -> None:
        """The flag was parsed and then dropped on the floor.

        `main` never passed it to `PickerWindow`, so the gold lane always used
        geo_dataread's default and the header said so regardless of what was
        typed — a flag that documents a behaviour it does not have.
        """
        import inspect

        from gps_plot import detrend_picker_qt as dp

        src = inspect.getsource(dp.main)
        assert "provisional_days=args.provisional_days" in src
        assert "tot_dir=args.tot_dir" in src

        w = self._window()
        w.provisional_days = 0.0
        w.refit()
        assert "prov 0.0 d" in w.header.text()

    @pytest.mark.parametrize(
        "payload",
        [
            "[]",
            '{"domain": "2009:2019"}',
            '{"domain": [2009.0]}',
            '{"domain": [2019.0, 2009.0]}',
            '{"steps": ["soon"]}',
            '{"steps": 2015.5}',
            '{"stage": [1, 2]}',
            '{"term": {"on": true, "epoch": "later"}}',
            "not json at all",
        ],
    )
    def test_a_corrupt_session_never_takes_the_window_down(self, payload: str) -> None:
        """`load_session` runs at LAUNCH, so nothing in it may raise.

        Only a JSON syntax error was caught. A structurally wrong session —
        a list at top level, a two-element domain that is not, a step epoch
        that is not a number — crashed the application before the window
        appeared, leaving no way in to clear the file that was killing it.
        """
        w = self._window()
        w._session_path().parent.mkdir(parents=True, exist_ok=True)
        w._session_path().write_text(payload)
        assert w.load_session() is False
        assert "NOT restored" in w.summary.toPlainText()
        # and nothing half-applied: the domain is still the declared one
        assert tuple(round(v, 4) for v in w.domain_regions[0].getRegion()) == tuple(
            round(v, 4) for v in w.default_domain
        )

    def test_a_good_session_still_round_trips_after_the_hardening(self) -> None:
        w = self._window()
        for r in w.domain_regions:
            r.blockSignals(True)
            r.setRegion((2005.0, 2015.0))
            r.blockSignals(False)
        w._add_step(2011.25)
        w.save_session()
        w2 = self._window()
        assert w2.load_session() is True
        assert tuple(round(v, 4) for v in w2.domain_regions[0].getRegion()) == (
            2005.0,
            2015.0,
        )
        assert [round(g[0].value(), 4) for g in w2.step_lines] == [2011.25]


class TestStageLaneKnowsTheDeclaredSteps:
    """The long stage must free `step` when the FIT carries a step column.

    Fifth violation of the picker's one invariant, and the same shape as the
    four before it: a second place reasoning about steps that knew only
    about the PICKED ones. ``refit`` built the final stage's free-group list
    from ``self.step_lines``, but ``steps.csv`` is a FLOOR that
    ``_override_settings`` merges in -- so on a station with a declared step
    an untouched stage plan freed only ``secular`` while the fit still
    carried ``step_amp_1``, and every fit was refused with "never estimated
    and not held in the final stage".

    Both sides refused identically, so the emitted command still reproduced
    the figure -- but the stage lane was unusable on exactly the two
    stations in ``steps.csv`` (SELF, HOFN), and nothing on screen said that
    re-declaring the already-declared step was the way out.
    """

    @staticmethod
    def _staged(sta: str):
        w = TestQtPickerBorrowedFeatures._window(sta=sta)
        w.cb_stage.setChecked(True)
        w.stage_regions[0].setRegion(
            (round(w.span[0] + 0.5, 4), round(w.span[0] + 6.0, 4))
        )
        w.refit()
        return w

    def test_a_declared_step_frees_the_step_group(self) -> None:
        w = self._staged("SELF")
        assert w.record is not None, (
            "the long stage did not free `step`, so the declared 2008 "
            f"coseismic is estimated nowhere: {w.summary.toPlainText()[:200]}"
        )
        cmd = w.command.text()
        assert "--stage long:secular,step" in cmd
        # The declared step is fitted without ever being picked or emitted.
        assert not w.step_lines
        assert "--step" not in cmd
        assert any(abs(float(v) - 2008.4085) < 0.01 for v in w.record["step_epochs"])

    def test_a_station_with_no_declared_step_does_not_free_it(self) -> None:
        """The fix has to be CONDITIONAL, or every plan frees a dead group."""
        from gps_plot.detrend_workbench import _declared_step_epochs

        w = self._staged("RHOF")
        if _declared_step_epochs("RHOF", w.base_settings.steps):  # pragma: no cover
            pytest.skip("RHOF gained a declared step; pick another station")
        cmd = w.command.text()
        assert "--stage long:secular " in cmd + " "
        assert "step" not in cmd.split("--stage long:")[1].split(" ")[0]

    def test_the_emitted_stage_plan_reproduces_the_picker_fit(self) -> None:
        """Round-trip the NEW emission path, not just its exit code."""
        import shlex

        from geo_dataread.stage_plan import build_stage_plan
        from gps_plot.detrend_workbench import (
            _build_parser,
            _override_settings,
            estimate_record,
        )

        w = self._staged("SELF")
        assert w.record is not None
        ns = _build_parser().parse_args(shlex.split(w.command.text())[1:])
        plan = build_stage_plan(ns.stage, ns.hold)
        cli = _override_settings(w.base_settings, "SELF", quiet=True)
        est = estimate_record(
            "SELF", w.yearf, w.data, w.sigma, settings=cli, stage_plan=plan
        )
        assert est is not None
        assert est.record["step_epochs"] == w.record["step_epochs"]
        assert est.record["n_rejected"] == w.record["n_rejected"]
        for a, b in zip(est.record["components"], w.record["components"]):
            assert a["params"] == pytest.approx(b["params"])


class TestBothPickersEmitTheSameRunFlags:
    """`run_flags` exists because two pickers forgot the same flag.

    ``--tot-dir`` was read and never emitted in the Qt picker (fixed
    2026-08-16) and independently in the marimo one, and ``--uncert`` was a
    float against the workbench's ``type=int`` in both. One assembly site is
    the only thing that makes a third picker safe.
    """

    def test_uncert_is_spelled_the_way_the_workbench_parses_it(self) -> None:
        from gps_plot.detrend_workbench import _build_parser, run_flags

        flags = run_flags(uncert=12)
        assert flags == ["--uncert", "12"], flags
        # the float spelling is what argparse refused
        with pytest.raises(SystemExit):
            _build_parser().parse_args(["SELF", "--uncert", "12.0"])
        ns = _build_parser().parse_args(["SELF", *flags])
        assert ns.uncert == 12

    def test_the_default_screen_is_omitted_not_restated(self) -> None:
        from gps_plot.detrend_workbench import (
            WORKBENCH_UNCERT_DEFAULT,
            _build_parser,
            run_flags,
        )

        assert run_flags(uncert=WORKBENCH_UNCERT_DEFAULT) == []
        # omission is only correct because the workbench defaults to the same
        assert _build_parser().parse_args(["SELF"]).uncert == WORKBENCH_UNCERT_DEFAULT

    def test_tot_dir_reaches_the_command(self) -> None:
        from gps_plot.detrend_workbench import _build_parser, run_flags

        flags = run_flags(tot_dir="/data/alt-tot", max_gap_years=1.5)
        assert flags[:2] == ["--tot-dir", "/data/alt-tot"]
        ns = _build_parser().parse_args(["SELF", *flags])
        assert ns.tot_dir == "/data/alt-tot" and ns.max_gap_years == 1.5

    def test_both_pickers_declare_uncert_as_the_same_type(self) -> None:
        """A float here and an int there is how the spelling diverged."""
        from gps_plot import detrend_picker, detrend_picker_qt
        from gps_plot.detrend_workbench import WORKBENCH_UNCERT_DEFAULT

        for mod in (detrend_picker, detrend_picker_qt):
            assert mod.WORKBENCH_UNCERT_DEFAULT is WORKBENCH_UNCERT_DEFAULT

    def test_the_marimo_picker_parses_uncert_as_an_int(self) -> None:
        import subprocess
        import sys

        out = subprocess.run(
            [
                sys.executable,
                "-c",
                "import gps_plot.detrend_picker as m; m.main()",
                "SELF",
                "--uncert",
                "12.5",
            ],
            capture_output=True,
            text=True,
        )
        assert out.returncode != 0
        assert "invalid int value" in out.stderr

    def test_a_non_integral_screen_raises_rather_than_rounding(self) -> None:
        """Rounding would be the fixed bug wearing a disguise.

        `--uncert 12.5` used to emit a command argparse REFUSES -- loud, and
        the operator knew. Silently emitting `--uncert 12` instead would give
        a command that parses, runs, and screens a different set of epochs
        than the figure was fitted on. `PickerWindow(uncert=...)` is only an
        annotation, so a float can still reach here.
        """
        from gps_plot.detrend_workbench import run_flags

        with pytest.raises(ValueError, match="not integral"):
            run_flags(uncert=12.5)
        # an integral float is the same screen, and stays spellable
        assert run_flags(uncert=12.0) == ["--uncert", "12"]


class TestSplitLayout:
    """Plots left, controls right, command full width along the bottom.

    Everything used to stack vertically, which spent height -- the scarce
    dimension, with three component panels plus a periodogram -- on a one-line
    control strip that had itself run out of width.
    """

    @staticmethod
    def _window():
        return TestQtPickerBorrowedFeatures._window()

    def test_plots_are_left_of_the_controls(self) -> None:
        w = self._window()
        assert w.split.orientation() == w.pg.QtCore.Qt.Horizontal
        assert w.split.count() == 2
        assert w.split.widget(0) is w.glw, "the plots must be the left pane"
        # the control column carries the controls AND the record
        right = w.split.widget(1)
        assert w.summary in right.findChildren(type(w.summary))

    def test_the_command_spans_the_full_width(self) -> None:
        """It is the window's output and runs past 200 chars on a staged fit.

        Putting it in the right column would make the one thing you copy the
        least readable widget on screen.
        """
        w = self._window()
        right = w.split.widget(1)
        assert w.command not in right.findChildren(type(w.command))
        assert w.command.parent() is w.split.parent()

    def test_every_control_survived_the_move(self) -> None:
        """Re-parenting must not drop a widget or its signal."""
        w = self._window()
        for name in ("cb_stage", "cb_term", "kind", "tau"):
            assert hasattr(w, name), f"{name} lost in the layout change"
        # and they still drive a refit: toggling the term changes the command
        before = w.command.text()
        w.cb_term.setChecked(True)
        assert w.command.text() != before
        assert "--term" in w.command.text()

    def test_rms_is_rounded_for_the_narrow_column(self) -> None:
        """Raw floats wrapped over three lines and buried the comparison."""
        w = self._window()
        w.refit()
        rms_line = [
            ln for ln in w.summary.toPlainText().splitlines() if ln.startswith("rms")
        ]
        assert rms_line, w.summary.toPlainText()[:200]
        # 2 dp, matching the PDF's summarise()
        assert "0000" not in rms_line[0] and len(rms_line[0]) < 60, rms_line[0]
