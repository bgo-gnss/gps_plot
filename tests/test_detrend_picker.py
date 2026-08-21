"""The picker's contract: it writes the grammar the CLI parses.

The round trip is the whole point — anything the picker emits must parse back
into the plan it came from, or "one grammar, two producers" is a slogan.
"""

from __future__ import annotations

import shlex

import pytest

from geo_dataread.stage_plan import build_stage_plan
from gps_plot.detrend_picker import render_command
from gps_plot.detrend_picker_qt import FIT_COLOR as FIT_COLOR_RGB


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
        settings, _extra, _terms, _stages, _holds = w._current()
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
        # `step` must be FREE in the long stage. The exact spelling moved when
        # the default became "hold the background" (secular is now held, so
        # long frees only step) -- what this asserts is the freedom, not the
        # literal, because the default is allowed to change and the freedom
        # is not.
        long_free = cmd.split("--stage long:")[1].split(" ")[0].split(",")
        assert "step" in long_free, cmd
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
        assert "step" not in cmd.split("--stage long:")[1].split(" ")[0]
        # RHOF has no step and no term, so holding the whole background leaves
        # the long stage with nothing to estimate and the plan is refused. The
        # command must still SAY what was asked for -- emitting the unstaged
        # spelling here made the window show a refusal while the command
        # described a different, perfectly fittable fit.
        assert w.record is None
        assert "--stage clean:secular,periodic@" in cmd
        assert "--hold long:secular=stage:clean" in cmd

    def test_a_refused_plan_is_still_emitted(self) -> None:
        """Violation eight, introduced while building slice 2 and caught here.

        When the stage plan would not build, `_command` fell through to the
        UNSTAGED spelling — so the window showed a refusal while the command
        described a different, perfectly fittable fit. On RHOF (no declared
        step, nothing left free once the whole background is held) that was
        `gps-detrend-workbench RHOF --max-gap-years 1.5`: copy it and you get
        the figure the picker had just refused to show.

        The command must say what was ASKED FOR, so the workbench refuses it
        for the same reason — which it does, with the same message.
        """
        from gps_plot.detrend_workbench import _build_parser

        w = self._staged("RHOF")
        assert w.record is None, "RHOF unexpectedly fittable; pick another case"
        cmd = w.command.text()
        assert "--stage" in cmd and "--hold" in cmd, cmd
        # and it is a command the workbench can at least PARSE, so its refusal
        # is the plan's refusal rather than a syntax error
        import shlex

        ns = _build_parser().parse_args(shlex.split(cmd)[1:])
        assert ns.stage and ns.hold

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


class TestRunParameterControls:
    """`uncert`, `max-gap` and `provisional-days` as live controls.

    Each writes the attribute `_command` already reads, so a control cannot
    move the figure without moving the emitted command with it — which is the
    only way to add a knob to this window without adding a sixth way to break
    its one invariant.
    """

    @staticmethod
    def _window(gap=1.5, catalog_gap=None):
        import os

        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        import numpy as np

        import geo_dataread.gps_read as gpsr
        from geo_dataread.detrend_estimate import (
            FitCatalogRow,
            FitDefaults,
            resolve_fit_settings,
        )
        from gps_plot.detrend_picker_qt import PickerWindow, _require_qt

        _pg, qtw = _require_qt()
        qtw.QApplication.instance() or qtw.QApplication([])
        try:
            yearf, data, sigma, _ = gpsr.getData(
                "SELF", ref="plate", Dir=None, tType="TOT", uncert=10
            )
        except Exception:  # pragma: no cover
            pytest.skip("SELF data not available")
        catalog = (
            {"SELF": FitCatalogRow(max_gap_years=catalog_gap)}
            if catalog_gap is not None
            else None
        )
        settings = resolve_fit_settings(
            "SELF", catalog, FitDefaults(), catalog_source="test.csv"
        )
        return PickerWindow(
            "SELF",
            np.asarray(yearf, float),
            np.atleast_2d(np.asarray(data, float)),
            np.atleast_2d(np.asarray(sigma, float)),
            settings,
            max_gap_years=gap,
            uncert=10,
        )

    def test_the_gap_flag_beats_the_catalog_row(self) -> None:
        """Regression: --max-gap-years used to be discarded by the catalog.

        `main` baked the flag into `FitDefaults` BEFORE `resolve_fit_settings`,
        which puts it BELOW the catalog row — so on a station whose
        fit_windows.csv sets its own gate, the picker fitted the catalog's
        value while emitting a command that fits the flag's. Measured on DYNG
        (catalog 1.0): `--max-gap-years 2.0` fitted at 1.0. The workbench
        applies it as an override AFTER resolving, and the picker must match.
        """
        w = self._window(gap=2.0, catalog_gap=1.0)
        assert w.base_settings.max_gap_years == 1.0, "catalog row not in play"

        # the settings the picker FITS with -- not a re-derivation of them,
        # which is what made the first version of this test pass against the
        # bug it was written for
        settings, _extra, _terms, _stages, _holds = w._current()
        assert settings.max_gap_years == 2.0, "the flag lost to the catalog again"
        # ... and the command the operator would copy says the same number
        assert "--max-gap-years 2.0" in w.command.text()

    def test_the_gap_control_moves_fit_and_command_together(self) -> None:
        w = self._window(gap=1.5)
        w.sp_gap.setValue(3.0)
        w._set_max_gap()
        assert w.max_gap_years == 3.0
        assert "--max-gap-years 3.0" in w.command.text()

    def test_uncert_rereads_the_series_rather_than_refitting(self) -> None:
        """It screens sigma at READ time, so it changes which epochs exist.

        A refit alone would move the command while the figure kept the old
        series — the same divergence one lever over.
        """
        w = self._window()
        before = w.yearf.size
        w.sp_uncert.setValue(8)
        w._set_uncert()
        assert w.yearf.size != before, "the series was not re-read"
        assert w.uncert == 8
        assert "--uncert 8" in w.command.text()
        assert f"{before} -> {w.yearf.size}" in w.summary.toPlainText()

    def test_a_failed_reread_keeps_the_series_and_says_so(self) -> None:
        """A picker that silently empties itself is worse than one that refuses."""
        w = self._window()
        before, kept = w.yearf.size, w.uncert
        w.sp_uncert.setValue(1)  # screens essentially everything away
        w._set_uncert()
        if w.yearf.size == before:  # the read was refused
            assert w.uncert == kept, "state and widget drifted apart"
            assert w.sp_uncert.value() == kept


class TestTermStateControls:
    """Three-state term controls, and the invariant across every combination.

    `--model` and the stage plan are orthogonal in the estimator: one decides
    which terms are in the design matrix, the other where each is estimated.
    That is what makes three states meaningful rather than two.
    """

    @staticmethod
    def _window():
        return TestQtPickerBorrowedFeatures._window()

    def test_the_states_compose_the_stored_model(self) -> None:
        from gps_plot.detrend_picker_qt import STATE_ABSENT, STATE_ESTIMATE

        w = self._window()
        for sec, per, expected in (
            (STATE_ESTIMATE, STATE_ESTIMATE, "lineperiodic"),
            (STATE_ESTIMATE, STATE_ABSENT, "linear"),
            (STATE_ABSENT, STATE_ESTIMATE, "periodic"),
        ):
            w.grp["secular"].setCurrentText(sec)
            w.grp["periodic"].setCurrentText(per)
            w.refit()
            assert w.model == expected, (sec, per)
            assert w.record is not None and w.record["model"] == expected
            # the default stays out of the command; anything else is emitted
            if expected == "lineperiodic":
                assert "--model" not in w.command.text()
            else:
                assert f"--model {expected}" in w.command.text()

    def test_both_absent_is_refused_not_crashed(self) -> None:
        """No --model value carries neither term, so there is nothing to emit."""
        from gps_plot.detrend_picker_qt import STATE_ABSENT

        w = self._window()
        w.refit()
        good = w.record
        assert good is not None
        w.grp["secular"].setCurrentText(STATE_ABSENT)
        w.grp["periodic"].setCurrentText(STATE_ABSENT)
        w.refit()
        assert w.model is None
        assert "refused" in w.summary.toPlainText().lower()

    def test_hold_is_disabled_until_there_is_a_window_to_hold_from(self) -> None:
        """Greyed, not hidden: removing the item would renumber the rest."""
        from gps_plot.detrend_picker_qt import GROUP_STATES, STATE_HOLD

        w = self._window()
        for name in ("secular", "periodic"):
            combo = w.grp[name]
            item = combo.model().item(GROUP_STATES.index(STATE_HOLD))
            assert not item.isEnabled(), f"{name}: hold reachable with no stage"
            assert combo.count() == len(GROUP_STATES), "an item was removed"

    def test_the_abort_fallback_is_shared_with_the_workbench(self) -> None:
        """Regression: the picker gave up where the CLI produced a figure.

        The leaf RETURNS None on an outlier abort (recoverable, retry S0) and
        RAISES on a failed gate (not). `build_record` handled that; the picker
        called `estimate_record` directly and did not — so the same command
        rendered fine from the CLI and showed 'no record' in the window.
        Reachable the moment `--model periodic` became selectable, since a
        periodic-only model leaves the trend in the residuals and the
        candidate fraction trips the abort.
        """
        from gps_plot.detrend_picker_qt import STATE_ABSENT

        w = self._window()
        w.grp["secular"].setCurrentText(STATE_ABSENT)  # -> --model periodic
        w.cb_term.setChecked(True)
        w.refit()
        assert w.record is not None, (
            "the picker refused a model the workbench fits: "
            + w.summary.toPlainText()[:120]
        )
        assert w.model == "periodic"


class TestPerGroupHold:
    """Slice 2: what carries from the clean window to the full span.

    This is the background model — whatever is HELD is estimated on the quiet
    window and extended across the span, and what stays free is estimated
    against that background. Removing it leaves residuals in which short-term
    deviations can be read.
    """

    @staticmethod
    def _staged():
        w = TestQtPickerBorrowedFeatures._window()
        w.stage_regions[0].setRegion((2009.5443, 2021.0394))
        w.cb_stage.setChecked(True)
        return w

    def test_staging_defaults_to_holding_the_background(self) -> None:
        """Regression: the plan hardcoded 'hold periodic' only.

        The trend was therefore re-estimated over the whole span including the
        unrest it was supposed to be a background FOR.
        """
        from gps_plot.detrend_picker_qt import STATE_HOLD

        w = self._staged()
        assert w.grp["secular"].currentText() == STATE_HOLD
        assert w.grp["periodic"].currentText() == STATE_HOLD
        cmd = w.command.text()
        assert "--hold long:secular=stage:clean" in cmd
        assert "--hold long:periodic=stage:clean" in cmd
        assert w.record is not None

    def test_the_old_periodic_only_plan_is_still_reachable(self) -> None:
        from gps_plot.detrend_picker_qt import STATE_ESTIMATE

        w = self._staged()
        w.grp["secular"].setCurrentText(STATE_ESTIMATE)
        w.refit()
        cmd = w.command.text()
        assert "--hold long:periodic=stage:clean" in cmd
        assert "--hold long:secular=stage:clean" not in cmd
        assert "--stage long:secular,step" in cmd

    def test_holding_nothing_is_refused_with_a_reason(self) -> None:
        """Staging exists to carry something; carrying nothing is not staging."""
        from gps_plot.detrend_picker_qt import STATE_ESTIMATE

        w = self._staged()
        w.grp["secular"].setCurrentText(STATE_ESTIMATE)
        w.grp["periodic"].setCurrentText(STATE_ESTIMATE)
        w.refit()
        assert w.record is None
        assert "no group is held" in w.summary.toPlainText()

    def test_unstaging_clears_an_unreachable_hold(self) -> None:
        """A control showing a value the fit cannot use is the divergence."""
        from gps_plot.detrend_picker_qt import GROUP_STATES, STATE_HOLD

        w = self._staged()
        assert w.grp["secular"].currentText() == STATE_HOLD
        w.cb_stage.setChecked(False)
        for name in ("secular", "periodic"):
            combo = w.grp[name]
            assert combo.currentText() != STATE_HOLD, f"{name} left holding"
            item = combo.model().item(GROUP_STATES.index(STATE_HOLD))
            assert not item.isEnabled()

    def test_the_clean_stage_keeps_secular_as_a_nuisance(self) -> None:
        """A seasonal fitted on a window that ignores the trend absorbs it."""
        from gps_plot.detrend_picker_qt import STATE_ESTIMATE

        w = self._staged()
        w.grp["secular"].setCurrentText(STATE_ESTIMATE)  # free in the long stage
        w.refit()
        # ... but still free in clean, so the held seasonal is unbiased
        assert "--stage clean:secular,periodic@" in w.command.text()


class TestRefineTau:
    """Slice 3: solve the one nonlinear parameter instead of eyeballing it.

    The visual fit fixes everything except tau, which is exactly what
    `gps_analysis.profile_transient_tau` exists to refine. The spinbox stays
    the single source the fit and the command both read, so a refinement
    cannot move one without the other.
    """

    @staticmethod
    def _synthetic(tau=1.5, t0=2012.0, seed=7):
        """A series with a KNOWN tau — the only honest way to test recovery."""
        import os

        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        import numpy as np

        from geo_dataread.detrend_estimate import FitDefaults, resolve_fit_settings
        from gps_plot.detrend_picker_qt import PickerWindow, _require_qt

        _pg, qtw = _require_qt()
        qtw.QApplication.instance() or qtw.QApplication([])
        rng = np.random.default_rng(seed)
        t = np.arange(2010.0, 2020.0, 1 / 365.25)
        post = np.maximum(t - t0, 0.0)
        sig = (
            3.0 * t - 6000.0 + 4.0 * np.sin(2 * np.pi * t) + 25.0 * np.log1p(post / tau)
        )
        y = np.vstack([sig + rng.normal(0, 1.0, t.size) for _ in range(3)])
        settings = resolve_fit_settings(
            "SYNT", None, FitDefaults(max_gap_years=2.0), catalog_source="test"
        )
        w = PickerWindow(
            "SYNT", t, y, np.full_like(y, 1.0), settings, max_gap_years=2.0, uncert=10
        )
        w.onset_lines[0].setValue(t0)
        return w

    def test_it_recovers_a_known_tau_and_applies_it(self) -> None:
        w = self._synthetic(tau=1.5)
        w.tau.setValue(4.0)  # a deliberately wrong seed
        w.cb_term.setChecked(True)
        w._refine_tau()
        assert w.tau.value() == pytest.approx(1.5, abs=0.1), w.summary.toPlainText()
        # and the command moved with the figure
        assert f"tau={w.tau.value()}" in w.command.text(), w.command.text()

    def test_an_unclosed_interval_is_a_bound_and_is_not_applied(self) -> None:
        """The profiler's own words: publish a BOUND then, not a measurement.

        Applying it would silently turn "tau is at least this" into "tau is
        this". SELF's transient placed on its declared 2008 coseismic is the
        real case — onset and step are collinear, so tau runs to the bound.
        """
        w = TestQtPickerBorrowedFeatures._window()
        w.onset_lines[0].setValue(2008.4085)  # the declared step epoch
        w.cb_term.setChecked(True)
        before = w.tau.value()
        w._refine_tau()
        text = w.summary.toPlainText()
        assert "BOUND, not applied" in text, text[:200]
        assert w.tau.value() == before, "a bound was applied as a measurement"

    def test_the_profiler_cannot_return_a_tau_the_spinbox_cannot_hold(self) -> None:
        """Regression: bounds wider than the control silently clamp.

        Profiling over (0.02, 40) while the spinbox held (0.05, 50) made the
        summary report tau = 0.020 and the command carry 0.05 — the reported
        number and the fitted one disagreed.
        """
        w = self._synthetic()
        w.cb_term.setChecked(True)
        w._refine_tau()
        for line in w.summary.toPlainText().splitlines():
            if " τ = " in line:
                value = float(line.split("τ = ")[1].split()[0])
                assert w.tau.minimum() <= value <= w.tau.maximum(), line

    def test_it_needs_a_transient_and_says_so(self) -> None:
        w = self._synthetic()
        w.cb_term.setChecked(False)
        w._refine_tau()
        assert "needs a transient" in w.summary.toPlainText()


class TestGroupStatePersistence:
    """Group states survive a save/reload, and cannot come back unreachable."""

    @staticmethod
    def _window():
        return TestQtPickerBorrowedFeatures._window()

    def test_states_round_trip_through_a_session(self) -> None:
        from gps_plot.detrend_picker_qt import STATE_ABSENT

        w = self._window()
        w.grp["periodic"].setCurrentText(STATE_ABSENT)
        w.refit()
        assert w.model == "linear"
        w.save_session()

        w2 = self._window()  # __init__ auto-loads the session
        assert w2.grp["periodic"].currentText() == STATE_ABSENT
        assert w2.model == "linear"

    def test_a_legacy_session_restores_the_plan_it_described(self) -> None:
        """Sessions outlive the code that reads them.

        A session written before per-group hold recorded a staged fit under
        the OLD plan, which held `periodic` only. Restoring it under the
        current default (hold the whole background) does not reproduce that
        fit -- and on a station with no declared step and no transient it
        leaves the long stage with nothing to estimate, so a session that used
        to work comes back REFUSED. Measured on a real SEYD session from
        2026-08-16.
        """
        import json

        from gps_plot.detrend_picker_qt import STATE_ESTIMATE, STATE_HOLD

        w = self._window()
        w.save_session()
        path = w._session_path()
        d = json.loads(path.read_text())
        d["stage"]["on"] = True
        d.pop("groups")  # as a pre-slice-2 file has it
        path.write_text(json.dumps(d))

        w2 = self._window()
        assert w2.grp["secular"].currentText() == STATE_ESTIMATE
        assert w2.grp["periodic"].currentText() == STATE_HOLD
        assert "--hold long:periodic=stage:clean" in w2.command.text()
        assert "--hold long:secular=stage:clean" not in w2.command.text()

    def test_a_hold_is_not_restored_without_a_window_to_hold_from(self) -> None:
        """Restoring an unreachable state would show a value the fit cannot use."""
        from gps_plot.detrend_picker_qt import STATE_HOLD

        w = self._window()
        w.stage_regions[0].setRegion((2009.5443, 2021.0394))
        w.cb_stage.setChecked(True)
        assert w.grp["secular"].currentText() == STATE_HOLD
        w.save_session()
        # hand-edit the payload to claim staging was off while a hold persists
        import json

        path = w._session_path()
        d = json.loads(path.read_text())
        d["stage"]["on"] = False
        path.write_text(json.dumps(d))

        w2 = self._window()
        assert w2.grp["secular"].currentText() != STATE_HOLD

    def test_an_unknown_group_or_state_is_dropped_not_fatal(self) -> None:
        """This key is newer than the files already in the field."""
        import json

        w = self._window()
        w.save_session()
        path = w._session_path()
        d = json.loads(path.read_text())
        d["groups"]["gravitational_wave"] = "estimate here"
        d["groups"]["secular"] = "wobble"
        path.write_text(json.dumps(d))

        w2 = self._window()
        assert w2.record is not None, "a forward-compatible session was rejected"
        assert "NOT restored" not in w2.summary.toPlainText()


class TestStepGroupControl:
    """`step` has free/held but no ABSENT — there is no way to un-declare one."""

    def test_absent_is_disabled_for_step(self) -> None:
        from gps_plot.detrend_picker_qt import GROUP_STATES, STATE_ABSENT

        w = TestQtPickerBorrowedFeatures._window()
        item = w.grp["step"].model().item(GROUP_STATES.index(STATE_ABSENT))
        assert not item.isEnabled(), (
            "offering 'not in the model' for step promises something no "
            "emitted command can carry out: steps.csv is a floor"
        )

    def test_holding_a_step_reaches_the_command(self) -> None:
        from gps_plot.detrend_picker_qt import STATE_ESTIMATE, STATE_HOLD

        w = TestQtPickerBorrowedFeatures._window()
        # a window that CONTAINS SELF's 2008 step, so clean can estimate it
        w.stage_regions[0].setRegion((2005.0, 2015.0))
        w.cb_stage.setChecked(True)
        w.grp["secular"].setCurrentText(STATE_ESTIMATE)
        w.grp["periodic"].setCurrentText(STATE_HOLD)
        w.grp["step"].setCurrentText(STATE_HOLD)
        w.refit()
        assert "--hold long:step=stage:clean" in w.command.text(), w.command.text()


class TestModelEquation:
    """The panel shows the general FORM; the numbers go to the terminal."""

    def test_the_equation_is_read_off_the_record(self) -> None:
        """Written down once, it would rot; derived, it cannot."""
        from gps_plot.detrend_picker_qt import model_equation

        eq = model_equation(
            ["offset", "rate", "cos_annual", "sin_annual", "step_amp_1", "log_amp_1"]
        )
        assert eq.startswith("x(t) = a₀ + a₁·(t−t₀)")
        assert "c₁·cos(2πt)" in eq and "s₁·sin(2πt)" in eq
        assert "h1·H(t−t1)" in eq, "a declared step has no symbol"
        assert "ln(1+(t−tₑ)/τ)" in eq, "the log transient is not spelled out"

    def test_a_linear_model_says_so(self) -> None:
        from gps_plot.detrend_picker_qt import model_equation

        eq = model_equation(["offset", "rate"])
        assert "cos" not in eq and "sin" not in eq

    def test_the_panel_leads_with_the_form(self) -> None:
        w = TestQtPickerBorrowedFeatures._window()
        w.refit()
        assert w.summary.toPlainText().startswith("x(t) = ")


class TestCompareAndAdopt:
    """Compare is an OVERLAY; only adopt moves the command.

    Unchecking 'stage the fit' to peek at the plain fit is destructive — the
    toggle rewrites the group states on the way out and again on the way back
    — so the comparison exists to leave the setup alone.
    """

    @staticmethod
    def _staged():
        from gps_plot.detrend_picker_qt import STATE_ESTIMATE

        w = TestQtPickerBorrowedFeatures._window()
        w.stage_regions[0].setRegion((2003.0, 2015.0))
        w.cb_stage.setChecked(True)
        w.grp["secular"].setCurrentText(STATE_ESTIMATE)
        w.refit()
        return w

    def test_comparing_leaves_the_command_and_setup_alone(self) -> None:
        w = self._staged()
        before_cmd = w.command.text()
        before_state = w.grp["periodic"].currentText()
        w.compare_unstaged()
        assert len(w.compare_curves[0].getData()[0]) > 0, "no overlay drawn"
        assert w.command.text() == before_cmd, "the command moved without the figure"
        assert "--stage" in w.command.text(), "the staged setup was lost"
        assert w.grp["periodic"].currentText() == before_state
        assert w.btn_adopt.isEnabled()

    def test_adopting_makes_the_command_describe_what_is_drawn(self) -> None:
        w = self._staged()
        w.compare_unstaged()
        w.adopt_comparison()
        assert "--stage" not in w.command.text(), w.command.text()
        assert not w.cb_stage.isChecked()
        assert w.record is not None
        assert len(w.compare_curves[0].getData()[0] or []) == 0, "overlay left behind"

    def test_a_stale_overlay_is_cleared_on_the_next_refit(self) -> None:
        """It is a snapshot of a DIFFERENT configuration.

        Left on screen after the blue line moves, the two read as one fit.
        """
        w = self._staged()
        w.compare_unstaged()
        assert len(w.compare_curves[0].getData()[0]) > 0
        w.sp_gap.setValue(3.0)
        w._set_max_gap()
        assert len(w.compare_curves[0].getData()[0] or []) == 0
        assert not w.btn_adopt.isEnabled()


class TestRefusedFitLooksRefused:
    """A stale trajectory must not read as a result.

    Measured on RHOF 2026-08-21: staging with the whole background held leaves
    the long stage nothing to estimate, so the plan is refused — but the
    previous configuration's blue line stayed on screen, solid, and was read
    as "the fit within the orange window". The header said NO RECORD; the
    curve said otherwise, and the curve wins.
    """

    def test_a_refused_fit_greys_and_dashes_the_trajectory(self) -> None:
        w = TestQtPickerBorrowedFeatures._window(sta="RHOF", gap=2.0)
        w.stage_regions[0].setRegion((2001.7327, 2016.2057))
        w.cb_stage.setChecked(True)  # both held; RHOF has no step or transient
        assert w.record is None, "RHOF unexpectedly fittable; pick another case"
        pen = w.fit_curves[0].opts["pen"]
        assert pen.color().getRgb()[:3] != FIT_COLOR_RGB, "still drawn as a fit"
        assert pen.style().name == "DashLine"

    def test_a_good_fit_restores_the_live_pen(self) -> None:
        w = TestQtPickerBorrowedFeatures._window(sta="RHOF", gap=2.0)
        w.stage_regions[0].setRegion((2001.7327, 2016.2057))
        w.cb_stage.setChecked(True)
        w.cb_stage.setChecked(False)  # back to a fit that works
        assert w.record is not None
        pen = w.fit_curves[0].opts["pen"]
        assert pen.color().getRgb()[:3] == FIT_COLOR_RGB
        assert pen.style().name == "SolidLine"


class TestParametersAreVisible:
    """The picker is launched from a sway binding — stdout reaches nobody."""

    def test_the_panel_carries_the_parameter_table(self) -> None:
        w = TestQtPickerBorrowedFeatures._window()
        w.refit()
        text = w.summary.toPlainText()
        assert "parameters" in text, text[:200]
        assert "rate" in text and "cos_annual" in text
        # one column per component
        header = [ln for ln in text.splitlines() if "North" in ln and "Up" in ln]
        assert header, "no per-component header"

    def test_dismiss_keeps_the_configured_fit(self) -> None:
        w = TestQtPickerBorrowedFeatures._window()
        w.refit()
        before = w.command.text()
        w.compare_unstaged()
        assert w.btn_dismiss.isEnabled()
        w._dismiss_comparison()
        assert len(w.compare_curves[0].getData()[0] or []) == 0
        assert w.command.text() == before, "dismissing changed the command"
        assert not w.btn_adopt.isEnabled()
