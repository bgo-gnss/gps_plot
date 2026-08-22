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
from gps_plot.detrend_picker_qt import StageDraft, render_stage_flags


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


def _jump_count(curve) -> int:
    """Discontinuities in a drawn curve, not merely non-zero slope.

    `trajectory_curve` brackets each step epoch at +/- 1e-6 yr, so a jump
    appears between two grid points ~2e-6 yr apart while the rest are ~1/365
    apart. The jump's difference is therefore orders of magnitude above the
    typical one -- which is what to test for. Thresholding the raw difference
    instead counts every point of a sloping line.
    """
    import numpy as np

    d = np.abs(np.diff(np.asarray(curve, dtype=float)))
    d = d[np.isfinite(d)]
    if d.size == 0:
        return 0
    typical = float(np.median(d))
    return int((d > max(20.0 * typical, 1e-9)).sum())


def _estimated_groups(cmd: str) -> set[str]:
    """Every group estimated by SOME stage of an emitted command.

    Which stage estimates what is a layout decision the preset is allowed to
    change; that a group is estimated at all is not. Asserting on the union
    keeps the tests measuring the science.
    """
    parts = shlex.split(cmd)
    out: set[str] = set()
    for i, arg in enumerate(parts):
        if arg == "--stage":
            spec = parts[i + 1].split("@")[0]
            out.update(g for g in spec.split(":", 1)[1].split(",") if g)
    return out


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


class TestStageDraftIsTheOnlySpeller:
    """``StageDraft`` + :func:`render_stage_flags` are the picker's plan.

    These run without Qt: the drafts are plain data, which is the point of
    lifting them out of the window class. Everything the panel will let the
    operator build in the next slice has to survive this round trip first.
    """

    def test_no_stages_is_unstaged(self) -> None:
        assert render_stage_flags([]) == ([], [])

    def test_a_window_of_none_omits_the_at_sign(self) -> None:
        # None means "inherit the caller's domain"; '@:' means the full span.
        # They differ whenever --segment was passed, so the renderer must not
        # turn one into the other.
        specs, _ = render_stage_flags([StageDraft("long", groups=["secular"])])
        assert specs == ["long:secular"]

    def test_one_stage_leaves_the_hold_unqualified(self) -> None:
        specs, holds = render_stage_flags(
            [StageDraft("fit", groups=["periodic"], holds={"secular": "donor:OLAC"})]
        )
        assert (specs, holds) == (["fit:periodic"], ["secular=donor:OLAC"])

    def test_more_than_one_stage_qualifies_every_hold(self) -> None:
        _, holds = render_stage_flags(
            [
                StageDraft(
                    "clean", groups=["secular", "periodic"], window=(2001.6, 2019.5)
                ),
                StageDraft("long", groups=["step"], holds={"periodic": "stage:clean"}),
            ]
        )
        assert holds == ["long:periodic=stage:clean"]

    def test_empty_groups_render_and_are_refused_downstream(self) -> None:
        # The RHOF case: nothing left free once the background is held. The
        # renderer does NOT swallow it -- the operator must meet the same
        # refusal whether they are looking at the figure or at the command.
        specs, holds = render_stage_flags(
            [
                StageDraft("clean", groups=["secular"], window=(2001.0, 2016.0)),
                StageDraft("long", groups=[], holds={"secular": "stage:clean"}),
            ]
        )
        assert specs[1] == "long:"
        with pytest.raises(ValueError):
            build_stage_plan(specs, holds)

    def test_the_three_slot_composition_round_trips(self) -> None:
        # lin+per on a quiet window, steps against that background, then
        # transients against both -- the composition the redesign exists for.
        drafts = [
            StageDraft("lin", groups=["secular"], window=(2001.0, 2016.0)),
            StageDraft("per", groups=["periodic"], window=(2005.0, 2012.0)),
            StageDraft(
                "st",
                groups=["step"],
                holds={"secular": "stage:lin", "periodic": "stage:per"},
            ),
            StageDraft(
                "tr",
                groups=["transient"],
                holds={
                    "secular": "stage:lin",
                    "periodic": "stage:per",
                    "step": "stage:st",
                },
            ),
        ]
        specs, holds = render_stage_flags(drafts)
        plan = build_stage_plan(specs, holds)
        again = shlex.split(render_command("SELF", plan))
        assert (
            build_stage_plan(
                [again[i + 1] for i, a in enumerate(again) if a == "--stage"],
                [again[i + 1] for i, a in enumerate(again) if a == "--hold"],
            )
            == plan
        )


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
        # `step` must be estimated SOMEWHERE. Which stage moved when the
        # preset opened out into the compositional build (steps now get their
        # own stage against the held background) -- what this asserts is that
        # the group is fitted at all, not the literal, because the layout is
        # allowed to change and the freedom is not.
        assert "step" in _estimated_groups(cmd), cmd
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
        assert "step" not in _estimated_groups(cmd), cmd
        # RHOF has no step and no transient, so the preset lays out the
        # background stage and nothing else. ONE windowed stage is a real
        # fit -- the model is estimated inside the window and evaluated
        # across the whole span, which is what "hold the background from the
        # quiet window" means when there is nothing else to estimate. The
        # two-stage default used to refuse this, and once the grammar began
        # accepting the single-stage spelling that refusal broke the
        # invariant: the window said refused, the emitted command fitted.
        assert [c.name for c in w.stage_cards] == ["clean"]
        assert "--stage clean:secular,periodic@" in cmd
        assert "--hold" not in cmd
        assert w.record is not None, w.summary.toPlainText()[:200]

    def test_a_refused_plan_is_still_emitted(self) -> None:
        """Violation eight, introduced while building slice 2 and caught here.

        When the stage plan would not build, `_command` fell through to the
        UNSTAGED spelling — so the window showed a refusal while the command
        described a different, perfectly fittable fit: copy it and you get
        the figure the picker had just refused to show.

        The command must say what was ASKED FOR. The case moved when the
        preset opened out into N stages — RHOF now lays out ONE windowed
        stage, which is a perfectly good fit and no longer refused — so the
        refusal exercised here is the one that survives: two stages neither
        of which carries anything from the other.
        """
        import shlex

        from gps_plot.detrend_workbench import _build_parser

        w = self._staged("SELF")
        for group in ("secular", "periodic"):
            w.stage_cards[-1].groups[group].setChecked(True)
        w.refit()
        assert w.record is None, "expected a refusal to emit"
        cmd = w.command.text()
        assert "--stage clean:" in cmd and "--stage st:" in cmd, cmd
        assert "--hold" not in cmd, "nothing is held — that IS the refusal"
        # and it is a command the workbench can at least PARSE, so its refusal
        # is the plan's refusal rather than a syntax error
        ns = _build_parser().parse_args(shlex.split(cmd)[1:])
        assert len(ns.stage) == 2 and not ns.hold

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
        w = self._window()
        for sec, per, expected in (
            (True, True, "lineperiodic"),
            (True, False, "linear"),
            (False, True, "periodic"),
        ):
            w.model_in["secular"].setChecked(sec)
            w.model_in["periodic"].setChecked(per)
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
        w = self._window()
        w.refit()
        good = w.record
        assert good is not None
        w.model_in["secular"].setChecked(False)
        w.model_in["periodic"].setChecked(False)
        w.refit()
        assert w.model is None
        assert "refused" in w.summary.toPlainText().lower()

    def test_nothing_can_be_held_without_a_stage_to_hold_from(self) -> None:
        """Unstaged means no cards at all, not one card covering everything.

        A single stage over the whole domain would be `--segment` spelled a
        second way, and it would make `hold` look reachable when there is
        nothing above to hold from.
        """
        w = self._window()
        assert w.stage_cards == []
        assert not w.stages_box.isVisible()
        assert "--hold" not in w.command.text()
        assert "--stage" not in w.command.text()

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
        w = self._window()
        w.model_in["secular"].setChecked(False)  # -> --model periodic
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
        w = self._staged()
        assert [c.name for c in w.stage_cards] == ["clean", "st"], (
            "SELF has a declared step, so the preset is background then steps"
        )
        assert w.stage_cards[0].estimates() == ["secular", "periodic"]
        cmd = w.command.text()
        assert "--hold st:secular=stage:clean" in cmd
        assert "--hold st:periodic=stage:clean" in cmd
        assert w.record is not None

    def test_the_old_periodic_only_plan_is_still_reachable(self) -> None:
        # Re-estimating the trend in the later stage is now spelled by ticking
        # it there, rather than by a state that meant it implicitly.
        w = self._staged()
        w.stage_cards[-1].groups["secular"].setChecked(True)
        w.refit()
        cmd = w.command.text()
        assert "--hold st:periodic=stage:clean" in cmd
        assert "--hold st:secular=stage:clean" not in cmd
        assert "--stage st:secular,step" in cmd

    def test_holding_nothing_is_refused_with_a_reason(self) -> None:
        """Staging exists to carry something; carrying nothing is not staging."""
        w = self._staged()
        for group in ("secular", "periodic"):
            w.stage_cards[-1].groups[group].setChecked(True)
        w.refit()
        assert w.record is None
        assert "did no work" in w.summary.toPlainText()

    def test_unstaging_drops_the_cards(self) -> None:
        """Cards left on screen would describe a plan that no longer runs."""
        w = self._staged()
        assert w.stage_cards
        w.cb_stage.setChecked(False)
        assert w.stage_cards == []
        assert not w.stages_box.isVisible()
        assert "--hold" not in w.command.text()

    def test_the_clean_stage_keeps_secular_as_a_nuisance(self) -> None:
        """A seasonal fitted on a window that ignores the trend absorbs it.

        So a group may be estimated in TWO stages — which the three-state
        combo could not express, and is why membership and assignment had to
        come apart.
        """
        w = self._staged()
        w.stage_cards[-1].groups["secular"].setChecked(True)
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
        w = self._window()
        w.model_in["periodic"].setChecked(False)
        w.refit()
        assert w.model == "linear"
        w.save_session()

        w2 = self._window()  # __init__ auto-loads the session
        assert not w2.model_in["periodic"].isChecked()
        assert w2.model == "linear"

    def test_an_assignment_round_trips_through_a_session(self) -> None:
        w = self._window()
        w.stage_regions[0].setRegion((2009.5443, 2021.0394))
        w.cb_stage.setChecked(True)
        w.stage_cards[-1].groups["secular"].setChecked(True)
        w.refit()
        before = w.command.text()
        w.save_session()

        w2 = self._window()
        assert w2.cb_stage.isChecked()
        assert w2.stage_cards[-1].groups["secular"].isChecked()
        assert w2.command.text() == before

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

        w = self._window()
        w.save_session()
        path = w._session_path()
        d = json.loads(path.read_text())
        d["stage"]["on"] = True
        for key in ("groups", "in_model", "assignments"):
            d.pop(key, None)  # as a pre-slice-2 file has it
        path.write_text(json.dumps(d))

        w2 = self._window()
        assert w2.stage_cards[0].estimates() == ["periodic"], (
            "the old plan held periodic only; restoring it under today's "
            "default would not reproduce the fit the session described"
        )
        assert "--hold st:periodic=stage:clean" in w2.command.text()
        assert "--hold st:secular=stage:clean" not in w2.command.text()

    def test_a_hold_is_not_restored_without_a_window_to_hold_from(self) -> None:
        """Restoring an unreachable assignment would show a plan that cannot run."""
        w = self._window()
        w.stage_regions[0].setRegion((2009.5443, 2021.0394))
        w.cb_stage.setChecked(True)
        assert w.stage_cards
        w.save_session()
        # hand-edit the payload to claim staging was off while cards persist
        import json

        path = w._session_path()
        d = json.loads(path.read_text())
        d["stage"]["on"] = False
        path.write_text(json.dumps(d))

        w2 = self._window()
        assert w2.stage_cards == []
        assert "--hold" not in w2.command.text()

    def test_an_unknown_group_or_state_is_dropped_not_fatal(self) -> None:
        """This key is newer than the files already in the field."""
        import json

        w = self._window()
        w.stage_regions[0].setRegion((2009.5443, 2021.0394))
        w.cb_stage.setChecked(True)
        w.save_session()
        path = w._session_path()
        d = json.loads(path.read_text())
        d["in_model"]["gravitational_wave"] = True
        d["assignments"]["clean"].append("wobble")
        d["assignments"]["a_stage_that_never_existed"] = ["secular"]
        path.write_text(json.dumps(d))

        w2 = self._window()
        assert w2.record is not None, "a forward-compatible session was rejected"
        assert "NOT restored" not in w2.summary.toPlainText()


class TestStepGroupControl:
    """`step` is assignable but has no membership control.

    There is no CLI spelling for un-declaring a step: ``steps.csv`` is a
    FLOOR that merges in, and a picked step is removed by removing the pick.
    A "not in the model" control for it would promise something no emitted
    command could carry out.
    """

    def test_step_has_no_membership_checkbox(self) -> None:
        w = TestQtPickerBorrowedFeatures._window()
        assert "step" not in w.model_in
        assert set(w.model_in) == {"secular", "periodic"}

    def test_step_is_assignable_on_every_card(self) -> None:
        w = TestQtPickerBorrowedFeatures._window()
        w.stage_regions[0].setRegion((2005.0, 2015.0))
        w.cb_stage.setChecked(True)
        assert all("step" in c.groups for c in w.stage_cards)

    def test_holding_a_step_reaches_the_command(self) -> None:
        w = TestQtPickerBorrowedFeatures._window()
        # a window that CONTAINS SELF's 2008 step, so clean can estimate it
        w.stage_regions[0].setRegion((2005.0, 2015.0))
        w.cb_stage.setChecked(True)
        w.stage_cards[0].groups["step"].setChecked(True)
        w.stage_cards[-1].groups["step"].setChecked(False)
        w.stage_cards[-1].groups["secular"].setChecked(True)
        w.refit()
        assert "--hold st:step=stage:clean" in w.command.text(), w.command.text()


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
        w = TestQtPickerBorrowedFeatures._window()
        w.stage_regions[0].setRegion((2003.0, 2015.0))
        w.cb_stage.setChecked(True)
        w.stage_cards[-1].groups["secular"].setChecked(True)
        w.refit()
        return w

    def test_comparing_leaves_the_command_and_setup_alone(self) -> None:
        w = self._staged()
        before_cmd = w.command.text()
        before_cards = [(c.name, c.estimates()) for c in w.stage_cards]
        w.compare_unstaged()
        assert len(w.compare_curves[0].getData()[0]) > 0, "no overlay drawn"
        assert w.command.text() == before_cmd, "the command moved without the figure"
        assert "--stage" in w.command.text(), "the staged setup was lost"
        assert [(c.name, c.estimates()) for c in w.stage_cards] == before_cards
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

    RHOF is no longer that case — one windowed stage is a real fit — so the
    refusal driven here is the surviving one: a later stage that re-estimates
    everything the earlier one did, leaving the earlier stage no work.
    """

    def test_a_refused_fit_greys_and_dashes_the_trajectory(self) -> None:
        w = TestQtPickerBorrowedFeatures._window()
        w.stage_regions[0].setRegion((2009.5443, 2021.0394))
        w.cb_stage.setChecked(True)
        for group in ("secular", "periodic"):
            w.stage_cards[-1].groups[group].setChecked(True)
        w.refit()
        assert w.record is None, "expected a refusal; pick another case"
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


class TestDetrendedView:
    """data − f(t), display only.

    Same convention as --hide-outliers: the masks, the record and the emitted
    command are untouched. It subtracts the model that was already fitted; it
    does not fit anything different.
    """

    def test_it_shows_residuals_and_changes_nothing_else(self) -> None:
        import numpy as np

        w = TestQtPickerBorrowedFeatures._window(sta="RHOF", gap=2.0)
        raw = w.kept_scatters[1].getData()[1]
        cmd, rec = w.command.text(), w.record
        assert abs(float(np.mean(raw))) > 1.0, "RHOF East is already centred?"

        w.cb_detrend.setChecked(True)
        res = w.kept_scatters[1].getData()[1]
        assert abs(float(np.mean(res))) < 1.0, "residuals are not centred on zero"
        assert w.command.text() == cmd, "a display toggle moved the command"
        assert w.record == rec, "a display toggle moved the record"

    def test_the_axis_says_which_quantity_is_drawn(self) -> None:
        w = TestQtPickerBorrowedFeatures._window(sta="RHOF", gap=2.0)
        assert w.plots[0].getAxis("left").labelText == "North [mm]"
        w.cb_detrend.setChecked(True)
        assert w.plots[0].getAxis("left").labelText == "North residual [mm]"

    def test_the_trajectory_is_not_drawn_over_its_own_residuals(self) -> None:
        """Subtracted, the model IS the zero line."""
        w = TestQtPickerBorrowedFeatures._window(sta="RHOF", gap=2.0)
        assert len(w.fit_curves[0].getData()[0]) > 0
        w.cb_detrend.setChecked(True)
        assert len(w.fit_curves[0].getData()[0] or []) == 0


class TestGroupLabels:
    """The GUI says `linear`; every emitted flag still says `secular`.

    The stage grammar's `secular` names the linear term alone, but "secular"
    properly names the long-term background as a whole — linear AND periodic,
    which is what `lineperiodic` composes. Showing the grammar's word invites
    reading it as the whole background.
    """

    def test_the_row_is_labelled_linear_but_the_flag_is_secular(self) -> None:
        from gps_plot.detrend_picker_qt import GROUP_LABELS

        assert GROUP_LABELS["secular"] == "linear"
        w = TestQtPickerBorrowedFeatures._window()
        w.stage_regions[0].setRegion((2003.0, 2015.0))
        w.cb_stage.setChecked(True)
        assert w.stage_cards[0].groups["secular"].text() == "linear"
        w.stage_cards[-1].groups["secular"].setChecked(True)
        w.refit()
        # the grammar is unchanged where it matters — in the command
        assert "secular" in w.command.text()
        assert "linear" not in w.command.text()

    def test_the_tooltip_names_both(self) -> None:
        w = TestQtPickerBorrowedFeatures._window()
        tip = w.model_in["secular"].toolTip()
        assert "linear" in tip and "secular" in tip
        assert "secular background" in tip

    def test_the_header_says_how_many_epochs_the_domain_kept(self) -> None:
        """The blue region is a CONTROL, but shaded background does not read
        as one — and the plots show the whole series either way, so a fit
        restricted to a window looked identical to a fit over everything.
        """
        import re

        w = TestQtPickerBorrowedFeatures._window(sta="RHOF", gap=2.0)
        strip = lambda t: re.sub("<[^>]+>", "", t)  # noqa: E731
        assert "fitting all" in strip(w.header.text())
        w.domain_regions[0].setRegion((2001.7327, 2016.2057))
        w.refit()
        head = strip(w.header.text())
        assert "fitting 1718 of 4812 epochs" in head, head


class TestCardsAreTheStageEditor:
    """Slice 2: the panel is N stage cards, and the cards ARE the plan.

    Two decisions the three-state combo used to fuse are separated here,
    because the estimator already treats them as orthogonal: ``--model``
    decides which terms are in the design matrix, the stage plan decides
    where each is estimated. So membership is one control that works staged
    or not, and assignment is a checkbox per stage.

    A group may be estimated in MORE than one stage on purpose: the clean
    stage frees ``secular`` as a nuisance even when the kept value comes from
    a later stage, because a seasonal fitted on a window that ignores the
    trend inside it absorbs part of that trend.
    """

    @staticmethod
    def _staged():
        w = TestQtPickerBorrowedFeatures._window()
        w.stage_regions[0].setRegion((2009.5443, 2021.0394))
        w.cb_stage.setChecked(True)
        return w

    # -- 1. --model composed from membership, read off _current() ---------

    def test_model_follows_membership_not_assignment(self) -> None:
        """Read ``_current()``'s own output, never a re-derivation.

        Violation 6 was a control that moved the figure without moving the
        command, and the first test written for it PASSED against the bug
        because it re-derived the value instead of asking the method that
        assembles it.
        """
        w = self._staged()
        assert w._current()[0] is not None
        assert w.model == "lineperiodic"

        w.model_in["secular"].setChecked(False)
        w._current()
        assert w.model == "periodic"

        w.model_in["secular"].setChecked(True)
        w.model_in["periodic"].setChecked(False)
        w._current()
        assert w.model == "linear"

    def test_both_out_is_refused_not_guessed(self) -> None:
        # No --model value carries neither, so it must refuse rather than
        # send something the estimator would fail on later and less clearly.
        w = self._staged()
        w.model_in["secular"].setChecked(False)
        w.model_in["periodic"].setChecked(False)
        w.refit()
        assert w.model is None
        assert w.record is None
        assert "both" in w.summary.toPlainText()

    def test_an_unassigned_group_is_still_in_the_model(self) -> None:
        """Absent-from-every-card is NOT absent-from-the-model.

        This is the regression the split exists to prevent: a freshly added
        empty stage, and an unstaged picker, both have a group assigned
        nowhere -- and neither means the term should leave the design matrix.
        """
        w = self._staged()
        for card in w.stage_cards:
            card.groups["periodic"].setChecked(False)
        w._current()
        assert w.model == "lineperiodic", "membership must not follow assignment"

    # -- 2. session restore is a migration, not a rename ------------------

    def test_a_legacy_group_state_session_migrates(self) -> None:
        """Sessions outlive the code that reads them.

        The states map onto the new pair exactly: ABSENT is out of the model,
        HOLD is estimated in the FIRST stage (and so carried by the later
        ones), ESTIMATE is estimated in the LAST.
        """
        from gps_plot.detrend_picker_qt import (
            STATE_ABSENT,
            STATE_ESTIMATE,
            STATE_HOLD,
        )

        from gps_plot.detrend_picker_qt import migrate_group_states

        w = self._staged()
        parsed = w._parse_session(
            {
                "station": "SELF",
                "stage": {"on": True, "window": [2009.5443, 2021.0394]},
                "groups": {
                    "secular": STATE_ESTIMATE,
                    "periodic": STATE_HOLD,
                    "step": STATE_ABSENT,
                },
            }
        )
        assert parsed["in_model"] == {
            "secular": True,
            "periodic": True,
            "step": False,
        }
        # The names are resolved against the REAL cards, which do not exist
        # while the payload is being parsed -- reading them off an empty card
        # list collapsed `first` and `last` onto one stage and put a held
        # group in the same place as a freely estimated one.
        assert parsed["assignments"] is None
        assignments = migrate_group_states(parsed["legacy_groups"], "clean", "st")
        assert "periodic" in assignments["clean"]
        assert "secular" in assignments["st"]
        assert "periodic" not in assignments["st"]

    def test_a_session_with_no_groups_keeps_its_old_meaning(self) -> None:
        # Predates per-group hold: that build held `periodic` only, and
        # restoring it under the current default would not reproduce the fit
        # it described.
        from gps_plot.detrend_picker_qt import migrate_group_states

        w = self._staged()
        parsed = w._parse_session(
            {"station": "SELF", "stage": {"on": True, "window": [2009.0, 2021.0]}}
        )
        assignments = migrate_group_states(parsed["legacy_groups"], "clean", "st")
        assert assignments["clean"] == ["periodic"]
        assert "secular" in assignments["st"]


class TestPeelFollowsTheActiveStage:
    """Slice 3: the plot shows what the active stage is actually fitted to.

    The standing complaint this answers: an estimate made in the orange
    window was not the model the next decision got read against. Selecting
    stage k now subtracts stages 1..k-1 — steps against data − s(t),
    transients against data − s(t) − st(t) — which is the operator's own
    description of the workflow, made into the display rule.
    """

    @staticmethod
    def _staged():
        w = TestQtPickerBorrowedFeatures._window()
        w.stage_regions[0].setRegion((2009.5443, 2021.0394))
        w.cb_stage.setChecked(True)
        return w

    def test_the_first_stage_peels_nothing(self) -> None:
        w = self._staged()
        assert w.stage_cards[0].box.isChecked(), "the first card starts active"
        assert w._peeled_groups() == []
        assert "minus" not in w.plots[0].getAxis("left").labelText

    def test_a_later_stage_peels_exactly_the_earlier_ones(self) -> None:
        import numpy as np

        from gps_plot.detrend_workbench import group_contribution

        w = self._staged()
        assert [c.name for c in w.stage_cards] == ["clean", "st"]
        w.stage_cards[1].box.setChecked(True)
        w.refit()
        assert w._peeled_groups() == ["secular", "periodic"]
        assert w.record is not None

        # The drawn series IS data minus that contribution -- not merely
        # "something smaller".
        expected = w.data - group_contribution(
            w.record, w.yearf, ["secular", "periodic"]
        )
        shown_x, shown_y = w.kept_scatters[0].getData()
        finite = np.isfinite(w.data[0])
        keep = finite & np.isin(w.yearf, shown_x)
        assert keep.sum() > 100, "no kept epochs to compare"
        assert np.allclose(
            shown_y[: keep.sum()],
            expected[0][keep][: keep.sum()],
            atol=1e-9,
        )

    def test_the_axis_names_what_was_subtracted(self) -> None:
        w = self._staged()
        w.stage_cards[1].box.setChecked(True)
        w.refit()
        label = w.plots[0].getAxis("left").labelText
        assert "minus linear, periodic" in label, label

    def test_the_curve_is_peeled_with_the_data(self) -> None:
        """A residual series under an unpeeled curve is a model of other data."""
        import numpy as np

        w = self._staged()
        w.stage_cards[1].box.setChecked(True)
        w.refit()
        _, curve_y = w.fit_curves[0].getData()
        assert curve_y is not None and len(curve_y) > 0
        # What is left of SELF's model after removing lin+per is the step
        # alone: piecewise constant with exactly ONE jump. Asserting the
        # SHAPE rather than a magnitude is deliberate -- with the background
        # held from a window that starts after the 2008 coseismic, the
        # amplitude here is a few hundredths of a millimetre, and a
        # threshold on it would be testing this station's numbers instead of
        # the peel.
        jumps = np.flatnonzero(np.abs(np.diff(curve_y)) > 1e-12)
        assert jumps.size == 1, f"expected one step, got {jumps.size}"
        assert float(np.nanstd(curve_y)) < float(np.nanstd(w.data[0])) / 10.0, (
            "the curve still carries the trend that was peeled off the data"
        )

    def test_the_view_only_checkbox_still_wins(self) -> None:
        """It asks about the whole model, not about one stage."""
        w = self._staged()
        w.stage_cards[1].box.setChecked(True)
        w.cb_detrend.setChecked(True)
        w.refit()
        assert "residual" in w.plots[0].getAxis("left").labelText
        assert len(w.fit_curves[0].getData()[0] or []) == 0

    def test_peeling_moves_no_fitted_quantity(self) -> None:
        """DISPLAY ONLY, on the same terms as --hide-outliers."""
        w = self._staged()
        before_cmd = w.command.text()
        before_rms = list(w.record["rms"])
        w.stage_cards[1].box.setChecked(True)
        w.refit()
        assert w.command.text() == before_cmd, "the peel moved the command"
        assert list(w.record["rms"]) == before_rms, "the peel moved the fit"

    def test_the_card_shows_the_holds_it_emits(self) -> None:
        w = self._staged()
        w.refit()
        text = w.stage_cards[1].held_label.text()
        assert "linear ← clean" in text and "periodic ← clean" in text
        for group in ("secular", "periodic"):
            assert f"--hold st:{group}=stage:clean" in w.command.text()
        assert w.stage_cards[0].held_label.text().startswith("holds nothing")


class TestGroupContribution:
    """The peel's arithmetic, without a window."""

    @staticmethod
    def _record():
        import json
        import pathlib

        p = pathlib.Path.home() / ".config/gpsconfig/detrend_params.json"
        if not p.is_file():  # pragma: no cover
            pytest.skip("no deployed detrend_params.json")
        recs = json.loads(p.read_text())
        recs = recs.get("stations", recs)
        for sta, rec in recs.items():
            if isinstance(rec, dict) and rec.get("step_epochs"):
                return sta, rec
        pytest.skip("no deployed record carries a step")  # pragma: no cover

    def test_the_groups_sum_back_to_the_whole_model(self) -> None:
        """Linear in every parameter it solves for, so this must be exact."""
        import numpy as np

        from gps_analysis import evaluate_record
        from gps_plot.detrend_workbench import group_contribution

        _, rec = self._record()
        t = np.linspace(2000.0, 2024.0, 400)
        full = np.asarray(evaluate_record(rec, t, terms="all"))
        parts = group_contribution(rec, t, ["secular", "periodic"]) + (
            group_contribution(rec, t, ["step", "transient"])
        )
        assert np.nanmax(np.abs(full - parts)) == 0.0

    def test_the_step_tail_is_classified_as_step(self) -> None:
        """`to_record` APPENDS step_amp_k, so no classifier ever sees them."""
        from gps_plot.detrend_workbench import group_param_mask

        _, rec = self._record()
        names = rec["param_names"]
        mask = group_param_mask(rec, ["step"])
        assert [n for n, m in zip(names, mask) if m] == [
            n for n in names if n.startswith("step_amp_")
        ]

    def test_periodic_alone_leaves_the_offset_in(self) -> None:
        """The offset belongs to secular; peeling periodic must not take it."""
        from gps_plot.detrend_workbench import group_param_mask

        _, rec = self._record()
        names = rec["param_names"]
        mask = group_param_mask(rec, ["periodic"])
        assert "offset" not in [n for n, m in zip(names, mask) if m]


class TestUseFlaggedEpochs:
    """Slice 6: the screened epochs can go back INTO the fit.

    Two controls that look alike and are not: drawing the grey points is
    `--hide-outliers`-shaped (same masks, same counts, same record), while
    fitting them changes all three. They sit on opposite sides of the
    view-only divider for that reason.
    """

    @staticmethod
    def _window():
        return TestQtPickerBorrowedFeatures._window()

    def test_it_emits_the_stage_set_the_cli_already_has(self) -> None:
        """No new grammar: S1/S2 are structural, so naming only those is
        exactly "flag nothing" -- inventing a flag would have been a second
        spelling for a state the CLI could already reach."""
        from gps_plot.detrend_picker_qt import USE_FLAGGED_STAGES

        w = self._window()
        assert "--stages" not in w.command.text()
        w.cb_use_flagged.setChecked(True)
        assert f"--stages {USE_FLAGGED_STAGES}" in w.command.text()

    def test_the_emitted_stage_set_is_the_one_the_fit_ran(self) -> None:
        from gps_plot.detrend_picker_qt import USE_FLAGGED_STAGES

        w = self._window()
        w.cb_use_flagged.setChecked(True)
        assert w.stages_spec == USE_FLAGGED_STAGES
        assert f"--stages {w.stages_spec}" in w.command.text()

    def test_nothing_is_flagged_and_the_record_moves(self) -> None:
        w = self._window()
        w.refit()
        before = w.record
        assert before is not None
        assert sum(before["n_rejected"]) > 0, "no screening to switch off here"

        w.cb_use_flagged.setChecked(True)
        assert w.record is not None
        assert sum(w.record["n_rejected"]) == 0, w.record["n_rejected"]
        assert w.record["rms"] != before["rms"], (
            "putting the screened epochs back changed nothing — this control "
            "is supposed to move the estimate"
        )

    def test_the_command_reproduces_the_figure(self) -> None:
        """The invariant, on the new flag."""
        import shlex

        from gps_plot.detrend_workbench import (
            _build_parser,
            _override_settings,
            estimate_with_abort_fallback,
        )

        w = self._window()
        w.cb_use_flagged.setChecked(True)
        ns = _build_parser().parse_args(shlex.split(w.command.text())[1:])
        assert ns.stages == "S1,S2"
        settings = _override_settings(
            w.base_settings,
            w.sta,
            quiet=True,
            max_gap_years=ns.max_gap_years,
        )
        est, _ = estimate_with_abort_fallback(
            w.sta, w.yearf, w.data, w.sigma, settings=settings, stages=ns.stages
        )
        assert est is not None
        assert [round(v, 9) for v in est.record["rms"]] == [
            round(v, 9) for v in w.record["rms"]
        ]

    def test_drawing_the_grey_points_moves_no_fitted_quantity(self) -> None:
        w = self._window()
        w.refit()
        before_cmd, before_rms = w.command.text(), list(w.record["rms"])
        assert len(w.flag_scatters[0].getData()[0]) > 0, "nothing flagged to hide"

        w.cb_draw_flagged.setChecked(False)
        assert len(w.flag_scatters[0].getData()[0] or []) == 0
        assert w.command.text() == before_cmd, "a view toggle moved the command"
        assert list(w.record["rms"]) == before_rms, "a view toggle moved the fit"

    def test_the_two_controls_are_independent(self) -> None:
        """Hiding is not fitting, and fitting is not hiding."""
        w = self._window()
        w.cb_use_flagged.setChecked(True)
        assert w.cb_draw_flagged.isChecked(), "fitting them must not hide them"
        assert sum(w.record["n_rejected"]) == 0
        # with nothing flagged there is nothing grey left to draw
        assert len(w.flag_scatters[0].getData()[0] or []) == 0


class TestJointRefit:
    """Slice 4: staging identifies the structure, the joint solve reports it.

    Every stage after the first conditions on earlier values treated as
    known, so its uncertainties are conditional and the covariance between a
    held group and a free one is missing. The joint solve has neither
    problem. The staged→joint movement is the diagnostic: more than one
    sigma means two stages were fitting the same signal.
    """

    @staticmethod
    def _staged():
        w = TestQtPickerBorrowedFeatures._window()
        w.stage_regions[0].setRegion((2009.5443, 2021.0394))
        w.cb_stage.setChecked(True)
        return w

    def test_the_flag_is_only_emitted_with_a_stage_plan(self) -> None:
        """Unstaged IS joint; the flag would claim a choice never made."""
        w = TestQtPickerBorrowedFeatures._window()
        w.cb_final_joint.setChecked(True)
        assert "--final" not in w.command.text()
        assert not w.final_joint

    def test_the_shown_fit_is_the_joint_one(self) -> None:
        """Emitting --final joint while drawing the staged curve would put a
        command and a figure on screen that are not the same fit."""
        w = self._staged()
        staged_rms = list(w.record["rms"])
        w.cb_final_joint.setChecked(True)
        assert "--final joint" in w.command.text()
        assert w.record is not None
        assert list(w.record["rms"]) != staged_rms, (
            "the joint solve produced the staged numbers — nothing was re-fitted"
        )

    def test_the_movement_is_reported(self) -> None:
        w = self._staged()
        w.cb_final_joint.setChecked(True)
        assert w.joint_deltas, "no deltas computed"
        assert {"param", "staged", "joint", "delta", "sigma", "ratio"} <= set(
            w.joint_deltas[0]
        )
        assert "joint re-fit" in w.summary.toPlainText()

    def test_the_command_reproduces_the_figure(self) -> None:
        """The invariant, on the new flag: the whole point of --final joint
        changing the RUN rather than only the commit."""
        import shlex

        from gps_plot.detrend_workbench import _build_parser

        w = self._staged()
        w.cb_final_joint.setChecked(True)
        parts = shlex.split(w.command.text())
        ns = _build_parser().parse_args(parts[1:])
        assert ns.final == "joint"
        assert ns.stage and ns.hold, "the identifying plan is still emitted"

    def test_unticking_returns_the_staged_answer(self) -> None:
        w = self._staged()
        before = list(w.record["rms"])
        w.cb_final_joint.setChecked(True)
        w.cb_final_joint.setChecked(False)
        assert list(w.record["rms"]) == before
        assert "--final" not in w.command.text()
        assert w.joint_deltas == []


class TestStagedJointDeltas:
    """The delta arithmetic, without a window."""

    def test_identical_records_move_nothing(self) -> None:
        import json
        import pathlib

        from gps_plot.detrend_workbench import staged_joint_deltas

        p = pathlib.Path.home() / ".config/gpsconfig/detrend_params.json"
        if not p.is_file():  # pragma: no cover
            pytest.skip("no deployed detrend_params.json")
        recs = json.loads(p.read_text())
        recs = recs.get("stations", recs)
        rec = next(v for v in recs.values() if isinstance(v, dict))
        rows = staged_joint_deltas(rec, rec)
        assert rows, "no parameters compared"
        assert all(r["delta"] == 0.0 for r in rows)
        assert all(r["ratio"] == 0.0 for r in rows)

    def test_the_table_marks_what_crossed_a_sigma(self) -> None:
        from gps_plot.detrend_workbench import format_staged_joint_deltas

        rows = [
            {
                "component": "north",
                "param": "rate",
                "staged": 1.0,
                "joint": 3.0,
                "delta": 2.0,
                "sigma": 0.5,
                "ratio": 4.0,
            },
            {
                "component": "east",
                "param": "offset",
                "staged": 1.0,
                "joint": 1.01,
                "delta": 0.01,
                "sigma": 1.0,
                "ratio": 0.01,
            },
        ]
        text = format_staged_joint_deltas(rows)
        lines = text.splitlines()
        assert "rate" in lines[1], "the loudest row must come first"
        assert "stages disagreed" in lines[1]
        assert "stages disagreed" not in lines[2]


class TestInvariantSweep:
    """The one invariant, swept: the emitted command reproduces the figure.

    Not a review pass — a matrix. Every configuration the redesign added is
    driven offscreen, the command it emits is parsed back and re-estimated
    through the workbench's OWN entry points, and the two records are diffed
    elementwise. This is the method that caught violations 7 and 8 when
    reading the code did not, and slices 1-6 each added a new way for the
    figure and the command to come apart.

    Divergences that are ALLOWED, stated up front so the sweep neither
    false-alarms nor quietly ignores something real:

    - provenance only: ``refs`` (``uncert``, ``window_source``) and
      ``fitted_at`` — the picker records how it was driven, the CLI how it
      was invoked, and neither is a fitted quantity;
    - view state is never diffed at all: which card is expanded, the peel,
      and whether the grey points are drawn move nothing.

    Everything else — every group assignment, window, hold source, the model,
    the stage set and the final-solve choice — must match exactly.
    """

    FITTED = ("model", "param_names", "step_epochs", "n_epochs", "n_rejected")

    @staticmethod
    def _configure(w, *, cards=None, model_out=(), joint=False, flagged=False):
        """Drive the panel, then return it. No CLI knowledge here."""
        for name in model_out:
            w.model_in[name].setChecked(False)
        if cards is not None:
            w.stage_regions[0].setRegion((2009.5443, 2021.0394))
            w.cb_stage.setChecked(True)
            for index, groups in cards.items():
                card = w.stage_cards[index]
                for group, cb in card.groups.items():
                    cb.setChecked(group in groups)
        if flagged:
            w.cb_use_flagged.setChecked(True)
        if joint:
            w.cb_final_joint.setChecked(True)
        w.refit()
        return w

    @staticmethod
    def _replay(w):
        """Re-fit from the EMITTED command, through the workbench's parser."""
        import shlex

        from geo_dataread.stage_plan import build_stage_plan
        from gps_plot.detrend_workbench import (
            _build_parser,
            _override_settings,
            estimate_with_abort_fallback,
        )

        ns = _build_parser().parse_args(shlex.split(w.command.text())[1:])
        plan = build_stage_plan(ns.stage, ns.hold) if ns.stage else None
        if ns.final == "joint":
            # what --final joint means: the plan identified, it does not fit
            plan = None
        settings = _override_settings(
            w.base_settings,
            w.sta,
            quiet=True,
            segments=tuple(_parse_segments(ns.segment)) or None,
            steps=tuple(float(s) for s in ns.step) or None,
            max_gap_years=ns.max_gap_years,
        )
        est, _ = estimate_with_abort_fallback(
            w.sta,
            w.yearf,
            w.data,
            w.sigma,
            settings=settings,
            terms=tuple(ns.term) or None,
            stage_plan=plan,
            model=ns.model,
            stages=ns.stages,
        )
        return est

    @pytest.mark.parametrize(
        "label, kwargs",
        [
            ("unstaged", {}),
            ("unstaged, linear only", {"model_out": ("periodic",)}),
            ("unstaged, periodic only", {"model_out": ("secular",)}),
            ("preset", {"cards": {}}),
            ("secular re-estimated late", {"cards": {1: ("secular", "step")}}),
            ("periodic freed late", {"cards": {1: ("periodic", "step")}}),
            ("linear only, staged", {"cards": {}, "model_out": ("periodic",)}),
            ("flagged epochs fitted", {"flagged": True}),
            ("flagged epochs, staged", {"cards": {}, "flagged": True}),
            ("joint final", {"cards": {}, "joint": True}),
            (
                "joint final, freed late",
                {"cards": {1: ("secular", "step")}, "joint": True},
            ),
            ("joint final, flagged", {"cards": {}, "joint": True, "flagged": True}),
        ],
    )
    def test_the_command_reproduces_the_figure(self, label, kwargs) -> None:
        w = self._configure(TestQtPickerBorrowedFeatures._window(), **kwargs)
        if w.record is None:
            pytest.skip(f"{label}: refused — covered by the refusal tests")
        est = self._replay(w)
        assert est is not None, f"{label}: the emitted command produced no record"
        for key in self.FITTED:
            assert est.record[key] == w.record[key], f"{label}: {key} diverged"
        for c, (a, b) in enumerate(
            zip(est.record["components"], w.record["components"], strict=True)
        ):
            assert a["params"] == pytest.approx(b["params"], rel=0, abs=1e-9), (
                f"{label}: component {c} parameters diverged"
            )


def _parse_segments(specs):
    out = []
    for spec in specs or ():
        lo, _, hi = spec.partition(":")
        out.append((float(lo) if lo else None, float(hi) if hi else None))
    return out


class TestTheCurveIsTheActiveStage:
    """A stage that fits two groups must not be drawn fitting three.

    Reported from the GUI 2026-08-22: with `clean — linear, periodic` active
    and a step stage below it, the figure drew lin+per+step over an orange
    window containing no step. The PARAMETERS were right the whole time --
    stage 1's rate is identical to the same fit run standalone on its window
    -- but the curve is what gets believed, and it showed a model the stage
    had not been asked for.
    """

    @staticmethod
    def _staged():
        w = TestQtPickerBorrowedFeatures._window()
        w.stage_regions[0].setRegion((2009.1188, 2020.8294))
        w.cb_stage.setChecked(True)
        return w

    def test_stage_one_draws_no_step(self) -> None:

        w = self._staged()
        assert [c.name for c in w.stage_cards] == ["clean", "st"]
        assert w.stage_cards[0].estimates() == ["secular", "periodic"]
        assert "step" in w.stage_cards[1].estimates()
        assert w.record is not None and w.record["step_epochs"], "no step in the model"

        _, curve = w.fit_curves[0].getData()
        assert _jump_count(curve) == 0, (
            "the active stage estimates linear+periodic, but the curve drawn "
            "for it contains a discontinuity — that is the step stage's term"
        )

    def test_stage_two_draws_only_the_step(self) -> None:

        w = self._staged()
        w.stage_cards[1].box.setChecked(True)
        w.refit()
        _, curve = w.fit_curves[0].getData()
        n = _jump_count(curve)
        assert n == 1, f"expected the step alone, got {n} jumps"

    def test_the_title_names_what_is_drawn(self) -> None:
        w = self._staged()
        title = w.plots[0].titleLabel.text
        assert "clean" in title and "linear" in title and "periodic" in title
        assert "step" not in title

    def test_the_joint_view_shows_the_whole_model(self) -> None:
        """A joint solve has no stage structure, so nothing is peeled."""

        w = self._staged()
        w.cb_final_joint.setChecked(True)
        assert "minus" not in w.plots[0].getAxis("left").labelText
        _, curve = w.fit_curves[0].getData()
        assert _jump_count(curve) == 1, "the joint model's step is not drawn"

    def test_the_parameters_were_never_wrong(self) -> None:
        """The bug was the curve, not the fit — pin that down.

        Not bit-identical, and the reason is worth recording: the staged run
        screens outliers over the WHOLE domain (SELF: [32, 24, 11]) while the
        standalone windowed fit screens only inside its window ([14, 7, 1]),
        so the two see slightly different epoch sets. The rates agree to
        ~7e-4 mm/yr. The tolerance below is two orders of magnitude tighter
        than the difference this test exists to catch -- stage 1 silently
        picking up the full-span rate, which is 0.08 mm/yr away.
        """
        from gps_plot.detrend_workbench import _override_settings, estimate_record

        w = self._staged()
        standalone = estimate_record(
            "SELF",
            w.yearf,
            w.data,
            w.sigma,
            settings=_override_settings(
                w.base_settings,
                "SELF",
                quiet=True,
                segments=((2009.1188, 2020.8294),),
                max_gap_years=2.0,
            ),
        )
        assert standalone is not None
        for c in range(3):
            staged_rate = w.record["components"][c]["params"][1]
            alone_rate = standalone.record["components"][c]["params"][1]
            assert staged_rate == pytest.approx(alone_rate, abs=5e-3), (
                f"component {c}: the staged stage-1 rate is not the "
                f"standalone fit on the same window"
            )
        assert w.record["n_rejected"] != standalone.record["n_rejected"], (
            "the epoch sets are expected to differ — if they stop differing, "
            "the tolerance above can be tightened to exact"
        )
