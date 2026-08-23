"""The Qt picker: two phases, one invariant.

The invariant is unchanged from every earlier shape of this window — **the
emitted command reproduces the figure** — and so is the failure mode it
guards against: a second place assembling the same decision. What changed is
that the panel now has the model's own two halves as two modes, so the
commands are two pure functions and can be tested as plain strings.
"""

from __future__ import annotations

import shlex

import pytest

from gps_plot.detrend_picker_qt import (
    MODE_BACKGROUND,
    MODE_EVENTS,
    background_command,
    events_command,
    model_equation,
)


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


class TestTheTwoCommands:
    """Both spellings, as plain strings — no Qt, no data."""

    def test_the_background_emits_one_segment_per_interval(self) -> None:
        """The union is the point: one interval cannot span an event."""
        cmd = background_command("SELF", segments=[(2001.5, 2008.4), (2009.5, 2020.83)])
        parts = shlex.split(cmd)
        assert [parts[i + 1] for i, a in enumerate(parts) if a == "--segment"] == [
            "2001.5:2008.4",
            "2009.5:2020.83",
        ]

    def test_the_default_model_stays_out_of_the_command(self) -> None:
        assert "--model" not in background_command("SELF", model="lineperiodic")
        assert "--model linear" in background_command("SELF", model="linear")

    def test_saving_is_opt_in(self) -> None:
        assert "--save-secular" not in background_command("SELF")
        assert "--save-secular" in background_command("SELF", save=True)

    def test_the_events_command_holds_both_background_groups(self) -> None:
        cmd = events_command("SELF", free=["step"], hold_from="self")
        assert "--hold secular=store:self" in cmd
        assert "--hold periodic=store:self" in cmd
        assert "--stage ev:step" in cmd

    def test_borrowing_names_the_donor_station(self) -> None:
        """The legacy UseSTA, in the grammar's own spelling."""
        cmd = events_command("OLAC", free=["step"], hold_from="DYNG")
        assert "--hold periodic=store:DYNG" in cmd

    def test_store_is_not_donor(self) -> None:
        """They resolve against different objects, so they are spelled apart.

        `donor:` reads a finished record (s(t) + events); `store:` reads the
        background store. Inferring between them would let a rename change
        which object a stored hold points at.
        """
        assert "store:" in events_command("SELF", free=["step"])
        assert "donor:" not in events_command("SELF", free=["step"])

    def test_committing_is_opt_in(self) -> None:
        assert "--commit" not in events_command("SELF", free=["step"])
        assert "--commit" in events_command("SELF", free=["step"], commit=True)

    @pytest.mark.parametrize(
        "cmd",
        [
            background_command("SELF", segments=[(2001.5, 2008.4)], save=True),
            events_command(
                "SELF",
                free=["step"],
                steps=[2011.25],
                segment=(2001.5, 2026.0),
                commit=True,
            ),
        ],
    )
    def test_the_workbench_can_parse_what_the_picker_emits(self, cmd) -> None:
        """The invariant's floor: a command that will not parse reproduces
        nothing, and `--uncert 12.0` once emitted exactly that."""
        from gps_plot.detrend_workbench import _build_parser

        ns = _build_parser().parse_args(shlex.split(cmd)[1:])
        assert ns.station.upper() == "SELF"

    def test_the_emitted_holds_build_a_real_plan(self) -> None:
        from geo_dataread.stage_plan import StoreRef, build_stage_plan
        from gps_plot.detrend_workbench import _build_parser

        cmd = events_command("SELF", free=["step"], hold_from="self")
        ns = _build_parser().parse_args(shlex.split(cmd)[1:])
        plan = build_stage_plan(ns.stage, ns.hold)
        assert plan.stages[0].held == {
            "secular": StoreRef(None),
            "periodic": StoreRef(None),
        }


class TestModelEquation:
    """The panel shows the general FORM; the numbers go to the table."""

    def test_it_is_read_off_the_record(self) -> None:
        eq = model_equation(
            ["offset", "rate", "cos_annual", "sin_annual", "step_amp_1", "log_amp_1"]
        )
        assert eq.startswith("x(t) = a₀ + a₁·(t−t₀)")
        assert "h1·H(t−t1)" in eq, "a declared step has no symbol"
        assert "ln(1+(t−tₑ)/τ)" in eq, "the log transient is not spelled out"

    def test_a_linear_model_says_so(self) -> None:
        eq = model_equation(["offset", "rate"])
        assert "cos" not in eq and "sin" not in eq


class TestTheWindow:
    """Driven offscreen. A GUI, so test what can be tested without a screen."""

    @staticmethod
    def _window(sta="SELF", gap=2.0, tmp_state=None):
        import os

        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        if tmp_state is not None:
            os.environ["XDG_STATE_HOME"] = str(tmp_state)
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
        settings = resolve_fit_settings(sta, None, FitDefaults(max_gap_years=gap))
        return PickerWindow(
            sta,
            np.asarray(yearf, float),
            np.atleast_2d(np.asarray(data, float)),
            np.atleast_2d(np.asarray(sigma, float)),
            settings,
            max_gap_years=gap,
            uncert=10,
        )

    def test_it_opens_in_the_background_phase(self, tmp_path) -> None:
        """s(t) must exist before events can be estimated against it."""
        w = self._window(tmp_state=tmp_path)
        assert w.mode.currentText() == MODE_BACKGROUND
        assert w.bg_box.isVisibleTo(w.win) and not w.ev_box.isVisibleTo(w.win)

    def test_intervals_reach_the_fit_and_the_command(self, tmp_path) -> None:
        """The whole contract, on the control the redesign added."""
        w = self._window(tmp_state=tmp_path)
        w.add_segment((2001.5, 2008.40))
        w.add_segment((2009.5, 2020.83))
        assert w.segments() == [(2001.5, 2008.4), (2009.5, 2020.83)]
        assert w.record is not None
        cmd = w.command.text()
        assert "--segment 2001.5:2008.4" in cmd
        assert "--segment 2009.5:2020.83" in cmd
        # and the fit really is the one that command describes
        assert w.record["rms"][0] < 2.0, w.record["rms"]

    def test_the_union_is_not_the_hull(self, tmp_path) -> None:
        """Two intervals either side of an event must EXCLUDE the event.

        If the picker collapsed them to a hull it would re-include the
        postseismic months, and the whole reason for the union is gone.
        """
        w = self._window(tmp_state=tmp_path)
        w.add_segment((2001.5, 2008.40))
        w.add_segment((2009.5, 2020.83))
        segs = w.record.get("segments")
        assert segs and len(segs) == 2, f"collapsed to {segs}"

    def test_both_terms_off_is_refused_not_guessed(self, tmp_path) -> None:
        w = self._window(tmp_state=tmp_path)
        w.cb_linear.setChecked(False)
        w.cb_periodic.setChecked(False)
        assert w.record is None
        assert "refused" in w.summary.toPlainText().lower()

    def test_the_events_phase_swaps_the_controls_and_the_view(self, tmp_path) -> None:
        w = self._window(tmp_state=tmp_path)
        w.mode.setCurrentText(MODE_EVENTS)
        assert w.ev_box.isVisibleTo(w.win) and not w.bg_box.isVisibleTo(w.win)
        # the view events are READ in, chosen for you on arrival
        assert w.view.currentText() == "data − s(t)"
        assert "--hold secular=store:self" in w.command.text()

    def test_events_without_a_saved_background_refuse_with_the_fix(
        self, tmp_path
    ) -> None:
        """Never silently estimate what the operator asked to hold."""
        w = self._window(tmp_state=tmp_path)
        w.mode.setCurrentText(MODE_EVENTS)
        if w.record is not None:  # pragma: no cover - a store already exists
            pytest.skip("SELF already has a saved background on this host")
        text = w.summary.toPlainText()
        assert "no saved background" in text
        assert "save s(t)" in text, "the refusal must name the way out"

    def test_a_view_change_moves_no_fitted_quantity(self, tmp_path) -> None:
        """DISPLAY ONLY, on the same terms as --hide-outliers."""
        w = self._window(tmp_state=tmp_path)
        w.add_segment((2001.5, 2020.83))
        before_cmd, before_rms = w.command.text(), list(w.record["rms"])
        w.view.setCurrentText("data − f(t)")
        assert w.command.text() == before_cmd
        assert list(w.record["rms"]) == before_rms
        w.cb_draw_flagged.setChecked(False)
        assert w.command.text() == before_cmd
        assert list(w.record["rms"]) == before_rms

    def test_fitting_the_flagged_epochs_does_move_it(self, tmp_path) -> None:
        """The one in the run box is not a view control, and says so."""
        w = self._window(tmp_state=tmp_path)
        w.add_segment((2001.5, 2020.83))
        before = list(w.record["n_rejected"])
        assert sum(before) > 0, "no screening to switch off here"
        w.cb_use_flagged.setChecked(True)
        assert "--stages S1,S2" in w.command.text()
        assert sum(w.record["n_rejected"]) == 0

    def test_the_session_round_trips(self, tmp_path) -> None:
        w = self._window(tmp_state=tmp_path)
        w.add_segment((2003.0, 2007.0))
        w.add_segment((2011.0, 2019.0))
        w.save_session()
        before = w.command.text()

        w2 = self._window(tmp_state=tmp_path)
        assert w2.segments() == [(2003.0, 2007.0), (2011.0, 2019.0)]
        assert w2.command.text() == before

    def test_a_corrupt_session_degrades_instead_of_dying(self, tmp_path) -> None:
        """This runs at LAUNCH: a raise here leaves no way in to fix the file."""
        w = self._window(tmp_state=tmp_path)
        w.save_session()
        w._session_path().write_text("{ not json")
        w2 = self._window(tmp_state=tmp_path)
        assert "NOT restored" in w2.summary.toPlainText()
        assert w2.segments() == []


class TestTheTwoPhaseWorkflow:
    """The thing the redesign exists for, driven end to end.

    Estimate s(t) on the clean intervals either side of the event, save it,
    then hold it and estimate only the events. The background must come back
    EXACTLY — it is held, not re-fitted — and data − f(t) must go to zero.
    """

    @staticmethod
    def _scratch_config(tmp_path, monkeypatch):
        """A disposable config tree, dereferenced.

        `~/gps-data/testcfg` is symlinks INTO the deployed config, so a
        shallow copy still writes through to the live document. copytree with
        symlinks=False is what makes this safe.
        """
        import shutil
        from pathlib import Path

        src = Path.home() / ".config/gpsconfig"
        if not (src / "analysis.yaml").is_file():  # pragma: no cover
            pytest.skip("no deployed gpsconfig on this host")
        dst = tmp_path / "cfg"
        shutil.copytree(src, dst, symlinks=False)
        monkeypatch.setenv("GPS_CONFIG_PATH", str(dst))
        monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
        return dst

    def test_save_then_hold_reproduces_the_background_exactly(
        self, tmp_path, monkeypatch
    ) -> None:
        from geo_dataread.secular_store import read_secular

        cfg = self._scratch_config(tmp_path, monkeypatch)
        w = TestTheWindow._window()
        w.add_segment((2001.5, 2008.40))
        w.add_segment((2009.5, 2020.83))
        assert w.record is not None
        background = [round(c["params"][1], 6) for c in w.record["components"]]

        w.save_secular()
        entry = read_secular(cfg / "analysis.yaml")["SELF"]
        assert "step_amp_1" not in entry.param_names, "an event leaked into s(t)"
        assert entry.segments == ((2001.5, 2008.4), (2009.5, 2020.83))

        w.mode.setCurrentText(MODE_EVENTS)
        assert w.record is not None, w.summary.toPlainText()[:200]
        held = [round(c["params"][1], 6) for c in w.record["components"]]
        assert held == background, "the held background was re-fitted"
        # and the event it was held in order to measure
        step = float(w.record["components"][0]["params"][-1])
        assert -160.0 < step < -140.0, f"SELF's 2008 coseismic came out {step}"

    def test_data_minus_f_goes_to_zero(self, tmp_path, monkeypatch) -> None:
        """The operator's own acceptance criterion, asserted."""
        import numpy as np

        self._scratch_config(tmp_path, monkeypatch)
        w = TestTheWindow._window()
        w.add_segment((2001.5, 2008.40))
        w.add_segment((2009.5, 2020.83))
        w.save_secular()
        w.mode.setCurrentText(MODE_EVENTS)
        assert w.record is not None

        w.view.setCurrentText("data − f(t)")
        assert "− f(t)" in w.plots[0].getAxis("left").labelText
        _x, residual = w.kept_scatters[0].getData()
        residual = np.asarray(residual, dtype=float)
        assert abs(float(np.nanmean(residual))) < 0.5, "data − f(t) is not centred"
        # and it is genuinely flatter than the series it came from
        assert float(np.nanstd(residual)) < float(np.nanstd(w.data[0])) / 10.0

    def test_the_saved_background_carries_the_offset(
        self, tmp_path, monkeypatch
    ) -> None:
        """Unlike the legacy CSV, and for the reason the whole redesign exists.

        Detrending does not need an offset. HOLDING s(t) to measure a step
        does: with the level unanchored the step is measured against a
        floating background, and with it anchored on the wrong side of the
        event the step cannot be measured at all — SELF returned 0.0 mm
        against a true −150.8 (2026-08-22).
        """
        from geo_dataread.secular_store import read_secular

        cfg = self._scratch_config(tmp_path, monkeypatch)
        w = TestTheWindow._window()
        w.add_segment((2001.5, 2008.40))
        w.add_segment((2009.5, 2020.83))
        w.save_secular()
        entry = read_secular(cfg / "analysis.yaml")["SELF"]
        assert entry.param_names[0] == "offset"
        assert len(entry.components["north"]) == len(entry.param_names)

    def test_saving_with_no_fit_says_so(self, tmp_path, monkeypatch) -> None:
        self._scratch_config(tmp_path, monkeypatch)
        w = TestTheWindow._window()
        w.cb_linear.setChecked(False)
        w.cb_periodic.setChecked(False)
        assert w.record is None
        w.save_secular()
        assert "nothing to save" in w.summary.toPlainText()


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


class TestTransientsAreOutOfScope:
    """Taken back out on purpose, after trying — so the removal is pinned.

    Two measurements decided it. A short-τ saturating exp is 0.993 correlated
    with the step it sits on, so over an 18-year record the pair is not
    separable and only their SUM is trustworthy. And on SELF what looks like
    a transient across the 2008 event is a rate change — pre 2.17, post 3.05
    mm/yr on north — which is not a transient at all and breaks the
    one-regime assumption the background rests on.

    `gps-detrend-workbench --term` still exists. The picker does not offer
    what it cannot help an operator judge.
    """

    def test_the_events_command_carries_no_term(self) -> None:
        cmd = events_command("SELF", free=["step"], steps=[2008.4085])
        assert "--term" not in cmd
        assert "--stage ev:step" in cmd

    def test_the_window_has_no_transient_control(self, tmp_path) -> None:
        w = TestTheWindow._window(tmp_state=tmp_path)
        for gone in ("cb_term", "kind", "tau", "onset_lines", "btn_refine"):
            assert not hasattr(w, gone), f"{gone} survived the removal"

    def test_the_event_stage_estimates_steps_only(self, tmp_path) -> None:
        w = TestTheWindow._window(tmp_state=tmp_path)
        w.mode.setCurrentText(MODE_EVENTS)
        assert "transient" not in w.command.text()

    def test_the_whole_model_view_still_peels_a_transient(self, tmp_path) -> None:
        """A record fitted ELSEWHERE may carry one, and data − f(t) must
        subtract the whole model or it is not what it says it is."""
        from gps_plot.detrend_picker_qt import PEEL_ALL

        assert "transient" in PEEL_ALL
