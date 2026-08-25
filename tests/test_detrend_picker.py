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
    declare_spec,
    events_command,
    model_equation,
    store_command,
)


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


class TestStoreCommand:
    """The store dialog's command assembly — pure, no Qt, no data.

    ``store_command`` is what the dialog shows and what the store run parses:
    a declared pick becomes a ``--declare-step`` and its ``--step`` flag is
    dropped (the floor carries it once the declaration lands), ``--force``
    replaces an existing stored record.
    """

    def test_a_declared_step_becomes_declare_and_loses_step(self) -> None:
        base = events_command("SELF", free=["step"], steps=[2008.4085], commit=True)
        assert "--step 2008.4085" in base
        cmd = store_command(base, [(2008.4085, "earthquake", "Ölfus M6.3")])
        assert "--step 2008.4085" not in cmd, "a declared pick must not stay fit-only"
        assert "--declare-step" in cmd
        assert "kind=earthquake" in cmd and "comment=Ölfus M6.3" in cmd

    def test_an_unticked_pick_stays_a_step(self) -> None:
        base = events_command("SELF", free=["step"], steps=[2008.4085], commit=True)
        assert store_command(base) == base

    def test_only_the_declared_epoch_loses_its_flag(self) -> None:
        base = events_command(
            "SELF", free=["step"], steps=[2008.4085, 2010.5], commit=True
        )
        cmd = store_command(base, [(2008.4085, "earthquake", "")])
        assert "--step 2010.5" in cmd, "the undeclared pick must survive"
        assert "--step 2008.4085" not in cmd

    def test_declare_spec_carries_epoch_and_date(self) -> None:
        spec = declare_spec(2008.4085, "earthquake", "Ölfus M6.3")
        assert "epoch=2008.408500" in spec
        assert "date=2008-05-29" in spec, "the date must be the noon-convention day"
        assert "kind=earthquake" in spec and "comment=Ölfus M6.3" in spec

    def test_declare_spec_round_trips_through_the_workbench_parser(self) -> None:
        from gps_plot.detrend_workbench import parse_declare_step

        spec = declare_spec(2008.4085, "equipment", "antenna swap")
        rec = parse_declare_step(spec, "SELF")
        assert rec.marker == "SELF" and rec.kind == "equipment"
        assert abs(rec.epoch_yearf - 2008.4085) < 1e-4
        assert rec.date == "2008-05-29"

    def test_force_appends_the_replace_flag(self) -> None:
        base = events_command("SELF", free=["step"], steps=[2008.4085], commit=True)
        assert "--force" not in base
        assert store_command(base, force=True).endswith("--force")

    def test_near_declared_step_flags_a_twin(self) -> None:
        """The Ölfus case: a pick 4 days from the declared earthquake must be
        recognised as the SAME event, not stored as a second step in one gap."""
        from gps_plot.detrend_picker_qt import near_declared_step

        assert near_declared_step(2008.3973, [2008.4085]) == 2008.4085
        # distinct events, months apart, are not twins
        assert near_declared_step(2010.5, [2008.4085]) is None
        # an exact match is a twin (and merge_station_steps would dedup it)
        assert near_declared_step(2008.4085, [2008.4085]) == 2008.4085


class TestSegmentAndCoverageGuards:
    """Two states the window used to restore silently: overlapping clean
    intervals (the fit refuses them → "NO RECORD" on every restart), and a
    held background that never spanned the step it was meant to measure
    (the offset comes out ~0)."""

    def test_normalize_drops_the_overlap_keeps_the_chain(self) -> None:
        from gps_plot.detrend_picker_qt import normalize_segments

        kept, dropped = normalize_segments(
            [(2002.162, 2008.4448), (2008.4206, 2009.3298), (2009.3294, 2021.4796)]
        )
        # the middle interval overlaps the first, so it is dropped; the
        # post-event one survives — the pre+post structure the user meant
        assert kept == [(2002.162, 2008.4448), (2009.3294, 2021.4796)]
        assert dropped == [(2008.4206, 2009.3298)]

    def test_normalize_is_idempotent_on_clean_segments(self) -> None:
        from gps_plot.detrend_picker_qt import normalize_segments

        kept, dropped = normalize_segments([(2009.0, 2015.0), (2002.0, 2008.0)])
        assert kept == [(2002.0, 2008.0), (2009.0, 2015.0)]  # sorted
        assert dropped == []

    def test_adjacent_segments_are_not_an_overlap(self) -> None:
        from gps_plot.detrend_picker_qt import normalize_segments

        kept, dropped = normalize_segments([(2002.0, 2008.5), (2008.5, 2020.0)])
        assert dropped == [] and len(kept) == 2

    def test_step_outside_background_names_the_side(self) -> None:
        from gps_plot.detrend_picker_qt import step_outside_background

        # SELF's coseismic is BEFORE a post-event-only background
        gaps = step_outside_background([2008.4085], [(2009.31, 2021.28)])
        assert gaps == [
            (2008.4085, "before the background's earliest data (2009.3100)")
        ]
        # a step inside the span is fine
        assert step_outside_background([2015.0], [(2009.31, 2021.28)]) == []
        # and a step after it is named too
        (gap,) = step_outside_background([2025.0], [(2009.31, 2021.28)])
        assert "after" in gap[1]


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


class TestTheBackgroundCanBeCommitted:
    """A station with no events must still reach detrend_params.json.

    Measured hole 2026-08-23: RHOF has no declared step, so the events phase
    reports "nothing to estimate" and produces no record — leaving no route
    from the picker to the document plot-gps-timeseries reads. Its background
    IS its whole model, so it commits from the background phase.

    The two writes stay DISTINCT even in one command: --save-secular writes
    s(t) to the store as a reusable component, --commit writes the finished
    record. Conflating them is the hazard the separate stores exist to
    prevent — a background committed for a station that HAS events would make
    production serve a series with its offset still in it.
    """

    def test_the_background_phase_emits_both_writes(self, tmp_path) -> None:
        w = TestTheWindow._window(tmp_state=tmp_path)
        w.add_segment((2002.0, 2020.0))
        assert w.record is not None
        cmd = w._commit_command()
        assert "--save-secular" in cmd, "the reusable s(t) is not written"
        assert "--commit" in cmd, "the finished record is not written"
        assert "--stage" not in cmd, "the background phase declares no stage"

    def test_the_events_phase_commits_the_event_stage(self, tmp_path) -> None:
        w = TestTheWindow._window(tmp_state=tmp_path)
        w.mode.setCurrentText(MODE_EVENTS)
        cmd = w._commit_command()
        assert "--stage" in cmd and "--commit" in cmd
        assert "--save-secular" not in cmd, "the events phase stores only the record"

    def test_a_stepless_station_is_committed_from_the_background(
        self, tmp_path
    ) -> None:
        w = TestTheWindow._window(sta="RHOF", tmp_state=tmp_path)
        w.add_segment((2002.0, 2020.0))
        if w.record is None:  # pragma: no cover
            pytest.skip("RHOF not fittable here")
        if w.record.get("step_epochs"):  # pragma: no cover
            pytest.skip("RHOF gained a declared step; pick another station")
        cmd = w._commit_command()
        assert "--save-secular" in cmd and "--commit" in cmd

    def test_nothing_to_store_is_said_not_crashed(self, tmp_path) -> None:
        w = TestTheWindow._window(tmp_state=tmp_path)
        w.cb_linear.setChecked(False)
        w.cb_periodic.setChecked(False)
        assert w.record is None
        w.store()
        assert "nothing to store" in w.summary.toPlainText()


class TestTheStoreRun:
    """The store button runs the command in-process through
    ``detrend_workbench.main``.  ``main`` takes the RAW argv (parse_args
    drops argv[0]), but the command spells its own program name — so the
    store drops it.  Getting that wrong means the program name becomes the
    ``station`` positional and every store fails with an argparse error; the
    offscreen picker never surfaced it because the dialog is modal.
    """

    def test_the_store_drops_the_program_name(self, tmp_path, monkeypatch) -> None:
        import gps_plot.detrend_workbench as wb

        w = TestTheWindow._window(tmp_state=tmp_path)
        received: dict[str, list[str]] = {}

        def fake_main(argv):
            received["argv"] = list(argv)
            return 0

        monkeypatch.setattr(wb, "main", fake_main)
        w._run_store("gps-detrend-workbench RHOF --commit", [])
        assert received["argv"][0] == "RHOF", (
            "the program name must be dropped before main() — it is argv[0]"
        )
        assert "gps-detrend-workbench" not in received["argv"]

    def test_a_refused_store_is_reported_not_silent(
        self, tmp_path, monkeypatch
    ) -> None:
        import gps_plot.detrend_workbench as wb

        w = TestTheWindow._window(tmp_state=tmp_path)
        monkeypatch.setattr(wb, "main", lambda argv: 3)
        w._run_store("gps-detrend-workbench RHOF --commit", [])
        assert "FAILED" in w.summary.toPlainText()
        assert "exit 3" in w.summary.toPlainText()


class TestAutoSaveBackground:
    """Switching to events auto-saves s(t) when it differs from the on-screen
    segments — the "what you see gets held" contract."""

    def test_switch_to_events_auto_saves_when_background_differs(
        self, tmp_path, monkeypatch
    ) -> None:
        from geo_dataread.secular_store import read_secular

        cfg = TestTheTwoPhaseWorkflow._scratch_config(tmp_path, monkeypatch)
        w = TestTheWindow._window(tmp_state=tmp_path)
        w.add_segment((2002.0, 2008.35))
        w.add_segment((2009.0, 2020.83))
        assert w.record is not None

        # initial saved background (from the deployed post-event copy) differs
        saved_before = read_secular(cfg / "analysis.yaml").get("SELF")
        on_screen = w.segments()
        assert saved_before is None or tuple(on_screen) != tuple(
            (float(s[0]), float(s[1])) for s in (saved_before.segments or ())
        ), "test only valid when saved bg ≠ on-screen segments"

        w.mode.setCurrentText(MODE_EVENTS)

        # auto-save wrote the on-screen segments
        saved_after = read_secular(cfg / "analysis.yaml")["SELF"]
        assert tuple(on_screen) == tuple(
            (float(s[0]), float(s[1])) for s in saved_after.segments
        )

        # the offset is NOT ~0 (it was measured against a spanning background)
        amp = abs(float(w.record["components"][0]["params"][-1]))
        assert amp > 50.0, f"step_amp {amp:.1f} — not the ~0 mm of an extrapolated bg"
        assert "auto-saved" in w.summary.toPlainText()
