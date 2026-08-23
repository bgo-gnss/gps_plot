"""Qt picker for the detrend lane — two phases, because the model has two.

    f(t) = s(t) + Σ step(t − tₖ)

``s(t)`` is the BACKGROUND: rate plus annual and semiannual, estimated once
on clean data, saved, and reused — held fixed while events are estimated
against it, and borrowed by stations too short or too noisy to constrain a
seasonal of their own.  The events are short-lived departures from it.

Those are different acts on different data, and the previous panel made them
one.  It offered a general N-stage editor in which the operator had to
rediscover, per station, that a background held from a window on ONE side of
an event pins the level to that side and leaves the step nothing to measure
(SELF, 2026-08-22: step estimated at 0.0 mm against a true −150.8).  This
window has the two phases as two modes, and their order IS the workflow:

**background** — pick the clean intervals (a UNION, usually one either side
of the event), fit lin+per on them, look at it, ``save s(t)``.  The union is
the point: one interval cannot span an event, and a background fitted only
after one extrapolates backwards through it.

**events** — the saved s(t) is held, the plot shows ``data − s(t)``, and
the offsets are estimated against a background that no longer moves.
``data − f(t)`` goes to zero when the model is right.

Transients are deliberately NOT here.  They were tried and taken back out:
a short-τ saturating exp is 0.993 correlated with the step it sits on, so
over an 18-year record the two are not separable, and what looks like a
transient across an event is often a RATE CHANGE — which breaks the
one-regime assumption the background rests on and needs a term the grammar
does not have.  ``gps-detrend-workbench --term`` still exists for anyone who
wants one; this window does not offer what it cannot help you judge.

The invariant is unchanged and is still the whole promise: **the emitted
command reproduces the figure.**  Every divergence found in this window has
had one shape — a second place assembling the same decision — so the two
commands come from two pure functions below, the run-flag tail from
``detrend_workbench.run_flags`` (shared with the marimo picker), the settings
from the workbench's own ``_override_settings``, and what a background IS
from ``geo_dataread.secular_store``.

Requires ``pyqtgraph`` and ``PySide6``, which live in the DEV group — a local
development tool, never a production dependency.
"""

from __future__ import annotations

import os
import shlex
import sys
from collections.abc import Sequence
from typing import Any

# The workbench's own default, IMPORTED rather than mirrored so the emitted
# command omits the flag exactly when the workbench would default to the same
# screen. A copy could drift; this cannot.
from gps_plot.detrend_workbench import (
    WORKBENCH_UNCERT_DEFAULT,
    estimate_with_abort_fallback,
    group_contribution,
    run_flags,
    trajectory_curve,
)

__all__ = [
    "EVENT_STAGE",
    "MODE_BACKGROUND",
    "MODE_EVENTS",
    "background_command",
    "events_command",
    "main",
    "model_equation",
]

# --- the cleaned-view vocabulary, kept in step with timesmatplt's constants
KEPT_COLOR = (214, 39, 40)  # red    — in the fit
FLAG_COLOR = (150, 150, 150)  # grey   — flagged by the fit
OUTSIDE_COLOR = (150, 150, 150)  # grey   — flagged outside the window
PROV_FACE = (255, 215, 0)  # gold   — provisional, kept
PROV_EDGE = (184, 134, 11)
FIT_COLOR = (31, 119, 180)  # blue   — the model
SEGMENT_COLOR = (255, 165, 0, 60)  # orange — a clean interval for s(t)
DOMAIN_COLOR = (100, 150, 220, 45)  # blue   — the events fit domain
STEP_COLOR = (140, 20, 20)
TOS_EVENT_COLOR = (0, 100, 0)
SEISMIC_EVENT_COLOR = (139, 0, 0)
STALE_COLOR = (150, 150, 150)  # a curve whose fit was REFUSED
COMPONENTS = ("North", "East", "Up")

#: The two phases. Not a preference — s(t) must exist before events can be
#: estimated against it, and the mode makes that order visible rather than
#: leaving it to be rediscovered per station.
MODE_BACKGROUND = "background — s(t)"
MODE_EVENTS = "events — offsets"

#: The GUI says `linear`; every emitted flag still says `secular`. The stage
#: grammar's `secular` names the linear term alone, but "secular" properly
#: names the long-term background as a whole, which invites misreading.
GROUP_LABELS = {
    "secular": "linear",
    "periodic": "periodic",
    "step": "step",
    "transient": "transient",
}

#: The groups s(t) is made of, and the groups that are events. A step is
#: what gets estimated AGAINST the background, never part of it — the same
#: split `geo_dataread.secular_store` enforces when saving.
#:
#: TRANSIENTS ARE OUT OF SCOPE HERE, deliberately and after trying. They are
#: not a missing feature but an unfinished conversation: on SELF a short-τ
#: saturating exp is 0.993 correlated with the step it sits on, so the two
#: cannot be separated over an 18-year record, and the rate change either
#: side of the event is not a transient at all — it breaks the one-regime
#: assumption this whole two-phase design rests on. `--term` is still there
#: in the workbench for anyone who wants it; the picker does not offer what
#: it cannot help you judge.
BACKGROUND_GROUPS = ("secular", "periodic")
EVENT_GROUPS = ("step",)
#: Peeled for the `data − f(t)` view: a record fitted elsewhere may carry a
#: transient even though this window will not create one, and the view must
#: still subtract the WHOLE model.
PEEL_ALL = ("secular", "periodic", "step", "transient")

#: `station_record_from_arrays`' own default, so the emitted command stays
#: clean when the operator has not moved away from it.
DEFAULT_MODEL = "lineperiodic"

#: (linear, periodic) → the stored `--model`. Both off has no spelling:
#: every value in the vocabulary carries at least one of them.
MODEL_BY_TERMS: dict[tuple[bool, bool], str] = {
    (True, True): "lineperiodic",
    (True, False): "linear",
    (False, True): "periodic",
}

#: The one stage the events phase declares. A fixed name, because there is
#: exactly one stage in this shape and a vocabulary for it would be a second
#: thing to keep in step with the command.
EVENT_STAGE = "ev"

#: "Flag nothing", as the detection pipeline already spells it: S1 and S2 are
#: structural and always run, so naming only those turns despike, global,
#: window and protection all off. No new flag was needed for it.
USE_FLAGGED_STAGES = "S1,S2"

#: Symbolic spelling of each parameter family, keyed by the prefix
#: `param_names` uses. Built from the record's OWN names so the equation
#: cannot drift from the model that was actually fitted.
TERM_FORMS: tuple[tuple[str, str], ...] = (
    ("offset", "a₀"),
    ("rate", "a₁·(t−t₀)"),
    ("cos_annual", "c₁·cos(2πt)"),
    ("sin_annual", "s₁·sin(2πt)"),
    ("cos_semiannual", "c₂·cos(4πt)"),
    ("sin_semiannual", "s₂·sin(4πt)"),
)


def model_equation(param_names: Sequence[str]) -> str:
    """The general FORM of a record's model, read off its parameter names.

    Written down once it would rot; derived, it cannot. The numbers go to the
    panel and the terminal separately — an equation with twenty-one
    coefficients substituted into it is not readable as an equation.
    """
    names = list(param_names)
    parts = [form for key, form in TERM_FORMS if key in names]
    steps = sum(1 for n in names if n.startswith("step_amp"))
    parts += [f"h{k + 1}·H(t−t{k + 1})" for k in range(steps)]
    for n in names:
        if n.startswith("log_amp"):
            parts.append("A·ln(1+(t−tₑ)/τ)")
        elif n.startswith("exp_amp"):
            parts.append("A·(1−e^(−(t−tₑ)/τ))")
    return "x(t) = " + " + ".join(parts) if parts else "x(t) = (no terms)"


def _provisional_days_default() -> float:
    """geo_dataread's own default, imported so the two cannot drift."""
    try:
        from geo_dataread.gps_views import PROVISIONAL_DAYS_DEFAULT

        return float(PROVISIONAL_DAYS_DEFAULT)
    except Exception:  # pragma: no cover - older sibling
        return 14.0


def _require_qt() -> tuple[Any, Any]:
    """Import pyqtgraph, explaining itself if the dev group is absent."""
    try:
        import pyqtgraph as pg
        from pyqtgraph.Qt import QtWidgets
    except Exception as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "the Qt picker needs pyqtgraph and PySide6, which live in the dev "
            "group: run `uv sync` in gps_plot, or use the CLI "
            "(`gps-detrend-workbench`), which needs neither."
        ) from exc
    return pg, QtWidgets


# --- the two commands, as pure functions -----------------------------------
#
# ONE speller per phase, module-level and Qt-free, so what the window shows
# and what the clipboard carries are assembled in a single place. Every
# violation of this window's invariant has been a second such place.


def background_command(
    station: str,
    *,
    segments: Sequence[tuple[float, float]] = (),
    model: str | None = DEFAULT_MODEL,
    flags: Sequence[str] = (),
    save: bool = False,
) -> str:
    """The command that fits s(t) on the clean intervals, and maybe saves it.

    Every interval becomes its own ``--segment``. The union is what a
    background needs when an event sits inside the record, and it is the one
    thing a single dragged region could never express.
    """
    parts = ["gps-detrend-workbench", station]
    for lo, hi in segments:
        parts += ["--segment", f"{lo}:{hi}"]
    if model is not None and model != DEFAULT_MODEL:
        parts += ["--model", model]
    if save:
        parts.append("--save-secular")
    return shlex.join(parts + list(flags))


def events_command(
    station: str,
    *,
    free: Sequence[str],
    hold_from: str = "self",
    steps: Sequence[float] = (),
    segment: tuple[float, float] | None = None,
    flags: Sequence[str] = (),
    commit: bool = False,
) -> str:
    """The command that estimates events against a held background.

    ``hold_from`` is a ``store:`` value — ``self``, or another station's code
    (the legacy ``UseSTA``). ``store:`` is a DIFFERENT hold kind from
    ``donor:``, which reads a finished record; this one reads the background
    store, and the grammar names the kind precisely because the two resolve
    against different objects.
    """
    parts = ["gps-detrend-workbench", station]
    if segment is not None:
        parts += ["--segment", f"{segment[0]}:{segment[1]}"]
    for epoch in steps:
        parts += ["--step", str(epoch)]
    parts += ["--stage", f"{EVENT_STAGE}:{','.join(free)}"]
    for group in BACKGROUND_GROUPS:
        parts += ["--hold", f"{group}=store:{hold_from}"]
    if commit:
        parts.append("--commit")
    return shlex.join(parts + list(flags))


class PickerWindow:  # pragma: no cover - GUI
    """Plots on the left, the phase controls on the right."""

    def __init__(
        self,
        sta: str,
        yearf: Any,
        data: Any,
        sigma: Any,
        settings: Any,
        *,
        max_gap_years: float | None = None,
        uncert: int = WORKBENCH_UNCERT_DEFAULT,
        provisional_days: float | None = None,
        tot_dir: str | None = None,
    ) -> None:
        import numpy as np

        pg, QtWidgets = _require_qt()
        self.pg, self.QtWidgets, self.np = pg, QtWidgets, np

        self.sta = sta
        self.yearf = yearf
        self.data = data
        self.sigma = sigma
        self.base_settings = settings
        self.max_gap_years = max_gap_years
        self.uncert = int(uncert)
        self.provisional_days = provisional_days
        self.tot_dir = tot_dir

        finite = yearf[np.isfinite(yearf)]
        self.span = (float(finite.min()), float(finite.max()))
        lo, hi = getattr(settings, "window", (None, None)) or (None, None)
        self.default_domain = (
            self.span[0] if lo is None else max(float(lo), self.span[0]),
            self.span[1] if hi is None else min(float(hi), self.span[1]),
        )

        self.record: dict[str, Any] | None = None
        self.model: str | None = DEFAULT_MODEL
        self._est: Any = None
        self.segment_regions: list[list[Any]] = []
        self.step_lines: list[list[Any]] = []
        self._prov_counts = [0, 0, 0]
        self.command_text = ""
        # A session that failed to load must still be SAID, and `refit`
        # rewrites the summary the moment it runs. Held here and re-applied
        # after the first fit, or the warning is on screen for microseconds.
        self._session_note = ""

        self._build_plots()
        controls = self._build_controls()
        self._assemble(controls)
        self._draw_declared_events()
        self.load_session()
        self.refit()
        if self._session_note:
            self.summary.setPlainText(
                f"{self._session_note}\n\n{self.summary.toPlainText()}"
            )

    # -- construction ----------------------------------------------------
    def _build_plots(self) -> None:
        pg = self.pg
        pg.setConfigOptions(antialias=True, background="w", foreground="k")
        self.layout = pg.GraphicsLayoutWidget()
        self.plots: list[Any] = []
        self.kept_scatters: list[Any] = []
        self.flag_scatters: list[Any] = []
        self.outside_scatters: list[Any] = []
        self.prov_scatters: list[Any] = []
        self.fit_curves: list[Any] = []

        for row, name in enumerate(COMPONENTS):
            p = self.layout.addPlot(row=row, col=0)
            p.showGrid(x=True, y=True, alpha=0.25)
            p.setLabel("left", f"{name} [mm]")
            if row:
                p.setXLink(self.plots[0])
            self.plots.append(p)

            def scatter(face: Any, edge: Any = None, size: int = 4, _p: Any = p) -> Any:
                s = pg.ScatterPlotItem(
                    size=size,
                    pen=pg.mkPen(edge or face, width=1),
                    brush=pg.mkBrush(face),
                )
                _p.addItem(s)
                return s

            self.kept_scatters.append(scatter(KEPT_COLOR))
            self.flag_scatters.append(scatter(FLAG_COLOR, size=3))
            self.outside_scatters.append(scatter(OUTSIDE_COLOR, size=3))
            self.prov_scatters.append(scatter(PROV_FACE, PROV_EDGE, size=5))
            self.fit_curves.append(p.plot([], [], pen=pg.mkPen(FIT_COLOR, width=2)))

        self.plots[-1].setLabel("bottom", "fractional year")

        # The residual periodogram, borrowed from SARI / TSAnalyzer: a
        # seasonal the model missed shows as power at 1 or 2 cycles/yr, which
        # no amount of staring at the residual series reveals.
        self.spec = self.layout.addPlot(row=3, col=0)
        self.spec.setLabel("left", "residual power")
        self.spec.setLabel("bottom", "cycles per year")
        self.spec.setMaximumHeight(150)
        self.spec_curves = [
            self.spec.plot([], [], pen=pg.mkPen(c, width=1))
            for c in (KEPT_COLOR, (44, 160, 44), FIT_COLOR)
        ]
        for x in (1.0, 2.0):
            self.spec.addItem(
                pg.InfiniteLine(
                    x,
                    angle=90,
                    pen=pg.mkPen((160, 160, 160), style=pg.QtCore.Qt.DashLine),
                )
            )

        # The events domain (blue) and the clean intervals (orange) are
        # DIFFERENT objects: one says which epochs the events are fitted over,
        # the other which the background came from.
        self.domain_regions = self._add_region(self.default_domain, DOMAIN_COLOR)
        self.layout.scene().sigMouseClicked.connect(self._on_click)

    def _add_region(self, values: tuple[float, float], colour: Any) -> list[Any]:
        """One region per component panel, moving together.

        A fit domain that differed between components would be meaningless,
        and seeing the same interval on all three is most of the point.
        """
        pg = self.pg
        group: list[Any] = []
        for p in self.plots:
            r = pg.LinearRegionItem(values=values, brush=pg.mkBrush(colour))
            r.setZValue(-10)
            p.addItem(r)
            group.append(r)
        for r in group:
            r.sigRegionChanged.connect(lambda src, g=group: self._sync(src, g))
            r.sigRegionChangeFinished.connect(self.refit)
        return group

    def _add_line(self, pos: float, colour: Any) -> list[Any]:
        pg = self.pg
        group: list[Any] = []
        for p in self.plots:
            ln = pg.InfiniteLine(
                pos, angle=90, movable=True, pen=pg.mkPen(colour, width=2)
            )
            p.addItem(ln)
            group.append(ln)
        for ln in group:
            ln.sigPositionChanged.connect(lambda src, g=group: self._sync_line(src, g))
            ln.sigPositionChangeFinished.connect(self.refit)
        return group

    @staticmethod
    def _sync(src: Any, group: list[Any]) -> None:
        for r in group:
            if r is not src:
                r.blockSignals(True)
                r.setRegion(src.getRegion())
                r.blockSignals(False)

    @staticmethod
    def _sync_line(src: Any, group: list[Any]) -> None:
        for ln in group:
            if ln is not src:
                ln.blockSignals(True)
                ln.setValue(src.value())
                ln.blockSignals(False)

    @staticmethod
    def _set_visible(group: Sequence[Any], on: bool) -> None:
        for item in group:
            item.setVisible(on)

    def _draw_declared_events(self) -> None:
        """Equipment changes and declared earthquakes, from the catalogs.

        SARI's metadata fusion (Santamaria-Gomez 2019): a candidate
        discontinuity nobody has picked is still worth seeing, because the
        reason for a jump is usually in the station's own history.
        """
        pg = self.pg
        try:
            from gps_plot.detrend_workbench import declared_event_epochs

            seismic, other = declared_event_epochs(self.sta)
        except Exception:  # pragma: no cover - catalogs are optional
            return
        for epochs, colour in (
            (seismic, SEISMIC_EVENT_COLOR),
            (other, TOS_EVENT_COLOR),
        ):
            for item in epochs or ():
                # `declared_event_epochs` yields bare epochs for one catalog
                # and (epoch, description) pairs for the other; both are drawn.
                if isinstance(item, (tuple, list)):
                    epoch = float(item[0])
                    label = str(item[1]) if len(item) > 1 else ""
                else:
                    epoch, label = float(item), ""
                for index, p in enumerate(self.plots):
                    p.addItem(
                        pg.InfiniteLine(
                            epoch,
                            angle=90,
                            pen=pg.mkPen(colour, width=1, style=pg.QtCore.Qt.DotLine),
                            label=label if index == 0 else None,
                            labelOpts={"position": 0.95, "color": colour},
                        )
                    )

    def _build_controls(self) -> Any:
        """The right-hand column, grouped by WHAT A CONTROL DECIDES.

        `phase` chooses which half of the model is being worked on; the two
        phase boxes change the fit and therefore the record; `view` changes
        nothing at all; `run` holds the parameters that were once CLI-only.
        That split is the one an operator needs to tell curation from
        navigation, and it is the same one the workbench draws between a
        stored decision and a look-only one.
        """
        QtWidgets = self.QtWidgets
        box = QtWidgets.QWidget()
        col = QtWidgets.QVBoxLayout(box)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(8)

        # --- phase --------------------------------------------------------
        phase_box = QtWidgets.QGroupBox("phase")
        pcol = QtWidgets.QVBoxLayout(phase_box)
        self.mode = QtWidgets.QComboBox()
        self.mode.addItems([MODE_BACKGROUND, MODE_EVENTS])
        self.mode.setToolTip(
            "s(t) first, then the events against it. The order is not a "
            "preference: a background held from a window on ONE side of an "
            "event pins the level to that side, and the step is then left "
            "with nothing to measure"
        )
        self.mode.currentIndexChanged.connect(self._mode_changed)
        pcol.addWidget(self.mode)
        self.phase_hint = QtWidgets.QLabel()
        self.phase_hint.setWordWrap(True)
        self.phase_hint.setStyleSheet("color: #555;")
        pcol.addWidget(self.phase_hint)
        col.addWidget(phase_box)

        # --- background ----------------------------------------------------
        self.bg_box = QtWidgets.QGroupBox("background s(t)")
        bcol = QtWidgets.QVBoxLayout(self.bg_box)

        trow = QtWidgets.QHBoxLayout()
        trow.addWidget(QtWidgets.QLabel("terms:"))
        self.cb_linear = QtWidgets.QCheckBox("linear")
        self.cb_linear.setChecked(True)
        self.cb_linear.setToolTip(
            "Rate and offset (emitted inside --model as 'secular'). The "
            "OFFSET matters here even though detrending does not need one: "
            "holding s(t) to measure a step needs the level anchored"
        )
        self.cb_periodic = QtWidgets.QCheckBox("periodic")
        self.cb_periodic.setChecked(True)
        self.cb_periodic.setToolTip("Annual and semiannual (emitted inside --model)")
        for cb in (self.cb_linear, self.cb_periodic):
            cb.toggled.connect(self.refit)
            trow.addWidget(cb)
        trow.addStretch(1)
        bcol.addLayout(trow)

        self.seg_list = QtWidgets.QListWidget()
        self.seg_list.setMaximumHeight(90)
        self.seg_list.setToolTip(
            "The clean intervals s(t) is fitted on — a UNION, which is the "
            "whole point: one interval cannot span an event, and a background "
            "fitted only after one extrapolates backwards through it"
        )
        bcol.addWidget(self.seg_list)

        srow = QtWidgets.QHBoxLayout()
        add = QtWidgets.QPushButton("+ interval")
        add.setToolTip("Add a clean interval; drag its orange region on the plot")
        add.clicked.connect(lambda: self.add_segment())
        srow.addWidget(add)
        rem = QtWidgets.QPushButton("− interval")
        rem.setToolTip("Remove the last interval")
        rem.clicked.connect(self.remove_segment)
        srow.addWidget(rem)
        bcol.addLayout(srow)

        self.btn_save = QtWidgets.QPushButton("save s(t)")
        self.btn_save.setToolTip(
            "Write this background to analysis.yaml as the station's reusable "
            "s(t) (--save-secular). Separate from committing, which stores "
            "the finished f(t) that plot-gps-timeseries reads"
        )
        self.btn_save.clicked.connect(self.save_secular)
        bcol.addWidget(self.btn_save)
        self.btn_bg_commit = QtWidgets.QPushButton("copy the commit command")
        self.btn_bg_commit.setToolTip(
            "For a station whose model IS the background — no events to "
            "estimate — this is the whole f(t). Emits --save-secular --commit, "
            "which writes s(t) to the store AND the finished record to "
            "detrend_params.json, where plot-gps-timeseries reads it"
        )
        self.btn_bg_commit.clicked.connect(self.copy_commit)
        bcol.addWidget(self.btn_bg_commit)
        self.saved_label = QtWidgets.QLabel()
        self.saved_label.setWordWrap(True)
        self.saved_label.setStyleSheet("color: #555;")
        bcol.addWidget(self.saved_label)
        col.addWidget(self.bg_box)

        # --- events ---------------------------------------------------------
        self.ev_box = QtWidgets.QGroupBox("events")
        ecol = QtWidgets.QVBoxLayout(self.ev_box)

        hrow = QtWidgets.QHBoxLayout()
        hrow.addWidget(QtWidgets.QLabel("hold s(t) from:"))
        self.hold_from = QtWidgets.QLineEdit("self")
        self.hold_from.setToolTip(
            "'self' holds this station's own saved background; a station code "
            "borrows that station's (the legacy UseSTA). Emitted as "
            "--hold secular=store:… --hold periodic=store:…"
        )
        self.hold_from.editingFinished.connect(self.refit)
        hrow.addWidget(self.hold_from)
        ecol.addLayout(hrow)

        self.btn_commit = QtWidgets.QPushButton("copy the commit command")
        self.btn_commit.setToolTip(
            "Put the command WITH --commit on the clipboard. The picker never "
            "writes the finished record itself: the workbench does, so there "
            "is one path to stored science and it is one you can read first"
        )
        self.btn_commit.clicked.connect(self.copy_commit)
        ecol.addWidget(self.btn_commit)

        clear = QtWidgets.QPushButton("clear picked steps")
        clear.setToolTip(
            "Remove the PICKED steps. Steps declared in steps.csv are a floor "
            "and stay in the fit"
        )
        clear.clicked.connect(self.clear_steps)
        ecol.addWidget(clear)
        hint = QtWidgets.QLabel(
            "double-click a jump to declare a step · right-click it to remove"
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #666;")
        ecol.addWidget(hint)
        col.addWidget(self.ev_box)

        # --- view: moves nothing that is fitted or stored --------------------
        view_box = QtWidgets.QGroupBox("view — display only")
        vcol = QtWidgets.QVBoxLayout(view_box)
        self.view = QtWidgets.QComboBox()
        self.view.addItems(["data", "data − s(t)", "data − f(t)"])
        self.view.setToolTip(
            "'data − s(t)' is where events are read: the background is gone "
            "and what remains is the departures from it. 'data − f(t)' goes "
            "to zero when the whole model is right"
        )
        self.view.currentIndexChanged.connect(self.refit)
        vcol.addWidget(self.view)
        self.cb_draw_flagged = QtWidgets.QCheckBox("draw flagged epochs (grey)")
        self.cb_draw_flagged.setChecked(True)
        self.cb_draw_flagged.setToolTip(
            "Show or hide the screened epochs, exactly like --hide-outliers: "
            "same masks, same counts, same record. To put them back in the "
            "FIT, use the checkbox in the run box"
        )
        self.cb_draw_flagged.toggled.connect(self.refit)
        vcol.addWidget(self.cb_draw_flagged)
        col.addWidget(view_box)

        # --- run: each writes the SAME attribute run_flags emits -------------
        run_box = QtWidgets.QGroupBox("run")
        rgrid = QtWidgets.QFormLayout(run_box)
        self.cb_use_flagged = QtWidgets.QCheckBox("fit the flagged epochs too")
        self.cb_use_flagged.setToolTip(
            "Put the screened epochs back into the estimation (emits "
            "--stages S1,S2, which is 'flag nothing'). This CHANGES the fit "
            "and the record — unlike drawing them, which is in the view box"
        )
        self.cb_use_flagged.toggled.connect(self.refit)
        rgrid.addRow(self.cb_use_flagged)

        self.sp_gap = QtWidgets.QDoubleSpinBox()
        self.sp_gap.setRange(0.1, 50.0)
        self.sp_gap.setSingleStep(0.5)
        self.sp_gap.setDecimals(2)
        # Show the EFFECTIVE gate, not the flag: with no --max-gap-years the
        # resolved settings carry the catalog's value, and a spinbox reading
        # 0.5 while the fit ran at 1.0 is a third place disagreeing about it.
        self.sp_gap.setValue(
            float(
                self.max_gap_years
                if self.max_gap_years is not None
                else getattr(self.base_settings, "max_gap_years", 0.5)
            )
        )
        self.sp_gap.setToolTip(
            "Per-segment gap gate [yr]. The 0.5 spec default rejects every "
            "station in the working set, so this is effectively required"
        )
        self.sp_gap.editingFinished.connect(self._set_max_gap)
        rgrid.addRow("max-gap [yr]", self.sp_gap)

        self.sp_prov = QtWidgets.QDoubleSpinBox()
        self.sp_prov.setRange(0.0, 400.0)
        self.sp_prov.setSingleStep(7.0)
        self.sp_prov.setDecimals(0)
        self.sp_prov.setValue(
            float(
                _provisional_days_default()
                if self.provisional_days is None
                else self.provisional_days
            )
        )
        self.sp_prov.setToolTip(
            "Recency bound of the GOLD provisional lane [days]; 0 disables. "
            "Display only — it moves no fitted quantity"
        )
        self.sp_prov.editingFinished.connect(self._set_provisional_days)
        rgrid.addRow("provisional [d]", self.sp_prov)

        self.sp_uncert = QtWidgets.QSpinBox()
        self.sp_uncert.setRange(1, 200)
        self.sp_uncert.setValue(self.uncert)
        self.sp_uncert.setToolTip(
            "Formal-sigma screen [mm], applied at READ time. Unlike the "
            "others this RE-READS the series from disk — it changes which "
            "epochs exist, not how they are fitted"
        )
        self.sp_uncert.editingFinished.connect(self._set_uncert)
        rgrid.addRow("uncert [mm]", self.sp_uncert)
        col.addWidget(run_box)

        save = QtWidgets.QPushButton("save session")
        save.setToolTip(
            "Convenience only — the emitted command is what reproduces the "
            "science, because it is re-parsed by the grammar the CLI uses"
        )
        save.clicked.connect(self.save_session)
        col.addWidget(save)

        self.summary = QtWidgets.QPlainTextEdit()
        self.summary.setReadOnly(True)
        self.summary.setMinimumHeight(180)
        self.summary.setStyleSheet("font-family: monospace;")
        col.addWidget(self.summary)
        col.addStretch(1)
        return box

    def _assemble(self, controls: Any) -> None:
        QtWidgets = self.QtWidgets
        self.win = QtWidgets.QWidget()
        self.win.setWindowTitle(f"detrend picker — {self.sta}")
        outer = QtWidgets.QVBoxLayout(self.win)

        self.header = QtWidgets.QLabel()
        self.header.setTextFormat(self.pg.QtCore.Qt.RichText)
        outer.addWidget(self.header)

        # Plots left, controls right, draggable divider. Stacking everything
        # vertically spent the scarcest dimension -- height -- on a one-line
        # control strip, while three component panels plus the periodogram
        # need every bit of it.
        split = QtWidgets.QSplitter(self.pg.QtCore.Qt.Horizontal)
        split.addWidget(self.layout)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(controls)
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(340)
        split.addWidget(scroll)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 1)
        outer.addWidget(split, 1)

        # The command runs FULL WIDTH along the bottom: it is the window's
        # output, runs past 200 characters on a staged fit with a term and a
        # segment, and exists to be read and copied -- which a third of the
        # window cannot do.
        self.command = QtWidgets.QLineEdit()
        self.command.setReadOnly(True)
        self.command.setStyleSheet("font-family: monospace;")
        outer.addWidget(self.command)
        self.win.resize(1500, 950)

    # -- segments ---------------------------------------------------------
    def segments(self) -> list[tuple[float, float]]:
        """The clean intervals, in time order, rounded as they are emitted."""
        out = [
            (round(float(g[0].getRegion()[0]), 4), round(float(g[0].getRegion()[1]), 4))
            for g in self.segment_regions
        ]
        return sorted(out)

    def add_segment(self, values: tuple[float, float] | None = None) -> None:
        if values is None:
            lo, hi = self.span
            width = (hi - lo) / 4.0
            start = lo + width * len(self.segment_regions)
            values = (round(start, 4), round(min(start + width, hi), 4))
        self.segment_regions.append(self._add_region(values, SEGMENT_COLOR))
        self.refit()

    def remove_segment(self) -> None:
        if not self.segment_regions:
            return
        for p, r in zip(self.plots, self.segment_regions.pop(), strict=True):
            p.removeItem(r)
        self.refit()

    def _clear_segments(self) -> None:
        while self.segment_regions:
            for p, r in zip(self.plots, self.segment_regions.pop(), strict=True):
                p.removeItem(r)

    def _update_segment_list(self) -> None:
        self.seg_list.clear()
        for lo, hi in self.segments():
            self.seg_list.addItem(f"{lo} : {hi}      ({hi - lo:.2f} yr)")

    # -- steps -------------------------------------------------------------
    def _on_click(self, ev: Any) -> None:
        """Double-click places a step; right-click on one removes it."""
        if self.mode.currentText() != MODE_EVENTS:
            return
        pos = ev.scenePos()
        for p in self.plots:
            if not p.sceneBoundingRect().contains(pos):
                continue
            x = float(p.vb.mapSceneToView(pos).x())
            if ev.double():
                self._add_step(round(x, 4))
                ev.accept()
            elif ev.button() == self.pg.QtCore.Qt.MouseButton.RightButton:
                self._remove_step_near(x)
                ev.accept()
            return

    def _add_step(self, epoch: float) -> None:
        self.step_lines.append(self._add_line(epoch, STEP_COLOR))
        self.refit()

    def _remove_step_near(self, x: float) -> None:
        if not self.step_lines:
            return
        idx = min(
            range(len(self.step_lines)),
            key=lambda i: abs(self.step_lines[i][0].value() - x),
        )
        if abs(self.step_lines[idx][0].value() - x) > 0.25:
            return
        for p, ln in zip(self.plots, self.step_lines.pop(idx), strict=True):
            p.removeItem(ln)
        self.refit()

    def clear_steps(self) -> None:
        while self.step_lines:
            for p, ln in zip(self.plots, self.step_lines.pop(), strict=True):
                p.removeItem(ln)
        self.refit()

    def _picked_steps(self) -> tuple[float, ...]:
        return tuple(round(float(g[0].value()), 4) for g in self.step_lines)

    # -- run parameters -----------------------------------------------------
    def _set_max_gap(self) -> None:
        self.max_gap_years = float(self.sp_gap.value())
        self.refit()

    def _set_provisional_days(self) -> None:
        self.provisional_days = float(self.sp_prov.value())
        self.refit()

    def _set_uncert(self) -> None:
        """RE-READS the series: this screen decides which epochs exist."""
        import geo_dataread.gps_read as gpsr

        want = int(self.sp_uncert.value())
        try:
            yearf, data, sigma, _ = gpsr.getData(
                self.sta, ref="plate", Dir=self.tot_dir, tType="TOT", uncert=want
            )
        except Exception as exc:
            self.summary.setPlainText(f"re-read at uncert {want} failed: {exc}")
            self.sp_uncert.setValue(self.uncert)
            return
        if yearf is None or len(yearf) == 0:
            self.summary.setPlainText(
                f"uncert {want} mm screens every epoch away — keeping {self.uncert}"
            )
            self.sp_uncert.setValue(self.uncert)
            return
        np = self.np
        before = self.yearf.size
        self.yearf = np.asarray(yearf, float)
        self.data = np.atleast_2d(np.asarray(data, float))
        self.sigma = np.atleast_2d(np.asarray(sigma, float))
        self.uncert = want
        self.summary.setPlainText(
            f"re-read at uncert {want} mm: {before} → {self.yearf.size} epochs"
        )
        self.refit()

    def _mode_changed(self, *_: Any) -> None:
        events = self.mode.currentText() == MODE_EVENTS
        self.bg_box.setVisible(not events)
        self.ev_box.setVisible(events)
        self._set_visible(self.domain_regions, events)
        for group in self.segment_regions:
            self._set_visible(group, not events)
        for group in self.step_lines:
            self._set_visible(group, events)
        self.phase_hint.setText(
            "Estimate the departures against a background that no longer "
            "moves. data − s(t) is where they are read."
            if events
            else "Pick the CLEAN intervals — usually one either side of an "
            "event — then save s(t)."
        )
        # Entering the events phase, the useful view is the one events are
        # read in; leaving it, the useful view is the data itself.
        self.view.blockSignals(True)
        self.view.setCurrentIndex(1 if events else 0)
        self.view.blockSignals(False)
        self.refit()

    # -- the fit -------------------------------------------------------------
    def _terms_model(self) -> str | None:
        return MODEL_BY_TERMS.get(
            (self.cb_linear.isChecked(), self.cb_periodic.isChecked())
        )

    def _run_flags(self) -> list[str]:
        """The shared tail. TWO pickers need it; ONE function builds it.

        Two pickers assembling it independently is two places to forget the
        same flag, and each of them forgot `--tot-dir`.
        """
        flags = list(
            run_flags(
                tot_dir=self.tot_dir,
                max_gap_years=self.max_gap_years,
                uncert=self.uncert,
                provisional_days=self.provisional_days,
            )
        )
        if self.cb_use_flagged.isChecked():
            flags = ["--stages", USE_FLAGGED_STAGES, *flags]
        return flags

    def _events_free(self, settings: Any) -> list[str]:
        """Which groups the event stage estimates.

        `step` only when the FIT will carry one — the MERGED declaration
        (steps.csv floor ∪ picked), not the picked lines, because a declared
        step has parameters whether or not anyone clicked it. Naming a group
        the model has no parameters for is refused by the estimator, rightly,
        so asking for it unconditionally would make every stepless station
        unfittable.
        """
        from gps_plot.detrend_workbench import _declared_step_epochs

        return ["step"] if _declared_step_epochs(self.sta, settings.steps) else []

    def refit(self, *_: Any) -> None:
        from gps_plot.detrend_workbench import _override_settings

        events = self.mode.currentText() == MODE_EVENTS
        self._update_segment_list()
        flags = self._run_flags()
        note = ""
        plan = lookup = None

        if events:
            steps = self._picked_steps()
            lo, hi = (round(float(v), 4) for v in self.domain_regions[0].getRegion())
            moved = (lo, hi) != tuple(round(v, 4) for v in self.default_domain)
            settings = _override_settings(
                self.base_settings,
                self.sta,
                quiet=True,
                segments=((lo, hi),) if moved else None,
                steps=steps or None,
                max_gap_years=self.max_gap_years,
            )
            free = self._events_free(settings)
            hold_from = self.hold_from.text().strip() or "self"
            self.command_text = events_command(
                self.sta,
                free=free or ["step"],
                hold_from=hold_from,
                steps=steps,
                segment=(lo, hi) if moved else None,
                flags=flags,
            )
            self.model = None
            if not free:
                note = (
                    "nothing to estimate: this station has no declared step. "
                    "Double-click a jump to "
                    "declare a step by double-clicking a jump on the plot."
                )
            else:
                plan, lookup, note = self._events_plan(free, hold_from)
        else:
            self.model = self._terms_model()
            segs = self.segments()
            settings = _override_settings(
                self.base_settings,
                self.sta,
                quiet=True,
                segments=tuple(segs) or None,
                max_gap_years=self.max_gap_years,
            )
            self.command_text = background_command(
                self.sta, segments=segs, model=self.model, flags=flags
            )
            if self.model is None:
                note = (
                    "fit refused: linear and periodic are both off, which no "
                    "--model value can express. Leave at least one of them in."
                )

        est = None
        if not note:
            try:
                # The SAME fallback the workbench applies: a full-detection
                # abort is recoverable (retry S0-only), a failed gate is not.
                est, fell_back = estimate_with_abort_fallback(
                    self.sta,
                    self.yearf,
                    self.data,
                    self.sigma,
                    settings=settings,
                    stage_plan=plan,
                    lookup_secular=lookup,
                    model=self.model,
                    stages=(
                        USE_FLAGGED_STAGES if self.cb_use_flagged.isChecked() else None
                    ),
                )
            except (ValueError, RuntimeError) as exc:
                # A refused fit is a RESULT -- a rank-deficient stage, an
                # unfittable term, a domain with too few epochs. Say so, and
                # keep the previous trajectory on screen but greyed.
                note = f"fit refused: {exc}"
            else:
                if fell_back:
                    from gps_plot.detrend_workbench import abort_fallback_note

                    note = f"note: {abort_fallback_note(self.sta)}"

        self._render(est, note)
        self.command.setText(self.command_text)
        self._update_header(events)

    def _events_plan(self, free: Sequence[str], hold_from: str) -> tuple[Any, Any, str]:
        """The one-stage plan and the store lookup, or an explanation."""
        from geo_dataread.stage_plan import build_stage_plan, default_analysis_yaml_path
        from gps_plot.detrend_workbench import _secular_lookup

        class _Args:
            analysis_yaml = None

        try:
            plan = build_stage_plan(
                [f"{EVENT_STAGE}:{','.join(free)}"],
                [f"{g}=store:{hold_from}" for g in BACKGROUND_GROUPS],
            )
        except ValueError as exc:
            return None, None, f"stage plan refused: {exc}"
        try:
            lookup = _secular_lookup(_Args(), self.sta)
            # Resolved EAGERLY so a missing background is reported here, with
            # the action that fixes it, rather than surfacing later as an
            # opaque fit failure.
            lookup(None if hold_from == "self" else hold_from)
        except (RuntimeError, ValueError) as exc:
            return (
                None,
                None,
                f"{exc}\n\nSwitch to the background phase, pick the clean "
                f"intervals and press 'save s(t)'.\n"
                f"Store: {default_analysis_yaml_path()}",
            )
        return plan, lookup, ""

    # -- rendering -----------------------------------------------------------
    def _render(self, est: Any, note: str) -> None:
        np = self.np
        if est is None:
            # Grey and DASHED: a solid blue line reads as a result whatever
            # the summary says, and this curve belongs to a configuration that
            # is no longer on screen.
            for curve in self.fit_curves:
                curve.setPen(
                    self.pg.mkPen(
                        STALE_COLOR, width=1, style=self.pg.QtCore.Qt.DashLine
                    )
                )
            for group in (
                self.flag_scatters,
                self.outside_scatters,
                self.prov_scatters,
            ):
                for sc in group:
                    sc.setData([], [])
            for c in range(len(COMPONENTS)):
                finite = np.isfinite(self.data[c])
                self.kept_scatters[c].setData(self.yearf[finite], self.data[c][finite])
            self.record = None
            self.summary.setPlainText(
                note
                or "no record: the outlier stage aborted, or a validity gate "
                "rejected this domain (span / epochs / max-gap)"
            )
            return

        import gps_analysis

        self.record = est.record
        self._est = est
        fit = np.asarray(gps_analysis.evaluate_record(est.record, self.yearf))
        outl = np.atleast_2d(np.asarray(est.outliers, dtype=bool))
        empty = np.zeros(self.yearf.shape, dtype=bool)

        # The fit passes NO verdict outside its window, so drawn plain those
        # epochs claim "clean" and one blunder owns the y-axis. Fill that
        # silence with the view detector -- the same chain
        # `plot-gps-timeseries --view cleaned` uses, and the only thing that
        # HAS a provisional category.
        try:
            from gps_plot.detrend_workbench import screen_outside_window

            outside, prov = screen_outside_window(
                self.sta,
                self.yearf,
                self.data,
                self.sigma,
                est,
                steps=self._picked_steps() or None,
                provisional_days=self.provisional_days,
            )
        except Exception:
            outside = prov = None
        outside = None if outside is None else np.atleast_2d(np.asarray(outside, bool))
        prov = None if prov is None else np.atleast_2d(np.asarray(prov, bool))

        view = self.view.currentText()
        peel: list[str] = []
        if view == "data − s(t)":
            peel = list(BACKGROUND_GROUPS)
        elif view == "data − f(t)":
            peel = list(PEEL_ALL)
        # DISPLAY ONLY: the masks, the record and the emitted command are the
        # same either way -- this subtracts a model that was already fitted,
        # it does not fit anything different.
        shown = (
            self.data
            if not peel
            else self.data - group_contribution(est.record, self.yearf, peel)
        )

        # The curve is drawn on its own dense grid: joining the data epochs
        # draws a straight chord over every gap, which claims linear motion
        # the model never fitted. Steps are bracketed there so a jump stays
        # vertical at any zoom.
        fit_x, fit_y = trajectory_curve(est.record, self.yearf)
        if peel:
            remaining = [g for g in list(PEEL_ALL) if g not in peel]
            fit_y = (
                group_contribution(est.record, fit_x, remaining)
                if remaining
                else np.zeros((self.data.shape[0], fit_x.size))
            )
        for curve in self.fit_curves:
            curve.setPen(self.pg.mkPen(FIT_COLOR, width=2))

        suffix = "" if not peel else (" − s(t)" if len(peel) == 2 else " − f(t)")
        draw = self.cb_draw_flagged.isChecked()
        blank = self.yearf[:0]
        for c, name in enumerate(COMPONENTS):
            self.plots[c].setLabel("left", f"{name}{suffix} [mm]")
            # connect="finite" so a component the model cannot evaluate breaks
            # the line instead of being joined across.
            self.fit_curves[c].setData(fit_x, fit_y[c], connect="finite")
            finite = np.isfinite(self.data[c])
            flagged = outl[c] & finite
            out_c = outside[c] & finite if outside is not None else empty
            prov_c = prov[c] & finite if prov is not None else empty
            # Both greys MASK; gold does not, so gold stays in the kept series
            # and is only overlaid.
            kept = finite & ~outl[c] & ~out_c
            self.kept_scatters[c].setData(self.yearf[kept], shown[c][kept])
            self.flag_scatters[c].setData(
                self.yearf[flagged] if draw else blank,
                shown[c][flagged] if draw else blank,
            )
            self.outside_scatters[c].setData(
                self.yearf[out_c] if draw else blank,
                shown[c][out_c] if draw else blank,
            )
            self.prov_scatters[c].setData(self.yearf[prov_c], shown[c][prov_c])
            self._prov_counts[c] = int(prov_c.sum())

        self._update_spectrum(fit)
        text = model_equation(est.record.get("param_names") or []) + "\n\n"
        if note:
            text = f"{note}\n\n{text}"
        text += self._summary(est.record)
        block = self.params_block(est.record)
        if block:
            text += "\n\nparameters\n" + block
        self.print_params(est.record, "current fit")
        self.summary.setPlainText(text)

    def _update_spectrum(self, fit: Any) -> None:
        """Lomb-Scargle of the residuals, per component."""
        np = self.np
        try:
            from scipy.signal import lombscargle
        except Exception:  # pragma: no cover
            return
        freqs = np.linspace(0.2, 6.0, 600)
        omega = 2.0 * np.pi * freqs
        for c in range(len(COMPONENTS)):
            resid = self.data[c] - fit[c]
            ok = np.isfinite(resid) & np.isfinite(self.yearf)
            if int(ok.sum()) < 32:
                self.spec_curves[c].setData([], [])
                continue
            y = resid[ok] - resid[ok].mean()
            try:
                power = lombscargle(self.yearf[ok], y, omega, normalize=True)
            except Exception:  # pragma: no cover
                self.spec_curves[c].setData([], [])
                continue
            self.spec_curves[c].setData(freqs, power)

    def _summary(self, rec: dict[str, Any]) -> str:
        # `rms` is rounded for DISPLAY: raw it printed as
        # [1.8299068022476694, ...] and wrapped over three lines in the
        # control column, burying the number being compared between iterations.
        keys = ("model", "window", "n_epochs", "n_rejected", "rms", "step_epochs")
        lines = []
        for k in keys:
            value = rec.get(k)
            if k == "rms" and value is not None:
                value = [round(float(v), 2) for v in value]
            lines.append(f"{k:14s} {value}")
        rate = [round(float(c["params"][1]), 2) for c in rec["components"]]
        lines.append(f"{'rate [mm/yr]':14s} {rate}")
        return "\n".join(lines)

    def params_block(self, record: dict[str, Any]) -> str:
        """The parameter table as text, for the panel.

        The picker is normally launched from a sway keybinding, which `exec`s
        it with no terminal attached -- so printing to stdout alone put the
        numbers nowhere an operator could see them.
        """
        names = record.get("param_names") or []
        comps = record.get("components") or []
        if not names or not comps:
            return ""
        rows = [f"{'':18s}" + "".join(f"{n:>12s}" for n in COMPONENTS)]
        for j, name in enumerate(names):
            cells = "".join(
                f"{float(c['params'][j]):12.3f}"
                if j < len(c["params"])
                else f"{'':12s}"
                for c in comps
            )
            rows.append(f"{name:18s}{cells}")
        return "\n".join(rows)

    def print_params(self, record: dict[str, Any], tag: str) -> None:
        block = self.params_block(record)
        if not block:
            return
        print(f"\n{self.sta} — {tag}")
        print("  " + model_equation(record.get("param_names") or []))
        for line in block.splitlines():
            print("  " + line)

    def _update_header(self, events: bool) -> None:
        """Station, phase and the run parameters, always on screen.

        `uncert` screens sigma at READ time, so it changes WHICH epochs are
        fitted while leaving no trace in any fitted quantity, and
        `max_gap_years` decides whether the station is estimable at all. Two
        records that differ only in these are indistinguishable from their
        numbers.
        """
        bits = [
            f"<b>{self.sta}</b>",
            f"phase <b>{'events' if events else 'background'}</b>",
            f"uncert {self.uncert} mm",
            f"max-gap {self.sp_gap.value():.1f} yr",
        ]
        if events:
            bits.append(f"holding s(t) from <b>{self.hold_from.text().strip()}</b>")
            bits.append(f"steps picked {len(self.step_lines)}")
        else:
            segs = self.segments()
            if segs:
                span = sum(hi - lo for lo, hi in segs)
                bits.append(f"{len(segs)} interval(s), {span:.1f} yr")
            else:
                bits.append(
                    "<span style='color:#a00'>no interval — using the catalog "
                    "domain</span>"
                )
        if self.record is None:
            bits.append("<span style='color:#a00'>NO RECORD</span>")
        else:
            bits.append(f"fitting {self.record.get('n_epochs')} epochs")
            if any(self._prov_counts):
                bits.append(
                    f"<span style='color:#8a6d0b'>provisional "
                    f"{self._prov_counts}</span>"
                )
        self.header.setText(" · ".join(bits))

    # -- actions --------------------------------------------------------------
    def save_secular(self) -> None:
        """Write s(t) to the store, through the store's OWN writer.

        The picker does not decide what a background is: `secular_from_record`
        does (linear + periodic, never events), and two answers to that
        question is exactly the divergence this window keeps producing.
        """
        if self.record is None:
            self.summary.setPlainText(
                "nothing to save: there is no fit on screen. Adjust the "
                "intervals until a record appears."
            )
            return
        from geo_dataread.secular_store import secular_from_record, write_secular
        from geo_dataread.stage_plan import default_analysis_yaml_path

        path = default_analysis_yaml_path()
        if path is None:
            self.summary.setPlainText(
                "no analysis.yaml is reachable on this host, so there is "
                "nowhere to save the background. Run the emitted command with "
                "--save-secular --analysis-yaml <path> instead."
            )
            return
        try:
            entry = secular_from_record(
                self.record, fitted_at=self.record.get("fitted_at")
            )
            write_secular(path, self.sta, entry)
        except (ValueError, OSError) as exc:
            self.summary.setPlainText(f"save refused: {exc}")
            return
        spans = ", ".join(f"{a}:{b}" for a, b in (entry.segments or ())) or "the domain"
        self.saved_label.setText(
            f"saved {len(entry.param_names)} parameters per component, "
            f"fitted on {spans} → {path}"
        )
        cmd = background_command(
            self.sta,
            segments=self.segments(),
            model=self.model,
            flags=self._run_flags(),
            save=True,
        )
        self.summary.setPlainText(
            f"background saved for {self.sta}.\n\nSwitch to the events phase "
            f"to estimate the offsets against it.\n\nThe same thing "
            f"from the CLI:\n\n{cmd}"
        )

    def copy_commit(self) -> None:
        """Put the committing command on the clipboard — never run it here.

        The workbench stores; the picker proposes. One path to stored science,
        and it is the one an operator can read before it runs.
        """
        if self.mode.currentText() != MODE_EVENTS:
            # A station with no events HAS no events phase to commit from --
            # it says "nothing to estimate" and produces no record, which
            # left such a station with no route to detrend_params.json at
            # all. Its background IS its whole model, so it commits here.
            #
            # The two writes stay distinct even in one command:
            # --save-secular writes s(t) to the store as a reusable
            # COMPONENT, --commit writes the finished record production
            # reads. Declared steps are already in this record (steps.csv is
            # a floor), so for a station whose only events are declared, the
            # background phase produces a complete f(t).
            if self.record is None:
                self.summary.setPlainText(
                    "nothing to commit: there is no fit on screen."
                )
                return
            cmd = background_command(
                self.sta,
                segments=self.segments(),
                model=self.model,
                flags=self._run_flags(),
                save=True,
            )
            cmd += " --commit"
            self.QtWidgets.QApplication.clipboard().setText(cmd)
            declared = self.record.get("step_epochs") or []
            note = (
                f"\n\nThis record carries {len(declared)} declared step(s) "
                f"{[round(float(e), 4) for e in declared]}, so it is already a "
                f"complete f(t)."
                if declared
                else "\n\nThis station has no declared step, so s(t) IS f(t)."
            )
            self.summary.setPlainText(
                f"copied to the clipboard:\n\n{cmd}\n\n"
                f"--save-secular writes the reusable s(t); --commit writes the "
                f"finished record plot-gps-timeseries reads.{note}"
            )
            return
        from gps_plot.detrend_workbench import _override_settings

        steps = self._picked_steps()
        settings = _override_settings(
            self.base_settings,
            self.sta,
            quiet=True,
            steps=steps or None,
            max_gap_years=self.max_gap_years,
        )
        cmd = events_command(
            self.sta,
            free=self._events_free(settings) or ["step"],
            hold_from=self.hold_from.text().strip() or "self",
            steps=steps,
            flags=self._run_flags(),
            commit=True,
        )
        self.QtWidgets.QApplication.clipboard().setText(cmd)
        self.summary.setPlainText(f"copied to the clipboard:\n\n{cmd}")

    # -- session ---------------------------------------------------------------
    def _session_path(self) -> Any:
        from pathlib import Path

        base = (
            Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local/state"))
            / "gps-detrend-picker"
        )
        base.mkdir(parents=True, exist_ok=True)
        return base / f"{self.sta}.json"

    def save_session(self) -> None:
        """Write the picks so they survive closing the window.

        Borrowed from TSAnalyzer (Wu et al. 2017, doi:10.1007/s10291-017-0637-2).
        A CONVENIENCE, not a record: the emitted command is what reproduces
        the science, because it is re-parsed by the same grammar the CLI uses.
        A session that could be mistaken for provenance would be a second path
        to stored results, which this tool does not have.
        """
        import json

        payload = {
            "station": self.sta,
            "note": "picker convenience only — the emitted command is the record",
            "mode": self.mode.currentText(),
            "terms": {
                "linear": self.cb_linear.isChecked(),
                "periodic": self.cb_periodic.isChecked(),
            },
            "segments": [list(s) for s in self.segments()],
            "domain": [round(float(v), 4) for v in self.domain_regions[0].getRegion()],
            "steps": list(self._picked_steps()),
            "hold_from": self.hold_from.text().strip(),
            "params": {
                "uncert": self.uncert,
                "max_gap_years": self.max_gap_years,
                "provisional_days": self.provisional_days,
            },
        }
        self._session_path().write_text(json.dumps(payload, indent=2))
        self.summary.setPlainText(f"session saved → {self._session_path()}")

    def load_session(self) -> bool:
        """Restore picks. Nothing here may raise: this runs at LAUNCH.

        A session file that is unreadable OR structurally wrong used to take
        the application down before the window appeared, leaving no way in to
        clear the very file that was killing it. A bad session now degrades to
        the declared defaults and says where the file is, because the file is
        somebody's curation and deleting it unasked is the worse failure.
        """
        import json

        path = self._session_path()
        if not path.exists():
            self._mode_changed()
            return False
        try:
            d = json.loads(path.read_text())
            if not isinstance(d, dict):
                raise ValueError("top level must be an object")
            segments = [(float(a), float(b)) for a, b in (d.get("segments") or [])]
            steps = [float(e) for e in (d.get("steps") or [])]
            domain = d.get("domain")
            terms = d.get("terms") or {}
            if not isinstance(terms, dict):
                raise ValueError("'terms' must be an object")
        except (OSError, ValueError, TypeError) as exc:
            self._session_note = (
                f"session NOT restored — {path} is unusable ({exc}).\n"
                f"Starting from the declared defaults; the file is left as is."
            )
            self.summary.setPlainText(self._session_note)
            self._mode_changed()
            return False

        self.cb_linear.setChecked(bool(terms.get("linear", True)))
        self.cb_periodic.setChecked(bool(terms.get("periodic", True)))
        self._clear_segments()
        for lo, hi in segments:
            self.segment_regions.append(self._add_region((lo, hi), SEGMENT_COLOR))
        if isinstance(domain, (list, tuple)) and len(domain) == 2:
            for r in self.domain_regions:
                r.blockSignals(True)
                r.setRegion((float(domain[0]), float(domain[1])))
                r.blockSignals(False)
        for e in steps:
            self.step_lines.append(self._add_line(e, STEP_COLOR))
        if d.get("hold_from"):
            self.hold_from.setText(str(d["hold_from"]))
        if d.get("mode") in (MODE_BACKGROUND, MODE_EVENTS):
            self.mode.blockSignals(True)
            self.mode.setCurrentText(str(d["mode"]))
            self.mode.blockSignals(False)
        self._mode_changed()
        return True


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - GUI
    import argparse

    p = argparse.ArgumentParser(
        prog="gps-detrend-picker-qt",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("station")
    p.add_argument("--tot-dir", default=None)
    p.add_argument(
        "--uncert",
        type=int,
        default=WORKBENCH_UNCERT_DEFAULT,
        help=f"formal-sigma screen [mm] applied at READ time (default: "
        f"{WORKBENCH_UNCERT_DEFAULT}, the workbench's own). int, not float, "
        f"because the emitted command has to be executable by "
        f"gps-detrend-workbench",
    )
    p.add_argument("--max-gap-years", type=float, default=None)
    p.add_argument(
        "--provisional-days",
        type=float,
        default=None,
        help="recency bound for the GOLD provisional lane [days]; 0 disables",
    )
    args = p.parse_args(argv)
    sta = args.station.upper()

    try:
        _pg, QtWidgets = _require_qt()
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 4

    import numpy as np

    import geo_dataread.gps_read as gpsr
    from geo_dataread.detrend_estimate import (
        FitDefaults,
        default_fit_catalog_path,
        read_fit_catalog,
        resolve_fit_settings,
    )

    yearf, data, sigma, _off = gpsr.getData(
        sta, ref="plate", Dir=args.tot_dir, tType="TOT", uncert=args.uncert
    )
    if yearf is None or len(yearf) == 0:
        print(f"error: no data for station {sta}", file=sys.stderr)
        return 2

    from pathlib import Path

    catalog = source = None
    path = default_fit_catalog_path()
    if path and Path(path).is_file():
        catalog, source = read_fit_catalog(path), str(path)
    # PLAIN FitDefaults, with --max-gap-years applied later as an OVERRIDE --
    # the order the workbench uses. Baking it into the defaults put it BELOW
    # the catalog row, so on a station whose fit_windows.csv sets its own gate
    # the flag was silently discarded.
    settings = resolve_fit_settings(sta, catalog, FitDefaults(), catalog_source=source)

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    # Pin the Wayland app_id rather than letting Qt derive one from argv[0]:
    # the sway scratchpad rules match on app_id, and a binding that depends on
    # how the program happened to be invoked is a binding that breaks.
    app.setApplicationName("gps-detrend-picker")
    app.setDesktopFileName("gps-detrend-picker")
    window = PickerWindow(
        sta,
        np.asarray(yearf, float),
        np.atleast_2d(np.asarray(data, float)),
        np.atleast_2d(np.asarray(sigma, float)),
        settings,
        max_gap_years=args.max_gap_years,
        uncert=args.uncert,
        provisional_days=args.provisional_days,
        tot_dir=args.tot_dir,
    )
    window.win.show()
    return int(app.exec())


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
