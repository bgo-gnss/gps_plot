"""Pick a detrend setup by pointing at the data, and print the command.

The desktop picker. Its predecessors were rejected for good reasons and both
lessons are built in here:

* the PDF workflow (edit a flag, re-run, open a viewer) is too slow to take
  as many looks as curation needs;
* the marimo notebook made it *reactive* but not *pointed* — sliders and
  dropdowns are typing numbers with extra steps, which is "the terminal in a
  browser".

So everything here is placed **on the data**: drag the fit domain to where
the series stops being quiet, double-click the jump you want declared as a
step, drag the transient onset to where deformation starts. The fit re-runs
on release (~0.2–0.4 s) and the trajectory redraws, so the consequence of a
pick is visible immediately.

It still **never stores anything**. The output is a
``gps-detrend-workbench`` command line, which remains the only path to
committed science — so every refusal the CLI makes still applies, and this
window cannot store what the CLI would reject. One grammar, several
producers (``geo_dataread.stage_plan`` and ``geo_dataread.term_spec`` are
the grammar; this is a third producer alongside the CLI and the notebook).

Honest trade, stated because it is the reason matplotlib was the earlier
choice: this is a FAITHFUL BUT DIFFERENT view from the PDF. pyqtgraph draws
the same numbers with the same vocabulary (red kept, grey flagged, gold
provisional, blue trajectory, coloured event lines) but not the same
renderer, so judge borderline cosmetics on the PDF. What it buys is
interaction the publication figure cannot give: grab-handled regions,
draggable markers and redraw fast enough to explore with.

Prior art, surveyed 2026-08-03; each idea below is attributed where it is
used:

* **SARI** — Santamaría-Gómez, A. (2019), *SARI: interactive GNSS position
  time series analysis software*, GPS Solutions 23:52,
  doi:10.1007/s10291-019-0846-y. R/Shiny, browser-based. Same term algebra
  (polynomial + offsets + sinusoids + exp/log decays); WLS and Kalman
  fitters. Notably **digests equipment changes from IGS site logs, GAMIT
  ``station.info``, the NGL offsets file or a custom offsets file** to flag
  candidate discontinuities, and can subtract a nearby station's series.
* **TSAnalyzer** — Wu, D., Yan, H., Shen, Y. (2017), *TSAnalyzer, a GNSS
  time series analysis software*, GPS Solutions 21:1389–1394,
  doi:10.1007/s10291-017-0637-2. Python + Qt, the same stack as this module,
  arrived at independently. Semi-automatic offset detection via ``sigseg``
  (Vitti 2012) then manual inspection, and **records picked offsets in a
  JSON file that can be reloaded**, replotted as coloured lines by kind.
* **Snuffler** — Heimann et al., Pyrocko seismological toolbox,
  https://pyrocko.org/docs/current/apps/snuffler/manual.html. The mature
  interactive-picking tradition: distinct marker TYPES, double-click to
  pick, click-and-drag for a time span, markers attached to a trace or
  global, and a plugin system (*snufflings*).
* Both GNSS tools cite **Gazeaux et al. (2013)**, doi:10.1002/jgrb.50152,
  for the finding this whole tool rests on: automatic offset detection has
  not matched expert manual inspection.

What is NOT borrowed, deliberately: both published GNSS tools are
GUI-PRIMARY, with the model inside the interface. This one is CLI-first and
emits a command, so a picked record is reproducible from a line of text
rather than from a session file.

Requires ``pyqtgraph`` and ``PySide6``, which live in the DEV group — a
local development tool, never a production dependency (``uv sync`` installs
them; a production install is unaffected).
"""

from __future__ import annotations

import os
import shlex
import sys
from typing import Any

# ``gps-detrend-workbench``'s own ``--uncert`` default, IMPORTED rather than
# mirrored so the emitted command omits the flag exactly when the workbench
# would default to the same screen. A copy could drift; this cannot.
from gps_plot.detrend_workbench import WORKBENCH_UNCERT_DEFAULT

__all__ = ["main"]

# The cleaned-view vocabulary, kept deliberately in step with
# ``timesmatplt``'s constants so the two tools cannot drift apart.
KEPT_COLOR = (214, 39, 40)  # red   — in the fit
FLAG_COLOR = (150, 150, 150)  # grey  — flagged outlier, masked
FIT_COLOR = (31, 119, 180)  # blue  — the fitted trajectory
DOMAIN_COLOR = (31, 119, 180, 40)
STAGE_COLOR = (255, 127, 14, 55)
OUTSIDE_COLOR = (105, 105, 105)  # dimgrey — view-flagged OUTSIDE the fit window
PROV_FACE = (255, 215, 0)  # gold          — provisional: verdict PENDING
PROV_EDGE = (184, 134, 11)  # darkgoldenrod
TOS_EVENT_COLOR = (0, 100, 0)  # darkgreen — TOS equipment change
SEISMIC_EVENT_COLOR = (139, 0, 0)  # darkred — declared seismic step
STEP_COLOR = (140, 20, 20)  # dark red   — declared step
ONSET_COLOR = (44, 120, 44)  # dark green — transient onset
COMPONENTS = ("North", "East", "Up")

#: The three things a term group can be, and they are three CLAIMS, not two.
#: A seasonal HELD from the quiet window asserts it continues across the whole
#: span; a seasonal ABSENT asserts there is none. A two-state toggle collapses
#: that distinction, and it is the distinction the background model rests on.
#: ``--model`` and the stage plan are orthogonal in the estimator -- one
#: decides which terms are in the design matrix, the other where each is
#: estimated -- which is what makes three states expressible at all.
STATE_ESTIMATE = "estimate here"
STATE_HOLD = "hold from window"
STATE_ABSENT = "not in the model"
GROUP_STATES = (STATE_ESTIMATE, STATE_HOLD, STATE_ABSENT)

#: ``(secular, periodic)`` states -> the stored ``--model``.  Both absent has
#: no spelling: every model in the vocabulary carries at least one of them.
MODEL_BY_STATE: dict[tuple[bool, bool], str] = {
    (True, True): "lineperiodic",
    (True, False): "linear",
    (False, True): "periodic",
}

#: ``station_record_from_arrays``' own default, so the emitted command stays
#: clean when the operator has not moved away from it.
DEFAULT_MODEL = "lineperiodic"


def _provisional_days_default() -> float:
    """geo_dataread's own gold-lane recency bound, for the spinbox's start.

    Imported rather than restated: the spinbox is only showing what the fit
    would use anyway, and a number copied here would be a second definition
    of the same default that could drift from the leaf's.
    """
    try:
        from geo_dataread.gps_views import PROVISIONAL_DAYS

        return float(PROVISIONAL_DAYS)
    except Exception:  # pragma: no cover - leaf rename/removal
        return 14.0


def _require_qt() -> tuple[Any, Any]:
    """Import the Qt stack, or explain how to get it."""
    try:
        import pyqtgraph as pg
        from PySide6 import QtWidgets
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "the Qt picker needs pyqtgraph and PySide6, which are a local "
            "development dependency rather than a production one. Install "
            "with:  uv sync   (or use gps-detrend-picker for the matplotlib "
            "fallback, or the CLI's --segment/--stage/--term directly)"
        ) from exc
    return pg, QtWidgets


class PickerWindow:  # pragma: no cover - GUI
    """The window: three linked panels, draggable picks, a live refit."""

    def __init__(
        self,
        sta: str,
        yearf: Any,
        data: Any,
        sigma: Any,
        settings: Any,
        *,
        max_gap_years: float | None,
        uncert: int,
        provisional_days: float | None = None,
        tot_dir: str | None = None,
    ) -> None:
        pg, QtWidgets = _require_qt()
        import numpy as np

        self.pg, self.QtWidgets, self.np = pg, QtWidgets, np
        self.sta = sta
        self.yearf, self.data, self.sigma = yearf, data, sigma
        self.base_settings = settings
        self.max_gap_years, self.uncert = max_gap_years, uncert
        self.provisional_days = provisional_days
        # Kept only so the emitted command can carry it: a picker run against
        # a non-default TOT directory used to emit a command that reads the
        # DEFAULT one, i.e. a different series entirely.
        self.tot_dir = tot_dir
        self.span = (float(np.nanmin(yearf)), float(np.nanmax(yearf)))
        # The domain region opens on the CATALOG's fit window, not on the data
        # span.  Starting at the span asserted "fit everything" over a station
        # whose `fit_windows.csv` row says otherwise, and then -- because
        # `--segment` was emitted only when the region differed from the SPAN
        # -- an untouched region emitted no flag at all, so the copied command
        # fitted the catalog window while the figure showed the full span.
        # The same divergence as the picked-step bug, one lever over.
        lo, hi = self.base_settings.window  # hull; either side may be open
        self.default_domain = (
            self.span[0] if lo is None else max(float(lo), self.span[0]),
            self.span[1] if hi is None else min(float(hi), self.span[1]),
        )
        self.step_lines: list[Any] = []
        self._prov_counts: list[int] = [0, 0, 0]
        self.record: dict[str, Any] | None = None
        # Composed from the group states on every `_current()`; seeded here so
        # the attribute exists before the first refit.
        self.model: str | None = DEFAULT_MODEL

        pg.setConfigOptions(antialias=False, background="w", foreground="k")
        self.win = QtWidgets.QMainWindow()
        self.win.setWindowTitle(f"{sta} — detrend picker")
        central = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central)

        self.header = QtWidgets.QLabel()
        self.header.setStyleSheet(
            "font-family: monospace; font-size: 13px; padding: 4px;"
        )
        self.header.setTextInteractionFlags(self.pg.QtCore.Qt.TextSelectableByMouse)
        layout.addWidget(self.header)

        self.glw = pg.GraphicsLayoutWidget()
        layout.addWidget(self.glw, stretch=1)
        self.plots: list[Any] = []
        self.fit_curves: list[Any] = []
        self.kept_scatters: list[Any] = []
        self.flag_scatters: list[Any] = []
        self.outside_scatters: list[Any] = []
        self.prov_scatters: list[Any] = []
        for c, name in enumerate(COMPONENTS):
            p = self.glw.addPlot(row=c, col=0)
            p.setLabel("left", f"{name} [mm]")
            p.showGrid(x=True, y=True, alpha=0.25)
            if c:
                p.setXLink(self.plots[0])
            finite = np.isfinite(yearf) & np.isfinite(data[c])
            # Red is the KEPT series only. Production masks a flagged epoch
            # (NaN) and redraws it grey, so drawing it red with a grey ring
            # would say "in the fit, and also flagged" -- two different
            # claims. Re-masked on every refit.
            self.kept_scatters.append(
                p.plot(
                    yearf[finite],
                    data[c][finite],
                    pen=None,
                    symbol="o",
                    symbolSize=3,
                    symbolBrush=KEPT_COLOR,
                    symbolPen=None,
                )
            )
            self.flag_scatters.append(
                p.plot(
                    [],
                    [],
                    pen=None,
                    symbol="o",
                    symbolSize=5,
                    symbolBrush=None,
                    symbolPen=pg.mkPen(FLAG_COLOR, width=1),
                )
            )
            # Second grey lane: the VIEW detector's verdict on epochs the fit
            # window excluded. Hollow and dimmer so the two greys stay
            # countable apart -- they are masked for different reasons ("not
            # in the fit" cannot justify it, since NO out-of-window epoch is).
            self.outside_scatters.append(
                p.plot(
                    [],
                    [],
                    pen=None,
                    symbol="o",
                    symbolSize=5,
                    symbolBrush=None,
                    symbolPen=pg.mkPen(OUTSIDE_COLOR, width=1),
                )
            )
            # Gold: PROVISIONAL. These epochs stay IN the series -- the marker
            # says only that the verdict is pending, because nothing follows
            # them yet and a blunder looks identical to the onset of
            # deformation. Drawn as an overlay on the kept series, never
            # instead of it, and it WILL change as epochs arrive.
            self.prov_scatters.append(
                p.plot(
                    [],
                    [],
                    pen=None,
                    symbol="o",
                    symbolSize=6,
                    symbolBrush=PROV_FACE,
                    symbolPen=pg.mkPen(PROV_EDGE, width=1),
                )
            )
            self.fit_curves.append(p.plot([], [], pen=pg.mkPen(FIT_COLOR, width=2)))
            self.plots.append(p)
        self.plots[-1].setLabel("bottom", "fractional year")
        self._draw_declared_events()

        # Fourth panel: the residual periodogram. Both published GNSS pickers
        # carry one (SARI's Lomb-Scargle + wavelet; TSAnalyzer's Lomb-Scargle),
        # and it answers the question the time-domain panels cannot: is the
        # SEASONAL adequately modelled? Leftover power at 1 cycle/yr says the
        # stage-1 window is drawing the seasonal from the wrong stretch.
        # Lomb-Scargle rather than an FFT because GNSS series are gappy and
        # unevenly sampled (Lomb 1976; Scargle 1982) -- resampling to force an
        # FFT would invent the very structure being tested for.
        self.spec = self.glw.addPlot(row=3, col=0)
        self.spec.setLabel("left", "residual power")
        self.spec.setLabel("bottom", "cycles per year")
        self.spec.showGrid(x=True, y=True, alpha=0.25)
        self.spec.setMaximumHeight(190)
        self.spec_curves = [
            self.spec.plot([], [], pen=pg.mkPen(col, width=1))
            for col in ((200, 60, 60), (60, 140, 60), (60, 60, 200))
        ]
        for f, lbl in ((1.0, "annual"), (2.0, "semi")):
            mark = pg.InfiniteLine(
                pos=f,
                angle=90,
                movable=False,
                pen=pg.mkPen((150, 150, 150), width=1, style=self.pg.QtCore.Qt.DotLine),
                label=lbl,
                labelOpts={"position": 0.9, "color": (110, 110, 110)},
            )
            self.spec.addItem(mark)

        # --- the picks -------------------------------------------------
        # Regions live on every panel and move together: a fit domain that
        # differed between components would be meaningless, and seeing the
        # same window on all three is most of the point.
        self.domain_regions = self._add_region(self.default_domain, DOMAIN_COLOR)
        self.stage_regions = self._add_region(
            (self.span[0], min(self.span[0] + 8.0, self.span[1])), STAGE_COLOR
        )
        self._set_visible(self.stage_regions, False)
        self.onset_lines = self._add_line(
            (self.span[0] + self.span[1]) / 2.0, ONSET_COLOR
        )
        self._set_visible(self.onset_lines, False)

        self.summary = QtWidgets.QPlainTextEdit()
        self.summary.setReadOnly(True)
        self.summary.setMinimumHeight(190)
        self.summary.setStyleSheet("font-family: monospace;")

        # Controls and record go in a right-hand column, the plots on the left,
        # with a draggable divider between them. Stacking everything vertically
        # spent the scarcest dimension -- height -- on a one-line control strip,
        # while three stacked component panels plus the periodogram need every
        # bit of it. A column also gives the controls somewhere to GROW: as a
        # single row they were already running out of width.
        right = QtWidgets.QWidget()
        rcol = QtWidgets.QVBoxLayout(right)
        rcol.setContentsMargins(0, 0, 0, 0)
        rcol.addWidget(self._controls())
        rcol.addWidget(self.summary, stretch=1)

        self.split = QtWidgets.QSplitter(self.pg.QtCore.Qt.Horizontal)
        self.split.addWidget(self.glw)
        self.split.addWidget(right)
        # Plots take the space when the window resizes; the control column is
        # sized to its contents and stays put.
        self.split.setStretchFactor(0, 1)
        self.split.setStretchFactor(1, 0)
        self.split.setSizes([880, 370])
        self.split.setChildrenCollapsible(False)
        layout.addWidget(self.split, stretch=1)

        # The command stays FULL WIDTH along the bottom rather than joining the
        # right column. It is the output of the whole window -- a staged fit
        # with a term and a segment runs past 200 characters -- and it exists to
        # be read and copied, which a third of the window cannot do.
        self.command = QtWidgets.QLineEdit()
        self.command.setReadOnly(True)
        self.command.setStyleSheet("font-family: monospace;")
        layout.addWidget(self.command)

        self.win.setCentralWidget(central)
        self.win.resize(1420, 950)
        self.glw.scene().sigMouseClicked.connect(self._on_click)
        if not self.load_session():
            self.refit()

    def _draw_declared_events(self) -> None:
        """Draw already-DECLARED events, so a pick is informed rather than guessed.

        Borrowed from SARI (Santamaría-Gómez 2019, doi:10.1007/s10291-019-0846-y),
        which digests IGS site logs / GAMIT ``station.info`` / the NGL offsets
        file to flag candidate discontinuities. Our equivalent sources are the
        TOS device history and ``steps.csv``, already resolved by the
        workbench, so the vocabulary matches the PDF exactly: **darkgreen** =
        new antenna/receiver install, **darkred** = declared seismic step.

        These are DECLARATIONS, not detections — nothing here is fitted. They
        are drawn thin and behind the picks so an operator can see that an
        antenna changed on the day they are about to double-click, which is
        the difference between declaring a step and inventing one.
        """
        pg = self.pg
        from gps_plot.detrend_workbench import declared_event_epochs

        events: list[tuple[float, str, tuple[int, int, int]]] = []
        try:
            seismic, _other = declared_event_epochs(self.sta)
            events += [(e, lbl, SEISMIC_EVENT_COLOR) for e, lbl in seismic]
        except Exception:
            pass  # a missing catalog must never cost the window
        try:
            from gps_plot.detrend_workbench import tos_equipment_epochs

            events += [
                (e, lbl, TOS_EVENT_COLOR) for e, lbl in tos_equipment_epochs(self.sta)
            ]
        except Exception:
            # TOS is a live service; being offline is not an error here.
            pass

        lo, hi = self.span
        self.declared_events = [(e, lbl) for e, lbl, _c in events if lo <= e <= hi]
        for epoch, label, colour in events:
            if not (lo <= epoch <= hi):
                continue  # clipped: an off-axis line loses its line and keeps its caption
            for i, p in enumerate(self.plots):
                ln = pg.InfiniteLine(
                    pos=epoch,
                    angle=90,
                    movable=False,
                    pen=pg.mkPen(colour, width=1, style=self.pg.QtCore.Qt.DotLine),
                    label=label if i == 0 else None,
                    labelOpts={
                        "position": 0.92,
                        "color": colour,
                        "fill": (255, 255, 255, 180),
                    },
                )
                ln.setZValue(-20)
                p.addItem(ln)

    # -- construction helpers ------------------------------------------
    def _add_region(self, values: tuple[float, float], colour: Any) -> list[Any]:
        pg = self.pg
        items = []
        for p in self.plots:
            r = pg.LinearRegionItem(values=values, brush=pg.mkBrush(*colour))
            r.setZValue(-10)
            p.addItem(r)
            items.append(r)
        for r in items:
            r.sigRegionChanged.connect(lambda src, g=items: self._sync(src, g))
            r.sigRegionChangeFinished.connect(self.refit)
        return items

    def _add_line(self, pos: float, colour: Any) -> list[Any]:
        pg = self.pg
        items = []
        for p in self.plots:
            ln = pg.InfiniteLine(
                pos=pos,
                angle=90,
                movable=True,
                pen=pg.mkPen(colour, width=2, style=self.pg.QtCore.Qt.DashLine),
            )
            p.addItem(ln)
            items.append(ln)
        for ln in items:
            ln.sigPositionChanged.connect(lambda src, g=items: self._sync_line(src, g))
            ln.sigPositionChangeFinished.connect(self.refit)
        return items

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
    def _set_visible(group: list[Any], on: bool) -> None:
        for item in group:
            item.setVisible(on)

    def _state_combo(self, name: str) -> Any:
        """A three-state control for one term group.

        ``hold`` starts disabled rather than hidden: an operator should be able
        to SEE that holding a group from the clean window is a thing this
        window does, and why it is currently unavailable, instead of the
        option appearing from nowhere when staging is switched on.
        """
        QtWidgets = self.QtWidgets
        combo = QtWidgets.QComboBox()
        combo.addItems(GROUP_STATES)
        combo.setCurrentText(STATE_ESTIMATE)
        combo.setToolTip(
            f"{name}: estimate it here, hold it from the clean window "
            f"(needs 'stage the fit'), or leave it out of the model entirely"
        )
        self._set_state_enabled(combo, STATE_HOLD, False)
        combo.currentIndexChanged.connect(self.refit)
        return combo

    @staticmethod
    def _set_state_enabled(combo: Any, state: str, on: bool) -> None:
        """Grey out one state rather than removing it.

        Removing an item would renumber the rest and silently move whatever
        the operator had selected -- a control changing meaning underneath a
        stored pick is this window's whole family of bugs.
        """
        item = combo.model().item(GROUP_STATES.index(state))
        if item is not None:
            item.setEnabled(on)

    def _group_state(self, name: str) -> str:
        """The selected state for a group, or ``estimate`` when absent."""
        combo = self.grp.get(name)
        return combo.currentText() if combo is not None else STATE_ESTIMATE

    def _controls(self) -> Any:
        """The right-hand control column.

        Grouped by what a control DECIDES, not by widget type: "model" changes
        what is fitted and therefore the record, "picks" only moves what is
        already on the plot. That split is the one an operator needs when
        deciding whether an action is curation or navigation -- the same
        distinction the workbench draws between a stored decision and a
        look-only one.

        Laid out as a column because the old single row had run out of width;
        every widget below is the same object with the same signal as before,
        only re-parented.
        """
        QtWidgets = self.QtWidgets
        box = QtWidgets.QWidget()
        col = QtWidgets.QVBoxLayout(box)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(8)

        # --- model: these change the FIT, and so the emitted command ------
        model_box = QtWidgets.QGroupBox("model")
        mcol = QtWidgets.QVBoxLayout(model_box)

        self.cb_stage = QtWidgets.QCheckBox("stage the fit")
        self.cb_stage.setToolTip(
            "Fit the seasonal on a clean sub-window and hold it for the long "
            "stage (emits --stage/--hold)"
        )
        self.cb_stage.toggled.connect(self._toggle_stage)
        mcol.addWidget(self.cb_stage)

        # Per-group state. `secular` and `periodic` compose the stored
        # --model; `hold` needs a second stage to hold FROM, so it is disabled
        # until staging is on (slice 2 wires it, along with step/transient).
        self.grp: dict[str, Any] = {}
        grid = QtWidgets.QFormLayout()
        for name in ("secular", "periodic"):
            self.grp[name] = self._state_combo(name)
            grid.addRow(name, self.grp[name])
        mcol.addLayout(grid)

        self.cb_term = QtWidgets.QCheckBox("transient")
        self.cb_term.setToolTip(
            "Add a log/exp transient at the green onset line (emits --term)"
        )
        self.cb_term.toggled.connect(self._toggle_term)
        mcol.addWidget(self.cb_term)

        trow = QtWidgets.QHBoxLayout()
        self.kind = QtWidgets.QComboBox()
        self.kind.addItems(["log", "exp"])
        self.kind.currentIndexChanged.connect(self.refit)
        trow.addWidget(self.kind)
        trow.addWidget(QtWidgets.QLabel("tau [yr]"))
        self.tau = QtWidgets.QDoubleSpinBox()
        self.tau.setRange(0.05, 50.0)
        self.tau.setSingleStep(0.1)
        self.tau.setValue(2.0)
        self.tau.editingFinished.connect(self.refit)
        trow.addWidget(self.tau)
        mcol.addLayout(trow)
        col.addWidget(model_box)

        # --- picks: these move what is on the plot, and nothing else ------
        picks_box = QtWidgets.QGroupBox("picks")
        pcol = QtWidgets.QVBoxLayout(picks_box)
        reset = QtWidgets.QPushButton("reset domain")
        reset.setToolTip("Back to the catalog's fit window, not the data span")
        reset.clicked.connect(self._reset_domain)
        pcol.addWidget(reset)
        clear = QtWidgets.QPushButton("clear steps")
        clear.setToolTip(
            "Remove the PICKED steps. Steps declared in steps.csv are a floor "
            "and stay in the fit"
        )
        clear.clicked.connect(self._clear_steps)
        pcol.addWidget(clear)
        save = QtWidgets.QPushButton("save session")
        save.clicked.connect(self.save_session)
        pcol.addWidget(save)

        hint = QtWidgets.QLabel(
            "double-click a jump to declare a step · right-click it to remove"
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #666;")
        pcol.addWidget(hint)
        col.addWidget(picks_box)

        # --- run: parameters that were command-line-only ------------------
        # Each writes the SAME attribute `run_flags` emits, so a control cannot
        # move the figure without moving the command with it.
        run_box = QtWidgets.QGroupBox("run")
        rgrid = QtWidgets.QFormLayout(run_box)

        self.sp_gap = QtWidgets.QDoubleSpinBox()
        self.sp_gap.setRange(0.1, 50.0)
        self.sp_gap.setSingleStep(0.5)
        self.sp_gap.setDecimals(2)
        # Show the EFFECTIVE gate, not the flag: with no --max-gap-years the
        # resolved settings carry the catalog's value (or the spec default),
        # and a spinbox reading 0.5 while the fit ran at 1.0 would be a third
        # place disagreeing about the same number.
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
            "Display only -- it moves no fitted quantity"
        )
        self.sp_prov.editingFinished.connect(self._set_provisional_days)
        rgrid.addRow("provisional [d]", self.sp_prov)

        self.sp_uncert = QtWidgets.QSpinBox()
        self.sp_uncert.setRange(1, 200)
        self.sp_uncert.setValue(self.uncert)
        self.sp_uncert.setToolTip(
            "Formal-sigma screen [mm], applied at READ time. Unlike the others "
            "this RE-READS the series from disk — it changes which epochs "
            "exist, not how they are fitted"
        )
        self.sp_uncert.editingFinished.connect(self._set_uncert)
        rgrid.addRow("uncert [mm]", self.sp_uncert)
        col.addWidget(run_box)

        return box

    # -- run parameters ------------------------------------------------
    # These were command-line-only. Each sets the attribute that `_command`
    # already reads, so the emitted command follows the control by
    # construction rather than by remembering to update it in two places.

    def _set_max_gap(self) -> None:
        value = round(float(self.sp_gap.value()), 4)
        if value == self.max_gap_years:
            return
        self.max_gap_years = value
        self.refit()

    def _set_provisional_days(self) -> None:
        value = round(float(self.sp_prov.value()), 4)
        if value == self.provisional_days:
            return
        self.provisional_days = value
        self.refit()

    def _set_uncert(self) -> None:
        """Re-READ the series under a new sigma screen, then refit.

        The odd one out: ``uncert`` is applied by ``getData`` when the file is
        read, so it decides which epochs EXIST rather than how they are
        fitted. A refit alone would move the emitted command while the figure
        kept showing the old series -- this window's recurring bug wearing a
        new hat -- so the data is re-read here.

        The picks are deliberately left where they are. Screening sigma is not
        a statement about which window to fit, so the domain region, steps and
        onset stay put; only the underlying series changes. A failed read
        keeps the current data and says so, because a picker that silently
        empties itself is worse than one that refuses.
        """
        value = int(self.sp_uncert.value())
        if value == self.uncert:
            return

        import geo_dataread.gps_read as gpsr

        np = self.np
        try:
            yearf, data, sigma, _off = gpsr.getData(
                self.sta, ref="plate", Dir=self.tot_dir, tType="TOT", uncert=value
            )
            yearf = np.asarray(yearf, float)
            if yearf.size == 0:
                raise ValueError("no epochs survive that screen")
        except Exception as exc:
            self.sp_uncert.setValue(self.uncert)  # keep widget and state in step
            self.summary.setPlainText(
                f"uncert {value} mm: re-read failed, keeping {self.uncert} mm\n{exc}"
            )
            return

        before = self.yearf.size
        self.yearf = yearf
        self.data = np.atleast_2d(np.asarray(data, float))
        self.sigma = np.atleast_2d(np.asarray(sigma, float))
        self.span = (float(np.nanmin(yearf)), float(np.nanmax(yearf)))
        self.uncert = value
        self.refit()
        self._note_epoch_change(before, yearf.size, value)

    def _note_epoch_change(self, before: int, after: int, uncert: int) -> None:
        """Say how many epochs the screen took, above the record.

        The count is the whole point of the control and it is invisible
        otherwise: a tighter screen that drops 40 of 4713 epochs looks
        identical on a plot of 4673.
        """
        delta = after - before
        if not delta:
            return
        self.summary.setPlainText(
            f"uncert {uncert} mm: {before} -> {after} epochs ({delta:+d})\n\n"
            + self.summary.toPlainText()
        )

    # -- session (TSAnalyzer's reloadable pick file) --------------------
    def _session_path(self) -> Any:
        from pathlib import Path

        base = os.environ.get("XDG_STATE_HOME") or os.path.expanduser("~/.local/state")
        d = Path(base) / "gps-detrend-picker" / "sessions"
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{self.sta}.json"

    def save_session(self) -> None:
        """Write the current picks so they survive closing the window.

        Borrowed from TSAnalyzer (Wu et al. 2017,
        doi:10.1007/s10291-017-0637-2), which "uses a JSON text format to
        record all these data offset events … and all the data offset events
        can thus be recorded and reloaded".

        This is a CONVENIENCE, not a record: the emitted command is what
        reproduces the science, because it is re-parsed by the same grammar
        the CLI uses. A session that could be mistaken for provenance would
        be a second path to stored results, which this tool does not have.
        """
        import json

        lo, hi = (round(v, 4) for v in self.domain_regions[0].getRegion())
        s_lo, s_hi = (round(v, 4) for v in self.stage_regions[0].getRegion())
        payload = {
            "station": self.sta,
            "note": "picker convenience only — the emitted command is the record",
            "domain": [lo, hi],
            "steps": [round(g[0].value(), 4) for g in self.step_lines],
            "stage": {"on": self.cb_stage.isChecked(), "window": [s_lo, s_hi]},
            "term": {
                "on": self.cb_term.isChecked(),
                "kind": self.kind.currentText(),
                "epoch": round(self.onset_lines[0].value(), 4),
                "tau": round(self.tau.value(), 3),
            },
            "params": {
                "uncert": self.uncert,
                "max_gap_years": self.max_gap_years,
                "provisional_days": self.provisional_days,
            },
        }
        self._session_path().write_text(json.dumps(payload, indent=2))
        self.summary.setPlainText(f"session saved -> {self._session_path()}")

    def _parse_session(self, d: Any) -> dict[str, Any]:
        """Validate a session payload into plain values, touching no widget.

        Separate from :meth:`load_session` so a malformed field cannot
        HALF-apply: every field is parsed first, and only a payload that
        parses in full is allowed to move anything on screen. Applying as it
        went would leave the picker in a state that is neither the session
        nor the defaults, with a fit on screen matching no command.

        Raises:
            ValueError: naming the offending field.
        """

        def pair(v: Any, field: str) -> tuple[float, float]:
            if not isinstance(v, (list, tuple)) or len(v) != 2:
                raise ValueError(f"{field}: expected two numbers, got {v!r}")
            try:
                lo, hi = float(v[0]), float(v[1])
            except (TypeError, ValueError):
                raise ValueError(f"{field}: not numeric: {v!r}") from None
            if not (lo < hi):
                raise ValueError(f"{field}: {lo} is not below {hi}")
            return lo, hi

        if not isinstance(d, dict):
            raise ValueError(f"top level must be an object, got {type(d).__name__}")
        stage = d.get("stage") or {}
        term = d.get("term") or {}
        if not isinstance(stage, dict) or not isinstance(term, dict):
            raise ValueError("'stage' and 'term' must be objects")
        steps_raw = d.get("steps") or []
        if not isinstance(steps_raw, (list, tuple)):
            raise ValueError(f"'steps' must be a list, got {steps_raw!r}")
        try:
            steps = [float(e) for e in steps_raw]
        except (TypeError, ValueError):
            raise ValueError(f"'steps': not all numeric: {steps_raw!r}") from None

        out: dict[str, Any] = {
            "domain": (
                pair(d["domain"], "domain")
                if d.get("domain") is not None
                else self.default_domain
            ),
            "stage_window": (
                pair(stage["window"], "stage.window")
                if stage.get("window") is not None
                else self.span
            ),
            "stage_on": bool(stage.get("on")),
            "term_on": bool(term.get("on")),
            "steps": steps,
            "kind": term.get("kind") if term.get("kind") in ("log", "exp") else None,
        }
        for key, field in (("epoch", "term.epoch"), ("tau", "term.tau")):
            raw = term.get(key)
            if raw in (None, 0, 0.0):
                out[key] = None
                continue
            try:
                out[key] = float(raw)
            except (TypeError, ValueError):
                raise ValueError(f"{field}: not numeric: {raw!r}") from None
        return out

    def load_session(self) -> bool:
        """Restore picks written by :meth:`save_session`. Returns True if any.

        Called at launch, which is why nothing here may raise: a session file
        that is unreadable OR structurally wrong used to take the whole
        application down before the window appeared — leaving the operator no
        way in to clear the very file that was killing it. A bad session now
        degrades to the defaults and says where the file is, because the file
        is somebody's curation and deleting it unasked is the worse failure.
        """
        import json

        path = self._session_path()
        if not path.exists():
            return False
        try:
            d = self._parse_session(json.loads(path.read_text()))
        except (OSError, ValueError, TypeError) as exc:
            self.summary.setPlainText(
                f"session NOT restored — {path} is unusable ({exc}).\n"
                f"Starting from the declared defaults; the file is left as is."
            )
            return False
        for r in self.domain_regions:
            r.blockSignals(True)
            # a session with no stored domain falls back to the DECLARED one,
            # for the same reason the initial region does
            r.setRegion(d["domain"])
            r.blockSignals(False)
        for r in self.stage_regions:
            r.blockSignals(True)
            r.setRegion(d["stage_window"])
            r.blockSignals(False)
        self.cb_stage.setChecked(d["stage_on"])
        self.cb_term.setChecked(d["term_on"])
        if d["kind"] is not None:
            self.kind.setCurrentText(d["kind"])
        if d["epoch"] is not None:
            for ln in self.onset_lines:
                ln.blockSignals(True)
                ln.setValue(d["epoch"])
                ln.blockSignals(False)
        if d["tau"] is not None:
            self.tau.setValue(d["tau"])
        self._clear_steps_quiet()
        for e in d["steps"]:
            self.step_lines.append(self._add_line(e, STEP_COLOR))
        self.refit()
        return True

    def _clear_steps_quiet(self) -> None:
        for group in self.step_lines:
            for p, ln in zip(self.plots, group, strict=True):
                p.removeItem(ln)
        self.step_lines.clear()

    # -- interaction ----------------------------------------------------
    def _toggle_stage(self, on: bool) -> None:
        self._set_visible(self.stage_regions, on)
        self.refit()

    def _toggle_term(self, on: bool) -> None:
        self._set_visible(self.onset_lines, on)
        self.refit()

    def _reset_domain(self) -> None:
        """Back to the DECLARED domain, which is what "reset" has to mean.

        Resetting to the data span would be a third way to say "fit
        everything" and would leave the picker asserting something the
        catalog does not.
        """
        for r in self.domain_regions:
            r.blockSignals(True)
            r.setRegion(self.default_domain)
            r.blockSignals(False)
        self.refit()

    def _clear_steps(self) -> None:
        for group in self.step_lines:
            for p, ln in zip(self.plots, group, strict=True):
                p.removeItem(ln)
        self.step_lines.clear()
        self.refit()

    def _on_click(self, ev: Any) -> None:
        """Double-click places a step; right-click on one removes it."""
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
        group = self._add_line(epoch, STEP_COLOR)
        self.step_lines.append(group)
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

    # -- the fit --------------------------------------------------------
    def _current(self) -> tuple[Any, list[str], list[str], list[str]]:
        """Read the picks off the plot into settings + CLI flags.

        Settings and flags are assembled by the SAME function the workbench
        CLI uses (``_override_settings``), because the picker's promise is
        that the emitted command reproduces the figure.  Building the
        settings here instead is what broke it: a picked step REPLACED the
        station's declared ones, while ``--step`` on the command line merges
        with them.  On SELF that was the difference between an aborted fit
        showing a stale curve and a clean one -- the undeclared 2008 Ölfus
        coseismic trips the excess-candidate rule -- and on a milder station
        it would have been a silent difference in rate.
        """
        from gps_plot.detrend_workbench import _override_settings

        lo, hi = (round(v, 4) for v in self.domain_regions[0].getRegion())
        extra: list[str] = []
        # An UNTOUCHED region defers to the catalog entirely -- no flag, and
        # `segments=None` so `_override_settings` leaves the row's own
        # segments in place.  That matters beyond tidiness: a catalog row may
        # declare a UNION of intervals, and one region cannot express a union,
        # so overriding it with the hull would silently re-include an
        # excision the operator never touched.  Once the region IS moved, the
        # single interval is both fitted and emitted, and `--segment`
        # replaces the row -- figure and command agree either way.
        moved = (lo, hi) != tuple(round(v, 4) for v in self.default_domain)
        segments = ((lo, hi),) if moved else None
        if moved:
            extra += ["--segment", f"{lo}:{hi}"]

        # Only the PICKED steps become --step flags; the declared ones are
        # already the workbench's floor, so emitting them would be a no-op
        # at best and a double-declaration at worst.
        steps = tuple(round(g[0].value(), 4) for g in self.step_lines)
        for e in steps:
            extra += ["--step", str(e)]

        # max_gap_years goes through the SAME override the workbench applies,
        # from the same live attribute the emitted command reads -- so the
        # spinbox cannot change the command without changing the fit.
        settings = _override_settings(
            self.base_settings,
            self.sta,
            quiet=True,
            segments=segments,
            steps=steps or None,
            max_gap_years=self.max_gap_years,
        )

        # The stored --model is COMPOSED from the group states, in the same
        # place the flags are built, so the design matrix the picker fits and
        # the model the copied command asks for cannot come apart.
        self.model = MODEL_BY_STATE.get(
            (
                self._group_state("secular") != STATE_ABSENT,
                self._group_state("periodic") != STATE_ABSENT,
            )
        )
        if self.model is not None and self.model != DEFAULT_MODEL:
            extra += ["--model", self.model]

        terms: list[str] = []
        if self.cb_term.isChecked():
            terms.append(
                f"{self.kind.currentText()}@{round(self.onset_lines[0].value(), 4)}"
                f",tau={round(self.tau.value(), 3)}"
            )

        stages: list[str] = []
        if self.cb_stage.isChecked():
            s_lo, s_hi = (round(v, 4) for v in self.stage_regions[0].getRegion())
            stages = [f"clean:secular,periodic@{s_lo}:{s_hi}", "long:secular"]
        return settings, extra, terms, stages

    def refit(self, *_: Any) -> None:
        from geo_dataread.stage_plan import build_stage_plan
        from gps_plot.detrend_workbench import (
            abort_fallback_note,
            estimate_with_abort_fallback,
        )

        np = self.np
        settings, extra, terms, stage_specs = self._current()

        plan = None
        note = ""
        if self.model is None:
            # Both secular and periodic absent. There is no --model spelling
            # for it -- every value in the vocabulary carries at least one --
            # so this is refused HERE rather than sent to the estimator, which
            # would fail later and less clearly. Refused loudly, previous
            # curve kept, per the composer's own convention.
            note = (
                "fit refused: secular and periodic are both 'not in the "
                "model', which no --model value can express. Leave at least "
                "one of them in."
            )
        if stage_specs and not note:
            # The long stage must free `step` whenever the fit CARRIES a step
            # column, and the picked lines are only half of where those come
            # from: `steps.csv` is a FLOOR that `_override_settings` merges
            # in, so ask the merged settings, never the picks.  Reading
            # `self.step_lines` here meant that on a station with a declared
            # step (SELF, 2008 Ölfus) an untouched stage plan freed only
            # `secular` and EVERY fit was refused -- `step_amp_1` estimated
            # in no stage and held in none -- with no hint that
            # re-declaring the already-declared step was the way out. The
            # same shape as the picked-step bug one lever over: a second
            # place reasoning about steps that knew only about the picked
            # ones.
            from gps_plot.detrend_workbench import _declared_step_epochs

            has_step = bool(_declared_step_epochs(self.sta, settings.steps))
            groups = ("secular", "periodic", "step", "transient")
            free2 = [
                g
                for g in groups
                if g == "secular"
                or (g == "step" and has_step)
                or (g == "transient" and terms)
            ]
            stage_specs = [stage_specs[0], f"long:{','.join(free2)}"]
            try:
                plan = build_stage_plan(stage_specs, ["long:periodic=stage:clean"])
            except ValueError as exc:
                note = f"stage plan refused: {exc}"

        est = None
        if not note:
            try:
                # The SAME fallback the workbench applies: a full-detection
                # abort is recoverable (retry S0-only), a failed gate is not.
                # Calling estimate_record directly meant the picker gave up
                # where the emitted command produced a perfectly good figure --
                # reachable as soon as `--model periodic` became selectable,
                # since a periodic-only model leaves the trend in the residuals
                # and the candidate fraction trips the abort.
                est, fell_back = estimate_with_abort_fallback(
                    self.sta,
                    self.yearf,
                    self.data,
                    self.sigma,
                    settings=settings,
                    terms=terms or None,
                    stage_plan=plan,
                    model=self.model,
                )
            except (ValueError, RuntimeError) as exc:
                # A refused fit is a RESULT -- rank-deficient stage, an
                # unfittable term, a domain with too few epochs. Say so and
                # keep the previous trajectory on screen.
                note = f"fit refused: {exc}"

        if est is not None:
            self.record = est.record
            fit = np.asarray(
                __import__("gps_analysis").evaluate_record(est.record, self.yearf)
            )
            outl = np.atleast_2d(np.asarray(est.outliers, dtype=bool))
            empty = np.zeros(self.yearf.shape, dtype=bool)
            # The fit passes NO verdict outside its window, so drawn plain
            # those epochs claim "clean" and one blunder owns the y-axis.
            # Fill that silence with the view detector -- the same chain
            # `plot-gps-timeseries --view cleaned` uses -- which is also the
            # only thing that HAS a provisional category.
            try:
                from gps_plot.detrend_workbench import screen_outside_window

                outside, prov = screen_outside_window(
                    self.sta,
                    self.yearf,
                    self.data,
                    self.sigma,
                    est,
                    steps=tuple(round(g[0].value(), 4) for g in self.step_lines)
                    or None,
                    provisional_days=self.provisional_days,
                )
            except Exception:
                # The screen is an AID; never lose the figure over it.
                outside = prov = None
            outside = (
                None
                if outside is None
                else np.atleast_2d(np.asarray(outside, dtype=bool))
            )
            prov = None if prov is None else np.atleast_2d(np.asarray(prov, dtype=bool))
            # The CURVE is drawn on its own dense grid, while `fit` above
            # stays at the data epochs because the residual spectrum needs
            # data minus model at the SAME epochs. Two samplings, two
            # purposes -- joining the data epochs drew a straight chord over
            # every gap, which claims linear motion the model never fitted.
            from gps_plot.detrend_workbench import trajectory_curve

            fit_x, fit_y = trajectory_curve(est.record, self.yearf)
            for c in range(3):
                # connect="finite" so a component the model cannot evaluate
                # breaks the line instead of being joined across. The step
                # epochs do NOT rely on it -- trajectory_curve brackets those
                # so the jump draws vertical.
                self.fit_curves[c].setData(fit_x, fit_y[c], connect="finite")
                finite = np.isfinite(self.data[c])
                flagged = outl[c] & finite
                out_c = outside[c] & finite if outside is not None else empty
                prov_c = prov[c] & finite if prov is not None else empty
                # Both greys MASK; gold does not. So gold stays in the kept
                # series and is only overlaid.
                kept = finite & ~outl[c] & ~out_c
                self.kept_scatters[c].setData(self.yearf[kept], self.data[c][kept])
                self.flag_scatters[c].setData(
                    self.yearf[flagged], self.data[c][flagged]
                )
                self.outside_scatters[c].setData(self.yearf[out_c], self.data[c][out_c])
                self.prov_scatters[c].setData(self.yearf[prov_c], self.data[c][prov_c])
                self._prov_counts[c] = int(prov_c.sum())
            self._update_spectrum(fit)
            text = self._summary(est.record)
            if fell_back:
                # Same words the CLI prints to stderr -- an S0 record is
                # not the same object as a full-detection one, and the
                # operator has to know which one they are judging.
                text = f"note: {abort_fallback_note(self.sta)}\n\n{text}"
            self.summary.setPlainText(text)
        else:
            for group in (
                self.flag_scatters,
                self.outside_scatters,
                self.prov_scatters,
            ):
                for sc in group:
                    sc.setData([], [])
            for c in range(3):
                finite = np.isfinite(self.data[c])
                self.kept_scatters[c].setData(self.yearf[finite], self.data[c][finite])
            self.record = None
        if est is None and not note:
            note = (
                "no record: the outlier stage aborted, or a validity gate "
                "rejected this domain (span / epochs / max-gap)"
            )
        if note:
            self.summary.setPlainText(note)

        self.command.setText(self._command(plan, extra, terms))
        self._update_header(terms, plan)

    def _update_header(self, terms: list[str], plan: Any) -> None:
        """Station, run parameters and the live picks, always on screen.

        The run parameters matter enough to display rather than remember:
        ``uncert`` screens sigma at READ time, so it changes WHICH epochs are
        fitted while leaving no trace in any fitted quantity, and
        ``max_gap_years`` decides whether the station is estimable at all.
        Two records that differ only in these are indistinguishable from
        their numbers.
        """
        lo, hi = (round(v, 4) for v in self.domain_regions[0].getRegion())
        rec = self.record
        bits = [
            f"<b>{self.sta}</b>",
            f"uncert {self.uncert:g} mm",
            f"max-gap {self.max_gap_years if self.max_gap_years is not None else 'default'} yr",
            f"prov {self.provisional_days if self.provisional_days is not None else 'default 14'} d",
            f"domain {lo}:{hi}",
        ]
        # One region cannot draw a union, so when the catalog declares several
        # segments and the region is untouched, the picture is the HULL while
        # the fit is the union. Say so rather than let the excision be
        # invisible; moving the region collapses it to the single interval
        # drawn, which is then also what the command says.
        n_seg = len(self.base_settings.segments)
        if n_seg > 1:
            moved = (lo, hi) != tuple(round(v, 4) for v in self.default_domain)
            bits.append(
                f"⚠ shown as hull of {n_seg} catalog segments"
                if not moved
                else f"⚠ collapses {n_seg} catalog segments to one"
            )
        picked = [round(g[0].value(), 4) for g in self.step_lines]
        # What was FITTED, not what was clicked. steps.csv and the fit
        # catalog are a floor the picks add to, and the window filter drops
        # any step outside the domain, so the two lists routinely differ --
        # and the operator has to see which, because the merge is exactly
        # what used to be missing from this fit.
        fitted = [round(float(v), 4) for v in (rec or {}).get("step_epochs", [])]
        if fitted and fitted != picked:
            bits.append(f"steps picked {picked} → fitting {fitted}")
        elif picked:
            bits.append(f"steps {picked}")
        if terms:
            bits.append(f"term {terms[0]}")
        if plan is not None:
            bits.append("staged")
        if rec is not None:
            n = rec.get("n_rejected")
            method = str(rec.get("detrend_method", ""))
            stages = str((rec.get("refs") or {}).get("outlier_stages", "?"))
            # An abort is the difference between "nothing was wrong" and
            # "nothing was judged", so say which.
            aborted = method == "plain_wls" or (n and not any(n))
            flag = (
                f"<span style='color:#a00'>outliers ABORTED (stages {stages}) "
                f"— nothing judged</span>"
                if aborted
                else f"outliers {n} (stages {stages})"
            )
            bits.append(flag)
            if self._prov_counts and any(self._prov_counts):
                bits.append(
                    f"<span style='color:#8a6d0b'>provisional {self._prov_counts} "
                    f"(gold; KEPT, verdict pending)</span>"
                )
        else:
            # The case where the operator most needs telling: an abort or a
            # failed gate leaves the PREVIOUS trajectory on screen, which
            # otherwise reads as a successful fit of the current picks.
            bits.append(
                "<span style='color:#a00'>NO RECORD — outlier stage aborted, "
                "or a gate rejected this domain; the curve shown is stale"
                "</span>"
            )
        self.header.setText("  ·  ".join(bits))

    def _update_spectrum(self, fit: Any) -> None:
        """Lomb-Scargle periodogram of the fit residuals, per component.

        Equation (Lomb 1976, Astrophys. Space Sci. 39; Scargle 1982, ApJ 263,
        eq. 10): the least-squares power of a sinusoid at angular frequency ω
        fitted to unevenly sampled data, which is why it is the right tool
        here — GNSS series are gappy, and resampling onto a regular grid to
        permit an FFT would manufacture the structure being tested for.

        Read it as a MODEL-ADEQUACY check, not as science: a peak left at
        1 cycle/yr means the stored seasonal does not describe this series,
        usually because the stage-1 window drew it from an unrepresentative
        stretch.
        """
        import numpy as np
        from scipy.signal import lombscargle

        freqs = np.linspace(0.2, 6.0, 400)  # cycles per year
        omega = 2.0 * np.pi * freqs
        for c in range(3):
            good = np.isfinite(self.data[c]) & np.isfinite(fit[c])
            t = self.yearf[good]
            resid = self.data[c][good] - fit[c][good]
            if t.size < 32 or not np.isfinite(resid).all():
                self.spec_curves[c].setData([], [])
                continue
            resid = resid - resid.mean()
            if not np.any(resid):
                self.spec_curves[c].setData([], [])
                continue
            try:
                power = lombscargle(t, resid, omega, normalize=True)
            except (ValueError, ZeroDivisionError):
                self.spec_curves[c].setData([], [])
                continue
            self.spec_curves[c].setData(freqs, np.asarray(power, dtype=float))

    def _summary(self, rec: dict[str, Any]) -> str:
        # `rms` is rounded for DISPLAY, to 2 dp like the PDF's `summarise`.
        # Raw it prints as [1.8299068022476694, 2.8976389308583466, ...], which
        # wrapped over three lines in the control column and buried the values
        # that are actually being compared between iterations. Sub-micrometre
        # digits on a millimetre residual are noise either way.
        keys = ("model", "window", "n_epochs", "n_rejected", "rms", "step_epochs")
        lines = []
        for k in keys:
            value = rec.get(k)
            if k == "rms" and value is not None:
                value = [round(float(v), 2) for v in value]
            lines.append(f"{k:14s} {value}")
        rate = [round(float(c["params"][1]), 2) for c in rec["components"]]
        lines.append(f"{'rate [mm/yr]':14s} {rate}")
        lines.append(f"{'record_version':14s} {rec.get('record_version')}")
        return "\n".join(lines)

    def _command(self, plan: Any, extra: list[str], terms: list[str]) -> str:
        from gps_plot.detrend_picker import render_command
        from gps_plot.detrend_workbench import run_flags

        # Assembled by the workbench's own `run_flags`, shared with the
        # marimo picker: two pickers building this list independently is two
        # places to forget the same flag, and each of them forgot
        # `--tot-dir`. It also formats `--uncert` as the int the workbench's
        # `type=int` will accept -- "12.0" emitted a command that does not
        # parse.
        flags = list(extra) + run_flags(
            tot_dir=self.tot_dir,
            max_gap_years=self.max_gap_years,
            uncert=self.uncert,
            provisional_days=self.provisional_days,
        )
        if plan is None:
            parts = ["gps-detrend-workbench", self.sta]
            for t in terms:
                parts += ["--term", t]
            return shlex.join(parts + flags)
        return render_command(self.sta, plan, flags, terms=terms)


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
        help="recency bound for the GOLD provisional lane [days]; 0 disables. "
        "Default is geo_dataread's 14. The bound matters: indeterminate "
        "clusters also sit at old mid-series gaps and would otherwise "
        "dominate the lane",
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
    # PLAIN FitDefaults, with --max-gap-years applied later as an OVERRIDE in
    # `_current` -- the order the workbench uses (`build_record`). Baking it
    # into the defaults here instead put it BELOW the catalog row, so on a
    # station whose fit_windows.csv sets its own gate the flag was silently
    # discarded: `gps-detrend-picker-qt DYNG --max-gap-years 2.0` fitted at the
    # catalog's 1.0 while emitting a command that fits at 2.0. The sixth
    # instance of this window's one recurring bug, and the same shape as the
    # rest -- a second place assembling the same decision.
    settings = resolve_fit_settings(sta, catalog, FitDefaults(), catalog_source=source)

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    # Pin the Wayland app_id rather than letting Qt derive one from argv[0]:
    # the sway scratchpad rules match on app_id, and a binding that depends
    # on how the program happened to be invoked is a binding that breaks.
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
