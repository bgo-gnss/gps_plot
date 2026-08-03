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

Requires ``pyqtgraph`` and ``PySide6``, which live in the DEV group — a
local development tool, never a production dependency (``uv sync`` installs
them; a production install is unaffected).
"""

from __future__ import annotations

import dataclasses
import shlex
import sys
from typing import Any

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
STEP_COLOR = (140, 20, 20)  # dark red   — declared step
ONSET_COLOR = (44, 120, 44)  # dark green — transient onset
COMPONENTS = ("North", "East", "Up")


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
        uncert: float,
        provisional_days: float | None = None,
    ) -> None:
        pg, QtWidgets = _require_qt()
        import numpy as np

        self.pg, self.QtWidgets, self.np = pg, QtWidgets, np
        self.sta = sta
        self.yearf, self.data, self.sigma = yearf, data, sigma
        self.base_settings = settings
        self.max_gap_years, self.uncert = max_gap_years, uncert
        self.provisional_days = provisional_days
        self.span = (float(np.nanmin(yearf)), float(np.nanmax(yearf)))
        self.step_lines: list[Any] = []
        self._prov_counts: list[int] = [0, 0, 0]
        self.record: dict[str, Any] | None = None

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

        # --- the picks -------------------------------------------------
        # Regions live on every panel and move together: a fit domain that
        # differed between components would be meaningless, and seeing the
        # same window on all three is most of the point.
        self.domain_regions = self._add_region(self.span, DOMAIN_COLOR)
        self.stage_regions = self._add_region(
            (self.span[0], min(self.span[0] + 8.0, self.span[1])), STAGE_COLOR
        )
        self._set_visible(self.stage_regions, False)
        self.onset_lines = self._add_line(
            (self.span[0] + self.span[1]) / 2.0, ONSET_COLOR
        )
        self._set_visible(self.onset_lines, False)

        layout.addWidget(self._controls())
        self.summary = QtWidgets.QPlainTextEdit()
        self.summary.setReadOnly(True)
        self.summary.setMaximumHeight(150)
        self.summary.setStyleSheet("font-family: monospace;")
        layout.addWidget(self.summary)
        self.command = QtWidgets.QLineEdit()
        self.command.setReadOnly(True)
        self.command.setStyleSheet("font-family: monospace;")
        layout.addWidget(self.command)

        self.win.setCentralWidget(central)
        self.win.resize(1250, 950)
        self.glw.scene().sigMouseClicked.connect(self._on_click)
        self.refit()

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

    def _controls(self) -> Any:
        QtWidgets = self.QtWidgets
        box = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(box)

        self.cb_stage = QtWidgets.QCheckBox("stage the fit")
        self.cb_stage.toggled.connect(self._toggle_stage)
        row.addWidget(self.cb_stage)

        self.cb_term = QtWidgets.QCheckBox("transient")
        self.cb_term.toggled.connect(self._toggle_term)
        row.addWidget(self.cb_term)

        self.kind = QtWidgets.QComboBox()
        self.kind.addItems(["log", "exp"])
        self.kind.currentIndexChanged.connect(self.refit)
        row.addWidget(self.kind)

        row.addWidget(QtWidgets.QLabel("tau [yr]"))
        self.tau = QtWidgets.QDoubleSpinBox()
        self.tau.setRange(0.05, 50.0)
        self.tau.setSingleStep(0.1)
        self.tau.setValue(2.0)
        self.tau.editingFinished.connect(self.refit)
        row.addWidget(self.tau)

        reset = QtWidgets.QPushButton("reset domain")
        reset.clicked.connect(self._reset_domain)
        row.addWidget(reset)
        clear = QtWidgets.QPushButton("clear steps")
        clear.clicked.connect(self._clear_steps)
        row.addWidget(clear)

        row.addWidget(
            QtWidgets.QLabel(
                "  double-click a jump to declare a step · right-click it to remove"
            )
        )
        row.addStretch(1)
        return box

    # -- interaction ----------------------------------------------------
    def _toggle_stage(self, on: bool) -> None:
        self._set_visible(self.stage_regions, on)
        self.refit()

    def _toggle_term(self, on: bool) -> None:
        self._set_visible(self.onset_lines, on)
        self.refit()

    def _reset_domain(self) -> None:
        for r in self.domain_regions:
            r.blockSignals(True)
            r.setRegion(self.span)
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
        """Read the picks off the plot into settings + CLI flags."""
        lo, hi = (round(v, 4) for v in self.domain_regions[0].getRegion())
        settings = dataclasses.replace(self.base_settings, segments=((lo, hi),))
        extra: list[str] = []
        if (lo, hi) != tuple(round(v, 4) for v in self.span):
            extra += ["--segment", f"{lo}:{hi}"]

        steps = tuple(round(g[0].value(), 4) for g in self.step_lines)
        if steps:
            settings = dataclasses.replace(settings, steps=steps)
            for e in steps:
                extra += ["--step", str(e)]

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
        from geo_dataread.detrend_estimate import station_estimate_from_arrays
        from geo_dataread.stage_plan import build_stage_plan

        np = self.np
        settings, extra, terms, stage_specs = self._current()

        plan = None
        note = ""
        if stage_specs:
            groups = ("secular", "periodic", "step", "transient")
            free2 = [
                g
                for g in groups
                if g == "secular"
                or (g == "step" and self.step_lines)
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
                est = station_estimate_from_arrays(
                    self.sta,
                    self.yearf,
                    self.data,
                    self.sigma,
                    settings=settings,
                    terms=terms or None,
                    stage_plan=plan,
                    lookup_donor=None,
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
            for c in range(3):
                good = np.isfinite(self.yearf) & np.isfinite(fit[c])
                self.fit_curves[c].setData(self.yearf[good], fit[c][good])
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
            self.summary.setPlainText(self._summary(est.record))
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
        if self.step_lines:
            bits.append(f"steps {[round(g[0].value(), 4) for g in self.step_lines]}")
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

    def _summary(self, rec: dict[str, Any]) -> str:
        keys = ("model", "window", "n_epochs", "n_rejected", "rms", "step_epochs")
        lines = [f"{k:14s} {rec.get(k)}" for k in keys]
        rate = [round(float(c["params"][1]), 2) for c in rec["components"]]
        lines.append(f"{'rate [mm/yr]':14s} {rate}")
        lines.append(f"{'record_version':14s} {rec.get('record_version')}")
        return "\n".join(lines)

    def _command(self, plan: Any, extra: list[str], terms: list[str]) -> str:
        from gps_plot.detrend_picker import render_command

        flags = list(extra)
        if self.max_gap_years is not None:
            flags += ["--max-gap-years", str(self.max_gap_years)]
        if self.uncert != 10.0:
            flags += ["--uncert", str(self.uncert)]
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
    p.add_argument("--uncert", type=float, default=10.0)
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
    defaults = FitDefaults()
    if args.max_gap_years is not None:
        defaults = dataclasses.replace(defaults, max_gap_years=args.max_gap_years)
    settings = resolve_fit_settings(sta, catalog, defaults, catalog_source=source)

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
    )
    window.win.show()
    return int(app.exec())


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
