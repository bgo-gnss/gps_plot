"""Development visualization for the ``gps_analysis`` lane (thread C / L5).

Renders, for one station and one component (north/east/up), the analysis
products of the sibling leaf library ``gps_analysis`` as they develop
(``PLAN-analysis-lane.md`` §2 thread C, §4 task L5):

1. **Trajectory fit** — observed series with the fitted
   :func:`gps_analysis.models.lineperiodic` trajectory overlaid
   (fit by :func:`gps_analysis.fitting.fit_components`), annotated with the
   WLS secular velocity v̂ ± σ_v from
   :func:`gps_analysis.velocity.estimate_velocity` and, when north+east are
   present, the horizontal magnitude/azimuth (±σ) derived on the same result.
2. **Detrended residuals** — ``r = y − f(t; p̂)`` via
   :func:`gps_analysis.fitting.remove_trend`.
3. **Velocity break points** — the GBIS4TS optimal one/two-break
   piecewise-linear model (:func:`gps_analysis.transient.detect_breakpoints`
   → :func:`gps_analysis.transient.bpd1_forward` /
   :func:`gps_analysis.transient.bpd2_forward`) overlaid on the break-input
   series (zero-referenced raw by default — see :func:`_break_input_series`
   for the modes and their measured behaviour), with each break epoch t_b
   marked and its rate change g [mm/yr] labelled.

All math lives in ``gps_analysis`` (leaf: numpy/scipy/gtimes); this module
only calls it and draws — no equations are re-derived here
(``gps_analysis/docs/MATH_STANDARDS.md``).

Data sources (parameter, so real data slots in later):

- :func:`load_neu` — a published ``.NEU`` product file
  (``date time dN DN dE DE dU DU`` rows, mm; the format the aflogun v0
  loader verified against the live CDN 2026-07-08).
- :func:`synthetic_station` — BPD1/BPD2 trajectory + seasonal + white noise,
  so the viz is runnable end-to-end without station data.

Run it::

    gps-analysis-devviz --synthetic --component east --out /tmp/devviz.png
    gps-analysis-devviz --neu /path/STA.NEU --station STA --component north \\
        --out /tmp/sta_north.png

This module requires the ``dev`` dependency group
(``uv sync`` in ``gps_plot/`` installs the sibling ``gps_analysis``
editable — see ``pyproject.toml``).
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from collections.abc import Sequence
from pathlib import Path

import matplotlib
import numpy as np
from gtimes.timefunc import TimefromYearf, dTimetoYearf
from numpy.typing import NDArray

from gps_analysis.baseline import estimate_offset
from gps_analysis.fitting import fit_components, remove_trend
from gps_analysis.models import TrajectoryParams, lineperiodic, periodic
from gps_analysis.transient import (
    BPD1Params,
    BPD2Params,
    InversionResult,
    bpd1_forward,
    bpd2_forward,
    detect_breakpoints,
)
from gps_analysis.velocity import VelocityEstimate, estimate_velocity

FloatArray = NDArray[np.float64]

COMPONENTS: tuple[str, str, str] = ("north", "east", "up")

#: Colors per panel role (matplotlib default cycle members, kept explicit
#: so the three panels read consistently).
_C_OBS = "0.55"
_C_FIT = "tab:red"
_C_RES = "tab:blue"
_C_BPD = "tab:orange"
_C_BREAK = "tab:green"


# --------------------------------------------------------------------------
# Data containers + sources
# --------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class StationSeries:
    """One station's N/E/U displacement series in dev-viz form.

    Attributes:
        station: Station marker (e.g. ``"REYK"`` or ``"SYNTH"``).
        t: Epochs, shape (N,), fractional years (``yearf``) [yr], ascending.
        y: Displacements, shape (3, N) [mm], rows in ``components`` order.
        sigma: 1-σ observation uncertainties, shape (3, N) [mm], or None.
        components: Row labels, default ``("north", "east", "up")``.
        true_breaks: Synthetic-data provenance only — the BPD parameters
            each component was generated with (None for real data).
    """

    station: str
    t: FloatArray
    y: FloatArray
    sigma: FloatArray | None
    components: tuple[str, str, str] = COMPONENTS
    true_breaks: tuple[BPD1Params | BPD2Params, ...] | None = None

    def component_index(self, component: str) -> int:
        """Row index of ``component`` (case-insensitive)."""
        labels = tuple(c.lower() for c in self.components)
        try:
            return labels.index(component.lower())
        except ValueError:
            raise ValueError(
                f"unknown component {component!r}; have {self.components}"
            ) from None


def synthetic_station(
    *,
    seed: int = 0,
    n_days: int = 1096,
    t0: float = 2020.0,
    n_breaks: int = 1,
    wn_amp: float = 2.0,
    station: str = "SYNTH",
) -> StationSeries:
    """Generate a synthetic N/E/U series with a velocity break.

    Composition (all forward models from ``gps_analysis`` — no new math):
    per component, ``y = bpd*_forward(params, t) + periodic(t, a, b, c, d)
    + ε``, ε ~ N(0, wn_amp²) white noise. Daily sampling (Δt = 1/365.25 yr).

    Args:
        seed: RNG seed (numpy ``default_rng``) — reproducible series.
        n_days: Number of daily epochs N.
        t0: First epoch [yr, fractional year].
        n_breaks: 1 (BPD1) or 2 (BPD2) velocity breaks per component.
        wn_amp: White-noise standard deviation [mm].
        station: Marker stored on the result.

    Returns:
        :class:`StationSeries` with ``sigma`` filled at ``wn_amp`` and the
        true break parameters kept on ``true_breaks`` for overlay checks.
    """
    if n_breaks not in (1, 2):
        raise ValueError(f"n_breaks must be 1 or 2, got {n_breaks}")
    rng = np.random.default_rng(seed)
    t = t0 + np.arange(n_days, dtype=np.float64) / 365.25
    t_b1 = t0 + 0.55 * (t[-1] - t0)  # break past mid-series
    t_b2 = t0 + 0.80 * (t[-1] - t0)

    params: list[BPD1Params | BPD2Params] = []
    # (intercept mm, secular rate mm/yr, rate change(s) mm/yr) per component.
    if n_breaks == 1:
        params = [
            BPD1Params(2.0, 12.0, 18.0, t_b1, kappa=0.0, amp=0.0),
            BPD1Params(-3.0, -8.0, 25.0, t_b1, kappa=0.0, amp=0.0),
            BPD1Params(0.0, 3.0, -30.0, t_b1, kappa=0.0, amp=0.0),
        ]
    else:
        params = [
            BPD2Params(2.0, 12.0, 18.0, t_b1, -12.0, t_b2, kappa=0.0, amp=0.0),
            BPD2Params(-3.0, -8.0, 25.0, t_b1, -18.0, t_b2, kappa=0.0, amp=0.0),
            BPD2Params(0.0, 3.0, -30.0, t_b1, 22.0, t_b2, kappa=0.0, amp=0.0),
        ]
    # Annual/semiannual amplitudes [mm] per component (cos_a, sin_a, cos_sa, sin_sa).
    seasonal = ((2.0, 1.0, 0.5, 0.3), (1.5, -0.8, 0.4, -0.2), (4.0, 2.5, 1.0, 0.5))

    rows = []
    for p, amps in zip(params, seasonal, strict=True):
        trend = bpd1_forward(p, t) if isinstance(p, BPD1Params) else bpd2_forward(p, t)
        noise = rng.normal(0.0, wn_amp, size=t.size)
        rows.append(trend + periodic(t, *amps) + noise)
    y = np.vstack(rows)
    sigma = np.full_like(y, wn_amp)
    return StationSeries(
        station=station, t=t, y=y, sigma=sigma, true_breaks=tuple(params)
    )


def load_neu(path: str | Path, *, station: str | None = None) -> StationSeries:
    """Load one published ``.NEU`` product file into a :class:`StationSeries`.

    Format (verified by the aflogun v0 loader against the live CDN
    2026-07-08): whitespace-separated ``date time dN DN dE DE dU DU`` rows in
    mm, time format ``%Y/%m/%d %H:%M:%S``, ``#`` comments and stray header
    lines skipped defensively. Epochs are converted to fractional years with
    :func:`gtimes.timefunc.dTimetoYearf` (the lane's ``yearf`` convention).

    Args:
        path: ``.NEU`` file path.
        station: Marker; defaults to the file stem.

    Returns:
        :class:`StationSeries` with rows (north, east, up) [mm].
    """
    import datetime as _dt

    path = Path(path)
    t_list: list[float] = []
    rows: list[list[float]] = []
    for line in path.read_text().splitlines():
        stripped = line.lstrip()
        if not stripped[:1].isdigit():  # headers, comments, junk lines
            continue
        fields = stripped.split()
        if len(fields) < 8:
            continue
        stamp = _dt.datetime.strptime(f"{fields[0]} {fields[1]}", "%Y/%m/%d %H:%M:%S")
        t_list.append(float(dTimetoYearf(stamp)))
        rows.append([float(v) for v in fields[2:8]])
    if not rows:
        raise ValueError(f"no data rows parsed from {path}")
    data = np.asarray(rows, dtype=np.float64)  # columns dN DN dE DE dU DU
    order = np.argsort(np.asarray(t_list, dtype=np.float64))
    t = np.asarray(t_list, dtype=np.float64)[order]
    data = data[order]
    return StationSeries(
        station=station or path.stem,
        t=t,
        y=data[:, (0, 2, 4)].T.copy(),
        sigma=data[:, (1, 3, 5)].T.copy(),
    )


# --------------------------------------------------------------------------
# Analysis (thin orchestration over gps_analysis)
# --------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ComponentAnalysis:
    """Everything the dev-viz panels need for one station/component.

    Attributes:
        series: The input data.
        component: Plotted component label.
        fits: Per-component :class:`~gps_analysis.models.TrajectoryParams`
            of the :func:`~gps_analysis.models.lineperiodic` fit (absolute
            ``yearf`` epochs), row order of ``series.y``.
        residuals: Detrended series, shape (3, N) [mm]
            (:func:`~gps_analysis.fitting.remove_trend`).
        velocity: :class:`~gps_analysis.velocity.VelocityEstimate` for all
            components (carries rate ± σ and horizontal magnitude/azimuth).
        break_result: MCMC :class:`~gps_analysis.transient.InversionResult`
            for the plotted component's break-input series, or None when
            detection was skipped.
        break_params: BPD parameters used for the overlay — the MCMC optimum
            when detection ran, otherwise caller-provided (e.g. synthetic
            truth); None disables the break panel overlay.
        break_series: The series break detection ran on / the overlay is
            drawn against (see ``break_input`` of :func:`analyze_station`),
            shape (N,) [mm].
        break_input: The input mode that produced ``break_series``
            (``"raw"``/``"seasonal_removed"``/``"residuals"``).
    """

    series: StationSeries
    component: str
    fits: tuple[TrajectoryParams, ...]
    residuals: FloatArray
    velocity: VelocityEstimate
    break_result: InversionResult | None
    break_params: BPD1Params | BPD2Params | None
    break_series: FloatArray
    break_input: str


def _optimal_to_params(result: InversionResult) -> BPD1Params | BPD2Params:
    """Wrap ``InversionResult.optimal`` (MATLAB order) in its typed params."""
    values = tuple(float(v) for v in result.optimal)
    if result.model == "BPD1":
        return BPD1Params(*values)
    return BPD2Params(*values)


#: Reference window for zero-referencing the break-input series [yr]
#: (~18 days — enough samples for a stable weighted mean at the start).
_ZERO_REF_WINDOW_YR = 0.05

BREAK_INPUTS = ("raw", "seasonal_removed", "residuals")


def _break_input_series(
    series: StationSeries,
    idx: int,
    fit: TrajectoryParams,
    residuals: FloatArray,
    mode: str,
) -> FloatArray:
    """Build the series break detection runs on (all math from gps_analysis).

    Modes (measured behaviour on the built-in synthetic, N = 365,
    2000-run chain — dev-indicative, see :func:`analyze_station`):

    - ``"raw"`` (default): the observed component zero-referenced by the
      weighted mean of its first ~18 days
      (:func:`gps_analysis.baseline.estimate_offset`). BPD estimates its own
      trend + break; unmodeled seasonal signal is soaked up by the power-law
      noise term. Break epoch recovers well; the BPD rates absorb part of
      the seasonal signal and are biased on short (< 2 yr) windows.
    - ``"seasonal_removed"``: raw minus the fitted
      :func:`~gps_analysis.models.periodic` seasonal part of the
      lineperiodic fit. Cleaner rates on long windows, but on short windows
      the seasonal estimate is itself ramp-contaminated (trend/seasonal/
      break degeneracy — Blewitt & Lavallée 2002) and drags the break off.
    - ``"residuals"``: the lineperiodic-detrended residuals. **Known-bad
      for detection** — the single-rate fit absorbs the ramp, the kink is
      distorted and the optimum tends to a boundary break with unidentified
      Δv. Kept so the failure mode stays visible in the dev-viz.
    """
    if mode == "residuals":
        return np.asarray(residuals[idx], dtype=np.float64)
    y = series.y[idx]
    if mode == "seasonal_removed":
        y = y - periodic(series.t, *fit.params[2:6])
    elif mode != "raw":
        raise ValueError(f"break_input must be one of {BREAK_INPUTS}, got {mode!r}")
    sig = None if series.sigma is None else series.sigma[idx]
    offset = estimate_offset(
        series.t,
        y,
        sig,
        start=float(series.t[0]),
        end=float(series.t[0]) + _ZERO_REF_WINDOW_YR,
    )
    return np.asarray(y - offset, dtype=np.float64)


def analyze_station(
    series: StationSeries,
    component: str,
    *,
    detect: bool = True,
    n_breaks: int = 1,
    n_runs: int = 2_000,
    t_runs: int = 100,
    seed: int | None = 0,
    wn_amp: float | None = None,
    break_params: BPD1Params | BPD2Params | None = None,
    break_input: str = "raw",
) -> ComponentAnalysis:
    """Run the gps_analysis chain the dev-viz panels display.

    Chain: :func:`~gps_analysis.fitting.fit_components` (lineperiodic, all
    components) → :func:`~gps_analysis.fitting.remove_trend` →
    :func:`~gps_analysis.velocity.estimate_velocity` (rates ± σ +
    horizontal products) → optionally
    :func:`~gps_analysis.transient.detect_breakpoints` on the plotted
    component's break-input series (zero-referenced so the intercept fits
    the BPD ±5 mm prior — see :func:`_break_input_series` for the modes
    and their measured behaviour).

    Args:
        series: Input data (real or synthetic).
        component: Component to run break detection on / to plot.
        detect: Run the GBIS4TS MCMC (dominant cost). If False,
            ``break_params`` (or synthetic truth, when present) drives the
            overlay instead.
        n_breaks: 1 (BPD1) or 2 (BPD2).
        n_runs: Kept MCMC iterations — dev default 2 000 is far below the
            production 1e6; the optimum is indicative, the posterior is not
            converged. Cost note (pre-H3, plan §4): every noise-parameter
            perturbation (~⅓ of iterations) rebuilds + factorizes the dense
            N×N covariance — measured ~10 ms at N = 365, ~0.3 s at N = 730,
            ~0.5 s at N = 1096 — so wall time ≈ ``n_runs/3`` × that. Keep
            dev series ≲ 1–2 yr or chains short until the Toeplitz/Cholesky
            speedup (H3) lands.
        t_runs: Annealing iterations per temperature (16 temperatures; must
            satisfy ``16 * t_runs < n_runs``).
        seed: MCMC RNG seed.
        wn_amp: Fixed white-noise amplitude [mm] for the likelihood; default
            is the median observation σ of the component (or the residual
            std when the series carries no σ) — a dev-viz heuristic, not the
            GBIS4TS noise pre-processing.
        break_params: Overlay parameters when ``detect=False``.
        break_input: What detection runs on — ``"raw"`` (default),
            ``"seasonal_removed"``, or ``"residuals"``
            (:func:`_break_input_series`).

    Returns:
        :class:`ComponentAnalysis` for the requested component.
    """
    idx = series.component_index(component)
    fits = tuple(
        fit_components(
            lineperiodic,
            series.t,
            series.y,
            sigma=series.sigma,
            names=series.components,
        )
    )
    residuals = np.asarray(
        remove_trend(lineperiodic, series.t, series.y, fits), dtype=np.float64
    )
    velocity = estimate_velocity(
        series.t,
        series.y,
        series.sigma,
        model="lineperiodic",
        names=series.components,
    )

    break_series = _break_input_series(series, idx, fits[idx], residuals, break_input)
    break_result: InversionResult | None = None
    if detect:
        if wn_amp is None:
            if series.sigma is not None:
                wn_amp = float(np.median(series.sigma[idx]))
            else:
                wn_amp = float(np.std(residuals[idx]))
        break_result = detect_breakpoints(
            series.t,
            break_series,
            wn_amp,
            n_breaks=n_breaks,
            n_runs=n_runs,
            t_runs=t_runs,
            seed=seed,
        )
        break_params = _optimal_to_params(break_result)
    elif break_params is None and series.true_breaks is not None:
        break_params = series.true_breaks[idx]

    return ComponentAnalysis(
        series=series,
        component=series.components[idx],
        fits=fits,
        residuals=residuals,
        velocity=velocity,
        break_result=break_result,
        break_params=break_params,
        break_series=break_series,
        break_input=break_input,
    )


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------


def _break_epochs(params: BPD1Params | BPD2Params) -> tuple[tuple[float, float], ...]:
    """(break epoch t_b [yr], rate change g [mm/yr]) pairs of a BPD model."""
    if isinstance(params, BPD1Params):
        return ((params.breakpoint, params.trend_change),)
    return (
        (params.breakpoint1, params.trend_change1),
        (params.breakpoint2, params.trend_change2),
    )


def _yearf_label(t_b: float) -> str:
    """Break epoch as ``yearf (yyyy-mm-dd)`` via gtimes."""
    day = TimefromYearf(t_b, String="%Y-%m-%d")
    return f"{t_b:.3f} ({day})"


def plot_component(analysis: ComponentAnalysis) -> "matplotlib.figure.Figure":
    """Render the three dev-viz panels for one station/component.

    Panels (shared time axis): observed + trajectory fit with the velocity
    annotation; detrended residuals; residuals + optimal BPD break model
    with break epochs marked. Returns the figure — the caller saves or
    shows it.
    """
    import matplotlib.pyplot as plt

    series = analysis.series
    idx = series.component_index(analysis.component)
    t = series.t
    y = series.y[idx]
    res = analysis.residuals[idx]
    fit = analysis.fits[idx]
    vel = analysis.velocity

    fig, axes = plt.subplots(
        3, 1, figsize=(11, 8.5), sharex=True, constrained_layout=True
    )
    ax_fit, ax_res, ax_bpd = axes

    # Panel 1 — observed + lineperiodic trajectory + velocity annotation.
    if series.sigma is not None:
        ax_fit.errorbar(
            t,
            y,
            yerr=series.sigma[idx],
            fmt=".",
            ms=2.5,
            color=_C_OBS,
            ecolor="0.85",
            elinewidth=0.5,
            label="observed",
        )
    else:
        ax_fit.plot(t, y, ".", ms=2.5, color=_C_OBS, label="observed")
    t_dense = np.linspace(float(t[0]), float(t[-1]), 4 * t.size)
    ax_fit.plot(
        t_dense,
        lineperiodic(t_dense, *fit.params),
        "-",
        lw=1.5,
        color=_C_FIT,
        label="lineperiodic fit",
    )
    rate, sig = float(vel.rates[idx]), float(vel.sigmas[idx])
    note = [f"v = {rate:+.2f} ± {sig:.2f} mm/yr ({analysis.component}, WLS)"]
    if vel.magnitude is not None and vel.azimuth is not None:
        mag_s = f" ± {vel.magnitude_sigma:.2f}" if vel.magnitude_sigma else ""
        azi_s = f" ± {vel.azimuth_sigma:.1f}" if vel.azimuth_sigma else ""
        note.append(
            f"|v_h| = {vel.magnitude:.2f}{mag_s} mm/yr, "
            f"az = {vel.azimuth:.1f}{azi_s}° (CW from N)"
        )
    ax_fit.annotate(
        "\n".join(note),
        xy=(0.02, 0.96),
        xycoords="axes fraction",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round", "fc": "white", "ec": "0.7", "alpha": 0.9},
    )
    ax_fit.set_ylabel(f"{analysis.component} [mm]")
    ax_fit.legend(loc="lower right", fontsize=8)
    ax_fit.set_title(
        f"{series.station} — {analysis.component}: trajectory fit + WLS velocity"
    )

    # Panel 2 — detrended residuals.
    ax_res.plot(t, res, ".", ms=2.5, color=_C_RES)
    ax_res.axhline(0.0, color="0.3", lw=0.8)
    ax_res.set_ylabel("residual [mm]")
    ax_res.set_title("detrended residuals (lineperiodic removed)")

    # Panel 3 — break model on the break-input series.
    ax_bpd.plot(t, analysis.break_series, ".", ms=2.5, color=_C_RES, alpha=0.5)
    ax_bpd.axhline(0.0, color="0.3", lw=0.8)
    params = analysis.break_params
    if params is not None:
        forward = (
            bpd1_forward(params, t)
            if isinstance(params, BPD1Params)
            else bpd2_forward(params, t)
        )
        if analysis.break_result is not None:
            source = f"GBIS4TS optimum ({analysis.break_result.model})"
        else:
            source = "provided parameters"
            # Re-anchor to the same early-window reference level as the
            # break-input series (provided params were not fitted to it).
            forward = forward - estimate_offset(
                t,
                forward,
                start=float(t[0]),
                end=float(t[0]) + _ZERO_REF_WINDOW_YR,
            )
        ax_bpd.plot(t, forward, "-", lw=1.8, color=_C_BPD, label=source)
        for t_b, g in _break_epochs(params):
            ax_bpd.axvline(t_b, color=_C_BREAK, lw=1.2, ls="--")
            ax_bpd.annotate(
                f"t_b = {_yearf_label(t_b)}\nΔv = {g:+.1f} mm/yr",
                xy=(t_b, 0.02),
                xycoords=("data", "axes fraction"),
                fontsize=8,
                rotation=90,
                va="bottom",
                ha="right",
                color=_C_BREAK,
            )
        ax_bpd.legend(loc="upper left", fontsize=8)
        title = f"velocity break points (BPD on {analysis.break_input} series)"
    else:
        title = "velocity break points — detection skipped, no parameters"
    ax_bpd.set_title(title)
    ax_bpd.set_ylabel(f"{analysis.break_input} [mm]")
    ax_bpd.set_xlabel("time [fractional year]")
    return fig


def render_dev_viz(
    series: StationSeries,
    component: str = "east",
    out_path: str | Path | None = None,
    **analyze_kwargs: object,
) -> tuple["matplotlib.figure.Figure", ComponentAnalysis]:
    """One-call entry: analyze + plot + optionally save.

    Args:
        series: Data from :func:`load_neu` / :func:`synthetic_station` (or
            any :class:`StationSeries` — the data source is a parameter).
        component: Component to display (``north``/``east``/``up``).
        out_path: When given, the figure is saved there (format by suffix).
        **analyze_kwargs: Forwarded to :func:`analyze_station`.

    Returns:
        ``(figure, analysis)``.
    """
    analysis = analyze_station(series, component, **analyze_kwargs)  # type: ignore[arg-type]
    fig = plot_component(analysis)
    if out_path is not None:
        fig.savefig(out_path, dpi=150)
    return fig, analysis


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gps-analysis-devviz",
        description=(
            "Development visualization of gps_analysis outputs: trajectory "
            "fit, detrended residuals, velocity break points, velocity."
        ),
    )
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--neu", type=Path, help=".NEU product file to plot (real data)")
    src.add_argument(
        "--synthetic",
        action="store_true",
        help="use the built-in synthetic series (default when no --neu)",
    )
    parser.add_argument("--station", help="station marker (default: file stem)")
    parser.add_argument(
        "--component",
        choices=COMPONENTS,
        default="east",
        help="component to display (default: east)",
    )
    parser.add_argument(
        "--breaks",
        type=int,
        choices=(1, 2),
        default=1,
        help="number of velocity breaks to fit (BPD1/BPD2, default 1)",
    )
    parser.add_argument(
        "--no-detect",
        action="store_true",
        help="skip the MCMC; overlay synthetic truth / nothing instead",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=2_000,
        help=(
            "kept MCMC iterations (dev default 2000 — quick, indicative; "
            "production GBIS4TS uses 1e6). Pre-H3 cost: ~runs/3 dense "
            "N-by-N factorizations (~10 ms at N=365, ~0.3 s at N=730)"
        ),
    )
    parser.add_argument(
        "--t-runs",
        type=int,
        default=100,
        help="annealing iterations per temperature (default 100; 16*t_runs < runs)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=365,
        help=(
            "synthetic series length in daily epochs (default 365; the MCMC "
            "cost grows steeply past ~1.5 yr until the H3 speedup lands)"
        ),
    )
    parser.add_argument(
        "--break-input",
        choices=BREAK_INPUTS,
        default="raw",
        help=(
            "series break detection runs on: raw (zero-referenced observed, "
            "default), seasonal_removed, or residuals (known-bad — kept to "
            "keep the failure mode visible)"
        ),
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed")
    parser.add_argument(
        "--wn-amp",
        type=float,
        help="fixed white-noise amplitude [mm] (default: median sigma)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("devviz.png"),
        help="output figure path (default: ./devviz.png)",
    )
    parser.add_argument(
        "--show", action="store_true", help="open an interactive window too"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Console entry point (``gps-analysis-devviz``)."""
    args = _build_parser().parse_args(argv)
    if not args.show:
        matplotlib.use("Agg")

    if args.neu is not None:
        series = load_neu(args.neu, station=args.station)
    else:
        series = synthetic_station(
            seed=args.seed,
            n_days=args.days,
            n_breaks=args.breaks,
            station=args.station or "SYNTH",
        )

    fig, analysis = render_dev_viz(
        series,
        component=args.component,
        out_path=args.out,
        detect=not args.no_detect,
        n_breaks=args.breaks,
        n_runs=args.runs,
        t_runs=args.t_runs,
        seed=args.seed,
        wn_amp=args.wn_amp,
        break_input=args.break_input,
    )
    idx = series.component_index(args.component)
    vel = analysis.velocity
    print(f"wrote {args.out}")
    print(
        f"{series.station} {args.component}: "
        f"v = {vel.rates[idx]:+.2f} ± {vel.sigmas[idx]:.2f} mm/yr (WLS)"
    )
    if vel.magnitude is not None:
        print(
            f"horizontal: |v| = {vel.magnitude:.2f} mm/yr, "
            f"azimuth = {vel.azimuth:.1f} deg CW from N"
        )
    if analysis.break_params is not None:
        for t_b, g in _break_epochs(analysis.break_params):
            print(f"break: t_b = {_yearf_label(t_b)}, dv = {g:+.2f} mm/yr")
    if args.show:
        import matplotlib.pyplot as plt

        plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
