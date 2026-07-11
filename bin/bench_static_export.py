#!/usr/bin/env python3
"""Wall-clock benchmark for the static-plot-modernize export path.

Times figure build + export over N synthetic stations for four scenarios:

- ``legacy``      -- pre-modernization production path: per-plot rcParams
  reset, a fresh pyplot figure per station, EPS via ``bbox_inches="tight"``,
  then ImageMagick ``convert -density 90`` + ``-trim`` to PNG.
- ``modern3``     -- modernized path: style once, reused Figure, one native
  ``savefig`` each for EPS + PDF + PNG (shared precomputed tight bbox).
- ``modern_png``  -- modernized path, PNG only (the brunnur product).
- ``notex_png``   -- PNG only with ``text.usetex=False`` and a plain
  mathtext-free title: the Path-A cost control that isolates the per-plot
  TeX cost (look differs -- it is a timing control, not a product).

All stations share identical data (identical tick strings) but carry
UNIQUE station names, so per-station TeX work is exactly the per-title
work -- as in production, where tick labels repeat across stations and
each station contributes fresh title strings.  Each scenario runs two
passes: pass 1 is TeX-cold for its titles, pass 2 fully warm; their
difference isolates the LaTeX cost per plot.  A warm-up render outside
the timers pre-caches the shared tick/ylabel strings and the font cache.

Run from the gps_plot repo (foreground)::

    uv run python bin/bench_static_export.py --scenario all --stations 12
"""

from __future__ import annotations

import argparse
import datetime
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from gps_plot import timesmatplt as tplt

FROZEN_CREATED = "(Plot created on Jul 11 2026 12:00 GMT)"


def series(n_days: int = 90) -> tuple[list[datetime.datetime], np.ndarray, np.ndarray]:
    """One fixed series reused by every station (identical tick strings)."""
    end = datetime.datetime.now().replace(
        hour=12, minute=0, second=0, microsecond=0
    ) - datetime.timedelta(days=1)
    x = [end - datetime.timedelta(days=d) for d in range(n_days - 1, -1, -1)]
    rng = np.random.default_rng(0)
    y = np.cumsum(rng.normal(0.0, 1.5, size=(3, n_days)), axis=1)
    dy = np.full((3, n_days), 2.0)
    return x, y, dy


def modern_title(sta: str, last: datetime.datetime) -> tplt.StationTitle:
    """StationTitle with a frozen created-string (deterministic TeX work)."""
    live = tplt.make_title(sta, last, ref="PLATE")
    return tplt.StationTitle(
        main=live.main,
        status=live.status,
        created=r"\large %s" % FROZEN_CREATED,
        status_color=live.status_color,
    )


def legacy_title(sta: str, last: datetime.datetime) -> list[str]:
    """Legacy \\textcolor two-string title with a frozen created-string."""
    t = modern_title(sta, last)
    dcolor = "green" if t.status_color == tplt.STATUS_CURRENT_COLOR else "red"
    name_str, ref_str = t.main.replace(r"\Huge ", "").split(r" \ \Large ")
    status_str = t.status.replace(r"\LARGE ", "")
    return [
        r"\Huge\textcolor{black}{%s} \  \Large\textcolor{black}{%s}  "
        % (name_str, ref_str),
        r"\LARGE\textcolor{%s}{%s} \  \large\textcolor{black}{%s}  "
        % (dcolor, status_str, FROZEN_CREATED),
    ]


def run_legacy(names: list[str], outdir: Path) -> list[float]:
    """Old production path: reset + fresh figure + EPS tight + convert."""
    convert = shutil.which("convert")
    if convert is None:
        raise SystemExit("legacy scenario needs ImageMagick convert")
    x, y, dy = series()
    times: list[float] = []
    for sta in names:
        t0 = time.perf_counter()
        tplt.init_plot_style(force=True)  # legacy per-plot rcParams reset
        fig = tplt.stdTimesPlot(x, y, dy, Title=legacy_title(sta, x[-1]))
        eps = outdir / f"{sta}.eps"
        fig.savefig(eps, bbox_inches="tight")
        tmp = outdir / f"{sta}-tmp.png"
        subprocess.run([convert, "-density", "90", str(eps), str(tmp)], check=True)
        subprocess.run(
            [convert, "-trim", str(tmp), str(outdir / f"{sta}.png")], check=True
        )
        tmp.unlink(missing_ok=True)
        plt.close(fig)
        times.append(time.perf_counter() - t0)
    return times


def run_modern(names: list[str], outdir: Path, formats: tuple[str, ...]) -> list[float]:
    """Modern path: style once, reused Figure, native multi-format export."""
    tplt.init_plot_style(force=True)
    x, y, dy = series()
    times: list[float] = []
    for sta in names:
        t0 = time.perf_counter()
        fig = tplt.stdTimesPlot(
            x, y, dy, Title=modern_title(sta, x[-1]), fig=tplt._reusable_figure()
        )
        tplt.saveFig(str(outdir / sta), formats, fig)
        times.append(time.perf_counter() - t0)
    return times


def run_notex(names: list[str], outdir: Path) -> list[float]:
    """Path-A cost control: usetex off, plain title, PNG only."""
    tplt.init_plot_style(force=True)
    mpl.rcParams["text.usetex"] = False
    x, y, dy = series()
    times: list[float] = []
    try:
        for sta in names:
            t0 = time.perf_counter()
            title = tplt.StationTitle(
                main=f"{sta} ({sta})   Reference frame: PLATE",
                status="Last datapoint: 10 Jul 2026",
                created=FROZEN_CREATED,
                status_color=tplt.STATUS_CURRENT_COLOR,
            )
            fig = tplt.stdTimesPlot(x, y, dy, Title=title, fig=tplt._reusable_figure())
            tplt.saveFig(str(outdir / sta), ("png",), fig)
            times.append(time.perf_counter() - t0)
    finally:
        mpl.rcParams["text.usetex"] = True
    return times


SCENARIOS = {
    "legacy": lambda names, out: run_legacy(names, out),
    "modern3": lambda names, out: run_modern(names, out, ("eps", "pdf", "png")),
    "modern_png": lambda names, out: run_modern(names, out, ("png",)),
    "notex_png": lambda names, out: run_notex(names, out),
}

#: unique station-name prefixes so every scenario's pass 1 is TeX-cold
SCENARIO_PREFIX = {
    "legacy": "LG",
    "modern3": "M3",
    "modern_png": "MP",
    "notex_png": "NT",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scenario", choices=[*SCENARIOS, "all"], default="all")
    parser.add_argument("--stations", type=int, default=12)
    parser.add_argument(
        "--out", type=Path, default=None, help="output dir (default: temp dir)"
    )
    args = parser.parse_args(argv)

    outdir = args.out or Path(tempfile.mkdtemp(prefix="bench_static_"))
    outdir.mkdir(parents=True, exist_ok=True)

    # warm-up outside the timers: shared tick/ylabel TeX strings, font cache
    x, y, dy = series()
    tplt.init_plot_style(force=True)
    fig = tplt.stdTimesPlot(
        x, y, dy, Title=modern_title("WARM", x[-1]), fig=tplt._reusable_figure()
    )
    tplt.saveFig(str(outdir / "WARM"), ("eps", "pdf", "png"), fig)

    wanted = list(SCENARIOS) if args.scenario == "all" else [args.scenario]
    print(f"stations={args.stations}  out={outdir}")
    print(
        f"{'scenario':<12} {'pass':<6} {'total[s]':>9} {'per-plot[s]':>12} "
        f"{'min':>6} {'max':>6}"
    )
    for name in wanted:
        # identical names in both passes: pass1 is TeX-cold for these titles
        # (fresh names per scenario), pass2 re-renders them fully warm
        names = [f"{SCENARIO_PREFIX[name]}{i:03d}" for i in range(args.stations)]
        for pass_no in (1, 2):
            times = SCENARIOS[name](names, outdir)
            print(
                f"{name:<12} {pass_no:<6} {sum(times):>9.2f} "
                f"{sum(times) / len(times):>12.3f} "
                f"{min(times):>6.2f} {max(times):>6.2f}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
