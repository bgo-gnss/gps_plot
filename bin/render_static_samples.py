#!/usr/bin/env python3
"""Render static-plot-modernize sign-off samples (PDF + PNG + EPS).

Produces, into a clearly named output directory, the SAME standard
three-component plot in all three formats via the modernized native
export path (``gps_plot.timesmatplt``), for:

- real ``.NEU`` stations (plate-fixed products; their last data point is
  in the past, so the "Last datapoint" header is semantically RED), and
- one synthetic station whose series ends yesterday (header GREEN).

For comparison it also renders a legacy-pipeline reference for selected
stations: the old single-string ``\\textcolor`` title saved as EPS
(dvips keeps the color) and rasterized with ImageMagick ``convert``,
exactly like the pre-modernization production path.

Run from the gps_plot repo::

    uv run python bin/render_static_samples.py \\
        --out samples_static_modernize \\
        --neu ~/Downloads/AFST-plate.NEU --neu ~/Downloads/ISAF-plate.NEU
"""

from __future__ import annotations

import argparse
import datetime
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: samples are files, never windows

import matplotlib.pyplot as plt
import numpy as np

from gps_plot import timesmatplt as tplt

FORMATS = ("eps", "pdf", "png")


def load_neu_datetimes(
    path: Path,
) -> tuple[list[datetime.datetime], np.ndarray, np.ndarray]:
    """Parse a published ``.NEU`` product (date time dN DN dE DE dU DU, mm).

    Same format contract as ``gps_plot.dev_viz.load_neu`` (aflogun-verified),
    but keeping datetimes since the static plots are datetime-based.
    """
    times: list[datetime.datetime] = []
    rows: list[list[float]] = []
    for line in path.read_text().splitlines():
        stripped = line.lstrip()
        if not stripped[:1].isdigit():
            continue
        fields = stripped.split()
        if len(fields) < 8:
            continue
        times.append(
            datetime.datetime.strptime(f"{fields[0]} {fields[1]}", "%Y/%m/%d %H:%M:%S")
        )
        rows.append([float(v) for v in fields[2:8]])
    if not rows:
        raise ValueError(f"no data rows parsed from {path}")
    order = np.argsort(np.array([t.timestamp() for t in times]))
    times = [times[i] for i in order]
    data = np.asarray(rows, dtype=np.float64)[order]
    y = data[:, (0, 2, 4)].T.copy()  # north, east, up [mm]
    dy = data[:, (1, 3, 5)].T.copy()
    return times, y, dy


def synthetic_green_station(
    n_days: int = 90, seed: int = 7
) -> tuple[list[datetime.datetime], np.ndarray, np.ndarray]:
    """Synthetic daily series ending YESTERDAY noon => green header."""
    rng = np.random.default_rng(seed)
    yesterday = datetime.datetime.now().replace(
        hour=12, minute=0, second=0, microsecond=0
    ) - datetime.timedelta(days=1)
    x = [yesterday - datetime.timedelta(days=d) for d in range(n_days - 1, -1, -1)]
    trend = np.outer([12.0, -8.0, 4.0], np.linspace(0.0, 1.0, n_days))
    y = trend + np.cumsum(rng.normal(0.0, 1.2, size=(3, n_days)), axis=1)
    dy = np.abs(rng.normal(2.0, 0.5, size=(3, n_days))) + 1.0
    return x, y, dy


def last_window(
    x: list[datetime.datetime], y: np.ndarray, dy: np.ndarray, days: int
) -> tuple[list[datetime.datetime], np.ndarray, np.ndarray]:
    """Slice the trailing ``days`` days of the series (production 90d/year)."""
    cutoff = x[-1] - datetime.timedelta(days=days)
    keep = [i for i, t in enumerate(x) if t >= cutoff]
    return [x[i] for i in keep], y[:, keep], dy[:, keep]


def render_modern(
    sta: str,
    x: list[datetime.datetime],
    y: np.ndarray,
    dy: np.ndarray,
    base: Path,
    ref: str = "PLATE",
) -> tuple[float, float, float]:
    """Modern path: reusable figure -> one native savefig per format."""
    title = tplt.make_title(sta, x[-1], ref=ref)
    fig = tplt.stdTimesPlot(x, y, dy, Title=title, fig=tplt._reusable_figure())
    tplt.inpLogo(fig)  # parity with plotTime (empty logo axes)
    tplt.saveFig(str(base), FORMATS, fig)
    return title.status_color


def render_legacy_reference(
    sta: str,
    x: list[datetime.datetime],
    y: np.ndarray,
    dy: np.ndarray,
    base: Path,
    ref: str = "PLATE",
) -> None:
    """Legacy pipeline: \\textcolor title -> EPS (dvips) -> convert -> PNG."""
    tplt.init_plot_style(force=True)  # legacy reset per plot
    title = tplt.makelatexTitle(sta, x[-1], ref=ref)
    fig = tplt.stdTimesPlot(x, y, dy, Title=title)  # fresh pyplot figure
    tplt.inpLogo(fig)
    eps = base.with_suffix(".eps")
    fig.savefig(eps, bbox_inches="tight")
    plt.close(fig)

    convert = shutil.which("convert")
    if convert is None:
        print(f"  [legacy] ImageMagick convert missing; EPS only: {eps}")
        return
    tmp = base.parent / "tmp-legacy.png"
    subprocess.run([convert, "-density", "90", str(eps), str(tmp)], check=True)
    subprocess.run(
        [convert, "-trim", str(tmp), str(base.with_suffix(".png"))], check=True
    )
    tmp.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("samples_static_modernize"),
        help="output directory (created if missing)",
    )
    parser.add_argument(
        "--neu",
        type=Path,
        action="append",
        default=None,
        help="real .NEU product file(s); repeatable",
    )
    parser.add_argument(
        "--no-legacy",
        action="store_true",
        help="skip the legacy EPS+convert reference renders",
    )
    args = parser.parse_args(argv)

    neu_files = args.neu or [
        Path.home() / "Downloads" / "AFST-plate.NEU",
        Path.home() / "Downloads" / "ISAF-plate.NEU",
    ]
    out: Path = args.out
    legacy_dir = out / "legacy-reference"
    out.mkdir(parents=True, exist_ok=True)
    if not args.no_legacy:
        legacy_dir.mkdir(parents=True, exist_ok=True)

    index: list[str] = []

    # --- real stations (red headers: last data point is in the past) -----
    windows = [("90d", 91), ("year", 366)]
    for i, neu in enumerate(neu_files):
        if not neu.exists() or neu.stat().st_size == 0:
            print(f"skipping {neu} (missing or empty)")
            continue
        sta = neu.stem.split("-")[0]
        x, y, dy = load_neu_datetimes(neu)
        wname, wdays = windows[min(i, len(windows) - 1)]
        xs, ys, dys = last_window(x, y, dy, wdays)
        base = out / f"{sta}-plate-{wname}"
        color = render_modern(sta, xs, ys, dys, base)
        status = "GREEN" if color == tplt.STATUS_CURRENT_COLOR else "RED"
        print(
            f"{sta}: modern {wname} window, header {status} -> {base}.{{eps,pdf,png}}"
        )
        index.append(f"{base.name}: real data ({neu}), window {wname}, header {status}")
        if not args.no_legacy:
            render_legacy_reference(
                sta, xs, ys, dys, legacy_dir / f"{sta}-plate-{wname}-legacy"
            )
            index.append(
                f"legacy-reference/{sta}-plate-{wname}-legacy: old EPS+convert path"
            )

    # --- synthetic green station (series ends yesterday) -----------------
    x, y, dy = synthetic_green_station()
    base = out / "SYNT-plate-90d"
    color = render_modern("SYNT", x, y, dy, base)
    status = "GREEN" if color == tplt.STATUS_CURRENT_COLOR else "RED"
    print(f"SYNT: synthetic, header {status} -> {base}.{{eps,pdf,png}}")
    index.append(f"{base.name}: SYNTHETIC series ending yesterday, header {status}")
    if not args.no_legacy:
        render_legacy_reference("SYNT", x, y, dy, legacy_dir / "SYNT-plate-90d-legacy")
        index.append("legacy-reference/SYNT-plate-90d-legacy: old EPS+convert path")

    (out / "INDEX.txt").write_text(
        "static-plot-modernize sign-off samples\n"
        f"generated: {datetime.datetime.now():%Y-%m-%d %H:%M}\n\n"
        + "\n".join(index)
        + "\n\nNotes:\n"
        "- station full names fall back to the marker (gps_parser not\n"
        "  installed in this env); production shows the configured name.\n"
        "- real .NEU products are already plate-fixed; ref label 'PLATE'.\n"
        "- PNG at 90 dpi to match the legacy convert -density 90 output.\n"
    )
    print(f"index written: {out / 'INDEX.txt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
