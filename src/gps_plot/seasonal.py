"""``gpsplot seasonal``: a station's seasonal, folded onto one year.

What the detrend record removes as "periodic" (annual + semiannual), shown
against the data it was fitted to, and against what the station's
neighbours predict for it (the regional seasonal, docs/regional-seasonal.md
in geo_dataread):

- dots: the residual with the stored secular (offset + rate + steps)
  removed and the seasonal left in, folded on ``yearf mod 1`` — over the
  record's own fit window by default (outside it the secular is an
  extrapolation and any unrest swamps the fold); its level is set so that
  the median of (residual − own seasonal) is zero (:func:`align_datum`);
- black: robust bin medians of the dots;
- red: the record's own periodic terms, ±1σ from their covariance;
- blue dashed: the regional prediction from the other stations' estimated
  seasonals (leave-one-out: the station itself is never its own donor),
  ±1σ the leave-one-out RMS of the annual vector.

The periodic basis is cos/sin(2π·yearf) on absolute ``yearf``
(``gps_analysis.models.periodic``), so ``cos(2πt) = cos(2π·frac(t))``:
folding on the fractional year puts the data on the same phase as the
curve with no calendar conversion.

A seasonal-only record (a regional one, no secular of its own) has no
residual to fold: the stored regional curve is drawn alone, never the raw
series in its place.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

COMPONENTS = ("north", "east", "up")
LABELS = ("North", "East", "Up")
#: Julian year, the fractional-year convention of the records
DAYS_PER_YEAR = 365.25
#: curve sampling over one year
GRID_N = 366
#: 1.4826·MAD is σ for Gaussian data
MAD_TO_SIGMA = 1.4826

OWN_COLOR = "#c0392b"
REGIONAL_COLOR = "#1f4e9c"
DOT_COLOR = "0.55"
DONOR_COLOR = "0.6"


# ---------------------------------------------------------------------------
# math (pure numpy)
# ---------------------------------------------------------------------------


def fold(yearf: Any) -> FloatArray:
    """Fractional year in [0, 1): the phase of the periodic basis."""
    out: FloatArray = np.mod(np.asarray(yearf, dtype=float), 1.0)
    return out


def periodic_basis(x: Any) -> FloatArray:
    """(n, 4) design ``[cos 2πx, sin 2πx, cos 4πx, sin 4πx]`` — the order of
    ``PERIODIC_NAMES`` (cos/sin annual, cos/sin semiannual)."""
    w = 2 * np.pi * np.asarray(x, dtype=float)
    out: FloatArray = np.column_stack(
        [np.cos(w), np.sin(w), np.cos(2 * w), np.sin(2 * w)]
    )
    return out


def seasonal_curve(coeffs: Any, x: Any) -> FloatArray:
    """(3, n) seasonal per N/E/U at phases ``x`` from (3, 4) coefficients."""
    out: FloatArray = np.asarray(coeffs, dtype=float) @ periodic_basis(x).T
    return out


def curve_sigma(cov: Any, x: Any) -> FloatArray:
    """(3, n) 1σ of the curve from (3, 4, 4) coefficient covariances:
    ``σ(x)² = b(x)ᵀ C b(x)``."""
    b = periodic_basis(x)
    c = np.asarray(cov, dtype=float)
    var = np.einsum("ni,kij,nj->kn", b, c, b)
    out: FloatArray = np.sqrt(np.clip(var, 0.0, None))
    return out


@dataclass(frozen=True)
class Harmonics:
    """One component's seasonal as amplitudes and the annual peak.

    ``a cos 2πt + b sin 2πt = A cos(2π(t − t_peak))`` with
    ``A = √(a² + b²)`` and ``t_peak = atan2(b, a) / 2π mod 1``.
    """

    annual: float
    peak: float
    semiannual: float


def harmonics(coeffs4: Any) -> Harmonics:
    """Annual amplitude, annual peak (fraction of year) and semiannual
    amplitude of one component's four coefficients."""
    a, b, c, d = (float(v) for v in coeffs4)
    peak = float(np.mod(np.arctan2(b, a) / (2 * np.pi), 1.0))
    return Harmonics(float(np.hypot(a, b)), peak, float(np.hypot(c, d)))


def peak_label(frac: float) -> str:
    """A fraction of the year as a day and month, e.g. ``12 Aug``."""
    day = dt.date(2001, 1, 1) + dt.timedelta(days=frac * DAYS_PER_YEAR)
    return f"{day.day} {day:%b}"


def binned_profile(
    x: Any, y: Any, *, bin_days: float = 15.0, min_count: int = 5
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """(centres, medians, robust standard errors) of ``y`` in phase bins.

    The standard error is 1.4826·MAD/√n; bins with fewer than ``min_count``
    finite values are NaN.
    """
    xs, ys = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    ok = np.isfinite(xs) & np.isfinite(ys)
    xs, ys = xs[ok], ys[ok]
    n_bins = max(int(round(DAYS_PER_YEAR / bin_days)), 1)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    med = np.full(n_bins, np.nan)
    err = np.full(n_bins, np.nan)
    idx = np.clip(np.searchsorted(edges, xs, side="right") - 1, 0, n_bins - 1)
    for i in range(n_bins):
        v = ys[idx == i]
        if v.size >= min_count:
            m = float(np.median(v))
            med[i] = m
            err[i] = MAD_TO_SIGMA * float(np.median(np.abs(v - m))) / np.sqrt(v.size)
    return centres, med, err


def rms_difference(c1: Any, c2: Any) -> FloatArray:
    """(3,) RMS over the year of the difference of two seasonals.

    Exact for the four-term basis: the cross terms vanish over a full
    period, so RMS² = ½ Σ Δcoeff².
    """
    d = np.asarray(c1, dtype=float) - np.asarray(c2, dtype=float)
    out: FloatArray = np.sqrt(0.5 * np.sum(d**2, axis=1))
    return out


def align_datum(t: Any, y: Any, coeffs: Any) -> FloatArray:
    """``y`` shifted per component so that median(y − seasonal(t)) = 0.

    The series' datum (zeroed on its first samples) is not the record's
    offset, so the secular-removed residual sits at an arbitrary level; the
    seasonal is about the shape, so the level is set by the data less the
    record's own seasonal, robustly.
    """
    yy = np.asarray(y, dtype=float)
    r = yy - seasonal_curve(coeffs, fold(t))
    shift = np.array(
        [np.nanmedian(row) if np.isfinite(row).any() else 0.0 for row in r]
    )
    out: FloatArray = yy - shift[:, None]
    return out


# ---------------------------------------------------------------------------
# records
# ---------------------------------------------------------------------------


def periodic_cov(record: Mapping[str, Any]) -> FloatArray | None:
    """(3, 4, 4) covariance of a record's periodic coefficients (by name),
    or None when the record carries no usable covariance."""
    from geo_dataread.regional_seasonal import PERIODIC_NAMES

    names = list(record.get("param_names") or ())
    comps = record.get("components") or ()
    if len(comps) != 3 or not all(n in names for n in PERIODIC_NAMES):
        return None
    n = len(names)
    idx = [names.index(p) for p in PERIODIC_NAMES]
    out = np.empty((3, 4, 4))
    iu = np.triu_indices(n)
    for k, comp in enumerate(comps):
        upper = comp.get("cov_upper")
        if upper is None or len(upper) != len(iu[0]):
            return None
        full = np.zeros((n, n))
        full[iu] = np.asarray(upper, dtype=float)
        full = full + np.triu(full, 1).T
        out[k] = full[np.ix_(idx, idx)]
    return out if np.all(np.isfinite(out)) else None


@dataclass
class SeasonalView:
    """Everything one figure shows (no I/O; see :func:`station_view`)."""

    sta: str
    name: str
    frame: str
    record: Mapping[str, Any]
    seasonal_only: bool
    own: FloatArray | None
    own_cov: FloatArray | None
    regional: FloatArray | None
    regional_why: str | None
    regional_sigma: FloatArray | None
    weights: dict[str, float]
    donors: dict[str, FloatArray]
    #: (t, y (3, N)) secular-removed residual; None for seasonal-only
    data: tuple[FloatArray, FloatArray] | None
    window: tuple[float, float] | None


def regional_prediction(
    sta: str,
    stations: Mapping[str, Mapping[str, Any]],
    coords: Callable[[str], tuple[float, float]],
) -> tuple[Any, dict[str, Any]]:
    """The leave-one-out regional seasonal of ``sta`` and the donor table.

    Returns (``gps_analysis.regional.RegionalSeasonal``, donors as
    code → (lon, lat, coeffs)). ``sta`` is excluded from its own donors.
    """
    from geo_dataread.regional_seasonal import collect_donors
    from gps_analysis.regional import interpolate_seasonal

    donors, _ = collect_donors(stations, coords)
    lon, lat = coords(sta)
    got = interpolate_seasonal((float(lon), float(lat)), donors, exclude=sta)
    return got, donors


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------


def station_view(
    sta: str,
    doc: Mapping[str, Any],
    *,
    frame: str,
    pos_id: str,
    pos_dir: str,
    start: float | None,
    end: float | None,
    clean: bool,
    uncert: float,
    coords: Callable[[str], tuple[float, float]],
    name: str = "",
) -> SeasonalView:
    """Read one station and gather its seasonals.

    Raises:
        ValueError: The station has no record in ``doc``, or its record
            has no periodic terms.
    """
    from geo_dataread.regional_seasonal import periodic_coeffs

    stations = doc.get("stations") or {}
    record = stations.get(sta)
    if not record:
        raise ValueError(f"{sta}: no detrend record in the {frame} document")
    own = periodic_coeffs(record)
    if own is None:
        raise ValueError(f"{sta}: the record has no periodic terms")
    seasonal_only = bool(record.get("seasonal_only"))

    got, donors = regional_prediction(sta, stations, coords)
    rs = doc.get("regional_seasonal") or {}
    sigma = rs.get("sigma_pred")
    if sigma is None and seasonal_only:
        sigma = ((record.get("groups") or {}).get("periodic") or {}).get("sigma_pred")

    data = None
    window = None
    if not seasonal_only:
        w = record.get("window")
        lo = start if start is not None else (float(w[0]) if w else None)
        hi = end if end is not None else (float(w[1]) if w else None)
        data = _read_residual(
            sta, lo, hi, doc=doc, pos_id=pos_id, pos_dir=pos_dir, clean=clean,
            uncert=uncert,
        )  # fmt: skip
        if data[0].size:
            window = (float(np.nanmin(data[0])), float(np.nanmax(data[0])))
            data = (data[0], align_datum(data[0], data[1], own))

    return SeasonalView(
        sta=sta,
        name=name,
        frame=frame,
        record=record,
        seasonal_only=seasonal_only,
        own=None if seasonal_only else own,
        own_cov=None if seasonal_only else periodic_cov(record),
        regional=own if seasonal_only else (None if got.refused else got.coeffs),
        regional_why=got.refused,
        regional_sigma=None if sigma is None else np.asarray(sigma, dtype=float),
        weights=dict(got.weights),
        donors={c: donors[c][2] for c in got.weights},
        data=data,
        window=window,
    )


def _read_residual(
    sta: str,
    start: float | None,
    end: float | None,
    *,
    doc: Mapping[str, Any],
    pos_id: str,
    pos_dir: str,
    clean: bool,
    uncert: float,
) -> tuple[FloatArray, FloatArray]:
    """(t, y (3, N)): plate removed, the stored secular (offset, rate,
    steps) removed, the seasonal left in; outliers NaN when ``clean``."""
    from geo_dataread.gps_views import read_gps_view

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = read_gps_view(
            sta, start, end, view="detrended", terms="secular", clean=clean,
            params=doc, tType="POS", pos_id=pos_id, Dir=pos_dir, uncert=uncert,
        )  # fmt: skip
    cols = [f"{c}_detrended" for c in COMPONENTS]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        reason = df.attrs.get("gps_view", {}).get("degrade_reason") or missing
        raise ValueError(f"{sta}: secular not removed ({reason})")
    t = df["yearf"].to_numpy(dtype=np.float64)
    y = np.vstack([df[c].to_numpy(dtype=np.float64) for c in cols])
    if clean:
        for i, c in enumerate(COMPONENTS):
            mask = df.get(f"{c}_outlier")
            if mask is not None:
                y[i, mask.to_numpy(dtype=bool)] = np.nan
    return t, y


# ---------------------------------------------------------------------------
# drawing
# ---------------------------------------------------------------------------


def _summary(h: Harmonics) -> str:
    return (
        f"annual {h.annual:.1f} mm (max {peak_label(h.peak)}), "
        f"semiannual {h.semiannual:.1f} mm"
    )


def seasonal_figure(
    view: SeasonalView, *, bin_days: float = 15.0, n_donors: int = 0, fig: Any = None
) -> Any:
    """The three-panel (N/E/U) figure of one :class:`SeasonalView`."""
    import matplotlib.pyplot as plt

    if fig is None:
        fig = plt.figure(figsize=(8.5, 11))
    axes = fig.subplots(3, 1, sharex=True)
    x = np.linspace(0.0, 1.0, GRID_N)
    folded = None if view.data is None else fold(view.data[0])

    for k, ax in enumerate(axes):
        if view.data is not None and folded is not None:
            yk = view.data[1][k]
            ax.plot(folded, yk, ".", ms=2, color=DOT_COLOR, alpha=0.35,
                    rasterized=True, zorder=1)  # fmt: skip
            cx, med, err = binned_profile(folded, yk, bin_days=bin_days)
            ax.errorbar(cx, med, yerr=err, fmt="o", ms=4, color="k", lw=1,
                        zorder=4, label=f"{bin_days:g}-day bin median")  # fmt: skip
            lo, hi = np.nanpercentile(yk, [1, 99]) if np.isfinite(yk).any() else (-1, 1)
            pad = 0.15 * (hi - lo)
            ax.set_ylim(lo - pad, hi + pad)

        for code in list(view.weights)[:n_donors]:
            ax.plot(x, seasonal_curve(view.donors[code], x)[k], lw=0.6,
                    color=DONOR_COLOR, zorder=2)  # fmt: skip

        lines = []
        if view.own is not None:
            c = seasonal_curve(view.own, x)[k]
            if view.own_cov is not None:
                s = curve_sigma(view.own_cov, x)[k]
                ax.fill_between(x, c - s, c + s, color=OWN_COLOR, alpha=0.25, lw=0)
            ax.plot(x, c, color=OWN_COLOR, lw=2, zorder=5, label="own (estimated)")
            lines.append(("own", harmonics(view.own[k])))
        if view.regional is not None:
            c = seasonal_curve(view.regional, x)[k]
            if view.regional_sigma is not None:
                s = float(view.regional_sigma[k])
                ax.fill_between(x, c - s, c + s, color=REGIONAL_COLOR, alpha=0.12, lw=0)
            label = (
                "regional (stored)"
                if view.seasonal_only
                else "regional (leave-one-out)"
            )
            ax.plot(x, c, "--", color=REGIONAL_COLOR, lw=2, zorder=5, label=label)
            lines.append(("regional", harmonics(view.regional[k])))

        text = [f"{who}: {_summary(h)}" for who, h in lines]
        if view.own is not None and view.regional is not None:
            d = rms_difference(view.own, view.regional)[k]
            text.append(f"own − regional: RMS {d:.1f} mm over the year")
        elif view.regional is None and view.regional_why:
            text.append(f"regional refused: {view.regional_why}")
        ax.text(0.01, 0.98, "\n".join(text), transform=ax.transAxes, va="top",
                fontsize=8, family="monospace",
                bbox={"facecolor": "white", "alpha": 0.8, "lw": 0})  # fmt: skip
        ax.axhline(0, color="0.3", lw=0.5)
        ax.set_ylabel(f"{LABELS[k]} [mm]")
        ax.grid(True, lw=0.3, alpha=0.6)
        if k == 0:
            ax.legend(loc="upper right", fontsize=7, framealpha=0.8)

    months = [dt.date(2001, m, 1) for m in range(1, 13)]
    axes[-1].set_xticks(
        [(d - dt.date(2001, 1, 1)).days / DAYS_PER_YEAR for d in months]
    )
    axes[-1].set_xticklabels([f"{d:%b}" for d in months])
    axes[-1].set_xlim(0, 1)
    axes[-1].set_xlabel("time of year (yearf mod 1)")

    fig.suptitle(_title(view), fontsize=12)
    fig.text(0.5, 0.005, _footer(view, n_donors), ha="center", va="bottom",
             fontsize=7, color="0.3", wrap=True)  # fmt: skip
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    return fig


def _title(view: SeasonalView) -> str:
    who = f"{view.name} ({view.sta})" if view.name else view.sta
    if view.seasonal_only:
        return f"{who}: regional seasonal, {view.frame}"
    w = view.window
    span = f"{w[0]:.2f}–{w[1]:.2f}" if w else "no data"
    n = 0 if view.data is None else int(np.isfinite(view.data[1][0]).sum())
    return f"{who}: seasonal, {view.frame}; secular + steps removed, {span}, {n} days"


def _footer(view: SeasonalView, n_donors: int) -> str:
    parts = []
    if view.seasonal_only:
        parts.append("seasonal-only record: no secular of its own, so no data folded")
    elif view.record.get("window"):
        w = view.record["window"]
        parts.append(f"record fit window {float(w[0]):.2f}–{float(w[1]):.2f}")
    if view.weights:
        top = ", ".join(f"{c} {w:.2f}" for c, w in list(view.weights.items())[:6])
        parts.append(f"{len(view.weights)} donors (top: {top})")
    if view.regional_sigma is not None:
        s = "/".join(f"{v:.2f}" for v in view.regional_sigma)
        parts.append(f"regional band ±{s} mm N/E/U = leave-one-out annual-vector RMS")
    if n_donors:
        parts.append(f"grey lines: the {n_donors} heaviest donors")
    return "; ".join(parts)


# ---------------------------------------------------------------------------
# command
# ---------------------------------------------------------------------------


def build_parser(
    parser: argparse.ArgumentParser, *, config: Any
) -> argparse.ArgumentParser:
    from gps_plot.cli import common

    parser.description = (
        "A station's seasonal (annual + semiannual) folded onto one year: the "
        "data with the stored secular removed, the record's own seasonal, and "
        "the regional prediction from its neighbours (leave-one-out)."
    )
    parser.add_argument("Stations", nargs="+", help="station codes")
    parser.add_argument("-s", "--start", default=None,
                        help="start of the folded data, YYYYMMDD "
                        "(default: the record's fit window)")  # fmt: skip
    parser.add_argument("-e", "--end", default=None,
                        help="end of the folded data, YYYYMMDD "
                        "(default: the record's fit window)")  # fmt: skip
    parser.add_argument("--frame", default=common.DEFAULT_PLOT_FRAME,
                        help="reference frame of the .pos series and the "
                        "detrend document (default itrf2014)")  # fmt: skip
    parser.add_argument("--view", choices=("raw", "cleaned"), default="cleaned",
                        help="data view (default cleaned: outliers masked)")  # fmt: skip
    parser.add_argument("-u", "--uncert", type=float, default=10,
                        help="formal-σ screen at read time [mm] (default 10)")  # fmt: skip
    parser.add_argument("--bin-days", type=float, default=15.0,
                        help="width of the median bins [days] (default 15)")  # fmt: skip
    parser.add_argument("--donors", type=int, default=0, metavar="N",
                        help="also draw the N heaviest donors' seasonals")  # fmt: skip
    parser.add_argument("--params", default=None,
                        help="detrend document (default: the deployed one "
                        "of the frame)")  # fmt: skip
    parser.add_argument("-i", "--Dir", default=None,
                        help="the .pos directory (default: pos_dir)")  # fmt: skip
    common.add_output_options(parser, config)
    return parser


def _yearf(value: str | None) -> float | None:
    if value is None:
        return None
    from geo_dataread.velocity import yearf

    return float(yearf(dt.datetime.strptime(value, "%Y%m%d").date()))


def run(args: argparse.Namespace, parser: argparse.ArgumentParser, config: Any) -> int:
    """Plot each station; 0 when all plotted, 1 otherwise."""
    import matplotlib as mpl

    from geo_dataread.gps_views import default_params_path, read_detrend_params
    from geo_dataread.pos import frame_name, pos_product
    from geo_dataread.velocity import _coords

    from gps_plot.cli import common

    try:
        pos_id = pos_product(args.frame)
    except ValueError as exc:
        parser.error(str(exc))
    frame = frame_name(pos_id)
    path = args.params or default_params_path(pos_id)
    if path is None:
        parser.error(f"no detrend document for {frame}")
    doc = read_detrend_params(path)
    pos_dir = args.Dir or common.config_dir(config, "pos_dir")
    fig_dir = (
        args.figDir
        if args.figDir is not None
        else common.config_dir(config, "fig_dir", "")
    )
    mpl.use("Agg" if args.save else "WebAgg")

    failed = 0
    for sta in (s.upper() for s in args.Stations):
        try:
            view = station_view(
                sta, doc, frame=frame, pos_id=pos_id, pos_dir=str(pos_dir),
                start=_yearf(args.start), end=_yearf(args.end),
                clean=args.view == "cleaned", uncert=args.uncert, coords=_coords,
                name=_station_name(config, sta),
            )  # fmt: skip
        except ValueError as exc:
            print(f"{sta}: {exc}", file=sys.stderr)
            failed += 1
            continue
        fig = seasonal_figure(view, bin_days=args.bin_days, n_donors=args.donors)
        if not args.save:
            import matplotlib.pyplot as plt

            plt.show()
            continue
        from gps_plot import timesmatplt as tplt

        stem = args.name or f"{sta}-seasonal-{frame}" + (
            "" if args.view == "cleaned" else "-raw"
        )
        tplt.saveFig(os.path.join(str(fig_dir), stem), args.save, fig)
        print(f"{sta}: {os.path.join(str(fig_dir), stem)}.{args.save}")
        import matplotlib.pyplot as plt

        plt.close(fig)
    return 1 if failed else 0


def _station_name(config: Any, sta: str) -> str:
    try:
        return str(
            dict(dict(config.getStationInfo(sta))["station"]).get("station_name") or ""
        )
    except Exception:  # noqa: BLE001 - an unknown station has no name
        return ""


__all__: Sequence[str] = (
    "Harmonics",
    "SeasonalView",
    "binned_profile",
    "build_parser",
    "curve_sigma",
    "fold",
    "harmonics",
    "peak_label",
    "periodic_basis",
    "periodic_cov",
    "regional_prediction",
    "rms_difference",
    "run",
    "seasonal_curve",
    "seasonal_figure",
    "station_view",
)
