"""``gpsplot stack``: several stations and/or baselines in one N/E/U figure.

Every series comes from :func:`gps_plot.sources.gps.fetch` (the same read,
clean and detrend as ``gpsplot ts``) and is drawn by
:func:`gps_plot.compose.stack_figure`: overlaid with a legend by default,
``--offset MM`` spaces them apart. Station codes may be shell-style
patterns matched against the station list, on either side of a baseline:
``"SEY?-SEYD"`` is the nine SEY1…SEY9 − SEYD baselines (the self-pair
SEYD-SEYD is dropped). Quote patterns: an unquoted ``?`` is the shell's.

Each series is zeroed at its own first epoch. A series that starts later
than the others is put on the common start instead (:func:`backcast_late_starts`,
BGÓ 2026-10-23): its rate over its first stretch of data is extended back to
the stack's first epoch and the series is shifted so that line is zero there;
the figure draws that line dashed. Too little data to fit a rate leaves the
series zeroed at its start, and the legend says so.
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import sys
import textwrap
import warnings
import dataclasses
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from gps_plot.cli import common
from gps_plot.series import Series

SPECIALS = ["90d", "year", "full", "fixedstart"]
REFS = ["plate", "detrend", common.REF_FRAME]
VIEWS = ["raw", "cleaned"]
TYPES = ["TOT", "POS"]

EXAMPLES = """\
examples:
  gpsplot stack "SEY?-SEYD" --special year --save pdf
                                 the nine SEY1..SEY9 - SEYD baselines
  gpsplot stack HOFN SELF RHOF --ref plate --offset 20 --save pdf
                                 three stations, 20 mm apart
  gpsplot stack "SEY?-SEYD" SEYD --ref detrend --save pdf
                                 baselines and a station together
Patterns (? * [..]) match the station list; quote them for the shell."""


def build_parser(
    parser: argparse.ArgumentParser | None = None, config: Any = None
) -> argparse.ArgumentParser:
    """The ``gpsplot stack`` options (the series ones shared with ``ts``)."""
    if config is None:
        from gps_plot.plot_gps_timeseries import _load_config

        config = _load_config()
    if parser is None:
        parser = argparse.ArgumentParser(prog="gpsplot stack")
    parser.description = textwrap.fill(
        "Several GPS time series in one figure: stations and/or baselines "
        "AAAA-BBBB, overlaid with a legend (--offset spaces them). Each series "
        "is read, cleaned and detrended exactly as by gpsplot ts and zeroed at "
        "its own first epoch; one that starts late is put on the common start "
        "along its early rate (dashed line).",
        width=78,
    )
    parser.formatter_class = argparse.RawDescriptionHelpFormatter
    parser.epilog = EXAMPLES
    parser.add_argument(
        "Series",
        nargs="+",
        help="stations, baselines AAAA-BBBB, or patterns of either "
        '("SEY?-SEYD", "HV*")',
    )
    parser.add_argument(
        "--offset",
        type=float,
        default=0.0,
        metavar="MM",
        help="vertical spacing between consecutive series [mm] (default 0: "
        "overlaid); the legend gives each series' offset",
    )
    common.add_period_options(parser, specials=SPECIALS)
    common.add_event_options(parser)
    common.add_output_options(parser, config)
    common.add_series_options(
        parser,
        refs=REFS,
        ref_help="what is removed from the positions, the same for every "
        "series: ref_frame = nothing, plate = each station's plate velocity, "
        "detrend = plate + the stored rate and seasonals (--detrend-terms). "
        "Default: plate if any series is a baseline, ref_frame otherwise",
        views=VIEWS,
        types=TYPES,
        sessions=["24h", "8h"],
        session_help="24h daily solutions (default) or 8h sessions (itrf2020 "
        "only; the frame defaults to itrf2020)",
    )
    parser.add_argument(
        "--baseline-sigma",
        choices=("formal", "empirical"),
        default="formal",
        help="baselines: 'formal' (default) = quadrature of the two stations' "
        "GLOBK sigma; 'empirical' = the fitted baseline noise record, as "
        "gpsplot ts --baseline-sigma. Adds -empsig",
    )
    parser.add_argument(
        "--show-outliers",
        type=int,
        nargs="?",
        const=0,
        default=None,
        metavar="N",
        help="with --view cleaned: draw each series' outliers in its own colour "
        "with a bold black outline (all of them; N: only among its last N "
        "epochs). Off by default",
    )
    parser.add_argument(
        "--outlier-sigma",
        type=float,
        default=None,
        metavar="K",
        help="with --view cleaned: the outlier tolerance in local noise "
        "units, as gpsplot ts (default 4.5; the published plots use 3.5)",
    )
    parser.add_argument(
        "--show-obstruction",
        type=int,
        nargs="?",
        const=0,
        default=None,
        metavar="N",
        help="with --view cleaned: draw the icing and snow-burial days (set "
        "aside) light blue, edged in the series' colour; N: only among its "
        "last N epochs. Off by default",
    )
    parser.add_argument(
        "--show-network",
        type=int,
        nargs="?",
        const=0,
        default=None,
        metavar="N",
        help="with --view cleaned: draw the network days (set aside) dark "
        "grey, edged in the series' colour; N as --show-obstruction",
    )
    parser.add_argument(
        "--colors",
        nargs="+",
        default=None,
        metavar="SERIES=COLOR",
        help="a series' own colour, e.g. --colors SVIN-SKFC=red "
        "SVIE-SKFC=#1f77b4 (any matplotlib colour); the rest keep the "
        "palette colour of their name",
    )
    parser.add_argument(
        "--label",
        default=None,
        metavar="NAME",
        help="name the stack in the file name: stack_<NAME>-<ref>-...-<period> "
        "instead of stack_<first series>_<count> (plots.yaml baseline groups)",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        metavar="FILE",
        help="append the files written to FILE, one absolute path per line",
    )
    parser.add_argument(
        "--detrend-terms",
        default=None,
        metavar="TERM[,TERM...]",
        help="with --ref detrend: which stored terms to subtract (linear, "
        "periodic, transient); default linear,periodic",
    )
    return parser


# ---------------------------------------------------------------------------
# series list
# ---------------------------------------------------------------------------


def outlier_params(sigma: float | None) -> Any:
    """``--outlier-sigma K`` as the cleaned view's OutlierParams (the
    running window's k, as gpsplot ts); None keeps the default."""
    from gps_plot import plot_gps_timeseries as pgt

    return pgt._build_outlier_params([], base=pgt._method_overrides(None, sigma))


def parse_colors(items: Sequence[str] | None) -> dict[str, str]:
    """``--colors SERIES=COLOR ...`` as ``{series: colour}``; a bad colour
    or a missing ``=`` raises ValueError."""
    from matplotlib.colors import is_color_like

    out: dict[str, str] = {}
    for item in items or ():
        name, sep, color = item.partition("=")
        if not (sep and name and is_color_like(color)):
            raise ValueError(f"--colors {item!r}: give SERIES=COLOR")
        out[name.upper()] = color
    return out


def series_colors(series: Sequence[Series], own: Mapping[str, str]) -> list[str]:
    """Each series' colour: its own (``--colors``) or the palette's."""
    from gps_plot import compose

    palette = compose.series_colors([s.label for s in series])
    return [own.get(s.label.upper(), c) for s, c in zip(series, palette, strict=True)]


#: a series starting more than this after the stack's first epoch is backcast
BACKCAST_MIN_GAP_DAYS = 3.0
#: the rate is fitted over the series' first max(gap, this) days of data
BACKCAST_FIT_DAYS = 90.0
#: and needs at least this many epochs spanning at least this many days
BACKCAST_MIN_POINTS = 10
BACKCAST_MIN_SPAN_DAYS = 30.0
_DAYS_PER_YEAR = 365.25


#: the backcast line rejects epochs whose residual exceeds this many robust
#: σ (1.4826·MAD), refitting until the kept set settles
BACKCAST_CLIP_SIGMA = 4.0
BACKCAST_CLIP_ITER = 5


def _rate_line(
    t: np.ndarray, y: np.ndarray, sigma: np.ndarray, t_ref: float
) -> tuple[float, float] | None:
    """σ-weighted straight line through (t, y): (value at ``t_ref``, rate/yr).

    Robust to the odd undetected outlier at a series' start (OLAC, July
    2021: one Up epoch ~330 mm above its neighbours lifted the whole
    backcast): after each fit, epochs further than
    :data:`BACKCAST_CLIP_SIGMA` robust σ (1.4826·MAD of the residuals)
    from the line are left out and the line refitted — the median/MAD
    counterpart of the old plots' window median and rolling z-score.
    """
    ok = np.isfinite(t) & np.isfinite(y) & np.isfinite(sigma) & (sigma > 0)
    keep = ok.copy()
    fit: tuple[float, float] | None = None
    for _ in range(BACKCAST_CLIP_ITER):
        if keep.sum() < 2:
            return fit
        dt = t[keep] - t_ref
        w = 1.0 / sigma[keep] ** 2
        a = np.array([[w.sum(), (w * dt).sum()], [(w * dt).sum(), (w * dt * dt).sum()]])
        b = np.array([(w * y[keep]).sum(), (w * dt * y[keep]).sum()])
        try:
            level, rate = np.linalg.solve(a, b)
        except np.linalg.LinAlgError:
            return fit
        fit = (float(level), float(rate))
        resid = np.where(ok, y - (level + rate * (t - t_ref)), np.nan)
        scale = 1.4826 * float(
            np.nanmedian(np.abs(resid[keep] - np.nanmedian(resid[keep])))
        )
        if not scale > 0:
            break
        new_keep = ok & (np.abs(resid) <= BACKCAST_CLIP_SIGMA * scale)
        if np.array_equal(new_keep, keep):
            break
        keep = new_keep
    return fit


def backcast_late_starts(series: Sequence[Series]) -> tuple[list[Series], list[bool]]:
    """Put series that start late on the stack's common zero (best effort).

    Every series arrives zeroed at its own first epoch. One that starts more
    than :data:`BACKCAST_MIN_GAP_DAYS` after the earliest series gets a
    straight line (σ-weighted, per component) through its first
    max(gap, :data:`BACKCAST_FIT_DAYS`) days of data; the line is extended
    back to the stack's first epoch ``t0`` and the series shifted so that the
    line is zero at ``t0`` — where it would sit had it been measured from the
    start and moved at that rate. ``meta["backcast"]`` keeps ``(t0, t1,
    values at t1)`` of the line for the figure. A series with fewer than
    :data:`BACKCAST_MIN_POINTS` epochs (or :data:`BACKCAST_MIN_SPAN_DAYS`) in
    the fit stretch is left as it is.

    Returns the series and, per series, whether it was left zeroed at its
    own start although it starts late.
    """
    starts = [float(np.nanmin(np.asarray(s.t, dtype=float))) for s in series]
    t0 = min(starts)
    out: list[Series] = []
    unfitted: list[bool] = []
    for s, t1 in zip(series, starts, strict=True):
        gap = (t1 - t0) * _DAYS_PER_YEAR
        if gap <= BACKCAST_MIN_GAP_DAYS:
            out.append(s)
            unfitted.append(False)
            continue
        t = np.asarray(s.t, dtype=float)
        y = np.asarray(s.y, dtype=float)
        sigma = np.asarray(s.sigma, dtype=float)
        stretch = (t >= t1) & (t <= t1 + max(gap, BACKCAST_FIT_DAYS) / _DAYS_PER_YEAR)
        lines = []
        for c in range(y.shape[0]):
            ok = stretch & np.isfinite(y[c])
            span = (t[ok].max() - t[ok].min()) * _DAYS_PER_YEAR if ok.any() else 0.0
            if ok.sum() < BACKCAST_MIN_POINTS or span < BACKCAST_MIN_SPAN_DAYS:
                lines.append(None)
                continue
            lines.append(_rate_line(t[ok], y[c][ok], sigma[c][ok], t1))
        if any(line is None for line in lines):
            out.append(s)
            unfitted.append(True)
            continue
        # value of each line at t0 (the shift) and at t1 (where it meets the data)
        at_t0 = np.array([lv + r * (t0 - t1) for lv, r in lines])  # type: ignore[misc]
        at_t1 = np.array([lv for lv, _ in lines])  # type: ignore[misc]

        def moved(overlay: Any) -> Any:  # an overlay moves with its series
            if overlay is None:
                return None
            oy, ody = overlay
            return (np.asarray(oy, dtype=float) - at_t0[:, None], ody)

        out.append(
            dataclasses.replace(
                s,
                y=y - at_t0[:, None],
                outliers=moved(s.outliers),
                iced=moved(s.iced),
                buried=moved(s.buried),
                netday=moved(s.netday),
                meta={**s.meta, "backcast": (t0, t1, at_t1 - at_t0)},
            )
        )
        unfitted.append(False)
    return out, unfitted


def _is_pattern(code: str) -> bool:
    return any(c in code for c in "*?[")


def _match(code: str, stations: Sequence[str]) -> list[str]:
    if not _is_pattern(code):
        return [code.upper()]
    return sorted(s for s in stations if fnmatch.fnmatchcase(s, code.upper()))


def split_spec(spec: str) -> tuple[str, str] | None:
    """``"A-B"`` → ``("A", "B")`` at a ``-`` outside ``[...]``, else None.

    Not ``split_baseline``: a pattern side need not be a 4-char code, and
    a ``-`` inside a character class (``SEY[1-4]``) is a range.
    """
    depth = 0
    for i, c in enumerate(spec):
        if c == "[":
            depth += 1
        elif c == "]":
            depth = max(0, depth - 1)
        elif c == "-" and depth == 0:
            a, b = spec[:i], spec[i + 1 :]
            return (a, b) if a and b else None
    return None


def expand_series(specs: Sequence[str], stations: Sequence[str]) -> list[str]:
    """Patterns → concrete stations/baselines, in order, without repeats.

    A baseline expands each side (``SEY?-SEYD`` → SEY1-SEYD … SEY9-SEYD)
    and drops the self-pairs. A pattern that matches nothing is an error,
    not an empty plot.
    """
    out: list[str] = []
    for spec in specs:
        pair = split_spec(spec)
        if pair is not None:
            a_side, b_side = _match(pair[0], stations), _match(pair[1], stations)
            got = [f"{a}-{b}" for a in a_side for b in b_side if a != b]
        else:
            got = _match(spec, stations)
        if not got:
            raise ValueError(f"{spec!r} matches no station")
        out.extend(g for g in got if g not in out)
    return out


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


def _ref_title(
    ref: str, tType: str, pos_id: str | None, terms: Any, frame: bool = True
) -> str:
    """The title's frame text: what was removed, and in which frame
    (``frame=False``: not named -- a stack of baselines, where it cancels)."""
    if ref == "plate":
        text = "Plate velocity removed"
    elif ref == "detrend":
        from gps_plot.timesmatplt import DEFAULT_DETREND_TERMS

        text = "Detrended"
        if terms and tuple(terms) != DEFAULT_DETREND_TERMS:
            text += f" ({' + '.join(terms)})"
    else:
        text = "Reference frame"
    if frame and tType == "POS" and pos_id is not None:
        from geo_dataread.pos import POS_PRODUCTS, pos_product

        name = POS_PRODUCTS.get(pos_product(pos_id), ("", ""))[0]
        if name:
            text = f"{text}, {name}"
    return text


def file_stem(series: Sequence[str], label: str | None = None) -> str:
    """``stack_SEY1_SEYD_9`` (or ``stack_<label>``): no ``-`` before the ref
    (publish globs split there)."""
    if label:
        return "stack_" + label.replace("-", "_")
    return f"stack_{series[0].replace('-', '_')}_{len(series)}"


def output_stem(
    series: Sequence[str],
    *,
    label: str | None,
    ref: str,
    detrend_terms: Sequence[str] | None,
    view: str | None,
    offset: float,
    tType: str,
    pos_id: str | None,
    special: str | None,
    first: Any,
    last: Any,
    baseline_sigma: str = "formal",
) -> str:
    """The figure's file name (no directory, no suffix).

    ``stack_<label or first series_count>-<ref>[-terms][-cleaned][-empsig]
    [-offsetN][-<frame>][-8h]-<period>`` (``-empsig``: empirical baseline σ,
    when a series is a baseline — as gpsplot ts names it). A labelled (plots.yaml) stack
    leaves out the offset (a display setting, not another product), and a
    stack of baselines the frame (it cancels in a baseline).
    """
    from geo_dataread.baseline import split_baseline

    from gps_plot import timesmatplt as tplt

    tag = "ref_frame" if ref == common.REF_FRAME_INTERNAL else ref
    stem = f"{file_stem(series, label)}-{tag}"
    if (
        ref == "detrend"
        and detrend_terms
        and tuple(detrend_terms) != tplt.DEFAULT_DETREND_TERMS
    ):
        stem += "-" + "-".join(detrend_terms)
    if view == "cleaned":
        stem += "-cleaned"
    if baseline_sigma == "empirical" and any(split_baseline(x) for x in series):
        stem += "-empsig"
    if offset and not label:
        stem += f"-offset{offset:g}"
    if tType == "POS" and pos_id is not None:
        from geo_dataread.pos import frame_name, pos_product

        if not all(split_baseline(x) for x in series):
            stem += f"-{frame_name(pos_product(pos_id))}"
        if pos_id.startswith("imo.08hr"):
            stem += "-8h"
    elif tType != "TOT":
        stem += f"-{tType}"
    if special == "fixedstart":
        stem += "_since-%s" % first.strftime("%Y%m%d")
    elif special:
        stem += f"-{special}"
    else:
        stem += "_%s-%s" % (first.strftime("%Y%m%d"), last.strftime("%Y%m%d"))
    return stem


def run(args: argparse.Namespace, parser: argparse.ArgumentParser, config: Any) -> int:
    """Plot the stack; 0 when at least one series plotted, 1 otherwise."""
    import matplotlib as mpl
    from gtimes.timefunc import currDatetime

    from gps_plot import compose
    from gps_plot import timesmatplt as tplt
    from gps_plot.plot_gps_timeseries import _detrend_terms
    from gps_plot.sources import gps as source

    kwargs = vars(args).copy()
    try:
        series_specs = expand_series(args.Series, config.getStationInfo())
        detrend_terms = _detrend_terms(args.detrend_terms)
    except ValueError as exc:
        parser.error(str(exc))
    if detrend_terms is not None and args.ref != "detrend":
        parser.error("--detrend-terms needs --ref detrend")
    for opt in ("show_outliers", "show_obstruction", "show_network"):
        if (getattr(args, opt) or 0) < 0:
            parser.error(f"--{opt.replace('_', '-')} N: N must be positive")
    if args.outlier_sigma is not None and args.outlier_sigma <= 0:
        parser.error("--outlier-sigma: must be > 0")
    try:
        own_colors = parse_colors(args.colors)
    except ValueError as exc:
        parser.error(str(exc))
    start, end = common.resolve_dates(args, currDatetime)
    try:
        events = common.resolve_events(args)
    except ValueError as exc:
        parser.error(str(exc))
    common.resolve_source(kwargs, args, parser, config)
    start, end, fstart, fend = tplt.resolve_period(start, end, args.special, args.fix)

    from geo_dataread.baseline import BASELINE_EPOCH_TOLERANCE_YEARS, split_baseline

    ref = args.ref
    if ref is None:
        ref = "plate" if any(split_baseline(s) for s in series_specs) else "itrf2008"
    elif ref == common.REF_FRAME:
        ref = common.REF_FRAME_INTERNAL
    pos_id = kwargs.get("pos_id")
    is_8h = pos_id is not None and pos_id.startswith("imo.08hr")
    request = source.SeriesRequest(
        fstart=fstart,
        fend=fend,
        ref=ref,
        Dir=kwargs["Dir"],
        tType=kwargs["tType"],
        uncert=args.uncert,
        view=args.view,
        detrend_terms=detrend_terms,
        pos_id=pos_id,
        outlier_params=outlier_params(args.outlier_sigma),
    )
    empirical = args.baseline_sigma == "empirical"
    if empirical and is_8h:
        parser.error(
            "--baseline-sigma empirical: the noise record is fitted on daily "
            "solutions; use formal for 8 h sessions"
        )
    bl_opts = source.BaselineOptions(
        tol=2.0 / 24.0 / 365.25 if is_8h else BASELINE_EPOCH_TOLERANCE_YEARS,
        empirical=empirical,
    )

    got, labels, dropped = [], [], []
    for spec in series_specs:
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                s = source.fetch(spec, request, bl_opts)
                if ref == "detrend":
                    s = source.rezero_detrended(spec, request, s)
        except Exception as exc:  # one bad series must not sink the stack
            dropped.append(f"{spec}: {exc}")
            continue
        for w in caught:
            print(f"warning: {spec}: {w.message}", file=sys.stderr)
        label = spec
        if (ref == "detrend" or args.view == "detrended") and not s.detrend_applied:
            # no stored record: the series is the plate series, not
            # detrended; never let it pass as detrended in the legend
            label += " (not detrended)"
        got.append(s)
        labels.append(label)
    for line in dropped:
        print(f"dropped {line}", file=sys.stderr)
    if not got:
        print("nothing plotted", file=sys.stderr)
        return 1

    got, unfitted = backcast_late_starts(got)
    labels = [
        f"{label} (zeroed at its start)" if late else label
        for label, late in zip(labels, unfitted, strict=True)
    ]

    if not args.save:
        mpl.use("WebAgg")
    last = max(tplt._as_datetime_list(s.t)[-1] for s in got)
    all_baselines = all(split_baseline(x) for x in series_specs)
    title_text = _ref_title(
        ref, kwargs["tType"], pos_id, detrend_terms, frame=not all_baselines
    )
    status = tplt.make_title(series_specs[0], last, ref=title_text)
    title = tplt.StationTitle(
        main=(
            r"\Huge %s \ \Large %d series, %s"
            % (args.label.replace("_", r"\_"), len(got), title_text)
            if args.label
            else r"\Huge %d series \ \Large %s" % (len(got), title_text)
        ),
        status=status.status,
        created=status.created,
        status_color=status.status_color,
    )
    fig = compose.stack_figure(
        got,
        title=title,
        labels=labels,
        start=start,
        end=end,
        ylim=args.ylim,
        offset=args.offset,
        colors=series_colors(got, own_colors),
        show_outliers=args.show_outliers if args.view == "cleaned" else None,
        show_obstruction=args.show_obstruction if args.view == "cleaned" else None,
        show_network=args.show_network if args.view == "cleaned" else None,
        fig=tplt._reusable_figure() if args.save else None,
    )
    if events:
        tplt.addEvent(events, fig)
    tplt.inpLogo(fig)  # type: ignore[no-untyped-call]

    if not args.save:
        import matplotlib.pyplot as plt

        plt.show()
        return 0
    first = min(tplt._as_datetime_list(s.t)[0] for s in got)
    stem = args.name or output_stem(
        series_specs,
        label=args.label,
        ref=ref,
        detrend_terms=detrend_terms,
        view=args.view,
        offset=args.offset,
        tType=kwargs["tType"],
        pos_id=pos_id,
        special=args.special,
        first=first,
        last=last,
        baseline_sigma=args.baseline_sigma,
    )
    tplt.saveFig(os.path.join(kwargs["figDir"], stem), args.save, fig)
    if args.manifest:
        with open(args.manifest, "a") as fh:
            fh.writelines(
                os.path.abspath(f) + "\n" for f in getattr(fig, "_gps_files", ())
            )
    if dropped:
        print(f"plotted {len(got)} of {len(series_specs)} series", file=sys.stderr)
    return 0
