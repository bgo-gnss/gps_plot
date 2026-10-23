"""``gpsplot stack``: several stations and/or baselines in one N/E/U figure.

Every series comes from :func:`gps_plot.sources.gps.fetch` (the same read,
clean and detrend as ``gpsplot ts``) and is drawn by
:func:`gps_plot.compose.stack_figure`: overlaid with a legend by default,
``--offset MM`` spaces them apart. Station codes may be shell-style
patterns matched against the station list, on either side of a baseline:
``"SEY?-SEYD"`` is the nine SEY1…SEY9 − SEYD baselines (the self-pair
SEYD-SEYD is dropped). Quote patterns: an unquoted ``?`` is the shell's.
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import sys
import textwrap
import warnings
from collections.abc import Sequence
from typing import Any

from gps_plot.cli import common

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
        "its own first epoch.",
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
) -> str:
    """The figure's file name (no directory, no suffix).

    ``stack_<label or first series_count>-<ref>[-terms][-cleaned]
    [-offsetN][-<frame>][-8h]-<period>``. A labelled (plots.yaml) stack
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
    start, end = common.resolve_dates(args, currDatetime)
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
    )
    bl_opts = source.BaselineOptions(
        tol=2.0 / 24.0 / 365.25 if is_8h else BASELINE_EPOCH_TOLERANCE_YEARS
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
        fig=tplt._reusable_figure() if args.save else None,
    )
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
