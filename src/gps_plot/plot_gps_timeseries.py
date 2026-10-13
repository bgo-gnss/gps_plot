#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse
import concurrent.futures
import datetime as dt
import functools
import os
import sys
import textwrap
import traceback

# Only the three helpers the MODERN lane calls are annotated below. This
# module is `ignore_errors` for mypy (see pyproject's typing policy), so
# these signatures type the SEAM, not the module: detrend_workbench is
# strict-checked and cannot call an untyped function.
from collections.abc import Mapping, Sequence
from typing import Any

# from gtimes.timefunc import TimefromYearf, currTime, TimetoYearf
from gtimes.timefunc import currDatetime

import gps_plot.timesmatplt as tplt

# gps_parser is imported lazily inside main() (it needs a deployed
# gpsconfig dir), so the module stays importable without it.


# functions
def tryTimes(sta, **kwargs):
    """
    Catching exceptions from tplt.plotTime
    """

    try:  # Trying to plot
        print("%s Plotting" % sta)
        fig = tplt.plotTime(sta, **kwargs)
        print(
            "plotted %s using: %s, %s"
            % (sta, getattr(fig, "_gps_label", kwargs["ref"]), kwargs["special"])
        )
    except IndexError as e:
        top = traceback.extract_stack()[-1]
        errorstr = "%s: %s, %s: " % (sta, kwargs["ref"], kwargs["special"])
        errorstr += ", ".join(
            [
                type(e).__name__,
                os.path.basename(top[0]),
                str(top[1]),
                "For station %s" % sta,
            ]
        )
        print(">> {}, {}".format(sys.stderr, errorstr))

    except Exception:
        traceback.print_exc()
        # print >>sys.stderr, top
        print(
            sys.stderr,
            "Unexpected error: %s during processing of %s" % (sys.exc_info()[0], sta),
        )


def _install_raw_read_cache(maxsize=4):
    """Cache raw GLOBK file reads per ``(sta, Dir, tType)`` in this process.

    ``getData`` re-reads and re-parses the same three ``mb_*.dat{1,2,3}``
    files once per sub-period (90d/year/full/fixedstart) although only the
    date window differs.  Only the deterministic disk read
    (``openGlobkTimes``) is cached: the window-DEPENDENT math downstream
    (``dPeriod`` windowing, ``iprep`` zero-referencing at the window start,
    plate-velocity removal accumulating from the window's first epoch) must
    still run per call, so every plot gets bit-identical data to an
    uncached read.  Each cache hit returns fresh copies because ``getData``
    mutates the arrays in place (``iprep``'s m->mm conversion, plate
    removal).
    """
    import geo_dataread.gps_read as gpsr

    if getattr(gpsr.openGlobkTimes, "_gps_plot_raw_read_cache", False):
        return

    uncached = gpsr.openGlobkTimes
    raw_read = functools.lru_cache(maxsize=maxsize)(uncached)

    @functools.wraps(gpsr.openGlobkTimes)
    def cached_openGlobkTimes(sta, Dir=None, tType="TOT", **kw):
        # kw (POS ref_position/convention/pos_id) are normally strings and so
        # hashable cache keys; a reference POSITION (an XYZ array, as the 8 h
        # overlay passes) is not, and such a read goes around the cache
        try:
            yearf, data, ddata = raw_read(sta, Dir, tType, **kw)
        except TypeError as exc:
            if "unhashable" not in str(exc):
                raise
            yearf, data, ddata = uncached(sta, Dir=Dir, tType=tType, **kw)
        return yearf.copy(), data.copy(), ddata.copy()

    cached_openGlobkTimes._gps_plot_raw_read_cache = True
    gpsr.openGlobkTimes = cached_openGlobkTimes


def plotStation(sta, variants):
    """Render every requested (ref, special) plot variant for one station.

    This is the unit of (process-)parallel work: one station, all its
    sub-period/reference variants in sequence, so the raw-read cache turns
    the former once-per-variant file read into one read per station.
    Per-plot fault tolerance is unchanged (:func:`tryTimes`).
    """
    _install_raw_read_cache()
    for kwargs in variants:
        tryTimes(sta, **kwargs)


def _expand_variants(kwargs, refs, specials):
    """Expand ``--ref``/``--special`` "all" into per-station plot kwargs.

    ``kwargs["ref"]``/``kwargs["special"]`` equal to ``"all"`` expand to
    the allowed values (``refs`` outer, ``specials`` inner -- the legacy
    loop order); anything else passes through unchanged.  Always returns
    fresh dicts safe to ship to worker processes.
    """
    ref_all = kwargs.get("ref") == "all"
    special_all = kwargs.get("special") == "all"
    base = {
        key: value
        for key, value in kwargs.items()
        if not ((key == "ref" and ref_all) or (key == "special" and special_all))
    }

    if ref_all and special_all:
        return [dict(base, ref=r, special=s) for r in refs for s in specials]
    if special_all:
        return [dict(base, special=s) for s in specials]
    if ref_all:
        return [dict(base, ref=r) for r in refs]
    return [base]


def _coerce_outlier_param(name, typ, raw):
    """Coerce one ``NAME=VALUE`` string to an OutlierParams field type."""
    if typ is bool:
        low = raw.lower()
        if low in ("1", "true", "yes", "on"):
            return True
        if low in ("0", "false", "no", "off"):
            return False
        # bool("False") is True -- never let str fall through to the cast
        raise SystemExit(
            "--outlier-param %s: expected a boolean (true/false), got %r" % (name, raw)
        )
    try:
        return typ(raw)
    except (ValueError, TypeError):
        raise SystemExit(
            "--outlier-param %s: expected %s, got %r" % (name, typ.__name__, raw)
        ) from None


def _build_outlier_params(
    assignments: Sequence[str], base: Mapping[str, Any] | None = None
) -> Any:
    """Build an ``OutlierParams`` from repeated ``NAME=VALUE`` CLI args.

    Field names and types are read off the dataclass itself, so this flag
    stays in sync with ``gps_analysis`` and NO detection default is
    restated here -- unassigned fields keep their spec values.  Types come
    from :func:`typing.get_type_hints`, not ``field.type``, which is a bare
    string whenever the leaf adopts ``from __future__ import annotations``.

    Returns None when nothing was assigned; that (rather than a
    default-valued object) is what lets the per-station
    ``outlier_overrides.csv`` row win downstream.
    """
    if not assignments and not base:
        return None

    import dataclasses
    import typing

    from gps_analysis import OutlierParams

    hints = typing.get_type_hints(OutlierParams)
    values = dict(base or {})
    valid = sorted(f.name for f in dataclasses.fields(OutlierParams))
    for item in assignments:
        name, sep, raw = item.partition("=")
        name = name.strip()
        if not sep:
            raise SystemExit("--outlier-param expects NAME=VALUE, got %r" % (item,))
        if name not in valid:
            raise SystemExit(
                "--outlier-param: unknown field %r; valid fields: %s"
                % (name, ", ".join(valid))
            )
        values[name] = _coerce_outlier_param(name, hints[name], raw.strip())

    try:
        # OutlierParams.__post_init__ validates ranges/enums -- surface it as a
        # clean CLI error rather than a traceback
        return OutlierParams(**values)
    except (ValueError, TypeError) as exc:
        raise SystemExit("--outlier-param: %s" % (exc,)) from None


#: ``--outlier-param`` NAMEs grouped by the stage each acts on, so ``--help``
#: reads as something an operator can navigate rather than 31 alphabetical
#: fields.  Only the ORDERING lives here -- names, types and defaults are read
#: off the dataclass, and any field this map does not know lands in a trailing
#: "other" group.  So the listing can go stale in arrangement, never in
#: COVERAGE: a leaf that adds a threshold still documents it.
OUTLIER_PARAM_GROUPS = (
    (
        "S0 despike",
        ("despike", "despike_n_sigma", "despike_return_sigma", "despike_gap_days"),
    ),
    (
        "S1/S2 fit+whiten",
        ("loss", "f_scale", "whiten_sigma_clip", "scale_estimator", "scale_floor"),
    ),
    ("S3 global", ("enable_global", "global_n_sigma")),
    (
        "S4 windowed",
        (
            "enable_window",
            "window_days",
            "window_n_sigma",
            "window_min_count",
            "window_order",
            "window_robust_iterations",
        ),
    ),
    (
        "S5 protection",
        (
            "enable_protection",
            "min_outlier",
            "max_run_days",
            "cluster_gap_days",
            "run_sign_fraction",
            "step_evidence_sigma",
            "step_window_days",
            "step_flank_max_reach_days",
            "step_magnitude_ratio",
            "protect_on_indeterminate",
        ),
    ),
    (
        "iteration/abort",
        (
            "max_iterations",
            "max_flag_fraction",
            "min_abort_candidates",
            "epoch_policy",
        ),
    ),
)


def outlier_param_help(width: int = 78) -> str:
    """The ``--outlier-param`` NAME list, for a parser epilog.

    Generated from ``gps_analysis.OutlierParams`` at parse time rather
    than written down, for the same reason
    :func:`_build_outlier_params` reads the dataclass: a help text that
    restated the fields would drift from the detector the moment the leaf
    gained one, and it would do so SILENTLY -- an operator would read a
    name that no longer exists, or never learn about one that does.

    The shown values are the SPEC defaults, and are labelled as such
    because they are not necessarily what a given station gets: an unset
    field defers to that station's ``outlier_overrides.csv`` row first.
    Booleans print as ``true``/``false``, one of the spellings
    :func:`_coerce_outlier_param` accepts, so a line can be copied
    straight onto the command line.

    Returns:
        The block, or "" if ``gps_analysis`` cannot be imported --
        ``--help`` must never be the thing that fails.
    """
    import dataclasses

    try:
        from gps_analysis import OutlierParams
    except Exception:
        return ""

    fields = {f.name: f for f in dataclasses.fields(OutlierParams)}
    grouped = [
        (title, [n for n in names if n in fields])
        for title, names in OUTLIER_PARAM_GROUPS
    ]
    known = {n for _t, names in grouped for n in names}
    other = [n for n in fields if n not in known]
    if other:
        grouped.append(("other", other))

    def _shown(name):
        default = fields[name].default
        if isinstance(default, bool):
            return "%s=%s" % (name, str(default).lower())
        return "%s=%s" % (name, default)

    label_w = max(len(t) for t, _n in grouped) + 2
    lines = textwrap.wrap(
        "--outlier-param NAME=VALUE, repeatable. Valid NAMEs, shown with "
        "the SPEC default (a station's outlier_overrides.csv row wins "
        "wherever a NAME is left unset, and is REPLACED wholesale by any "
        "override):",
        width=width,
    ) + [""]
    for title, names in grouped:
        if not names:
            continue
        body = ", ".join(_shown(n) for n in names)
        lines.extend(
            textwrap.wrap(
                body,
                width=width,
                initial_indent="  %-*s" % (label_w, title),
                subsequent_indent="  " + " " * label_w,
            )
        )
    return "\n".join(lines)


#: Detection stages selectable with ``--stages`` (§14).  S1/S2 (the robust
#: trajectory fit and the whitening) are STRUCTURAL — every later stage is
#: defined on their residuals — so they are always on and are accepted only
#: for documentation symmetry.
STAGE_FLAGS = {
    "S0": "despike",
    "S3": "enable_global",
    "S4": "enable_window",
    "S5": "enable_protection",
}
STAGE_ALWAYS_ON = ("S1", "S2")


def _stage_overrides(spec: str | None) -> dict[str, bool] | None:
    """``--stages`` -> OutlierParams field overrides, or None if unset.

    Named stages are ON, unnamed ones OFF.  This replaces the sentinel-
    threshold idiom (``global_n_sigma=1e9``), which could not express intent,
    could not be asserted on, and for the protection stage could not express
    "off" at all -- its indeterminate arm ignored every threshold (§3.4.2a).
    """
    if not spec:
        return None
    if spec.strip().lower() == "all":
        return {v: True for v in STAGE_FLAGS.values()}
    want = [x.strip().upper() for x in spec.split(",") if x.strip()]
    bad = [x for x in want if x not in STAGE_FLAGS and x not in STAGE_ALWAYS_ON]
    if bad:
        raise SystemExit(
            "--stages: unknown stage(s) %s; valid: %s (S1/S2 are structural "
            "and always run)" % (", ".join(bad), ", ".join(sorted(STAGE_FLAGS)))
        )
    return {field: (name in want) for name, field in STAGE_FLAGS.items()}


def exit_gracefully(signum, frame):
    """Exit gracefully on Ctrl-C"""

    # restore the original signal handler as otherwise evil things will happen
    # in input() when CTRL+C is pressed, and our signal handler is not re-entrant
    signal.signal(signal.SIGINT, original_sigint)

    try:
        if input("\nReally quit? (y/n)> ").lower().startswith("y"):
            sys.exit(1)

    except KeyboardInterrupt:
        print("Ok ok, quitting")
        sys.exit(1)

    # restore the exit gracefully handler here
    signal.signal(signal.SIGINT, exit_gracefully)

    # Method borrowed from:
    # http://stackoverflow.com/questions/18114560/python-catch-ctrl-c-command-prompt-


####    ------------------------------------------------   ####


EXAMPLES = """examples:
  plot-gps-timeseries REYK --save pdf                 itrf2014 daily, nothing removed
  plot-gps-timeseries REYK --ref plate --save pdf     plate velocity removed
  plot-gps-timeseries REYK --frame itrf2020 --save pdf
                                                      the itrf2020 series instead
  plot-gps-timeseries RHOF --ref ref_frame --frame itrf2008 --special full --save pdf
                                                      whole series, nothing removed
  plot-gps-timeseries SENG --view detrended --save pdf
  plot-gps-timeseries SENG --session both -s 20250101 --save pdf
                                                      24 h + 8 h, itrf2020
  plot-gps-timeseries SENG --session 8h -u 20 --save pdf
  plot-gps-timeseries VFLN-VFLS --show-ends --save pdf
                                                      baseline + both ends
  plot-gps-timeseries VFLN-VFLS --baseline-sigma empirical --save pdf
  plot-gps-timeseries REYK --tType TOT --save pdf     the old TOT series
  plot-gps-timeseries REYK --name REYK-test --save png
                                                      one fixed file to keep open
Always pass --save (without it the plot opens interactively)."""

#: the reference frame plotted when neither --tType nor --frame is given
DEFAULT_PLOT_FRAME = "itrf2014"

#: ``--ref`` value for "nothing removed": the positions in the frame that
#: ``--frame`` picks.  Internally (plotTime, geo_dataread) this is still the
#: historical token ``"itrf2008"``, which the CLI also accepts as an alias.
REF_FRAME = "ref_frame"
_REF_FRAME_INTERNAL = "itrf2008"
_ITRF_REF_NAMES = ("itrf2014", "itrf2020", "itrf08", "itrf14", "itrf20")


def _ref_choice(value: str) -> str:
    """argparse type for ``--ref``: map the legacy alias, explain frame names."""
    v = value.strip().lower()
    if v == _REF_FRAME_INTERNAL:
        return REF_FRAME
    if v in _ITRF_REF_NAMES:
        raise argparse.ArgumentTypeError(
            f"{value!r} is a reference frame, not a --ref choice: use "
            f"--ref ref_frame --frame {value} (--frame picks the frame)"
        )
    return v


def _config_dir(config, option, default=None):
    """Resolve a directory from postprocess.cfg, or `default` if unavailable.

    Directories (figDir, totPath) live in [PATHS], with a legacy [Configs]
    fallback — that is `getPostProcessDir`. `getPostProcessConfig` is a
    different method for config *files* in [FILES]; calling it here is what
    broke the console script.

    This must not raise: it runs while argparse is being built, so a missing key
    (totPath is commented out in the deployed config) or no deployed config at
    all — a container image before gpsconfig lands — would otherwise kill even
    `--help`. Callers that genuinely need the value fail at use time instead.
    """
    try:
        return config.getPostProcessDir(option)
    except Exception:
        return default


#: ``--remove-steps`` aliases for the operator's own vocabulary.  ``kind``
#: values come from ``steps.yaml`` (``gps_parser.outlier_catalogs.STEP_KINDS``);
#: these are the friendly spellings of the same concept.
_KIND_ALIASES = {"instrumental": "equipment", "instrument": "equipment"}


def _normalize_kinds(raw: Sequence[str]) -> list[str] | None:
    """Alias + validate ``--remove-steps`` / ``--annotate-steps`` kinds.

    Each item in ``raw`` is split on comma so ``equipment,earthquake`` works
    alongside repeatable ``--remove-steps equipment --remove-steps earthquake``.
    Returns ``None`` when no kinds were given; otherwise a de-duplicated list
    of canonical ``STEP_KINDS`` values.  An unknown kind is a hard error.
    """
    if not raw:
        return None
    from gps_parser.outlier_catalogs import STEP_KINDS

    kinds: list[str] = []
    for item in raw:
        for part in item.split(","):
            part = part.strip()
            if not part:
                continue
            k = _KIND_ALIASES.get(part.lower(), part.lower())
            if k not in STEP_KINDS:
                raise SystemExit(
                    f"--remove-steps/--annotate-steps: kind {part!r} is not "
                    f"one of {STEP_KINDS} (aliases accepted: {sorted(_KIND_ALIASES)})"
                )
            if k not in kinds:
                kinds.append(k)
    return kinds or None
    return kinds


# Main
def main():
    """ """
    import gps_parser as cp

    config = cp.ConfigParser()
    # date to use defaults to today ----
    dstr = "%Y%m%d"  # Default input string

    # initialising  few variables
    start = end = None
    eventDict = {}
    Type_allow = ["TOT", "08h", "JOIN", "POS"]
    ref_allow = ["all", "plate", "detrend", REF_FRAME]
    special_allow = ["all", "90d", "year", "full", "fixedstart"]
    view_allow = ["raw", "cleaned", "detrended"]

    # Raw*Description* leaves the epilog alone (argument help is still
    # auto-wrapped), which is what lets the generated NAME table keep its
    # columns instead of being reflowed into a comma wall.
    parser = argparse.ArgumentParser(
        description="Plot GPS time series: one station, or a baseline AAAA-BBBB.\n\n"
        + textwrap.fill(
            "By default the daily .pos series in itrf2014 is read from "
            "pos_dir; --frame switches the reference frame (itrf2008, "
            "itrf2014, itrf2020), --ref says what is removed (ref_frame = "
            "nothing, plate, detrend) and --session picks 24 h, 8 h or both. "
            "Plate removal and the detrend parameters always follow the frame.",
            width=78,
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n\n".join(
            block
            for block in (
                EXAMPLES,
                outlier_param_help(),
                textwrap.fill(
                    "For any issues regarding this program or the GPS system "
                    "contact, Benni, gsm: 847 4985, email: bgo@vedur.is, or "
                    "Hildur email: hildur@vedur.is",
                    width=78,
                ),
            )
            if block
        ),
    )

    parser.add_argument(
        "Stations",
        nargs="+",
        help="stations (4-char codes), or 'all'. AAAA-BBBB plots the BASELINE "
        "AAAA minus BBBB (N/E/U differences on common epochs — days, or 8 h "
        "sessions within 2 h — re-zeroed at the first; sigma in quadrature, or "
        "--baseline-sigma empirical); it defaults to --ref plate and can be "
        "mixed with single stations, e.g. VFLN-VFLS VFLN",
    )
    parser.add_argument(
        "--events", nargs="+", default=None, help="list of individual events to add"
    )
    parser.add_argument(
        "--eventf",
        nargs="?",
        type=argparse.FileType("r"),
        default=None,
        const=sys.stdin,
        help="read in a list of events from a file: defaults to stdin",
    )
    parser.add_argument(
        "-s", "--start", type=str, default=None, help="start of the plot, YYYYMMDD"
    )
    parser.add_argument(
        "-e", "--end", type=str, default=None, help="end of the plot, YYYYMMDD"
    )
    parser.add_argument(
        "-D",
        type=int,
        default=0,
        help="Number of days to shift the given day positive subracts negativ adds",
    )
    parser.add_argument(
        "--save",
        type=str,
        nargs="?",
        default=None,
        const="eps",
        help="save figure to file(s): a format suffix or comma-separated "
        "formats, each written natively by matplotlib "
        "(e.g. 'png', 'pdf' or 'eps,pdf,png'); defaults to eps",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="fixed output basename (no directory, no suffix): the figure is "
        "written as <figDir>/<name>.<fmt>, bypassing the usual "
        "ref/view/period filename. Handy for re-rendering to one stable path "
        "you keep open in a viewer (e.g. --name RHOF-test). Every station and "
        "--ref/--special variant writes the SAME file, so only the last "
        "render survives a multi-variant run",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="plot strictly the period specified, regardless of the data",
    )
    parser.add_argument(
        "--ref",
        type=_ref_choice,
        default=None,
        choices=ref_allow,
        help="what is removed from the positions: ref_frame = nothing (the "
        "positions in the reference frame picked by --frame; the old name "
        "itrf2008 still works), "
        "plate = the station's plate velocity in that frame's plate-motion "
        "model, detrend = plate + the stored rate and seasonals (step offsets "
        "stay visible); all = one plot of each. Default: plate for a baseline "
        "(AAAA-BBBB) or --view detrended, ref_frame otherwise",
    )
    parser.add_argument(
        "--view",
        type=str,
        default="raw",
        choices=view_allow,
        help="Data view (geo_dataread apply-on-read toggle): raw (default), "
        "cleaned (outlier epochs masked from the main series; the grey "
        "overlay is OFF by default -- add --show-outliers to see what was "
        "set aside), detrended (subtract full trajectory including steps "
        "— unlike --ref detrend which keeps step offsets visible)",
    )
    parser.add_argument(
        "--show-ends",
        action="store_true",
        help="baselines only (AAAA-BBBB): also draw each station's own "
        "series behind the baseline, AAAA light grey and BBBB dark grey, "
        "both referenced to the baseline's first common epoch so the red "
        "baseline is their visible difference. Adds -ends to the file name",
    )
    parser.add_argument(
        "--baseline-sigma",
        choices=("formal", "empirical"),
        default="formal",
        help="baselines only: 'formal' (default) = quadrature of the two "
        "stations' GLOBK sigma; 'empirical' = realistic daily sigma from the "
        "fitted baseline noise record (shared daily error + station scale, "
        "geo_dataread.baseline_noise / gps-baseline-noise). Adds -empsig",
    )
    parser.add_argument(
        "--baseline-noise",
        default=None,
        help="path of the baseline noise record (default: "
        "$GPS_API_STORE/noise/baseline_noise.json)",
    )
    parser.add_argument(
        "--tType",
        type=str,
        default=None,
        choices=Type_allow,
        help='the series to read. Default (since 2026-10-13): "POS" in the '
        f"{DEFAULT_PLOT_FRAME} frame — the PBO .pos files of the daily SINEX "
        "archive (N/E/U from X/Y/Z; okada's junction offsets applied to the "
        "itrf2008/itrf2014 sets); --frame picks "
        'another frame. "TOT": the GLOBK mb_STA_TOT.dat[123] files (the '
        'previous default), "08h" their 8-hour series, "JOIN" both',
    )
    parser.add_argument(
        "--ref-position",
        dest="ref_position",
        default=None,
        choices=["wmean", "mean", "median", "first", "apriori", "tssum"],
        help="--tType POS: the reference position of the N/E/U, kept as the "
        "zero of the plot: least-squares mean (wmean), mean, median, first "
        "epoch, the station's a-priori coordinate (station_coord.xyz) or "
        "tssum's own columns. Default: zero on the first samples, as for TOT",
    )
    parser.add_argument(
        "--pos-id",
        dest="pos_id",
        default=None,
        help="--tType POS: which .pos product in pos_dir — a frame (as --frame, "
        "which is the simpler switch) or a full product id, e.g. imo.24hr_itrf14 "
        "(the .pos file names keep the short frame key: tssum's 15-character "
        "product id has no room for itrf2014)",
    )
    parser.add_argument(
        "--frame",
        default=None,
        help="reference frame of the .pos series: itrf2008 (okada, 1995 on), "
        "itrf2014 (IGb14: okada re-aligned, 1995 on; the DEFAULT) or itrf2020 "
        "(IGb20: gamit-p01, 2017 on; the only frame with 8 h sessions). Short "
        "names itrf08/itrf14/itrf20 and igb14/igb20 also work. "
        "Implies --tType POS; plate removal and the detrend parameters follow "
        "the frame (detrend_params[.<frame>].json)",
    )
    parser.add_argument(
        "--session",
        default="24h",
        choices=["24h", "8h", "both"],
        help="24h daily solutions (default), 8h sessions "
        "(<SITE>.imo.08hr_itrf20.pos: itrf2020 only, gamit-p01 2023 on, ~13 IMO "
        "stations; the frame defaults to itrf2020), or both: the 8h sessions "
        "(blue) behind the daily series (red), same frame, view and reference. "
        "8 h sigmas are larger: -u 10 drops the noisier sessions, -u 20 keeps "
        "more",
    )
    parser.add_argument(
        "--convention",
        default="geodetic",
        choices=["geodetic", "globk"],
        help="--tType POS: local frame of the N/E/U — geodetic (true local "
        "level, default) or globk (glorg Loc., the convention of the TOT files)",
    )
    parser.add_argument(
        "--ylim",
        type=int,
        default=[],
        nargs="+",
        help="set limit for y-axis: list of one or two numbers in mm. "
        + "One number x will expand the y axis by x mm in each direciton. "
        + "Two numbers will give absolute lower and upper boundary of the y-axis",
    )
    parser.add_argument(
        "-u",
        "--uncert",
        type=int,
        default=10,
        help="formal-uncertainty screen at read time [mm] — epochs with σ above "
        "this threshold are excluded from the series. Default 10 (the "
        "same screen as the detrend workbench; the batch estimator and "
        "geo_dataread.gps_read.getData default is 15)",
    )
    parser.add_argument(
        "--special",
        type=str,
        default=None,
        choices=special_allow,
        help="for routine plots: whole time series (all), One year long (year), ninety days (90d)",
    )
    parser.add_argument(
        "-d",
        "--figDir",
        type=str,
        nargs="?",
        default=None,
        const=None,
        help="figure save directory (default: fig_dir in postprocess.cfg "
        "[PATHS], deployed from gps-config-data; currently "
        f"{_config_dir(config, 'fig_dir', None) or 'unset -> current directory'}). "
        "Use -d . for the current directory.",
    )
    parser.add_argument(
        "-i",
        "--Dir",
        type=str,
        nargs="?",
        default=None,
        help="time series input directory (default: postprocess pos_dir for "
        "the .pos series, totPath for --tType TOT/08h/JOIN)",
    )
    parser.add_argument(
        "--outlier-param",
        action="append",
        metavar="NAME=VALUE",
        default=None,
        help="override one gps_analysis.OutlierParams detection threshold of "
        "the '--view cleaned' detector; repeatable (e.g. --outlier-param "
        "window_n_sigma=3.5 --outlier-param despike=true). Names/types are "
        "read off the dataclass: every valid NAME with its spec default is "
        "listed at the END of this help, and an unknown one lists them too. "
        "ANY override replaces the station's outlier_overrides.csv row "
        "wholesale; leave it unset to let that catalog (else the spec "
        "defaults) decide. Careful: min_outlier= is the SCALAR floor, a "
        "different knob from the per-component [N,E,U] vector the catalog "
        "carries. Ignored unless --view cleaned",
    )
    parser.add_argument(
        "--outlier-overrides",
        type=str,
        default=None,
        metavar="PATH",
        help="read per-station outlier overrides from this CSV instead of the "
        "deployed one -- the lever that reproduces a production cleaned "
        "view locally. Ignored unless --view cleaned",
    )
    parser.add_argument(
        "--stages",
        type=str,
        default=None,
        metavar="LIST",
        help="run only the named detection stages, e.g. --stages S0 or "
        "--stages S0,S4 ('all' restores everything). S0 gross-blunder "
        "despike, S3 global identifier, S4 windowed Hampel, S5 signal "
        "protection; S1 (robust fit) and S2 (whitening) are structural and "
        "always run. Named stages are ON, unnamed OFF. Isolation is an "
        "ATTRIBUTION tool -- with S5 off every candidate becomes a flag, "
        "which is never a production setting. Ignored unless --view cleaned",
    )
    parser.add_argument(
        "--show-outliers",
        action="store_true",
        help="overlay the grey outlier points on the --view cleaned figure. "
        "Off by default: the flagged epochs are masked from the series and "
        "the overlay is HIDDEN, so the plot shows only the cleaned series "
        "(and the y-axis tightens to it). Turning this on restores the "
        "evidence of what the detector set aside, at the cost of the "
        "y-axis stretching back out. Ignored unless --view cleaned",
    )
    parser.add_argument(
        "--provisional-days",
        type=float,
        default=None,
        metavar="DAYS",
        help="recency bound [days] of the PROVISIONAL marker (gold): recent "
        "candidates the detector could not rule on because the step evidence "
        "is indeterminate -- no data follows them yet, so a blunder and the "
        "onset of real deformation look identical. They stay IN the series; "
        "the marker only says the verdict is pending. 0 disables the marker; "
        "unset uses the geo_dataread default (14). Ignored unless "
        "--view cleaned",
    )
    parser.add_argument(
        "--remove-steps",
        action="append",
        default=[],
        metavar="KIND[,KIND...]",
        help="subtract the RECORDED step amplitudes (detrend_params.json) "
        "for declared steps of these kinds. Repeatable, and each argument "
        "accepts a comma-separated list "
        "(e.g. --remove-steps equipment,earthquake). "
        "Kind is the steps.yaml vocabulary: equipment (= instrument / "
        "antenna / receiver changes), earthquake, icing, manual, other "
        "('instrumental' is accepted as an alias for 'equipment'). Needs a "
        "committed detrend record for the station; otherwise it warns and "
        "plots the series unchanged. Pure apply, no re-fit; composable "
        "with --view cleaned (cleaning runs first, then step removal).  "
        "Independent of --ref detrend (which subtracts rate + seasonal "
        "only; they compose — rate+seasonal removal + selective step "
        "removal).  Redundant with --view detrended (which already "
        "subtracts the full trajectory including steps)",
    )
    parser.add_argument(
        "--annotate-steps",
        nargs="?",
        const="*",
        default=None,
        metavar="KIND[,KIND...]",
        help="draw the declared steps from steps.yaml on the figure: a "
        "solid vertical line at each step epoch plus a rotated label "
        "('kind: comment'), seismic dark-red, equipment dark-green. "
        "Optionally filter by kind: --annotate-steps (all kinds) or "
        "--annotate-steps equipment,earthquake (selected). Display-only "
        "and independent of --remove-steps",
    )
    parser.add_argument(
        "-t", action="store_true", help="join gamit pre and rap time series"
    )
    parser.add_argument(
        "-j",
        "--workers",
        type=int,
        default=None,
        help="number of parallel station-plotting processes when saving; "
        "defaults to min(cpu count, number of stations). Interactive "
        "mode (no --save) is always serial.",
    )

    args = parser.parse_args()

    # processing command line arguments --------------
    if args.fix or args.D:  # set the end date as yesterday
        end = currDatetime(-1)

    # defining sub-periods to plot, 'full'
    if args.start:  # start the plot at
        start = dt.datetime.strptime(args.start, dstr)
    if args.end:  # end the plot at
        end = dt.datetime.strptime(args.end, dstr)

    if args.D:
        start = currDatetime(days=-args.D, refday=end)

    if args.eventf:  # reading a list of events from a file or stdin
        eventDict.update(
            dict(
                [
                    [line.split(",")[0], line.split(",")[1:]]
                    for line in args.eventf.read().splitlines()
                ]
            )
        )

    if args.events:  # adding individual events from the command line
        eventDict.update(
            dict([[event.split(",")[0], event.split(",")[1:]] for event in args.events])
        )

    # preparing the args for plot function
    kwargs = vars(args)
    kwargs["start"] = start
    kwargs["end"] = end
    kwargs["events"] = eventDict
    # argparse dest 'outlier_param' (repeatable strings) -> plotTime's
    # 'outlier_params' (one OutlierParams, or None to defer to the catalog)
    # --stages sets the enable flags; an explicit --outlier-param still wins,
    # so the two compose rather than one silently overriding the other.
    stage_over = _stage_overrides(kwargs.pop("stages"))
    kwargs["outlier_params"] = _build_outlier_params(
        kwargs.pop("outlier_param"), base=stage_over
    )
    # plotTime's knob is ``hide_outliers`` (overlay drawn at the API by
    # default); the CLI inverts it -- the grey overlay is OFF unless
    # ``--show-outliers`` asks for it, so ``hide_outliers = not show``.
    kwargs["hide_outliers"] = not kwargs.pop("show_outliers")
    # ``--remove-steps`` kinds -> plotTime's ``remove_kinds`` (validated +
    # aliased); ``--annotate-steps`` -> ``True`` (all kinds), a list of
    # kinds (filtered), or ``False`` (the flag was not passed at all).
    kwargs["remove_kinds"] = _normalize_kinds(kwargs.pop("remove_steps"))
    annotate_raw = kwargs.pop("annotate_steps")
    if annotate_raw is None:
        kwargs["annotate_steps"] = False
    elif annotate_raw == "*":
        kwargs["annotate_steps"] = True
    else:
        kwargs["annotate_steps"] = _normalize_kinds([annotate_raw])

    # --frame/--session pick the .pos product; --pos-id stays as an alias.
    # With neither --tType nor a product, the default is the .pos series in
    # DEFAULT_PLOT_FRAME.
    frame, session = kwargs.pop("frame"), kwargs.pop("session")
    if args.tType is None:
        kwargs["tType"] = args.tType = "POS"
    if args.tType == "POS" and frame is None and kwargs.get("pos_id") is None:
        # the 8 h sessions exist only in itrf2020 (gamit-p01) so far
        frame = DEFAULT_PLOT_FRAME if session == "24h" else "itrf2020"
    if frame is not None or session != "24h":
        from geo_dataread.pos import pos_product

        if kwargs.get("pos_id") is not None:
            parser.error("give --frame/--session or --pos-id, not both")
        try:
            kwargs["pos_id"] = pos_product(
                frame, "24h" if session == "both" else session
            )
            if session == "both":
                kwargs["overlay_pos_id"] = pos_product(frame, "8h")
        except ValueError as exc:
            parser.error(str(exc))
        kwargs["tType"] = args.tType = "POS"

    if kwargs["figDir"] is None:
        kwargs["figDir"] = _config_dir(config, "fig_dir", "")

    if kwargs["Dir"] is None:
        kwargs["Dir"] = _config_dir(
            config, "pos_dir" if args.tType == "POS" else "totPath"
        )

    stations = args.Stations  # station list
    del kwargs["Stations"]
    del kwargs["eventf"]
    del kwargs["D"]

    if "all" in stations:  # geting a list of all the GPS stations
        stations = config.getStationInfo()

    # ------------------------

    if args.t:  # join GPS time series
        raise NotImplementedError(
            "--t (join GPS time series) is not implemented in this driver"
        )

    del kwargs["t"]

    # worker-count policy: interactive mode (no --save) shows figures in a
    # browser backend and stays in-process; batch mode defaults to one
    # process per station up to the core count.
    workers = kwargs.pop("workers")
    if not args.save:
        workers = 1
    elif workers is None:
        workers = min(os.cpu_count() or 1, len(stations))
    workers = max(1, workers)

    # plotting: expand --ref/--special "all" into the per-station variant
    # list.  The legacy loops were special-outer/station-inner, re-reading
    # each station's data files once per sub-period; station-outer order
    # lets plotStation read each station's raw series once and reuse it
    # across all its variants.
    if kwargs.get("ref") == REF_FRAME:
        kwargs["ref"] = _REF_FRAME_INTERNAL
    variants = _expand_variants(
        kwargs,
        [_REF_FRAME_INTERNAL if r == REF_FRAME else r for r in ref_allow[1:]],
        special_allow[1:],
    )

    # --name is a FIXED path by design (re-render into one open viewer), so
    # a fan-out silently overwrites itself; warn rather than auto-suffix.
    if args.name and (len(variants) > 1 or len(stations) > 1):
        print(
            "warning: --name %s is a fixed basename -- %d variant(s) x %d "
            "station(s) all write the same file; only the last render survives"
            % (args.name, len(variants), len(stations)),
            file=sys.stderr,
        )

    if workers > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(plotStation, sta, variants): sta for sta in stations}
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception:  # one bad station must not sink the batch
                    traceback.print_exc()
                    print(
                        "Unexpected error: %s during processing of %s"
                        % (sys.exc_info()[0], futures[future]),
                        file=sys.stderr,
                    )
    else:
        for sta in stations:
            plotStation(sta, variants)


if __name__ == "__main__":
    import signal

    # This is used to catch Ctrl-C exits
    original_sigint = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, exit_gracefully)

    main()
