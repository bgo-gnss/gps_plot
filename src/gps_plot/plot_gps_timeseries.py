#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse
import concurrent.futures
import dataclasses
import functools
import os
import sys
import textwrap
import traceback
import warnings

# Only the three helpers the MODERN lane calls are annotated below. This
# module is `ignore_errors` for mypy (see pyproject's typing policy), so
# these signatures type the SEAM, not the module: detrend_workbench is
# strict-checked and cannot call an untyped function.
from collections.abc import Mapping, Sequence
from typing import Any

# from gtimes.timefunc import TimefromYearf, currTime, TimetoYearf
from gtimes.timefunc import currDatetime

import gps_plot.timesmatplt as tplt
from gps_plot.cli import common

# gps_parser is imported lazily inside main() (it needs a deployed
# gpsconfig dir), so the module stays importable without it.


# functions

#: warnings that mean a plot would show something other than its file name
#: says (a raw series under a -cleaned or -detrend name, a "plate-removed"
#: series with no plate removed).  --strict turns them into failures before
#: anything is written; without it they stay warnings (interactive use).
DEGRADE_WARNINGS = (
    r".*serving raw series",
    # ...except too few points to run detection at all: nothing to clean
    r"(?!.*requires N >=).*serving unflagged data",
    r"Cannot determine plate velocity",
)


#: what --operational sets (argparse dest -> value), unless given explicitly:
#: the cleaned view at 3.5 sigma with the recent outliers and obstruction
#: days shown, strict (no degraded file) and the publish names (BGÓ 2026-10-21)
OPERATIONAL = {
    "view": "cleaned",
    "outlier_sigma": 3.5,
    "show_outliers": 10,
    "show_obstruction": 30,
    "show_network": 0,
    "strict": True,
    "legacy_names": True,
}
OPERATIONAL_FLAGS = (
    "--view cleaned",
    "--outlier-sigma 3.5",
    "--show-outliers 10",
    "--show-obstruction 30",
    "--show-network",
    "--strict",
    "--legacy-names",
)


def apply_operational(args, parser):
    """Fill :data:`OPERATIONAL` into ``args`` where the option is at its default."""
    for dest, value in OPERATIONAL.items():
        if getattr(args, dest) == parser.get_default(dest):
            setattr(args, dest, value)


@dataclasses.dataclass
class PlotOutcome:
    """What one (station, variant) render did — the batch contract's unit.

    ``status`` is ``"ok"`` (``files`` written), ``"skipped"`` (no data in
    the plotted period: a dead or down station, expected in a network run)
    or ``"failed"`` (``reason`` says why; nothing was written).
    """

    sta: str
    variant: str
    status: str
    reason: str = ""
    files: list[str] = dataclasses.field(default_factory=list)


def tryTimes(sta, strict=False, show_traceback=False, **kwargs):
    """Render one plot variant; never raise — return a :class:`PlotOutcome`.

    One bad station must not sink a batch, but a failure must not vanish
    either: it is one line on stderr and a ``failed`` outcome the summary
    counts.  ``strict`` refuses degraded output (:data:`DEGRADE_WARNINGS`).
    """
    variant = "%s/%s" % (kwargs.get("ref"), kwargs.get("special"))
    try:  # Trying to plot
        print("%s Plotting" % sta)
        with warnings.catch_warnings():
            if strict:
                for pattern in DEGRADE_WARNINGS:
                    warnings.filterwarnings("error", message=pattern)
            fig = tplt.plotTime(sta, **kwargs)
        print(
            "plotted %s using: %s, %s"
            % (sta, getattr(fig, "_gps_label", kwargs["ref"]), kwargs["special"])
        )
        files = list(getattr(fig, "_gps_files", []))
        if kwargs.get("save"):
            import matplotlib.pyplot as plt

            plt.close(fig)  # a worker renders many stations; don't keep them
        return PlotOutcome(sta, variant, "ok", files=files)
    except ValueError as exc:
        if not tplt.is_no_data(exc):
            return _failed(sta, variant, exc, show_traceback)
        print("skipped %s %s: %s" % (sta, variant, exc), file=sys.stderr)
        return PlotOutcome(sta, variant, "skipped", str(exc))
    except Exception as exc:  # noqa: BLE001 - reported, counted, exit status
        return _failed(sta, variant, exc, show_traceback)


def _failed(sta, variant, exc, show_traceback):
    if show_traceback:
        traceback.print_exception(exc)
    reason = "%s: %s" % (type(exc).__name__, exc)
    print("FAILED %s %s: %s" % (sta, variant, reason), file=sys.stderr)
    return PlotOutcome(sta, variant, "failed", reason)


def summarize(outcomes, manifest=None):
    """Print the batch summary, write the manifest; return the exit status.

    The manifest lists every file written, one absolute path per line,
    sorted — the publish step copies exactly these instead of globbing the
    figure directory.  Exit status 1 when any plot failed, else 0 (skips
    are not failures).
    """
    ok = [o for o in outcomes if o.status == "ok"]
    skipped = [o for o in outcomes if o.status == "skipped"]
    failed = [o for o in outcomes if o.status == "failed"]
    files = sorted({os.path.abspath(f) for o in ok for f in o.files})
    if manifest:
        with open(manifest, "w") as fh:
            fh.writelines(f + "\n" for f in files)
    print(
        "gpsplot ts: %d plotted (%d files), %d skipped (no data in period), "
        "%d failed" % (len(ok), len(files), len(skipped), len(failed)),
        file=sys.stderr,
    )
    for o in sorted(failed, key=lambda o: (o.sta, o.variant)):
        print("  failed %s %s: %s" % (o.sta, o.variant, o.reason), file=sys.stderr)
    return 1 if failed else 0


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


def plotStation(sta, variants, strict=False, show_traceback=False):
    """Render every requested (ref, special) plot variant for one station.

    This is the unit of (process-)parallel work: one station, all its
    sub-period/reference variants in sequence, so the raw-read cache turns
    the former once-per-variant file read into one read per station.
    Per-plot fault tolerance is unchanged (:func:`tryTimes`).
    """
    _install_raw_read_cache()
    return [
        tryTimes(sta, strict=strict, show_traceback=show_traceback, **kwargs)
        for kwargs in variants
    ]


def _expand_variants(kwargs, refs, specials):
    """Expand ``--ref``/``--special`` "all" into per-station plot kwargs.

    ``kwargs["ref"]``/``kwargs["special"]`` equal to ``"all"`` expand to
    the allowed values, and a comma list of specials to those listed (``refs`` outer, ``specials`` inner -- the legacy
    loop order); anything else passes through unchanged.  Always returns
    fresh dicts safe to ship to worker processes.
    """
    special = kwargs.get("special")
    if special and "," in special:  # --special 90d,year,full: one plot each
        listed = [list(specials) if s == "all" else [s] for s in special.split(",")]
        specials = list(dict.fromkeys(s for group in listed for s in group))
        kwargs = dict(kwargs, special="all")
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


def _model_variants(variants, model):
    """Give each variant its ``--model`` plot(s): both = curve then residual."""
    kinds = ["curve", "residual"] if model == "both" else [model]
    return [dict(v, ref=MODEL_REFS[k], model=k) for v in variants for k in kinds]


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
    # --stages alone: assignments is None (the flag was not given)
    for item in assignments or ():
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
    (
        "running (§15)",
        (
            "method",
            "running_half_window_days",
            "running_n_sigma",
            "running_min_count",
            "running_scale_days",
            "running_max_reach_days",
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


#: ``--outlier-sigma`` sets the decision threshold of the active method:
#: the running window's k, or the model method's windowed-Hampel k_w.
SIGMA_FIELD = {"running": "running_n_sigma", "model": "window_n_sigma"}
#: the cleaned view's detector when neither --outlier-method nor the
#: station's outlier_overrides.csv row names one (OutlierParams default)
DEFAULT_METHOD = "running"


def _method_overrides(method: str | None, sigma: float | None) -> dict[str, Any] | None:
    """``--outlier-method`` / ``--outlier-sigma`` -> OutlierParams overrides."""
    if method is None and sigma is None:
        return None
    out: dict[str, Any] = {}
    if method is not None:
        out["method"] = method
    if sigma is not None:
        if sigma <= 0.0:
            raise SystemExit("--outlier-sigma: must be > 0, got %s" % (sigma,))
        out[SIGMA_FIELD[method or DEFAULT_METHOD]] = sigma
    return out


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


EXAMPLES = """examples (gpsplot STA ... is short for gpsplot ts STA ...; plot-gps-timeseries
is the old name of the same command):
  gpsplot REYK --save pdf                      itrf2014 daily, plate velocity removed
  gpsplot REYK --ref ref_frame --save pdf      nothing removed
  gpsplot REYK --frame itrf2020 --save pdf     the itrf2020 series instead
  gpsplot RHOF --ref ref_frame --frame itrf2008 --special full --save pdf
                                               whole series, nothing removed
  gpsplot SENG --ref detrend --save pdf        rate + seasonals removed
  gpsplot SENG --ref detrend --detrend-terms periodic --save pdf
                                               seasonals only removed
  gpsplot SENG --ref detrend --remove-steps all --save pdf
                                               residuals (steps removed too)
  gpsplot HOFN --model both --save pdf         stored model over the plate
                                               series, and over the detrended
  gpsplot HOFN --model residual --detrend-terms periodic --model-terms linear,periodic --save pdf
                                               seasonal removed from the data,
                                               not from the line
  gpsplot SENG --view cleaned --show-outliers 5 --save pdf
                                               outliers in the last 5 days only
  gpsplot SENG --session both -s 20250101 --save pdf
                                               24 h + 8 h, itrf2020
  gpsplot SENG --session 8h -u 20 --save pdf
  gpsplot VFLN-VFLS --show-ends --save pdf     baseline + both ends
  gpsplot VFLN-VFLS --baseline-sigma empirical --save pdf
  gpsplot REYK --tType TOT --save pdf          the old TOT series
  gpsplot REYK --name REYK-test --save png     one fixed file to keep open
Always pass --save (without it the plot opens interactively)."""

# shared with gpsplot stack: cli/common.py (names kept for importers)
DEFAULT_PLOT_FRAME = common.DEFAULT_PLOT_FRAME
REF_FRAME = common.REF_FRAME
_REF_FRAME_INTERNAL = common.REF_FRAME_INTERNAL
_ITRF_REF_NAMES = common.ITRF_REF_NAMES
_ref_choice = common.ref_choice


_config_dir = common.config_dir


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
            if k == "all":
                kinds.extend(x for x in STEP_KINDS if x not in kinds)
                continue
            if k not in STEP_KINDS:
                raise SystemExit(
                    f"--remove-steps/--annotate-steps: kind {part!r} is not "
                    f"one of {STEP_KINDS} or 'all' (aliases accepted: "
                    f"{sorted(_KIND_ALIASES)})"
                )
            if k not in kinds:
                kinds.append(k)
    return kinds or None
    return kinds


def _rate_window(value: str) -> tuple[float, float]:
    """``START:END`` (YYYYMMDD) as a velocity-product window [yr], whole days."""
    import datetime as _dt

    from geo_dataread.velocity import window_bounds

    a, sep, b = value.partition(":")
    try:
        start = _dt.datetime.strptime(a, "%Y%m%d").date()
        end = _dt.datetime.strptime(b, "%Y%m%d").date()
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"{value!r}: use START:END (YYYYMMDD)"
        ) from None
    if not sep or end < start:
        raise argparse.ArgumentTypeError(f"{value!r}: END must not be before START")
    return window_bounds(start, end)


def _detrend_terms(raw: str | None) -> tuple[str, ...] | None:
    """Parse ``--detrend-terms``: comma list → canonical-order tuple."""
    if raw is None:
        return None
    from gps_plot.timesmatplt import DETREND_TERM_GROUPS

    want = {p.strip().lower() for p in raw.split(",") if p.strip()}
    bad = want - set(DETREND_TERM_GROUPS)
    if bad or not want:
        raise ValueError(
            f"--detrend-terms: {', '.join(sorted(bad)) or 'nothing given'}; "
            f"use any of {', '.join(DETREND_TERM_GROUPS)}"
        )
    return tuple(t for t in DETREND_TERM_GROUPS if t in want)


def _model_terms(raw: str | None) -> tuple[str, ...] | None:
    """Parse ``--model-terms``: comma list → canonical-order tuple."""
    if raw is None:
        return None
    from gps_plot.sources.gps import MODEL_TERM_GROUPS

    want = {p.strip().lower() for p in raw.split(",") if p.strip()}
    bad = want - set(MODEL_TERM_GROUPS)
    if bad or not want:
        raise ValueError(
            f"--model-terms: {', '.join(sorted(bad)) or 'nothing given'}; "
            f"use any of {', '.join(MODEL_TERM_GROUPS)}"
        )
    return tuple(t for t in MODEL_TERM_GROUPS if t in want)


# Main
TYPE_ALLOW = ["TOT", "08h", "JOIN", "POS"]
REF_ALLOW = ["all", "plate", "detrend", REF_FRAME]
SPECIAL_ALLOW = ["all", "90d", "year", "full", "fixedstart"]
VIEW_ALLOW = ["raw", "cleaned"]
MODEL_ALLOW = ["curve", "residual", "both"]
#: the ref each --model plot is drawn in (the record is plate-frame)
MODEL_REFS = {"curve": "plate", "residual": "detrend"}


def _load_config() -> Any:
    import gps_parser as cp

    return cp.ConfigParser()


def build_parser(
    parser: argparse.ArgumentParser | None = None, config: Any = None
) -> argparse.ArgumentParser:
    """Add the time-series options to ``parser`` (a new one if None).

    ``gpsplot ts`` passes its subparser here, ``plot-gps-timeseries``
    builds a standalone one, so both carry the same flags and help.
    """
    if config is None:
        config = _load_config()
    Type_allow = TYPE_ALLOW
    ref_allow = REF_ALLOW
    special_allow = SPECIAL_ALLOW
    view_allow = VIEW_ALLOW
    if parser is None:
        parser = argparse.ArgumentParser(prog="plot-gps-timeseries")

    # Raw*Description* leaves the epilog alone (argument help is still
    # auto-wrapped), which is what lets the generated NAME table keep its
    # columns instead of being reflowed into a comma wall.
    _set_parser_text(
        parser,
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
        help="stations (4-char codes), 'active' (the production set: "
        "stations.cfg stations that are not discontinued or "
        "inactive (passive included), in the Iceland box or --region, with a series), or 'all' "
        "(every station with a .pos series of the selected frame in the "
        "Iceland box, or in --region; for --tType TOT the stations.cfg "
        "list). AAAA-BBBB plots the BASELINE "
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
        "--quake",
        action="append",
        default=None,
        metavar="TIME,LAT,LON,MAG[,NAME]",
        help="mark an earthquake: a dashed line at TIME (ISO, e.g. 2026-10-19 "
        "or 2026-10-19T14:32) on every panel, labelled 'NAME M5.2, dist 34 km' "
        "(epicentral distance to the station, or to each end of a baseline). "
        "Repeatable. E.g. --quake 2026-10-19,64.63,-17.53,5.2,BB",
    )
    common.add_period_options(parser, specials=special_allow, multi=True)
    common.add_selection_options(parser)
    common.add_output_options(parser, config)
    common.add_series_options(
        parser,
        refs=ref_allow,
        ref_help="what is removed from the positions: ref_frame = nothing (the "
        "positions in the reference frame picked by --frame; the old name "
        "itrf2008 still works), "
        "plate = the station's plate velocity in that frame's plate-motion "
        "model, detrend = plate + the stored rate and seasonals (--detrend-terms; step offsets "
        "stay visible); all = one plot of each. Default: plate (with --model: "
        "the model's own ref)",
        views=view_allow,
        types=Type_allow,
        sessions=["24h", "8h", "both"],
        session_help="24h daily solutions (default), 8h sessions "
        "(<SITE>.imo.08hr_itrf20.pos: itrf2020 only, gamit-p01 2023 on, ~13 IMO "
        "stations; the frame defaults to itrf2020), or both: the 8h sessions "
        "(blue) behind the daily series (red), same frame, view and reference. "
        "8 h sigmas are larger: -u 10 drops the noisier sessions, -u 20 keeps "
        "more",
    )
    parser.add_argument(
        "--model",
        default=None,
        choices=MODEL_ALLOW,
        help="draw the stored detrend model as a line: curve = over the "
        "plate series (--ref plate), residual = over the detrended series "
        "(--ref detrend; the line is what the model still has in the data, "
        "e.g. the steps), both = one file of each. Solid in the record's fit "
        "window, dashed where extrapolated. Files end in -model / -resid. "
        "Stations only; leave --ref out (or give the matching one)",
    )
    parser.add_argument(
        "--model-terms",
        default=None,
        metavar="TERM[,TERM...]",
        help="with --model: draw exactly these model terms as the line, any "
        "of linear (offset + rate), periodic, steps, transient, independent "
        "of what --detrend-terms/--remove-steps take out of the data (e.g. "
        "the seasonal line over data with the seasonal removed). Default: "
        "what the model still has in the plotted data. Files add -line-<terms>",
    )
    parser.add_argument(
        "--rate",
        action="store_true",
        help="draw the velocity product's rate over the plotted window: a line "
        "with its slope and, per panel, v = rate ± σ mm/yr (honest σ: formal "
        "× the station's empirical k). The number gpsplot map velocity and "
        "gps-velocities give for the same ref and view (those default to "
        "plate-periodic, cleaned; this follows the plot): --ref plate → ref "
        "plate; --ref detrend → background (--detrend-terms periodic → "
        "plate-periodic). Files end -rate",
    )
    parser.add_argument(
        "--rate-window",
        action="append",
        default=None,
        type=_rate_window,
        metavar="START:END",
        help="a window of its own for --rate (YYYYMMDD:YYYYMMDD, both days "
        "in; repeatable, one colour each), e.g. the quiet stretch between two "
        "episodes; implies --rate",
    )
    parser.add_argument(
        "--rate-segments",
        default=None,
        metavar="SOURCE[,SOURCE]",
        help="one rate per span BETWEEN events over the plotted window: "
        "'steps' splits at the station's declared steps (steps.yaml), "
        "'events' at the events of the event catalog (events.yaml) whose "
        "region holds the station (--segment-kinds, --segment-cut); "
        "'steps,events' at both. Each line is labelled rate ± σ mm/yr at its "
        "middle (× = refused) and the table is printed. Seasonal removed "
        "with --ref detrend --detrend-terms periodic (plate-periodic); "
        "implies --rate. Files end -rate-seg<source>",
    )
    parser.add_argument(
        "--segment-kinds",
        default="dike,eruption",
        metavar="KIND[,KIND...]",
        help="--rate-segments events: the event kinds that split (default "
        "dike,eruption; also inflation, deflation, deep_inflation, swarm, "
        "earthquake)",
    )
    parser.add_argument(
        "--segment-cut",
        default="span",
        choices=("span", "onset"),
        help="--rate-segments events: take each event out from its onset to "
        "its end (span, default: an eruption of weeks is its own regime) or "
        "its onset day only (onset: the spans run through the eruptions)",
    )
    parser.add_argument(
        "--segment-min-days",
        type=int,
        default=30,
        metavar="DAYS",
        help="--rate-segments: shortest span that gets a rate (default 30)",
    )
    parser.add_argument(
        "--events-catalog",
        default=None,
        metavar="PATH",
        help="--rate-segments events: the event catalog (default: the "
        "deployed events.yaml)",
    )
    parser.add_argument(
        "--detrend-terms",
        default=None,
        metavar="TERM[,TERM...]",
        help="with --ref detrend: which stored terms to subtract, any "
        "combination of linear (offset + rate), periodic (annual + "
        "semi-annual) and transient (if the record has one). Default "
        "linear,periodic. Steps are not terms here: --remove-steps removes "
        "them. Non-default terms are named in the title and the file name",
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
        "--convention",
        default="geodetic",
        choices=["geodetic", "globk"],
        help="--tType POS: local frame of the N/E/U — geodetic (true local "
        "level, default) or globk (glorg Loc., the convention of the TOT files)",
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
        "--outlier-method",
        choices=["model", "running"],
        default=None,
        help="the '--view cleaned' detector: running (default) = prior-free "
        "running window -- each day judged against robust local lines over "
        "the 10 days before and the 10 days after it, in units of the local "
        "noise; no steps, events or model needed, so it does not give up on "
        "deforming stations (gps_analysis design §15). model = against one "
        "robust trajectory fit of the plotted span, using declared steps "
        "(the pre-2026-10-19 detector; adds -modeldetect to the file name; "
        "--stages implies it). Per station: method in outlier_overrides.csv. "
        "Like --outlier-param, it replaces the station's override row. "
        "Ignored unless --view cleaned",
    )
    parser.add_argument(
        "--outlier-sigma",
        type=float,
        default=None,
        metavar="K",
        help="the outlier tolerance in sigma (local noise units) of the "
        "chosen --outlier-method: running_n_sigma (default 4.5) or, for "
        "model, the windowed identifier's window_n_sigma (default 4). Lower "
        "flags more; below ~2.5 the noise itself exceeds the 5%% abort "
        "(max_flag_fraction). Not in the file name, so automatic graphs keep "
        "theirs. Per station: running_n_sigma in outlier_overrides.csv. "
        "Ignored unless --view cleaned",
    )
    parser.add_argument(
        "--icing",
        dest="icing_mode",
        choices=["remove", "keep", "off"],
        default=None,
        help="days with icing suspected (GAMIT autcln index, harvested by "
        "gps-autcln-harvest into pos_dir; none there = none): remove "
        "(default) = taken out like outliers; keep = left in the series "
        "(they are suspected, not decided -- e.g. for automatic runs); "
        "off = ignored. Either way they never reach the outlier detector; "
        "--show-icing draws them. Ignored unless --view cleaned",
    )
    parser.add_argument(
        "--no-icing",
        dest="icing",
        action="store_false",
        help="same as --icing off",
    )
    parser.add_argument(
        "--show-icing",
        type=int,
        nargs="?",
        const=0,
        default=None,
        metavar="N",
        help="draw the icing days (light blue), like --show-outliers does the "
        "outliers and independent of it: --show-icing alone all of them, "
        "--show-icing N only those among the last N data points. Off by "
        "default (the note still counts them). Ignored unless --view cleaned",
    )
    parser.add_argument(
        "--snow",
        dest="snow_mode",
        choices=["remove", "keep", "off"],
        default=None,
        help="days with the antenna suspected buried in snow (Up lifted for "
        "weeks, horizontals quiet, GAMIT noise raised evenly; "
        "geo_dataread.snow_burial on the autcln harvest + the station's .pos): "
        "remove (default) / keep / off as --icing, for Up only. "
        "--show-snow draws them. Ignored unless --view cleaned",
    )
    parser.add_argument(
        "--no-snow",
        dest="snow",
        action="store_false",
        help="same as --snow off",
    )
    parser.add_argument(
        "--show-snow",
        type=int,
        nargs="?",
        const=0,
        default=None,
        metavar="N",
        help="draw the snow days (pale lilac, Up), as --show-icing: alone all "
        "of them, N only those among the last N data points",
    )
    parser.add_argument(
        "--network",
        dest="network_mode",
        choices=["remove", "keep", "off"],
        default=None,
        help="network days: the whole network solution bad that day "
        "(gps-network-days writes network_days.itrf14.csv into pos_dir; none "
        "there = none): remove (default) = taken out of every component; "
        "keep = left in; off = ignored. Never reach the outlier detector; "
        "--show-network draws them dark grey. Ignored unless --view cleaned",
    )
    parser.add_argument(
        "--no-network",
        dest="network",
        action="store_false",
        help="same as --network off",
    )
    parser.add_argument(
        "--show-network",
        type=int,
        nargs="?",
        const=0,
        default=None,
        metavar="N",
        help="draw the network days (dark grey), as --show-icing: alone all of "
        "them, N only those among the last N data points",
    )
    parser.add_argument(
        "--common-mode",
        dest="common_mode",
        choices=["subtract", "off"],
        default="subtract",
        help="the daily network common mode (gps-network-days writes "
        "network_cm.itrf14.csv into pos_dir from the stations outside the "
        "volcanic areas): subtract (default; okada 24 h itrf08/itrf14 series "
        "only, before outlier detection) or off",
    )
    parser.add_argument(
        "--obstruction",
        choices=["remove", "keep", "off"],
        default=None,
        help="icing AND snow together (antenna obstruction): sets both "
        "--icing and --snow; an explicit --icing/--snow still wins",
    )
    parser.add_argument(
        "--show-obstruction",
        type=int,
        nargs="?",
        const=0,
        default=None,
        metavar="N",
        help="--show-icing and --show-snow together (an explicit one wins)",
    )
    parser.add_argument(
        "--icing-ratio",
        type=float,
        default=None,
        metavar="R",
        help="icing threshold: the normalised azimuth mean over the station's "
        "own low level (default 2.0). Lower flags more. Not in the file name. "
        "Ignored unless --view cleaned",
    )
    parser.add_argument(
        "--show-outliers",
        type=int,
        nargs="?",
        const=0,
        default=None,
        metavar="N",
        help="overlay the outlier points on the --view cleaned figure. "
        "Off by default: the flagged epochs are masked from the series and "
        "the overlay is HIDDEN, so the plot shows only the cleaned series "
        "(and the y-axis tightens to it). --show-outliers alone draws all "
        "of them (grey); --show-outliers N only those among the last N "
        "data points -- what was set aside recently, without stretching the "
        "y-axis for old ones. Ignored unless --view cleaned",
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
        "'all' = every declared kind. "
        "Kind is the steps.yaml vocabulary: equipment (= instrument / "
        "antenna / receiver changes), earthquake, icing, manual, other "
        "('instrumental' is accepted as an alias for 'equipment'). Needs a "
        "committed detrend record for the station; otherwise it warns and "
        "plots the series unchanged. Pure apply, no re-fit; composable "
        "with --view cleaned (cleaning runs first, then step removal).  "
        "Independent of --ref detrend (which subtracts rate + seasonal "
        "only; they compose — rate+seasonal removal + selective step "
        "removal): --ref detrend --remove-steps all removes the whole "
        "stored trajectory, leaving the residuals",
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
    parser.add_argument(
        "--strict",
        action="store_true",
        help="production mode: refuse (write nothing, count as failed) any "
        "plot that would degrade — a raw series under a -cleaned/-detrend "
        "name, a plate-removed series without a plate. Without it these "
        "stay warnings",
    )
    parser.add_argument(
        "--manifest",
        metavar="FILE",
        default=None,
        help="write the absolute path of every file written, one per line "
        "(sorted) — publish exactly these instead of globbing the figure "
        "directory",
    )
    parser.add_argument(
        "--operational",
        action="store_true",
        help="the operational products in one flag: "
        + " ".join(OPERATIONAL_FLAGS)
        + ". Any of these given explicitly (with a non-default value) wins",
    )
    parser.add_argument(
        "--legacy-names",
        action="store_true",
        help="publish file names: leave the frame out ({STA}-plate-90d.png, "
        "as brunnur/cdn and their hotlinks expect); 8 h products keep -8h. "
        "Only for one frame per figure directory, or the names collide",
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="leave out (one line, not a failure) stations without a .pos "
        "series of the selected frame/session — for area lists, where most "
        "stations have no 8 h series",
    )
    parser.add_argument(
        "--traceback",
        action="store_true",
        help="print the full traceback of a failed plot (default: one line)",
    )

    return parser


def _set_parser_text(parser, description, formatter_class, epilog):
    parser.description = description
    parser.formatter_class = formatter_class
    parser.epilog = epilog


def _resolve_obstruction(kwargs: dict[str, Any], parser: Any) -> None:
    """--icing/--snow (+ --show-*) with the --obstruction umbrella under them.

    An explicit per-kind flag wins over the umbrella, the umbrella over the
    default (remove, not drawn). "off" (or --no-icing/--no-snow) turns the
    index off: no days kept out of detection, none flagged, none drawn.
    """
    umbrella = kwargs.pop("obstruction")
    show_umbrella = kwargs.pop("show_obstruction")
    for kind in ("icing", "snow"):
        mode = kwargs[f"{kind}_mode"] or umbrella or "remove"
        if mode == "off" or not kwargs[kind]:
            kwargs[kind], mode = False, "remove"
        kwargs[f"{kind}_mode"] = mode
        show = kwargs.pop(f"show_{kind}")
        if show is None:
            show = show_umbrella
        if show is not None and show < 0:
            parser.error(f"--show-{kind} N: N must be positive")
        kwargs[f"hide_{kind}"] = show is None
        kwargs[f"{kind}_tail"] = show or None
    kwargs["common_mode"] = kwargs.get("common_mode", "subtract") != "off"
    # network days: no umbrella, same rules
    mode = kwargs["network_mode"] or "remove"
    if mode == "off" or not kwargs["network"]:
        kwargs["network"], mode = False, "remove"
    kwargs["network_mode"] = mode
    show = kwargs.pop("show_network")
    if show is not None and show < 0:
        parser.error("--show-network N: N must be positive")
    kwargs["hide_network"] = show is None
    kwargs["network_tail"] = show or None


def run(
    args: argparse.Namespace, parser: argparse.ArgumentParser, config: Any = None
) -> int:
    """Plot what the parsed ``args`` ask for; return the exit status (1 = a plot failed)."""
    if config is None:
        config = _load_config()
    if args.operational:
        apply_operational(args, parser)
    eventDict = {}
    ref_allow = REF_ALLOW
    special_allow = SPECIAL_ALLOW

    # processing command line arguments --------------
    start, end = common.resolve_dates(args, currDatetime)

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
    from gps_plot.layers.quakes import parse_quake

    try:
        kwargs["quakes"] = [parse_quake(q) for q in kwargs.pop("quake") or ()]
    except ValueError as err:
        parser.error(str(err))
    # argparse dest 'outlier_param' (repeatable strings) -> plotTime's
    # 'outlier_params' (one OutlierParams, or None to defer to the catalog)
    # --stages sets the enable flags; an explicit --outlier-param still wins,
    # so the two compose rather than one silently overriding the other.
    stage_over = _stage_overrides(kwargs.pop("stages"))
    method = kwargs.pop("outlier_method")
    if method is None and stage_over is not None:
        method = "model"  # the S0-S5 stages exist only in the model detector
    method_over = _method_overrides(method, kwargs.pop("outlier_sigma"))
    base = {**(stage_over or {}), **(method_over or {})} or None
    kwargs["outlier_params"] = _build_outlier_params(
        kwargs.pop("outlier_param"), base=base
    )
    # plotTime's knob is ``hide_outliers`` (overlay drawn at the API by
    # default); the CLI inverts it -- the grey overlay is OFF unless
    # ``--show-outliers`` asks for it, so ``hide_outliers = not show``.
    _resolve_obstruction(kwargs, parser)
    show_outliers = kwargs.pop("show_outliers")
    kwargs["hide_outliers"] = show_outliers is None
    if show_outliers is not None and show_outliers < 0:
        parser.error("--show-outliers N: N must be positive")
    kwargs["outlier_tail"] = show_outliers or None
    # ``--remove-steps`` kinds -> plotTime's ``remove_kinds`` (validated +
    # aliased); ``--annotate-steps`` -> ``True`` (all kinds), a list of
    # kinds (filtered), or ``False`` (the flag was not passed at all).
    kwargs["remove_kinds"] = _normalize_kinds(kwargs.pop("remove_steps"))
    try:
        kwargs["detrend_terms"] = _detrend_terms(kwargs["detrend_terms"])
    except ValueError as exc:
        parser.error(str(exc))
    kwargs["rate_windows"] = kwargs.pop("rate_window")
    if kwargs["rate_segments"] is not None:
        if kwargs["rate_windows"]:
            parser.error("--rate-segments and --rate-window: give one")
        try:
            from gps_plot.layers.rate import parse_segment_sources

            kwargs["rate_segments"] = parse_segment_sources(kwargs["rate_segments"])
        except ValueError as exc:
            parser.error(str(exc))
        kwargs["segment_kinds"] = tuple(
            k.strip() for k in kwargs["segment_kinds"].split(",") if k.strip()
        )
        kwargs["rate"] = True
    if kwargs.get("ref") is None and kwargs.get("model") is None:
        kwargs["ref"] = "plate"  # the default since 2026-10-19
    if kwargs["rate"] or kwargs["rate_windows"]:
        if kwargs["session"] != "24h":
            parser.error("--rate: daily series only (no --session 8h/both)")
        if kwargs.get("ref") in (None, "all"):
            parser.error("--rate needs one --ref: plate or detrend")
    model = kwargs.pop("model")
    if model is not None:
        wanted = MODEL_REFS.get(model)
        if kwargs.get("ref") is not None and kwargs.get("ref") != wanted:
            parser.error(
                "--model %s draws over --ref %s; leave --ref out"
                % (model, wanted or "plate and detrend")
            )
        from geo_dataread.baseline import split_baseline

        if any(split_baseline(s) for s in args.Stations):
            parser.error("--model: baselines (AAAA-BBBB) are not supported yet")
    try:
        kwargs["model_terms"] = _model_terms(kwargs["model_terms"])
    except ValueError as exc:
        parser.error(str(exc))
    if kwargs["model_terms"] is not None and model is None:
        parser.error("--model-terms needs --model")
    if kwargs["detrend_terms"] is not None and not (
        kwargs.get("ref") in ("detrend", "all") or model in ("residual", "both")
    ):
        parser.error("--detrend-terms needs --ref detrend")
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
    if model is not None and kwargs["session"] != "24h":
        parser.error("--model: daily series only (no --session 8h/both) for now")
    common.resolve_source(kwargs, args, parser, config)

    stations = args.Stations  # station list
    del kwargs["Stations"]
    del kwargs["eventf"]
    del kwargs["D"]

    region, with_record = kwargs.pop("region"), kwargs.pop("with_record")
    skip_missing = kwargs.pop("skip_missing")
    stations = common.resolve_stations(
        stations,
        kwargs,
        config,
        region=region,
        with_record=with_record,
        skip_missing=skip_missing,
    )
    if not stations:
        if skip_missing:  # nothing of this product here: not an error
            print("gpsplot ts: no station with a series of this product; nothing to do")
            return 0
        parser.error("no station left to plot after 'all'/--region/--with-record")

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
    strict = kwargs.pop("strict")
    kwargs.pop("operational")
    show_traceback = kwargs.pop("traceback")
    manifest = kwargs.pop("manifest")
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
    if model is not None:
        variants = _model_variants(variants, model)

    # --name is a FIXED path by design (re-render into one open viewer), so
    # a fan-out silently overwrites itself; warn rather than auto-suffix.
    if args.name and (len(variants) > 1 or len(stations) > 1):
        print(
            "warning: --name %s is a fixed basename -- %d variant(s) x %d "
            "station(s) all write the same file; only the last render survives"
            % (args.name, len(variants), len(stations)),
            file=sys.stderr,
        )

    batch = {"strict": strict, "show_traceback": show_traceback}
    outcomes = []
    if workers > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(plotStation, sta, variants, **batch): sta
                for sta in stations
            }
            for future in concurrent.futures.as_completed(futures):
                try:
                    outcomes.extend(future.result())
                except Exception as exc:  # a worker died: count it, go on
                    sta = futures[future]
                    reason = "%s: %s" % (type(exc).__name__, exc)
                    print("FAILED %s: %s" % (sta, reason), file=sys.stderr)
                    outcomes.append(PlotOutcome(sta, "*", "failed", reason))
    else:
        for sta in stations:
            outcomes.extend(plotStation(sta, variants, **batch))
    return summarize(outcomes, manifest)


def main(argv: Sequence[str] | None = None) -> int:
    """``plot-gps-timeseries``: the same as ``gpsplot ts`` (returns the exit status)."""
    config = _load_config()
    parser = build_parser(config=config)
    return run(parser.parse_args(argv), parser, config) or 0


if __name__ == "__main__":
    import signal

    # This is used to catch Ctrl-C exits
    original_sigint = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, exit_gracefully)

    sys.exit(main())
