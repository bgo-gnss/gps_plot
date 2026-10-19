#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Static matplotlib production plots for GPS displacement time series.

Public functions:
   init_plot_style(force=False)      -- apply the classic/usetex style ONCE
   make_title(sta, lastData, ...)    -- typed StationTitle (semantic status color)
   makelatexTitle(sta, lastData,...) -- legacy single-string title (EPS/dvips only)
   stdFrame(Ylabel=None, Title=None, fig=None)
   stdTimesPlot(x, y, Dy, ...)       -- build (and return) the 3-component Figure
   plotTime(sta, ...)                -- read data, build figure, save/show
   addData / addPoints / addEvent
   saveFig(fileName, fType, fig)     -- native multi-format export (eps/pdf/png/...)
   setXlim / setYlim / tsTickLabels / tslabels
   convIcelCh / convLatex

   Under construction:
       fancytitle(x, y, titlestr, font, ax, **kw)

Modernization notes (static-plot-modernize, Path B):

- Export is a native ``Figure.savefig`` per requested suffix (eps, pdf, png,
  ...); the old ImageMagick ``convert`` EPS->PNG subprocess is gone.
- The semantic red/green "Last datapoint" header is now a matplotlib-level
  text color on a usetex fragment (see :func:`make_title` /
  :func:`_add_status_subtitle`).  The old single-string
  ``\\textcolor{...}{...}`` title only survived the PS/dvips pipeline --
  matplotlib's own DVI reader (used by the Agg/PNG and PDF backends)
  ignores LaTeX color specials, which is why PNG/PDF used to lose the
  color.  Coloring the fragment at the matplotlib level renders
  identically on every backend while keeping the exact Computer-Modern
  usetex look.
- Style/rcParams (``classic`` + usetex) are applied once per process by
  :func:`init_plot_style` instead of per figure, and the production save
  path reuses a single non-pyplot Figure across stations.
"""

from __future__ import annotations

__author__ = "Benedikt G. Ofeigsson <bgo@vedur.is>"
__date__ = "$Date: Feb 2016"
__version__ = "$Revision: 0.2 $"[11:-2]

import dataclasses
import datetime
import os
import warnings
from collections.abc import Mapping, Sequence
from datetime import timedelta
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.figure
import matplotlib.image as image
import matplotlib.pyplot as plt
import matplotlib.style
import numpy as np
from gtimes.timefunc import currDate, currDatetime, currTime, currYearfDate, toDatetimel

# The baseline layer (parsing, matching, differencing, the referenced ends,
# the mixed-view check) is numpy-only and lives in geo_dataread so every
# consumer forms a baseline the same way; gps_plot.sources.gps calls it and
# this module only draws the result.
from geo_dataread.baseline import split_baseline
from matplotlib import transforms
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea

#: Semantic status colors of the "Last datapoint" header.  These match the
#: LaTeX ``color`` package values the legacy ``\textcolor{green|red}`` title
#: produced under dvips (pure RGB green/red -- NOT matplotlib's ``"green"``,
#: which is (0, 0.5, 0)).
STATUS_CURRENT_COLOR: tuple[float, float, float] = (0.0, 1.0, 0.0)
STATUS_STALE_COLOR: tuple[float, float, float] = (1.0, 0.0, 0.0)

#: Outlier-overlay colors of the ``cleaned`` view.  Flagged epochs are drawn
#: de-emphasised in grey (error bars grey, dark-grey outline, light-grey
#: fill) so they read as "set aside" instead of competing with the main
#: series the way the earlier red overlay did.
OUTLIER_ERRORBAR_COLOR: str = "grey"
OUTLIER_EDGE_COLOR: str = "dimgrey"
OUTLIER_FACE_COLOR: str = "lightgrey"

#: PROVISIONAL epochs -- flagged by the identifiers but protected because the
#: step evidence is indeterminate, and recent enough that data has not yet
#: settled it (``gps_views.detect_view_outliers`` provenance).  Deliberately
#: neither red (kept) nor grey (removed): the point IS in the series, and the
#: marker says the verdict is pending, not made.  Gold reads as "caution"
#: without colliding with the red/grey pair or the green last-point highlight.
PROVISIONAL_ERRORBAR_COLOR: str = "darkgoldenrod"
PROVISIONAL_EDGE_COLOR: str = "darkgoldenrod"
PROVISIONAL_FACE_COLOR: str = "gold"
#: icing suspected (GAMIT autcln index): removed like an outlier, but its own
#: colour — ice on the antenna, not a blunder
ICING_ERRORBAR_COLOR: str = "lightsteelblue"
ICING_EDGE_COLOR: str = "steelblue"
ICING_FACE_COLOR: str = "#e8f4fd"  # very, very light blue (BGÓ)
#: a provisional LAST point (yesterday's, when the data are current): the
#: gold "verdict pending" face keeps a green outline, so the newest day stays
#: recognisable instead of the yellow marker hiding the green highlight.
PROVISIONAL_LAST_EDGE_COLOR: str = "limegreen"
PROVISIONAL_LAST_EDGE_WIDTH: float = 2.0

#: ABORT badge.  A component whose detection aborted is served RAW with only
#: a UserWarning, so an aborted axis is otherwise indistinguishable from a
#: clean one -- the worst failure mode for a monitoring product.  Red on a
#: pale wash, per axis (the abort is per component, §3.5a).
ABORT_BADGE_TEXT: str = r"DETECTION ABORTED --- raw shown"
ABORT_BADGE_COLOR: str = "red"
ABORT_BADGE_FACE: str = "mistyrose"

#: Raster resolution for PNG export.  Matches the legacy ImageMagick
#: ``convert -density 90`` EPS->PNG conversion, so direct-PNG output keeps
#: the accustomed pixel dimensions.
PNG_DPI: float = 90.0

_STYLE_INITIALIZED: bool = False
_FRAME_FIG: Figure | None = None


def init_plot_style(force: bool = False) -> None:
    """Apply the production plot style (classic + usetex) once per process.

    The legacy code re-ran ``mpl.style.use("classic")`` and reset the usetex
    rcParams inside every ``stdFrame`` call; this is now done a single time
    (``force=True`` re-applies, e.g. after another module changed rcParams).
    """
    global _STYLE_INITIALIZED
    if _STYLE_INITIALIZED and not force:
        return

    mpl.style.use("classic")
    mpl.rcParams["legend.handlelength"] = 0
    mpl.rcParams["text.usetex"] = True
    # NOTE: the legacy code called mpl.rc("text.latex", preamble=...) twice,
    # so only the second line (color) was actually active; both are kept here.
    mpl.rcParams["text.latex.preamble"] = "\n".join(
        [r"\usepackage[utf8]{inputenc}", r"\usepackage{color}"]
    )
    mpl.rc("font", family="Times New Roman")

    _STYLE_INITIALIZED = True


def _reusable_figure() -> Figure:
    """Module-level Figure reused across stations on the save path.

    A plain (non-pyplot) Figure with an Agg canvas: no GUI window manager,
    no pyplot figure accumulation, cleared and rebuilt per station by
    :func:`stdFrame`.
    """
    global _FRAME_FIG
    if _FRAME_FIG is None:
        init_plot_style()  # Figure snapshots rcParams (dpi, subplotpars)
        _FRAME_FIG = matplotlib.figure.Figure(figsize=(13, 20))
        FigureCanvasAgg(_FRAME_FIG)
    return _FRAME_FIG


def data_is_current(
    last: datetime.datetime | datetime.date | np.datetime64, warnp: int = -1
) -> bool:
    """True if ``last`` falls on the reference day ``currDate(warnp)``.

    This is THE semantic red/green decision: ``warnp=-1`` means "data
    through yesterday counts as current" (green); anything older is stale
    (red).  Used by both the title status color and the highlighted last
    data point so the two can never disagree.
    """
    if isinstance(last, np.datetime64):
        last = last.astype("datetime64[us]").item()
    if isinstance(last, datetime.datetime):
        last = last.date()
    return last == currDate(warnp)


@dataclasses.dataclass(frozen=True)
class StationTitle:
    """Title of a standard GPS time-series plot, split for rendering.

    ``main`` becomes the (all-black) suptitle; ``status`` is rendered in
    ``status_color`` next to the black ``created`` fragment (the split is
    what lets every backend color the status correctly -- see module notes).
    """

    main: str
    status: str
    created: str
    status_color: tuple[float, float, float]


def baseline_file_stem(sta: str) -> str:
    """Filename stem: ``VFLN_VFLS-baseline`` for a baseline, else ``sta``.

    Not ``VFLN-VFLS``: the publish globs and the ``{STA}-{ref}-…`` naming
    convention split on ``-``, so a hyphenated pair would read as station
    ``VFLN`` with reference frame ``VFLS``.
    """
    pair = split_baseline(sta)
    return f"{pair[0]}_{pair[1]}-baseline" if pair else sta


def make_title(
    sta: str,
    lastData: datetime.datetime,
    ref: str = "ITRF2008",
    warnp: int = -1,
) -> StationTitle:
    """Create the standard plot title with the semantic status color.

    The status fragment ("Last datapoint: ...") is green when the last data
    point is current (see :func:`data_is_current`), red otherwise.
    """
    lastpoint = currDate(refday=lastData, String="Last datapoint: %d %b %Y")
    status_color = (
        STATUS_CURRENT_COLOR if data_is_current(lastData, warnp) else STATUS_STALE_COLOR
    )

    timeofPlot = currTime("(Plot created on %b %d %Y %H:%M %Z)")

    def full_name(marker: str) -> str:
        # station full name from the station config, marker as fallback
        try:
            import gps_parser as cp

            name = cp.ConfigParser().get_config(marker, "station_name")
        except Exception:
            name = marker
        return "%s (%s)" % (convLatex(name), marker)

    pair = split_baseline(sta)
    if pair:
        # a baseline names both ends by MARKER only: two full station names
        # overflow the title line (BGÓ 2026-10-08, VFLN–VFLS)
        NameStr = "%s -- %s" % pair
        refFr = "Baseline, reference frame: %s" % ref
    else:
        NameStr = full_name(sta)
        refFr = "Reference frame: %s" % ref
    if sta == "hekla":
        NameStr = "Hekla (Summit)"
    if ref == "Multigas":
        refFr = "%s: %s" % (ref, "Uncorrected")

    return StationTitle(
        main=r"\Huge %s \ \Large %s" % (NameStr, refFr),
        status=r"\LARGE %s" % lastpoint,
        created=r"\large %s" % timeofPlot,
        status_color=status_color,
    )


def makelatexTitle(
    sta: str,
    lastData: datetime.datetime,
    ref: str = "ITRF2008",
    warnp: int = -1,
) -> list[str]:
    """Legacy single-string LaTeX title (``\\textcolor``-based).

    Kept for callers that render through the PS/dvips pipeline only
    (e.g. ``gasmatplt``): the embedded ``\\textcolor`` is IGNORED by the
    Agg/PNG and PDF backends.  New code should use :func:`make_title`.
    """
    title = make_title(sta, lastData, ref=ref, warnp=warnp)
    dcolor = "green" if title.status_color == STATUS_CURRENT_COLOR else "red"

    name_str, ref_str = title.main.split(r" \ \Large ")
    name_str = name_str.replace(r"\Huge ", "")
    status_str = title.status.replace(r"\LARGE ", "")
    created_str = title.created.replace(r"\large ", "")

    return [
        r"\Huge\textcolor{black}{%s} \  \Large\textcolor{black}{%s}  "
        % (name_str, ref_str),
        r"\LARGE\textcolor{%s}{%s} \  \large\textcolor{black}{%s}  "
        % (dcolor, status_str, created_str),
    ]


def _add_status_subtitle(ax: Any, title: StationTitle) -> None:
    """Center the colored status + black created fragments above ``ax``.

    Replaces the legacy ``ax.set_title(latex_string, y=1.01)``: the two
    fragments keep their LaTeX sizes (``\\LARGE`` / ``\\large``) while the
    status color is applied at the matplotlib level so it renders
    identically in EPS, PDF and PNG.
    """
    status = TextArea(title.status, textprops={"color": title.status_color})
    created = TextArea(title.created, textprops={"color": "black"})
    row = HPacker(children=[status, created], align="baseline", pad=0, sep=8)
    box = AnchoredOffsetbox(
        loc="lower center",
        child=row,
        pad=0.0,
        borderpad=0.0,
        frameon=False,
        bbox_to_anchor=(0.5, 1.01),
        bbox_transform=ax.transAxes,
    )
    ax.add_artist(box)


def icing_epochs(
    sta: str, *, icing: bool = True, icing_ratio: float | None = None
) -> tuple[float, ...]:
    """The station's icing days (GAMIT autcln index), or () when off/absent.

    Graceful like every station catalog here: an older geo_dataread without
    the index, or no harvest in ``pos_dir``, is simply no icing days.
    """
    if not icing:
        return ()
    from geo_dataread import gps_views

    resolve = getattr(gps_views, "resolve_icing_epochs", None)
    if resolve is None:
        return ()
    epochs, _src = resolve(sta, ratio=icing_ratio)
    return tuple(epochs)


def view_flags(
    sta: str,
    yearf: Any,
    data: Any,
    Ddata: Any,
    *,
    outlier_params: Any = None,
    outlier_overrides: str | None = None,
    provisional_days: float | None = None,
    step_epochs: Any = None,
    restrict: Any = None,
    icing: bool = True,
    icing_ratio: float | None = None,
) -> tuple[Any, Any, list[bool] | None]:
    """Run the station-aware view detector and return its raw verdicts.

    The resolution chain of ``geo_dataread.gps_views.read_gps_view`` lives
    here, in ONE place: declared step epochs and protect windows from the
    deployed catalogs, thresholds by explicit-arg > per-station catalog >
    spec default.  :func:`_mask_outliers` is the plotting wrapper around
    it; a caller that needs the verdicts themselves rather than a split
    series (the detrend workbench, screening epochs its fit window left
    unjudged) calls this directly, so there is exactly one detector call
    site and the two tools cannot drift apart.

    Args:
        sta: Station four-letter name — the key of every catalog lookup.
        yearf: Epochs, fractional years, shape (N,).
        data: Observations, shape (C, N) or (N,).
        Ddata: Formal 1-σ uncertainties, shape of ``data``.
        outlier_params: ``gps_analysis.OutlierParams`` thresholds; None
            lets the station's catalog row (else the spec defaults) win.
        outlier_overrides: Explicit ``outlier_overrides.csv`` path.
        provisional_days: Recency bound of the provisional mask [d].
        step_epochs: Declared step epochs, REPLACING the catalog lookup.
            A caller that declared its own steps (the workbench's
            ``--step``) must pass them: detection around an undeclared
            step over-flags at ANY threshold, so a detector fed fewer
            steps than its caller knows about answers a different
            question.  None resolves the deployed catalog.
        restrict: Optional (N,) bool mask — verdicts are forced False
            outside it.  Detection still runs over the FULL series, and
            deliberately so: the detector fits its own trajectory and
            needs the whole span to place it.  Only the REPORTED verdict
            is narrowed.  This is how a caller says "rule on these epochs
            only" without handing the detector a truncated series it
            would fit worse.

    Returns:
        ``(flags, provisional, aborted)`` — two bool arrays shaped like
        ``data`` and the per-component abort list (None when the
        provenance lacks it).  ``flags`` and ``provisional`` are disjoint
        by construction: a flagged epoch is a removal, a provisional one
        is a pending verdict on an epoch that stays in the series.
    """
    from geo_dataread import gps_views

    # Station catalogs are ENHANCEMENTS: each resolver degrades to "nothing
    # declared" with a UserWarning rather than raising, so a laptop with no
    # deployed gpsconfig still plots (plotTime never fails for a view reason).
    if step_epochs is None:
        steps, _steps_src = gps_views.station_step_epochs(sta)
    else:
        steps = np.atleast_1d(np.asarray(step_epochs, dtype=float))
    pwindows, _pw_src = gps_views.resolve_protect_windows(sta)
    xepochs, _xe_src = gps_views.resolve_excluded_epochs(sta)
    iepochs = icing_epochs(sta, icing=icing, icing_ratio=icing_ratio)
    resolved = gps_views.resolve_outlier_detection(
        sta, outlier_params=outlier_params, outlier_overrides=outlier_overrides
    )

    # flags come back shaped like data (1D in -> 1D out); keep it that way so
    # the returned series never changes shape on the caller
    detect_kwargs: dict[str, Any] = {
        "outlier_params": resolved.params,
        "step_epochs": steps if steps.size else None,
        "protect_windows": pwindows,
        "min_outlier": resolved.min_outlier,
    }
    if provisional_days is not None:
        detect_kwargs["provisional_days"] = provisional_days
    if xepochs:
        detect_kwargs["exclude_epochs"] = xepochs
    if iepochs:
        detect_kwargs["icing_epochs"] = iepochs
    flags, prov = gps_views.detect_view_outliers(yearf, data, Ddata, **detect_kwargs)
    flags = np.asarray(flags, dtype=bool)

    # DIAGNOSTIC, and absent from older geo_dataread -- never let its absence
    # break a plot (the same graceful-degrade rule the rest of this path follows)
    pflags = np.asarray(prov.get("provisional", False), dtype=bool)
    if pflags.shape != flags.shape:
        pflags = np.zeros_like(flags)

    if restrict is not None:
        keep = np.asarray(restrict, dtype=bool)
        flags = flags & keep
        pflags = pflags & keep

    # §3.5a: which components the detector refused to rule on. A component
    # that aborted is served RAW with only a warning, so without this the
    # figure is indistinguishable from a clean one -- the failure mode is
    # invisible exactly where it matters most.
    aborted = prov.get("component_abort")
    aborted = None if aborted is None else [bool(v) for v in np.atleast_1d(aborted)]
    return flags, pflags, aborted


def _mask_outliers(
    sta: str,
    yearf: Any,
    data: Any,
    Ddata: Any,
    *,
    outlier_params: Any = None,
    outlier_overrides: str | None = None,
    provisional_days: float | None = None,
    icing: bool = True,
    icing_ratio: float | None = None,
) -> tuple[
    Any,
    tuple[Any, Any] | None,
    tuple[Any, Any] | None,
    list[bool] | None,
    tuple[Any, Any] | None,
]:
    """Split a series into a cleaned main series and an outlier overlay.

    The station-aware resolution chain lives in :func:`view_flags` (which
    mirrors ``geo_dataread.gps_views.read_gps_view``, so a plotted cleaned
    view matches the canonical read/write path for the SAME station); this
    is the plotting half — turning verdicts into a series and overlays.

    Flags come from ``geo_dataread.gps_views.detect_view_outliers`` (the
    model-aware, signal-protecting detector of the analysis leaf) — a MASK,
    never a filter: the returned main series has flagged epochs set to NaN
    per component, and the overlay carries exactly those epochs so the plot
    shows what was flagged. Detection failure/abort degrades inside
    geo_dataread (warning + all-False flags), in which case the overlay is
    None and the main series is untouched.

    Args:
        sta: Station four-letter name — the key of every catalog lookup.
        yearf: Epochs, fractional years.
        data: Observations, shape (C, N) or (N,).
        Ddata: Formal 1-σ uncertainties, shape of ``data``.
        outlier_params: ``gps_analysis.OutlierParams`` thresholds; None
            lets the station's catalog row (else the spec defaults) win.
        outlier_overrides: Explicit ``outlier_overrides.csv`` path; None
            resolves the deployed catalog.

    Returns:
        ``(data, overlay, provisional, aborted)`` — ``overlay`` is
        ``(out_data, out_Ddata)`` NaN everywhere except the flagged
        epochs, or None when nothing was flagged.  ``provisional`` has the
        same shape convention for the epochs the detector could not rule
        on yet (recent + indeterminate step evidence), or None.
        ``aborted`` is :func:`view_flags`' per-component abort list, passed
        through so the caller can draw the badge.

        overlay and provisional are disjoint and behave differently on
        purpose: flagged epochs are REMOVED from the returned series,
        provisional ones are KEPT in it.  A provisional marker annotates a
        point that is still part of the data.

        The fourth member is not decoration: a component whose detection
        aborted is served RAW with only a ``UserWarning``, so without it an
        aborted axis is indistinguishable from a clean one.  It was missing
        from this annotation while both ``return`` statements and every
        caller had four members.
    """
    flags, pflags, aborted = view_flags(
        sta,
        yearf,
        data,
        Ddata,
        outlier_params=outlier_params,
        outlier_overrides=outlier_overrides,
        provisional_days=provisional_days,
        icing=icing,
        icing_ratio=icing_ratio,
    )

    provisional = None
    if pflags.any():
        provisional = (
            np.where(pflags, data, np.nan),
            np.where(pflags, Ddata, np.nan),
        )

    # the icing days (flagged by the view, every component) get their own
    # overlay; the grey one keeps the detector's outliers only
    imask = np.zeros_like(flags)
    iep = icing_epochs(sta, icing=icing, icing_ratio=icing_ratio)
    if iep:
        from gps_parser import outlier_catalogs as _oc

        idx = list(_oc.match_declared_epochs(np.asarray(yearf, float), np.asarray(iep)))
        if idx:
            if imask.ndim == 1:
                imask[idx] = True
            else:
                imask[:, idx] = True
        imask &= flags
    iced = None
    if imask.any():
        iced = (np.where(imask, data, np.nan), np.where(imask, Ddata, np.nan))

    if not flags.any():
        return data, None, provisional, aborted, iced
    cleaned = np.where(flags, np.nan, data)
    ofl = flags & ~imask
    overlay = None
    if ofl.any():
        overlay = (np.where(ofl, data, np.nan), np.where(ofl, Ddata, np.nan))
    return cleaned, overlay, provisional, aborted, iced


@dataclasses.dataclass
class StationSeries:
    """One station's series after the per-station view pipeline.

    ``detrend_applied`` says whether a stored trajectory was actually
    subtracted (``--view detrended`` / ``--ref detrend``): a station with
    no record degrades to its plate series with a warning, which a
    single-station plot can live with but a baseline cannot (it would
    difference a detrended series against a plate one).
    """

    yearf: Any
    data: Any
    Ddata: Any
    outliers: tuple[Any, Any] | None
    provisional: tuple[Any, Any] | None
    aborted: list[bool] | None
    detrend_applied: bool
    iced: tuple[Any, Any] | None = None


def pos_read_kwargs(
    ref_position: Any, convention: str, pos_id: str | None
) -> dict[str, Any]:
    """The .pos keywords ``getData`` is called with (none for the defaults)."""
    pos_kw: dict[str, Any] = {}
    if ref_position is not None or convention != "geodetic":
        pos_kw = dict(ref_position=ref_position, convention=convention)
    if pos_id is not None:
        pos_kw["pos_id"] = pos_id
    return pos_kw


def record_context(
    tType: str, pos_id: str | None, params: str | None
) -> tuple[str, str | None]:
    """The series frame and the detrend document that belongs to it.

    Detrend parameters belong to one reference frame: the series' frame tag
    guards the record, and a .pos frame other than ITRF2008 reads its own
    document (``detrend_params.<frame>.json``) unless ``params`` is given.
    """
    from geo_dataread import gps_views as _gv

    frame_of = pos_id if tType == "POS" else None
    series_frame = _gv.series_frame(frame_of)
    if params is None and series_frame != _gv.PLATE_REMOVED_FRAME:
        _p = _gv.default_params_path(frame_of)
        params = str(_p) if _p is not None else None
    return series_frame, params


def trajectory_views(
    sta: str,
    yearf: Any,
    data: Any,
    *,
    ref: str,
    view: str,
    tType: str,
    pos_id: str | None,
    remove_kinds: Sequence[str] | None = None,
    detrend_terms: Sequence[str] | None = None,
    steps_catalog: str | None = None,
    params: str | None = None,
) -> tuple[Any, bool]:
    """Remove the requested parts of the stored trajectory from ``data``.

    ``remove_kinds`` step removal, then ``view="detrended"`` (the whole
    trajectory) or ``ref="detrend"`` (the ``detrend_terms`` groups; step
    offsets stay). Returns ``(data, detrend_applied)``. Every term is a
    function of time subtracted from the data, so applied to zeros this
    returns minus what it removes: ``gpsplot ts --model`` draws the model
    line through this same function, so line and data cannot disagree.
    """
    series_frame = None
    if view == "detrended" or ref == "detrend" or remove_kinds:
        series_frame, params = record_context(tType, pos_id, params)
    detrend_applied = False

    # Selective step removal: subtract the RECORDED step amplitudes (from
    # the deployed detrend record) for declared steps whose kind is in
    # ``remove_kinds``.  steps.yaml has the epochs + kinds, the record has
    # the amplitudes — a pure apply, no re-fit, and it degrades to a
    # warning (series unchanged) when there is no record or no matching
    # step.  Runs BEFORE the inline detrended view (below), so
    # --remove-steps + --view detrended compose: selective steps removed
    # first, then the remaining trajectory is subtracted.
    if remove_kinds:
        from geo_dataread import gps_views

        data, _prov = gps_views.remove_declared_steps(
            sta,
            yearf,
            data,
            kinds=remove_kinds,
            params=params,
            steps_catalog=steps_catalog,
        )

    # Inline detrended view: subtract the stored trajectory from the
    # SAME plate data — one data stream, so cleaning + detrending
    # compose without a double-read.  Runs AFTER --remove-steps, so the
    # two compose: selective steps removed first, then the remainder is
    # subtracted.
    #
    # --view detrended: subtract the FULL trajectory (rate + seasonal +
    # steps).  --ref detrend: subtract rate + seasonal only — step
    # offsets remain visible (the user wants trends removed, not discrete
    # events).
    if view == "detrended" or ref == "detrend":
        from geo_dataread import gps_views as _gv

        try:
            doc = _gv.read_detrend_params(params)
            record, _src = _gv.station_detrend_record(doc, sta)
        except Exception:
            record = None
        if record is None:
            warnings.warn(
                f"{sta}: no detrend record for this station; serving raw series",
                UserWarning,
                stacklevel=3,
            )
        else:
            try:
                record_frame = record.get("frame")
                if record_frame is not None and record_frame != series_frame:
                    raise RuntimeError(
                        f"{sta}: frame mismatch: record frame {record_frame!r} != "
                        f"series frame {series_frame!r} - refusing to apply"
                    )
                # --view detrended (legacy): the whole trajectory. --ref
                # detrend: the chosen terms (default linear + periodic) as
                # staged groups, whose "secular" excludes the steps, so the
                # step offsets stay visible (they are --remove-steps' job)
                if view == "detrended":
                    data = _gv.apply_stored_detrend(
                        record, yearf, data, frame=series_frame
                    )
                else:
                    data = _gv.apply_stored_detrend(
                        record,
                        yearf,
                        data,
                        groups=tuple(
                            DETREND_TERM_GROUPS[t]
                            for t in (detrend_terms or DEFAULT_DETREND_TERMS)
                        ),
                        frame=series_frame,
                    )
                detrend_applied = True
            except ValueError as exc:
                warnings.warn(
                    f"{sta}: could not apply detrend record ({exc}); "
                    "serving raw series",
                    UserWarning,
                    stacklevel=3,
                )

    return data, detrend_applied


def station_series(
    sta: str,
    *,
    fstart: float | None,
    fend: float | None,
    ref: str,
    Dir: str | None,
    tType: str,
    uncert: int,
    view: str,
    outlier_params: Any = None,
    outlier_overrides: str | None = None,
    provisional_days: float | None = None,
    remove_kinds: Sequence[str] | None = None,
    detrend_terms: Sequence[str] | None = None,
    steps_catalog: str | None = None,
    params: str | None = None,
    ref_position: str | None = None,
    icing: bool = True,
    icing_ratio: float | None = None,
    convention: str = "geodetic",
    pos_id: str | None = None,
) -> StationSeries:
    """Read one station and apply the requested view: clean, remove steps, detrend.

    The single per-station pipeline behind :func:`plotTime`. A baseline
    runs BOTH of its stations through this same function, so the two ends
    can never be prepared differently. Order: read (``getData``, in
    ``ref``) → ``cleaned`` masking → ``remove_kinds`` step removal →
    stored-trajectory subtraction (``view="detrended"`` subtracts the full
    trajectory, ``ref="detrend"`` keeps the step offsets visible).
    """
    import geo_dataread.gps_read as gpsr

    yearf, data, Ddata, _offset = gpsr.getData(
        sta,
        fstart=fstart,
        fend=fend,
        ref=ref,
        Dir=Dir,
        tType=tType,
        uncert=uncert,
        **pos_read_kwargs(ref_position, convention, pos_id),
    )
    if yearf is None or len(yearf) == 0:
        raise ValueError("no data for station %s" % sta)

    outliers = provisional = aborted = iced = None
    if view == "cleaned":
        data, outliers, provisional, aborted, iced = _mask_outliers(
            sta,
            yearf,
            data,
            Ddata,
            outlier_params=outlier_params,
            outlier_overrides=outlier_overrides,
            provisional_days=provisional_days,
            icing=icing,
            icing_ratio=icing_ratio,
        )

    views = dict(
        ref=ref,
        view=view,
        tType=tType,
        pos_id=pos_id,
        remove_kinds=remove_kinds,
        detrend_terms=detrend_terms,
        steps_catalog=steps_catalog,
        params=params,
    )
    data, detrend_applied = trajectory_views(sta, yearf, data, **views)
    if detrend_applied or remove_kinds:
        # the outlier / provisional overlays are values of the SAME series
        # (NaN off their flags): they get the same steps and trajectory
        # removed, or they sit at their plate values beside detrended data
        def _moved(pair: Any) -> Any:
            if pair is None:
                return None
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # already said for the data
                vals, _ = trajectory_views(sta, yearf, pair[0], **views)
            return (vals, pair[1])

        outliers, provisional = _moved(outliers), _moved(provisional)
        iced = _moved(iced)

    return StationSeries(
        yearf, data, Ddata, outliers, provisional, aborted, detrend_applied, iced
    )


def resolve_period(
    start: datetime.datetime | None,
    end: datetime.datetime | None,
    special: str | None,
    fix: bool,
) -> tuple[Any, Any, float | None, float | None]:
    """``(start, end, fstart, fend)``: the axis span and the read window.

    ``special`` (90d, year, full, fixedstart) sets the routine sub-periods,
    ending yesterday unless ``end`` is given. ``start``/``end`` (datetimes)
    bound the x axis only with ``fix`` or ``special``; otherwise the axis
    spans the data. ``fstart``/``fend`` (fractional years) are what is read.
    """
    fstart = fend = None
    if start:
        fstart = currYearfDate(refday=start)
    if end:
        fend = currYearfDate(refday=end)

    # standard sub-periods for routine plots
    if special:
        if not end:
            end = currDatetime(-1)
            fend = currYearfDate(refday=end)

        if special == "90d":
            start = currDatetime(days=-91, refday=end)
            fstart = currYearfDate(refday=start)
        if special == "year":
            start = currDatetime(days=-366, refday=end)
            fstart = currYearfDate(refday=start)
        if special == "fixedstart":
            pass
        if special == "full":
            start = None
            fstart = None

    if not (fix or special):  # only plot the extent of the data
        start = end = None
    return start, end, fstart, fend


def plotTime(
    sta: str,
    start: datetime.datetime | None = None,
    end: datetime.datetime | None = None,
    save: str | Sequence[str] | None = None,
    ylim: Sequence[float] | Sequence[Sequence[float]] = (),
    special: str | None = None,
    ref: str | None = "itrf2008",
    figDir: str = "",
    events: dict[Any, Any] | None = None,
    fix: bool = False,
    Dir: str | None = None,
    tType: str = "TOT",
    ref_position: str | None = None,
    convention: str = "geodetic",
    pos_id: str | None = None,
    uncert: int = 15,
    logo: bool = True,
    fig: Figure | None = None,
    view: str = "raw",
    name: str | None = None,
    outlier_params: Any = None,
    outlier_overrides: str | None = None,
    hide_outliers: bool = False,
    outlier_tail: int | None = None,
    provisional_days: float | None = None,
    icing: bool = True,
    icing_ratio: float | None = None,
    remove_kinds: Sequence[str] | None = None,
    detrend_terms: Sequence[str] | None = None,
    annotate_steps: bool | Sequence[str] | None = False,
    steps_catalog: str | None = None,
    params: str | None = None,
    show_ends: bool = False,
    baseline_rho: Sequence[float] | None = None,
    baseline_sigma_scale: tuple[Sequence[float], Sequence[float]] | None = None,
    baseline_sigma: str = "formal",
    baseline_noise: str | None = None,
    overlay_pos_id: str | None = None,
    model: str | None = None,
    model_terms: Sequence[str] | None = None,
    rate: bool = False,
    rate_windows: Sequence[tuple[float, float]] | None = None,
    rate_segments: Sequence[str] | None = None,
    segment_kinds: Sequence[str] = ("dike", "eruption"),
    segment_cut: str = "span",
    segment_min_days: int = 30,
    events_catalog: str | None = None,
) -> Figure:
    """Plot a standard GPS North/East/Up time series for one station.

    Reads the series (``geo_dataread``), builds the figure and either saves
    it (``save`` = format or list/comma-string of formats, e.g. ``"png"``
    or ``"eps,pdf,png"`` -- each written by one native ``savefig``) or
    shows it interactively.  Returns the Figure (REPL-friendly).

    ``view`` is the first-class raw|cleaned toggle of the internal delivery
    path (geo_dataread ``gps_views``, design DESIGN_live_detrending §0):
    ``"raw"`` (default) plots exactly before; ``"cleaned"`` masks outlier
    epochs from the main series and overlays them as GREY points (mask only
    — raw stays retrievable).  Use ``remove_kinds`` (--remove-steps) for
    selective step removal per-kind from the deployed record. Degrades
    gracefully (warning + undegraded series), so a plot never fails for a
    view reason.

    ``outlier_params`` / ``outlier_overrides`` reach the ``cleaned`` view
    only.  Both are thresholds-level levers of
    :func:`geo_dataread.gps_views.resolve_outlier_detection`: an explicit
    ``OutlierParams`` beats the station's catalog row, which beats the
    spec defaults; ``outlier_overrides`` points at a specific
    ``outlier_overrides.csv`` instead of the deployed one.  Declared step
    epochs and protect windows always resolve from their own catalogs —
    see :func:`_mask_outliers` for why they are not optional.

    ``outlier_tail`` = N draws only the outliers among the series' last N
    epochs (``--show-outliers N``).

    ``hide_outliers`` suppresses the grey overlay of the ``cleaned``
    view.  It changes DISPLAY ONLY -- the flagged epochs are already
    absent from the main series (masked to NaN by :func:`_mask_outliers`),
    so this decides whether the plot still SHOWS what was set aside.  Two
    consequences worth knowing: the y-axis tightens to the cleaned series
    (nothing sets explicit limits unless ``ylim`` is given, so matplotlib
    autoscales over whatever artists exist, overlay included), and the
    figure no longer carries the evidence of what the detector removed --
    which is exactly why the overlay is the default.

    ``model`` (``--model``) draws the stored trajectory model as a line:
    ``"curve"`` over the plate series (``ref="plate"``), ``"residual"``
    over the detrended one (``ref="detrend"``), where the line is the part
    of the model still in the data (the steps, unless removed). Solid in
    the record's fit window, dashed where it is extrapolated. Files get
    ``-model`` / ``-resid``. Stations only, for now. ``model_terms``
    (``--model-terms``) draws exactly those model groups instead, whatever
    was removed from the data; files then add ``-line-<terms>``.

    ``name`` fixes the output basename: with ``save`` set the figure is
    written to ``<figDir>/<name>.<fmt>``, BYPASSING the
    ref/view/tType/period filename construction entirely.  Every call with
    the same ``name`` overwrites the same path -- that is the point (watch
    one file in a viewer while re-rendering), but it also means a
    multi-station or multi-variant run leaves only the last render.

    ``baseline_rho`` / ``baseline_sigma_scale`` (baselines only) replace
    the quadrature σ of the baseline by the empirical one —
    :func:`geo_dataread.baseline.baseline` ``rho=`` / ``sigma_scale=``:
    the shared daily-error fraction per component and each station's
    σ scale factor (``gps_analysis.empirical_sigma``). The ends drawn by
    ``show_ends`` keep their FORMAL σ.

    ``baseline_sigma="empirical"`` derives them from the fitted record
    (``geo_dataread.baseline_noise``; ``baseline_noise`` overrides its
    path): the pair's OWN shared fraction once it has enough consecutive
    common days, else ρ at the stations' distance (stations.cfg
    coordinates); each station's scale k, or the network median. Missing
    record/coordinates warn and fall back to the formal σ; a note on the
    plot says which σ is shown. Adds ``-empsig`` to the file name.
    """
    if view not in ("raw", "cleaned", "detrended"):
        raise ValueError(f"view must be 'raw', 'cleaned' or 'detrended', got {view!r}")

    # heavy production deps are imported lazily so the module (and the
    # figure-building seam) stays importable without them
    import geo_dataread.gps_read as gpsr

    if not save:
        mpl.use("WebAgg")

    start, end, fstart, fend = resolve_period(start, end, special, fix)

    # graph title reference-frame string
    if ref is None:  # baselines and the detrended view default to plate
        ref = "plate" if (split_baseline(sta) or view == "detrended") else "itrf2008"
    if ref == "plate":
        from geofunc.plates import plate_name

        from geo_dataread.station_catalog import plate_table

        table = plate_table()
        ends = split_baseline(sta) or (sta,)
        unknown = [s for s in ends if s not in table]
        if unknown:
            # the plate-removed series cannot be formed without a plate; say
            # which catalog to fix instead of surfacing a bare KeyError
            raise ValueError(
                f"{sta}: no plate assigned to {', '.join(unknown)} in the "
                "deployed station-plate catalog (postprocess.cfg platefile). "
                "Add it in gps-config-data and deploy, or plot with --ref ref_frame."
            )
        plates = [plate_name(table[s]) for s in ends]
        # a cross-rift baseline has two plates; say so rather than pick one
        refTitle = (
            plates[0]
            if len(set(plates)) == 1
            else " / ".join(
                f"{s}: {p}" for s, p in zip(split_baseline(sta) or (), plates)
            )
        )
    elif ref == "detrend":
        refTitle = "Detrended"
        if detrend_terms and tuple(detrend_terms) != DEFAULT_DETREND_TERMS:
            refTitle += f" ({' + '.join(detrend_terms)})"
    else:
        refTitle = ref.upper()
        if ref == "itrf2008" and tType == "POS" and pos_id is not None:
            # "itrf2008" means "no plate removal": name the series' own frame
            from geo_dataread.pos import POS_PRODUCTS, pos_product

            refTitle = POS_PRODUCTS.get(pos_product(pos_id), (refTitle, ""))[0]
    if tType == "POS" and ref != "itrf2008":
        # name the series' frame: plate-removed or detrended in which ITRF
        from geo_dataread.pos import POS_PRODUCTS, pos_product

        _label = POS_PRODUCTS.get(pos_product(pos_id), ("", ""))[0]
        if _label:
            refTitle = f"{refTitle}, {_label}"
    if view == "detrended" and ref != "detrend":
        # the full stored trajectory is subtracted from the plate series; the
        # title must say so, or it reads exactly like the plain plate plot
        refTitle = f"{refTitle}, detrended"

    pair = split_baseline(sta)
    if model is not None:
        from gps_plot.sources.gps import MODEL_REFS

        if model not in ("curve", "residual"):
            raise ValueError(f"model must be 'curve' or 'residual', got {model!r}")
        if pair is not None:
            raise ValueError(f"{sta}: --model over a baseline is not supported yet")
        if ref not in MODEL_REFS:
            raise ValueError(
                f"{sta}: --model draws the stored model, which is fitted to the "
                f"plate series: use --ref plate or detrend, not {ref!r}"
            )
        if (
            overlay_pos_id is not None
            or ref_position is not None
            or convention != "geodetic"
            or (pos_id is not None and pos_id.startswith("imo.08hr"))
        ):
            # the record's datum is the full daily read, zeroed by getData;
            # an 8 h series or a chosen reference position is zeroed
            # elsewhere and shares no epoch to align on exactly
            raise ValueError(
                f"{sta}: --model is for the daily series only (no 8 h "
                "sessions, --session both or a reference position) for now"
            )
    if baseline_sigma not in ("formal", "empirical"):
        raise ValueError(
            f"baseline_sigma must be 'formal' or 'empirical', got {baseline_sigma!r}"
        )
    # 8 h sessions: epochs are matched within 2 h (a missing session must not
    # pair with its neighbour 8 h away); the daily default is half a day
    from geo_dataread.baseline import BASELINE_EPOCH_TOLERANCE_YEARS

    def _is_8h(pid: str | None) -> bool:
        if pid is None:
            return False
        from geo_dataread.pos import pos_product

        return pos_product(pid).startswith("imo.08hr")

    epoch_tol = (
        2.0 / 24.0 / 365.25
        if tType == "POS" and _is_8h(pos_id)
        else BASELINE_EPOCH_TOLERANCE_YEARS
    )
    if (
        overlay_pos_id is not None
        and pair is None
        and ref_position is None
        and tType == "POS"
    ):
        # one zero for both series: the main series' first position IN THE
        # PLOTTED WINDOW, so the plot starts at zero and the overlay sits
        # where its positions are, not re-zeroed on its own
        from geo_dataread import pos as _pos

        _main = _pos.read_pos(_pos.pos_path(sta, Dir, _pos.pos_product(pos_id)))
        _first = 0
        if fstart is not None:
            _first = min(
                # fstart (gtimes day/365) and .pos yearf (GLOBK, noon epochs,
                # 365.25) differ by up to half a day: allow it, or the first
                # plotted day is skipped
                int(np.searchsorted(np.asarray(_main.yearf), fstart - 0.5 / 365.25)),
                len(_main.xyz) - 1,
            )
        ref_position = _main.xyz[_first]
    if baseline_sigma == "empirical" and pair is not None and _is_8h(pos_id):
        raise ValueError(
            f"baseline {sta}: the empirical baseline noise record is fitted on "
            "daily solutions; use --baseline-sigma formal for 8 h sessions"
        )
    series_kw = dict(
        fstart=fstart,
        fend=fend,
        ref=ref,
        Dir=Dir,
        tType=tType,
        uncert=uncert,
        view=view,
        outlier_params=outlier_params,
        outlier_overrides=outlier_overrides,
        provisional_days=provisional_days,
        remove_kinds=remove_kinds,
        detrend_terms=detrend_terms,
        steps_catalog=steps_catalog,
        params=params,
        ref_position=ref_position,
        convention=convention,
        pos_id=pos_id,
        icing=icing,
        icing_ratio=icing_ratio,
    )
    from gps_plot.sources import gps as gps_source

    request = gps_source.SeriesRequest(**series_kw)
    main_series = gps_source.fetch(
        sta,
        request,
        gps_source.BaselineOptions(
            tol=epoch_tol,
            empirical=baseline_sigma == "empirical",
            rho=baseline_rho,
            sigma_scale=baseline_sigma_scale,
            noise_path=baseline_noise,
        ),
    )
    if ref == "detrend" or view == "detrended":
        # re-zero a detrended station at its start (vshift again, after the
        # detrend): the model was subtracted after getData's zeroing
        main_series = gps_source.rezero_detrended(sta, request, main_series)
    yearf, data, Ddata = main_series.t, main_series.y, main_series.sigma
    outliers, provisional = main_series.outliers, main_series.provisional
    aborted = main_series.aborted
    iced = main_series.iced
    sigma_note = main_series.meta.get("sigma_note")
    # the rho the baseline σ used (from the noise record when empirical);
    # None means formal quadrature, which names the file without -empsig
    baseline_rho = main_series.meta.get("rho", baseline_rho)
    if show_ends and main_series.ends is not None:
        # the data layer already referenced both ends to the baseline's
        # zero epoch; only the time axis is converted here
        ends_zeroed = [
            (list(gpsr.toDateTime(e.t)), e.y, e.sigma) for e in main_series.ends
        ]

    # single yearf -> datetime conversion (was done twice before)
    x = list(gpsr.toDateTime(yearf))
    firstpoint, lastpoint = x[0], x[-1]

    warnp = -1  # data through currDate(warnp) is labelled green
    Title = make_title(sta, lastpoint, ref=refTitle, warnp=warnp)
    # ONE currency decision drives both the green title fragment and the
    # green last-point marker, computed here on the UNMASKED epoch list so
    # the cleaned view cannot make the two disagree (see stdTimesPlot).
    highlight_last = data_is_current(lastpoint, warnp)

    if save and fig is None:
        fig = _reusable_figure()

    fig = stdTimesPlot(
        x,
        data,
        Ddata,
        Title=Title,
        start=start,
        end=end,
        ylim=ylim,
        warnp=warnp,
        fig=fig,
        highlight_last=highlight_last,
    )

    if sigma_note is not None:
        fig.axes[2].text(
            0.01,
            0.02,
            # the figure renders text through LaTeX (usetex): no raw σ/ρ
            "baseline $\\sigma$: "
            + sigma_note.replace("ρ", "$\\rho$").replace("σ", "$\\sigma$"),
            transform=fig.axes[2].transAxes,
            fontsize="small",
            color="0.3",
            va="bottom",
        )

    if show_ends:
        if pair is None:
            warnings.warn(
                f"{sta}: --show-ends applies to baselines (AAAA-BBBB) only; ignored",
                UserWarning,
                stacklevel=2,
            )
        else:
            fig = add_baseline_ends(fig, pair, ends_zeroed)

    if outliers is not None and not hide_outliers:
        # cleaned view: flagged epochs overlaid in grey on top of the masked
        # main series (mask only — nothing is deleted, matching the
        # geo_dataread raw-preservation rule).  Colors are passed at the call
        # site, not as addData defaults, because the JOIN overlay below still
        # wants the standard red markers.  ``outlier_tail`` = N keeps only
        # those among the last N epochs.
        oy, ody = outliers
        if outlier_tail:
            oy, ody = last_flagged(oy, ody, outlier_tail)
        fig = addData(
            x,
            oy,
            ody,
            fig,
            ecolor=OUTLIER_ERRORBAR_COLOR,
            markerfacecolor=OUTLIER_FACE_COLOR,
            markeredgecolor=OUTLIER_EDGE_COLOR,
        )

    if iced is not None:
        # icing suspected (GAMIT autcln index, geo_dataread.autcln_stats):
        # removed from the series like outliers, drawn in their own colour
        # under the same --show-outliers rule, and counted in a note even
        # when hidden -- an iced spell must never pass for clean data
        n_iced = int(np.count_nonzero(np.isfinite(np.atleast_2d(iced[0])).any(axis=0)))
        if not hide_outliers:
            iy, idy = iced
            if outlier_tail:
                iy, idy = last_flagged(iy, idy, outlier_tail)
            fig = addData(
                x,
                iy,
                idy,
                fig,
                ecolor=ICING_ERRORBAR_COLOR,
                markerfacecolor=ICING_FACE_COLOR,
                markeredgecolor=ICING_EDGE_COLOR,
            )
        fig.axes[0].text(
            0.01,
            0.02,
            "icing suspected: %d day%s (GAMIT autcln)"
            % (n_iced, "" if n_iced == 1 else "s"),
            transform=fig.axes[0].transAxes,
            fontsize="small",
            color=ICING_EDGE_COLOR,
            va="bottom",
        )

    if provisional is not None:
        # Epochs the detector cannot rule on yet.  Unlike the grey overlay
        # these are NOT masked out -- the point is still in the main series
        # underneath, so the marker is drawn larger to cover the red one and
        # say "verdict pending", not "removed".  --hide-outliers does NOT
        # suppress these: hiding a decided outlier declutters, hiding an
        # undecided one hides the thing most worth looking at.
        fig = addData(
            x,
            provisional[0],
            provisional[1],
            fig,
            markersize=5.5,
            ecolor=PROVISIONAL_ERRORBAR_COLOR,
            markerfacecolor=PROVISIONAL_FACE_COLOR,
            markeredgecolor=PROVISIONAL_EDGE_COLOR,
        )
        if highlight_last:
            fig = add_provisional_last(x, provisional[0], fig)

    if overlay_pos_id is not None:
        fig = _add_session_overlay(
            fig,
            sta,
            pair,
            overlay_pos_id,
            series_kw,
            main=(yearf, data, Ddata, x),
        )

    if model is not None:
        # after every data overlay: the line goes over the points, and its
        # layer keeps the axis limits the data set
        from gps_plot.layers.model import add_model_curve, model_note

        model_series: Any = None
        try:
            model_series = gps_source.fetch_model(sta, request, yearf, model_terms)
        except ValueError as exc:
            # e.g. a seasonal-only (regional) record has no full model line:
            # the plot is still worth drawing, without the line
            warnings.warn(f"{sta}: no model line ({exc})", UserWarning, stacklevel=2)
    if model is not None and model_series is not None:
        _shift = main_series.meta.get("zero_shift")
        if _shift is not None:
            # the data were re-zeroed: the line moves with them
            model_series = dataclasses.replace(
                model_series,
                y=np.asarray(model_series.y) + np.asarray(_shift)[:, None],
            )
        fig = add_model_curve(fig, model_series)
        fig.axes[0].text(
            0.01,
            0.02,
            model_note(model_series),
            transform=fig.axes[0].transAxes,
            fontsize="small",
            color="0.3",
            va="bottom",
        )

    if rate or rate_windows:
        # the velocity product's rate per window (gpsplot ts --rate): the
        # same number as the map and the velocity files
        from gps_plot.layers.rate import (
            add_rate_lines,
            default_window,
            rate_note,
            rate_settings,
        )

        if pair is not None:
            raise ValueError(f"{sta}: --rate over a baseline is not supported yet")
        if overlay_pos_id is not None or tType != "POS":
            raise ValueError(f"{sta}: --rate reads the daily .pos series only")
        settings = rate_settings(
            ref=ref, detrend_terms=detrend_terms, view=view, pos_id=pos_id,
            uncert=uncert, Dir=Dir,
        )  # fmt: skip
        if rate_segments is not None:
            from gps_plot.layers.rate import (
                add_segment_rates,
                segment_gaps,
                segment_table,
                segment_windows,
            )

            gaps = segment_gaps(
                sta, rate_segments, kinds=segment_kinds, cut=segment_cut,
                steps=steps_catalog, events=events_catalog,
            )  # fmt: skip
            windows = segment_windows(
                yearf, fstart, fend, gaps, min_days=segment_min_days
            )
            records = add_segment_rates(fig, sta, yearf, data, Ddata, windows, settings)
            print(segment_table(sta, records))
        else:
            windows = list(rate_windows or [default_window(yearf, fstart, fend)])
            records = add_rate_lines(fig, sta, yearf, data, Ddata, windows, settings)
        fig.axes[2].text(
            0.99, 0.02, rate_note(records, segments=rate_segments is not None),
            transform=fig.axes[2].transAxes,
            fontsize="small", color="0.3", va="bottom", ha="right",
        )  # fmt: skip

    if aborted is not None and any(aborted):
        # after the overlays so the badge is never painted under them
        fig = addAbortBadge(fig, aborted)

    if tType == "JOIN":
        Pdata = gpsr.convGlobktopandas(yearf, data, Ddata)
        yearf8, data8, Ddata8, _offset8 = gpsr.getData(
            sta,
            fstart=fstart,
            fend=fend,
            ref=ref,
            Dir=Dir,
            tType="08h",
            uncert=uncert,
            offset=None,
        )
        Pdata8 = gpsr.convGlobktopandas(yearf8, data8, Ddata8)
        shift = [
            (Pdata8.north - Pdata.north).mean(),
            (Pdata8.east - Pdata.east).mean(),
            (Pdata8.up - Pdata.up).mean(),
        ]
        data8 = np.array([data8[i, :] - shift[i] for i in range(3)])
        x8 = list(gpsr.toDateTime(yearf8))
        fig = addData(x8, data8, Ddata8, fig, markerfacecolor="b", markeredgecolor="b")

    if events:
        addEvent(events, fig)

    if annotate_steps:
        # Declared steps from steps.yaml, drawn as a vertical line + rotated
        # label (seismic dark-red, equipment dark-green) — independent of the
        # --remove-steps data transformation.  ``annotate_steps`` is ``True``
        # (all kinds) or a sequence of kind strings (filtered).
        kind_filter = (
            annotate_steps
            if isinstance(annotate_steps, Sequence)
            and not isinstance(annotate_steps, str)
            else None
        )
        pair = split_baseline(sta)
        for end in pair or (sta,):
            for epoch, label, color in step_annotations(
                end, steps_catalog, kinds=kind_filter
            ):
                tag = f"{end}: {label}" if pair else label
                add_event_lines(fig, [(epoch, tag)], color)

    if logo:
        inpLogo(fig)

    if save and name:
        # fixed basename: bypass the whole filend construction so repeated
        # renders land on exactly the same path
        saveFig(os.path.join(figDir, name), save, fig)
    elif save:
        # .pos series name their frame below; "itrf2008" (= nothing removed)
        # would read as a second frame there
        filend = "" if (tType == "POS" and ref == "itrf2008") else "-%s" % (ref,)
        if (
            ref == "detrend"
            and detrend_terms
            and tuple(detrend_terms) != DEFAULT_DETREND_TERMS
        ):
            filend += "-" + "-".join(detrend_terms)
        if view == "cleaned":
            filend += "-cleaned"
            if getattr(outlier_params, "method", "running") == "model":
                # the default (running window, gps_analysis §15) is unmarked
                # and carries no sigma, so operational graphs keep their names
                # while the tolerance is tuned; only the old model method is
                filend += "-modeldetect"
        elif view == "detrended":
            filend += "-detrended"
        if show_ends and pair is not None:
            filend += "-ends"
        if pair is not None and baseline_rho is not None:
            filend += "-empsig"
        if remove_kinds:
            filend += f"-rm-{'-'.join(sorted(remove_kinds))}"
        if model is not None:
            filend += "-model" if model == "curve" else "-resid"
            if model_terms:
                filend += "-line-" + "-".join(model_terms)
        if rate or rate_windows:
            filend += "-rate"
            if rate_segments is not None:
                filend += "-seg" + "-".join(rate_segments)
                if "events" in rate_segments and segment_cut == "onset":
                    filend += "-onset"
        if tType == "POS":
            # one file per frame (and session): itrf2008/itrf2014/itrf2020, 8h
            from geo_dataread.pos import frame_name, pos_product

            _pid = pos_product(pos_id)
            filend += f"-{frame_name(_pid)}"
            if _pid.startswith("imo.08hr"):
                filend += "-8h"
            elif overlay_pos_id is not None:
                filend += "-24h8h"
        elif tType != "TOT":
            filend += "-{0:s}".format(tType)

        if special:
            if special == "fixedstart":
                filend += "_since-%s" % (firstpoint.strftime("%Y%m%d"),)
            else:
                filend += "-%s" % special
        else:
            filend += "_%s-%s" % (
                firstpoint.strftime("%Y%m%d"),
                lastpoint.strftime("%Y%m%d"),
            )

        fileName = os.path.join(figDir, baseline_file_stem(sta) + filend)
        saveFig(fileName, save, fig)
    else:
        plt.show()

    fig._gps_ref = ref  # type: ignore[attr-defined]  # the RESOLVED frame, for logs
    fig._gps_label = _ref_label(ref, tType, pos_id, overlay_pos_id)  # type: ignore[attr-defined]
    return fig


def last_flagged(y: Any, dy: Any, n: int) -> tuple[Any, Any]:
    """Keep the flagged epochs among the series' last ``n`` epochs only.

    ``y``/``dy`` are the outlier overlay (components × epochs, NaN where not
    flagged), aligned with the main series; anything flagged before its
    last ``n`` epochs is dropped.
    """
    y = np.array(y, dtype=float)
    dy = np.array(dy, dtype=float)
    cut = max(y.shape[1] - n, 0)
    y[:, :cut] = np.nan
    dy[:, :cut] = np.nan
    return y, dy


def _ref_label(
    ref: str, tType: str, pos_id: str | None, overlay_pos_id: str | None = None
) -> str:
    """Human label for the console log: what was removed, in which frame.

    ``ref == "itrf2008"`` is the internal token for "nothing removed" (the CLI
    calls it ``ref_frame``); for .pos series the frame and session come from
    the product id, e.g. ``ref_frame, itrf2020 24h+8h``.
    """
    name = "ref_frame" if ref == "itrf2008" else ref
    if tType == "POS" and pos_id is not None:
        from geo_dataread.pos import frame_name, pos_product

        prod = pos_product(pos_id)  # imo.24hr_itrf20
        session = prod[4:8].lstrip("0").replace("hr", "h")
        if overlay_pos_id is not None:
            session += "+" + pos_product(overlay_pos_id)[4:8].lstrip("0").replace(
                "hr", "h"
            )
        return f"{name}, {frame_name(prod)} {session}"
    return f"{name}, {tType}"


#: --detrend-terms vocabulary → gps_analysis staged term group
#: (select_record_groups). "linear" is offset + rate, the staged "secular",
#: which leaves the step amplitudes out (steps are --remove-steps' business)
DETREND_TERM_GROUPS: dict[str, str] = {
    "linear": "secular",
    "periodic": "periodic",
    "transient": "transient",
}
#: what --ref detrend removes when --detrend-terms is not given
DEFAULT_DETREND_TERMS: tuple[str, ...] = ("linear", "periodic")


#: marker colour of the overlaid sub-daily (8 h) series
SESSION_OVERLAY_COLOR = "tab:blue"


def _add_session_overlay(
    fig: Figure,
    sta: str,
    pair: tuple[str, str] | None,
    overlay_pos_id: str,
    series_kw: dict[str, Any],
    *,
    main: tuple[Any, Any, Any, Any],
) -> Figure:
    """Draw another session's series (8 h) behind the main (24 h) one.

    Same frame, reference, view and plate removal as the main series. A
    single station shares the main series' reference position, so the two
    sit where their positions are. A baseline is re-zeroed at its own first
    epoch, so the overlay is shifted by the median difference on common days
    (stated in the legend). The main series is re-drawn on top.
    """
    from gps_plot.sources import gps as gps_source

    overlay = gps_source.fetch(
        sta,
        gps_source.SeriesRequest(**series_kw).replace(pos_id=overlay_pos_id),
        gps_source.BaselineOptions(tol=2.0 / 24.0 / 365.25, check_view=False),
    )
    t8, d8, s8 = overlay.t, np.asarray(overlay.y), np.asarray(overlay.sigma)
    note = None
    if pair is not None:
        # align on the days both have: the median of 8 h minus the daily value
        yearf, data = main[0], np.asarray(main[1])
        day = np.floor(np.asarray(yearf) * 365.25)
        idx = {int(k): i for i, k in enumerate(day)}
        diffs = [
            d8[:, j] - data[:, idx[int(k)]]
            for j, k in enumerate(np.floor(np.asarray(t8) * 365.25))
            if int(k) in idx
        ]
        if diffs:
            d8 = d8 - np.nanmedian(np.array(diffs), axis=0)[:, None]
        note = "median-aligned"
    import geo_dataread.gps_read as gpsr

    x8 = list(gpsr.toDateTime(t8))
    label = "8 h sessions" + (f" ({note})" if note else "")
    fig = addData(
        x8,
        d8,
        s8,
        fig,
        markersize=2.0,
        ecolor="0.75",
        elinewidth=0.3,
        markerfacecolor=SESSION_OVERLAY_COLOR,
        markeredgecolor=SESSION_OVERLAY_COLOR,
        label=label,
    )
    fig = addData(main[3], main[1], main[2], fig, label="24 h")
    fig.axes[0].legend(loc="upper left", fontsize="small")
    return fig


def stdFrame(
    Ylabel: Sequence[str] | None = None,
    Title: str | Sequence[str] | StationTitle | None = None,
    fig: Figure | None = None,
) -> tuple[Figure, Any]:
    """Empty 3-axes frame for a standard GPS time-series plot.

    ``fig=None`` creates a new pyplot figure; passing an existing Figure
    reuses it (cleared) -- the production save path passes the module-level
    reusable figure.  Style is applied once via :func:`init_plot_style`.
    """
    init_plot_style()

    if fig is None:
        fig = plt.figure(figsize=(13, 20))
    else:
        fig.clear()
        fig.set_size_inches(13, 20)
    axes = fig.subplots(nrows=3, ncols=1)
    # explicit geometry (classic-style values + the production hspace) so a
    # reused Figure lays out identically no matter when it was created
    fig.subplots_adjust(
        left=0.125, right=0.9, bottom=0.1, top=0.9, wspace=0.2, hspace=0.1
    )

    if isinstance(Title, StationTitle):
        fig.suptitle(Title.main, y=0.935, x=0.51)
        _add_status_subtitle(axes[0], Title)
    elif isinstance(Title, (list, tuple)):
        fig.suptitle(Title[0], y=0.935, x=0.51)
        axes[0].set_title(Title[1], y=1.01)
    elif isinstance(Title, str):
        axes[0].set_title(Title)

    if Ylabel is None:
        Ylabel = ("North [mm]", "East [mm]", "Up [mm]")

    for i in range(3):
        axes[i].set_ylabel(Ylabel[i], fontsize="x-large", labelpad=0)

    for ax in axes[-1:-4:-1]:
        # tickmarks labels and other axis stuff on each axes
        xax = ax.get_xaxis()

        xax.set_tick_params(
            which="minor", reset=True, direction="inout", length=4, top=False
        )
        xax.set_tick_params(
            which="major", reset=True, direction="inout", length=10, top=False
        )

        if ax is axes[0]:  # special case of top axes
            xax.set_tick_params(
                which="major", reset=True, direction="inout", length=10, top=True
            )
            xax.set_tick_params(
                which="minor", reset=True, direction="inout", length=4, top=True
            )
        else:
            ax.spines["top"].set_visible(False)

    return fig, axes


def setXlim(
    axes: Any,
    xmin: datetime.datetime,
    xmax: datetime.datetime,
    start: datetime.datetime | None = None,
    end: datetime.datetime | None = None,
) -> timedelta:
    """Set the x extent of the plot; returns the plotted period."""
    if start and end:
        space = (end - start) / 40
    elif start and not end:
        space = (xmax - start) / 40
    elif not start and end:
        space = (end - xmin) / 40
    else:
        space = (xmax - xmin) / 40

    if start:
        start = start - space
    else:
        start = xmin - space

    if end:
        end = end + space
    else:
        end = xmax + space

    for ax in axes:
        ax.set_xlim(start, end)

    return end - start


def setYlim(
    fig: Figure,
    ymin: Sequence[float] = (0, 0, 0),
    ymax: Sequence[float] = (0, 0, 0),
    ylim: Sequence[Any] = (),
) -> Figure:
    """Set the y extent of the three component axes.

    ``ylim`` semantics (unchanged from the legacy CLI): one number expands
    each axis by that many mm; two numbers are absolute bounds for all
    axes; three items are per-component ``(lower, upper)`` pairs.
    """
    for i in range(3):
        if len(ylim) == 1:
            fig.axes[i].set_ylim(ymin[i] - float(ylim[0]), ymax[i] + float(ylim[0]))

        if len(ylim) == 2:
            fig.axes[i].set_ylim(float(ylim[0]), float(ylim[1]))

        if len(ylim) == 3:
            fig.axes[i].set_ylim(float(ylim[i][0]), float(ylim[i][1]))

    return fig


def _as_datetime_list(x: Any) -> list[datetime.datetime]:
    """Coerce an epoch array to a list of datetimes.

    Fractional-year float input (the legacy call convention) is converted
    via ``geo_dataread`` (lazy production dep); datetime-like input
    (datetime64 arrays, datetime sequences, pandas indexes) passes through
    without the redundant re-conversion the old code performed.
    """
    arr = np.asarray(x)
    if arr.dtype.kind == "f":
        from geo_dataread.gps_read import toDateTime

        return list(toDateTime(arr))
    if arr.dtype.kind == "M":
        return list(arr.astype("datetime64[us]").astype(object))
    return [v.to_pydatetime() if hasattr(v, "to_pydatetime") else v for v in x]


def stdTimesPlot(
    x: Any,
    y: Any,
    Dy: Any,
    Ylabel: Sequence[str] | None = None,
    Title: str | Sequence[str] | StationTitle | None = None,
    start: datetime.datetime | None = None,
    end: datetime.datetime | None = None,
    ylim: Sequence[float] | Sequence[Sequence[float]] = (),
    warnp: int = -1,
    label: str | None = None,
    fig: Figure | None = None,
    highlight_last: bool | None = None,
) -> Figure:
    """Build a three-component time-series Figure and return it.

    This is the figure-returning seam: it only constructs the Figure --
    saving/showing is the caller's separate step (see :func:`saveFig`), so
    it can be driven from a REPL.  ``x`` may be datetimes or fractional
    years; ``y``/``Dy`` are shape (3, N).

    ``highlight_last`` re-enables the legacy last-point highlight (a
    light-green dot with a black rim, :func:`addPoints`) and is the SAME
    decision as the green "Last datapoint" title fragment.  ``None``
    derives it locally from ``warnp`` via :func:`data_is_current`;
    :func:`plotTime` passes it explicitly, computed on the unmasked epoch
    list, so the marker and the title can never contradict each other.

    One asymmetry is deliberate: in the ``cleaned`` view a flagged final
    epoch is NaN in that component, so no marker is drawn on that axis
    while the title stays green -- the EPOCH is current, the value was set
    aside.  The other two components still get their dot.
    """
    x = _as_datetime_list(x)

    fig, axes = stdFrame(Ylabel, Title, fig=fig)
    period = setXlim(axes, x[0], x[-1], start=start, end=end)
    fig = tsTickLabels(fig, axes, period=period)

    # NaN-safe (the cleaned view masks outlier epochs with NaN); identical
    # values for all-finite data, so the raw path is unchanged
    ymin = [np.nanmin(y[0, :]), np.nanmin(y[1, :]), np.nanmin(y[2, :])]
    ymax = [np.nanmax(y[0, :]), np.nanmax(y[1, :]), np.nanmax(y[2, :])]
    fig = setYlim(fig, ymin=ymin, ymax=ymax, ylim=ylim)

    fig = addData(x, y, Dy, fig, label=label)

    # Last-point highlight (revived BGÓ 2026-07-27, suppressed 2026-07-11):
    # pairs with the green title fragment -- see the docstring.
    if highlight_last is None:
        highlight_last = data_is_current(x[-1], warnp)
    if highlight_last:
        fig = addPoints(x[-1], [y[i][-1] for i in range(3)], fig)

    return fig


#: ``--show-ends``: the baseline's two stations behind the red baseline
#: points — A (the minuend) light, B (the subtrahend) dark.
BASELINE_END_COLORS = ("#a6a6a6", "#4d4d4d")


def _empirical_pair_inputs(
    pair: tuple[str, str], ends: Sequence[Any], path: str | None
) -> tuple[Any, Any, str | None]:
    """``(rho, (k_a, k_b), note)`` from the baseline noise record, or formal.

    Never fails the plot: anything missing warns and returns
    ``(None, None, "formal (…)")`` so the baseline keeps its quadrature σ.
    """
    try:
        import gps_parser as cp
        from geo_dataread import baseline_noise as bn
        from geofunc import local

        rec = bn.read_baseline_noise(Path(path) if path else None)
        cfg = cp.ConfigParser()
        xyz = []
        for s in pair:
            info = cfg.getStationInfo(s)["station"]
            xyz.append(
                local.geodetic_to_ecef(
                    np.radians(float(info["latitude"])),
                    np.radians(float(info["longitude"])),
                    float(info.get("height") or 0.0),
                )
            )
        d_km = float(np.linalg.norm(xyz[0] - xyz[1])) / 1e3
        own = bn.own_shared_fraction(
            (ends[0].yearf, ends[0].data, ends[0].Ddata),
            (ends[1].yearf, ends[1].data, ends[1].Ddata),
        )
        rho, k, how = bn.pair_inputs(rec, *pair, distance_km=d_km, own_rho=own)
        return (
            rho.tolist(),
            (k[0].tolist(), k[1].tolist()),
            f"empirical, {how}; fit {rec.fitted_at[:10]}",
        )
    except Exception as exc:  # noqa: BLE001 - a plot never fails for a sigma reason
        warnings.warn(
            f"baseline {'-'.join(pair)}: empirical σ unavailable ({exc}); showing formal σ",
            UserWarning,
            stacklevel=3,
        )
        return None, None, "formal (empirical unavailable)"


def add_baseline_ends(
    fig: Figure,
    labels: tuple[str, str],
    ends: Sequence[tuple[Any, ...]],
) -> Figure:
    """Draw a baseline's two stations behind it, in two greys, with a legend.

    Presentation only: ``ends`` comes from
    :attr:`geo_dataread.baseline.Baseline.ends` (time axis converted to
    datetimes by the caller), ``labels`` names the two series. Works on any
    figure with three component axes, not only one built by plotTime.

    ``ends`` holds ``(x, data)`` or ``(x, data, sigma)`` per station,
    already referenced to the baseline's first common epoch, so the red
    baseline points are exactly the visible gap between the two grey
    series. With ``sigma`` each station gets its FORMAL error bars, thin
    and in its own grey (BGÓ 2026-10-09: compare them with the baseline's
    empirical σ); drawn at a lower z-order so the baseline stays on top.
    """
    from matplotlib.lines import Line2D

    for end, color in zip(ends, BASELINE_END_COLORS, strict=True):
        x, data = end[0], end[1]
        sigma = end[2] if len(end) > 2 else None
        for i in range(3):
            if sigma is not None:
                fig.axes[i].errorbar(
                    x,
                    data[i],
                    yerr=sigma[i],
                    fmt="none",
                    ecolor=color,
                    elinewidth=0.6,
                    capsize=0,
                    alpha=0.6,
                    zorder=1.4,
                )
            fig.axes[i].plot(
                x,
                data[i],
                linestyle="none",
                marker="o",
                markersize=3.5,  # same as the baseline points (addData)
                markerfacecolor=color,
                markeredgecolor=color,
                zorder=1.5,
            )
    handles = [
        Line2D([], [], ls="none", marker="o", markersize=4, color="r"),
        *(
            Line2D([], [], ls="none", marker="o", markersize=4, color=c)
            for c in BASELINE_END_COLORS
        ),
    ]
    names = [f"{labels[0]} $-$ {labels[1]}", labels[0], labels[1]]
    fig.axes[0].legend(
        handles, names, loc="best", fontsize="small", framealpha=0.8, numpoints=1
    )
    return fig


def addData(
    x: Any,
    y: Any,
    Dy: Any,
    fig: Figure,
    ls: str = "none",
    ecolor: str = "grey",
    elinewidth: float = 0.4,
    marker: str = "o",
    markersize: float = 3.5,
    markerfacecolor: str = "r",
    markeredgecolor: str = "r",
    label: str | None = None,
) -> Figure:
    """Add a three-component data series to the figure."""
    for i in range(3):
        fig.axes[i].errorbar(
            x, y[i], yerr=Dy[i], ls=ls, ecolor=ecolor, elinewidth=elinewidth
        )
        fig.axes[i].plot(
            x,
            y[i],
            linestyle="none",
            marker=marker,
            markersize=markersize,
            markerfacecolor=markerfacecolor,
            markeredgecolor=markeredgecolor,
            label=label,
        )

    return fig


def addPoints(
    x: Any,
    y: Sequence[float],
    fig: Figure,
    marker: str = "o",
    markersize: float = 5.5,
    markerfacecolor: str = "lightgreen",
    markeredgecolor: str = "black",
) -> Figure:
    """Highlight a single epoch on all three component axes."""
    for i in range(3):
        fig.axes[i].plot(
            x,
            y[i],
            linestyle="none",
            marker=marker,
            markersize=markersize,
            markerfacecolor=markerfacecolor,
            markeredgecolor=markeredgecolor,
        )

    return fig


def add_provisional_last(x: Any, py: Any, fig: Figure) -> Figure:
    """Outline a provisional LAST epoch in green, per component.

    The last-point highlight is green; a provisional marker drawn on top of
    it is gold and hid it. Where the newest epoch is provisional in a
    component, redraw it gold with a green edge: pending AND yesterday's.
    """
    last = np.asarray(py, dtype=float)[:, -1]
    for i in range(min(3, last.size)):
        if np.isfinite(last[i]):
            fig.axes[i].plot(
                x[-1],
                last[i],
                linestyle="none",
                marker="o",
                markersize=6.0,
                markerfacecolor=PROVISIONAL_FACE_COLOR,
                markeredgecolor=PROVISIONAL_LAST_EDGE_COLOR,
                markeredgewidth=PROVISIONAL_LAST_EDGE_WIDTH,
            )
    return fig


def addAbortBadge(fig: Figure, component_abort: Sequence[bool]) -> Figure:
    """Mark each aborted component's axis (§3.5a).

    Per axis, NOT per figure: the abort is per component, and a figure-level
    banner would hide the granularity the leaf change exists to provide --
    a station where north aborted but east and up cleaned normally is a
    different object from one where all three failed.

    Deliberately NOT suppressed by ``hide_outliers``: decluttering removes
    DECIDED outliers, and an aborted component is the opposite of decided.
    """
    for c, aborted in enumerate(component_abort):
        if not aborted or c >= len(fig.axes):
            continue
        fig.axes[c].text(
            0.5,
            0.97,
            ABORT_BADGE_TEXT,
            transform=fig.axes[c].transAxes,
            ha="center",
            va="top",
            color=ABORT_BADGE_COLOR,
            fontsize=13,
            bbox={
                "facecolor": ABORT_BADGE_FACE,
                "edgecolor": ABORT_BADGE_COLOR,
                "boxstyle": "round,pad=0.35",
                "alpha": 0.9,
            },
            zorder=10,
        )
    return fig


def addEvent(
    eventDict: Mapping[Any, Sequence[Any]],
    fig: Figure,
    dstr: str = "%Y%m%d",
    **kwargs: Any,
) -> Figure:
    """
    Adding events
    """

    events = eventDict.keys()
    for event in events:
        if len(eventDict[event]) > 0:
            color = eventDict[event][0]
        else:
            color = "r"

        if not isinstance(event, datetime.datetime):
            event = toDatetimel(event, dstr)

        axes = fig.axes
        [ax.axvline(x=event, color=color, zorder=2, **kwargs) for ax in axes]

    return fig


#: Annotation colours for declared steps, split by :attr:`StepRecord.is_seismic`.
#: (The workbench carries its own ``SEISMIC_COLOR``/``TOS_COLOR`` for its event
#: lines; these are the plot driver's spellings of the same two colours.)
SEISMIC_STEP_COLOR: str = "darkred"
EQUIPMENT_STEP_COLOR: str = "darkgreen"


def add_event_lines(
    fig: Figure, events: Sequence[tuple[float, str]], color: str
) -> Figure:
    """Vertical lines with a label on the top axis.

    Lines go through :func:`addEvent` (the existing primitive — ``axvline``
    on every axis); only the text is new, and only on axis 0, because
    repeating it on all three is noise.

    The label is drawn in FULL.  It used to be ``label.split(" ")[0]`` —
    the date and nothing else — which was right while the rest of the
    string was a device count, and silently threw away the equipment names
    the moment they existed.  Date and equipment go on two rotated lines
    so the identifying part stays at the axis edge and the detail runs
    beside it rather than after it.
    """
    from gtimes.timefunc import TimefromYearf

    if not events:
        return fig
    addEvent({TimefromYearf(e): [color] for e, _ in events}, fig)
    ax = fig.axes[0]
    lo, hi = ax.get_ylim()
    mid = (lo + hi) / 2.0
    for epoch, label in events:
        head, sep, tail = label.partition(" (")
        text = f"{head}\n{tail.rstrip(')')}" if sep else label
        ax.text(
            TimefromYearf(epoch),
            mid,
            text,
            rotation=90,
            va="center",
            ha="right",
            fontsize=9,
            linespacing=0.95,
            color=color,
            zorder=6,
        )
    return fig


def step_annotations(
    sta: str,
    steps_catalog: str | Path | None = None,
    *,
    kinds: Sequence[str] | None = None,
) -> list[tuple[float, str, str]]:
    """Declared steps for annotation: ``(epoch, label, colour)`` triples.

    Reads ``steps.yaml`` (the durable declaration catalog) and returns one
    triple per declared step — the label via :attr:`StepRecord.label`, the
    colour from the seismic/equipment split (:attr:`StepRecord.is_seismic`).
    When ``kinds`` is given, only steps whose ``kind`` is in that set are
    annotated (``None`` = all).
    A missing or empty catalog degrades to ``[]``: annotation is an
    enhancement, never a plot failure.
    """
    try:
        from gps_parser.outlier_catalogs import read_steps

        catalog = read_steps(steps_catalog)
    except Exception:
        return []
    kind_set = {k.strip().lower() for k in kinds} if kinds else None
    out: list[tuple[float, str, str]] = []
    for row in catalog.get(sta.upper(), ()):
        if kind_set is not None and (row.kind or "").strip().lower() not in kind_set:
            continue
        out.append(
            (
                float(row.epoch_yearf),
                row.label,
                SEISMIC_STEP_COLOR if row.is_seismic else EQUIPMENT_STEP_COLOR,
            )
        )
    return sorted(out)


def saveFig(
    fileName: str,
    fType: str | Sequence[str],
    fig: Figure,
    png_dpi: float = PNG_DPI,
) -> Figure:
    """Save the figure natively to one or more formats.

    ``fType`` is a single suffix (``"png"``), a comma-separated string
    (``"eps,pdf,png"``) or a sequence of suffixes; each format is one
    direct ``savefig`` -- no external conversion.  The tight bounding box
    is computed once and shared by all formats (the legacy per-save
    ``bbox_inches="tight"`` re-rendered the figure to measure it).
    """
    formats = _save_formats(fType)

    # A missing output directory used to surface as a bare FileNotFoundError
    # deep inside matplotlib (typically a relative -d run from another cwd).
    # Create it, and say where, so a mistyped path is still noticed.
    out_dir = os.path.dirname(fileName)
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)
        print(f"created output directory {os.path.abspath(out_dir)}")

    bbox: Any
    try:
        pad = float(mpl.rcParams["savefig.pad_inches"])
        bbox = fig.get_tightbbox().padded(pad)
    except Exception:  # pragma: no cover - defensive fallback
        bbox = "tight"

    for fmt in formats:
        fig.savefig(
            "%s.%s" % (fileName, fmt),
            bbox_inches=bbox,
            dpi=png_dpi if fmt.lower() == "png" else "figure",
        )

    return fig


def _save_formats(fType: str | Sequence[str]) -> list[str]:
    """Normalize a format spec to a list of file suffixes."""
    if isinstance(fType, str):
        formats = [f.strip() for f in fType.split(",")]
    else:
        formats = [str(f).strip() for f in fType]
    formats = [f.lstrip(".") for f in formats if f]
    if not formats:
        raise ValueError("no output format given: %r" % (fType,))
    return formats


#
# Other functions
#


def convLatex(string: str) -> str:
    """
    Converting Icelandic letters to latex (th as þ and d as ð)
    """
    charlist = {
        "Á": r"\'A",
        "á": r"\'a",
        "ð": r"d",
        "Ó": r"\'O",
        "ó": r"\'o",
        "Ö": r"\"O",
        "ö": r"\"o",
        "Ú": r"\'U",
        "ú": r"\'u",
        "Æ": r"{\AE}",
        "æ": r"{\ae}",
        "Í": r"\'I",
        "í": r"\'i",
        "Ý": r"\'Y",
        "ý": r"\'y",
        "Þ": "TH",
        "þ": "\th",
    }
    for key in charlist:
        if key in string:
            string = string.replace(key, charlist[key])

    return string


def convIcelCh(string):
    """
    Converting Icelandic letters
    """
    charlist = {
        "Á": "\\301",
        "á": "\\341",
        "ð": "\\360",
        "Ó": "\\323",
        "ó": "\\363",
        "Ö": "\\326",
        "ö": "\\366",
        "ú": "\\372",
        "æ": "\\346",
        "Í": "\\315",
        "í": "\\355",
        "Ý": "\\335",
        "ý": "\\375",
        "Þ": "\\336",
        "þ": "\\376",
    }
    for key in charlist:
        if key in string:
            string = string.replace(key, charlist[key])

    return string


#
#   ---  labels and text ---
#


def inpLogo(fig, Logo=None):
    """ """

    asp = 0.45  # Image aspect ratio
    xlen = 0.07
    ylen = xlen * asp
    xpos = fig.axes[0].get_position().xmin + 0.005
    ypos = (fig.axes[0].get_position().ymax - xlen * asp) - 0.005

    aximage = fig.add_axes(
        [xpos, ypos, xlen, ylen], frameon=False, xticks=[], yticks=[]
    )

    if Logo:
        im = image.imread(Logo)
        aximage.imshow(
            im, aspect="auto", interpolation="quadric", filternorm=1, alpha=0.7
        )


def tsTickLabels(fig, axes, period=timedelta(90)):
    """ """

    # major labels in separate layer
    # -----------
    axes[-1].get_xaxis().set_tick_params(
        which="major", direction="inout", length=14, reset=True, pad=10
    )
    axes[0].get_xaxis().set_tick_params(
        which="major",
        direction="inout",
        length=14,
        reset=True,
        labelbottom=True,
        pad=10,
    )
    axes[1].get_xaxis().set_tick_params(
        which="major",
        direction="inout",
        length=14,
        reset=True,
        labelbottom=True,
        pad=10,
    )
    #
    for ax in axes[-1:-4:-1]:
        # tickmarks lables and other and other axis stuff on each axes
        # needs improvement
        ax.grid(
            True,
            which="minor",
            axis="x",
        )
        ax.grid(
            True,
            which="major",
            axis="x",
            color="lightgray",
            linestyle="-",
            linewidth=0.8,
        )
        ax.grid(
            True,
            axis="y",
            color="lightgray",
            linestyle="-",
            linewidth=0.8,
        )

        # --- X axis ---
        xax = ax.get_xaxis()
        xax = tslabels(xax, period=period, locator="minor", formater="minor")
        xax = tslabels(xax, period=period, locator="major", formater="major")

        for tick in xax.get_major_ticks():
            tick.label1.set_horizontalalignment("left")

        xax.label.set_horizontalalignment("center")

        # --- Y axis ---
        yax = ax.get_yaxis()
        yax.set_minor_locator(mpl.ticker.AutoMinorLocator())
        yax.set_tick_params(which="minor", direction="in", length=4)
        yax.set_tick_params(which="major", direction="in", length=10)

    return fig


def tslabels(xax, period=timedelta(90), locator=None, formater=None):
    """
    first attempt to make the time scale on the x axis look good
    """

    if period <= timedelta(hours=36):
        # RTK spans (gpsplot rtk: 6h, 12h, day)
        minorLoc = mpl.dates.HourLocator(byhour=range(0, 24, 2))
        minorFmt = mpl.dates.DateFormatter("%H:%M")

        majorLoc = mpl.dates.DayLocator()
        majorFmt = mpl.dates.DateFormatter("%d %b %y")

    elif period <= timedelta(6):
        minorLoc = mpl.dates.HourLocator(byhour=[0, 8, 16])
        minorFmt = mpl.dates.DateFormatter("%H:%M")

        majorLoc = mpl.dates.DayLocator()
        majorFmt = mpl.dates.DateFormatter("%d %b %y")

    elif period <= timedelta(12):
        minorLoc = mpl.dates.HourLocator(byhour=[0, 8, 16])
        minorFmt = mpl.dates.DateFormatter("%H")

        majorLoc = mpl.dates.AutoDateLocator(maxticks=8)
        majorFmt = mpl.dates.DateFormatter("%d %b %y")

    elif period <= timedelta(18):
        minorLoc = mpl.dates.HourLocator(byhour=[0, 12])
        minorFmt = mpl.dates.DateFormatter("%H")

        majorLoc = mpl.dates.AutoDateLocator()
        majorFmt = mpl.dates.DateFormatter("%d %b %y")

    elif period <= timedelta(30):
        minorLoc = mpl.dates.DayLocator()
        minorFmt = mpl.dates.DateFormatter("%d")
        for label in xax.get_ticklabels("major")[1::2]:
            label.set_visible(False)

        majorLoc = mpl.dates.MonthLocator()
        majorFmt = mpl.dates.DateFormatter("%d %b %Y")

    elif period <= timedelta(193):
        minorLoc = mpl.dates.DayLocator(bymonthday=[1, 7, 14, 21, 28])
        minorFmt = mpl.dates.DateFormatter("%d")

        majorLoc = mpl.dates.MonthLocator(interval=1)
        majorFmt = mpl.dates.DateFormatter("%b %Y")

    elif period <= timedelta(500):
        minorLoc = mpl.dates.MonthLocator()
        minorFmt = mpl.dates.DateFormatter("%b")

        majorLoc = mpl.dates.YearLocator(1, month=1)
        majorFmt = mpl.dates.DateFormatter("%Y")

    elif period <= timedelta(1200):
        minorLoc = mpl.dates.MonthLocator()
        minorFmt = mpl.dates.DateFormatter("%b")
        for label in xax.get_ticklabels("major")[1::2]:
            label.set_visible(False)

        majorLoc = mpl.dates.YearLocator(1)
        majorFmt = mpl.dates.DateFormatter("%Y")

    elif period <= timedelta(2300):
        minorLoc = mpl.dates.MonthLocator(bymonth=[1, 4, 7, 10])
        minorFmt = mpl.dates.DateFormatter("%b")

        majorLoc = mpl.dates.YearLocator(1)
        majorFmt = mpl.dates.DateFormatter("%Y")

    elif period <= timedelta(5000):
        minorLoc = mpl.dates.MonthLocator(bymonth=[1, 4, 7, 10])
        minorFmt = mpl.dates.DateFormatter("%b")
        for label in xax.get_ticklabels("minor")[1::2]:
            label.set_visible(False)

        majorLoc = mpl.dates.YearLocator()
        majorFmt = mpl.dates.DateFormatter("%Y")
        for label in xax.get_ticklabels()[1::2]:
            label.set_visible(False)

    elif period <= timedelta(10000):
        minorLoc = mpl.dates.YearLocator(base=1)
        minorFmt = mpl.dates.DateFormatter("")
        for label in xax.get_ticklabels("minor")[:]:
            label.set_visible(False)

        majorLoc = mpl.dates.YearLocator(base=3)
        majorFmt = mpl.dates.DateFormatter("%Y")

    else:
        minorLoc = mpl.dates.YearLocator(1)
        minorFmt = mpl.dates.DateFormatter("")

        majorLoc = mpl.dates.AutoDateLocator()
        majorFmt = mpl.dates.AutoDateFormatter(majorLoc)

    if locator:
        if locator == "minor":
            xax.set_minor_locator(minorLoc)

        else:
            xax.set_major_locator(majorLoc)

    if formater:
        if formater == "minor":
            xax.set_minor_formatter(minorFmt)

        else:
            xax.set_major_formatter(majorFmt)

        return xax

    else:
        return xax


#
#   --- Private functions ---
#


def __converter(x):
    """
    The data extracted are converted to float and
    occational ******* in the data files needs to handled as NAN

    """
    if x == "********":
        return np.nan
    else:
        return float(x)


#
#   --- develpoment ---
#


def fancytitle(x, y, titlestr, font, ax, **kw):
    """
    function under construction intended to implement a fancy title when plotting time series.
    Using stuff like colorcoating the Last datapoint to indicate if yesterdays data was plotted.

    eventually the aim is to implement something in like ax.fancytext(titlestrlist, fontdictlist)
    where each string in titlestrlist is mapped with corresponding fontdict in fontdictlist

    """
    # fancytitle(0.0, 1.07, titlestr, font, axes[0])

    t = ax.transData
    canvas = ax.figure.canvas

    textlist = [None, None, None, None]
    ex = [None, None, None, None]

    axex = ax.get_window_extent()
    axwidth = axex.width

    for i in range(0, len(titlestr), 2):
        textlist[i] = ax.text(
            x, y, " " + titlestr[i] + " ", fontdict=font[i], transform=t
        )
        textlist[i].draw(canvas.get_renderer())
        ex[i] = textlist[i].get_window_extent()
        textlist[i + 1] = ax.text(
            x, y, " " + titlestr[i + 1] + " ", fontdict=font[i + 1], transform=t
        )
        textlist[i + 1].draw(canvas.get_renderer())
        ex[i + 1] = textlist[i + 1].get_window_extent()

        textwidth = ex[i].width + ex[i + 1].width
        x_tmp = (axwidth - textwidth) / 2 / axwidth

        textlist[i].remove()
        textlist[i + 1].remove()

        textlist[i] = ax.text(
            x_tmp, y, " " + titlestr[i] + " ", fontdict=font[i], transform=t
        )
        textlist[i].draw(canvas.get_renderer())
        t = transforms.offset_copy(textlist[i]._transform, x=ex[i].width, units="dots")
        textlist[i + 1] = ax.text(
            x_tmp, y, " " + titlestr[i + 1] + " ", fontdict=font[i + 1], transform=t
        )
        textlist[i + 1].draw(canvas.get_renderer())
        t = transforms.offset_copy(textlist[i]._transform, x=0, units="dots")
        y = y - 0.06

    return textlist
