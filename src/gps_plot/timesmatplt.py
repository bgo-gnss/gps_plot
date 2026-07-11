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
from collections.abc import Sequence
from datetime import timedelta
from typing import Any

import matplotlib as mpl
import matplotlib.figure
import matplotlib.image as image
import matplotlib.pyplot as plt
import matplotlib.style
import numpy as np
from gtimes.timefunc import currDate, currDatetime, currTime, currYearfDate, toDatetimel
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

    # station full name from the station config, marker as fallback
    try:
        import gps_parser as cp

        stName = cp.ConfigParser().get_config(sta, "station_name")
    except Exception:
        stName = sta
    stName = convLatex(stName)

    NameStr = "%s (%s)" % (stName, sta)
    if sta == "hekla":
        NameStr = "Hekla (Summit)"
    refFr = "Reference frame: %s" % ref
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


def plotTime(
    sta: str,
    start: datetime.datetime | None = None,
    end: datetime.datetime | None = None,
    save: str | Sequence[str] | None = None,
    ylim: Sequence[float] | Sequence[Sequence[float]] = (),
    special: str | None = None,
    ref: str = "itrf2008",
    figDir: str = "",
    events: dict[Any, Any] | None = None,
    fix: bool = False,
    Dir: str | None = None,
    tType: str = "TOT",
    uncert: int = 15,
    logo: bool = True,
    fig: Figure | None = None,
) -> Figure:
    """Plot a standard GPS North/East/Up time series for one station.

    Reads the series (``geo_dataread``), builds the figure and either saves
    it (``save`` = format or list/comma-string of formats, e.g. ``"png"``
    or ``"eps,pdf,png"`` -- each written by one native ``savefig``) or
    shows it interactively.  Returns the Figure (REPL-friendly).
    """
    # heavy production deps are imported lazily so the module (and the
    # figure-building seam) stays importable without them
    import geo_dataread.gps_read as gpsr

    if not save:
        mpl.use("WebAgg")

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

    # graph title reference-frame string
    if ref == "plate":
        import geofunc.geofunc as gf

        refTitle = gf.plateFullname(gf.plateDict()[sta])
    elif ref == "detrend":
        refTitle = "Detrended"
    else:
        refTitle = ref.upper()

    yearf, data, Ddata, offset = gpsr.getData(
        sta, fstart=fstart, fend=fend, ref=ref, Dir=Dir, tType=tType, uncert=uncert
    )
    if yearf is None or len(yearf) == 0:
        raise ValueError("no data for station %s" % sta)

    # single yearf -> datetime conversion (was done twice before)
    x = list(gpsr.toDateTime(yearf))
    firstpoint, lastpoint = x[0], x[-1]

    warnp = -1  # data through currDate(warnp) is labelled green
    Title = make_title(sta, lastpoint, ref=refTitle, warnp=warnp)

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
    )

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

    if logo:
        inpLogo(fig)

    if save:
        filend = "-%s" % (ref,)
        if tType != "TOT":
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

        fileName = os.path.join(figDir, sta + filend)
        saveFig(fileName, save, fig)
    else:
        plt.show()

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
) -> Figure:
    """Build a three-component time-series Figure and return it.

    This is the figure-returning seam: it only constructs the Figure --
    saving/showing is the caller's separate step (see :func:`saveFig`), so
    it can be driven from a REPL.  ``x`` may be datetimes or fractional
    years; ``y``/``Dy`` are shape (3, N).
    """
    x = _as_datetime_list(x)

    fig, axes = stdFrame(Ylabel, Title, fig=fig)
    period = setXlim(axes, x[0], x[-1], start=start, end=end)
    fig = tsTickLabels(fig, axes, period=period)

    ymin = [min(y[0, :]), min(y[1, :]), min(y[2, :])]
    ymax = [max(y[0, :]), max(y[1, :]), max(y[2, :])]
    fig = setYlim(fig, ymin=ymin, ymax=ymax, ylim=ylim)

    fig = addData(x, y, Dy, fig, label=label)
    # Last-point highlight suppressed (BGÓ 2026-07-11): the legacy trigger
    # (Timestamp == date) never fired in production, so no marker is drawn.
    # ``addPoints`` is retained as a reusable helper if the highlight is revived.

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


def addEvent(eventDict, fig, dstr="%Y%m%d", **kwargs):
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

    if period <= timedelta(6):
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
