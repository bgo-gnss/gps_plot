#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse
import concurrent.futures
import datetime as dt
import functools
import os
import sys
import traceback

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
        tplt.plotTime(sta, **kwargs)
        print("plotted %s using: %s, %s" % (sta, kwargs["ref"], kwargs["special"]))
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

    raw_read = functools.lru_cache(maxsize=maxsize)(gpsr.openGlobkTimes)

    @functools.wraps(gpsr.openGlobkTimes)
    def cached_openGlobkTimes(sta, Dir=None, tType="TOT"):
        yearf, data, ddata = raw_read(sta, Dir, tType)
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
    Type_allow = ["TOT", "08h", "JOIN"]
    ref_allow = ["all", "plate", "detrend", "itrf2008"]
    special_allow = ["all", "90d", "year", "full", "fixedstart"]
    view_allow = ["raw", "cleaned", "detrended"]

    parser = argparse.ArgumentParser(
        description="Plot tool for GPS time series.",
        epilog="For any issues regarding this program or the GPS system contact, Benni, gsm: 847 4985, email: bgo@vedur.is, or Hildur email: hildur@vedur.is",
    )

    parser.add_argument("Stations", nargs="+", help="List of stations")
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
    parser.add_argument("-s", "--start", type=str, default=None, help="Start of plot")
    parser.add_argument("-e", "--end", type=str, default=None, help="End of plot")
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
        "--fix",
        action="store_true",
        help="plot strictly the period specified, regardless of the data",
    )
    parser.add_argument(
        "--ref",
        type=str,
        default="itrf2008",
        choices=ref_allow,
        help="Reference frame: defaults to itrf2008, remove plate  velocity (plate), Detrend the time series (detrend)",
    )
    parser.add_argument(
        "--view",
        type=str,
        default="raw",
        choices=view_allow,
        help="Data view (geo_dataread apply-on-read toggle): raw (default), "
        "cleaned (outlier epochs masked from the main series and overlaid "
        "as red points), detrended (stored-parameter detrended series — "
        "pure apply of the deployed detrend record, plate-first)",
    )
    parser.add_argument(
        "--tType",
        type=str,
        default="TOT",
        choices=Type_allow,
        help='Choose a three letter string to determain the ??? in mb_STAT_???.data[123] (Gamit time series name convention). defaults to "TOT" standard 24 hour time seriess. "08h" for 8 hour time series',
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
        default=15,
        help="set limit for uncertainty of values ploted in mm.",
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
        default="",
        const=config.getPostprocessConfig("figDir"),
        help="Figure save directory",
    )
    parser.add_argument(
        "-i",
        "--Dir",
        type=str,
        nargs="?",
        default=config.getPostprocessConfig("totPath"),
        help="Time series input directory",
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
    variants = _expand_variants(kwargs, ref_allow[1:], special_allow[1:])

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
