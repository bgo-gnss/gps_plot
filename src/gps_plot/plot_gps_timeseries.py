#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse
import datetime as dt
import os
import sys
import traceback

import gps_parser as cp

# from gtimes.timefunc import TimefromYearf, currTime, TimetoYearf
from gtimes.timefunc import currDatetime

import gps_plot.timesmatplt as tplt


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

    except:
        traceback.print_exc()
        # print >>sys.stderr, top
        print(
            sys.stderr,
            "Unexpected error: %s during processing of %s" % (sys.exc_info()[0], sta),
        )


def exit_gracefully(signum, frame):
    """Exit gracefully on Ctrl-C"""

    current_func = sys._getframe().f_code.co_name + "() >> "

    # restore the original signal handler as otherwise evil things will happen
    # in raw_input when CTRL+C is pressed, and our signal handler is not re-entrant
    signal.signal(signal.SIGINT, original_sigint)

    try:
        if raw_input("\nReally quit? (y/n)> ").lower().startswith("y"):
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

    config = cp.ConfigParser()
    # date to use defaults to today ----
    dstr = "%Y%m%d"  # Default input string

    # initialising  few variables
    start = end = None
    eventDict = {}
    Type_allow = ["TOT", "08h", "JOIN"]
    ref_allow = ["all", "plate", "detrend", "itrf2008"]
    special_allow = ["all", "90d", "year", "full", "fixedstart"]

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
        for sta in statlist:
            tp.compGlobkTimes(sta)

    del kwargs["t"]

    # plotting
    if not (args.special == "all" or args.ref == "all"):
        for sta in stations:
            tryTimes(sta, **kwargs)

    else:
        if args.special == "all" and args.ref == "all":
            del kwargs["ref"]
            del kwargs["special"]
            del ref_allow[0]
            del special_allow[0]

            for ref in ref_allow:
                for special in special_allow:
                    for sta in stations:
                        tryTimes(sta, ref=ref, special=special, **kwargs)

        elif args.special == "all" and not args.ref == "all":
            del kwargs["special"]
            del special_allow[0]

            for special in special_allow:
                for sta in stations:
                    tryTimes(sta, special=special, **kwargs)

        else:
            del kwargs["ref"]
            del ref_allow[0]

            for ref in ref_allow:
                for sta in stations:
                    tryTimes(sta, ref=ref, **kwargs)


if __name__ == "__main__":
    import signal

    # This is used to catch Ctrl-C exits
    original_sigint = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, exit_gracefully)

    main()
