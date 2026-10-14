"""``gpsplot ts``: the time-series plot (formerly ``plot-gps-timeseries``).

The options and the plotting live in :mod:`gps_plot.plot_gps_timeseries`
(``build_parser`` + ``run``), shared with the old command name, so the two
cannot drift apart.
"""

from __future__ import annotations

import argparse

from gps_plot import plot_gps_timeseries as driver


def run(prog: str, argv: list[str]) -> int:
    config = driver._load_config()
    parser = driver.build_parser(argparse.ArgumentParser(prog=prog), config=config)
    driver.run(parser.parse_args(argv), parser, config)
    return 0
