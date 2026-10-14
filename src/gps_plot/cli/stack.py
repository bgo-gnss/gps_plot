"""``gpsplot stack``: the runner; options and plotting in :mod:`gps_plot.stack`."""

from __future__ import annotations

import argparse


def run(prog: str, argv: list[str]) -> int:
    from gps_plot import stack
    from gps_plot.plot_gps_timeseries import _load_config

    config = _load_config()
    parser = stack.build_parser(argparse.ArgumentParser(prog=prog), config=config)
    return stack.run(parser.parse_args(argv), parser, config)
