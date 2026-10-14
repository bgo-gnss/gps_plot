"""``gpsplot rtk``: the runner; options and plotting in :mod:`gps_plot.rtk`."""

from __future__ import annotations

import argparse


def run(prog: str, argv: list[str]) -> int:
    from gps_plot import rtk
    from gps_plot.plot_gps_timeseries import _load_config

    config = _load_config()
    parser = rtk.build_parser(argparse.ArgumentParser(prog=prog), config=config)
    return rtk.run(parser.parse_args(argv), parser, config)
