"""``gpsplot direction``: the runner; options in :mod:`gps_plot.direction`."""

from __future__ import annotations

import argparse


def run(prog: str, argv: list[str]) -> int:
    from gps_plot import direction
    from gps_plot.plot_gps_timeseries import _load_config

    config = _load_config()
    parser = direction.build_parser(argparse.ArgumentParser(prog=prog), config=config)
    return direction.run(parser.parse_args(argv), parser, config)
