"""``gpsplot map``: the runner; options and drawing in :mod:`gps_plot.map`."""

from __future__ import annotations

import argparse

#: options whose value starts with a minus in Iceland (west longitudes)
NEGATIVE_VALUE_OPTIONS = ("--region", "--mogi")


def join_negative_values(argv: list[str]) -> list[str]:
    """``--region -22.8/…`` → ``--region=-22.8/…`` (argparse reads a
    leading ``-`` as an option)."""
    out: list[str] = []
    it = iter(argv)
    for a in it:
        if a in NEGATIVE_VALUE_OPTIONS:
            nxt = next(it, None)
            out.append(a if nxt is None else f"{a}={nxt}")
        else:
            out.append(a)
    return out


def run(prog: str, argv: list[str]) -> int:
    from gps_plot import map as gmap
    from gps_plot.plot_gps_timeseries import _load_config

    config = _load_config()
    parser = gmap.build_parser(argparse.ArgumentParser(prog=prog), config=config)
    return gmap.run(parser.parse_args(join_negative_values(argv)), parser, config)
