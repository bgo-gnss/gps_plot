"""``gpsplot``: one command for the gps_plot plots, one subcommand per kind.

Design: ``docs/cli-subcommands-design.md``. Each subcommand is a
:class:`Command` in :data:`COMMANDS`; adding a plot kind is one entry here
plus a ``run`` function in its own module.

Dispatch is by hand rather than through ``add_subparsers`` so that only the
chosen subcommand is imported and built: the ``ts`` parser reads the
deployed config for its ``--figDir`` help, and the picker pulls in Qt, and
``gpsplot --help`` must need neither.

A first argument that is not a subcommand means ``ts`` (the default), so
``gpsplot HOFN --save pdf`` is ``gpsplot ts HOFN --save pdf``.
"""

from __future__ import annotations

import argparse
import importlib
import sys
import textwrap
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

#: what a subcommand runs: (prog, its argv) -> exit status (None = 0)
Runner = Callable[[str, list[str]], "int | None"]


@dataclass(frozen=True)
class Command:
    name: str
    help: str
    #: "module:function" taking (prog, argv); imported only when chosen
    target: str
    aliases: tuple[str, ...] = field(default=())

    def run(self, argv: list[str]) -> int | None:
        module, func = self.target.split(":")
        runner: Runner = getattr(importlib.import_module(module), func)
        return runner(f"gpsplot {self.name}", argv)


DEFAULT_COMMAND = "ts"

COMMANDS: tuple[Command, ...] = (
    Command(
        "ts",
        "time series of one station or a baseline AAAA-BBBB "
        "(the former plot-gps-timeseries)",
        "gps_plot.cli.ts:run",
        aliases=("timeseries",),
    ),
    Command(
        "stack",
        'several stations and/or baselines in one figure, e.g. "SEY?-SEYD"',
        "gps_plot.cli.stack:run",
    ),
    Command(
        "rtk",
        'RTK baselines from rtk.vedur.is, e.g. "SEY?-SEYD" (--daily adds GLOBK)',
        "gps_plot.cli.rtk:run",
    ),
    Command(
        "rtk-run",
        "the RTK monitoring plots of plots.yaml rtk: (every few minutes)",
        "gps_plot.rtk_run:run",
    ),
    Command(
        "run",
        "every standard-plot product of plots.yaml (or one product/area)",
        "gps_plot.cli.run:run",
    ),
    Command(
        "seasonal",
        "a station's seasonal folded onto one year: data, own and regional",
        "gps_plot.cli.seasonal:run",
    ),
    Command(
        "direction",
        "the direction a baseline or station moved, as a directions: line",
        "gps_plot.cli.direction:run",
    ),
    Command(
        "polar",
        "where a baseline's rover went since an onset: sector + hodograph",
        "gps_plot.cli.polar:run",
    ),
    Command(
        "map",
        "PyGMT maps: station, velocity (windowed fits), deformation, slip",
        "gps_plot.cli.map:run",
    ),
    Command(
        "devviz",
        "analysis-lane dev plot (same as gps-analysis-devviz)",
        "gps_plot.cli.tools:devviz",
    ),
    Command(
        "workbench",
        "detrend workbench (same as gps-detrend-workbench)",
        "gps_plot.cli.tools:workbench",
    ),
    Command(
        "picker",
        "Qt detrend picker (same as gps-detrend-picker-qt)",
        "gps_plot.cli.tools:picker",
    ),
)


def find_command(name: str) -> Command | None:
    for cmd in COMMANDS:
        if name == cmd.name or name in cmd.aliases:
            return cmd
    return None


def _top_parser() -> argparse.ArgumentParser:
    lines = []
    for cmd in COMMANDS:
        names = ", ".join((cmd.name, *cmd.aliases))
        lines.append(f"  {names:<16}{cmd.help}")
    return argparse.ArgumentParser(
        prog="gpsplot",
        usage="gpsplot [COMMAND] [ARGS ...]",
        description=textwrap.fill(
            "GPS plots. 'gpsplot COMMAND --help' shows a command's options; "
            f"without a command, '{DEFAULT_COMMAND}' is used "
            "(gpsplot HOFN --save pdf).",
            width=78,
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="commands:\n" + "\n".join(lines),
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help"):
        _top_parser().print_help()
        return 0
    cmd = find_command(args[0])
    if cmd is None:
        cmd = find_command(DEFAULT_COMMAND)
        assert cmd is not None
    else:
        args = args[1:]
    return cmd.run(args) or 0
