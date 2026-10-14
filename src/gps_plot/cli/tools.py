"""The existing gps_plot tools as ``gpsplot`` subcommands.

They keep their own console scripts and parsers (the picker's emitted
commands name ``gps-detrend-workbench``), so these only hand the argv on.
"""

from __future__ import annotations


def devviz(prog: str, argv: list[str]) -> int:
    from gps_plot.dev_viz import main

    return main(argv)


def workbench(prog: str, argv: list[str]) -> int:
    from gps_plot.detrend_workbench import main

    return main(argv)


def picker(prog: str, argv: list[str]) -> int:  # pragma: no cover - GUI
    from gps_plot.detrend_picker_qt import main

    return main(argv)
