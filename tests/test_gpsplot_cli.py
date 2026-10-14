"""gpsplot: the subcommand dispatcher and its default 'ts'."""

from __future__ import annotations

import argparse
import re

import pytest

from gps_plot import cli


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, list[str]]]:
    seen: list[tuple[str, list[str]]] = []

    def fake(self: cli.Command, argv: list[str]) -> int:
        seen.append((self.name, argv))
        return 0

    monkeypatch.setattr(cli.Command, "run", fake)
    return seen


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["ts", "HOFN", "--save", "pdf"], ("ts", ["HOFN", "--save", "pdf"])),
        (["timeseries", "HOFN"], ("ts", ["HOFN"])),
        (["HOFN", "--save", "pdf"], ("ts", ["HOFN", "--save", "pdf"])),
        (["VFLN-VFLS", "--show-ends"], ("ts", ["VFLN-VFLS", "--show-ends"])),
        (["--save", "pdf", "HOFN"], ("ts", ["--save", "pdf", "HOFN"])),
        (["workbench", "--status"], ("workbench", ["--status"])),
    ],
)
def test_dispatch_and_default_command(
    calls: list[tuple[str, list[str]]], argv: list[str], expected: tuple[str, list[str]]
) -> None:
    assert cli.main(argv) == 0
    assert calls == [expected]


def test_top_help_lists_every_command(
    calls: list[tuple[str, list[str]]], capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main([]) == 0
    out = capsys.readouterr().out
    assert calls == []
    for cmd in cli.COMMANDS:
        assert re.search(rf"^  {cmd.name}\b", out, re.M), cmd.name


def test_command_targets_resolve() -> None:
    import importlib

    for cmd in cli.COMMANDS:
        module, func = cmd.target.split(":")
        assert callable(getattr(importlib.import_module(module), func)), cmd.target


def test_ts_and_old_name_share_one_option_set() -> None:
    from gps_plot import plot_gps_timeseries as driver

    def options(p: argparse.ArgumentParser) -> list[tuple[tuple[str, ...], str]]:
        return [(tuple(a.option_strings), a.dest) for a in p._actions]

    old = driver.build_parser(config=None)
    new = driver.build_parser(argparse.ArgumentParser(prog="gpsplot ts"), config=None)
    assert options(old) == options(new)
    assert old.epilog == new.epilog and old.description == new.description
