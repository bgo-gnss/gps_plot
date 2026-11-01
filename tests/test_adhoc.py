"""--adhoc: argv handling, area resolution, folder and sidecars."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

import pytest

from gps_plot import adhoc
from gps_plot.cli import main

AREAS: dict[str, dict[str, Any]] = {
    "svinafellsheidi": {
        "root": "skridurogflod",
        "stations": ["SVIN", "SVIE"],
        "baselines": ["SVIN-SKFC", "SVIE-SKFC"],
        "stacks": {},
    },
    "oraefajokull": {
        "root": "volcanos",
        "stations": ["SKFC", "KVSK", "SVIN", "SVIE"],
        "baselines": [],
        "stacks": {},
    },
    "seydisfjordur": {
        "root": "skridurogflod",
        "stations": ["SEY1", "SEY2", "SEYD"],
        "baselines": [],
        "stacks": {"all": ["SEY1-SEYD", "SEY2-SEYD"]},
    },
}


def test_no_adhoc_is_none() -> None:
    assert adhoc.split_args(["SVIN-SKFC", "--save", "png"]) is None


@pytest.mark.parametrize(
    ("argv", "area", "label", "rest"),
    [
        (["SVIN-SKFC", "--adhoc"], None, None, ["SVIN-SKFC"]),
        (["--adhoc", "SVIN-SKFC"], None, None, ["SVIN-SKFC"]),
        (["--adhoc", "--save", "pdf", "A"], None, None, ["--save", "pdf", "A"]),
        (["SVIN-SKFC", "--adhoc", "svinafellsheidi"], "svinafellsheidi", None,
         ["SVIN-SKFC"]),
        (["X", "--adhoc=svinafellsheidi/plunge"], "svinafellsheidi", "plunge", ["X"]),
        (["X", "--adhoc", "/2026-10-run"], None, "2026-10-run", ["X"]),
        (["X", "--adhoc=/run-2"], None, "run-2", ["X"]),
    ],
)  # fmt: skip
def test_split_args(argv: list[str], area: Any, label: Any, rest: list[str]) -> None:
    req = adhoc.split_args(argv)
    assert req is not None
    assert (req.area, req.label, list(req.argv)) == (area, label, rest)


def test_note_taken_out() -> None:
    req = adhoc.split_args(["A", "--adhoc", "--note", "plunge vs. ground"])
    assert req is not None and req.note == "plunge vs. ground"
    assert list(req.argv) == ["A"]
    req = adhoc.split_args(["A", "--adhoc", "--note=x y"])
    assert req is not None and req.note == "x y"


@pytest.mark.parametrize(
    "argv",
    [
        ["A", "--note", "x"],  # no --adhoc
        ["A", "--adhoc", "-d", "out"],
        ["A", "--adhoc", "--figDir=out"],
        ["A", "--adhoc=area/../up"],
        ["A", "--adhoc", "--note"],
    ],
)
def test_split_args_refused(argv: list[str]) -> None:
    with pytest.raises(ValueError):
        adhoc.split_args(argv)


def test_targets_of() -> None:
    argv = ["velocity", "SVIN-SKFC", "SVIE", "--last", "90d", "SEY?-SEYD"]
    assert adhoc.targets_of(argv) == ["SVIN-SKFC", "SVIE", "SEY?-SEYD"]


def test_infer_area() -> None:
    pairs = ["SVIN-SKFC", "SVIE-SKFC"]
    assert adhoc.infer_area(pairs, AREAS) == "svinafellsheidi"
    assert adhoc.infer_area(["SKFC-SVIN"], AREAS) == "svinafellsheidi"  # reversed
    assert adhoc.infer_area(["SEY?-SEYD"], AREAS) == "seydisfjordur"  # a stack
    assert adhoc.infer_area(["KVSK"], AREAS) == "oraefajokull"
    with pytest.raises(ValueError, match="several areas"):
        adhoc.infer_area(["SKFC"], AREAS)
    with pytest.raises(ValueError, match="several areas"):
        adhoc.infer_area(["SVIN", "SKFC"], AREAS)  # stations only: no tie-break
    with pytest.raises(ValueError, match="no area"):
        adhoc.infer_area(["SVIN", "SEYD"], AREAS)
    with pytest.raises(ValueError, match="no stations"):
        adhoc.infer_area([], AREAS)


def test_folder(tmp_path: Path) -> None:
    got = adhoc.folder(tmp_path, "svinafellsheidi", AREAS, today=dt.date(2026, 11, 1))
    assert got == tmp_path / "locations/skridurogflod/svinafellsheidi/adhoc/2026-11-01"
    assert adhoc.folder(tmp_path, "oraefajokull", AREAS, "x").name == "x"
    with pytest.raises(ValueError, match="no area"):
        adhoc.folder(tmp_path, "nowhere", AREAS)


def _fake(tmp_path: Path) -> Any:
    return lambda: (AREAS, tmp_path)


def test_run_writes_figure_and_sidecar(tmp_path: Path) -> None:
    seen: list[list[str]] = []

    def runner(argv: list[str]) -> int:
        seen.append(argv)
        out = Path(argv[argv.index("-d") + 1])
        (out / "SVIN_SVIE-SKFC-polar-90d.png").write_bytes(b"png")
        return 0

    argv = ["SVIN-SKFC", "SVIE-SKFC", "--together", "--adhoc", "--note", "test"]
    status = adhoc.run("polar", argv, runner, load=_fake(tmp_path),
                       today=dt.date(2026, 11, 1), revision="abc1234")  # fmt: skip
    assert status == 0
    where = tmp_path / "locations/skridurogflod/svinafellsheidi/adhoc/2026-11-01"
    assert seen[0] == ["SVIN-SKFC", "SVIE-SKFC", "--together", "--save", "png",
                       "-d", str(where)]  # fmt: skip
    doc = json.loads((where / "SVIN_SVIE-SKFC-polar-90d.png.json").read_text())
    assert doc["area"] == "svinafellsheidi" and doc["label"] == "2026-11-01"
    assert doc["note"] == "test" and doc["gps_plot"] == "abc1234"
    assert doc["argv"] == ["gpsplot", "polar", *argv]


def test_run_sidecars_only_for_new_files(tmp_path: Path) -> None:
    where = tmp_path / "locations/volcanos/oraefajokull/adhoc/x"
    where.mkdir(parents=True)
    (where / "old.png").write_bytes(b"old")

    def runner(argv: list[str]) -> int:
        (where / "new.pdf").write_bytes(b"pdf")
        return 0

    adhoc.run("ts", ["SKFC", "--save", "pdf", "--adhoc=oraefajokull/x"], runner,
              load=_fake(tmp_path), revision="r")  # fmt: skip
    assert (where / "new.pdf.json").exists()
    assert not (where / "old.png.json").exists()


@pytest.mark.parametrize(
    ("tool", "argv"),
    [
        ("run", ["--adhoc"]),  # not a figure tool
        ("ts", ["SKFC", "--adhoc"]),  # ambiguous area
        ("ts", ["SKFC", "--adhoc", "-d", "x"]),
        ("ts", ["SKFC", "--adhoc", "nowhere"]),
    ],
)
def test_run_refused(tool: str, argv: list[str], tmp_path: Path,
                     capsys: pytest.CaptureFixture[str]) -> None:  # fmt: skip
    def runner(argv: list[str]) -> int:
        raise AssertionError("must not run")

    assert adhoc.run(tool, argv, runner, load=_fake(tmp_path)) == 2
    assert "--adhoc" in capsys.readouterr().out


def test_dispatcher_routes_adhoc(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, list[str]]] = []

    def fake(tool: str, argv: list[str], runner: Any, **_: Any) -> int:
        calls.append((tool, argv))
        return 0

    monkeypatch.setattr(adhoc, "run", fake)
    assert main(["polar", "SVIN-SKFC", "--adhoc"]) == 0
    assert calls == [("polar", ["SVIN-SKFC", "--adhoc"])]
