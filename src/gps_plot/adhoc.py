"""``--adhoc``: one-off figures kept by area, out of the production tree.

``gpsplot polar SVIN-SKFC SVIE-SKFC --panels profile --adhoc`` writes to
``<plots_dir>/locations/<root>/<area>/adhoc/<label>/`` instead of the
figure directory, next to the area's products but in a folder of its own
(``gpsplot run`` never lists, publishes or prunes it). ``--adhoc`` takes
``[AREA][/LABEL]``: AREA is a cdn slug of station_areas.yaml (left out:
the one area holding every station of the command), LABEL a folder name
(default: today, ``YYYY-MM-DD``). ``--note TEXT`` goes into the sidecar.

Beside each figure the run writes ``<figure>.json``: the command as typed,
the time, the gps_plot commit, the note — enough to make it again, or to
move it into plots.yaml. Design: docs/gpsplot-commands.md (ad hoc figures).
"""

from __future__ import annotations

import datetime as dt
import fnmatch
import getpass
import json
import re
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: subcommands that make figures and take -d (the others refuse --adhoc)
TOOLS = ("ts", "stack", "rtk", "seasonal", "polar", "map")

#: the folder under an area's plot directory
SUBDIR = "adhoc"

#: figure files a run may write (the sidecars are written for these)
FIGURE_SUFFIXES = (".png", ".pdf", ".eps", ".svg")

#: a station or baseline argument, patterns allowed ("SEY?-SEYD")
_STATION = re.compile(r"^[A-Z0-9?*\[\]]{4}(-[A-Z0-9?*\[\]]{4})?$")

#: a folder name: no path parts, no leading dot
_LABEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


@dataclass(frozen=True)
class Request:
    """What ``--adhoc``/``--note`` asked for, taken out of the argv."""

    area: str | None  # cdn slug; None = infer from the stations
    label: str | None  # folder; None = today
    note: str | None
    argv: tuple[str, ...]  # the tool's argv without --adhoc/--note


def split_args(argv: Sequence[str]) -> Request | None:
    """Take ``--adhoc [AREA][/LABEL]`` and ``--note TEXT`` out of ``argv``.

    None when there is no ``--adhoc``. A bare ``--adhoc`` (followed by an
    option, a station or nothing) infers the area.

    Raises:
        ValueError: ``--note`` without ``--adhoc``, ``-d``/``--figDir``
            given as well, or a bad LABEL.
    """
    rest: list[str] = []
    spec: str | None = None
    found = False
    note: str | None = None
    args = list(argv)
    i = 0
    while i < len(args):
        tok = args[i]
        if tok == "--adhoc" or tok.startswith("--adhoc="):
            found = True
            if "=" in tok:
                spec = tok.split("=", 1)[1]
            elif i + 1 < len(args) and _is_spec(args[i + 1]):
                spec = args[i + 1]
                i += 1
        elif tok == "--note" or tok.startswith("--note="):
            if "=" in tok:
                note = tok.split("=", 1)[1]
            elif i + 1 < len(args):
                note = args[i + 1]
                i += 1
            else:
                raise ValueError("--note needs a text")
        else:
            rest.append(tok)
        i += 1
    if not found:
        if note is not None:
            raise ValueError("--note goes with --adhoc")
        return None
    if any(t in ("-d", "--figDir") or t.startswith(("--figDir=", "-d")) for t in rest):
        raise ValueError("--adhoc chooses the directory: leave out -d/--figDir")
    area, _, label = (spec or "").partition("/")
    if label and not _LABEL.match(label):
        raise ValueError(f"--adhoc label {label!r}: letters, digits, _ . - only")
    return Request(area or None, label or None, note, tuple(rest))


def _is_spec(tok: str) -> bool:
    """An ``--adhoc`` value (a lower-case slug, maybe /label), not a station
    or an option."""
    if tok.startswith("-") or _STATION.match(tok):
        return False
    return tok[:1].islower() or tok.startswith("/")


def targets_of(argv: Sequence[str]) -> list[str]:
    """The stations and baselines of a tool's argv, as typed (patterns kept).

    Only bare ``AAAA``/``AAAA-BBBB`` words count; a map's KIND (lower case)
    and option values that look like neither are skipped.
    """
    out: list[str] = []
    for tok in argv:
        if not tok.startswith("-") and _STATION.match(tok) and tok not in out:
            out.append(tok)
    return out


def infer_area(targets: Sequence[str], areas: Mapping[str, Mapping[str, Any]]) -> str:
    """The one area (cdn slug) of a command's stations and baselines.

    Areas holding every station (patterns match any member; a baseline's
    stations count as members) are the candidates; when several are and the
    command has baselines, the one listing them all (``baselines:`` or a
    ``stacks:`` group, either way round) wins — SVIN-SKFC is Svínafellsheiði
    though Öræfajökull has both stations too.

    Raises:
        ValueError: No stations, or none or several areas fit.
    """
    stations = list(dict.fromkeys(s for t in targets for s in t.split("-")))
    if not stations:
        raise ValueError(
            "--adhoc: no stations to find the area from; give --adhoc AREA"
        )
    hits: dict[str, set[str]] = {}
    for slug, area in areas.items():
        pairs = set(area.get("baselines") or ())
        for group in (area.get("stacks") or {}).values():
            pairs.update(group)
        pairs |= {"-".join(reversed(p.split("-"))) for p in pairs}
        members = set(area.get("stations") or ())
        for pair in pairs:
            members.update(pair.split("-"))
        if all(fnmatch.filter(members, sta) for sta in stations):
            hits[slug] = pairs
    asked = [t for t in targets if "-" in t]
    if len(hits) > 1 and asked:
        listing = {
            s: p for s, p in hits.items() if all(fnmatch.filter(p, t) for t in asked)
        }
        hits = listing or hits
    if len(hits) == 1:
        return next(iter(hits))
    which = " ".join(targets)
    if not hits:
        raise ValueError(f"--adhoc: no area holds all of {which}; give --adhoc AREA")
    raise ValueError(
        f"--adhoc: {which} fit several areas ({', '.join(sorted(hits))}); "
        "give --adhoc AREA"
    )


def folder(
    plots_dir: Path,
    area: str,
    areas: Mapping[str, Mapping[str, Any]],
    label: str | None = None,
    today: dt.date | None = None,
) -> Path:
    """``<plots_dir>/locations/<root>/<area>/adhoc/<label or YYYY-MM-DD>``.

    Raises:
        ValueError: ``area`` is no cdn slug of station_areas.yaml.
    """
    if area not in areas:
        raise ValueError(
            f"--adhoc: {area!r} is no area (cdn slug of station_areas.yaml): "
            + ", ".join(sorted(areas))
        )
    name = label or f"{today or dt.date.today():%Y-%m-%d}"
    return plots_dir / "locations" / str(areas[area]["root"]) / area / SUBDIR / name


def snapshot(where: Path) -> dict[Path, float]:
    """Figure files under ``where`` and their modification times."""
    if not where.is_dir():
        return {}
    return {
        p: p.stat().st_mtime
        for p in where.rglob("*")
        if p.is_file() and p.suffix in FIGURE_SUFFIXES
    }


def written(before: Mapping[Path, float], after: Mapping[Path, float]) -> list[Path]:
    """Files new or rewritten between two snapshots."""
    return sorted(p for p, t in after.items() if before.get(p) != t)


def commit() -> str:
    """The gps_plot commit (``abc1234``, ``-dirty`` when changed), else the
    installed version, else ``unknown``."""
    here = Path(__file__).resolve().parent
    try:
        sha = subprocess.run(
            ["git", "-C", str(here), "describe", "--always", "--dirty", "--abbrev=7"],
            capture_output=True, text=True, check=True, timeout=10,
        ).stdout.strip()  # fmt: skip
        if sha:
            return sha
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        from importlib.metadata import version

        return f"v{version('gps_plot')}"
    except Exception:  # noqa: BLE001 - metadata missing in odd installs
        return "unknown"


def sidecar(
    figure: Path,
    *,
    command: Sequence[str],
    area: str,
    label: str,
    note: str | None,
    when: dt.datetime | None = None,
    revision: str | None = None,
) -> Path:
    """Write ``<figure>.json`` beside the figure; its path."""
    doc = {
        "figure": figure.name,
        "command": " ".join(_quote(a) for a in command),
        "argv": list(command),
        "area": area,
        "label": label,
        "note": note,
        "created": f"{when or dt.datetime.now(dt.UTC):%Y-%m-%dT%H:%M:%SZ}",
        "user": _user(),
        "gps_plot": revision or commit(),
    }
    out = figure.with_name(figure.name + ".json")
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n")
    return out


def _quote(arg: str) -> str:
    import shlex

    return shlex.quote(arg)


def _user() -> str:
    try:
        return getpass.getuser()
    except Exception:  # noqa: BLE001 - no login name in some containers
        return "unknown"


def load_areas_and_root() -> tuple[dict[str, dict[str, Any]], Path]:
    """station_areas.yaml areas and plots.yaml ``plots_dir`` of the deployed
    config (the files ``gpsplot run`` reads).

    Raises:
        ValueError: plots.yaml has no ``plots_dir``.
    """
    from gps_plot import plot_gps_timeseries as driver
    from gps_plot import products
    from gps_plot.cli.run import _cfg_file

    config = driver._load_config()
    cfg = products.load_yaml(Path(config.config_path).expanduser() / "plots.yaml")
    root = str(cfg.get("plots_dir") or "")
    if not root:
        raise ValueError("--adhoc: plots.yaml has no plots_dir")
    areas = products.area_table(products.load_yaml(_cfg_file(config, cfg, "areas")))
    return areas, Path(root).expanduser()


def run(
    tool: str,
    argv: Sequence[str],
    runner: Callable[[list[str]], int | None],
    *,
    load: Callable[[], tuple[dict[str, dict[str, Any]], Path]] = load_areas_and_root,
    today: dt.date | None = None,
    revision: str | None = None,
) -> int | None:
    """Run ``tool`` with ``--adhoc`` resolved, then write the sidecars.

    ``runner`` gets the tool's argv (``-d <folder>`` added, ``--save png``
    when no ``--save``). Returns the runner's status; 2 and a message for a
    refused request.
    """
    try:
        req = split_args(argv)
        if req is None:
            return runner(list(argv))
        if tool not in TOOLS:
            raise ValueError(f"--adhoc: only for {', '.join(TOOLS)}, not {tool}")
        areas, plots_dir = load()
        area = req.area or infer_area(targets_of(req.argv), areas)
        where = folder(plots_dir, area, areas, req.label, today)
    except ValueError as exc:
        print(f"gpsplot {tool}: {exc}")
        return 2
    where.mkdir(parents=True, exist_ok=True)
    tool_argv = list(req.argv)
    if not any(a == "--save" or a.startswith("--save=") for a in tool_argv):
        tool_argv += ["--save", "png"]
    tool_argv += ["-d", str(where)]
    before = snapshot(where)
    status = runner(tool_argv)
    files = written(before, snapshot(where))
    command = ["gpsplot", tool, *argv]
    rev = revision or commit()
    for f in files:
        sidecar(f, command=command, area=area, label=where.name, note=req.note,
                revision=rev)  # fmt: skip
    print(f"ad hoc ({area}): {where}" + ("" if files else " — no figure written"))
    return status
