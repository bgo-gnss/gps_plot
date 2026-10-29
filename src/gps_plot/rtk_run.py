"""``gpsplot rtk-run``: the RTK monitoring plots of plots.yaml, every few minutes.

Separate from ``gpsplot run`` (daily, with prepare steps and stale pruning):
RTK plots go stale in minutes, so a timer runs this every 5–10 min. The
plots.yaml ``rtk:`` section says where and what::

    rtk:
      dir: locations/{root}/{cdn}/graphs/rtk
      runs:
        - { name: 6h, args: [--special 6h] }          # RTK alone
        - { name: 12h, args: [--special 12h] }
        - { name: 2d, args: [--special twodays] }
        - { name: 4d, args: [--longterm 4d, --rtk-overlap 1] }
        - { name: 7d, args: [--longterm week, --rtk-overlap 2] }
        - { name: 14d, args: [--longterm 14d, --rtk-overlap 2] }
      overrides:                    # per area slug: args appended to each run
        svinafellsheidi: { args: [--rtk-span week] }

The pairs are each area's ``baselines:`` and ``stacks:`` pairs
(station_areas.yaml) that rtk.vedur.is processes, either way round
(``sources.rtk.rtk_direction``). One plot per pair and run:
``<dir>/A_B-rtk-<name>.png``, and per ``stacks:`` group with two or more RTK
pairs one figure of them all, as rtk_gps draws its multi-baseline plots:
``<dir>/stack_<group>-rtk-<name>.png`` (``stack_args:`` adds options to
every group figure, ``overrides: {<area>/<group>: {args: …}}`` to one) —
stable names, outside the ``A_B-baseline-*`` / ``stack_<g>-<frame>`` names
``gpsplot run`` prunes. Each file is written beside its target and
renamed over it, so a reader never sees half a PNG. A lock file keeps two
runs from overlapping (a run still going makes the next one exit 0). A pair
with nothing to plot (its stations down) is skipped, not an error; exit 1
only when a plot raised.
``rtk-*`` plots the config no longer has are deleted from the rtk dirs.
Nothing is published here (no CDN push yet).
"""

from __future__ import annotations

import argparse
import fcntl
import os
import re
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_DIR = "locations/{root}/{cdn}/graphs/rtk"
DEFAULT_RUNS: tuple[Mapping[str, Any], ...] = (
    # RTK alone, the rtk_gps standard spans (BGÓ 2026-10-29)
    {"name": "6h", "args": ["--special 6h"]},
    {"name": "12h", "args": ["--special 12h"]},
    {"name": "2d", "args": ["--special twodays"]},
    # RTK just overlapping the daily solutions
    {"name": "4d", "args": ["--longterm 4d", "--rtk-overlap 1"]},
    {"name": "7d", "args": ["--longterm week", "--rtk-overlap 2"]},
    {"name": "14d", "args": ["--longterm 14d", "--rtk-overlap 2"]},
)
LOCK_NAME = ".rtk-run.lock"
#: plot files this runner owns in an rtk dir: pairs and groups
_RTK_PNG = re.compile(r"([A-Z0-9]{4}_[A-Z0-9]{4}|stack_[A-Za-z0-9_]+)-rtk-.+\.png")
#: rtk options a run may not set: the runner does, per plot
RESERVED = ("--save", "--name", "-d", "--figDir")


@dataclass(frozen=True)
class RtkJob:
    """One plot of an area and run: a pair, or a group's pairs in one figure."""

    area: str
    pair: str
    run: str
    out_dir: str
    argv: tuple[str, ...]
    #: a group figure: the station_areas.yaml ``stacks:`` name and its pairs
    group: str | None = None
    members: tuple[str, ...] = ()

    @property
    def stem(self) -> str:
        if self.group:
            return f"stack_{self.group}-rtk-{self.run}"
        return f"{self.pair.replace('-', '_')}-rtk-{self.run}"

    @property
    def baselines(self) -> tuple[str, ...]:
        return self.members or (self.pair,)


def _args(value: Any) -> list[str]:
    from gps_plot.products import extra_args

    return extra_args(value, reserved={})  # rtk's own RESERVED, checked in plan()


def plan(
    cfg: Mapping[str, Any],
    areas: Mapping[str, Mapping[str, Any]],
    rtk_dirs: Sequence[str],
    area: Sequence[str] | None = None,
) -> list[RtkJob]:
    """The plots of a plots.yaml ``rtk:`` section (defaults without one).

    Raises:
        ValueError: A run without a name, a duplicate name, a reserved
            option, or an unknown ``--area``.
    """
    from gps_plot.products import area_pairs
    from gps_plot.sources.rtk import rtk_direction

    spec = cfg.get("rtk") or {}
    runs = list(spec.get("runs") or DEFAULT_RUNS)
    names = [str(r.get("name") or "") for r in runs]
    if not all(names) or len(set(names)) != len(names):
        raise ValueError(f"rtk runs need distinct names, got {names}")
    for r in runs:
        bad = [a for a in _args(r.get("args")) if a.split("=")[0] in RESERVED]
        if bad:
            raise ValueError(f"rtk run {r['name']}: {bad} is set by the runner")
    if area:
        unknown = [a for a in area if a not in areas]
        if unknown:
            raise ValueError(f"unknown area(s) {unknown}")
    overrides = spec.get("overrides") or {}
    jobs: list[RtkJob] = []
    for slug, entry in areas.items():
        if area and slug not in area:
            continue
        stacks: Mapping[str, Sequence[str]] = entry.get("stacks") or {}
        groups = {"": entry.get("baselines") or [], **stacks}

        def on_rtk(pairs: Sequence[str]) -> list[str]:
            out = []
            for pair in pairs:
                try:
                    rtk_direction(pair, rtk_dirs)
                except ValueError:
                    continue  # not an RTK baseline
                out.append(pair)
            return out

        pairs = on_rtk(area_pairs(groups))
        # a group figure needs two RTK pairs (rtk_gps's multi-baseline plots)
        figures = {g: m for g, ps in stacks.items() if len(m := on_rtk(ps)) >= 2}
        out_dir = str(spec.get("dir") or DEFAULT_DIR).format(
            cdn=slug, root=entry.get("root", "volcanos")
        )
        extra = _args((overrides.get(slug) or {}).get("args"))
        stack_args = _args(spec.get("stack_args"))
        for r in runs:
            argv = (*_args(r.get("args")), *extra)
            jobs.extend(
                RtkJob(slug, pair, str(r["name"]), out_dir, argv) for pair in pairs
            )
            for g, members in figures.items():
                own = _args((overrides.get(f"{slug}/{g}") or {}).get("args"))
                jobs.append(
                    RtkJob(
                        slug, " ".join(members), str(r["name"]), out_dir,
                        (*argv, *stack_args, *own), group=g, members=tuple(members),
                    )
                )  # fmt: skip
    return jobs


def stale(jobs: Sequence[RtkJob], root: Path) -> list[Path]:
    """``rtk-*`` plots in the jobs' dirs that no job writes any more."""
    keep = {root / j.out_dir / f"{j.stem}.png" for j in jobs}
    out: list[Path] = []
    for d in sorted({root / j.out_dir for j in jobs}):
        if d.is_dir():
            out += [
                f
                for f in sorted(d.iterdir())
                if _RTK_PNG.fullmatch(f.name) and f not in keep
            ]
    return out


def plot(job: RtkJob, root: Path, config: Any) -> bool:
    """Make one plot into a temp dir beside its target, then rename it over.

    False when there was nothing to plot (no RTK and no daily data: a
    station that is down); its previous plot, if any, is left as it was.
    """
    from gps_plot import rtk

    out = root / job.out_dir
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out, prefix=".rtk-") as tmp:
        parser = rtk.build_parser(
            argparse.ArgumentParser(prog=f"rtk-run {job.stem}"), config=config
        )
        argv = [
            *job.baselines,
            *job.argv,
            "--save",
            "png",
            "-d",
            tmp,
            "--name",
            job.stem,
        ]
        rtk.run(parser.parse_args(argv), parser, config)
        made = Path(tmp) / f"{job.stem}.png"
        if made.is_file():
            os.replace(made, out / made.name)
            return True
    return False


def build_parser(prog: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=prog,
        description=(__doc__ or "").split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "plots", nargs="?", default=None, help="plots.yaml (default: config dir)"
    )
    p.add_argument("--plots-dir", default=None, help="instead of plots.yaml plots_dir")
    p.add_argument(
        "--area", action="append", default=None,
        help="only these areas (cdn slug), comma separated or repeated",
    )  # fmt: skip
    p.add_argument("--dry-run", action="store_true", help="list the plots and exit")
    p.add_argument(
        "--keep-stale", action="store_true",
        help="keep rtk plots the config no longer has (default: delete)",
    )  # fmt: skip
    return p


def run(prog: str, argv: list[str]) -> int:
    from geo_dataread.rtk_read import list_baselines, rtk_dir

    from gps_plot import products
    from gps_plot.cli import common
    from gps_plot.cli.run import _cfg_file, _names
    from gps_plot.plot_gps_timeseries import _load_config

    args = build_parser(prog).parse_args(argv)
    config = _load_config()
    plots_path = (
        Path(args.plots)
        if args.plots
        else Path(config.config_path).expanduser() / "plots.yaml"
    )
    cfg = products.load_yaml(plots_path)
    areas = products.area_table(products.load_yaml(_cfg_file(config, cfg, "areas")))
    dirs = list_baselines(rtk_dir(common.config_dir(config, "rtk_dir")))
    try:
        jobs = plan(cfg, areas, dirs, _names(args.area))
    except ValueError as exc:
        print(f"{prog}: {plots_path}: {exc}", file=sys.stderr)
        return 2
    root = Path(args.plots_dir or cfg.get("plots_dir") or "").expanduser()
    if args.dry_run:
        for j in jobs:
            print(
                f"{root / j.out_dir / j.stem}.png\n    rtk {j.pair} {' '.join(j.argv)}"
            )
        return 0
    if not str(root):
        print(
            f"{prog}: no plots_dir (plots.yaml plots_dir or --plots-dir)",
            file=sys.stderr,
        )
        return 2
    root.mkdir(parents=True, exist_ok=True)
    with open(root / LOCK_NAME, "w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(f"{prog}: another rtk-run is still going; skipped")
            return 0
        status, skipped = 0, []
        for j in jobs:
            try:
                if not plot(j, root, config):
                    skipped.append(j.stem)
            except Exception as exc:  # one pair must not sink the rest
                print(f"{prog}: {j.stem}: {exc}", file=sys.stderr)
                status = 1
        if not args.keep_stale:
            for f in stale(jobs, root):
                f.unlink(missing_ok=True)
                print(f"{prog}: deleted stale {f}")
    if skipped:
        print(f"{prog}: nothing to plot (left as they were): {' '.join(skipped)}")
    print(
        f"{prog}: {len(jobs) - len(skipped)}/{len(jobs)} plot(s) under {root}, "
        f"exit {status}"
    )
    return status
