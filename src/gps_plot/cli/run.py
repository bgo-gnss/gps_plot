"""``gpsplot run``: every standard-plot product of plots.yaml in one go.

Expands plots.yaml (:mod:`gps_plot.products`) into ``gpsplot ts`` jobs and
runs them in this process through the same driver and batch contract.
Each run gets ``<plots_dir>/_runs/<UTC stamp>/`` with one manifest per job
group, ``publish.manifest`` (only ``publish: true`` runs: the rsync input)
and the summary; the oldest run directories beyond ``keep_runs`` are pruned.
Exit status 1 when any plot failed. Design: docs/plots-yaml-design.md.

Run order: the ``prepare:`` steps of plots.yaml run first, in their order
(``network_days`` = ``gps-network-days``: the network-day and common-mode
lists the cleaned view reads), then the products. ``--no-prepare`` skips
them (a quick re-run of one area); a failed step ends the run.

Stations: ``gpsplot run [plots.yaml] [STA ...]`` plots only those stations
of the jobs ``--product``/``--area`` select (each job keeps the ones it
has; ``active`` lists match the operated stations of stations.cfg).
"""

from __future__ import annotations

import argparse
import datetime as dt
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

from gps_plot import plot_gps_timeseries as driver
from gps_plot import products

DEFAULT_KEEP_RUNS = 14


def _network_days(workers: int | None) -> int:
    from geo_dataread import network_days

    return int(network_days.main(["-j", str(workers)] if workers else []))


#: plots.yaml ``prepare:`` step -> what it runs (exit status)
PREPARE_STEPS = {"network_days": _network_days}


DESCRIPTION = """\
Make the standard plots defined in plots.yaml (default: the one in the
gpsconfig directory, ~/.config/gpsconfig/plots.yaml).

plots.yaml lists products (standard = every active station; areas = one
plot set per area; baselines = the chosen station pairs of each area;
stacks = one figure per baseline group) and their runs (ref, frame, view,
periods, options). Each (product, area, run) becomes one `gpsplot ts` job
(a stack: one `gpsplot stack` job per group and period). The prepare: steps run first (network_days = gps-network-days: the
network bad days and common mode the cleaned view uses), then the jobs.

Output goes under plots_dir (the CDN tree). Each run also writes
plots_dir/_runs/<UTC stamp>/: one manifest per job and publish.manifest
(only the publish: true runs; the rsync input). Exit status 1 if any plot
failed, 2 if plots.yaml or a job's ts options are wrong (checked before
anything runs).

Selecting: --product and --area pick jobs; station codes after them keep
only those stations in the picked jobs (a job without any is dropped,
a station no job has is reported as "left out"). A station also picks the
baselines it is an end of; a stack is kept whole.
"""

EPILOG = """\
examples:
  gpsplot run --dry-run
      show every job (stations, output directory, full ts line); plots nothing
  gpsplot run -j 12
      the whole production run: prepare, then all products, 12 workers
  gpsplot run --product standard
      only the standard (all active stations) plots
  gpsplot run --product areas --area reykjanes
      only the Reykjanes area plots (area = the cdn slug in station_areas.yaml)
  gpsplot run SENG THOB
      SENG and THOB in every product that has them (standard + their areas)
  gpsplot run seng,thob --area reykjanes --no-prepare
      a quick re-plot of two stations in one area, without gps-network-days
  gpsplot run --product stacks --area karahnjukar
      the Kárahnjúkar baseline groups, one stack figure per group and period
  gpsplot run --product baselines,stacks --area vatnsfell
      every Vatnsfell baseline plot and stack (a comma list picks several products)
  gpsplot run SEYD --product baselines
      every Seyðisfjörður baseline with SEYD as an end (SEY1-SEYD ... SEY9-SEYD)
  gpsplot run RHOF --product standard --dry-run
      what would be plotted for RHOF in the standard product
  gpsplot run ~/test/plots.yaml --plots-dir /tmp/plots
      another plots.yaml, written to another directory (not the CDN tree)
"""


def build_parser(prog: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=prog,
        description=DESCRIPTION,
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "targets",
        nargs="*",
        metavar="plots.yaml|STA",
        help="a plots.yaml (a .yaml/.yml path) and/or station codes, space or "
        "comma separated, any case: only these stations of the selected jobs",
    )
    p.add_argument(
        "--plots-dir",
        default=None,
        help="write under this directory instead of plots.yaml plots_dir",
    )
    p.add_argument(
        "--product",
        default=None,
        help="only these products, comma separated (e.g. standard, baselines,stacks)",
    )
    p.add_argument(
        "--area",
        default=None,
        help="only this area (cdn slug, e.g. reykjanes, askja, tjornes)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="print the jobs (stations, directory, ts options) and exit",
    )
    p.add_argument(
        "-j",
        "--workers",
        type=int,
        default=None,
        help="parallel plot workers per job (and for the prepare steps)",
    )
    p.add_argument(
        "--no-prepare",
        action="store_true",
        help="skip the plots.yaml prepare: steps (e.g. gps-network-days)",
    )
    return p


def _products(arg: str | None) -> list[str] | None:
    """``--product a,b`` → ``["a", "b"]`` (None = every product)."""
    if arg is None:
        return None
    return [p for p in (s.strip() for s in arg.split(",")) if p]


def run(prog: str, argv: list[str]) -> int:
    parser = build_parser(prog)
    args = parser.parse_args(argv)
    plots_arg, wanted = _split_targets(args.targets)
    if len(plots_arg) > 1:
        parser.error(f"one plots.yaml, got {' '.join(plots_arg)}")
    config = driver._load_config()
    cfg_dir = Path(config.config_path).expanduser()
    plots_path = Path(plots_arg[0]) if plots_arg else cfg_dir / "plots.yaml"
    cfg = products.load_yaml(plots_path)
    events = products.load_yaml(_cfg_file(config, cfg, "events")).get("events") or []
    try:
        areas = products.area_table(products.load_yaml(_cfg_file(config, cfg, "areas")))
        jobs = products.expand(
            cfg,
            areas=areas,
            events=events,
            product=_products(args.product),
            area=args.area,
        )
    except ValueError as exc:
        print(f"{prog}: {plots_path}: {exc}")
        return 2
    bad = _check_options(jobs, config)
    if bad:
        print(f"{prog}: {plots_path}: ts options refused, nothing run:")
        for name, err in bad:
            print(f"  {name}: {err}")
        return 2
    if wanted:
        jobs, unused = products.select_stations(
            jobs, wanted, {"active": _active(config), "all": None}
        )
        if unused:
            print(f"{prog}: not in the selected jobs, left out: {' '.join(unused)}")
    if not jobs:
        print(
            f"{prog}: no jobs (product={args.product}, area={args.area}"
            + (f", stations={' '.join(wanted)})" if wanted else ")")
        )
        return 1
    root = Path(args.plots_dir or cfg.get("plots_dir") or "").expanduser()
    steps = [] if args.no_prepare else list(cfg.get("prepare") or [])
    unknown = [s for s in steps if s not in PREPARE_STEPS]
    if unknown:
        print(
            f"{prog}: unknown prepare step(s) {unknown} (known: {sorted(PREPARE_STEPS)})"
        )
        return 2
    if args.dry_run:
        for step in steps:
            print(f"prepare: {step}")
        for job in jobs:
            print(
                f"{job.name:28s} {'PUB' if job.publish else '   '} "
                f"{root / job.out_dir}\n    {len(job.stations)} stations: "
                f"{' '.join(job.stations[:12])}{' …' if len(job.stations) > 12 else ''}"
                f"\n    {job.tool} {' '.join(job.argv)}"
            )
        return 0
    if not str(root):
        print(f"{prog}: no plots_dir (plots.yaml plots_dir or --plots-dir)")
        return 2

    for step in steps:
        print(f"== prepare: {step}")
        if PREPARE_STEPS[step](args.workers):
            print(f"{prog}: prepare step {step} failed; no plots made")
            return 1

    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = root / "_runs" / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    status = 0
    published: list[str] = []
    for i, job in enumerate(jobs):
        if not job.stations:
            print(f"== {job.name} [{i + 1}/{len(jobs)}]: no stations, skipped")
            continue
        out = root / job.out_dir
        with tempfile.NamedTemporaryFile("r", suffix=".txt") as man:
            argv = [*job.stations, *job.argv, "-d", str(out), "--manifest", man.name]
            if args.workers and job.tool == "ts":
                argv += ["-j", str(args.workers)]
            print(
                f"== {job.name} [{i + 1}/{len(jobs)}]: {job.tool} {' '.join(job.argv)}"
            )
            run_tool, parser = _tool(job.tool, config)
            try:
                status |= run_tool(parser.parse_args(argv), parser, config) or 0
            except SystemExit as exc:  # an argument error ends this job only
                print(f"== {job.name}: job failed (exit {exc.code})")
                status = 1
            files = [line.strip() for line in open(man.name) if line.strip()]
        tag = job.name.replace("/", "_")
        with open(run_dir / f"{tag}.manifest", "a") as fh:
            fh.writelines(f + "\n" for f in files)
        if job.publish:
            published += files
    with open(run_dir / "publish.manifest", "w") as fh:
        # relative to plots_dir: rsync --files-from=publish.manifest plots_dir/ target
        fh.writelines(
            str(Path(f).relative_to(root.resolve())) + "\n"
            for f in sorted(set(published))
        )
    _prune(root / "_runs", int(cfg.get("keep_runs", DEFAULT_KEEP_RUNS)))
    print(f"{prog}: {len(jobs)} job(s), run dir {run_dir}, exit {status}")
    return status


def _tool(name: str, config: Any, **parser_kw: Any) -> tuple[Any, Any]:
    """``(run function, parser)`` of a job tool: ``ts`` or ``stack``."""
    if name == "stack":
        from gps_plot import stack

        parser = stack.build_parser(
            argparse.ArgumentParser(prog="gpsplot stack", **parser_kw), config=config
        )
        return stack.run, parser
    parser = driver.build_parser(
        argparse.ArgumentParser(prog="gpsplot ts", **parser_kw), config=config
    )
    return driver.run, parser


def _check_options(jobs: list[products.Job], config: Any) -> list[tuple[str, str]]:
    """Parse every job's options before anything runs (an ``args:`` typo
    must stop the run at the start, not fail one job an hour in)."""
    parsers: dict[str, Any] = {}
    bad: list[tuple[str, str]] = []
    seen: set[tuple[str, ...]] = set()
    for job in jobs:
        if (job.tool, *job.argv) in seen:
            continue
        seen.add((job.tool, *job.argv))
        if job.tool not in parsers:
            parsers[job.tool] = _tool(job.tool, config, exit_on_error=False)[1]
        parser = parsers[job.tool]
        try:
            _, rest = parser.parse_known_args(["XXXX", *job.argv])
        except argparse.ArgumentError as exc:
            bad.append((job.name, str(exc)))
            continue
        except SystemExit:  # the few errors argparse still exits on
            bad.append((job.name, "refused by the ts parser (see above)"))
            continue
        if rest:
            bad.append((job.name, f"unrecognized: {' '.join(rest)}"))
    return bad


def _split_targets(targets: list[str]) -> tuple[list[str], list[str]]:
    """Positional words -> (plots.yaml paths, station codes)."""
    plots: list[str] = []
    stations: list[str] = []
    for word in targets:
        if word.lower().endswith((".yaml", ".yml")) or "/" in word:
            plots.append(word)
        else:
            stations += [s.strip().upper() for s in word.split(",") if s.strip()]
    return plots, stations


def _active(config: Any) -> list[str]:
    """The operated stations (stations.cfg) an ``active`` job list stands for."""
    from gps_plot.cli.common import active_stations

    return list(active_stations(config))


def _cfg_file(config: Any, cfg: dict[str, Any], what: str) -> Path:
    """station_areas.yaml / events.yaml: plots.yaml key, else the deployed one."""
    default = {"areas": "station_areas.yaml", "events": "events.yaml"}[what]
    name = cfg.get(f"{what}_file") or default
    path = Path(name).expanduser()
    return path if path.is_absolute() else Path(config.config_path) / path


def _prune(runs: Path, keep: int) -> None:
    old = sorted(p for p in runs.iterdir() if p.is_dir())[:-keep] if keep > 0 else []
    for p in old:
        shutil.rmtree(p, ignore_errors=True)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(run("gpsplot run", sys.argv[1:]))
