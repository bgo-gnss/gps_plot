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


def build_parser(prog: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=prog,
        description="Plot every product of plots.yaml (or one product/area).",
    )
    p.add_argument(
        "plots",
        nargs="?",
        default=None,
        help="plots.yaml (default: plots.yaml in the gpsconfig directory)",
    )
    p.add_argument("--plots-dir", default=None, help="override plots_dir")
    p.add_argument("--product", default=None, help="only this product")
    p.add_argument("--area", default=None, help="only this area (cdn slug)")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="print the jobs (stations, directory, ts options) and exit",
    )
    p.add_argument("-j", "--workers", type=int, default=None)
    p.add_argument(
        "--no-prepare",
        action="store_true",
        help="skip the plots.yaml prepare: steps (e.g. gps-network-days)",
    )
    return p


def run(prog: str, argv: list[str]) -> int:
    args = build_parser(prog).parse_args(argv)
    config = driver._load_config()
    cfg_dir = Path(config.config_path).expanduser()
    plots_path = Path(args.plots) if args.plots else cfg_dir / "plots.yaml"
    cfg = products.load_yaml(plots_path)
    areas = products.area_table(products.load_yaml(_cfg_file(config, cfg, "areas")))
    events = products.load_yaml(_cfg_file(config, cfg, "events")).get("events") or []
    try:
        jobs = products.expand(
            cfg, areas=areas, events=events, product=args.product, area=args.area
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
    if not jobs:
        print(f"{prog}: no jobs (product={args.product}, area={args.area})")
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
                f"\n    ts {' '.join(job.argv)}"
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
            if args.workers:
                argv += ["-j", str(args.workers)]
            print(f"== {job.name} [{i + 1}/{len(jobs)}]: ts {' '.join(job.argv)}")
            parser = driver.build_parser(
                argparse.ArgumentParser(prog="gpsplot ts"), config=config
            )
            try:
                status |= driver.run(parser.parse_args(argv), parser, config) or 0
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


def _check_options(jobs: list[products.Job], config: Any) -> list[tuple[str, str]]:
    """Parse every job's ts options before anything runs (an ``args:`` typo
    must stop the run at the start, not fail one job an hour in)."""
    parser = driver.build_parser(
        argparse.ArgumentParser(prog="gpsplot ts", exit_on_error=False),
        config=config,
    )
    bad: list[tuple[str, str]] = []
    seen: set[tuple[str, ...]] = set()
    for job in jobs:
        if job.argv in seen:
            continue
        seen.add(job.argv)
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
