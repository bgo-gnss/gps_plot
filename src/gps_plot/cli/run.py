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
Watched areas go first (:func:`products.watched_areas`): plots.yaml
``watch:`` slugs and the region of every events.yaml event with no
``end`` whose onset is under ``watch_days:`` (90) old; within each part
the plots.yaml order holds. ``--changed`` draws only the jobs touching a
station whose files changed since the last whole run
(:mod:`gps_plot.changed`).
``--update-pos [SCRIPT]`` first brings the .pos series up to date (the
laptop's ``update_pos.sh``: new days from okada; plots.yaml
``update_pos:`` names it); when it fails or is missing, the run goes on
with the series as they are, with a warning.

Stale plots: after the jobs, baseline and stack plots in their directories
that the config no longer has (a pair dropped from ``baselines:``, a group
from ``stacks:``), and map-product plots no run or check series writes
any more (a renamed run, :func:`products.stale_map_plots`) are deleted;
``--dry-run`` lists them, ``--keep-stale`` keeps them.

Stations: ``gpsplot run [plots.yaml] [STA ...]`` plots only those stations
of the jobs ``--product``/``--area`` select (each job keeps the ones it
has; ``active`` lists match the operated stations of stations.cfg).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from gps_plot import plot_gps_timeseries as driver
from gps_plot import changed, products

DEFAULT_KEEP_RUNS = 14


def _network_days(workers: int | None) -> int:
    from geo_dataread import network_days

    return int(network_days.main(["-j", str(workers)] if workers else []))


#: plots.yaml ``prepare:`` step -> what it runs (exit status)
PREPARE_STEPS = {"network_days": _network_days}


def update_script(
    arg: str | None, cfg: dict[str, Any], config: Any = None
) -> Path | None:
    """``--update-pos [SCRIPT]``: the script to run, SCRIPT, postprocess.cfg
    ``update_pos`` (per host) or plots.yaml ``update_pos:``; None when the flag
    is not given."""
    if arg is None:
        return None
    from gps_plot.cli.common import config_dir

    per_host = config_dir(config, "update_pos", None) if config is not None else None
    named = arg or per_host or cfg.get("update_pos") or ""
    return Path(str(named)).expanduser() if named else Path()


def update_pos(script: Path, prog: str) -> bool:
    """Run the .pos update ``script`` (its output shown); False, with a
    warning, when it is missing or fails: the plots then use the series
    as they are."""
    if not str(script) or str(script) == ".":
        print(f"{prog}: --update-pos: no script (give one, or plots.yaml "
              "update_pos:); plotting the series as they are")  # fmt: skip
        return False
    if not script.is_file():
        print(f"{prog}: --update-pos: {script} not found; plotting the series "
              "as they are")  # fmt: skip
        return False
    print(f"== update: {script}", flush=True)
    status = subprocess.run([str(script)], check=False).returncode
    if status:
        print(f"{prog}: --update-pos: {script} failed (exit {status}); "
              "plotting the series as they are")  # fmt: skip
        return False
    return True


DESCRIPTION = """\
Make the standard plots defined in plots.yaml (default: the one in the
gpsconfig directory, ~/.config/gpsconfig/plots.yaml).

plots.yaml lists products (standard = every active station; areas = one
plot set per area; baselines = each pair in an area's station_areas.yaml
baselines: list; stacks = one figure per group of its stacks:) and their runs (ref, frame, view,
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
      the Kárahnjúkar stacks: groups, one stack figure per group and period
  gpsplot run --product baselines,stacks --area vatnsfell
      every Vatnsfell baseline plot and stack (a comma list picks several products)
  gpsplot run --product baselines,stacks --area vatnsfell,karahnjukar
      the same for Vatnsfell and Kárahnjúkar (--area takes a comma list too)
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
        action="append",
        default=None,
        help="only these products, comma separated or repeated "
        "(e.g. standard, baselines,stacks, --product a --product b)",
    )
    p.add_argument(
        "--area",
        action="append",
        default=None,
        help="only these areas, comma separated or repeated (cdn slug, e.g. "
        "reykjanes, askja,tjornes)",
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
        "--keep-stale",
        action="store_true",
        help="keep baseline/stack/map plots the config no longer has (default: delete)",
    )
    p.add_argument(
        "--update-pos",
        nargs="?",
        const="",
        default=None,
        metavar="SCRIPT",
        help="first bring the .pos series up to date: run SCRIPT (default: "
        "plots.yaml update_pos:, e.g. update_pos.sh); if it fails, plot the "
        "series as they are",
    )
    p.add_argument(
        "--changed",
        action="store_true",
        help="redraw only the plots of stations whose data changed since the "
        "last whole run (everything when the setup or gpsplot changed)",
    )
    p.add_argument(
        "--check-files",
        action="store_true",
        help="refuse the run when a map layer's file is not on this host "
        "(a map otherwise leaves such a layer out); with --dry-run, the check "
        "of a new configuration",
    )
    p.add_argument(
        "--state",
        metavar="NAME",
        help="this selection's own --changed baseline, saved after a clean "
        "run (a selected run otherwise saves none)",
    )
    p.add_argument(
        "--no-prepare",
        action="store_true",
        help="skip the plots.yaml prepare: steps (e.g. gps-network-days)",
    )
    return p


def _names(arg: str | list[str] | None) -> list[str] | None:
    """``--product a,b`` / ``--area a,b`` (each may be repeated) →
    ``["a", "b"]`` (None = all)."""
    if arg is None:
        return None
    parts = [arg] if isinstance(arg, str) else arg
    return [p for part in parts for p in (s.strip() for s in part.split(",")) if p]


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
            product=_names(args.product),
            area=_names(args.area),
        )
        configured = products.expand(cfg, areas=areas, events=events)
        watched = products.watched_areas(cfg, events, areas)
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
            f"{prog}: no jobs (product={_names(args.product)}, area={_names(args.area)}"
            + (f", stations={' '.join(wanted)})" if wanted else ")")
        )
        return 1
    jobs = products.watch_first(jobs, cfg, watched)
    have = {products.job_area(j, cfg) for j in jobs}
    for slug, why in watched.items():
        if slug in have:
            print(f"watch: {slug} first ({why})")
    from gps_plot.cli.common import plots_root

    root = plots_root(args.plots_dir, config, cfg)
    steps = [] if args.no_prepare else list(cfg.get("prepare") or [])
    unknown = [s for s in steps if s not in PREPARE_STEPS]
    if unknown:
        print(
            f"{prog}: unknown prepare step(s) {unknown} (known: {sorted(PREPARE_STEPS)})"
        )
        return 2
    update = update_script(args.update_pos, cfg, config)
    if args.check_files:
        missing = _check_files(jobs, config)
        if missing:
            print(f"{prog}: {plots_path}: files not on this host, nothing run:")
            print("".join(f"  {line}\n" for line in missing), end="")
            return 2
    whole = not (args.product or args.area or wanted)
    if args.dry_run:
        if update is not None:
            print(f"update: {update}")
        for step in steps:
            print(f"prepare: {step}")
        if args.changed and str(root):
            jobs, _ = _changed_jobs(jobs, config, root, prog, narrow=True)
        for job in jobs:
            who = (
                f"{len(job.stations)} stations: {' '.join(job.stations[:12])}"
                f"{' …' if len(job.stations) > 12 else ''}"
            )
            if job.tool == "map" and not job.stations:
                who = "stations: those in the map's --region"
            print(
                f"{job.name:28s} {'PUB' if job.publish else '   '} "
                f"{root / job.out_dir}\n    {who}"
                f"\n    {job.tool} {' '.join([*([job.kind] if job.kind else []), *job.argv])}"
            )
        if not args.keep_stale and str(root):
            for f in _stale(jobs, configured, root):
                print(f"stale (would delete): {f}")
        return 0
    if not str(root):
        print(f"{prog}: no plots_dir (plots.yaml plots_dir or --plots-dir)")
        return 2

    if update is not None:
        update_pos(update, prog)
    for step in steps:
        print(f"== prepare: {step}")
        if PREPARE_STEPS[step](args.workers):
            print(f"{prog}: prepare step {step} failed; no plots made")
            return 1
    # after the update and the prepare steps: the data as it will be plotted
    jobs, state = _changed_jobs(jobs, config, root, prog,
                                measure=whole or args.changed or bool(args.state),
                                narrow=args.changed, state_name=args.state)  # fmt: skip
    if args.changed and not jobs:
        print(f"{prog}: no station changed since the last whole run, nothing drawn")
        return 0

    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = root / "_runs" / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    status = 0
    published: list[str] = []
    for i, job in enumerate(jobs):
        if not job.stations and job.tool != "map":  # a map: those in --region
            print(f"== {job.name} [{i + 1}/{len(jobs)}]: no stations, skipped")
            continue
        out = root / job.out_dir
        with tempfile.NamedTemporaryFile("r", suffix=".txt") as man:
            argv = [*job.tool_argv(str(out)), "--manifest", man.name]
            if job.tool == "map":  # --region -22.7/… as gpsplot map reads it
                from gps_plot.cli.map import join_negative_values

                argv = join_negative_values(argv)
            if args.workers and job.tool == "ts":
                argv += ["-j", str(args.workers)]
            print(
                f"== {job.name} [{i + 1}/{len(jobs)}]: {job.tool} "
                f"{' '.join([*([job.kind] if job.kind else []), *job.argv])}"
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
    # the map captions beside the maps (aflogun Kort shows them)
    for out_dir, caps in products.map_captions(jobs, configured).items():
        path = root / out_dir / products.CAPTIONS_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(dict(sorted(caps.items())), ensure_ascii=False, indent=1) + "\n",
            encoding="utf-8",
        )
        if any(j.publish and any(d == out_dir for d, *_ in j.captions) for j in jobs):
            published.append(str(path.resolve()))
    with open(run_dir / "publish.manifest", "w") as fh:
        # relative to plots_dir: rsync --files-from=publish.manifest plots_dir/ target
        fh.writelines(
            str(Path(f).relative_to(root.resolve())) + "\n"
            for f in sorted(set(published))
        )
    _prune(root / "_runs", int(cfg.get("keep_runs", DEFAULT_KEEP_RUNS)))
    if not args.keep_stale:
        stale = _stale(jobs, configured, root)
        for f in stale:
            f.unlink(missing_ok=True)
        if stale:
            print(f"{prog}: deleted {len(stale)} plot(s) the config no longer has:")
            print("".join(f"  {f}\n" for f in stale), end="")
    if (whole or args.state) and status == 0 and state:
        # the baseline the next --changed compares with
        changed.save_state(root / changed.state_file(args.state), state)
    print(f"{prog}: {len(jobs)} job(s), run dir {run_dir}, exit {status}")
    return status


def _stale(
    jobs: list[products.Job], configured: list[products.Job], root: Path
) -> list[Path]:
    """Baseline, stack and map-product plots the config no longer has."""
    return [
        *products.stale_baseline_plots(jobs, configured, root),
        *products.stale_map_plots(jobs, configured, root),
    ]


def _tool(name: str, config: Any, **parser_kw: Any) -> tuple[Any, Any]:
    """``(run function, parser)`` of a job tool: ts, stack, map or polar."""
    if name == "map":
        from gps_plot import map as gmap

        parser = gmap.build_parser(
            argparse.ArgumentParser(prog="gpsplot map", **parser_kw), config=config
        )
        return gmap.run, parser
    if name == "polar":
        from gps_plot import polar

        parser = polar.build_parser(
            argparse.ArgumentParser(prog="gpsplot polar", **parser_kw), config=config
        )
        return polar.run, parser
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


def _check_files(jobs: list[products.Job], config: Any) -> list[str]:
    """One line per map job with a layer whose file is not on this host."""
    from gps_plot import gis_layers
    from gps_plot import map as gmap
    from gps_plot.cli.map import join_negative_values

    maps = [job for job in jobs if job.tool == "map"]
    if not maps:
        return []
    parser = _tool("map", config, exit_on_error=False)[1]
    missing = []
    for job in maps:
        args, _ = parser.parse_known_args(
            join_negative_values([job.kind or "station", *job.argv])
        )
        catalogue = gmap.layer_catalogue(args, config)
        try:
            gis_layers.overlays(
                gis_layers.expand(args.layers, catalogue),
                gmap.gis_dir(config), catalogue, strict=True, quiet=True,
            )  # fmt: skip
        except FileNotFoundError as exc:
            missing.append(f"{job.product}: {exc}")
    return missing


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
        probe = [job.kind, *job.argv] if job.kind else ["XXXX", *job.argv]
        if job.tool == "map":
            from gps_plot.cli.map import join_negative_values

            probe = join_negative_values(probe)
        try:
            _, rest = parser.parse_known_args(probe)
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


def _changed_jobs(
    jobs: list[products.Job],
    config: Any,
    root: Path,
    prog: str,
    *,
    measure: bool = True,
    narrow: bool,
    state_name: str | None = None,
) -> tuple[list[products.Job], dict[str, Any]]:
    """(the jobs to run, the new state to save). ``narrow`` (``--changed``):
    only the jobs touching a station whose files changed since the saved
    state (:mod:`gps_plot.changed`); all of them when the setup changed or
    there is no state. ``measure`` False: the jobs as they are, no state."""
    if not measure:
        return jobs, {}
    from gps_plot.cli.common import config_dir

    import gps_plot

    old = changed.load_state(root / changed.state_file(state_name))
    cache = dict(old.get("files") or {})
    pos = config_dir(config, "pos_dir", None)
    if not pos or not Path(pos).is_dir():
        print(f"{prog}: --changed: no pos_dir; drawing everything")
        return jobs, {}
    stations = changed.station_prints(Path(pos), cache)
    setup = changed.setup_print(
        Path(config.config_path).expanduser(), Path(gps_plot.__file__).parent, cache
    )
    state: dict[str, Any] = {"setup": setup, "stations": stations, "files": cache}
    if not narrow:
        return jobs, state
    if not old:
        print(f"{prog}: --changed: no saved state yet, drawing everything")
    elif old.get("setup") != setup:
        print(f"{prog}: --changed: the setup or gpsplot changed, drawing everything")
    else:
        moved = changed.changed_stations(old.get("stations") or {}, stations)
        jobs = changed.narrow(jobs, moved, {"active": _active(config), "all": None})
        print(
            f"{prog}: --changed: {len(moved)} station(s) changed"
            + (
                f" ({' '.join(sorted(moved)[:20])}{' …' if len(moved) > 20 else ''})"
                if moved
                else ""
            )  # fmt: skip
            + f", {len(jobs)} job(s) to draw"
        )
    return jobs, state


def _active(config: Any) -> list[str]:
    """The stations an ``active`` job list stands for: the operated ones
    (stations.cfg) inside the plotting box, as ``gpsplot ts active`` takes them.
    Without the box --changed drew foreign stations, which have no plate."""
    from geo_dataread.pos import pos_stations

    from gps_plot.cli.common import ICELAND_REGION, active_stations, config_dir

    inside = {
        sta
        for sta, _lon, _lat in pos_stations(
            "imo.24hr_itrf14", config_dir(config, "pos_dir", None), ICELAND_REGION
        )
    }
    return [sta for sta in active_stations(config) if sta in inside]


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
