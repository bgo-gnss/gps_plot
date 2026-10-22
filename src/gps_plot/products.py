"""plots.yaml → plot jobs: the standard-plot products (docs/plots-yaml-design.md).

Pure expansion, no plotting: :func:`expand` turns the products of a
plots.yaml into :class:`Job` s — one per (product, area, run), each a
``gpsplot ts`` argument list plus where its files go and whether they are
published. Station lists come from station_areas.yaml (areas) or the
``active`` keyword (standard); event lines from events.yaml by region. The
runner (:mod:`gps_plot.cli.run`) executes the jobs.
"""

from __future__ import annotations

import datetime as dt
import shlex
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

#: run keys understood in plots.yaml and the ts option each becomes
RUN_KEYS = (
    "ref",
    "detrend_terms",
    "special",
    "ylim",
    "uncert",
    "operational",
    "frame",
    "session",
    "view",
    "publish",
    "fixedstart",
    "args",
)

#: ts options a plots.yaml ``args:`` list may not carry: the runner sets
#: them per job (output dir, manifest, workers) or the job's own keys do
#: (``special``/``fixedstart`` make one job per period)
RESERVED_OPTIONS: Mapping[str, str] = {
    "-d": "the product's dir:",
    "--figDir": "the product's dir:",
    "--manifest": "the runner",
    "-j": "gpsplot run -j",
    "--workers": "gpsplot run -j",
    "--special": "the run's special:",
    "-s": "the run's fixedstart:",
    "--start": "the run's fixedstart:",
}

#: event colours when plots.yaml sets none (eruption red, dike blue as the
#: legacy Reykjanes list; the rest BGÓ 2026-10-21)
DEFAULT_EVENT_COLOURS: Mapping[str, str] = {
    "eruption": "r",
    "dike": "b",
    "intrusion": "b",
    "inflation": "g",
    "deep_inflation": "g",
    "deflation": "g",
    "earthquake": "m",
    "swarm": "m",
}


@dataclass(frozen=True)
class Job:
    """One ``gpsplot ts`` invocation of a product."""

    product: str
    area: str | None
    out_dir: str  # relative to plots_dir
    stations: tuple[str, ...]
    argv: tuple[str, ...]  # ts options (no stations, -d, --manifest)
    publish: bool

    @property
    def name(self) -> str:
        return self.product if self.area is None else f"{self.product}/{self.area}"


def load_yaml(path: str | Path) -> Any:
    import yaml

    with open(Path(path).expanduser()) as fh:
        return yaml.safe_load(fh) or {}


def area_table(doc: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """``{cdn slug: {"key", "stations"}}`` from a station_areas.yaml document.

    Only areas with a ``cdn:`` slug have a plot directory; the slug names
    the area everywhere (plot tree, aflogun, events.yaml regions).
    """
    out: dict[str, dict[str, Any]] = {}
    for section, areas in doc.items():
        if not section.endswith("_areas") or not isinstance(areas, Mapping):
            continue
        for key, spec in areas.items():
            cdn = (spec or {}).get("cdn")
            if not cdn:
                continue
            members = [str(s).split("#")[0].strip() for s in spec.get("stations") or ()]
            entry = out.setdefault(str(cdn), {"key": key, "stations": []})
            for sta in members:
                if sta and sta not in entry["stations"]:
                    entry["stations"].append(sta)
    return out


def event_specs(
    events: Sequence[Mapping[str, Any]],
    regions: Sequence[str],
    colours: Mapping[str, str] = DEFAULT_EVENT_COLOURS,
) -> list[str]:
    """``YYYYMMDD,colour`` (the ``--events`` form) for the rows of ``regions``.

    One line per event onset; kinds without a colour are left out.
    """
    out = []
    for ev in events:
        if ev.get("region") not in regions:
            continue
        colour = colours.get(str(ev.get("kind")))
        onset = ev.get("onset")
        if colour is None or onset is None:
            continue
        day = onset if isinstance(onset, dt.date) else dt.date.fromisoformat(str(onset))
        spec = f"{day:%Y%m%d},{colour}"
        if spec not in out:
            out.append(spec)
    return sorted(out)


def extra_args(value: Any) -> list[str]:
    """An ``args:`` value as ts tokens: a string or a list of strings, each
    shell-split (``"--outlier-sigma 3.5"`` is two tokens).

    Raises:
        ValueError: Not a string/list, or a runner-owned option
            (:data:`RESERVED_OPTIONS`).
    """
    if value is None or value == []:
        return []
    if isinstance(value, str):
        items: list[Any] = [value]
    elif isinstance(value, list | tuple):
        items = list(value)
    else:
        raise ValueError(f"plots.yaml args: a string or a list, got {value!r}")
    tokens = [tok for item in items for tok in shlex.split(str(item))]
    for tok in tokens:
        opt = tok.split("=", 1)[0]
        if opt in RESERVED_OPTIONS:
            raise ValueError(
                f"plots.yaml args: {opt} is not allowed here; "
                f"it is set by {RESERVED_OPTIONS[opt]}"
            )
    return tokens


def run_argv(
    run: Mapping[str, Any],
    defaults: Mapping[str, Any],
    product_args: Any = None,
    override_args: Any = None,
) -> list[list[str]]:
    """The ``gpsplot ts`` options of one run; a ``fixedstart`` list adds one each.

    ``args:`` lists are appended last, general to specific: defaults, the
    product, the run, an area override — a later one wins for a
    single-value option (argparse keeps the last), and any of them wins
    over the ``--operational`` presets (those fill only options left at
    their default).
    """
    unknown = set(run) - set(RUN_KEYS)
    if unknown:
        raise ValueError(f"plots.yaml run: unknown key(s) {sorted(unknown)}")
    opts = {**defaults, **run}
    base: list[str] = ["--save", str(opts.get("save", "png")), "--skip-missing"]
    if opts.get("legacy_names", True):
        base.append("--legacy-names")
    for key, flag in (
        ("ref", "--ref"),
        ("frame", "--frame"),
        ("session", "--session"),
        ("view", "--view"),
        ("uncert", "--uncert"),
    ):
        if opts.get(key) is not None:
            base += [flag, str(opts[key])]
    if opts.get("detrend_terms"):
        base += ["--detrend-terms", ",".join(_as_list(opts["detrend_terms"]))]
    if opts.get("ylim") is not None:
        base += ["--ylim", *(str(v) for v in _as_list(opts["ylim"]))]
    if opts.get("operational"):
        base.append("--operational")
    for value in (defaults.get("args"), product_args, run.get("args"), override_args):
        base += extra_args(value)
    out = []
    specials = _as_list(opts.get("special") or [])
    if specials:
        out.append(base + ["--special", ",".join(str(s) for s in specials)])
    for start in _as_list(opts.get("fixedstart") or []):
        out.append(base + ["--special", "fixedstart", "-s", _ymd(start)])
    if not out:
        raise ValueError("plots.yaml run: needs special and/or fixedstart")
    return out


def expand(
    cfg: Mapping[str, Any],
    *,
    areas: Mapping[str, Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]] = (),
    product: str | None = None,
    area: str | None = None,
) -> list[Job]:
    """All jobs of a plots.yaml document (optionally one product / one area)."""
    defaults = dict(cfg.get("defaults") or {})
    colours = {**DEFAULT_EVENT_COLOURS, **(cfg.get("event_colours") or {})}
    jobs: list[Job] = []
    for pname, spec in (cfg.get("products") or {}).items():
        if product is not None and pname != product:
            continue
        runs = spec.get("runs") or []
        if spec.get("stations") == "area":
            names = [a for a in areas if a not in set(spec.get("exclude") or ())]
            for slug in names:
                if area is not None and slug != area:
                    continue
                over = (spec.get("overrides") or {}).get(slug) or {}
                ev = (
                    event_specs(events, [slug, areas[slug]["key"]], colours)
                    if spec.get("events") == "catalog"
                    else []
                )
                for run in runs:
                    publish = bool({**run, **over}.get("publish", False))
                    merged = {
                        **run,
                        **{
                            k: v
                            for k, v in over.items()
                            if k in RUN_KEYS and k != "args"
                        },
                    }
                    if run.get("fixedstart") is False:  # the run opts out
                        merged["fixedstart"] = []
                    for argv in run_argv(
                        merged, defaults, spec.get("args"), over.get("args")
                    ):
                        if ev:
                            argv = argv + ["--events", *ev]
                        jobs.append(
                            Job(
                                pname,
                                slug,
                                str(spec["dir"]).format(cdn=slug),
                                tuple(areas[slug]["stations"]),
                                tuple(argv),
                                publish,
                            )
                        )
        else:
            if area is not None:
                continue
            stations = tuple(_as_list(spec.get("stations") or "active"))
            for run in runs:
                for argv in run_argv(run, defaults, spec.get("args")):
                    jobs.append(
                        Job(
                            pname,
                            None,
                            str(spec["dir"]),
                            stations,
                            tuple(argv),
                            bool(run.get("publish", False)),
                        )
                    )
    return jobs


def select_stations(
    jobs: Sequence[Job],
    wanted: Sequence[str],
    keywords: Mapping[str, Collection[str] | None] | None = None,
) -> tuple[list[Job], list[str]]:
    """Narrow jobs to the ``wanted`` stations (``gpsplot run STA ...``).

    Each job keeps the wanted stations it would plot, in the order given;
    a job left with none is dropped. A station keyword in a job's list
    (``active``) is matched through ``keywords``: ``{keyword: its
    stations}``, or ``None`` for "any station" (``all``).

    Returns:
        The narrowed jobs and the wanted stations no job plots.
    """
    keywords = keywords or {}
    want = [s.upper() for s in wanted]
    out: list[Job] = []
    used: set[str] = set()
    for job in jobs:
        members = {s.upper() for s in job.stations if s not in keywords}
        pool = [keywords[k] for k in job.stations if k in keywords]
        keep = tuple(
            s
            for s in want
            if s in members
            or any(p is None or s in {m.upper() for m in p} for p in pool)
        )
        if keep:
            out.append(replace(job, stations=keep))
            used.update(keep)
    return out, [s for s in want if s not in used]


def _as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list | tuple) else [value]


def _ymd(value: Any) -> str:
    day = value if isinstance(value, dt.date) else dt.date.fromisoformat(str(value))
    return f"{day:%Y%m%d}"
