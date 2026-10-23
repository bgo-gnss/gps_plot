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
import re
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
    "--label": "the baseline group name (station_areas.yaml baselines:)",
}

#: ``stations:`` values of a product that read station_areas.yaml: the area
#: members (ts), every baseline pair of the area (ts), or one stack figure
#: per baseline group of two or more pairs (``gpsplot stack``)
AREA_KINDS = ("area", "area_baselines", "area_stacks")

#: the plot root of an area whose category has no station_areas.yaml
#: ``category_roots:`` entry (``locations/<root>/<cdn>/...``)
DEFAULT_ROOT = "volcanos"

#: a baseline as written in station_areas.yaml: ``AAAA-BBBB``
_PAIR = re.compile(r"^[A-Z0-9]{4}-[A-Z0-9]{4}$")

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
    stations: tuple[str, ...]  # stations and/or baselines AAAA-BBBB
    argv: tuple[str, ...]  # tool options (no stations, -d, --manifest)
    publish: bool
    tool: str = "ts"  # "ts" or "stack" (one figure of all the stations)
    group: str | None = None  # the baseline group of a stack

    @property
    def name(self) -> str:
        parts = [self.product, self.area, self.group]
        return "/".join(p for p in parts if p)


def load_yaml(path: str | Path) -> Any:
    import yaml

    with open(Path(path).expanduser()) as fh:
        return yaml.safe_load(fh) or {}


def area_table(doc: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """``{cdn slug: {"key", "category", "root", "stations", "baselines"}}``
    from a station_areas.yaml document.

    Only areas with a ``cdn:`` slug have a plot directory; the slug names
    the area everywhere (plot tree, aflogun, events.yaml regions). Areas
    sharing a slug are merged (members and baseline groups); the category
    (``volcanic`` of ``volcanic_areas``) is the first one's, and the root
    its top-level ``category_roots:`` entry (default :data:`DEFAULT_ROOT`).

    Raises:
        ValueError: A malformed ``baselines:`` (see :func:`area_baselines`).
    """
    out: dict[str, dict[str, Any]] = {}
    roots = {str(k): str(v) for k, v in (doc.get("category_roots") or {}).items()}
    for section, areas in doc.items():
        if not section.endswith("_areas") or not isinstance(areas, Mapping):
            continue
        category = section.removesuffix("_areas")
        for key, spec in areas.items():
            cdn = (spec or {}).get("cdn")
            if not cdn:
                continue
            members = [str(s).split("#")[0].strip() for s in spec.get("stations") or ()]
            entry = out.setdefault(
                str(cdn),
                {
                    "key": key,
                    "category": category,
                    "root": roots.get(category, DEFAULT_ROOT),
                    "stations": [],
                    "baselines": {},
                },
            )
            for sta in members:
                if sta and sta not in entry["stations"]:
                    entry["stations"].append(sta)
            try:
                groups = area_baselines(spec.get("baselines"))
            except ValueError as exc:
                raise ValueError(f"station_areas {section}.{key}: {exc}") from None
            for name, pairs in groups.items():
                old = entry["baselines"].get(name)
                if old is not None and old != pairs:
                    raise ValueError(
                        f"station_areas {section}.{key}: baseline group {name!r} "
                        f"differs from the one of another {cdn!r} area"
                    )
                entry["baselines"][name] = pairs
    return out


def area_baselines(value: Any) -> dict[str, list[str]]:
    """An area's ``baselines:`` as ``{group: [AAAA-BBBB, ...]}``.

    A mapping names the groups (each group of two or more pairs is also one
    stack figure); a plain list is one unnamed group (``""``), plotted pair
    by pair only. Pairs are kept as written (no reference-second rule);
    ``# comments`` and case are dropped.

    Raises:
        ValueError: Not a mapping/list, a pair not ``AAAA-BBBB``, a station
            paired with itself, or a pair twice in one group (either way
            round). Across groups a pair may repeat or be reversed: each
            stack fans out from its own reference (AUSV-ELDC in the ELDC
            group, ELDC-AUSV in the AUSV one).
    """
    if value is None:
        return {}
    if isinstance(value, list):
        raw: Mapping[Any, Any] = {"": value}
    elif isinstance(value, Mapping):
        raw = value
    else:
        raise ValueError(f"baselines: a mapping of groups or a list, got {value!r}")
    out: dict[str, list[str]] = {}
    for name, items in raw.items():
        if not isinstance(items, list):
            raise ValueError(f"baselines {name!r}: a list of AAAA-BBBB, got {items!r}")
        pairs: list[str] = []
        for item in items:
            pair = str(item).split("#")[0].strip().upper()
            if not _PAIR.match(pair):
                raise ValueError(f"baselines {name!r}: {item!r} is not AAAA-BBBB")
            a, b = pair.split("-")
            if a == b:
                raise ValueError(
                    f"baselines {name!r}: {pair} pairs a station with itself"
                )
            if pair in pairs or f"{b}-{a}" in pairs:
                raise ValueError(f"baselines {name!r}: {a}-{b}/{b}-{a} twice")
            pairs.append(pair)
        out[str(name)] = pairs
    return out


def area_pairs(groups: Mapping[str, Sequence[str]]) -> list[str]:
    """Every baseline of an area once, in the order written; a reversed
    repeat (B-A after A-B) is the same baseline and is left out."""
    out: list[str] = []
    for pairs in groups.values():
        for pair in pairs:
            a, b = pair.split("-")
            if pair not in out and f"{b}-{a}" not in out:
                out.append(pair)
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
    tool: str = "ts",
) -> list[list[str]]:
    """The ``gpsplot ts`` options of one run; a ``fixedstart`` list adds one each.

    ``args:`` lists are appended last, general to specific: defaults, the
    product, the run, an area override — a later one wins for a
    single-value option (argparse keeps the last), and any of them wins
    over the ``--operational`` presets (those fill only options left at
    their default).

    ``tool="stack"``: the ``gpsplot stack`` options instead — no ts batch
    flags (``--skip-missing``, ``--legacy-names``, ``--operational``), and
    one argument list per period (stack takes a single ``--special``).
    """
    unknown = set(run) - set(RUN_KEYS)
    if unknown:
        raise ValueError(f"plots.yaml run: unknown key(s) {sorted(unknown)}")
    opts = {**defaults, **run}
    stack = tool == "stack"
    if stack and opts.get("operational"):
        raise ValueError("plots.yaml run: operational: is a ts preset, not for stacks")
    base: list[str] = ["--save", str(opts.get("save", "png"))]
    if not stack:
        base.append("--skip-missing")
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
    if specials and stack:
        out += [base + ["--special", str(s)] for s in specials]
    elif specials:
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
    product: str | Sequence[str] | None = None,
    area: str | Sequence[str] | None = None,
) -> list[Job]:
    """All jobs of a plots.yaml document (optionally some products / areas).

    ``product`` and ``area`` are each one name or several; a name plots.yaml
    (or station_areas.yaml) does not have is a ValueError (a typo would
    otherwise just drop it).

    A product's ``stations:`` is ``active``/a list (one job per run), or an
    area kind (:data:`AREA_KINDS`): one job per area and run with the area
    members (``area``), the area's baselines (``area_baselines``), or one
    ``gpsplot stack`` job per baseline group and period (``area_stacks``).
    The ``dir:`` of an area product may use ``{cdn}``, ``{root}`` (the area's
    category root, station_areas.yaml ``category_roots:``) and, for stacks,
    ``{group}``. A product's ``overrides:`` are keyed by area slug, or for
    stacks by ``<slug>/<group>`` (applied after the area's).
    """
    defaults = dict(cfg.get("defaults") or {})
    colours = {**DEFAULT_EVENT_COLOURS, **(cfg.get("event_colours") or {})}
    jobs: list[Job] = []
    all_products = cfg.get("products") or {}
    want_products = _picked(product, all_products, "product")
    want_areas = _picked(area, areas, "area")
    for pname, spec in all_products.items():
        if want_products is not None and pname not in want_products:
            continue
        runs = spec.get("runs") or []
        kind = spec.get("stations")
        if kind in AREA_KINDS:
            names = [a for a in areas if a not in set(spec.get("exclude") or ())]
            for slug in names:
                if want_areas is not None and slug not in want_areas:
                    continue
                jobs += _area_jobs(
                    pname,
                    spec,
                    runs,
                    slug,
                    areas[slug],
                    defaults,
                    colours,
                    events,
                )
        else:
            if want_areas is not None:
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


def _picked(
    value: str | Sequence[str] | None, known: Mapping[str, Any], what: str
) -> set[str] | None:
    """The names picked by ``value`` (None = all); unknown names are an error."""
    if value is None:
        return None
    picked = {value} if isinstance(value, str) else set(value)
    if missing := picked - set(known):
        raise ValueError(
            f"no {what} {', '.join(sorted(missing))} (have {', '.join(known)})"
        )
    return picked


def _area_jobs(
    pname: str,
    spec: Mapping[str, Any],
    runs: Sequence[Mapping[str, Any]],
    slug: str,
    entry: Mapping[str, Any],
    defaults: Mapping[str, Any],
    colours: Mapping[str, str],
    events: Sequence[Mapping[str, Any]],
) -> list[Job]:
    """The jobs of one area product for one area (see :func:`expand`)."""
    kind = spec["stations"]
    groups: Mapping[str, Sequence[str]] = entry.get("baselines") or {}
    if kind == "area":
        targets: list[tuple[str | None, tuple[str, ...]]] = [
            (None, tuple(entry["stations"]))
        ]
    elif kind == "area_baselines":
        pairs = area_pairs(groups)
        targets = [(None, tuple(pairs))] if pairs else []
    else:  # area_stacks: groups of two or more pairs
        targets = [(g, tuple(p)) for g, p in groups.items() if g and len(p) > 1]
    tool = "stack" if kind == "area_stacks" else "ts"
    overrides = spec.get("overrides") or {}
    ev = (
        event_specs(events, [slug, entry["key"]], colours)
        if spec.get("events") == "catalog" and tool == "ts"
        else []
    )
    root = entry.get("root", DEFAULT_ROOT)
    jobs: list[Job] = []
    for group, stations in targets:
        out_dir = str(spec["dir"]).format(cdn=slug, root=root, group=group or "")
        # an area override, then a group's own (``karahnjukar/short``) on top;
        # their args: both apply, the group's last
        area_over = overrides.get(slug) or {}
        group_over = (overrides.get(f"{slug}/{group}") or {}) if group else {}
        over = {**area_over, **group_over}
        over_args = [
            tok for o in (area_over, group_over) for tok in extra_args(o.get("args"))
        ]
        for run in runs:
            publish = bool({**run, **over}.get("publish", False))
            merged = {
                **run,
                **{k: v for k, v in over.items() if k in RUN_KEYS and k != "args"},
            }
            if run.get("fixedstart") is False:  # the run opts out
                merged["fixedstart"] = []
            for argv in run_argv(
                merged, defaults, spec.get("args"), over_args, tool=tool
            ):
                if ev:
                    argv = argv + ["--events", *ev]
                if group:
                    argv = argv + ["--label", group]
                jobs.append(
                    Job(
                        pname,
                        slug,
                        out_dir,
                        stations,
                        tuple(argv),
                        publish,
                        tool=tool,
                        group=group,
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
    stations}``, or ``None`` for "any station" (``all``). A baseline
    ``AAAA-BBBB`` is matched by itself or by either end (``SEYD`` keeps
    every ``SEY?-SEYD``); a stack job is one figure, so it is kept whole
    when any of its series matches.

    Returns:
        The narrowed jobs and the wanted stations no job plots.
    """
    keywords = keywords or {}
    want = [s.upper() for s in wanted]
    out: list[Job] = []
    used: set[str] = set()
    for job in jobs:
        members = [m for m in job.stations if m not in keywords]
        pools = [keywords[k] for k in job.stations if k in keywords]
        keep: list[str] = []
        for w in want:
            got = [m for m in members if w in {m.upper(), *m.upper().split("-")}]
            if not got and "-" not in w:  # through a keyword list (active, all)
                if any(p is None or w in {x.upper() for x in p} for p in pools):
                    got = [w]
            if got:
                used.add(w)
                keep += [m for m in got if m not in keep]
        if not keep:
            continue
        out.append(job if job.tool == "stack" else replace(job, stations=tuple(keep)))
    return out, [s for s in want if s not in used]


def _as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list | tuple) else [value]


def _ymd(value: Any) -> str:
    day = value if isinstance(value, dt.date) else dt.date.fromisoformat(str(value))
    return f"{day:%Y%m%d}"
