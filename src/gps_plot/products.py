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
import os
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
    "areas",  # only these area slugs (an area product's run)
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
    "--label": "the stack group name (station_areas.yaml stacks:)",
}

#: plots.yaml keys of a map product (``tool: map``) and of each of its runs
MAP_KEYS = ("tool", "map", "dir", "stations", "args", "runs", "ts_stations",
            "layers", "gmt", "gmt_config", "area", "label")  # fmt: skip
#: keys of a plots.yaml ``map_categories:`` / ``map_areas:`` entry: the
#: defaults of every map of that kind of area / of that area
MAP_AREA_KEYS = ("layers", "gmt", "gmt_config", "args")
MAP_RUN_KEYS = ("name", "args", "publish", "layers", "gmt", "gmt_config", "caption")
#: where gpsplot run writes the captions of a map dir (aflogun Kort shows
#: them under the images): {file stem: text}
CAPTIONS_FILE = "captions.json"
#: map kinds a product may draw (gpsplot map KIND)
MAP_KINDS = ("velocity", "deformation", "station")

#: map options a plots.yaml ``args:`` list may not carry
MAP_RESERVED_OPTIONS: Mapping[str, str] = {
    "-d": "the product's dir:",
    "--figDir": "the product's dir:",
    "--manifest": "the runner",
    "--name": "the run's name:",
    "--save": "plots.yaml defaults: save:",
}

#: ``stations:`` values of a product that read station_areas.yaml: the area
#: members (ts), every baseline pair of the area (ts), or one stack figure
#: per stacks: group (``gpsplot stack``)
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
    tool: str = "ts"  # "ts", "stack" (one figure of all), "map" or "polar"
    group: str | None = None  # the stacks: group of a stack
    kind: str | None = None  # a map's KIND (velocity, deformation, station)
    #: (dir, file stem, text) of the captions of the files it writes
    captions: tuple[tuple[str, str, str], ...] = ()

    def tool_argv(self, out_dir: str) -> list[str]:
        """The tool's whole argument list, writing to ``out_dir`` (a map's
        KIND first, then its stations)."""
        head = [self.kind] if self.kind else []
        return [*head, *self.stations, *self.argv, "-d", out_dir]

    @property
    def name(self) -> str:
        parts = [self.product, self.area, self.group]
        return "/".join(p for p in parts if p)


def load_yaml(path: str | Path) -> Any:
    import yaml

    with open(Path(path).expanduser()) as fh:
        return yaml.safe_load(fh) or {}


def area_table(doc: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """``{cdn slug: {"key", "category", "root", "stations", "baselines",
    "stacks", "directions"}}`` from a station_areas.yaml document.

    Only areas with a ``cdn:`` slug have a plot directory; the slug names
    the area everywhere (plot tree, aflogun, events.yaml regions). Areas
    sharing a slug are merged (members, baselines and stacks); the category
    (``volcanic`` of ``volcanic_areas``) is the first one's, and the root
    its top-level ``category_roots:`` entry (default :data:`DEFAULT_ROOT`).

    Raises:
        ValueError: A malformed ``baselines:`` / ``stacks:`` /
            ``directions:`` (see :func:`area_baselines`, :func:`area_stacks`,
            :func:`area_directions`).
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
                    "name": str(spec.get("name_is") or spec.get("name") or key),
                    "category": category,
                    "root": roots.get(category, DEFAULT_ROOT),
                    "stations": [],
                    "baselines": [],
                    "stacks": {},
                    "directions": {},
                },
            )
            for sta in members:
                if sta and sta not in entry["stations"]:
                    entry["stations"].append(sta)
            try:
                pairs = area_baselines(spec.get("baselines"))
                groups = area_stacks(spec.get("stacks"))
                dirs = area_directions(spec.get("directions"))
            except ValueError as exc:
                raise ValueError(f"station_areas {section}.{key}: {exc}") from None
            entry["baselines"] = area_pairs({"": entry["baselines"], key: pairs})
            for name, group in groups.items():
                old = entry["stacks"].get(name)
                if old is not None and old != group:
                    raise ValueError(
                        f"station_areas {section}.{key}: stack {name!r} "
                        f"differs from the one of another {cdn!r} area"
                    )
                entry["stacks"][name] = group
            entry["directions"].update(dirs)
    return out


def area_directions(value: Any) -> dict[str, dict[str, Any]]:
    """An area's ``directions:``: per pair, the fixed axes of ``--axes
    motion|slope`` (``gpsplot direction`` prints the lines)::

        directions:
          SVIE-SKFC: {motion: 334}             # azimuth [°] the rover moves
          SVIN-SKFC: {motion: 292, slope: [210, 28]}   # downslope az, dip [°]

    Returns ``{PAIR: {"motion": float, "slope": (az, dip)}}`` (each key only
    when given; pairs upper-cased, written as the plotted pair).

    Raises:
        ValueError: Not a mapping, a key not ``AAAA-BBBB``, an entry without
            motion or slope, an unknown key, or a value out of range.
    """
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"directions: a mapping of AAAA-BBBB, got {value!r}")
    out: dict[str, dict[str, Any]] = {}
    for key, entry in value.items():
        pair = str(key).strip().upper()
        if not _PAIR.match(pair):
            raise ValueError(f"directions: {key!r} is no AAAA-BBBB")
        if not isinstance(entry, Mapping) or not entry:
            raise ValueError(f"directions {pair}: {{motion: AZ, slope: [AZ, DIP]}}")
        unknown = set(entry) - {"motion", "slope"}
        if unknown:
            raise ValueError(f"directions {pair}: unknown {sorted(unknown)}")
        got: dict[str, Any] = {}
        if "motion" in entry:
            az = entry["motion"]
            if not isinstance(az, (int, float)) or not 0 <= az < 360:
                raise ValueError(f"directions {pair}: motion {az!r} not in 0..360")
            got["motion"] = float(az)
        if "slope" in entry:
            sl = entry["slope"]
            if (
                not isinstance(sl, (list, tuple))
                or len(sl) != 2
                or not all(isinstance(x, (int, float)) for x in sl)
                or not 0 <= sl[0] < 360
                or not 0 <= sl[1] < 90
            ):
                raise ValueError(
                    f"directions {pair}: slope {sl!r} is no [azimuth 0..360, dip 0..90]"
                )
            got["slope"] = (float(sl[0]), float(sl[1]))
        out[pair] = got
    return out


def directions_table(doc: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Every area's ``directions:`` in one ``{PAIR: {...}}``.

    Raises:
        ValueError: A malformed entry, or one pair given differently in two
            areas.
    """
    out: dict[str, dict[str, Any]] = {}
    for slug, area in area_table(doc).items():
        for pair, got in area["directions"].items():
            if out.get(pair, got) != got:
                raise ValueError(f"station_areas: directions {pair} differ ({slug})")
            out[pair] = got
    return out


def area_baselines(value: Any) -> list[str]:
    """An area's ``baselines:``: the pairs plotted one by one (``AAAA-BBBB``).

    Pairs are kept as written (no reference-second rule); ``# comments`` and
    case are dropped. Stacked figures are a separate list (:func:`area_stacks`).

    Raises:
        ValueError: Not a list (groups belong under ``stacks:``), a pair not
            ``AAAA-BBBB``, a station paired with itself, or a pair twice
            (either way round).
    """
    if value is None:
        return []
    if isinstance(value, Mapping):
        raise ValueError(
            "baselines: a list of AAAA-BBBB (named groups go under stacks:)"
        )
    if not isinstance(value, list):
        raise ValueError(f"baselines: a list of AAAA-BBBB, got {value!r}")
    return _pairs("baselines", value)


def area_stacks(value: Any) -> dict[str, list[str]]:
    """An area's ``stacks:`` as ``{group: [AAAA-BBBB, ...]}``: one stacked
    figure per group (named by it), two or more pairs each.

    Across groups (and with ``baselines:``) a pair may repeat or be
    reversed: each stack fans out from its own reference (AUSV-ELDC in the
    ELDC stack, ELDC-AUSV in the AUSV one).

    Raises:
        ValueError: Not a mapping of lists, a group with fewer than two
            pairs, or a bad pair (see :func:`area_baselines`).
    """
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"stacks: a mapping of groups, got {value!r}")
    out: dict[str, list[str]] = {}
    for name, items in value.items():
        if not isinstance(items, list):
            raise ValueError(f"stacks {name!r}: a list of AAAA-BBBB, got {items!r}")
        pairs = _pairs(f"stacks {name!r}", items)
        if len(pairs) < 2:
            raise ValueError(f"stacks {name!r}: a stack needs two or more pairs")
        out[str(name)] = pairs
    return out


def _pairs(where: str, items: Sequence[Any]) -> list[str]:
    """Validated ``AAAA-BBBB`` pairs, none twice (either way round)."""
    pairs: list[str] = []
    for item in items:
        pair = str(item).split("#")[0].strip().upper()
        if not _PAIR.match(pair):
            raise ValueError(f"{where}: {item!r} is not AAAA-BBBB")
        a, b = pair.split("-")
        if a == b:
            raise ValueError(f"{where}: {pair} pairs a station with itself")
        if pair in pairs or f"{b}-{a}" in pairs:
            raise ValueError(f"{where}: {a}-{b}/{b}-{a} twice")
        pairs.append(pair)
    return pairs


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


def extra_args(value: Any, reserved: Mapping[str, str] | None = None) -> list[str]:
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
    reserved = RESERVED_OPTIONS if reserved is None else reserved
    for tok in tokens:
        opt = tok.split("=", 1)[0]
        if opt in reserved:
            raise ValueError(
                f"plots.yaml args: {opt} is not allowed here; "
                f"it is set by {reserved[opt]}"
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
    if tool == "polar":
        return _polar_argv(opts, defaults, product_args, run, override_args)
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


#: run keys a polar product does not take (they are time-series options)
POLAR_UNUSED = ("ref", "detrend_terms", "ylim", "uncert", "operational", "session",
                "view")  # fmt: skip


def _polar_argv(
    opts: Mapping[str, Any],
    defaults: Mapping[str, Any],
    product_args: Any,
    run: Mapping[str, Any],
    override_args: Any,
) -> list[list[str]]:
    """The ``gpsplot polar`` options of one run: one list per period
    (``special:`` each, ``fixedstart:`` as ``-s``)."""
    bad = [k for k in POLAR_UNUSED if run.get(k) is not None]
    if bad:
        raise ValueError(f"plots.yaml polar run: {bad} are time-series keys")
    base: list[str] = ["--save", str(opts.get("save", "png"))]
    if opts.get("frame") is not None:
        base += ["--frame", str(opts["frame"])]
    for value in (defaults.get("args"), product_args, run.get("args"), override_args):
        base += extra_args(value)
    out = [base + ["--special", str(s)] for s in _as_list(opts.get("special") or [])]
    out += [base + ["-s", _ymd(s)] for s in _as_list(opts.get("fixedstart") or [])]
    if not out:
        raise ValueError("plots.yaml polar run: needs special and/or fixedstart")
    return out


def _area_filter(pname: str, value: Any, areas: Mapping[str, Any]) -> set[str] | None:
    """A product's or run's ``areas:`` (None: every area); unknown slugs refused."""
    if value is None:
        return None
    picked = {str(a) for a in _as_list(value)}
    if missing := picked - set(areas):
        raise ValueError(
            f"plots.yaml {pname}: areas: {', '.join(sorted(missing))} is no cdn "
            "slug of station_areas.yaml"
        )
    return picked


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
    ``gpsplot stack`` job per ``stacks:`` group and period (``area_stacks``).
    The ``dir:`` of an area product may use ``{cdn}``, ``{root}`` (the area's
    category root, station_areas.yaml ``category_roots:``) and, for stacks,
    ``{group}``. A product's ``overrides:`` are keyed by area slug, or for
    stacks by ``<slug>/<group>`` (applied after the area's). ``areas:`` on an
    area product, or on one of its runs, keeps only those area slugs (e.g.
    the rotated-axes runs for the landslide areas). ``tool: polar`` with
    ``stations: area_baselines`` is one ``gpsplot polar`` per area, run and
    period (a figure per pair); with ``area_stacks`` one ``--together``
    figure per ``stacks:`` group.
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
        if spec.get("tool") == "map":
            # an --area filter keeps the maps of those areas (area:)
            if want_areas is None or spec.get("area") in want_areas:
                jobs += map_jobs(pname, spec, defaults, areas, cfg, events)
            continue
        if spec.get("tool") == "polar" and kind not in (
            "area_baselines",
            "area_stacks",
        ):
            raise ValueError(
                f"plots.yaml {pname}: tool: polar needs area_baselines or area_stacks"
            )
        if kind in AREA_KINDS:
            only = _area_filter(pname, spec.get("areas"), areas)
            for run in runs:
                _area_filter(pname, run.get("areas"), areas)
            names = [a for a in areas if a not in set(spec.get("exclude") or ())]
            for slug in names:
                if want_areas is not None and slug not in want_areas:
                    continue
                if only is not None and slug not in only:
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


def map_option_args(
    pname: str, where: Mapping[str, Any], prefix: str = ""
) -> list[str]:
    """The ``layers:``, ``gmt:`` and ``gmt_config:`` of a map product, a run
    or ``defaults:`` (there ``map_layers:``, ``map_gmt:``, ``map_gmt_config:``)
    as gpsplot map options: ``layers: [volcanic, rural]`` → ``--layers
    volcanic,rural``; ``gmt: {calderas: {pen: 2p,red}}`` (or a list of
    ``STEP:KEY=VALUE``) → ``--gmt calderas:pen=2p,red``; ``gmt_config:
    {MAP_FRAME_TYPE: plain}`` → ``--gmt-config MAP_FRAME_TYPE=plain``. The
    run's come after the product's, so a run adds layers and its GMT
    arguments win.

    Raises:
        ValueError: a value of the wrong shape.
    """
    out: list[str] = []
    layers = where.get(f"{prefix}layers")
    if layers:
        names = layers.split(",") if isinstance(layers, str) else layers
        if not isinstance(names, list | tuple):
            raise ValueError(f"plots.yaml {pname}: layers: a list of names")
        # --layers=…: a -NAME (take it out) would read as an option
        out.append("--layers=" + ",".join(str(n).strip() for n in names))
    gmt = where.get(f"{prefix}gmt")
    if isinstance(gmt, Mapping):
        for step, kwargs in gmt.items():
            if not isinstance(kwargs, Mapping):
                raise ValueError(
                    f"plots.yaml {pname}: gmt: {step}: a mapping KEY: VALUE"
                )
            for key, value in kwargs.items():
                spec = f"{step}:{key}" if value is True else f"{step}:{key}={value}"
                out += ["--gmt", spec]
    elif gmt:
        out += [tok for spec in _as_list(gmt) for tok in ("--gmt", str(spec))]
    config = where.get(f"{prefix}gmt_config")
    if config:
        if not isinstance(config, Mapping):
            raise ValueError(f"plots.yaml {pname}: gmt_config: a mapping KEY: VALUE")
        out += [t for k, v in config.items() for t in ("--gmt-config", f"{k}={v}")]
    return out


def map_area_defaults(
    pname: str,
    spec: Mapping[str, Any],
    areas: Mapping[str, Mapping[str, Any]],
    cfg: Mapping[str, Any],
) -> tuple[str | None, list[Mapping[str, Any]]]:
    """A map product's ``area:`` (a station_areas.yaml cdn slug): its plot
    dir (``locations/<root>/<area>/maps``, used when the product gives no
    ``dir:``) and its defaults, in order: plots.yaml ``map_categories:`` of
    the area's category (``volcanic`` of ``volcanic_areas``), then
    ``map_areas:`` of the area. No ``area:``: (None, []).

    Raises:
        ValueError: an area station_areas.yaml has not, or an entry key
            not in :data:`MAP_AREA_KEYS`.
    """
    slug = spec.get("area")
    if not slug:
        return None, []
    area = areas.get(str(slug))
    if area is None:
        raise ValueError(
            f"plots.yaml {pname}: area: {slug!r} is no cdn slug of station_areas.yaml"
        )
    found = []
    for table, key in (("map_categories", area["category"]), ("map_areas", slug)):
        entry = (cfg.get(table) or {}).get(key) or {}
        bad = set(entry) - set(MAP_AREA_KEYS)
        if bad:
            raise ValueError(
                f"plots.yaml {table}.{key}: unknown key(s) {sorted(bad)} "
                f"(keys: {', '.join(MAP_AREA_KEYS)})"
            )
        found.append(entry)
    return f"locations/{area['root']}/{slug}/maps", found


def map_jobs(
    pname: str,
    spec: Mapping[str, Any],
    defaults: Mapping[str, Any],
    areas: Mapping[str, Mapping[str, Any]] | None = None,
    cfg: Mapping[str, Any] | None = None,
    events: Sequence[Mapping[str, Any]] = (),
) -> list[Job]:
    """The jobs of a map product (``tool: map``): one ``gpsplot map KIND``
    per run, each with its own ``name:`` (the file's basename) and
    ``args:`` after the product's. ``stations:`` is a list (or none: every
    station in the map's --region). With ``area:`` the area's defaults come
    between plots.yaml ``defaults:`` and the product's
    (:func:`map_area_defaults`), and ``dir:`` defaults to the area's maps.

    Raises:
        ValueError: An unknown key or map kind, a run without a name, a
            name used twice, or a runner-owned option in ``args:``.
    """
    unknown = set(spec) - set(MAP_KEYS)
    if unknown:
        raise ValueError(f"plots.yaml {pname}: unknown map key(s) {sorted(unknown)}")
    kind = spec.get("map")
    if kind not in MAP_KINDS:
        raise ValueError(f"plots.yaml {pname}: map: one of {MAP_KINDS}, got {kind!r}")
    area_dir, area_defaults = map_area_defaults(pname, spec, areas or {}, cfg or {})
    out_dir = str(spec.get("dir") or area_dir or "")
    if not out_dir:
        raise ValueError(f"plots.yaml {pname}: a map product needs dir: or area:")
    spec = {**spec, "dir": out_dir}
    area = (areas or {}).get(str(spec.get("area") or ""), {})
    label = str(spec.get("label") or area.get("name") or pname)
    stations = tuple(str(s).upper() for s in _as_list(spec.get("stations") or []))
    checks = [str(s).upper() for s in _as_list(spec.get("ts_stations") or [])]
    names: set[str] = set()
    jobs = []
    for run in spec.get("runs") or []:
        bad = set(run) - set(MAP_RUN_KEYS)
        if bad:
            raise ValueError(f"plots.yaml {pname} run: unknown key(s) {sorted(bad)}")
        name = str(run.get("name") or "")
        if not name or name in names:
            raise ValueError(f"plots.yaml {pname}: each run needs its own name:")
        names.add(name)
        argv = ["--save", str(defaults.get("save", "png"))]
        args = [defaults.get("map_args"), *(e.get("args") for e in area_defaults)]
        for value in (*args, spec.get("args"), run.get("args")):
            argv += extra_args(value, MAP_RESERVED_OPTIONS)
        # defaults:, the area's category and area, the product's, the run's
        levels = [(defaults, "map_"), *((e, "") for e in area_defaults),
                  (spec, ""), (run, "")]  # fmt: skip
        for where, prefix in levels:
            argv += map_option_args(pname, where, prefix)
        argv += ["--name", name]
        caps = [(out_dir, name, str(run.get("caption") or
                                     map_caption(str(kind), argv, label, events)))]  # fmt: skip
        if checks and kind == "station":  # the track map draws them itself
            if "--track" not in argv:
                raise ValueError(f"plots.yaml {pname}: ts_stations: needs --track")
            argv += ["--track-series", ",".join(checks)]
            caps += [(f"{out_dir}/{CHECK_SUBDIR}", f"{name}-{sta}",
                      track_series_caption(sta, argv, events)) for sta in checks]  # fmt: skip
        jobs.append(Job(pname, None, out_dir, stations, tuple(argv),
                        bool(run.get("publish", False)), tool="map", group=name,
                        kind=str(kind), captions=tuple(caps)))  # fmt: skip
    if not jobs:
        raise ValueError(f"plots.yaml {pname}: a map product needs runs:")
    return jobs + map_ts_jobs(pname, spec, jobs, defaults)


#: where a map product's check series go, under its dir (aflogun shows
#: them in a tab of their own, BGÓ 2026-10-28)
CHECK_SUBDIR = "ts"

#: map options that set a velocity map's windows and reference: the ones the
#: check time series (``ts_stations:``) must repeat
_MAP_WINDOW_OPTIONS = ("--last", "--window", "--ref", "-e", "--end")

#: a velocity map's --ref as the ts options that give the same rates
_TS_REF: Mapping[str, tuple[str, ...]] = {
    "plate-periodic": ("--ref", "detrend", "--detrend-terms", "periodic"),
    "background": ("--ref", "detrend"),
    "detrend": ("--ref", "detrend"),
    "plate": ("--ref", "plate"),
}


def map_ts_jobs(
    pname: str,
    spec: Mapping[str, Any],
    maps: Sequence[Job],
    defaults: Mapping[str, Any],
    today: dt.date | None = None,
) -> list[Job]:
    """The check time series of a velocity map (``ts_stations:``; a
    ``station --track`` map draws its own, ``--track-series``): each
    listed station's series with the map's windows fitted on it (``gpsplot
    ts --rate-last``/``--rate-window``, in the map's reference, so the
    numbers are the arrows' numbers), to see whether a line fits.

    One set per distinct set of windows (a horizontal and a vertical run of
    the same windows share it), named ``<the runs' common name>-<STA>``
    (``svartsengi-6man-3man-larett``/``-lodrett`` → ``svartsengi-6man-3man-
    SENG``), in ``<dir>/ts/``. The plot starts twice the longest window
    before the end.

    Raises:
        ValueError: ``ts_stations:`` on a map other than velocity, or a run
            whose windows or --ref the series cannot repeat.
    """
    wanted = [str(s).upper() for s in _as_list(spec.get("ts_stations") or [])]
    if not wanted or spec.get("map") == "station":  # a track map: --track-series
        return []
    if spec.get("map") != "velocity":
        raise ValueError(
            f"plots.yaml {pname}: ts_stations: needs map: velocity or station --track"
        )
    sets: dict[tuple[str, ...], list[Job]] = {}
    for job in maps:
        sets.setdefault(_window_options(pname, job.argv), []).append(job)
    jobs = []
    for opts, runs in sets.items():
        names = [str(r.group) for r in runs]
        # the common whole words: -larett/-lodrett share "l", not a word
        words = os.path.commonprefix([n.split("-") for n in names])
        stem = "-".join(words) or names[0]
        argv, days = _ts_check_argv(pname, opts, today or dt.date.today())
        start = (today or dt.date.today()) - dt.timedelta(days=2 * days)
        for sta in wanted:
            jobs.append(Job(pname, None, f"{spec['dir']}/{CHECK_SUBDIR}", (sta,),
                            ("--save", str(defaults.get("save", "png")),
                             "-s", f"{start:%Y%m%d}", *argv,
                             "--name", f"{stem}-{sta}"),
                            any(r.publish for r in runs), group=f"{stem}-{sta}",
                            captions=((f"{spec['dir']}/{CHECK_SUBDIR}", f"{stem}-{sta}",
                                       check_caption(sta, opts)),)))  # fmt: skip
    return jobs


def _window_options(pname: str, argv: Sequence[str]) -> tuple[str, ...]:
    """A map run's window and reference options, in order (``--opt value``)."""
    out: list[str] = []
    it = iter(argv)
    for tok in it:
        opt, eq, value = tok.partition("=")
        if opt in _MAP_WINDOW_OPTIONS:
            out += [opt, value if eq else next(it, "")]
    if "--last" not in out and "--window" not in out:
        raise ValueError(
            f"plots.yaml {pname}: ts_stations: the runs need --last/--window"
        )
    return tuple(out)


def _ts_check_argv(
    pname: str, opts: Sequence[str], today: dt.date
) -> tuple[list[str], int]:
    """The ts options of a map's windows, and the longest window [days]."""
    argv: list[str] = []
    ref = "plate-periodic"
    end = today
    days = 0
    pairs = list(zip(opts[::2], opts[1::2], strict=True))
    for opt, value in pairs:
        if opt in ("-e", "--end"):
            end = dt.datetime.strptime(value, "%Y%m%d").date()
            argv += ["-e", value]
        elif opt == "--ref":
            ref = value
    for opt, value in pairs:
        if opt == "--last":
            argv += ["--rate-last", value]
            days = max(days, int(value.partition(":")[0]))
        elif opt == "--window":
            a, _, b = value.partition(":")
            b = b or f"{end:%Y%m%d}"
            argv += ["--rate-window", f"{a}:{b}"]
            first = dt.datetime.strptime(a, "%Y%m%d").date()
            days = max(days, (end - first).days + 1)
    if ref not in _TS_REF:
        raise ValueError(
            f"plots.yaml {pname}: ts_stations: --ref {ref} has no time series "
            f"(one of {sorted(_TS_REF)})"
        )
    return [*_TS_REF[ref], *argv], days


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
    if kind == "area":
        targets: list[tuple[str | None, tuple[str, ...]]] = [
            (None, tuple(entry["stations"]))
        ]
    elif kind == "area_baselines":
        pairs = entry.get("baselines") or []
        targets = [(None, tuple(pairs))] if pairs else []
    else:  # area_stacks: one figure per group
        stacks: Mapping[str, Sequence[str]] = entry.get("stacks") or {}
        targets = [(g, tuple(p)) for g, p in stacks.items()]
    tool = "stack" if kind == "area_stacks" else "ts"
    if spec.get("tool") == "polar":
        tool = "polar"
    overrides = spec.get("overrides") or {}
    ev = (
        event_specs(events, [slug, entry["key"]], colours)
        if spec.get("events") == "catalog"  # ts and stack both draw them
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
            if run.get("areas") is not None and slug not in _as_list(run["areas"]):
                continue
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
                if ev and tool != "polar":
                    argv = argv + ["--events", *ev]
                if group and tool == "polar":  # a group: its pairs in one figure
                    argv = argv + ["--together"]
                elif group:
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
        whole = job.tool in ("stack", "map")  # one figure: kept whole
        out.append(job if whole else replace(job, stations=tuple(keep)))
    return out, [s for s in want if s not in used]


def _as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list | tuple) else [value]


def _ymd(value: Any) -> str:
    day = value if isinstance(value, dt.date) else dt.date.fromisoformat(str(value))
    return f"{day:%Y%m%d}"


_BASELINE_FILE = re.compile(r"([A-Z0-9]{4})_([A-Z0-9]{4})-baseline-.+\.(png|pdf)")
_STACK_FILE = re.compile(r"stack_.+\.(png|pdf)")


def stale_baseline_plots(
    jobs: Sequence[Job], configured: Sequence[Job], root: Path
) -> list[Path]:
    """Baseline / stack plots in the jobs' directories the config no longer has.

    ``configured`` is every job of the whole config (unfiltered), so a run of
    one area or product never takes another one's plots for stale. Only files
    named like gpsplot's own baseline (``A_B-baseline-*``) and stack
    (``stack_<group>-*``) plots are considered: a pair no baseline job of
    that directory plots, or a stack of a group no stack job there makes.
    Other files, and other variants (periods, frames) of a kept pair or
    group, are left alone.
    """
    pairs: dict[str, set[str]] = {}
    groups: dict[str, set[str]] = {}
    for job in configured:
        if job.tool == "stack" and job.group:
            groups.setdefault(job.out_dir, set()).add(job.group)
        elif job.stations and all(_PAIR.match(s) for s in job.stations):
            pairs.setdefault(job.out_dir, set()).update(job.stations)
    dirs = {
        j.out_dir
        for j in jobs
        if j.tool == "stack" or (j.stations and all(_PAIR.match(s) for s in j.stations))
    }
    stale: list[Path] = []
    for out_dir in sorted(dirs):
        folder = root / out_dir
        if not folder.is_dir():
            continue
        kept_pairs = pairs.get(out_dir, set())
        kept_groups = groups.get(out_dir, set())
        for f in sorted(folder.iterdir()):
            if m := _BASELINE_FILE.fullmatch(f.name):
                if f"{m.group(1)}-{m.group(2)}" not in kept_pairs:
                    stale.append(f)
            elif _STACK_FILE.fullmatch(f.name):
                if not any(f.name.startswith(f"stack_{g}-") for g in kept_groups):
                    stale.append(f)
    return stale


def stale_map_plots(
    jobs: Sequence[Job], configured: Sequence[Job], root: Path
) -> list[Path]:
    """Plots in a map product's directory that no configured job writes.

    A map product's files have fixed names (a run's ``name:``, a check
    series' ``<name>-<STA>``), so a renamed or dropped run, or a station
    taken out of ``ts_stations:``, leaves its old file behind. Only the
    directories of the selected map jobs are looked at, against the names
    of every configured job writing there (``configured`` is unfiltered).
    A directory other (time-series) products also write to is left alone.
    """
    names: dict[str, set[str]] = {}
    shared: set[str] = set()  # a dir other products write to: left alone
    for job in configured:
        if job.group and (job.tool == "map" or "--name" in job.argv):
            names.setdefault(job.out_dir, set()).add(job.group)
            if "--track-series" in job.argv:  # the track map's own ts/<name>-<STA>
                listed = job.argv[job.argv.index("--track-series") + 1]
                names.setdefault(f"{job.out_dir}/{CHECK_SUBDIR}", set()).update(
                    f"{job.group}-{sta}" for sta in listed.split(",")
                )
        else:
            shared.add(job.out_dir)
    stale: list[Path] = []
    dirs = {j.out_dir for j in jobs if j.tool == "map"}
    dirs |= {f"{d}/{CHECK_SUBDIR}" for d in dirs}
    for out_dir in sorted(dirs):
        folder = root / out_dir
        if out_dir in shared or not folder.is_dir():
            continue
        keep = names.get(out_dir, set())
        stale += [
            f
            for f in sorted(folder.iterdir())
            if f.is_file() and f.suffix in (".png", ".pdf") and f.stem not in keep
        ]
    return stale


def _ddmm(day: dt.date) -> str:
    return f"{day:%d.%m.}"


def _windows_text(argv: Sequence[str], today: dt.date | None = None) -> str:
    """The windows of map options (``--last``, ``--window``, ``-e``) in
    words: ``22.05.–28.10.2026 og 30.07.–28.10.2026``."""
    end = today or dt.date.today()
    tokens = list(argv)
    for i, tok in enumerate(tokens[:-1]):
        if tok in ("-e", "--end"):
            end = dt.datetime.strptime(tokens[i + 1], "%Y%m%d").date()
    spans = []
    for i, tok in enumerate(tokens[:-1]):
        value = tokens[i + 1]
        if tok == "--last":
            first, _, to = value.partition(":")
            a = end - dt.timedelta(days=int(first) - 1)
            spans.append((a, end - dt.timedelta(days=int(to or 0))))
        elif tok == "--window":
            a_s, _, b_s = value.partition(":")
            a = dt.datetime.strptime(a_s, "%Y%m%d").date()
            b = dt.datetime.strptime(b_s, "%Y%m%d").date() if b_s else end
            spans.append((a, b))
    words = [f"{_ddmm(a)}–{_ddmm(b)}{b:%Y}" for a, b in spans]
    return " og ".join(words)


def _option(argv: Sequence[str], name: str) -> str | None:
    tokens = list(argv)
    for i, tok in enumerate(tokens[:-1]):
        if tok == name:
            return tokens[i + 1]
    return None


def onset_text(spec: str | None, events: Sequence[Mapping[str, Any]] = ()) -> str:
    """``--track``'s onset as a date (``23.09.2026``): a YYYYMMDD, or the
    onset of the events.yaml event of that id (or the one id containing it);
    the spec itself when neither."""
    if not spec:
        return ""
    try:
        return f"{dt.datetime.strptime(spec, '%Y%m%d'):%d.%m.%Y}"
    except ValueError:
        pass
    hits = [e for e in events if str(e.get("id")) == spec] or [
        e for e in events if spec in str(e.get("id"))
    ]
    if len(hits) == 1 and hits[0].get("onset") is not None:
        onset = hits[0]["onset"]
        day = onset if isinstance(onset, dt.date) else dt.date.fromisoformat(str(onset))
        return f"{day:%d.%m.%Y}"
    return spec


def map_caption(
    kind: str,
    argv: Sequence[str],
    label: str,
    events: Sequence[Mapping[str, Any]] = (),
) -> str:
    """A map's caption in Icelandic, from its options (first draft, BGÓ
    2026-10-28; a run's ``caption:`` replaces it)."""
    if kind == "velocity":
        way = "lóðréttur" if "--vertical" in argv else "láréttur"
        windows = _windows_text(argv)
        return f"{label}: {way} hraði (mm/ár)" + (f", {windows}" if windows else "")
    if kind == "station" and _option(argv, "--track"):
        recent = _option(argv, "--track-recent") or "14"
        onset = onset_text(_option(argv, "--track"), events)
        kept = ("; leitni fyrir atburðinn haldið (hraði fyrir hann grár)"
                if _option(argv, "--track-trend") == "keep" else "")  # fmt: skip
        up = ("" if "--no-track-up" in argv
              else f"; lóðrétt færsla frá {onset} (ör upp/niður á stöð)")  # fmt: skip
        return (f"{label}: daglegar stöður frá {onset}{_ref_text(argv)}, litaðar "
                f"eftir dögum; hraði frá {onset} (blátt) og síðustu {recent} daga "
                f"(svart){up}{kept}")  # fmt: skip
    if kind == "deformation":
        return f"{label}: færsla"
    return f"{label}: stöðvakort"


def check_caption(sta: str, opts: Sequence[str]) -> str:
    """A velocity map's check series: the station and the map's windows."""
    windows = _windows_text(opts)
    return f"{sta}: tímaröð með hraða kortsins" + (f", {windows}" if windows else "")


def _ref_text(argv: Sequence[str]) -> str:
    """`` miðað við SKFC`` / `` miðað við meðaltal SKFC, SLEC og FAGC`` of a
    track map's ``--track-ref``; empty without one."""
    refs = [r.strip().upper() for r in (_option(argv, "--track-ref") or "").split(",")]
    refs = [r for r in refs if r]
    if not refs:
        return ""
    if len(refs) == 1:
        return f" miðað við {refs[0]}"
    return f" miðað við meðaltal {', '.join(refs[:-1])} og {refs[-1]}"


def track_series_caption(
    sta: str, argv: Sequence[str], events: Sequence[Mapping[str, Any]] = ()
) -> str:
    """A track map's series of one station."""
    recent = _option(argv, "--track-recent") or "14"
    onset = onset_text(_option(argv, "--track"), events)
    trend = ("árstíðasveifla fyrir {0} fjarlægð, leitni haldið"
             if _option(argv, "--track-trend") == "keep"
             else "leitni fyrir {0} fjarlægð").format(onset)  # fmt: skip
    comps = "norður og austur" if "--no-track-up" in argv else "norður, austur og upp"
    return (f"{sta}: {comps}{_ref_text(argv)}, {trend}; hraði frá "
            f"{onset} (blátt) og síðustu {recent} daga (svart)")  # fmt: skip


def map_captions(
    jobs: Sequence[Job], configured: Sequence[Job]
) -> dict[str, dict[str, str]]:
    """{dir: {file stem: caption}} of the dirs the ``jobs`` write to, from
    every ``configured`` job writing there (so a run of one product keeps
    the others' captions)."""
    dirs = {d for job in jobs for d, _stem, _text in job.captions}
    out: dict[str, dict[str, str]] = {d: {} for d in dirs}
    for job in configured:
        for d, stem, text in job.captions:
            if d in out:
                out[d][stem] = text
    return out
