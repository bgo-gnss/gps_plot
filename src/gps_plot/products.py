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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
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
)

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


def run_argv(run: Mapping[str, Any], defaults: Mapping[str, Any]) -> list[list[str]]:
    """The ``gpsplot ts`` options of one run; a ``fixedstart`` list adds one each."""
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
                    merged = {**run, **{k: v for k, v in over.items() if k in RUN_KEYS}}
                    if run.get("fixedstart") is False:  # the run opts out
                        merged["fixedstart"] = []
                    for argv in run_argv(merged, defaults):
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
                for argv in run_argv(run, defaults):
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


def _as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list | tuple) else [value]


def _ymd(value: Any) -> str:
    day = value if isinstance(value, dt.date) else dt.date.fromisoformat(str(value))
    return f"{day:%Y%m%d}"
