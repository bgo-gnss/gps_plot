"""The RTK warning of an area: one status file per area, and the bench.

``PLAN-rtk-trigger.md`` T1, the tracer. An area's ``monitoring: rtk:``
block in station_areas.yaml (the same block aflogun's RTK light reads)::

    rtk:
      figure: graphs/rtk/SVIN_SVIE-rtk-12h.png   # aflogun: the light's age
      baselines: [SVIN-SVIE, SVIN-SKFC]          # default: the area's RTK pairs
      window_minutes: 60                         # the newest window
      reference_minutes: 360                     # the level before it
      gap_minutes: 0                             # left out between them
      horizontal_mm: [10, 20, 40]                # yellow / orange / red edges
      up_mm: [20, 40, 80]

For each baseline: fixed RTK solutions only (Q 1; float and single out),
one-minute medians, and :func:`gps_analysis.rtk_alarm.window_offset` of the
newest window against the reference window, in mm. Horizontal is the length
of the N/E offset, Up the U offset; the baseline's level is the higher of
:func:`gps_analysis.rtk_alarm.alarm_level` of the two against the edges, the
area's level the highest over its baselines. No ``horizontal_mm`` = no
limits: no status file, and aflogun keeps "Mörk óskilgreind".

``gpsplot rtk-run`` writes ``<area rtk dir>/rtk.status.json`` (schema 1,
``provisional``: the edges are hand-set until the bench calibrates them,
T4) after its plots; ``gpsplot rtk-bench`` replays the same level function
on a 5-minute grid around an event and prints when each level was reached.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = 1
STATUS_NAME = "rtk.status.json"
LEVEL_NAMES = ("green", "yellow", "orange", "red")
#: a baseline whose newest epoch is older than this is left out [min]
STALE_MINUTES = 15.0
DEFAULTS = {"window_minutes": 60.0, "reference_minutes": 360.0, "gap_minutes": 0.0}
#: fixed solutions only: RTKLIB Q 2 (float) and 5 (single) out
FIXED_ONLY = (2, 5)


def rtk_spec(entry: Mapping[str, Any]) -> dict[str, Any] | None:
    """An area's ``monitoring: rtk:`` with defaults; None without limits.

    Raises:
        ValueError: malformed edges or window lengths.
    """
    mon = entry.get("monitoring") or {}
    raw = mon.get("rtk") if isinstance(mon, Mapping) else None
    if not isinstance(raw, Mapping) or raw.get("horizontal_mm") is None:
        return None
    spec: dict[str, Any] = {k: float(raw.get(k, v)) for k, v in DEFAULTS.items()}
    for k in ("horizontal_mm", "up_mm"):
        edges = [float(v) for v in raw.get(k) or []]
        if edges and not all(a < b for a, b in zip(edges, edges[1:], strict=False)):
            raise ValueError(f"rtk {k} must increase, got {edges}")
        spec[k] = edges
    if spec["window_minutes"] <= 0 or spec["reference_minutes"] <= 0:
        raise ValueError("rtk window_minutes and reference_minutes must be > 0")
    spec["baselines"] = [str(b) for b in raw.get("baselines") or []]
    return spec


def minutes(t: Any) -> np.ndarray:
    """UTC ``datetime64`` epochs as float minutes since 1970."""
    ns = np.asarray(t, dtype="datetime64[ns]").astype(np.int64)
    return ns.astype(np.float64) / 60e9


def baseline_level(
    t_min: np.ndarray, y_mm: np.ndarray, end_min: float, spec: Mapping[str, Any]
) -> dict[str, Any] | None:
    """One baseline's window offset and level at ``end_min``; None when a
    window has too few epochs (no verdict)."""
    from gps_analysis.rtk_alarm import alarm_level, window_offset

    w = window_offset(
        t_min,
        y_mm,
        end_min,
        window=spec["window_minutes"],
        reference=spec["reference_minutes"],
        gap=spec["gap_minutes"],
    )
    if w is None:
        return None
    horizontal = float(np.hypot(w.offset[0], w.offset[1]))
    up = float(w.offset[2])
    lh = alarm_level(horizontal, spec["horizontal_mm"])
    lu = alarm_level(up, spec["up_mm"]) if spec["up_mm"] else 0
    return {
        "level": max(lh, lu),
        "component": "up" if lu > lh else "horizontal",
        "horizontal_mm": round(horizontal, 1),
        "up_mm": round(up, 1),
        "sigma_mm": [round(float(s), 1) for s in w.sigma],
        "n_window": w.n_after,
        "n_reference": w.n_before,
    }


Reader = Callable[[str, dt.datetime, dt.datetime], tuple[np.ndarray, np.ndarray]]


def rtk_reader(rtk_dirs: Sequence[str], root: str | None = None) -> Reader:
    """``(pair, start, end) -> (t, y_mm)``: fixed one-minute medians of an
    RTK pair as written (a reversed directory negated), mm."""

    def read(pair: str, start: dt.datetime, end: dt.datetime) -> Any:
        from geo_dataread.rtk_read import read_rtk_baseline

        from gps_plot.sources.rtk import rtk_direction

        directory, flipped = rtk_direction(pair, rtk_dirs)
        r = read_rtk_baseline(
            directory, start, end, root=root, interval="60s", drop_q=FIXED_ONLY
        )
        return r.time, r.neu * (-1000.0 if flipped else 1000.0)

    return read


def area_status(
    slug: str,
    spec: Mapping[str, Any],
    pairs: Sequence[str],
    read: Reader,
    now: dt.datetime,
) -> dict[str, Any]:
    """The area's status record at ``now`` (naive UTC): per baseline, then
    the highest level over the baselines with fresh data."""
    span = spec["window_minutes"] + spec["gap_minutes"] + spec["reference_minutes"]
    start = now - dt.timedelta(minutes=span + 10)
    end_min = float(minutes(np.datetime64(now, "ns")))
    rows = []
    for pair in spec["baselines"] or pairs:
        row: dict[str, Any] = {"pair": pair}
        try:
            t, y = read(pair, start, now)
        except (OSError, ValueError) as exc:
            row["error"] = str(exc)
            rows.append(row)
            continue
        if t.size:
            row["newest"] = str(np.datetime_as_string(t[-1], unit="s")) + "Z"
        tm = minutes(t)
        if not t.size or end_min - tm[-1] > STALE_MINUTES:
            row["stale"] = True
            rows.append(row)
            continue
        got = baseline_level(tm, y, end_min, spec)
        if got is None:
            row["stale"] = True
        else:
            row.update(got)
        rows.append(row)
    live = [r for r in rows if "level" in r]
    record: dict[str, Any] = {
        "schema": SCHEMA,
        "kind": "rtk",
        "provisional": True,
        "area": slug,
        "made": now.strftime("%Y-%m-%dT%H:%MZ"),
        "window_minutes": spec["window_minutes"],
        "reference_minutes": spec["reference_minutes"],
        "gap_minutes": spec["gap_minutes"],
        "horizontal_mm": spec["horizontal_mm"],
        "up_mm": spec["up_mm"],
        "baselines": rows,
    }
    if not live:
        record["level"] = None
        record["level_name"] = "grey"
        return record
    top = max(live, key=lambda r: (r["level"], r[f"{r['component']}_mm"] ** 2))
    record["level"] = top["level"]
    record["level_name"] = LEVEL_NAMES[top["level"]]
    record["set_by"] = {
        "pair": top["pair"],
        "component": top["component"],
        "value_mm": top[f"{top['component']}_mm"],
    }
    return record


def write_status(path: Path, record: Mapping[str, Any]) -> None:
    """Write ``record`` as JSON: a temp file renamed over, never half."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(record, ensure_ascii=False, indent=1) + "\n")
    os.replace(tmp, path)


# --- the bench ----------------------------------------------------------


def onset_of(events: Sequence[Mapping[str, Any]], eid: str) -> dt.datetime:
    """An events.yaml event's onset (``onset`` + ``onset_time``, UTC).

    Raises:
        ValueError: no such event, or it has no ``onset_time``.
    """
    for e in events:
        if str(e.get("id")) == eid:
            hhmm = e.get("onset_time")
            if not hhmm:
                raise ValueError(f"event {eid} has no onset_time")
            day = dt.date.fromisoformat(str(e["onset"]))
            h, m = (int(x) for x in str(hhmm).split(":")[:2])
            return dt.datetime.combine(day, dt.time(h, m))
    raise ValueError(f"no event {eid}")


def replay(
    t_min: np.ndarray,
    y_mm: np.ndarray,
    grid: Sequence[dt.datetime],
    spec: Mapping[str, Any],
) -> list[tuple[dt.datetime, dict[str, Any] | None]]:
    """The level function at each grid time, on one series read once."""
    out = []
    for g in grid:
        end_min = float(minutes(np.datetime64(g, "ns")))
        out.append((g, baseline_level(t_min, y_mm, end_min, spec)))
    return out


def first_times(
    steps: Sequence[tuple[dt.datetime, dict[str, Any] | None]],
) -> dict[int, dt.datetime]:
    """The first grid time each level (1 yellow … 3 red) was reached."""
    first: dict[int, dt.datetime] = {}
    for g, got in steps:
        if got is None:
            continue
        for lvl in range(1, got["level"] + 1):
            first.setdefault(lvl, g)
    return first


def bench_parser(prog: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=prog,
        description="Replay the RTK warning on a 5-minute grid around an event "
        "and print when each level was reached (PLAN-rtk-trigger T1).",
    )
    p.add_argument("pair", help="RTK baseline A-B, e.g. SENG-ELDC")
    p.add_argument("--event", help="events.yaml id (its onset + onset_time)")
    p.add_argument("--onset", help="onset instead, UTC 'YYYY-MM-DD HH:MM'")
    p.add_argument("--area", help="the area whose monitoring: rtk: gives the edges")
    p.add_argument("--horizontal-mm", default=None, help="edges, e.g. 10,20,40")
    p.add_argument("--up-mm", default=None, help="edges, e.g. 20,40,80")
    p.add_argument("--window-minutes", type=float, default=None)
    p.add_argument("--reference-minutes", type=float, default=None)
    p.add_argument("--before-hours", type=float, default=6.0)
    p.add_argument("--after-hours", type=float, default=3.0)
    p.add_argument("--step-minutes", type=float, default=5.0)
    p.add_argument("--every", action="store_true", help="print every grid step")
    return p


def _edges(text: str | None) -> list[float] | None:
    return [float(v) for v in text.split(",")] if text else None


def bench(prog: str, argv: list[str]) -> int:
    from geo_dataread.rtk_read import list_baselines, rtk_dir

    from gps_plot import products
    from gps_plot.cli import common
    from gps_plot.cli.run import _cfg_file
    from gps_plot.plot_gps_timeseries import _load_config

    args = bench_parser(prog).parse_args(argv)
    config = _load_config()
    plots_path = Path(config.config_path).expanduser() / "plots.yaml"
    cfg = products.load_yaml(plots_path)
    if args.onset:
        onset = dt.datetime.strptime(args.onset, "%Y-%m-%d %H:%M")
    elif args.event:
        events = products.load_yaml(_cfg_file(config, cfg, "events")).get("events")
        onset = onset_of(events or [], args.event)
    else:
        print(f"{prog}: --event or --onset", file=sys.stderr)
        return 2
    spec: dict[str, Any] = {**DEFAULTS, "horizontal_mm": [], "up_mm": []}
    if args.area:
        areas = products.area_table(products.load_yaml(_cfg_file(config, cfg, "areas")))
        got = rtk_spec(areas.get(args.area) or {})
        if got:
            spec.update(got)
    for key, val in (
        ("horizontal_mm", _edges(args.horizontal_mm)),
        ("up_mm", _edges(args.up_mm)),
        ("window_minutes", args.window_minutes),
        ("reference_minutes", args.reference_minutes),
    ):
        if val is not None:
            spec[key] = val
    if not spec["horizontal_mm"]:
        print(
            f"{prog}: no edges (--area with limits, or --horizontal-mm)",
            file=sys.stderr,
        )
        return 2
    dirs = list_baselines(rtk_dir(common.config_dir(config, "rtk_dir")))
    read = rtk_reader(dirs)
    first_grid = onset - dt.timedelta(hours=args.before_hours)
    last_grid = onset + dt.timedelta(hours=args.after_hours)
    lead = spec["window_minutes"] + spec["gap_minutes"] + spec["reference_minutes"]
    t, y = read(args.pair, first_grid - dt.timedelta(minutes=lead + 5), last_grid)
    step = dt.timedelta(minutes=args.step_minutes)
    grid = []
    g = first_grid
    while g <= last_grid:
        grid.append(g)
        g += step
    steps = replay(minutes(t), y, grid, spec)
    print(
        f"{args.pair}: onset {onset:%Y-%m-%d %H:%M}Z, window "
        f"{spec['window_minutes']:g} min vs {spec['reference_minutes']:g} min, "
        f"edges H {spec['horizontal_mm']} U {spec['up_mm']} mm, "
        f"{t.size} fixed minutes"
    )
    if args.every:
        for g, got in steps:
            if got is None:
                print(f"  {g:%m-%d %H:%M}  no verdict")
            else:
                print(
                    f"  {g:%m-%d %H:%M}  {LEVEL_NAMES[got['level']]:6s} "
                    f"H {got['horizontal_mm']:7.1f}  U {got['up_mm']:7.1f} mm"
                )
    first = first_times(steps)
    before = [g for lvl, g in first.items() if g < onset]
    for lvl in (1, 2, 3):
        when = first.get(lvl)
        if when is None:
            print(f"  {LEVEL_NAMES[lvl]:6s} not reached")
        else:
            delay = (when - onset).total_seconds() / 60
            print(f"  {LEVEL_NAMES[lvl]:6s} {when:%H:%M}Z  {delay:+.0f} min")
    if before:
        print(f"  note: a level was reached before the onset ({min(before):%H:%M}Z)")
    return 0
