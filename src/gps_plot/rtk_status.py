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
#: T7: a level counts only when it held one grid step (5 min) before too
PERSIST_STEPS = 1
PERSIST_MINUTES = 5.0
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
    # T3: several newest windows, each against the same reference; the level
    # is the highest over them (default: window_minutes alone)
    windows = [float(w) for w in raw.get("windows_minutes") or [spec["window_minutes"]]]
    if any(w <= 0 for w in windows):
        raise ValueError(f"rtk windows_minutes must be > 0, got {windows}")
    spec["windows_minutes"] = sorted(set(windows))
    # T7b: independent pairs an area level needs (1, or 2 for every level)
    spec["agree"] = int(raw.get("agree", 1))
    if spec["agree"] not in (1, 2):
        raise ValueError(f"rtk agree must be 1 or 2, got {spec['agree']}")
    # T5: the common-mode mode; ``siblings`` = the sibling veto
    spec["common_mode"] = raw.get("common_mode")
    common_mode(spec)
    # T9: the area's level from the pairs (agree) or the coherent statistic
    spec["statistic"] = str(raw.get("statistic", "pairs"))
    if spec["statistic"] not in ("pairs", "coherent", "accel"):
        raise ValueError(
            "rtk statistic must be 'pairs', 'coherent' or 'accel', got "
            f"{spec['statistic']!r}"
        )
    return spec


def independent(a: str, b: str) -> bool:
    """Two ``ROVR-BASE`` pairs are independent witnesses unless the only
    station they share is the base of both: one base glitch then moves
    both (T7, BGÓ 2026-11-11). Sharing a rover, or nothing, is independent."""
    ra, ba = a[:4], a[5:]
    rb, bb = b[:4], b[5:]
    common = {ra, ba} & {rb, bb}
    return not (common and ba == bb and common == {ba})


def combine(levels: Mapping[str, int | None], agree: int = 1) -> int | None:
    """The area's level from its pairs' (T7): the highest, but red only when
    two red pairs are independent (:func:`independent`); a lone red pair,
    or reds that share only their base, give orange. None: no verdict.

    ``agree=2`` (BGÓ 2026-11-11, the area's ``agree:``): every level needs
    two independent pairs at it or above, a lone pair's alarm stays at the
    pair (an intrusion moves several stations, one pair's noise does not)."""
    from itertools import combinations

    got = {p: v for p, v in levels.items() if v is not None}
    if not got:
        return None
    if agree >= 2:
        for lvl in (3, 2, 1):
            at = [p for p, v in got.items() if v >= lvl]
            if any(independent(a, b) for a, b in combinations(at, 2)):
                return lvl
        return 0
    top = max(got.values())
    if top < 3:
        return top
    reds = [p for p, v in got.items() if v >= 3]
    if any(independent(a, b) for a, b in combinations(reds, 2)):
        return 3
    return 2


def pair_spec(spec: Mapping[str, Any], pair: str) -> Mapping[str, Any]:
    """``spec`` with the pair's own calibrated edges (``pair_edges``, from
    rtk_thresholds.yaml) in place of the area's, when it has them."""
    own = (spec.get("pair_edges") or {}).get(pair)
    if not own:
        return spec
    out = {**spec, **{k: own[k] for k in ("horizontal_mm", "up_mm") if own.get(k)}}
    if own.get("windows"):
        out["window_edges"] = own["windows"]
    return out


def load_pair_edges(
    doc: Mapping[str, Any], area: str
) -> dict[str, dict[str, list[float]]]:
    """An area's calibrated edges per pair from a loaded rtk_thresholds.yaml
    (T4); {} without one.

    Raises:
        ValueError: edges that do not increase.
    """

    def edges_of(e: Mapping[str, Any], where: str) -> dict[str, list[float]]:
        got = {}
        for k in ("horizontal_mm", "up_mm"):
            v = [float(x) for x in (e or {}).get(k) or []]
            if v and not all(a < b for a, b in zip(v, v[1:], strict=False)):
                raise ValueError(f"{area} {where} {k} must increase, got {v}")
            if v:
                got[k] = v
        return got

    out: dict[str, Any] = {}
    pairs = ((doc.get(area) or {}).get("pairs")) or {}
    for pair, e in pairs.items():
        edges: dict[str, Any] = edges_of(e or {}, str(pair))
        windows = (e or {}).get("windows") or {}
        if windows:  # T3: edges per window length [min]
            edges["windows"] = {
                float(w): edges_of(we, f"{pair} {w} min") for w, we in windows.items()
            }
        out[str(pair)] = edges
    return out


def minutes(t: Any) -> np.ndarray:
    """UTC ``datetime64`` epochs as float minutes since 1970."""
    ns = np.asarray(t, dtype="datetime64[ns]").astype(np.int64)
    return ns.astype(np.float64) / 60e9


def baseline_level(
    t_min: np.ndarray, y_mm: np.ndarray, end_min: float, spec: Mapping[str, Any]
) -> dict[str, Any] | None:
    """One baseline's level at ``end_min``: for each newest window (T3,
    ``windows_minutes``) its offset against the reference window, judged by
    that window's edges (``window_edges``, else the pair's or area's); the
    highest level wins (ties: the larger offset over its first edge). None
    when no window has enough epochs (no verdict)."""
    from gps_analysis.rtk_alarm import alarm_level, window_offset

    best: dict[str, Any] | None = None
    # T5: each window's offset vector, level and yellow edges, for the
    # sibling veto (:func:`sibling_veto`); private, never written
    per_window: dict[float, dict[str, Any]] = {}
    windows = spec.get("windows_minutes") or [spec["window_minutes"]]
    for minutes_w in windows:
        w = window_offset(
            t_min,
            y_mm,
            end_min,
            window=minutes_w,
            reference=spec["reference_minutes"],
            gap=spec["gap_minutes"],
        )
        if w is None:
            continue
        own = (spec.get("window_edges") or {}).get(float(minutes_w)) or {}
        h_edges = own.get("horizontal_mm") or spec["horizontal_mm"]
        u_edges = own.get("up_mm") or spec["up_mm"]
        horizontal = float(np.hypot(w.offset[0], w.offset[1]))
        up = float(w.offset[2])
        lh = alarm_level(horizontal, h_edges)
        lu = alarm_level(up, u_edges) if u_edges else 0
        ratio = max(horizontal / h_edges[0], abs(up) / u_edges[0] if u_edges else 0.0)
        per_window[float(minutes_w)] = {
            "offset": [float(v) for v in w.offset],
            "level": max(lh, lu),
            "yellow": (h_edges[0], u_edges[0] if u_edges else None),
        }
        got = {
            "level": max(lh, lu),
            "component": "up" if lu > lh else "horizontal",
            "window_minutes": minutes_w,
            "horizontal_mm": round(horizontal, 1),
            "up_mm": round(up, 1),
            "sigma_mm": [round(float(s), 1) for s in w.sigma],
            "n_window": w.n_after,
            "n_reference": w.n_before,
            "_ratio": ratio,
        }
        if best is None or (got["level"], ratio) > (best["level"], best["_ratio"]):
            best = got
    if best is not None:
        best.pop("_ratio")
        best["_windows"] = per_window
    return best


def base_of(pair: str) -> str:
    """The base of a ``ROVR-BASE`` pair."""
    return pair[5:]


def sibling_veto(
    got: Mapping[str, dict[str, Any] | None],
) -> dict[str, dict[str, Any] | None]:
    """T5, common mode ``siblings`` (BGÓ 2026-11-10): a jump of the base
    moves every pair on it by the same vector, so a pair's alarm in a window
    is vetoed when every sibling (the other pairs on the same base with an
    offset in that window) moved by the same vector: the rover-to-rover
    difference of the offsets, P − S, below P's yellow edges, horizontal
    and Up apart. That difference is the offset of the rover-rover baseline,
    in which the base cancels. A pair without siblings with data keeps its
    level. The level is the highest over the windows not vetoed; a lowered
    record keeps the raw one as ``level_raw`` and ``vetoed`` = the base.

    ``got``: one step's :func:`baseline_level` result per pair (None: no
    verdict). Returns new records; the input is not changed.

    Method: (R1 − B) − (R2 − B) = R1 − R2, so a shared base's motion
    cancels in the difference of two sibling offsets; Ohta et al. 2012 §2
    for the window offsets themselves.
    """
    out: dict[str, dict[str, Any] | None] = {}
    for pair, rec in got.items():
        if rec is None or rec["level"] < 1:
            out[pair] = rec
            continue
        sibs = [
            s
            for s, r in got.items()
            if s != pair and r is not None and base_of(s) == base_of(pair)
        ]
        if not sibs:
            out[pair] = rec
            continue
        level = 0
        for w, mine in rec["_windows"].items():
            if mine["level"] < 1:
                continue
            others = [got[s]["_windows"].get(w) for s in sibs]  # type: ignore[index]
            others = [o for o in others if o is not None]
            h_edge, u_edge = mine["yellow"]
            same = bool(others) and all(
                np.hypot(mine["offset"][0] - o["offset"][0],
                         mine["offset"][1] - o["offset"][1]) < h_edge
                and (u_edge is None or abs(mine["offset"][2] - o["offset"][2]) < u_edge)
                for o in others
            )  # fmt: skip
            if not same:
                level = max(level, mine["level"])
        if level < rec["level"]:
            out[pair] = {
                **rec,
                "level": level,
                "level_raw": rec.get("level_raw", rec["level"]),
                "vetoed": base_of(pair),
            }
        else:
            out[pair] = rec
    return out


def common_mode(spec: Mapping[str, Any]) -> str | None:
    """The area's common-mode mode (T5): ``siblings`` or None."""
    mode = spec.get("common_mode")
    if mode not in (None, "siblings"):
        raise ValueError(f"rtk common_mode must be 'siblings', got {mode!r}")
    return mode


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
    at_now: dict[str, dict[str, Any] | None] = {}
    at_before: dict[str, dict[str, Any] | None] = {}
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
        own = pair_spec(spec, pair)
        at_now[pair] = baseline_level(tm, y, end_min, own)
        at_before[pair] = baseline_level(tm, y, end_min - PERSIST_MINUTES, own)
        rows.append(row)
    coherent = None
    fit = spec.get("coherent_fit") if spec.get("statistic") == "coherent" else None
    if fit is not None:  # T9, on the offsets before any veto
        from gps_plot import rtk_coherent

        coherent = rtk_coherent.live(
            {p: (at_before.get(p), at_now[p]) for p in at_now}, fit
        )
    if common_mode(spec) == "siblings":  # T5, at both times
        at_now = sibling_veto(at_now)
        at_before = sibling_veto(at_before)
    for row in rows:
        if row["pair"] not in at_now or "stale" in row:
            continue
        got = at_now[row["pair"]]
        if got is None:
            row["stale"] = True
            continue
        row.update({k: v for k, v in got.items() if not k.startswith("_")})
        # T7: the level must have held 5 min before too
        before = at_before[row["pair"]]
        if before is not None and before["level"] < row["level"]:
            row.setdefault("level_raw", row["level"])
            row["level"] = before["level"]
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
        "common_mode": spec.get("common_mode"),
        "horizontal_mm": spec["horizontal_mm"],
        "up_mm": spec["up_mm"],
        "baselines": rows,
    }
    if not live:
        record["level"] = None
        record["level_name"] = "grey"
        return record
    top = max(live, key=lambda r: (r["level"], r[f"{r['component']}_mm"] ** 2))
    level = combine({r["pair"]: r["level"] for r in live}, spec.get("agree", 1))
    assert level is not None
    # which rule set the level: the coherent statistic, or the pairs (also
    # when coherent has too few template pairs with a verdict)
    record["rule"] = f"pairs, agree {spec.get('agree', 1)}"
    if coherent is not None:
        record["coherent"] = coherent
        if coherent["level"] is not None:
            level = coherent["level"]
            record["rule"] = "coherent"
        else:
            record["rule"] += " (coherent: too few pairs)"
    elif spec.get("statistic") == "coherent":
        record["rule"] += " (coherent: not calibrated)"
    # T10: the acceleration warning; the area takes the higher of it and
    # the offset rule (BGÓ 2026-11-12: the offset rule still catches steps)
    accel = spec.get("accel_fit") if spec.get("statistic") == "accel" else None
    if accel is not None:
        from gps_plot import rtk_accel

        got = rtk_accel.live(read, list(spec["baselines"] or pairs), accel, now)
        record["accel"] = got
        if got["level"] is not None and got["level"] > level:
            level = got["level"]
            record["rule"] = "accel"
    elif spec.get("statistic") == "accel":
        record["rule"] += " (accel: not calibrated)"
    record["level"] = level
    record["level_name"] = LEVEL_NAMES[level]
    if record["rule"] == "accel":
        pass  # one pair suffices: acceleration has no base-jump twin
    elif record["rule"] == "coherent":
        if coherent and coherent.get("red_unconfirmed"):  # the guard held it
            record["red_unconfirmed"] = True
    elif level < top["level"]:  # T7: a red pair alone, or reds sharing a base
        record["red_unconfirmed"] = True
    record["set_by"] = {
        "pair": top["pair"],
        "component": top["component"],
        "value_mm": top[f"{top['component']}_mm"],
    }
    return record


THRESHOLDS_NAME = "rtk_thresholds.yaml"


def merge_area(
    old: Mapping[str, Any], area: str, got: Mapping[str, Any]
) -> dict[str, Any]:
    """rtk_thresholds.yaml with ``got``'s keys set in ``area``, the area's
    other keys kept (a rate calibration keeps ``coherent:`` and back)."""
    return {**old, area: {**(old.get(area) or {}), **got}}


def write_thresholds(path: Path, doc: Mapping[str, Any]) -> None:
    """rtk_thresholds.yaml (T4), a temp file renamed over."""
    import yaml

    head = (
        "# RTK warning edges per pair (PLAN-rtk-trigger T4), written by\n"
        "# gpsplot rtk-bench --calibrate; rtk-run and rtk-bench read them.\n"
    )
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(
        head + yaml.safe_dump(dict(doc), sort_keys=False, allow_unicode=True)
    )
    os.replace(tmp, path)


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
    p.add_argument(
        "pair", nargs="?", default=None,
        help="RTK baseline A-B, e.g. SENG-ELDC; without it: the config bench",
    )  # fmt: skip
    p.add_argument(
        "--config", default=None,
        help="the bench (T2): rtk_bench.yaml (default: config dir), with --area",
    )  # fmt: skip
    p.add_argument(
        "--set", action="append", default=None, choices=("tune", "test"),
        help="the events to replay (config bench; repeatable; default tune)",
    )  # fmt: skip
    p.add_argument(
        "--quiet", action="append", default=None,
        help="quiet periods for the false-alarm rate, e.g. calibration (repeatable)",
    )  # fmt: skip
    p.add_argument("--out", default=None, help="write the report (markdown) here too")
    p.add_argument(
        "--calibrate", action="store_true",
        help="T4: per-pair edges from the calibration quiet periods -> "
        "rtk_thresholds.yaml (config dir), then the report with them",
    )  # fmt: skip
    p.add_argument(
        "--calibrate-rate", action="store_true",
        help="T4d: edges from the area's false-alarm targets (yellow 1/month, "
        "orange 4/year, red 1/year) on the calibration months -> "
        "rtk_thresholds.yaml, then the report with them",
    )  # fmt: skip
    p.add_argument(
        "--calibrate-coherent", action="store_true",
        help="T9: the coherent statistic (template from the tune events, scales "
        "and multipliers from the calibration months) -> rtk_thresholds.yaml "
        "<area>: coherent:, then the report",
    )  # fmt: skip
    p.add_argument(
        "--calibrate-accel", action="append", default=None, metavar="START,END",
        help="T10: the acceleration warning's quiet q99 and multipliers from "
        "these quiet spans (ISO dates, end excluded; repeatable) for the area's "
        "monitoring: rtk: baselines -> rtk_thresholds.yaml <area>: accel:",
    )  # fmt: skip
    p.add_argument(
        "--cache", default=None,
        help="with --calibrate-rate/--calibrate-coherent: keep each pair-period "
        "statistic here (npz)",
    )  # fmt: skip
    p.add_argument(
        "--no-thresholds", action="store_true",
        help="ignore rtk_thresholds.yaml: the area's edges for every pair",
    )  # fmt: skip
    p.add_argument(
        "--calibrate-pairs", default=None,
        help="with --calibrate: these pairs (comma separated, or BASE:ELDC for "
        "every pair to a base) instead of the bench's; added to the area's",
    )  # fmt: skip
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


def _config_bench(prog: str, args: argparse.Namespace, config: Any) -> int:
    """``rtk-bench --area A [--set tune|test] [--quiet NAME]``: the bench of
    :mod:`gps_plot.rtk_bench` over rtk_bench.yaml."""
    from geo_dataread.rtk_read import list_baselines, rtk_dir

    from gps_plot import products, rtk_bench
    from gps_plot.cli import common

    path = (
        Path(args.config)
        if args.config
        else Path(config.config_path).expanduser() / "rtk_bench.yaml"
    )
    if not args.area:
        print(f"{prog}: --area (an area of {path})", file=sys.stderr)
        return 2
    dirs = list_baselines(rtk_dir(common.config_dir(config, "rtk_dir")))
    read = rtk_reader(dirs)
    thresholds = Path(config.config_path).expanduser() / THRESHOLDS_NAME
    try:
        doc = products.load_yaml(path)
        if args.calibrate:
            chosen = None
            if args.calibrate_pairs:
                chosen = []
                for item in args.calibrate_pairs.split(","):
                    if item.startswith("BASE:"):
                        chosen += [d for d in dirs if d.endswith(f"-{item[5:]}")]
                    else:
                        chosen.append(item)
            got = rtk_bench.calibrate(
                rtk_bench.parse_bench(doc, args.area), read, pairs=chosen
            )
            old = products.load_yaml(thresholds) if thresholds.is_file() else {}
            if chosen and args.area in old:  # add to the area's pairs
                got["pairs"] = {**(old[args.area].get("pairs") or {}), **got["pairs"]}
            write_thresholds(thresholds, merge_area(old, args.area, got))
            print(f"{prog}: {len(got['pairs'])} pair(s) calibrated -> {thresholds}")
        if args.calibrate_rate:
            got = rtk_bench.rate_calibrate(
                rtk_bench.parse_bench(doc, args.area), read, cache=args.cache
            )
            old = products.load_yaml(thresholds) if thresholds.is_file() else {}
            write_thresholds(thresholds, merge_area(old, args.area, got))
            for name, f in got["multipliers"].items():
                print(
                    f"{prog}: {name}: k {f['k']}, {f['episodes']} episode(s) in "
                    f"{f['months']} months (target {f['target_per_month']}/month)"
                )
            print(f"{prog}: {len(got['pairs'])} pair(s) -> {thresholds}")
        if args.calibrate_coherent:
            from gps_plot import rtk_coherent

            got = rtk_coherent.calibrate(
                rtk_bench.parse_bench(doc, args.area), read, cache=args.cache
            )
            old = products.load_yaml(thresholds) if thresholds.is_file() else {}
            write_thresholds(thresholds, merge_area(old, args.area, {"coherent": got}))
            for name, f in got["fit"].items():
                print(
                    f"{prog}: coherent {name}: k {f['k']}, {f['episodes']} episode(s) "
                    f"in {f['months']} months (target {f['target_per_month']}/month)"
                )
            print(
                f"{prog}: coherent template {sorted(got['pairs'])}, left out "
                f"{sorted(got['left_out'])} -> {thresholds}"
            )
        edges: dict[str, Any] = {}
        fit = None
        if not args.no_thresholds and thresholds.is_file():
            from gps_plot import rtk_coherent

            tdoc = products.load_yaml(thresholds)
            edges = load_pair_edges(tdoc, args.area)
            fit = rtk_coherent.load(tdoc, args.area)
        report = rtk_bench.run_config(
            doc, args.area, read, args.set or ["tune"], args.quiet or [], edges, fit
        )
    except ValueError as exc:
        print(f"{prog}: {path}: {exc}", file=sys.stderr)
        return 2
    print(report)
    if args.out:
        Path(args.out).expanduser().write_text(report)
    return 0


def _calibrate_accel(prog: str, args: argparse.Namespace, config: Any, cfg: Any) -> int:
    """``rtk-bench --area A --calibrate-accel START,END ...`` (T10)."""
    from geo_dataread.rtk_read import list_baselines, rtk_dir

    from gps_plot import products, rtk_accel
    from gps_plot.cli import common
    from gps_plot.cli.run import _cfg_file

    areas = products.area_table(products.load_yaml(_cfg_file(config, cfg, "areas")))
    spec = rtk_spec(areas.get(args.area or "") or {})
    if spec is None or not spec["baselines"]:
        print(f"{prog}: --area with monitoring: rtk: baselines", file=sys.stderr)
        return 2
    spans = []
    for item in args.calibrate_accel:
        a, _, b = item.partition(",")
        spans.append((a.strip(), b.strip()))
    read = rtk_reader(list_baselines(rtk_dir(common.config_dir(config, "rtk_dir"))))
    thresholds = Path(config.config_path).expanduser() / THRESHOLDS_NAME
    try:
        got = rtk_accel.calibrate(read, spec["baselines"], spans)
    except ValueError as exc:
        print(f"{prog}: {exc}", file=sys.stderr)
        return 2
    old = products.load_yaml(thresholds) if thresholds.is_file() else {}
    write_thresholds(thresholds, merge_area(old, args.area, {"accel": got}))
    print(
        f"{prog}: accel {args.area}: k {got['multipliers']} (fitted {got['fitted']}), "
        f"{got['quiet_months']} quiet months, episodes {got['episodes']} -> {thresholds}"
    )
    return 0


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
    if args.calibrate_accel:
        return _calibrate_accel(prog, args, config, cfg)
    if args.pair is None:
        return _config_bench(prog, args, config)
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
