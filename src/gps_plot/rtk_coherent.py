"""The coherent RTK warning statistic (PLAN-rtk-trigger T9, BGÓ 2026-11-11).

The watch sees an intrusion as several stations moving together in a known
pattern; one pair's window offset is too weak early in an event. Here the
area gets ONE number per step, a matched projection on that pattern:

1. Template: each pair's direction of motion, the unit mean of its
   ``template_window`` offset [n, e, u] (A − B) over ``template_span``
   minutes after the swarm onset, on the bench's TUNE events only. A pair
   needs two tune events whose directions agree (every cosine to the mean
   > :data:`MIN_AGREEMENT`); the others are left out.
2. Per pair and window: the one-sided projection ``max(0, offset · u)`` (only
   motion along the pattern counts; noise in every other direction does
   not), divided by its quiet q99 → ``z``.
3. Per window: the matched sum ``Σ a·z / √Σ a²`` over the pairs with a
   verdict (at least ``min_pairs``; ``a`` = the pair's tune amplitude in
   noise units, BGÓ 2026-11-11), divided by its own quiet q99; ``S`` =
   the highest over the windows.
4. Level: the number of multipliers ``k`` at or below ``S``, one ``k`` per
   level fitted on the calibration months to the area targets of T4d
   (:data:`gps_plot.rtk_bench.RATE_TARGETS`); then the independence guard
   and persistence (one step, T7).

Independence guard (T7 analogue): a jump of a base moves every pair on it
by one vector, and several template pairs share ELDC or SUDV, so a large
enough base jump alone could reach red. Red therefore needs two pairs
independent by :func:`gps_plot.rtk_status.independent` with their own ``z``
≥ ``confirm_z`` at that step; otherwise it is held at orange. The guard
only lowers a level; the multipliers are fitted without it.

Calibrated products live in rtk_thresholds.yaml, ``<area>: coherent:``
(:func:`calibrate`, ``gpsplot rtk-bench --calibrate-coherent``); the
parameters in rtk_bench.yaml ``<area>: coherent:``; an area uses it with
``statistic: coherent`` in its rtk spec (live and bench).
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from itertools import combinations
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from gps_plot.rtk_bench import Bench
    from gps_plot.rtk_status import Reader

#: rtk_bench.yaml ``coherent:`` defaults (the 2026-11-11 sensitivity sweep,
#: bench-coherent-sweep.md: span and template window barely matter; the
#: 5-min window adds false yellows without catching more)
PARAMS = {
    "windows_minutes": [15.0, 30.0, 60.0],
    "min_pairs": 2,
    "template_window": 60.0,
    "template_span": [90.0, 150.0],
    "confirm_z": 1.0,
}
#: a pair joins the template only when every tune direction is this close
#: (cosine) to the mean
MIN_AGREEMENT = 0.5
#: and with at least this many tune events
MIN_TUNE_EVENTS = 2
SCALE_QUANTILE = 99.0


@dataclass(frozen=True)
class Coherent:
    """An area's calibrated coherent statistic (rtk_thresholds.yaml)."""

    windows: tuple[float, ...]
    min_pairs: int
    confirm_z: float
    template: dict[str, np.ndarray]  # pair -> unit [n, e, u]
    pair_scales: dict[str, np.ndarray]  # pair -> per window [mm]
    area_scales: np.ndarray  # per window
    ks: tuple[float, float, float]
    #: pair -> its expected tune amplitude in noise units (matched filter)
    weights: dict[str, float] = field(default_factory=dict)


def params(raw: Mapping[str, Any] | None) -> dict[str, Any]:
    """rtk_bench.yaml ``coherent:`` over :data:`PARAMS`, checked.

    Raises:
        ValueError: windows or span not positive, min_pairs < 1.
    """
    got: dict[str, Any] = {**PARAMS, **dict(raw or {})}
    out: dict[str, Any] = {
        "windows_minutes": sorted(float(w) for w in got["windows_minutes"]),
        "min_pairs": int(got["min_pairs"]),
        "template_window": float(got["template_window"]),
        "template_span": [float(x) for x in got["template_span"]],
        "confirm_z": float(got["confirm_z"]),
    }
    a, b = out["template_span"]
    if not out["windows_minutes"] or min(out["windows_minutes"]) <= 0:
        raise ValueError(f"coherent windows_minutes must be > 0: {out}")
    if out["template_window"] not in out["windows_minutes"]:
        raise ValueError(f"coherent template_window must be one of the windows: {out}")
    if out["min_pairs"] < 1 or not 0 <= a < b:
        raise ValueError(f"coherent min_pairs ≥ 1 and 0 ≤ span start < end: {out}")
    return out


def load(doc: Mapping[str, Any], area: str) -> Coherent | None:
    """The area's calibrated statistic from a loaded rtk_thresholds.yaml;
    None without one."""
    raw = (doc.get(area) or {}).get("coherent")
    if not raw:
        return None
    pairs = raw.get("pairs") or {}
    return Coherent(
        windows=tuple(float(w) for w in raw["windows_minutes"]),
        min_pairs=int(raw["min_pairs"]),
        confirm_z=float(raw["confirm_z"]),
        template={p: np.asarray(v["direction"], float) for p, v in pairs.items()},
        pair_scales={p: np.asarray(v["scales_mm"], float) for p, v in pairs.items()},
        area_scales=np.asarray(raw["area_scales"], float),
        ks=tuple(float(k) for k in raw["multipliers"]),  # type: ignore[arg-type]
        weights={p: float(v.get("weight", 1.0)) for p, v in pairs.items()},
    )


# --- the statistic --------------------------------------------------------


def pair_z(offsets: Mapping[str, np.ndarray], c: Coherent) -> dict[str, np.ndarray]:
    """``offsets``: pair -> (steps, len(c.windows), 3) [mm] (NaN: no
    verdict). Each template pair's one-sided projection over its quiet
    scale, (steps, windows)."""
    out = {}
    for p, o in offsets.items():
        if p in c.template:
            proj = np.clip(o @ c.template[p], 0.0, None)
            out[p] = proj / c.pair_scales[p][None, :]
    return out


def _combined_z(z: Mapping[str, np.ndarray], c: Coherent) -> np.ndarray:
    """(steps, windows) matched combination over the pairs with a verdict,
    ``Σ a·z / √Σ a²`` (``a`` = :attr:`Coherent.weights`, 1 without one),
    NaN with fewer than ``min_pairs``. With equal weights it is √n × the
    mean z; the weights put the pairs that move most (in noise units) first,
    which reaches yellow ~10 min sooner (bench-coherent-speed.md)."""
    names = list(z)
    m = np.stack([z[p] for p in names])
    a = np.array([c.weights.get(p, 1.0) for p in names])[:, None, None]
    have = ~np.isnan(m)
    num = np.sum(np.where(have, a * m, 0.0), axis=0)
    den = np.sqrt(np.sum(np.where(have, a**2, 0.0), axis=0))
    with np.errstate(invalid="ignore", divide="ignore"):
        s = num / den
    return np.where(have.sum(axis=0) >= c.min_pairs, s, np.nan)


def statistic(z: Mapping[str, np.ndarray], c: Coherent) -> np.ndarray:
    """S per step (:func:`pair_z` input), NaN without a verdict."""
    if not z:
        return np.empty(0)
    s = _combined_z(z, c) / c.area_scales[None, :]
    some = ~np.isnan(s).all(axis=1)
    top = np.max(np.where(np.isnan(s), -np.inf, s), axis=1)
    return np.where(some, top, np.nan)


def confirmed(z: Mapping[str, np.ndarray], confirm_z: float) -> np.ndarray:
    """Per step: two independent pairs each with z ≥ ``confirm_z`` (the
    highest over the windows)."""
    from gps_plot.rtk_status import independent

    names = list(z)
    if not names:
        return np.empty(0, bool)
    top = {p: np.max(np.where(np.isnan(z[p]), -np.inf, z[p]), axis=1) for p in names}
    out = np.zeros(next(iter(top.values())).shape[0], bool)
    for a, b in combinations(names, 2):
        if independent(a, b):
            out |= (top[a] >= confirm_z) & (top[b] >= confirm_z)
    return out


def levels(s: np.ndarray, ok: np.ndarray, ks: Sequence[float]) -> np.ndarray:
    """Levels (float, NaN = no verdict): the k at or below S, red held at
    orange unless ``ok`` (:func:`confirmed`), then persistence (the lowest
    of this step and the one before, when that one has a verdict)."""
    from gps_plot.rtk_bench import persist_steps

    raw = (s[:, None] >= np.asarray(ks)[None, :]).sum(axis=1).astype(float)
    raw[np.isnan(s)] = np.nan
    raw = np.where(~ok & (raw > 2), 2.0, raw)
    return persist_steps(raw)


def area_levels(offsets: Mapping[str, np.ndarray], c: Coherent) -> np.ndarray:
    """The area's coherent level per step from the pairs' offsets."""
    z = pair_z(offsets, c)
    if not z:
        n = next(iter(offsets.values())).shape[0] if offsets else 0
        return np.full(n, np.nan)
    return levels(statistic(z, c), confirmed(z, c.confirm_z), c.ks)


def offsets_of(
    records: Sequence[Mapping[str, Any] | None], windows: Sequence[float]
) -> np.ndarray:
    """(steps, windows, 3) signed offsets from :func:`gps_plot.rtk_status.
    baseline_level` records (their private ``_windows``), NaN where a record
    or window has no verdict."""
    out = np.full((len(records), len(windows), 3), np.nan)
    for i, rec in enumerate(records):
        if rec is None:
            continue
        per = rec.get("_windows") or {}
        for j, w in enumerate(windows):
            got = per.get(float(w))
            if got is not None:
                out[i, j] = got["offset"]
    return out


def live(
    records: Mapping[str, Sequence[Mapping[str, Any] | None]], c: Coherent
) -> dict[str, Any]:
    """The coherent part of the live status: ``records`` = pair -> the
    baseline_level records at (now − 5 min, now). ``level`` None: fewer than
    ``min_pairs`` template pairs with a verdict."""
    offs = {p: offsets_of(r, c.windows) for p, r in records.items() if p in c.template}
    z = pair_z(offs, c)
    used = sorted(p for p, v in z.items() if np.isfinite(v[-1]).any())
    if not z:
        return {"value": None, "level": None, "pairs": used}
    s = statistic(z, c)
    ok = confirmed(z, c.confirm_z)
    lv = levels(s, ok, c.ks)
    now = lv[-1]
    out: dict[str, Any] = {
        "value": None if np.isnan(s[-1]) else round(float(s[-1]), 2),
        "level": None if np.isnan(now) else int(now),
        "multipliers": list(c.ks),
        "pairs": used,
        "confirmed": bool(ok[-1]),
    }
    if not np.isnan(s[-1]) and s[-1] >= c.ks[2] and not ok[-1]:
        out["red_unconfirmed"] = True  # red by S, held at orange by the guard
    raw = levels(s[-1:], ok[-1:], c.ks)[0]
    if not np.isnan(raw) and not np.isnan(now) and raw > now:
        out["level_raw"] = int(raw)
    return out


# --- calibration ----------------------------------------------------------


def _window_offsets(
    t_min: np.ndarray,
    y_mm: np.ndarray,
    times: Sequence[dt.datetime],
    spec: Mapping[str, Any],
) -> np.ndarray:
    """(steps, spec windows, 3) signed offsets on a grid, edges ignored."""
    from gps_plot import rtk_status as rs

    blind = {**spec, "pair_edges": {}, "window_edges": {},
             "horizontal_mm": [1e12], "up_mm": [1e12]}  # fmt: skip
    recs = [rec for _, rec in rs.replay(t_min, y_mm, times, blind)]
    return offsets_of(recs, spec["windows_minutes"])


def period_offsets(
    bench: Bench,
    read: Reader,
    pair: str,
    start: dt.datetime,
    end: dt.datetime,
    cache: Any = None,
) -> np.ndarray | None:
    """Signed offsets of one pair on the 5-min grid ``[start, end)`` for
    every bench window, None without data; kept in ``cache`` as npz (the
    key of the 2026-11-11 experiments, ``_v3``) when there is data."""
    from gps_plot import rtk_bench as rb
    from gps_plot import rtk_status as rs

    spec = bench.spec
    ws = "_".join(f"{w:g}" for w in spec["windows_minutes"])
    key = (f"{pair}_{start:%Y%m%d%H%M}_{end:%Y%m%d%H%M}_w{ws}"
           f"_r{spec['reference_minutes']:g}_g{spec['gap_minutes']:g}_v3")  # fmt: skip
    path = Path(cache) / f"{key}.npz" if cache else None
    if path is not None and path.is_file():
        got = np.load(path)["offsets"]
        return None if got.size == 0 else got
    lead = dt.timedelta(minutes=max(spec["windows_minutes"]) + spec["gap_minutes"]
                        + spec["reference_minutes"] + 5)  # fmt: skip
    data = rb._read(read, pair, start - lead, end)
    out = None
    if data is not None and data[0].size:
        out = _window_offsets(
            rs.minutes(data[0]), data[1], rb.grid(start, end - rb.STEP), spec
        )
    # no data is never kept: an unmounted archive reads as empty, not as an
    # error, and would poison the cache (2026-11-11)
    if path is not None and out is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, offsets=out)
    return out


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def learn_template(
    tune: Sequence[tuple[str, int, Mapping[str, np.ndarray]]],
    windows: Sequence[float],
    template_window: float,
    span: Sequence[float],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Each pair's direction from the tune events: ``tune`` = (event id,
    grid index of the onset, pair -> (steps, windows, 3) offsets). Returns
    the kept unit vectors and, per pair, what was learned (events,
    cosines, offset sizes, kept)."""
    jt = list(windows).index(float(template_window))
    a, b = (int(round(x / 5.0)) for x in span)
    seen: dict[str, list[tuple[str, np.ndarray]]] = {}
    for eid, i0, offs in tune:
        for p, arr in offs.items():
            seg = arr[i0 + a : i0 + b + 1, jt]
            if np.isfinite(seg).all(axis=1).any():
                seen.setdefault(p, []).append((eid, np.nanmean(seg, axis=0)))
    template, report = {}, {}
    for p, got in seen.items():
        us = [_unit(v) for _, v in got]
        u = _unit(np.mean(us, axis=0))
        cos = [round(float(x @ u), 3) for x in us]
        keep = len(us) >= MIN_TUNE_EVENTS and min(cos) > MIN_AGREEMENT
        if keep:
            template[p] = u
        report[p] = {
            "events": [e for e, _ in got],
            "agreement": cos,
            "offset_mm": [round(float(np.linalg.norm(v)), 1) for _, v in got],
            "direction": [round(float(x), 6) for x in u],
            "kept": keep,
        }
    return template, report


def fit_ks(
    series: Sequence[np.ndarray], targets: Mapping[str, float]
) -> tuple[list[float], dict[str, Any]]:
    """Multipliers (yellow, orange, red) meeting the area targets [episodes
    per month] on ``series`` of S (no guard), scanned from the top as
    :func:`gps_plot.rtk_bench.rate_ks`."""
    from gps_plot import rtk_bench as rb

    steps = sum(int(np.sum(~np.isnan(s))) for s in series)
    months = steps * rb.STEP.total_seconds() / 86400 / rb.MONTH_DAYS
    ks: list[float] = []
    found: dict[str, Any] = {}
    for lvl, name in enumerate(("yellow", "orange", "red"), start=1):
        best: tuple[float, int | None] = (rb.K_GRID[-1], None)
        for k in sorted(rb.K_GRID, reverse=True):
            if ks and k <= ks[-1]:
                break
            trial = [*ks, k] + [k * 1e6] * (3 - lvl)
            n = sum(rb._episodes(rb._levels(s, trial))[lvl] for s in series if s.size)
            if n > targets[name] * months:
                break
            best = (k, n)
        ks.append(best[0])
        found[name] = {"k": round(best[0], 3), "episodes": best[1],
                       "months": round(months, 2),
                       "target_per_month": round(targets[name], 4)}  # fmt: skip
    return ks, found


def calibrate(
    bench: Bench,
    read: Reader,
    quiet: str = "calibration",
    targets: Mapping[str, float] | None = None,
    cache: Any = None,
) -> dict[str, Any]:
    """T9: the template from the tune events, the quiet scales and the
    multipliers from the ``quiet`` months; the rtk_thresholds.yaml
    ``<area>: coherent:`` document.

    Raises:
        ValueError: no such quiet periods, or fewer than ``min_pairs``
            template pairs.
    """
    from gps_plot import rtk_bench as rb

    if quiet not in bench.quiet:
        raise ValueError(f"{bench.area}: no quiet periods named {quiet!r}")
    par = bench.spec.get("coherent") or params(None)
    all_w = [float(w) for w in bench.spec["windows_minutes"]]
    missing = [w for w in [*par["windows_minutes"], par["template_window"]]
               if w not in all_w]  # fmt: skip
    if missing:
        raise ValueError(f"{bench.area}: coherent windows {missing} not in {all_w}")
    sel = [all_w.index(w) for w in par["windows_minutes"]]

    tune = []
    for e in bench.events:
        if e.set != "tune":
            continue
        start, end = e.onset - bench.before, e.onset + bench.after + rb.STEP
        offs = {}
        for p in bench.pairs:
            o = period_offsets(bench, read, p, start, end, cache)
            if o is not None:
                offs[p] = o
        tune.append((e.id, int((e.onset - start) / rb.STEP), offs))
    template, learned = learn_template(
        tune, all_w, par["template_window"], par["template_span"]
    )
    if len(template) < par["min_pairs"]:
        raise ValueError(
            f"{bench.area}: {len(template)} template pair(s), coherent needs "
            f"{par['min_pairs']}"
        )

    periods = []
    for start, end in bench.quiet[quiet]:
        got = {}
        for p in template:
            o = period_offsets(bench, read, p, start, end, cache)
            if o is not None:
                got[p] = o[:, sel]
        periods.append(got)
    # scales: per pair, then of the area mean (both q99 of quiet)
    proj: dict[str, list[np.ndarray]] = {}
    for got in periods:
        for p, o in got.items():
            proj.setdefault(p, []).append(np.clip(o @ template[p], 0.0, None))
    pair_scales = {
        p: np.nanpercentile(np.concatenate(v), SCALE_QUANTILE, axis=0)
        for p, v in proj.items()
    }
    c0 = Coherent(
        windows=tuple(par["windows_minutes"]),
        min_pairs=par["min_pairs"],
        confirm_z=par["confirm_z"],
        template={p: u for p, u in template.items() if p in pair_scales},
        pair_scales=pair_scales,
        area_scales=np.ones(len(sel)),
        ks=(1.0, 2.0, 3.0),
    )
    # weights: each pair's mean tune projection in its template window over
    # the template span, in units of its quiet q99 there
    jt = all_w.index(par["template_window"])
    a, b = (int(round(x / 5.0)) for x in par["template_span"])
    weights = {}
    for p, u in c0.template.items():
        vals = []
        for _, i0, offs in tune:
            if p in offs:
                seg = offs[p][i0 + a : i0 + b + 1, jt] @ u
                if np.isfinite(seg).any():
                    vals.append(float(np.nanmean(seg)))
        q = pair_scales[p][par["windows_minutes"].index(par["template_window"])]
        weights[p] = float(np.mean(vals) / q) if vals else 0.0
    c0 = replace(c0, weights=weights)
    combined = [_combined_z(pair_z(got, c0), c0) for got in periods if pair_z(got, c0)]
    area_scales = np.nanpercentile(np.concatenate(combined), SCALE_QUANTILE, axis=0)
    c1 = replace(c0, area_scales=area_scales)
    series = [statistic(pair_z(got, c1), c1) for got in periods]
    ks, found = fit_ks(series, targets or rb.RATE_TARGETS)
    spans = ", ".join(f"{a:%Y-%m-%d}..{b:%Y-%m-%d}" for a, b in bench.quiet[quiet])
    return {
        "calibrated_on": f"{quiet}: {spans}",
        "learned_on": [eid for eid, _, _ in tune],
        "rule": (
            "coherent (T9): one-sided projection of each pair's window offset on "
            "its tune-event direction / quiet q99, matched sum over the pairs "
            "(weight = tune amplitude in noise units), / quiet q99, highest over "
            "the windows; one multiplier per level to the area "
            "targets; red needs two independent pairs with z >= confirm_z"
        ),
        "windows_minutes": [float(w) for w in par["windows_minutes"]],
        "min_pairs": par["min_pairs"],
        "confirm_z": par["confirm_z"],
        "template_window": par["template_window"],
        "template_span": par["template_span"],
        "targets_per_month": {
            k: round(v, 4) for k, v in (targets or rb.RATE_TARGETS).items()
        },
        "multipliers": [round(k, 4) for k in ks],
        "fit": found,
        "area_scales": [round(float(s), 4) for s in area_scales],
        "pairs": {
            p: {
                "direction": learned[p]["direction"],
                "scales_mm": [round(float(s), 3) for s in pair_scales[p]],
                "weight": round(weights[p], 4),
                "agreement": learned[p]["agreement"],
                "offset_mm": learned[p]["offset_mm"],
                "events": learned[p]["events"],
            }
            for p in sorted(c1.template)
        },
        "left_out": {
            p: {"events": v["events"], "agreement": v["agreement"]}
            for p, v in sorted(learned.items())
            if not v["kept"]
        },
    }
