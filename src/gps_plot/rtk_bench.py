"""The RTK warning bench over a set of events and quiet periods (T2).

``gpsplot rtk-bench --config rtk_bench.yaml --area reykjanes --set tune``
replays the warning of :mod:`gps_plot.rtk_status` (the same level function
the live ``rtk-run`` uses) on a 5-minute grid:

- around each event of the set, from ``before_hours`` before its swarm
  onset to ``after_hours`` after: per baseline and for the area (the
  highest level over the baselines with a verdict), the first time each
  level was reached and its delay after the swarm onset;
- over the quiet periods (``--quiet calibration|evaluation``): the alarms,
  counted as episodes (a run of grid steps at or above a level counts once),
  per baseline and for the area, per month of grid steps with a verdict.

The report is a markdown table, reproducible: the same config and data give
the same text (no run time in it). PLAN-rtk-trigger.md, rtk_bench.yaml.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from gps_plot import rtk_status as rs

STEP = dt.timedelta(minutes=5)
#: days per month for the false-alarm rate
MONTH_DAYS = 30.44
LEVELS = (1, 2, 3)


@dataclass(frozen=True)
class BenchEvent:
    id: str
    onset: dt.datetime
    set: str


@dataclass(frozen=True)
class Bench:
    """One area of rtk_bench.yaml."""

    area: str
    pairs: tuple[str, ...]
    spec: dict[str, Any]
    before: dt.timedelta
    after: dt.timedelta
    events: tuple[BenchEvent, ...]
    quiet: dict[str, tuple[tuple[dt.datetime, dt.datetime], ...]]


def _when(text: Any) -> dt.datetime:
    s = str(text)
    return dt.datetime.strptime(s, "%Y-%m-%d %H:%M" if ":" in s else "%Y-%m-%d")


def parse_bench(doc: Mapping[str, Any], area: str) -> Bench:
    """An area of a loaded rtk_bench.yaml.

    Raises:
        ValueError: no such area, no pairs or edges, a malformed event or
            quiet period.
    """
    raw = doc.get(area)
    if not isinstance(raw, Mapping):
        raise ValueError(f"rtk_bench.yaml has no area {area!r}")
    pairs = tuple(str(p) for p in raw.get("pairs") or ())
    if not pairs:
        raise ValueError(f"{area}: no pairs")
    spec = rs.rtk_spec({"monitoring": {"rtk": raw}})
    if spec is None:
        raise ValueError(f"{area}: no horizontal_mm edges")
    from gps_plot import rtk_coherent

    spec["coherent"] = rtk_coherent.params(raw.get("coherent"))
    events = []
    for e in raw.get("events") or ():
        if (
            not e.get("id")
            or not e.get("swarm_onset")
            or e.get("set") not in ("tune", "test")
        ):
            raise ValueError(
                f"{area}: an event needs id, swarm_onset and set tune|test: {e}"
            )
        events.append(BenchEvent(str(e["id"]), _when(e["swarm_onset"]), str(e["set"])))
    quiet: dict[str, tuple[tuple[dt.datetime, dt.datetime], ...]] = {}
    for name, periods in (raw.get("quiet") or {}).items():
        spans = []
        for start, end in periods:
            a, b = _when(start), _when(end)
            if b <= a:
                raise ValueError(f"{area}: quiet {name} period {start}..{end} is empty")
            spans.append((a, b))
        quiet[str(name)] = tuple(spans)
    return Bench(
        area=area,
        pairs=pairs,
        spec=spec,
        before=dt.timedelta(hours=float(raw.get("before_hours", 6))),
        after=dt.timedelta(hours=float(raw.get("after_hours", 3))),
        events=tuple(events),
        quiet=quiet,
    )


def grid(start: dt.datetime, end: dt.datetime) -> list[dt.datetime]:
    """The 5-minute grid ``[start, end]``."""
    out = []
    g = start
    while g <= end:
        out.append(g)
        g += STEP
    return out


def levels_of(
    steps: Sequence[tuple[dt.datetime, dict[str, Any] | None]],
    persist: int = rs.PERSIST_STEPS,
) -> list[int | None]:
    """The levels of a replay, each held only if it lasted ``persist``
    grid steps before too (:func:`persistent`, T7)."""
    return persistent([s["level"] if s else None for _, s in steps], persist)


def persistent(levels: Sequence[int | None], steps: int) -> list[int | None]:
    """T7 (BGÓ 2026-11-11): a level counts only when it held for the
    ``steps`` grid steps before as well: the level at i is the lowest of
    i - steps .. i. A step without a verdict there is passed over (a gap
    does not stop a level); no verdict at i stays None."""
    out: list[int | None] = []
    for i, v in enumerate(levels):
        if v is None or steps <= 0:
            out.append(v)
            continue
        prev = [p for p in levels[max(0, i - steps) : i] if p is not None]
        out.append(min([v, *prev]))
    return out


def pair_levels(
    steps: Mapping[str, Sequence[tuple[dt.datetime, dict[str, Any] | None]]],
    spec: Mapping[str, Any],
) -> dict[str, list[int | None]]:
    """Each pair's levels over one grid, as the live area_status: the
    sibling veto first when the area's common mode is ``siblings`` (T5,
    :func:`gps_plot.rtk_status.sibling_veto`, step by step over the pairs),
    then persistence (:func:`levels_of`, T7)."""
    if rs.common_mode(spec) != "siblings" or len(steps) < 2:
        return {p: levels_of(s) for p, s in steps.items()}
    names = list(steps)
    vetoed: dict[str, list[tuple[dt.datetime, dict[str, Any] | None]]] = {
        p: [] for p in names
    }
    for row in zip(*(steps[p] for p in names), strict=True):
        got = rs.sibling_veto({p: s for p, (_, s) in zip(names, row, strict=True)})
        for p, (g, _) in zip(names, row, strict=True):
            vetoed[p].append((g, got[p]))
    return {p: levels_of(s) for p, s in vetoed.items()}


def area_levels(
    per_pair: Mapping[str, Sequence[int | None]], agree: int = 1
) -> list[int | None]:
    """The area's level per step (:func:`gps_plot.rtk_status.combine`, as the
    live area_status): the highest over the pairs with a verdict, red only
    with two independent red pairs; None where none has a verdict."""
    names = list(per_pair)
    out: list[int | None] = []
    for row in zip(*(per_pair[n] for n in names), strict=True):
        out.append(rs.combine(dict(zip(names, row, strict=True)), agree))
    return out


def area_rule(
    steps: Mapping[str, Sequence[tuple[dt.datetime, dict[str, Any] | None]]],
    per_pair: Mapping[str, Sequence[int | None]],
    spec: Mapping[str, Any],
    n: int,
) -> list[int | None]:
    """The area's level per step by the area's rule, as the live
    area_status: ``statistic: coherent`` with a calibration
    (``coherent_fit``, :mod:`gps_plot.rtk_coherent`) from the pairs' offsets,
    else the pairs (:func:`area_levels`, ``agree``)."""
    fit = spec.get("coherent_fit") if spec.get("statistic") == "coherent" else None
    if fit is None:
        return area_levels(per_pair, spec.get("agree", 1)) if per_pair else [None] * n
    from gps_plot import rtk_coherent

    offs = {
        p: rtk_coherent.offsets_of([r for _, r in s], fit.windows)
        for p, s in steps.items()
        if p in fit.template
    }
    if not offs:
        return [None] * n
    return [
        None if np.isnan(v) else int(v) for v in rtk_coherent.area_levels(offs, fit)
    ]


def episodes(levels: Sequence[int | None]) -> dict[int, int]:
    """Alarm episodes per level: the number of times the level went from
    below ``k`` (or no verdict) to ``k`` or above, for k = 1, 2, 3."""
    out = {}
    for k in LEVELS:
        n, on = 0, False
        for v in levels:
            now = v is not None and v >= k
            n += now and not on
            on = now
        out[k] = n
    return out


def first_reached(
    times: Sequence[dt.datetime], levels: Sequence[int | None]
) -> dict[int, dt.datetime]:
    first: dict[int, dt.datetime] = {}
    for t, v in zip(times, levels, strict=True):
        for k in LEVELS:
            if v is not None and v >= k:
                first.setdefault(k, t)
    return first


def _read(read: rs.Reader, pair: str, start: dt.datetime, end: dt.datetime) -> Any:
    try:
        return read(pair, start, end)
    except (OSError, ValueError):
        return None


def event_result(bench: Bench, event: BenchEvent, read: rs.Reader) -> dict[str, Any]:
    """Per pair and for the area: the first time of each level around one
    event, and whether the pair had data."""
    spec = bench.spec
    lead = dt.timedelta(minutes=spec["window_minutes"] + spec["gap_minutes"]
                        + spec["reference_minutes"] + 5)  # fmt: skip
    times = grid(event.onset - bench.before, event.onset + bench.after)
    steps = {}
    for pair in bench.pairs:
        got = _read(read, pair, times[0] - lead, times[-1])
        if got is None or not got[0].size:
            continue
        t, y = got
        steps[pair] = rs.replay(rs.minutes(t), y, times, rs.pair_spec(spec, pair))
    per_pair = pair_levels(steps, spec)
    area = area_rule(steps, per_pair, spec, len(times))
    return {
        "event": event,
        "pairs": {p: first_reached(times, lv) for p, lv in per_pair.items()},
        "area": first_reached(times, area),
        "missing": [p for p in bench.pairs if p not in per_pair],
    }


def quiet_result(
    bench: Bench, periods: Sequence[tuple[dt.datetime, dt.datetime]], read: rs.Reader
) -> dict[str, Any]:
    """Alarm episodes per pair and for the area over quiet periods, with the
    months of grid steps that had a verdict."""
    spec = bench.spec
    lead = dt.timedelta(minutes=spec["window_minutes"] + spec["gap_minutes"]
                        + spec["reference_minutes"] + 5)  # fmt: skip
    counts: dict[str, dict[int, int]] = {
        p: dict.fromkeys(LEVELS, 0) for p in bench.pairs
    }
    covered: dict[str, int] = dict.fromkeys(bench.pairs, 0)
    area_counts = dict.fromkeys(LEVELS, 0)
    area_covered = 0
    for start, end in periods:
        times = grid(start, end - STEP)
        steps = {}
        for pair in bench.pairs:
            got = _read(read, pair, start - lead, end)
            if got is None or not got[0].size:
                continue
            steps[pair] = rs.replay(
                rs.minutes(got[0]), got[1], times, rs.pair_spec(spec, pair)
            )
        per_pair = pair_levels(steps, spec)
        for pair, lv in per_pair.items():
            for k, n in episodes(lv).items():
                counts[pair][k] += n
            covered[pair] += sum(v is not None for v in lv)
        if steps:
            area = area_rule(steps, per_pair, spec, len(times))
            for k, n in episodes(area).items():
                area_counts[k] += n
            area_covered += sum(v is not None for v in area)

    def months(steps: int) -> float:
        return steps * STEP.total_seconds() / 86400 / MONTH_DAYS

    return {
        "pairs": {p: (counts[p], months(covered[p])) for p in bench.pairs},
        "area": (area_counts, months(area_covered)),
    }


def _delay(first: Mapping[int, dt.datetime], k: int, onset: dt.datetime) -> str:
    when = first.get(k)
    if when is None:
        return "—"
    return f"{(when - onset).total_seconds() / 60:+.0f}"


def event_report(bench: Bench, results: Sequence[Mapping[str, Any]]) -> str:
    """Markdown: the area's delay per event, then per pair."""
    spec = bench.spec
    head = [
        f"### {bench.area}: delay after the swarm onset [min]",
        "",
        f"Window {spec['window_minutes']:g} min vs {spec['reference_minutes']:g} min, "
        + (
            "each pair's calibrated edges (rtk_thresholds.yaml), "
            if spec.get("pair_edges")
            else f"edges horizontal {spec['horizontal_mm']} mm, Up {spec['up_mm']} mm, "
        )
        + f"fixed solutions; grid 5 min from {bench.before} before to {bench.after} after."
        + (
            " Area: the coherent statistic (T9, rtk_thresholds.yaml coherent:)."
            if spec.get("statistic") == "coherent" and spec.get("coherent_fit")
            else f" Area: the pairs, agree {spec.get('agree', 1)}."
        ),
        "",
        "| Event | Swarm onset | Yellow | Orange | Red | Pairs with data |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        e = r["event"]
        n = len(bench.pairs) - len(r["missing"])
        head.append(
            f"| {e.id} | {e.onset:%Y-%m-%d %H:%M} | {_delay(r['area'], 1, e.onset)} "
            f"| {_delay(r['area'], 2, e.onset)} | {_delay(r['area'], 3, e.onset)} "
            f"| {n}/{len(bench.pairs)} |"
        )
    head += ["", "Per pair (yellow / orange / red):", "",
             "| Event | " + " | ".join(bench.pairs) + " |",
             "|---|" + "---|" * len(bench.pairs)]  # fmt: skip
    for r in results:
        e = r["event"]
        cells = []
        for p in bench.pairs:
            f = r["pairs"].get(p)
            cells.append(
                "no data" if f is None
                else " / ".join(_delay(f, k, e.onset) for k in LEVELS)
            )  # fmt: skip
        head.append(f"| {e.id} | " + " | ".join(cells) + " |")
    return "\n".join(head) + "\n"


def quiet_report(bench: Bench, name: str, result: Mapping[str, Any]) -> str:
    """Markdown: alarm episodes per month over the quiet periods."""
    spans = ", ".join(f"{a:%Y-%m-%d}..{b:%Y-%m-%d}" for a, b in bench.quiet[name])
    lines = [
        f"### {bench.area}: alarms on quiet data ({name}: {spans})",
        "",
        "| Series | Months with a verdict | Yellow / month | Orange / month | Red / month |",
        "|---|---|---|---|---|",
    ]

    def row(label: str, counts: Mapping[int, int], months: float) -> str:
        rate = [
            f"{counts[k] / months:.1f} ({counts[k]})" if months else "—" for k in LEVELS
        ]
        return f"| {label} | {months:.2f} | " + " | ".join(rate) + " |"

    lines.append(row("**area**", *result["area"]))
    for p in bench.pairs:
        lines.append(row(p, *result["pairs"][p]))
    return "\n".join(lines) + "\n"


def run_config(
    doc: Mapping[str, Any],
    area: str,
    read: rs.Reader,
    sets: Sequence[str],
    quiet: Sequence[str],
    pair_edges: Mapping[str, Any] | None = None,
    coherent: Any = None,
) -> str:
    """The bench report for ``sets`` of events and ``quiet`` period names;
    ``pair_edges`` (rtk_thresholds.yaml) in place of the area's edges;
    ``coherent`` (:class:`gps_plot.rtk_coherent.Coherent`) the area's
    calibrated statistic, used with ``statistic: coherent``."""
    bench = parse_bench(doc, area)
    if pair_edges:
        bench.spec["pair_edges"] = dict(pair_edges)
    if coherent is not None:
        bench.spec["coherent_fit"] = coherent
    out = []
    events = [e for e in bench.events if e.set in sets]
    if events:
        out.append(event_report(bench, [event_result(bench, e, read) for e in events]))
    for name in quiet:
        if name not in bench.quiet:
            raise ValueError(f"{area}: no quiet periods named {name!r}")
        out.append(
            quiet_report(bench, name, quiet_result(bench, bench.quiet[name], read))
        )
    return "\n".join(out)


#: T4 (BGÓ 2026-11-11, second pass): yellow at the 99.9 % quantile of the
#: quiet offsets, orange at 99.99 %, red just above the quiet maximum (the
#: maximum alone let one quiet glitch set every edge)
CALIBRATION_QUANTILES = (99.9, 99.99)
CALIBRATION_RED_FACTOR = 1.1


def quiet_edges(values: Sequence[float]) -> list[float]:
    """Yellow/orange/red edges from a pair-window's quiet offsets, made
    strictly increasing (a flat tail is nudged up by 0.1 mm)."""
    v = np.asarray(values, dtype=float)
    edges = [float(np.percentile(v, q)) for q in CALIBRATION_QUANTILES]
    edges.append(float(v.max()) * CALIBRATION_RED_FACTOR)
    for k in range(1, len(edges)):
        edges[k] = max(edges[k], edges[k - 1] + 0.1)
    return [round(e, 1) for e in edges]


def calibrate(
    bench: Bench,
    read: rs.Reader,
    quiet: str = "calibration",
    pairs: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Per pair and per newest window (T3) edges from the quiet offsets of
    each window over the ``quiet`` periods (5-minute grid), horizontal and
    |Up| apart (:func:`quiet_edges`). A pair without data there
    gets no edges (the area's apply). ``pairs``: instead of the bench's
    (e.g. every pair drawn on a figure)."""
    if quiet not in bench.quiet:
        raise ValueError(f"{bench.area}: no quiet periods named {quiet!r}")
    lead = dt.timedelta(minutes=bench.spec["window_minutes"] + bench.spec["gap_minutes"]
                        + bench.spec["reference_minutes"] + 5)  # fmt: skip
    out: dict[str, Any] = {}
    for pair in pairs or bench.pairs:
        windows: dict[float, Any] = {}
        for minutes_w in bench.spec["windows_minutes"]:
            spec = {**bench.spec, "windows_minutes": [minutes_w],
                    "horizontal_mm": [1e12], "up_mm": [1e12]}  # fmt: skip
            hs: list[float] = []
            us: list[float] = []
            for start, end in bench.quiet[quiet]:
                got = _read(read, pair, start - lead, end)
                if got is None or not got[0].size:
                    continue
                steps = rs.replay(
                    rs.minutes(got[0]), got[1], grid(start, end - STEP), spec
                )
                for _, s in steps:
                    if s is not None:
                        hs.append(s["horizontal_mm"])
                        us.append(abs(s["up_mm"]))
            if hs:
                windows[minutes_w] = {
                    "horizontal_mm": quiet_edges(hs),
                    "up_mm": quiet_edges(us),
                    "quiet_max_mm": [round(max(hs), 1), round(max(us), 1)],
                    "verdicts": len(hs),
                }
        if windows:
            out[pair] = {"windows": windows}
    spans = ", ".join(f"{a:%Y-%m-%d}..{b:%Y-%m-%d}" for a, b in bench.quiet[quiet])
    return {
        "calibrated_on": f"{quiet}: {spans}",
        "rule": "per newest window (offset vs the 6 h before, 5-min grid, fixed "
        f"solutions): yellow = quiet q{CALIBRATION_QUANTILES[0]}, orange = "
        f"q{CALIBRATION_QUANTILES[1]}, red = quiet max x {CALIBRATION_RED_FACTOR}",
        "pairs": out,
    }


# --- T4d: edges from a false-alarm rate (BGÓ 2026-11-11) -----------------

#: area false-alarm targets per month (done criteria: yellow ≤ 1/month,
#: red ≤ 1/year; orange 4/year is provisional, not decided)
RATE_TARGETS = {"yellow": 1.0, "orange": 4.0 / 12.0, "red": 1.0 / 12.0}
#: each detector's scale: this quantile of its quiet values (well resolved
#: by months of 5-min steps, unlike the tail the edges sit in)
SCALE_QUANTILE = 99.0
#: the global multipliers tried, a geometric grid
K_GRID = tuple(float(k) for k in np.round(np.geomspace(0.5, 40.0, 400), 4))


def window_stats(
    t_min: np.ndarray,
    y_mm: np.ndarray,
    times: Sequence[dt.datetime],
    spec: Mapping[str, Any],
) -> np.ndarray:
    """(steps, windows, 2) of each newest window's statistic at each grid
    time: horizontal offset length and |Up| [mm], NaN without a verdict.
    The same window offsets as :func:`gps_plot.rtk_status.baseline_level`."""
    windows = [
        float(w) for w in spec.get("windows_minutes") or [spec["window_minutes"]]
    ]
    out = np.full((len(times), len(windows), 2), np.nan)
    blind = {**spec, "pair_edges": {}, "window_edges": {},
             "horizontal_mm": [1e12], "up_mm": [1e12]}  # fmt: skip
    for i, (_, rec) in enumerate(rs.replay(t_min, y_mm, times, blind)):
        if rec is None:
            continue
        for j, w in enumerate(windows):
            got = rec["_windows"].get(w)
            if got is not None:
                o = got["offset"]
                out[i, j] = (float(np.hypot(o[0], o[1])), abs(float(o[2])))
    return out


def pair_tops(
    stats: Mapping[str, np.ndarray], scales: Mapping[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """Each pair's largest normalised statistic per step (over windows and
    components, statistic / scale), NaN without a verdict. Its level with
    edges ``k · scale`` is the number of k at or below it (as
    :func:`gps_analysis.rtk_alarm.alarm_level`, the highest over windows)."""
    out = {}
    for p, s in stats.items():
        if p not in scales:
            continue
        with np.errstate(invalid="ignore", divide="ignore"):
            z = s / scales[p][None, :, :]
        z = np.where(np.isfinite(z), z, -np.inf)
        top = z.max(axis=(1, 2))
        out[p] = np.where(np.isfinite(top), top, np.nan)
    return out


def _levels(top: np.ndarray, ks: Sequence[float]) -> np.ndarray:
    """Levels (float, NaN = no verdict) of ``top`` against ``ks``, then
    persistence (:func:`persistent`, one step: the lowest of this step and
    the previous one, when that one has a verdict)."""
    lv = (top[:, None] >= np.asarray(ks)[None, :]).sum(axis=1).astype(float)
    lv[np.isnan(top)] = np.nan
    return persist_steps(lv)


def persist_steps(lv: np.ndarray) -> np.ndarray:
    """Persistence on a float level array (NaN = no verdict): the lowest of
    each step and the previous one, when that one has a verdict."""
    # the previous step only (a step without a verdict there: no check)
    prev = np.full(lv.size, np.nan)
    prev[1:] = lv[:-1]
    out = np.where(np.isnan(prev), lv, np.fmin(lv, prev))
    out[np.isnan(lv)] = np.nan  # no verdict stays none
    return out


def _area(levels: Mapping[str, np.ndarray], agree: int = 1) -> np.ndarray:
    """The area's level per step as :func:`gps_plot.rtk_status.combine`:
    the highest, red only with two independent red pairs; ``agree=2``:
    every level needs two independent pairs at it or above."""
    from itertools import combinations

    names = list(levels)
    m = np.vstack([levels[n] for n in names])
    have = ~np.isnan(m).all(axis=0)
    top = np.where(have, np.nanmax(np.where(np.isnan(m), -1, m), axis=0), np.nan)
    pairs_ok = [(a, b) for a, b in combinations(range(len(names)), 2)
                if rs.independent(names[a], names[b])]  # fmt: skip
    if agree >= 2:
        out = np.where(have, 0.0, np.nan)
        for lvl in (1, 2, 3):
            at = m >= lvl
            two = np.zeros(m.shape[1], dtype=bool)
            for a, b in pairs_ok:
                two |= at[a] & at[b]
            out[two] = lvl
        return out
    red = m >= 3
    two = np.zeros(m.shape[1], dtype=bool)
    for a, b in pairs_ok:
        two |= red[a] & red[b]
    return np.where((top >= 3) & ~two, 2.0, top)


def _episodes(area: np.ndarray) -> dict[int, int]:
    """:func:`episodes` on a float level array (NaN = no verdict)."""
    out = {}
    for k in LEVELS:
        on = np.nan_to_num(area, nan=-1) >= k
        out[k] = int(on[0]) + int(np.sum(on[1:] & ~on[:-1]))
    return out


def area_episodes(
    tops: Sequence[Mapping[str, np.ndarray]], ks: Sequence[float], agree: int = 1
) -> tuple[dict[int, int], float]:
    """Area alarm episodes per level over the periods (:func:`pair_tops`
    per period), edges ``k · scale``, persistence and the two-red rule as
    the bench (:func:`quiet_result`); and the months of steps with a
    verdict."""
    counts = dict.fromkeys(LEVELS, 0)
    steps = 0
    for per_pair in tops:
        if not per_pair:
            continue
        area = _area({p: _levels(t, ks) for p, t in per_pair.items()}, agree)
        for k, n in _episodes(area).items():
            counts[k] += n
        steps += int(np.sum(~np.isnan(area)))
    return counts, steps * STEP.total_seconds() / 86400 / MONTH_DAYS


def rate_ks(
    tops: Sequence[Mapping[str, np.ndarray]],
    targets: Mapping[str, float] = RATE_TARGETS,
    agree: int = 1,
) -> tuple[list[float], dict[str, Any]]:
    """The global multipliers (yellow, orange, red) meeting the area
    targets [episodes per month] on the calibration periods.

    Episodes are not monotone in an edge (a higher one can split a run), so
    the grid is scanned from the top: each level's k is the lowest one from
    which every higher k also meets its target. Levels are found in order,
    each above the one before (orange and red counted with the lower k's
    fixed, so the two-red rule sees the real yellow and orange)."""
    names = ("yellow", "orange", "red")
    ks: list[float] = []
    found: dict[str, Any] = {}
    for lvl, name in enumerate(names, start=1):
        best: tuple[float, int | None, float | None] | None = None
        for k in sorted(K_GRID, reverse=True):
            if ks and k <= ks[-1]:
                break
            trial = [*ks, k] + [k * 1e6] * (3 - lvl)  # higher levels out of reach
            counts, months = area_episodes(tops, trial, agree)
            if counts[lvl] > targets[name] * months:
                break
            best = (k, counts[lvl], months)
        if best is None:  # even the top of the grid is too often: take it
            best = (K_GRID[-1], None, None)
        k_best, n_best, m_best = best
        ks.append(k_best)
        found[name] = {"k": round(k_best, 3), "episodes": n_best,
                       "months": None if m_best is None else round(m_best, 2),
                       "target_per_month": round(targets[name], 4)}  # fmt: skip
    return ks, found


def detector_scales(
    per_period: Sequence[Mapping[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    """Each pair's (windows, 2) scale: the :data:`SCALE_QUANTILE` of its
    quiet statistic per window and component over all the periods."""
    pooled: dict[str, list[np.ndarray]] = {}
    for stats in per_period:
        for p, s in stats.items():
            pooled.setdefault(p, []).append(s)
    out = {}
    for p, parts in pooled.items():
        a = np.concatenate(parts, axis=0)
        if not np.isfinite(a).any():
            continue
        out[p] = np.nanpercentile(a, SCALE_QUANTILE, axis=0)
    return out


def rate_thresholds(
    windows: Sequence[float],
    scales: Mapping[str, np.ndarray],
    ks: Sequence[float],
) -> dict[str, Any]:
    """rtk_thresholds.yaml ``pairs:``: per pair and window, edges k · scale."""
    out: dict[str, Any] = {}
    for p, sc in scales.items():
        per_w = {}
        for j, w in enumerate(windows):
            if not np.isfinite(sc[j]).all():
                continue
            h = [round(float(k * sc[j, 0]), 1) for k in ks]
            u = [round(float(k * sc[j, 1]), 1) for k in ks]
            for e in (h, u):
                for i in range(1, len(e)):
                    e[i] = max(e[i], round(e[i - 1] + 0.1, 1))
            per_w[float(w)] = {"horizontal_mm": h, "up_mm": u,
                               "scale_mm": [round(float(sc[j, 0]), 2),
                                            round(float(sc[j, 1]), 2)]}  # fmt: skip
        if per_w:
            out[p] = {"windows": per_w}
    return out


def period_stats(
    bench: Bench,
    read: rs.Reader,
    pair: str,
    start: dt.datetime,
    end: dt.datetime,
    cache: Any = None,
) -> np.ndarray | None:
    """:func:`window_stats` of one pair over one quiet period on the 5-min
    grid, None without data; kept in ``cache`` (a directory) as npz, keyed
    by the pair, the period and the windows."""
    from pathlib import Path

    spec = bench.spec
    ws = "_".join(f"{w:g}" for w in spec["windows_minutes"])
    key = (f"{pair}_{start:%Y%m%d}_{end:%Y%m%d}_w{ws}_r{spec['reference_minutes']:g}"
           f"_g{spec['gap_minutes']:g}")  # fmt: skip
    path = Path(cache) / f"{key}.npz" if cache else None
    if path is not None and path.is_file():
        got = np.load(path)["stats"]
        return None if got.size == 0 else got
    lead = dt.timedelta(minutes=max(spec["windows_minutes"]) + spec["gap_minutes"]
                        + spec["reference_minutes"] + 5)  # fmt: skip
    data = _read(read, pair, start - lead, end)
    stats = None
    if data is not None and data[0].size:
        stats = window_stats(
            rs.minutes(data[0]), data[1], grid(start, end - STEP), spec
        )
    if path is not None and stats is not None:  # no data is never kept
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, stats=stats)
    return stats


def rate_calibrate(
    bench: Bench,
    read: rs.Reader,
    quiet: str = "calibration",
    pairs: Sequence[str] | None = None,
    targets: Mapping[str, float] = RATE_TARGETS,
    cache: Any = None,
) -> dict[str, Any]:
    """T4d (BGÓ 2026-11-11): edges from the area's false-alarm targets.

    Each detector (pair × window × component) is scaled by its own quiet
    :data:`SCALE_QUANTILE`; one multiplier per level, common to all, is
    chosen so the AREA (persistence, the two-red rule) has at most
    ``targets`` episodes per month on the ``quiet`` periods
    (:func:`rate_ks`). Rates under about one per calibration span cannot be
    checked on it: red ≤ 1/year rests on zero red episodes there.
    ``pairs``: the area light's pairs (default the bench's); the budget is
    theirs. Returns the rtk_thresholds.yaml area document."""
    if quiet not in bench.quiet:
        raise ValueError(f"{bench.area}: no quiet periods named {quiet!r}")
    chosen = list(pairs or bench.pairs)
    per_period = []
    for start, end in bench.quiet[quiet]:
        got = {}
        for pair in chosen:
            s = period_stats(bench, read, pair, start, end, cache)
            if s is not None:
                got[pair] = s
        per_period.append(got)
    scales = detector_scales(per_period)
    tops = [pair_tops(stats, scales) for stats in per_period]
    agree = int(bench.spec.get("agree", 1))
    ks, found = rate_ks(tops, targets, agree)
    spans = ", ".join(f"{a:%Y-%m-%d}..{b:%Y-%m-%d}" for a, b in bench.quiet[quiet])
    return {
        "calibrated_on": f"{quiet}: {spans}",
        "rule": (
            "rate (T4d): each pair-window-component scaled by its quiet "
            f"q{SCALE_QUANTILE:g}; one multiplier per level so the area has at most "
            "the target episodes per month on the quiet months (5-min grid, "
            "persistence, two-red rule)"
        ),
        "targets_per_month": {k: round(v, 4) for k, v in targets.items()},
        "agree": agree,
        "multipliers": found,
        "area_pairs": sorted(scales),
        "pairs": rate_thresholds(bench.spec["windows_minutes"], scales, ks),
    }
