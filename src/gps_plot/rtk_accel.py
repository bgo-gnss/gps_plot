"""The RTK acceleration warning (PLAN-rtk-trigger T10, BGÓ 2026-11-12).

What RTK should catch, and the daily GAMIT product cannot yet: an
ACCELERATION of the motion on 0.25–24 h scales (longer: the daily solutions
are more precise). One number per pair, time scale T and component:

    a_T = (v_now − v_background) · T    [mm]

the displacement over the last T that the background velocity of the two
days before does not explain. The chain is the bench's
(``~/gps-data/curation/rtk-trigger/slow/bank``, est_sg) step for step; the
calibration (quiet q99 and k) is only valid for exactly this chain:

1. 10-min medians of the fixed one-minute solutions, each labelled by its
   bin END (a value at t uses data ≤ t only).
2. Coarse despike: trailing Hampel, 3 h, 6σ, floor 5 mm.
3. Sidereal filter: minus the median, over the 7 previous days at the same
   sidereal time (23 h 56 min 4 s), of the HIGH-PASSED series (minus its
   centred 24 h median). Halves SVIN-SVIE's 1–6 h multipath noise.
4. Fine despike (trailing Hampel 3 h median, 12 h MAD, 5σ, floor 1 mm),
   then v_now = end-point Savitzky–Golay derivative (weighted LSQ order 2
   over the trailing L(T), gap-exact) and v_background = a line over the
   48 h before that window.
5. z = a_T / its quiet q99 per pair, scale and component (horizontal length,
   |Up|), z = the larger; level = the number of k at or below z, held for
   0.5·T on the 15-min grid; the pair's level = the highest over the scales,
   the area's = the highest over the pairs.

One k per level, fitted on the area's quiet span to the false-alarm targets
of T4d (yellow 1/month, orange 4/year, red 1/year), orange and red at
least :data:`MIN_RATIO` times yellow's. The level combines with
the T1 offset rule by the higher of the two (rtk_status.area_status).
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

#: time scales T [h]
SCALES: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 3.0, 6.0, 12.0, 24.0)
#: the trailing SG window L(T) [h]: ~3T, at least 2 h, 2T at 24 h
WINDOW_H: dict[float, float] = {0.25: 2.0, 0.5: 2.0, 1.0: 3.0, 2.0: 6.0,
                                3.0: 9.0, 6.0: 18.0, 12.0: 30.0, 24.0: 48.0}  # fmt: skip
ORDER = 2
BG_H = 48.0
MIN_FRAC = 0.6
MIN_FRAC_BG = 0.5
STEP = pd.Timedelta("10min")
STEP_H = 1.0 / 6.0
GRID = pd.Timedelta("15min")
GRID_H = 0.25
PERSIST_FRAC = 0.5
SID = pd.Timedelta(seconds=86164)  # GPS ground-track repeat
SIDEREAL_DAYS = 7
#: data before ``now`` the chain needs: 7 sidereal days + 12 h (centred 24 h
#: median) before the oldest row of 96 h of SG window + background, + slack
HISTORY = pd.Timedelta(days=12)
MONTH_H = 30.44 * 24
TARGETS = {"yellow": 1.0, "orange": 4.0 / 12.0, "red": 1.0 / 12.0}
K_GRID = np.round(np.geomspace(0.5, 40.0, 300), 4)
#: orange and red at least these times yellow's k: a short quiet span allows
#: no orange or red episode at all, so all three k fit just above its largest
#: excursion and the light would jump green → red (SVIN-SVIE 2026-11-12: 0.24
#: months, k 1.40/1.42/1.44). Provisional, until a few quiet months exist.
MIN_RATIO = (1.0, 1.5, 2.0)
#: σ of the current noise a level needs (BGÓ 2026-11-12), written by a
#: calibration; rtk_thresholds.yaml significance_sigma: 0 turns it off
SIGNIFICANCE = 2.0

#: ``(pair, start, end) -> (t, y_mm)``, rtk_status.rtk_reader's signature
Reader = Callable[[str, dt.datetime, dt.datetime], Any]


@dataclass
class Accel:
    """An area's calibrated acceleration warning (rtk_thresholds.yaml
    ``<area>: accel:``)."""

    #: pair -> (len(SCALES), 2) quiet q99 [mm] (horizontal, |Up|)
    scales: dict[str, np.ndarray]
    ks: tuple[float, float, float]
    quiet: list[list[str]] = field(default_factory=list)
    months: float = 0.0
    #: pair -> (len(SCALES), 2) the quiet robust σ of a_T [mm] (horizontal
    #: per axis, Up); empty: no significance test
    sigma: dict[str, np.ndarray] = field(default_factory=dict)
    #: pair -> (2,) the quiet input noise [mm] (horizontal per axis, Up)
    noise: dict[str, np.ndarray] = field(default_factory=dict)
    #: a level needs a_T ≥ this many σ of the CURRENT noise (BGÓ 2026-11-12:
    #: significant before yellow, as the daily light); 0: off
    significance: float = 0.0


# --- the chain (bench common.py + est_sg.py) ------------------------------


#: one-minute values a 10-min bin needs (BGÓ 2026-11-13: an hour of 3
#: sparse fixes in a gap, ~70 mm off, set the rtk-tie figure's scale)
MIN_PER_TEN = 4


def ten_min(
    t: np.ndarray, y_mm: np.ndarray, min_count: int = MIN_PER_TEN
) -> pd.DataFrame:
    """10-min medians (n, e, u [mm]) of one-minute rows, labelled by the
    bin's end; a bin with under ``min_count`` values is NaN. ``y_mm``:
    (3, N)."""
    df = pd.DataFrame(
        np.asarray(y_mm, float).T, index=pd.DatetimeIndex(t), columns=["n", "e", "u"]
    )
    bins = df.resample("10min", label="right", closed="right")
    return bins.median().where(bins.count() >= min_count)


def despike_coarse(df: pd.DataFrame, hours: float = 3.0, nsig: float = 6.0,
                   floor_mm: float = 5.0) -> pd.DataFrame:  # fmt: skip
    """Trailing Hampel: a value further than nsig robust sigmas (≥ floor)
    from the trailing median becomes NaN."""
    win = pd.Timedelta(hours=hours)
    med = df.rolling(win, min_periods=6).median()
    mad = (df - med).abs().rolling(win, min_periods=6).median() * 1.4826
    lim = np.maximum(nsig * mad, floor_mm)
    return df.where((df - med).abs() <= lim)


def sidereal(df: pd.DataFrame, days: int = SIDEREAL_DAYS) -> pd.DataFrame:
    """Minus the multipath model: the median over the previous ``days``
    sidereal days of the high-passed series (causal)."""
    hp = df - df.rolling(pd.Timedelta(hours=24), center=True, min_periods=36).median()
    shifted = [
        hp.shift(freq=SID * k).reindex(
            df.index, method="nearest", tolerance=pd.Timedelta("5min")
        )
        for k in range(1, days + 1)
    ]
    model = pd.concat(shifted).groupby(level=0).median()
    return df - model.reindex(df.index).fillna(0.0)


def despike_fine(y: pd.DataFrame, hours: float = 3.0, k: float = 5.0,
                 floor: float = 1.0) -> pd.DataFrame:  # fmt: skip
    """Trailing Hampel on a regular 10-min frame (the SG estimator's own)."""
    n = int(round(hours / STEP_H))
    med = y.rolling(n, min_periods=max(3, n // 3)).median()
    dev = (y - med).abs()
    mad = dev.rolling(4 * n, min_periods=max(3, n)).median()
    sig = np.maximum(1.4826 * mad, floor)
    return y.where(~(dev > k * sig))


def _causal_conv(a: np.ndarray, kern: np.ndarray) -> np.ndarray:
    """out[i] = Σ_j a[i − j] kern[j] along axis 0."""
    n = a.shape[0]
    if a.ndim == 1:
        return np.convolve(a, kern)[:n]
    return np.stack([np.convolve(a[:, c], kern)[:n] for c in range(a.shape[1])], axis=1)


def trailing_slope(
    y: np.ndarray, ok: np.ndarray, m: int, order: int, min_count: int
) -> np.ndarray:
    """Gap-exact end-point LSQ derivative [mm/h] over the trailing m rows."""
    w = ok.astype(float)
    yz = np.where(ok[:, None], y, 0.0)
    x = -np.arange(m, dtype=float) / m
    S = np.stack([_causal_conv(w, x**k) for k in range(2 * order + 1)], axis=1)
    R = np.stack([_causal_conv(yz, x**k) for k in range(order + 1)], axis=1)
    idx = np.add.outer(np.arange(order + 1), np.arange(order + 1))
    A = S[:, idx]
    count = np.rint(S[:, 0])
    good = count >= max(min_count, order + 2)
    out = np.full(y.shape, np.nan)
    if good.any():
        Ag = A[good]
        det = np.linalg.det(Ag)
        scale = np.abs(Ag).max(axis=(1, 2)) ** (order + 1)
        nonsing = np.abs(det) > 1e-10 * scale
        gi = np.flatnonzero(good)[nonsing]
        coef = np.linalg.solve(A[gi], R[gi])
        out[gi] = coef[:, 1, :] / (m * STEP_H)
    return out


def estimate(df: pd.DataFrame, grid: pd.DatetimeIndex) -> np.ndarray:
    """(len(grid), len(SCALES), 3) (v_now − v_bg)·T [mm], NaN where it
    cannot say; ``df`` = the sidereal-filtered 10-min frame."""
    out = np.full((len(grid), len(SCALES), 3), np.nan)
    d = df[["n", "e", "u"]].dropna(how="all")
    if d.empty or not len(grid):
        return out
    idx = pd.date_range(d.index.min().floor(STEP), d.index.max().floor(STEP), freq=STEP)
    y = d.groupby(d.index.floor(STEP)).median().reindex(idx)
    y = despike_fine(y - y.median())
    Y = y.to_numpy(float)
    ok = np.isfinite(Y).all(axis=1)
    N = len(idx)
    g = pd.DatetimeIndex(grid)
    pos = np.floor((g - idx[0]) / STEP).astype(int)
    inside = (pos >= 0) & (pos < N)
    pc = np.clip(pos, 0, N - 1)
    last_ok = pd.Series(np.where(ok, np.arange(N), np.nan)).ffill().to_numpy()
    lo = last_ok[pc]
    age_h = np.where(
        np.isfinite(lo),
        ((g - idx[0]) / pd.Timedelta("1h")).to_numpy() - np.nan_to_num(lo) * STEP_H,
        np.inf,
    )
    m_bg = int(round(BG_H / STEP_H))
    bg_all = trailing_slope(Y, ok, m_bg, 1, int(np.ceil(MIN_FRAC_BG * m_bg)))
    for j, T in enumerate(SCALES):
        m = int(round(WINDOW_H[T] / STEP_H))
        v = trailing_slope(Y, ok, m, ORDER, int(np.ceil(MIN_FRAC * m)))
        vbg = np.full_like(bg_all, np.nan)
        vbg[m:] = bg_all[:-m]
        acc = (v - vbg) * T
        sel = inside & (age_h <= max(0.25, T / 2))
        out[sel, j, :] = acc[pc[sel]]
    return out


def components(a: np.ndarray) -> np.ndarray:
    """(…, 3) n/e/u → (…, 2) horizontal length, |Up|."""
    return np.stack([np.hypot(a[..., 0], a[..., 1]), np.abs(a[..., 2])], -1)


#: the current input noise: the trailing span its scatter is taken over [h]
NOISE_H = 24.0


def input_noise(df: pd.DataFrame, grid: pd.DatetimeIndex) -> np.ndarray:
    """(len(grid), 2) the input's current noise [mm] (horizontal per axis,
    Up): the robust scatter (IQR/1.349) over the trailing ``NOISE_H`` of the
    10-min values' departure from their trailing 1-h median. Causal; it
    rises in storms, icing and bad multipath, when quiet-calibrated edges
    would let noise through."""
    out = np.full((len(grid), 2), np.nan)
    d = df[["n", "e", "u"]]
    if d.dropna(how="all").empty or not len(grid):
        return out
    resid = d - d.rolling(pd.Timedelta(hours=1), min_periods=3).median()
    win = pd.Timedelta(hours=NOISE_H)
    iqr = (
        resid.rolling(win, min_periods=36).quantile(0.75)
        - resid.rolling(win, min_periods=36).quantile(0.25)
    ) / 1.349
    h = np.sqrt((iqr["n"] ** 2 + iqr["e"] ** 2) / 2.0)
    both = pd.DataFrame({"h": h, "u": iqr["u"]})
    got = both.reindex(grid, method="ffill", tolerance=pd.Timedelta("20min"))
    return np.asarray(got.to_numpy(float), dtype=float)


def pair_series(
    read: Reader, pair: str, start: dt.datetime, end: dt.datetime,
    grid: pd.DatetimeIndex,
) -> tuple[np.ndarray, np.ndarray] | None:  # fmt: skip
    """``(estimate (len(grid), len(SCALES), 3), input noise (len(grid), 2))``
    for one pair from data in [start, end]; None without data."""
    t, y = read(pair, start, end)
    if not np.size(t):
        return None
    ten = despike_coarse(ten_min(t, y))
    df = sidereal(ten)
    return estimate(df, grid), input_noise(df, grid)


def pair_values(
    read: Reader, pair: str, start: dt.datetime, end: dt.datetime,
    grid: pd.DatetimeIndex,
) -> np.ndarray | None:  # fmt: skip
    """(len(grid), len(SCALES), 3) for one pair from data in [start, end],
    None without data."""
    got = pair_series(read, pair, start, end, grid)
    return None if got is None else got[0]


def axis_sigma(v: np.ndarray) -> np.ndarray:
    """(…, len(SCALES), 3) a_T → (len(SCALES), 2) robust σ [mm]: horizontal
    per axis (the RMS of N's and E's 1.4826·MAD), Up."""
    flat = v.reshape(-1, v.shape[-2], 3)
    med = np.nanmedian(flat, axis=0)
    mad = 1.4826 * np.nanmedian(np.abs(flat - med[None]), axis=0)  # (S, 3)
    return np.stack([np.sqrt((mad[:, 0] ** 2 + mad[:, 1] ** 2) / 2.0), mad[:, 2]], -1)


def significant(
    stats: np.ndarray, noise_now: np.ndarray, fit: Accel, pair: str
) -> np.ndarray | None:
    """(steps, scales, 2) whether each value stands ``fit.significance`` σ
    of the CURRENT noise above zero: σ_now = the quiet σ of a_T on that
    scale × (the input noise now / its quiet level). None without the test.
    The horizontal length is judged against √2 × the per-axis σ (its
    typical size under noise alone)."""
    if fit.significance <= 0 or pair not in fit.sigma or pair not in fit.noise:
        return None
    ratio = noise_now / fit.noise[pair][None, :]  # (steps, 2)
    ratio = np.where(np.isfinite(ratio), np.maximum(ratio, 1.0), 1.0)  # never laxer
    sig = fit.sigma[pair][None, :, :] * ratio[:, None, :]
    sig = sig * np.array([np.sqrt(2.0), 1.0])[None, None, :]
    return np.asarray(stats >= fit.significance * sig, dtype=bool)


def gated_z(
    stats: np.ndarray, q99: np.ndarray, ok: np.ndarray | None, k_yellow: float
) -> np.ndarray:
    """:func:`z_of` with each component that is not significant held just
    below yellow (it may still show how close it is, never light it)."""
    if ok is None:
        return z_of(stats, q99)
    z = stats / q99[None]
    z = np.where(ok, z, np.minimum(z, 0.999 * k_yellow))
    z = np.where(np.isfinite(z), z, -np.inf).max(axis=-1)
    return np.where(np.isfinite(z), z, np.nan)


# --- levels ---------------------------------------------------------------


def z_of(stats: np.ndarray, q99: np.ndarray) -> np.ndarray:
    """(steps, scales, 2) / (scales, 2) → (steps, scales) the larger z."""
    z = stats / q99[None]
    z = np.where(np.isfinite(z), z, -np.inf).max(axis=-1)
    return np.where(np.isfinite(z), z, np.nan)


def persist_steps(t_h: float) -> int:
    return max(1, int(round(PERSIST_FRAC * t_h / GRID_H)))


def scale_levels(z: np.ndarray, ks: Sequence[float]) -> np.ndarray:
    """(steps, scales) z → (steps, scales) level, each held 0.5·T."""
    out = np.full(z.shape, np.nan)
    k = np.asarray(ks, float)
    for j, t_h in enumerate(SCALES):
        s = z[:, j]
        lv = (s[:, None] >= k[None, :]).sum(axis=1).astype(float)
        lv[np.isnan(s)] = np.nan
        p = persist_steps(t_h)
        if p > 1:
            lv = (
                pd.Series(lv).rolling(p, min_periods=p // 2 + 1).min().to_numpy().copy()
            )
            lv[np.isnan(s)] = np.nan
        out[:, j] = lv
    return out


def held_z(z: np.ndarray) -> np.ndarray:
    """(steps, scales): the lowest z over each scale's persistence (what
    the level sees: z_held ≥ k ⇔ that level is lit, up to the rolling
    window's gap allowance)."""
    out = np.full(z.shape, np.nan)
    for j, t_h in enumerate(SCALES):
        p = persist_steps(t_h)
        s = pd.Series(z[:, j])
        out[:, j] = (
            s.rolling(p, min_periods=p // 2 + 1).min().to_numpy() if p > 1 else s
        )
    return out


def area_levels(z: Mapping[str, np.ndarray], ks: Sequence[float]) -> np.ndarray:
    """The highest level over pairs and scales per step (NaN: no verdict)."""
    m = np.concatenate([scale_levels(zz, ks).T for zz in z.values()])
    have = ~np.isnan(m).all(axis=0)
    return np.where(have, np.nanmax(np.where(np.isnan(m), -1, m), axis=0), np.nan)


def episodes(lv: np.ndarray, k: int) -> int:
    on = np.nan_to_num(lv, nan=-1) >= k
    return int(on[0]) + int(np.sum(on[1:] & ~on[:-1])) if on.size else 0


def months(lv: np.ndarray) -> float:
    return float(np.sum(~np.isnan(lv)) * GRID_H / MONTH_H)


def fit_ks(series: Sequence[Mapping[str, np.ndarray]]) -> list[float]:
    """One k per level, the lowest whose episodes stay within the target
    (the bench's fit: higher levels at 1e6 while the lower one is fitted)."""
    ks: list[float] = []
    for lvl, target in enumerate(TARGETS.values(), start=1):
        best = float(K_GRID[-1])
        for k in sorted(K_GRID, reverse=True):
            if ks and k <= ks[-1]:
                break
            trial = [*ks, float(k)] + [float(k) * 1e6] * (3 - lvl)
            lvs = [area_levels(z, trial) for z in series]
            mo = sum(months(lv) for lv in lvs)
            if sum(episodes(lv, lvl) for lv in lvs) > target * mo:
                break
            best = float(k)
        ks.append(best)
    return ks


# --- live + calibration -----------------------------------------------------


def live(
    read: Reader, pairs: Sequence[str], fit: Accel, now: dt.datetime
) -> dict[str, Any]:
    """The acceleration part of the area's status at ``now`` (naive UTC):
    the level, and what set it (pair, scale, component, a_T, Δv)."""
    end = pd.Timestamp(now).floor(GRID)
    hold = max(persist_steps(t) for t in SCALES)
    grid = pd.date_range(end - GRID * (hold + 1), end, freq=GRID)
    start = (end - HISTORY).to_pydatetime()
    rows: list[dict[str, Any]] = []
    best: tuple[float, float, dict[str, Any]] | None = None
    for pair in pairs:
        if pair not in fit.scales:
            continue
        row: dict[str, Any] = {"pair": pair}
        try:
            got = pair_series(read, pair, start, now, grid)
        except (OSError, ValueError) as exc:
            row["error"] = str(exc)
            rows.append(row)
            continue
        if got is None:
            row["stale"] = True
            rows.append(row)
            continue
        v, noise_now = got
        st = components(v)
        ok = significant(st, noise_now, fit, pair)
        z = gated_z(st, fit.scales[pair], ok, fit.ks[0])
        if ok is not None and pair in fit.noise:
            ratio = noise_now[-1] / fit.noise[pair]
            row["noise_ratio"] = [round(float(x), 2) for x in ratio]
        lv = scale_levels(z, fit.ks)[-1]
        if np.isnan(z[-1]).all():
            row["stale"] = True
            rows.append(row)
            continue
        zl = np.nan_to_num(z[-1], nan=-np.inf)
        zh = np.nan_to_num(held_z(z)[-1], nan=-np.inf)
        lvn = np.nan_to_num(lv, nan=-1.0)
        # the scale that holds the highest level, then the highest held z
        j = max(range(len(SCALES)), key=lambda i: (lvn[i], zh[i]))
        lvl = max(int(lvn[j]), 0)
        c = int(np.argmax(np.nan_to_num(st[-1, j] / fit.scales[pair][j], nan=-1.0)))
        t_h = SCALES[j]
        a = v[-1, j]
        row.update({
            "level": lvl,
            "scale_h": t_h,
            "component": ("horizontal", "up")[c],
            "z": round(float(zl[j]), 2),
            "z_held": round(float(zh[j]), 2),
            "value_mm": round(float(st[-1, j, c]), 1),
            "dv_mm_per_day": round(float(st[-1, j, c]) * 24.0 / t_h, 1),
            "neu_mm": [round(float(x), 1) for x in a],
        })  # fmt: skip
        rows.append(row)
        key = (lvl, float(zh[j]))
        if best is None or key > best[:2]:
            best = (*key, row)
    out: dict[str, Any] = {
        "multipliers": list(fit.ks),
        "scales_h": list(SCALES),
        "significance_sigma": fit.significance,
        "thresholds": thresholds(fit, pairs),
        "quiet": fit.quiet,
        "quiet_months": fit.months,
        "pairs": rows,
    }
    if best is None:
        out["level"] = None
        return out
    out["level"] = best[0]
    out["set_by"] = {k: best[2][k] for k in
                     ("pair", "scale_h", "component", "z", "z_held", "value_mm",
                      "dv_mm_per_day")}  # fmt: skip
    return out


def thresholds(fit: Accel, pairs: Sequence[str]) -> dict[str, list[dict[str, Any]]]:
    """What each level needs, per pair and scale (for the Vöktun page's
    table, BGÓ 2026-11-12): k · quiet q99 as extra displacement over T [mm]
    and as the change of velocity it means [mm/day], horizontal and Up."""
    out: dict[str, list[dict[str, Any]]] = {}
    for pair in pairs:
        q = fit.scales.get(pair)
        if q is None:
            continue
        rows = []
        for j, t_h in enumerate(SCALES):
            h = [round(float(k * q[j, 0]), 1) for k in fit.ks]
            u = [round(float(k * q[j, 1]), 1) for k in fit.ks]
            rows.append({
                "scale_h": t_h,
                "horizontal_mm": h,
                "up_mm": u,
                "horizontal_mm_per_day": [round(x * 24.0 / t_h, 1) for x in h],
                "up_mm_per_day": [round(x * 24.0 / t_h, 1) for x in u],
            })  # fmt: skip
        out[pair] = rows
    return out


def calibrate(
    read: Reader, pairs: Sequence[str], spans: Sequence[tuple[str, str]]
) -> dict[str, Any]:
    """rtk_thresholds.yaml ``<area>: accel:`` from the quiet ``spans``
    (ISO dates [start, end)): per pair and scale the quiet q99 of the
    horizontal length and |Up|, then the area's k. Data from ``HISTORY``
    before each span warm the chain up; only the span is scored."""
    raw: list[dict[str, np.ndarray]] = []
    raw_v: dict[str, list[np.ndarray]] = {}
    raw_noise: dict[str, list[np.ndarray]] = {}
    for a, b in spans:
        ta, tb = pd.Timestamp(a), pd.Timestamp(b)
        grid = pd.date_range(ta, tb, freq=GRID, inclusive="left")
        got = {}
        for p in pairs:
            pv = pair_series(
                read, p, (ta - HISTORY).to_pydatetime(), tb.to_pydatetime(), grid
            )
            if pv is not None:
                got[p] = components(pv[0])
                raw_v.setdefault(p, []).append(pv[0])
                raw_noise.setdefault(p, []).append(pv[1])
        raw.append(got)
    pooled: dict[str, list[np.ndarray]] = {}
    for d in raw:
        for p, s in d.items():
            pooled.setdefault(p, []).append(s)
    if not pooled:
        raise ValueError("accel: no data in the quiet spans")
    q99 = {
        p: np.nanpercentile(np.concatenate(v), 99, axis=0) for p, v in pooled.items()
    }
    zs = [{p: z_of(s, q99[p]) for p, s in d.items()} for d in raw]
    fitted = fit_ks(zs)
    ks = [max(k, r * fitted[0]) for k, r in zip(fitted, MIN_RATIO, strict=True)]
    lvs = [area_levels(z, ks) for z in zs if z]
    mo = sum(months(lv) for lv in lvs)
    return {
        "estimator": "savgol2-sidereal7",
        "scales_h": list(SCALES),
        "quiet": [[a, b] for a, b in spans],
        "quiet_months": round(mo, 2),
        "multipliers": [round(k, 3) for k in ks],
        "fitted": [round(k, 3) for k in fitted],
        "min_ratio": list(MIN_RATIO),
        "episodes": [sum(episodes(lv, k) for lv in lvs) for k in (1, 2, 3)],
        "significance_sigma": SIGNIFICANCE,
        "pairs": {
            p: {
                "q99_mm": [[round(float(x), 3) for x in r] for r in q],
                "sigma_mm": [
                    [round(float(x), 3) for x in r]
                    for r in axis_sigma(np.concatenate(raw_v[p]))
                ],
                "noise_mm": [
                    round(float(x), 3)
                    for x in np.nanmedian(np.concatenate(raw_noise[p]), axis=0)
                ],
            }
            for p, q in q99.items()
        },
    }


def load(doc: Mapping[str, Any], area: str) -> Accel | None:
    """The area's calibration from a loaded rtk_thresholds.yaml; None
    without one."""
    raw = (doc.get(area) or {}).get("accel")
    if not raw:
        return None
    if [float(s) for s in raw.get("scales_h") or []] != list(SCALES):
        raise ValueError(f"accel: {area} calibrated on other scales; recalibrate")
    return Accel(
        scales={p: np.asarray(v["q99_mm"], float) for p, v in raw["pairs"].items()},
        ks=tuple(float(k) for k in raw["multipliers"]),  # type: ignore[arg-type]
        quiet=[list(s) for s in raw.get("quiet") or []],
        months=float(raw.get("quiet_months") or 0.0),
        sigma={
            p: np.asarray(v["sigma_mm"], float)
            for p, v in raw["pairs"].items()
            if v.get("sigma_mm")
        },  # fmt: skip
        noise={
            p: np.asarray(v["noise_mm"], float)
            for p, v in raw["pairs"].items()
            if v.get("noise_mm")
        },  # fmt: skip
        significance=float(raw.get("significance_sigma") or 0.0),
    )
