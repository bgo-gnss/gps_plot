"""The RTK rate warning (BGÓ 2026-11-12): the SPEED of the motion over the
newest 24 h and 72 h, on the daily light's levels.

The offset rule (1 h against the 6 h before) catches a sudden shift, the
acceleration rule (:mod:`gps_plot.rtk_accel`) a speed step within 0.25–24 h.
Neither sees a new steady pace of a few mm/d that starts within a day: the
Svínafellsheiði episode of 8 Nov 2026 (SVIN-SVIE ≈ 2.5 mm/d along, ≈ −6 mm/d
Up) left the light green. The daily light has the right scale (1/3/5 mm/d
horizontal, 2/5/7 Up) but its 14 days lag. This rule puts that scale on the
RTK: the rate over each window, a level only above ``significance`` σ.

The chain is the acceleration rule's up to the sidereal filter: 10-min
medians of the fixed one-minute solutions, coarse despike, minus the
sidereal multipath model; then 1-h medians. Per window W and end time:

- the rate: a least-squares line over (end − W, end] [mm/d], with at least
  :data:`MIN_FRAC` of the hours present;
- its σ per component, the larger of (a) the line's white-noise σ and (b)
  the repeatability: the W-rates at every hour over the
  max(:data:`HISTORY_DAYS`, 4 W) before, their robust scatter (1.4826·MAD)
  about a line through them in time (the polar figure's empirical σ,
  :mod:`gps_plot.polar_rtk`); (b) holds the day-to-day multipath and
  atmosphere that one window cannot see;
- horizontal: the speed against its σ along it; Up: |rate| against its σ.

Level = the number of edges at or below the value, when the value stands
``significance`` σ above zero; the pair's level the highest over the
windows and the components asked for.

Bench (BGÓ 2026-11-13, SVIN-SVIE, hourly 27 Oct–12 Nov; the RTK archive
starts 27 Oct, the sidereal filter is warm from about 3 Nov): quiet 3–7 Nov,
24 h horizontal 5 false yellow hours at 2 σ, none at 3 σ; 72 h horizontal
none at either; Up far too noisy for the 2/5/7 mm/d edges (24 h σ ≈ 4.4,
72 h σ ≈ 1.3 mm/d: 13 false red, 15 false yellow hours at 2 σ). The 8 Nov
episode at 3 σ, horizontal: yellow + orange 8 Nov 17:00 (24 h, 3.6 mm/d),
the 72 h yellow 9 Nov 02:00. Hence Svínafellsheiði's rate: horizontal,
24 h and 72 h, 3 σ. Eleven quiet days only: provisional.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

#: the windows [h]
WINDOWS_H: tuple[float, ...] = (24.0, 72.0)
#: the daily light's edges [mm/d] (gps_plot.track_status)
HORIZONTAL_MM_PER_DAY = (1.0, 3.0, 5.0)
UP_MM_PER_DAY = (2.0, 5.0, 7.0)
#: the repeatability's look-back: at least this, and four windows
HISTORY_DAYS = 7.0
#: hours present a window needs
MIN_FRAC = 0.6
#: the sidereal filter's warm-up (rtk_accel: 7 sidereal days) and a margin
WARMUP = pd.Timedelta(days=8)
SIGNIFICANCE = 2.0
#: the components with a level (BGÓ 2026-11-13 bench: RTK Up is too noisy
#: for the daily light's Up edges, so Svínafellsheiði uses horizontal only)
COMPONENTS = ("horizontal", "up")

Reader = Callable[[str, dt.datetime, dt.datetime], Any]


def spec_of(raw: Any) -> dict[str, Any] | None:
    """``monitoring: rtk: rate:`` with defaults; None when absent or
    ``false``.

    Raises:
        ValueError: edges not increasing, or windows not positive.
    """
    if raw is None or raw is False:
        return None
    raw = raw if isinstance(raw, Mapping) else {}
    out: dict[str, Any] = {
        "windows_h": sorted(float(w) for w in raw.get("windows_h", WINDOWS_H)),
        "components": [str(c) for c in raw.get("components", COMPONENTS)],
        # None: the area's significance_sigma
        "significance_sigma": (float(raw["significance_sigma"])
                               if raw.get("significance_sigma") is not None else None),
        "horizontal_mm_per_day": [float(x) for x in
                                  raw.get("horizontal_mm_per_day", HORIZONTAL_MM_PER_DAY)],
        "up_mm_per_day": [float(x) for x in raw.get("up_mm_per_day", UP_MM_PER_DAY)],
    }  # fmt: skip
    if not out["components"] or set(out["components"]) - set(COMPONENTS):
        raise ValueError(
            f"rtk rate components from {COMPONENTS}, got {out['components']}"
        )
    if any(w <= 0 for w in out["windows_h"]):
        raise ValueError(f"rtk rate windows_h must be > 0, got {out['windows_h']}")
    for k in ("horizontal_mm_per_day", "up_mm_per_day"):
        e = out[k]
        if not all(a < b for a, b in zip(e, e[1:], strict=False)):
            raise ValueError(f"rtk rate {k} must increase, got {e}")
    return out


def hourly(t: Any, y: Any) -> pd.DataFrame:
    """The acceleration rule's chain to the sidereal filter, then 1-h
    medians (end-labelled) [mm], columns N, E, U."""
    from gps_plot import rtk_accel

    df = rtk_accel.sidereal(rtk_accel.despike_coarse(rtk_accel.ten_min(t, y)))
    out = df.resample("1h", label="right", closed="right").median()
    out.columns = ["N", "E", "U"]
    return out


def window_rates(h: pd.DataFrame, hours: float, ends: pd.DatetimeIndex) -> np.ndarray:
    """(len(ends), 3) rate [mm/d] and (len(ends), 3) white σ over (end −
    hours, end] at each end, NaN with too few hours; stacked (2, n, 3)."""
    x_all = (h.index - h.index[0]) / pd.Timedelta(days=1)
    x_all = np.asarray(x_all, float)
    y_all = h.to_numpy(float)
    out = np.full((2, len(ends), 3), np.nan)
    need = MIN_FRAC * hours
    lo_idx = np.searchsorted(h.index.values, (ends - pd.Timedelta(hours=hours)).values,
                             side="right")  # fmt: skip
    hi_idx = np.searchsorted(h.index.values, ends.values, side="right")
    for i, (a, b) in enumerate(zip(lo_idx, hi_idx, strict=True)):
        x, y = x_all[a:b], y_all[a:b]
        for c in range(3):
            ok = np.isfinite(y[:, c])
            if ok.sum() < max(need, 3):
                continue
            xx = x[ok] - x[ok].mean()
            sxx = float(np.sum(xx**2))
            if sxx <= 0:
                continue
            slope = float(np.sum(xx * (y[ok, c] - y[ok, c].mean())) / sxx)
            res = y[ok, c] - y[ok, c].mean() - slope * xx
            dof = max(int(ok.sum()) - 2, 1)
            out[0, i, c] = slope
            out[1, i, c] = float(np.sqrt(np.sum(res**2) / dof / sxx))
    return out


def repeat_sigma(rates: np.ndarray, t_days: np.ndarray) -> np.ndarray:
    """(3,) the robust scatter (1.4826·MAD) of ``rates`` (n, 3) about a
    line through them in ``t_days``; NaN with under 6."""
    out = np.full(3, np.nan)
    for c in range(3):
        ok = np.isfinite(rates[:, c])
        if ok.sum() < 6:
            continue
        p = np.polyfit(t_days[ok], rates[ok, c], 1)
        r = rates[ok, c] - np.polyval(p, t_days[ok])
        out[c] = 1.4826 * float(np.median(np.abs(r - np.median(r))))
    return out


def levels_of(value: float, sigma: float, edges: Sequence[float],
              significance: float) -> int:  # fmt: skip
    """The number of ``edges`` at or below ``value``, 0 unless it stands
    ``significance`` σ above zero (or σ is unknown)."""
    if not np.isfinite(value):
        return 0
    if significance > 0 and (not np.isfinite(sigma) or value <= significance * sigma):
        return 0
    return sum(1 for e in edges if value >= e)


def pair_rates(
    h: pd.DataFrame, spec: Mapping[str, Any], end: pd.Timestamp, significance: float
) -> list[dict[str, Any]]:
    """Per window, the rate at ``end`` and its level (one dict each)."""
    rows = []
    for w in spec["windows_h"]:
        back = max(HISTORY_DAYS, 4.0 * w / 24.0)
        ends = pd.date_range(end - pd.Timedelta(days=back), end, freq="1h")
        got = window_rates(h, w, ends)
        v, white = got[0, -1], got[1, -1]
        t_days = np.asarray((ends - ends[0]) / pd.Timedelta(days=1), float)
        rep = repeat_sigma(got[0, :-1], t_days[:-1])
        sig = np.fmax(white, rep)
        speed = float(np.hypot(v[0], v[1]))
        if speed > 0:
            un, ue = v[0] / speed, v[1] / speed
        else:
            un, ue = 1.0, 0.0
        sig_h = float(np.hypot(un * sig[0], ue * sig[1]))
        up = float(abs(v[2]))
        comps = spec.get("components", COMPONENTS)
        lh = (levels_of(speed, sig_h, spec["horizontal_mm_per_day"], significance)
              if "horizontal" in comps else 0)  # fmt: skip
        lu = (levels_of(up, float(sig[2]), spec["up_mm_per_day"], significance)
              if "up" in comps else 0)  # fmt: skip
        rows.append({
            "window_h": w,
            "level": max(lh, lu),
            "component": ("horizontal" if lh > lu or (lh == lu and "horizontal" in comps)
                          else "up"),
            "horizontal_mm_per_day": round(speed, 2),
            "horizontal_sigma": round(sig_h, 2),
            "azimuth": round(float(np.degrees(np.arctan2(v[1], v[0])) % 360.0), 0),
            "up_mm_per_day": round(float(v[2]), 2),
            "up_sigma": round(float(sig[2]), 2),
        })  # fmt: skip
    return rows


def live(
    read: Reader,
    pairs: Sequence[str],
    spec: Mapping[str, Any],
    now: dt.datetime,
    significance: float = SIGNIFICANCE,
) -> dict[str, Any]:
    """The rate part of the area's status at ``now`` (naive UTC): the
    level and what set it (pair, window, component, rate ± σ); the
    spec's own significance_sigma over ``significance`` when set."""
    if spec.get("significance_sigma") is not None:
        significance = float(spec["significance_sigma"])
    end = pd.Timestamp(now).floor("1h")
    longest = max(spec["windows_h"]) / 24.0
    back = pd.Timedelta(days=longest + max(HISTORY_DAYS, 4.0 * longest)) + WARMUP
    rows: list[dict[str, Any]] = []
    best: tuple[int, float, dict[str, Any]] | None = None
    for pair in pairs:
        row: dict[str, Any] = {"pair": pair}
        try:
            t, y = read(pair, (end - back).to_pydatetime(), now)
        except (OSError, ValueError) as exc:
            row["error"] = str(exc)
            rows.append(row)
            continue
        if not np.size(t):
            row["stale"] = True
            rows.append(row)
            continue
        windows = pair_rates(hourly(t, y), spec, end, significance)
        row["windows"] = windows
        top = max(windows, key=lambda r: (r["level"], -r["window_h"]))
        row["level"] = top["level"]
        rows.append(row)
        value = top[f"{top['component']}_mm_per_day"]
        key = (top["level"], abs(float(value)))
        if best is None or key > best[:2]:
            best = (*key, {"pair": pair, **top})
    out: dict[str, Any] = {
        "windows_h": list(spec["windows_h"]),
        "horizontal_mm_per_day": list(spec["horizontal_mm_per_day"]),
        "up_mm_per_day": list(spec["up_mm_per_day"]),
        "components": list(spec.get("components", COMPONENTS)),
        "significance_sigma": significance,
        "pairs": rows,
    }
    if best is None:
        out["level"] = None
        return out
    out["level"] = best[0]
    out["set_by"] = best[2]
    return out


def replay(
    h: pd.DataFrame, spec: Mapping[str, Any], ends: Sequence[pd.Timestamp],
    significance: float = SIGNIFICANCE,
) -> list[tuple[pd.Timestamp, list[dict[str, Any]]]]:  # fmt: skip
    """The bench: :func:`pair_rates` at each of ``ends`` on one series."""
    return [(e, pair_rates(h, spec, pd.Timestamp(e), significance)) for e in ends]
