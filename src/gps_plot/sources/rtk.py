"""RTK source: an RTK baseline (``geo_dataread.rtk_read``) as a :class:`Series`.

Independent of the daily GLOBK source (``sources.gps``): an RTK baseline is
plotted on its own (``gpsplot rtk``), and :func:`align_daily` places a daily
GLOBK baseline over it for the joined plot.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import fnmatch
from collections.abc import Sequence
from typing import Any

import numpy as np

from gps_plot.series import Series

#: the RTK series is zeroed on its first hour (1/sigma-weighted mean), the
#: 1-minute counterpart of getData's first-5-days zero
ZERO_WINDOW = np.timedelta64(1, "h")


def expand_rtk(specs: Sequence[str], baselines: Sequence[str]) -> list[str]:
    """``"SEY?-SEYD"`` → the RTK baseline directories it matches, in order.

    Matched against the directories that exist (rover and base sides
    separately), not a cross product of stations: a pair exists only if
    rtk.vedur.is processes it, and some exist in both directions.

    Raises:
        ValueError: a spec that is not ``ROVER-BASE`` or matches nothing.
    """
    from gps_plot.stack import split_spec

    out: list[str] = []
    for spec in specs:
        pair = split_spec(spec)
        if pair is None:
            raise ValueError(f"{spec!r}: an RTK baseline is ROVER-BASE")
        rover, base = pair[0].upper(), pair[1].upper()
        got = [
            b
            for b in baselines
            if fnmatch.fnmatchcase(b[:4], rover) and fnmatch.fnmatchcase(b[5:], base)
        ]
        if not got:
            raise ValueError(f"{spec!r} matches no RTK baseline")
        out.extend(g for g in got if g not in out)
    return out


def zero_level(t: Any, y: Any, sigma: Any) -> Any:
    """(3,) level of the first :data:`ZERO_WINDOW` (1/σ-weighted mean)."""
    from gps_analysis.baseline import estimate_offset

    t = np.asarray(t)
    level = np.zeros(3)
    if t.size == 0:
        return level
    first = t <= t[0] + ZERO_WINDOW
    for c in range(3):
        ok = first & np.isfinite(y[c]) & np.isfinite(sigma[c]) & (sigma[c] > 0)
        if ok.any():
            level[c] = float(np.asarray(estimate_offset(t[ok], y[c, ok], sigma[c, ok])))
    return level


def fetch_rtk(
    baseline: str,
    start: dt.datetime,
    end: dt.datetime,
    *,
    root: str | None = None,
    interval: str = "60s",
    drop_q: Sequence[int] = (5,),
    max_sigma: float = 0.1,
) -> Series:
    """An RTK baseline's interval medians in mm, zeroed on its first hour.

    ``t`` is ``datetime64`` (UTC). ``meta`` has ``interval`` and ``level``
    (the removed zero, mm) so a joined series can be placed against it.
    """
    from geo_dataread.rtk_read import read_rtk_baseline

    r = read_rtk_baseline(
        baseline,
        start,
        end,
        root=root,
        interval=interval,
        drop_q=drop_q,
        max_sigma=max_sigma,
    )
    y = r.neu * 1000.0
    sigma = r.sigma * 1000.0
    level = zero_level(r.time, y, sigma)
    return Series(
        label=baseline,
        t=r.time,
        y=y - level[:, None],
        sigma=sigma,
        meta={"interval": interval, "level": level, "source": "rtk"},
    )


def align_daily(daily: Series, rtk: Series) -> Series:
    """Move a daily GLOBK baseline onto an RTK one: median over the overlap.

    The two are independent solutions with independent zeros. Per
    component, the daily series is shifted by the median of
    ``rtk(t_day) − daily(t_day)`` over the days the RTK series covers
    (RTK interpolated to the daily epochs). An ESTIMATED alignment, not an
    exact datum: ``meta["aligned"]`` holds the shift [mm] and the number
    of days it rests on.

    Raises:
        ValueError: no daily epoch inside the RTK span.
    """
    t_rtk = np.asarray(rtk.t, dtype="datetime64[ns]").astype(np.int64).astype(float)
    t_day = np.asarray(daily.t, dtype="datetime64[ns]").astype(np.int64).astype(float)
    inside = (
        (t_day >= t_rtk.min()) & (t_day <= t_rtk.max()) if t_rtk.size else t_day < 0
    )
    if not inside.any():
        raise ValueError(f"{daily.label}: no daily solution inside the RTK span")
    shift = np.zeros(3)
    y_rtk = np.asarray(rtk.y, dtype=float)
    y_day = np.asarray(daily.y, dtype=float)
    for c in range(3):
        ok = np.isfinite(y_rtk[c])
        at_day = np.interp(t_day[inside], t_rtk[ok], y_rtk[c][ok])
        shift[c] = float(np.nanmedian(at_day - y_day[c][inside]))
    return dataclasses.replace(
        daily,
        y=y_day + shift[:, None],
        meta={**daily.meta, "aligned": {"shift": shift, "days": int(inside.sum())}},
    )
