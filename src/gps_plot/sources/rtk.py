"""RTK source: an RTK baseline (``geo_dataread.rtk_read``) as a :class:`Series`.

Independent of the daily GLOBK source (``sources.gps``): an RTK baseline is
plotted on its own (``gpsplot rtk``), and :func:`align` places one source on
the other for the joined plots (daily on RTK, or an RTK tail on years of
daily points with ``--longterm``).
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
        if not got and f"{base}-{rover}" in baselines:
            got = [f"{rover}-{base}"]  # a plain pair processed the other way
        if not got:
            raise ValueError(f"{spec!r} matches no RTK baseline")
        out.extend(g for g in got if g not in out)
    return out


def rtk_direction(name: str, baselines: Sequence[str]) -> tuple[str, bool]:
    """``(directory, flipped)`` of the RTK baseline ``A-B``.

    ``A-B`` itself when rtk.vedur.is processes it, else ``B-A`` flipped:
    rover − base is A − B (the daily GLOBK baseline), so the reversed
    directory is negated to keep the pair as written.

    Raises:
        ValueError: neither direction exists.
    """
    if name in baselines:
        return name, False
    reverse = f"{name[5:]}-{name[:4]}"
    if reverse in baselines:
        return reverse, True
    raise ValueError(f"no RTK baseline {name} (nor {reverse})")


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
    directory: str | None = None,
    flipped: bool = False,
) -> Series:
    """An RTK baseline's interval medians in mm, zeroed on its first hour.

    ``t`` is ``datetime64`` (UTC). ``meta`` has ``interval`` and ``level``
    (the removed zero, mm) so a joined series can be placed against it.
    ``directory``/``flipped`` (:func:`rtk_direction`) read a pair from its
    reversed directory, negated; the label stays ``baseline``.
    """
    from geo_dataread.rtk_read import read_rtk_baseline

    r = read_rtk_baseline(
        directory or baseline,
        start,
        end,
        root=root,
        interval=interval,
        drop_q=drop_q,
        max_sigma=max_sigma,
    )
    y = r.neu * (-1000.0 if flipped else 1000.0)
    sigma = r.sigma * 1000.0
    level = zero_level(r.time, y, sigma)
    return Series(
        label=baseline,
        t=r.time,
        y=y - level[:, None],
        sigma=sigma,
        meta={"interval": interval, "level": level, "source": "rtk"},
    )


def align(moving: Series, anchor: Series) -> Series:
    """Shift ``moving`` onto ``anchor``: the median over their overlap.

    The two are independent solutions with independent zeros. Per
    component, ``moving`` is shifted by the median of
    ``anchor(t) − moving(t)`` at the epochs of the sparser series inside
    the other's span (wider median spacing), the denser one interpolated there (daily epochs
    against an RTK line, either way round). An ESTIMATED alignment, not
    an exact datum: ``meta["aligned"]`` holds the shift [mm] and the
    number of epochs (days, for a daily series) it rests on.

    Raises:
        ValueError: the two do not overlap.
    """

    def ns(s: Series) -> Any:
        return np.asarray(s.t, dtype="datetime64[ns]").astype(np.int64).astype(float)

    t_mov, t_anc = ns(moving), ns(anchor)
    y_mov = np.asarray(moving.y, dtype=float)
    y_anc = np.asarray(anchor.y, dtype=float)
    sparse_is_moving = _spacing(t_mov) >= _spacing(t_anc)
    t_at, t_line = (t_mov, t_anc) if sparse_is_moving else (t_anc, t_mov)
    inside = (
        (t_at >= t_line.min()) & (t_at <= t_line.max()) if t_line.size else t_at < 0
    )
    if not inside.any():
        raise ValueError(f"{moving.label} and {anchor.label} do not overlap")
    shift = np.zeros(3)
    for c in range(3):
        if sparse_is_moving:
            ok = np.isfinite(y_anc[c])
            diff = np.interp(t_at[inside], t_anc[ok], y_anc[c][ok]) - y_mov[c][inside]
        else:
            ok = np.isfinite(y_mov[c])
            diff = y_anc[c][inside] - np.interp(t_at[inside], t_mov[ok], y_mov[c][ok])
        shift[c] = float(np.nanmedian(diff))
    return dataclasses.replace(
        moving,
        y=y_mov + shift[:, None],
        meta={**moving.meta, "aligned": {"shift": shift, "days": int(inside.sum())}},
    )


def _spacing(t: Any) -> float:
    """Median epoch spacing (inf for fewer than two epochs)."""
    return float(np.median(np.diff(t))) if t.size > 1 else float("inf")


def align_daily(daily: Series, rtk: Series) -> Series:
    """Move a daily GLOBK baseline onto an RTK one (:func:`align`).

    Raises:
        ValueError: no daily epoch inside the RTK span.
    """
    try:
        return align(daily, rtk)
    except ValueError:
        raise ValueError(
            f"{daily.label}: no daily solution inside the RTK span"
        ) from None
