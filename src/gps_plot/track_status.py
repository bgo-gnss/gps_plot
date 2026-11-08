"""The track map's status file (``map station --track … --track-status``).

The numbers a monitoring page shows beside the map, from the same run so
the page and the map always agree (aflogun's *Vöktun* tab reads it): per
station the rate over the last ``window_days`` (``--track-status-days``)
and since the onset, each with its σ (:mod:`gps_plot.track`: white noise
floored at the pre-onset scatter, times the block-mean wander scale), the
newest day's movement and whether it is significant, and the newest day.

Rates are in mm/yr, the products' unit; a page converts (mm/d = /365.25).
Written as ``<figure>.status.json`` next to the map.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import os
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from gps_plot import track
from gps_plot.track import Rate, Track, Vertical

SCHEMA = 1
#: the newest day's movement is significant above this many σ (as a moved track)
STEP_SIGMA = track.MOVED_SIGMA
SUFFIX = ".status.json"


def day_of(t: float) -> dt.date:
    """The calendar day of a fractional-year epoch (:func:`track.day_key`)."""
    return dt.date.fromordinal(int(track.day_key(np.array([t]))[0]))


def _pair(v: float, s: float) -> list[float]:
    return [round(float(v), 2), round(float(s), 2)]


def rate_record(rate: Rate | None, up: tuple[float, float] | None,
                window: float | None = None) -> dict[str, Any] | None:  # fmt: skip
    """A rate (N, E, the horizontal speed and its direction; U) as
    ``[value, σ]`` pairs [mm/yr]; None without a horizontal rate."""
    if rate is None:
        return None
    out: dict[str, Any] = {
        "N": _pair(rate.north, rate.sigma_north),
        "E": _pair(rate.east, rate.sigma_east),
        "speed": _pair(rate.speed, rate.sigma_speed),
        "azimuth": round(rate.azimuth, 1),
        "U": _pair(*up) if up is not None else None,
    }
    if window is not None:
        out["window_days"] = round(float(window), 1)
    return out


def last_step(tr: Track) -> dict[str, Any] | None:
    """The movement from the second newest day to the newest [mm], per
    component, with σ = √2 × the daily scatter before the onset; a
    component is significant above ``STEP_SIGMA`` σ. None with fewer than
    two days since the onset."""
    after = tr.days >= 0
    if int(after.sum()) < 2:
        return None
    t = tr.t[after]
    sn, se = tr.scatter_ne if any(tr.scatter_ne) else (tr.scatter / math.sqrt(2),) * 2
    dn = float(tr.north[after][-1] - tr.north[after][-2])
    de = float(tr.east[after][-1] - tr.east[after][-2])
    sig_n, sig_e = math.sqrt(2) * sn, math.sqrt(2) * se
    out: dict[str, Any] = {
        "from": day_of(float(t[-2])).isoformat(),
        "to": day_of(float(t[-1])).isoformat(),
        "N": _pair(dn, sig_n),
        "E": _pair(de, sig_e),
        "horizontal": _pair(
            math.hypot(dn, de), math.hypot(sig_n, sig_e) / math.sqrt(2)
        ),
        "significant": bool(
            abs(dn) > STEP_SIGMA * sig_n or abs(de) > STEP_SIGMA * sig_e
        ),
        "U": None,
        "significant_up": False,
    }
    vt = tr.vertical
    if vt is not None:
        k = vt.days >= 0
        if int(k.sum()) >= 2 and day_of(float(vt.t[k][-1])) == day_of(float(t[-1])):
            du = float(vt.up[k][-1] - vt.up[k][-2])
            su = math.sqrt(2) * vt.scatter
            out["U"] = _pair(du, su)
            out["significant_up"] = bool(abs(du) > STEP_SIGMA * su)
    return out


def _up(vt: Vertical | None, which: str) -> tuple[float, float] | None:
    return getattr(vt, which) if vt is not None else None


def station_record(tr: Track, role: str = "station") -> dict[str, Any]:
    """One station's row: its newest day, the window and since-onset
    rates, the newest day's movement and flags."""
    vt = tr.vertical
    flags = []
    if tr.zero_extrapolated:
        flags.append("zero_extrapolated")
    if tr.status is None:
        flags.append("too_few_days")
    return {
        "station": tr.marker,
        "role": role,
        "lon": round(tr.lon, 5),
        "lat": round(tr.lat, 5),
        "last_day": day_of(float(tr.t[-1])).isoformat() if tr.t.size else None,
        "days_since_onset": tr.n_after,
        "window": rate_record(tr.status, _up(vt, "status"), tr.status_window),
        "since": rate_record(tr.since, _up(vt, "since")),
        "last_step": last_step(tr),
        "moved": bool(tr.moved),
        "flags": flags,
    }


def status_record(
    tracks: Sequence[Track],
    *,
    onset: dt.date,
    event: str,
    window_days: int,
    frame: str,
    common_mode: bool,
    keep_trend: bool,
    reference: Mapping[str, Any] | None = None,
    made: dt.datetime | None = None,
) -> dict[str, Any]:
    """The status file's content (``SCHEMA`` 1). ``reference``: ``asked``,
    ``kept`` and ``dropped`` (``[{"station", "speed"}]``) reference
    stations; a station among them gets that role."""
    ref = dict(reference or {})
    kept = set(ref.get("kept", ()))
    dropped = {d["station"] for d in ref.get("dropped", ())}

    def role(m: str) -> str:
        return (
            "reference"
            if m in kept
            else "dropped_reference"
            if m in dropped
            else "station"
        )

    # the newest day of the watched stations (not the references: a day only
    # they have says nothing of the stations watched), else of any
    watched = [tr for tr in tracks if role(tr.marker) == "station" and tr.t.size]
    days = [day_of(float(tr.t[-1])) for tr in (watched or tracks) if tr.t.size]
    return {
        "schema": SCHEMA,
        "made": (made or dt.datetime.now(dt.UTC)).strftime("%Y-%m-%dT%H:%MZ"),
        "event": event,
        "onset": onset.isoformat(),
        "newest_day": max(days).isoformat() if days else None,
        "window_days": window_days,
        "rate_unit": "mm/yr",
        "step_unit": "mm",
        "step_sigma": STEP_SIGMA,
        "frame": frame,
        "view": "cleaned",
        "common_mode": "subtract" if common_mode else "off",
        "trend": "keep" if keep_trend else "remove",
        "reference": ref or None,
        "stations": [station_record(tr, role(tr.marker)) for tr in tracks],
    }


def status_path(figure: str) -> str:
    """``<figure without its extension>.status.json``."""
    return os.path.splitext(figure)[0] + SUFFIX


def write_status(path: str, record: Mapping[str, Any]) -> None:
    """Write ``record`` to ``path`` (via a temporary file: a page reading it
    never sees half a file)."""
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(record, fh, ensure_ascii=False, indent=1)
        fh.write("\n")
    os.replace(tmp, path)
