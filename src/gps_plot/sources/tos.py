"""Stations from TOS, for maps (``gpsplot map station --tos``,
``--tos-search``).

TOS is the authoritative station register: it has stations stations.cfg
does not (campaign marks, external stations, retired ones), with a
status. One bulk GET (``TOSClient.list_stations("geophysical")``) returns
every geophysical station with its dated attributes; this module keeps the
GPS stations and reads each attribute's *current* value.

**Current value.** TOS attributes are dated (``date_from``/``date_to``):
a moved station has several ``lat``/``lon`` periods, a re-opened one a
closed ``date_end``. The current value is the open period
(``date_to is None``) — the rule ``tostools.station_kind.open_attribute``
uses. Taking the last listed value instead silently drops or misplaces
stations.

The fleet is cached as JSON under ``$XDG_CACHE_HOME/gps_plot`` for
:data:`CACHE_MAX_AGE` (the station list changes slowly); ``refresh=True``
refetches. A failed fetch raises — it must never read as an empty fleet.
tostools is imported only when a fetch is needed.

**Any station.** ``--tos-search`` hands a ``tos search`` command line to
tostools (:func:`search_stations`): every expression and flag of ``tos
search`` selects, ``subtype = GPS stöð`` unless the search names a
subtype (``'subtype = all'`` lifts it). Its results keep their subtype
and open attributes, for grouping (:func:`group_key`).
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import os
import shlex
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

#: TOS ``subtype`` of a GPS station (the geophysical domain also holds SIL,
#: gas, infrasound … stations)
GPS_SUBTYPE = "GPS stöð"
#: the TOS domain listing that holds the GPS stations
DOMAIN = "geophysical"
#: cache lifetime [s]
CACHE_MAX_AGE = 24 * 3600
CACHE_NAME = "tos_geophysical_stations.json"

#: map status, in drawing order (bottom first)
STATUSES = ("ended", "campaign", "external", "active")
STATUS_LABELS = {
    "active": "Continuous",
    "external": "External",
    "campaign": "Campaign",
    "ended": "Discontinued",
}


@dataclasses.dataclass(frozen=True)
class TosStation:
    """One station from TOS, with its current attributes.

    Attributes:
        marker: 4-character marker, upper case.
        lon: Longitude [deg].
        lat: Latitude [deg].
        name: Station name, or None.
        status: ``active``, ``external`` (operated by others), ``campaign``
            (not continuous) or ``ended`` (a current ``date_end`` on or
            before ``today``).
        date_end: The current ``date_end`` (ISO text), or None.
        subtype: The TOS subtype (``GPS stöð``, ``SIL stöð``, …), or None.
        attributes: Every open attribute value, by code.
    """

    marker: str
    lon: float
    lat: float
    name: str | None
    status: str
    date_end: str | None = None
    subtype: str | None = None
    attributes: Mapping[str, str] = dataclasses.field(
        default_factory=dict, compare=False, hash=False, repr=False
    )


def open_value(entity: Mapping[str, Any], code: str) -> str | None:
    """The stripped value of ``entity``'s open (``date_to is None``) ``code``."""
    for attr in entity.get("attributes") or ():
        if attr.get("code") == code and attr.get("date_to") is None:
            value = attr.get("value")
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def current_end(entity: Mapping[str, Any]) -> str | None:
    """The station's current ``date_end``, or None.

    TOS records an end either as an open period or as a zero-length one
    (``date_to == date_from``: HLFJ, GLER, KALT, STOR); both mean ended.
    A period closed *after* it began revokes the end — the station runs
    again (GRVV, GRVM after Grindavík, GJAC).
    """
    for attr in entity.get("attributes") or ():
        if attr.get("code") != "date_end":
            continue
        value = attr.get("value")
        if not (isinstance(value, str) and value.strip()):
            continue
        start, stop = attr.get("date_from"), attr.get("date_to")
        if stop is None or (start is not None and stop <= start):
            return value.strip()
    return None


def _status(entity: Mapping[str, Any], today: dt.date) -> tuple[str, str | None]:
    end = current_end(entity)
    if end is not None:
        try:
            ended = dt.date.fromisoformat(end[:10]) <= today
        except ValueError:
            ended = True  # an unparsable end date still says "ended"
        if ended:
            return "ended", end
    if open_value(entity, "operational_status") == "external":
        return "external", end
    if open_value(entity, "continuity") == "campaign":
        return "campaign", end
    return "active", end


def open_values(entity: Mapping[str, Any]) -> dict[str, str]:
    """Every open (``date_to is None``) attribute value of ``entity``."""
    out: dict[str, str] = {}
    for attr in entity.get("attributes") or ():
        code, value = attr.get("code"), attr.get("value")
        if attr.get("date_to") is None and isinstance(value, str) and value.strip():
            out.setdefault(str(code), value.strip())
    return out


def stations(
    entities: Iterable[Mapping[str, Any]],
    *,
    subtype: str | None = GPS_SUBTYPE,
    today: dt.date | None = None,
) -> tuple[TosStation, ...]:
    """The stations among TOS station entities, one per marker, sorted.

    Only those of ``subtype`` (None: every subtype). Entities without an
    open marker or position are skipped (they cannot be placed on a map).
    A marker held by several entities of one subtype keeps the one not
    ended, else the latest ``date_start``; a GPS and a SIL station may
    share a marker.
    """
    today = today or dt.date.today()
    out: dict[tuple[str, str], tuple[tuple[bool, str], TosStation]] = {}
    for ent in entities:
        kind = open_value(ent, "subtype")
        if subtype is not None and kind != subtype:
            continue
        marker, lat, lon = (open_value(ent, c) for c in ("marker", "lat", "lon"))
        if not (marker and lat and lon):
            continue
        try:
            la, lo = float(lat), float(lon)
        except ValueError:
            continue
        status, end = _status(ent, today)
        station = TosStation(
            marker.upper(), lo, la, open_value(ent, "name"), status, end,
            kind, open_values(ent),
        )  # fmt: skip
        # one marker, several entities (SOHO: the retired station and its
        # successor): the running one wins, then the latest start
        rank = (status != "ended", open_value(ent, "date_start") or "")
        key = (station.marker, kind or "")
        if key not in out or rank > out[key][0]:
            out[key] = (rank, station)
    return tuple(out[k][1] for k in sorted(out))


def gps_stations(
    entities: Iterable[Mapping[str, Any]], *, today: dt.date | None = None
) -> tuple[TosStation, ...]:
    """The GPS stations (open ``subtype`` :data:`GPS_SUBTYPE`) among TOS
    station entities, one per marker, sorted (see :func:`stations`)."""
    return stations(entities, subtype=GPS_SUBTYPE, today=today)


def cache_path() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    return Path(base) / "gps_plot" / CACHE_NAME


def _fetch() -> list[Mapping[str, Any]]:
    try:
        from tostools.api.tos_client import TOSClient
    except ImportError as exc:  # pragma: no cover - depends on the env
        raise ImportError(
            "TOS stations need tostools: pip install 'gps_plot[tos]' "
            "(or the editable sibling in the gpslibrary env)"
        ) from exc
    return list(TOSClient().list_stations(DOMAIN))


def load_entities(
    *,
    refresh: bool = False,
    path: Path | None = None,
    fetch: Callable[[], Sequence[Mapping[str, Any]]] | None = None,
    max_age: float = CACHE_MAX_AGE,
) -> list[Mapping[str, Any]]:
    """TOS geophysical station entities: from the cache, else fetched.

    Raises:
        RuntimeError: the fetch failed or returned nothing (never an
            empty fleet).
    """
    path = path or cache_path()
    if not refresh and path.is_file() and time.time() - path.stat().st_mtime < max_age:
        return list(json.loads(path.read_text(encoding="utf-8")))
    entities = list((fetch or _fetch)())
    if not entities:
        raise RuntimeError("TOS returned no stations")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(entities), encoding="utf-8")
    tmp.replace(path)
    return entities


def tos_gps_stations(*, refresh: bool = False) -> tuple[TosStation, ...]:
    """Every GPS station in TOS (cached; ``refresh`` refetches)."""
    return gps_stations(load_entities(refresh=refresh))


def search_stations(
    search: str | Sequence[str],
    *,
    refresh: bool = False,
    run: Callable[..., Any] | None = None,
) -> tuple[TosStation, ...]:
    """The stations a ``tos search`` command line selects.

    ``search`` is what follows ``tos search`` (a string is split like a
    shell does: ``"--epos 'name ~ *vík*'"``); ``subtype = GPS stöð``
    unless it names a subtype. ``refresh`` refetches (tostools' cache).
    ``run`` stands in for ``tostools.tos.search_from_argv`` (tests).

    Raises:
        ValueError: a bad search (tostools' message).
    """
    argv = shlex.split(search) if isinstance(search, str) else list(search)
    if run is None:
        try:
            from tostools.tos import SearchUsage, search_from_argv
        except ImportError as exc:  # pragma: no cover - depends on the env
            raise ImportError(
                "--tos-search needs tostools with search_from_argv "
                "(the editable sibling in the gpslibrary env)"
            ) from exc
        run = search_from_argv
    else:
        SearchUsage = ValueError  # noqa: N806 - the stand-in's error
    try:
        result = run(
            [*argv, *(["--refresh"] if refresh else [])], default_subtype=GPS_SUBTYPE
        )
    except SearchUsage as exc:
        raise ValueError(str(exc)) from None
    except SystemExit:  # --help: tostools printed it
        raise ValueError("tos search --help is printed above") from None
    return stations(result.stations, subtype=None)


#: ``--group-by`` keys that are not a TOS attribute code
GROUP_STATUS, GROUP_SUBTYPE = "status", "subtype"


def group_key(station: TosStation, group_by: str | None) -> str | None:
    """The legend group of ``station``: its status (:data:`STATUSES`) for
    ``status``, its subtype for ``subtype``, else the open value of the
    attribute ``group_by`` (``"<code>: none"`` when it has none)."""
    if group_by in (None, GROUP_STATUS):
        return station.status
    if group_by == GROUP_SUBTYPE:
        return station.subtype or "no subtype"
    return station.attributes.get(str(group_by)) or f"{group_by}: none"


def default_group(fleet: Sequence[TosStation]) -> str:
    """``status`` when every station is a GPS station (or has no subtype),
    else ``subtype``."""
    kinds = {s.subtype for s in fleet} - {None}
    return GROUP_STATUS if kinds <= {GPS_SUBTYPE} else GROUP_SUBTYPE
