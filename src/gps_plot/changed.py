"""``gpsplot run --changed``: redraw only the plots whose data changed.

A fingerprint of every station's files in pos_dir (``STA.*``: its .pos
series in every frame and session, its autcln icing/snow indices) and one
of the setup (the plots.yaml/events/areas/detrend config files of
:data:`SETUP_FILES`, the stations.cfg time stamp, gps_plot's own code) are
kept in ``<plots_dir>/_runs/.fingerprints.json`` after a whole run (no
``--product``/``--area``/stations) that ended without a failure.

A ``--changed`` run compares with them: a changed setup, or no state yet,
redraws everything; else only the jobs touching a changed station run
(:func:`narrow`). The network files (``network_days``/``network_cm``) are
not part of it: a new day changes them for every station, but only on
that day, which a station without the day does not plot.

A plot is drawn relative to its day (the "Last datapoint" colour, the
90 d/year windows), so a station that stops sending would keep an old
green title: a whole run without ``--changed`` each night redraws it all
(BGÓ 2026-11-02: the nightly run sets the baseline).
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
from collections.abc import Collection, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from gps_plot.products import Job

STATE_FILE = Path("_runs") / ".fingerprints.json"

#: config files (gpsconfig dir) whose change redraws every plot; never the
#: credential files (stations.cfg counts by its time stamp only)
SETUP_FILES = (
    "plots.yaml",
    "events.yaml",
    "station_areas.yaml",
    "map_layers.yaml",
    "analysis.yaml",
    "steps.yaml",
    "detrend_params.json",
    "detrend_params.itrf14.json",
    "detrend_params.itrf20.json",
    "detrend_itrf2008.csv",
    "fit_windows.csv",
    "protect_windows.csv",
    "segment_exclusions.csv",
    "excluded_epochs.csv",
    "outlier_overrides.csv",
    "junction_offsets.csv",
)


def _file_hash(path: Path, cache: dict[str, Any]) -> str:
    """sha1 of a file's bytes, reused from ``cache`` while its size and
    mtime are those cached."""
    st = path.stat()
    key = str(path)
    seen = cache.get(key)
    if seen and seen[0] == st.st_size and seen[1] == st.st_mtime_ns:
        return str(seen[2])
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    cache[key] = [st.st_size, st.st_mtime_ns, h.hexdigest()]
    return h.hexdigest()


def station_prints(pos_dir: Path, cache: dict[str, Any]) -> dict[str, str]:
    """{STA: fingerprint of its ``STA.*`` files in ``pos_dir``}; files of
    other names (network lists, logs) are left out."""
    files: dict[str, list[Path]] = {}
    for f in sorted(pos_dir.iterdir()):
        sta = f.name.split(".", 1)[0]
        if f.is_file() and len(sta) == 4 and sta.isalnum() and sta == sta.upper():
            files.setdefault(sta, []).append(f)
    out: dict[str, str] = {}
    for sta, paths in files.items():
        h = hashlib.sha1()
        for f in paths:
            h.update(f.name.encode() + b"\0" + _file_hash(f, cache).encode())
        out[sta] = h.hexdigest()
    return out


def setup_print(config_dir: Path, code_dir: Path, cache: dict[str, Any]) -> str:
    """One fingerprint of the setup: :data:`SETUP_FILES`, the stations.cfg
    time stamp and gps_plot's ``*.py`` files."""
    h = hashlib.sha1()
    for name in SETUP_FILES:
        f = config_dir / name
        h.update(name.encode() + b"\0")
        h.update(_file_hash(f, cache).encode() if f.is_file() else b"-")
    cfg = config_dir / "stations.cfg"
    h.update(str(cfg.stat().st_mtime_ns if cfg.is_file() else "-").encode())
    for f in sorted(code_dir.rglob("*.py")):
        h.update(str(f.relative_to(code_dir)).encode() + b"\0")
        h.update(_file_hash(f, cache).encode())
    return h.hexdigest()


def load_state(path: Path) -> dict[str, Any]:
    """The saved state (``setup``, ``stations``, ``files``), {} when none
    or unreadable."""
    try:
        state = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return state if isinstance(state, dict) else {}


def save_state(path: Path, state: Mapping[str, Any]) -> None:
    """Write the state atomically (a crash leaves the old one)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, sort_keys=True))
    os.replace(tmp, path)


def changed_stations(old: Mapping[str, str], new: Mapping[str, str]) -> set[str]:
    """Stations whose fingerprint differs, is new, or is gone."""
    return {s for s in set(old) | set(new) if old.get(s) != new.get(s)}


def _hits(member: str, changed: Collection[str]) -> bool:
    """A job member (``STA``, ``A-B``, a pattern ``SEY?-SEYD``) touches a
    changed station."""
    ends = member.upper().split("-")
    return any(fnmatch.fnmatchcase(c, end) for end in ends for c in changed)


def narrow(
    jobs: Sequence[Job],
    changed: Collection[str],
    keywords: Mapping[str, Collection[str] | None],
) -> list[Job]:
    """The jobs to redraw for the ``changed`` stations.

    A map with a station list runs when one of them changed, one without
    (every station in its region) when anything changed; a stack
    or polar job is one figure, kept whole when any of its series touches a
    changed station; any other job keeps only the stations (or pairs, by
    either end) that changed. ``keywords``: a job's station keywords
    (``active``) and their stations, ``None`` for any station.
    """
    changed = {c.upper() for c in changed}
    out: list[Job] = []
    if not changed:
        return out
    for job in jobs:
        if job.tool == "map":  # its stations, else every one in its region
            if not job.stations or any(_hits(m, changed) for m in job.stations):
                out.append(job)
            continue
        members = [m for m in job.stations if m not in keywords]
        keep = [m for m in members if _hits(m, changed)]
        for k in job.stations:
            if k in keywords:
                pool = keywords[k]
                keep += sorted(
                    c for c in changed if pool is None or c in {p.upper() for p in pool}
                )
        if not keep:
            continue
        whole = job.tool in ("stack", "polar")
        out.append(job if whole else replace(job, stations=tuple(keep)))
    return out
