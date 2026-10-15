"""Stations read from files, for maps (``gpsplot map station --stations-file``).

Three layouts, told apart by the file:

- **GeoJSON** (``.geojson``/``.json``): Point features; the marker is
  the ``marker``, ``name`` or ``id`` property (the first one present),
  the name the ``name`` property.
- **CSV** (a first line with a comma): a header naming the columns
  ``marker`` (or ``station``/``id``), ``lon`` (or ``longitude``/``x``),
  ``lat`` (or ``latitude``/``y``) and optionally ``name``.
- **Text**: one station per line, ``MARKER LON LAT [NAME …]``, blank
  lines and ``#`` comments skipped.

Every station of one file carries the file's legend label as its map
group (``StationCoordinate.status``).
"""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

#: one station: marker, lon, lat, name
Row = tuple[str, float, float, str | None]

MARKER_KEYS = ("marker", "station", "id")
LON_KEYS = ("lon", "longitude", "x")
LAT_KEYS = ("lat", "latitude", "y")
GEOJSON_SUFFIXES = (".geojson", ".json")


def split_spec(spec: str) -> tuple[str, str]:
    """``FILE`` or ``FILE:LABEL`` -> (file, label); the label defaults to
    the file's stem. A colon that is part of an existing path stays."""
    path, sep, label = spec.rpartition(":")
    if not sep or not path or Path(spec).exists():
        return spec, Path(spec).stem
    return path, label or Path(path).stem


def _pick(row: dict[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if row.get(key) not in (None, ""):
            return row[key]
    return None


def _row(where: str, marker: Any, lon: Any, lat: Any, name: Any) -> Row:
    if marker in (None, "") or lon in (None, "") or lat in (None, ""):
        raise ValueError(f"{where}: needs a marker, lon and lat")
    try:
        return str(marker).strip(), float(lon), float(lat), name or None
    except (TypeError, ValueError):
        raise ValueError(f"{where}: lon/lat {lon!r}, {lat!r} are not numbers") from None


def _geojson(path: Path) -> list[Row]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("type") != "FeatureCollection":
        raise ValueError(f"{path}: not a GeoJSON FeatureCollection")
    rows: list[Row] = []
    for k, feat in enumerate(data.get("features") or (), 1):
        geom = feat.get("geometry") or {}
        if geom.get("type") != "Point":
            continue
        props = feat.get("properties") or {}
        lon, lat = (geom.get("coordinates") or [None, None])[:2]
        rows.append(
            _row(f"{path} feature {k}", _pick(props, ("marker", "name", "id")),
                 lon, lat, props.get("name"))
        )  # fmt: skip
    return rows


def _csv(path: Path, text: str) -> list[Row]:
    reader = csv.DictReader(io.StringIO(text))
    reader.fieldnames = [f.strip().lower() for f in reader.fieldnames or ()]
    missing = [
        keys[0]
        for keys in (MARKER_KEYS, LON_KEYS, LAT_KEYS)
        if not set(keys) & set(reader.fieldnames)
    ]
    if missing:
        raise ValueError(f"{path}: the CSV header has no {', '.join(missing)} column")
    return [
        _row(f"{path} line {k}", _pick(r, MARKER_KEYS), _pick(r, LON_KEYS),
             _pick(r, LAT_KEYS), (r.get("name") or "").strip())
        for k, r in enumerate(reader, 2)
    ]  # fmt: skip


def _text(path: Path, text: str) -> list[Row]:
    rows: list[Row] = []
    for k, line in enumerate(text.splitlines(), 1):
        t = line.split("#", 1)[0].split()
        if not t:
            continue
        if len(t) < 3:
            raise ValueError(f"{path} line {k}: use MARKER LON LAT [NAME]")
        rows.append(_row(f"{path} line {k}", t[0], t[1], t[2], " ".join(t[3:])))
    return rows


def read_stations(path: str | Path, label: str, maps: Any) -> list[Any]:
    """The stations of one file as ``maps.StationCoordinate``, grouped
    under ``label``.

    Raises:
        FileNotFoundError: no such file.
        ValueError: a line or feature without a marker and position.
    """
    path = Path(path).expanduser()
    if path.suffix.lower() in GEOJSON_SUFFIXES:
        rows = _geojson(path)
    else:
        text = path.read_text(encoding="utf-8")
        first = next((ln for ln in text.splitlines() if ln.strip()), "")
        rows = _csv(path, text) if "," in first else _text(path, text)
    if not rows:
        raise ValueError(f"{path}: no stations")
    return [
        maps.StationCoordinate(marker.upper(), lon, lat, name, label)
        for marker, lon, lat, name in rows
    ]
