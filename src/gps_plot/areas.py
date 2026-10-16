"""Names for map file stems: the area a map shows and how its stations
were chosen (``map_{kind}_{area}_{selection}[_{extras}]``).

A map is about a region or a concept, not about its first station. The
area is the best-matching named box of :data:`AREAS` (intersection over
union at least :data:`MIN_IOU`), else the region's own coordinates
(``22.8W-21.9W_63.75N-64.1N``). The boxes are approximate (W, E, S, N in
degrees) and meant to be edited: add a box, and maps there take its name.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence

#: named map areas, (W, E, S, N) [deg]; ASCII names (they go in file names)
AREAS: dict[str, tuple[float, float, float, float]] = {
    "iceland": (-24.6, -13.2, 63.2, 66.7),
    "reykjanes": (-22.9, -21.7, 63.75, 64.1),
    "svartsengi": (-22.57, -22.35, 63.825, 63.91),
    "hengill": (-21.6, -21.0, 63.9, 64.2),
    "sisz": (-21.0, -19.8, 63.85, 64.1),
    "snaefellsnes": (-24.1, -22.5, 64.7, 65.05),
    "hekla": (-19.95, -19.4, 63.9, 64.1),
    "torfajokull": (-19.3, -18.8, 63.8, 64.0),
    "eyjafjallajokull": (-19.9, -19.4, 63.55, 63.75),
    "katla": (-19.5, -18.7, 63.45, 63.8),
    "vatnajokull": (-18.2, -15.4, 63.95, 64.85),
    "grimsvotn": (-17.6, -17.0, 64.3, 64.55),
    "bardarbunga": (-17.9, -17.2, 64.5, 64.8),
    "oraefajokull": (-16.9, -16.4, 63.9, 64.1),
    "askja": (-17.0, -16.5, 64.95, 65.15),
    "krafla": (-17.0, -16.6, 65.6, 65.85),
    "theistareykir": (-17.2, -16.7, 65.8, 66.05),
    "north": (-19.5, -15.5, 65.3, 66.6),
}
#: the least overlap (intersection / union) for a name to fit a region
MIN_IOU = 0.35

_ASCII = str.maketrans({"ð": "d", "Ð": "D", "þ": "th", "Þ": "Th", "æ": "ae", "Æ": "Ae"})


def slug(text: str) -> str:
    """File-name safe ASCII: Icelandic letters folded, runs of anything
    else one ``-``, no leading/trailing ``-`` (``--epos`` -> ``epos``)."""
    folded = unicodedata.normalize("NFKD", text.translate(_ASCII))
    ascii_ = folded.encode("ascii", "ignore").decode()
    return re.sub(r"[^A-Za-z0-9.]+", "-", ascii_).strip("-.")


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    w, e = max(a[0], b[0]), min(a[1], b[1])
    s, n = max(a[2], b[2]), min(a[3], b[3])
    inter = max(e - w, 0.0) * max(n - s, 0.0)
    area = (a[1] - a[0]) * (a[3] - a[2]) + (b[1] - b[0]) * (b[3] - b[2]) - inter
    return inter / area if area > 0 else 0.0


def _deg(v: float, pos: str, neg: str) -> str:
    return f"{abs(round(v, 4)):g}{pos if v >= 0 else neg}"


def area_name(region: Sequence[float]) -> str:
    """The name of the area a ``(W, E, S, N)`` region shows: the named box
    it overlaps most (at least :data:`MIN_IOU`), else its coordinates."""
    box = tuple(float(v) for v in region[:4])
    best = max(AREAS, key=lambda k: _iou(box, AREAS[k]))
    if _iou(box, AREAS[best]) >= MIN_IOU:
        return best
    w, e, s, n = box
    return f"{_deg(w, 'E', 'W')}-{_deg(e, 'E', 'W')}_{_deg(s, 'N', 'S')}-{_deg(n, 'N', 'S')}"


def selection(
    *,
    tos: bool = False,
    tos_search: str | None = None,
    patterns: Sequence[str] = (),
    files: Sequence[str] = (),
) -> str:
    """How the stations were chosen: ``tos``, ``tos-<search>``, the
    station patterns (``SEY``), the station-file labels; joined by ``+``."""
    parts = []
    if tos_search is not None:
        found = slug(tos_search)
        parts.append(f"tos-{found}" if found else "tos")
    elif tos:
        parts.append("tos")
    pats = [slug(p) for p in patterns if slug(p)]
    if pats:
        parts.append("-".join(pats))
    parts += [slug(f) for f in files if slug(f)]
    return "+".join(parts)


def stem(kind: str, region: Sequence[float], *parts: str) -> str:
    """``map_{kind}_{area}_{part}_…``, empty parts left out."""
    return "_".join([f"map_{kind}", area_name(region), *[p for p in parts if p]])
